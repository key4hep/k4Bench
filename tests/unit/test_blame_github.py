"""Unit tests for :mod:`k4bench.blame.github` — commit range → PRs, with every
network response mocked. Covers the failure modes the module must survive
without raising: 404 (rewritten history), the 250-commit cap, non-squash commits
needing the pulls fallback — and the one it must raise on: a rate limit."""

from __future__ import annotations

import pytest

from k4bench.blame.github import (
    GitHubClient,
    RateLimitError,
    parse_pr_number,
    resolve_repo_prs,
)


class _Resp:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class _FakeSession:
    """Routes GET by URL suffix to a queued response. ``routes`` maps a path
    fragment to a :class:`_Resp` (or a list consumed in order)."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(url)
        for fragment, resp in self.routes.items():
            if fragment in url:
                if isinstance(resp, list):
                    return resp.pop(0)
                return resp
        return _Resp(404, {"message": "Not Found"})


def _client(routes: dict) -> GitHubClient:
    return GitHubClient(token="t", session=_FakeSession(routes))


def _commit(sha: str, message: str) -> dict:
    return {"sha": sha, "commit": {"message": message}}


def _pr_body(number: int) -> dict:
    return {
        "title": f"Title {number}", "user": {"login": "alice"},
        "html_url": f"https://github.com/key4hep/k4geo/pull/{number}",
        "merged_at": "2026-07-04T00:00:00Z", "additions": 12, "deletions": 3,
    }


# ── PR-number parsing ─────────────────────────────────────────────────────────

def test_parse_pr_number_reads_squash_suffix():
    assert parse_pr_number("Lower the tracker step limit (#1234)") == 1234


def test_parse_pr_number_ignores_issue_refs_in_prose():
    # Only the trailing (#N) on the first line is the merged PR; an earlier
    # issue reference in the subject, or a (#M) on a later body line, is not.
    assert parse_pr_number("Fix for #99: cleanup (#1234)") == 1234
    assert parse_pr_number("Real title (#1234)\nCloses (#1) in the body") == 1234


def test_parse_pr_number_none_when_absent():
    assert parse_pr_number("A plain merge commit") is None
    assert parse_pr_number("") is None


# ── resolve_repo_prs ──────────────────────────────────────────────────────────

def test_resolves_prs_from_compare_range():
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 2,
            "commits": [
                _commit("s1", "First change (#10)"),
                _commit("s2", "Second change (#11)"),
            ],
        }),
        "/pulls/10/files": _Resp(200, [{"filename": "FCCee/ALLEGRO/a.xml"}]),
        "/pulls/11/files": _Resp(200, [{"filename": "src/b.cpp"}]),
        "/pulls/10": _Resp(200, _pr_body(10)),
        "/pulls/11": _Resp(200, _pr_body(11)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert not res.commits_unavailable and not res.truncated
    assert sorted(c.number for c in res.candidates) == [10, 11]
    pr10 = next(c for c in res.candidates if c.number == 10)
    assert pr10.author == "alice"
    assert pr10.files == ("FCCee/ALLEGRO/a.xml",)
    assert pr10.additions == 12 and pr10.deletions == 3


def test_404_compare_marks_unavailable_without_raising():
    routes = {"/compare/": _Resp(404, {"message": "Not Found"})}
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert res.commits_unavailable is True
    assert res.candidates == []


def test_truncation_flag_when_compare_caps_commits():
    # GitHub caps compare at 250 commits: total_commits exceeds the returned list.
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 300,
            "commits": [_commit("s1", "Change (#10)")],
        }),
        "/pulls/10/files": _Resp(200, []),
        "/pulls/10": _Resp(200, _pr_body(10)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert res.truncated is True
    assert [c.number for c in res.candidates] == [10]


def test_falls_back_to_commit_pulls_when_no_squash_ref():
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 1,
            "commits": [_commit("deadbeef", "A plain merge commit, no ref")],
        }),
        "/commits/deadbeef/pulls": _Resp(200, [{"number": 55}]),
        "/pulls/55/files": _Resp(200, [{"filename": "x"}]),
        "/pulls/55": _Resp(200, _pr_body(55)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert [c.number for c in res.candidates] == [55]


def test_rate_limit_raises():
    # A throttled compare must abort the night, not degrade silently.
    routes = {"/compare/": _Resp(403, {"message": "API rate limit exceeded"},
                                 headers={"X-RateLimit-Remaining": "0"})}
    with pytest.raises(RateLimitError):
        resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)


def test_plain_403_is_not_a_rate_limit():
    # A permission 403 (e.g. a private repo) is one repo's problem, recorded as
    # unavailable — not a reason to abort the whole night.
    routes = {"/compare/": _Resp(403, {"message": "Must have admin rights"})}
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert res.commits_unavailable is True


def test_deduplicates_prs_across_commits():
    # Two commits, same PR (a rebase/backport) → one candidate.
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 2,
            "commits": [_commit("s1", "Part one (#10)"), _commit("s2", "Part two (#10)")],
        }),
        "/pulls/10/files": _Resp(200, []),
        "/pulls/10": _Resp(200, _pr_body(10)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert [c.number for c in res.candidates] == [10]


# ── Patch capture (the transient ranker input) ────────────────────────────────

def _one_pr_routes(files: list[dict]) -> dict:
    return {
        "/compare/": _Resp(200, {
            "total_commits": 1, "commits": [_commit("s1", "Change (#10)")],
        }),
        "/pulls/10/files": _Resp(200, files),
        "/pulls/10": _Resp(200, _pr_body(10)),
    }


def test_captures_patch_text_keyed_by_pr():
    routes = _one_pr_routes([
        {"filename": "FCCee/ALLEGRO/a.xml", "patch": "@@ -1 +1 @@\n-old\n+new steps"},
    ])
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    patch = res.patches[10]
    assert "FCCee/ALLEGRO/a.xml" in patch  # per-file header
    assert "+new steps" in patch           # the actual diff
    # The path still rides on the persisted candidate; the patch does not.
    assert res.candidates[0].files == ("FCCee/ALLEGRO/a.xml",)


def test_binary_and_rename_keep_path_but_contribute_no_diff():
    # Binary blobs and pure renames arrive with no ``patch`` field.
    routes = _one_pr_routes([
        {"filename": "img/logo.png"},
        {"filename": "new/name.py", "previous_filename": "old/name.py"},
    ])
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert res.candidates[0].files == ("img/logo.png", "new/name.py")  # paths kept
    assert 10 not in res.patches  # no diff text → nothing stored to rank on


def test_large_patch_is_truncated_and_marked():
    routes = _one_pr_routes([{"filename": "big.cpp", "patch": "x" * 5000}])
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    patch = res.patches[10]
    assert "… (truncated)" in patch
    assert len(patch) < 2500  # bounded well below the raw 5000 by the per-file cap


def test_total_patch_bounded_across_many_files():
    files = [{"filename": f"f{i}.cpp", "patch": "y" * 1500} for i in range(10)]
    res = resolve_repo_prs(_client(_one_pr_routes(files)), "key4hep/k4geo", "a" * 40, "c" * 40)
    patch = res.patches[10]
    assert "… (truncated)" in patch
    assert len(patch) < 7000  # per-PR cap holds even when each file is sizeable
