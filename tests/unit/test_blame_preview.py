"""``.github/scripts/blame_preview.py`` — the guards around a write-capable tool.

The rendering path is production code tested elsewhere (``test_blame_comment``,
``test_blame_publish``); what belongs here is everything this script adds on top
of it, all of which is about not doing something surprising: it writes only when
asked twice, it refuses to preview a comment production would not have produced,
and it redirects a body without rewriting it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"


def _load_preview():
    """Import the CLI by path, with its own directory importable — the script
    imports its sibling ``blame_comment`` by module name, which only resolves
    when ``.github/scripts`` is on ``sys.path`` (running it directly puts it
    there; pytest does not)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "blame_preview", _SCRIPTS / "blame_preview.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preview():
    return _load_preview()


_PLAN_NIGHT = "2026-06-27"


def _draft():
    """A real draft for ``key4hep/k4geo#607`` and the 2026-06-24 → 2026-06-25
    window — one ranked row, rendered by the production renderer."""
    from k4bench.blame.comment import CommentDraft, CommentPlan, RegressionRow, marker_for
    from k4bench.blame.models import CandidatePR
    from k4bench.regression.models import Direction, MetricVerdict, Severity

    verdict = MetricVerdict(
        detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e-_10GeV", label="baseline", metric_family="time",
        metric="mean_time_s", sub_detector=None, run_id=_PLAN_NIGHT,
        run_date="2026-06-25", value=120.0, baseline_median=100.0,
        baseline_mad=1.0, pct_change=0.20, z_score=6.0,
        severity=Severity.CONFIRMED, direction=Direction.UP, reason="step",
        onset_run_id="2026-06-25", onset_run_date="2026-06-25",
        last_accepted_run_id="2026-06-24", last_accepted_run_date="2026-06-24",
        first_confirmed_run_id=_PLAN_NIGHT,
    )
    subject = CandidatePR(
        repo="key4hep/k4geo", number=607, title="Change detector dimensions",
        author="alice", url="https://github.com/key4hep/k4geo/pull/607",
        merged_at="2026-06-24T09:00:00Z", files=("a.xml",), additions=1,
        deletions=1, score=91.0, ranked=True,
    )
    plan = CommentPlan(
        repo="key4hep/k4geo", number=607, subject=subject,
        base_release="2026-06-24", onset_release="2026-06-25",
        report_night=_PLAN_NIGHT,
        rows=[RegressionRow(
            verdict=verdict, stack="key4hep-2026-06-25", fact_id="r1",
            scope_score=91.0, scope_state="ranked",
        )],
    )
    return CommentDraft(
        repo="key4hep/k4geo", number=607,
        marker=marker_for("2026-06-24", "2026-06-25"), plan=plan,
        attribution=None, facts_payload={"window": ["2026-06-24", "2026-06-25"]},
    )


@pytest.fixture
def wired(monkeypatch, preview):
    """The script with every I/O boundary replaced: EOS reads, the selection,
    the renderer and the upsert. Returns the recorded publish calls."""
    from k4bench.blame import comment as comment_mod
    from k4bench.blame import github as github_mod
    from k4bench.blame import publish as publish_mod
    from k4bench.blame.models import BlameReport
    from k4bench import remote as remote_mod

    blame = BlameReport(generated_at="x", report_night=_PLAN_NIGHT, entries=())
    monkeypatch.setattr(
        remote_mod, "fetch_report", lambda *_a: {"generated_at": "x", "groups": []}
    )
    monkeypatch.setattr(
        remote_mod, "fetch_blame", lambda *_a: json.loads(json.dumps(blame.to_json()))
    )

    class _Plan:
        repo, number = "key4hep/k4geo", 607

    monkeypatch.setattr(comment_mod, "select", lambda *_a, **_k: [_Plan()])
    monkeypatch.setattr(
        comment_mod, "build_comments", lambda *_a, **_k: [_draft()]
    )
    monkeypatch.setattr(github_mod, "GitHubClient", lambda **_k: object())

    calls: list[dict] = []

    def _publish(_client, comments, *, dry_run=False):
        calls.append({"comments": comments, "dry_run": dry_run})
        return publish_mod.PublishResult(planned=[c.target for c in comments])

    monkeypatch.setattr(publish_mod, "publish", _publish)
    return calls


@pytest.fixture
def llm(monkeypatch):
    """A configured reviewer, so the fail-closed guard is not what is under test."""
    from k4bench.blame import attribute as attribute_mod

    monkeypatch.setattr(attribute_mod, "attributor_from_env", lambda: object())


# ── Writing is opt-in twice ───────────────────────────────────────────────────

def test_a_token_in_the_environment_is_not_consent_to_post(
    preview, wired, llm, monkeypatch
):
    # The whole point of the flag: an exported write token must not turn a
    # preview run into an edit on someone else's pull request.
    monkeypatch.setenv("K4BENCH_PR_COMMENT_TOKEN", "ghp_real_write_token")
    assert preview.main(["--night", _PLAN_NIGHT]) == 0
    assert [call["dry_run"] for call in wired] == [True]


def test_posting_needs_an_explicit_target(preview):
    with pytest.raises(SystemExit) as exc:
        preview.main(["--post", "--token", "t"])
    assert exc.value.code == 2


def test_posting_needs_a_write_token(preview, monkeypatch):
    monkeypatch.delenv("K4BENCH_PR_COMMENT_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        preview.main(["--post", "--post-to", "key4hep/k4Bench#106"])
    assert exc.value.code == 2


def test_post_writes_and_redirects_the_body_untouched(preview, wired, llm):
    # The redirect moves only the write target: the body still names --only, so
    # the reviewer reads exactly what would have landed there.
    assert preview.main([
        "--post", "--post-to", "key4hep/k4Bench#106", "--token", "t",
    ]) == 0
    (call,) = wired
    assert call["dry_run"] is False
    (comment,) = call["comments"]
    assert (comment.repo, comment.number) == ("key4hep/k4Bench", 106)
    # Everything under the re-keyed marker is the body rendered for --only.
    _, _, body = comment.render().body.partition("\n")
    _, _, original = _draft().render().body.partition("\n")
    assert body == original


def test_two_previews_of_one_window_do_not_overwrite_each_other(preview, wired, llm):
    # On a real pull request the window is the identity; on a shared test PR it
    # is not — two source PRs from the same night share a window, and a
    # window-only key would make the second preview edit the first away.
    assert preview.main([
        "--post", "--post-to", "key4hep/k4Bench#106", "--token", "t",
    ]) == 0
    (comment,) = wired[0]["comments"]
    assert "preview-of=key4hep/k4geo#607" in comment.marker
    # The upsert only recognises a marker on the body's first line, so the body
    # carries the same re-keying — and a re-run still edits its own comment.
    assert comment.render().body.startswith(comment.marker + "\n")
    assert "window=2026-06-24..2026-06-25" in comment.marker


def test_a_rerendered_preview_updates_even_though_the_facts_are_the_same(
    preview, monkeypatch
):
    # Production compares the hidden facts digest so a reworded summary of the
    # same regressions re-notifies nobody. A preview iterates on exactly that
    # wording against deliberately unchanged facts, so it compares bodies
    # instead — otherwise improving the renderer would leave the test comment
    # showing the old one. Runs the real upsert, not a stub of it.
    from k4bench.blame import publish as publish_mod
    from k4bench.blame.github import IssueComment

    posted = preview._redirect(_draft(), "key4hep/k4Bench", 106)
    body = posted.render().body
    # The digest line is still in the body, so the two share their facts.
    assert "<!-- k4bench-blame-facts:" in body.split("\n")[1]
    heading = "Possible performance change traced to this pull request"
    already_there = IssueComment(
        42, body.replace(heading, "An older wording"), author="k4bench-bot",
    )
    updated: list[tuple[int, str]] = []
    monkeypatch.setattr(publish_mod, "authenticated_login", lambda _c: "k4bench-bot")
    monkeypatch.setattr(
        publish_mod, "list_issue_comments", lambda *_a: [already_there]
    )
    monkeypatch.setattr(
        publish_mod, "update_issue_comment",
        lambda _c, _r, cid, body: updated.append((cid, body)) or "https://x",
    )
    result = publish_mod.publish(object(), [posted])
    assert result.updated == ["key4hep/k4Bench#106"]
    assert heading in updated[0][1]

    # …and a preview that really is identical still performs no write.
    monkeypatch.setattr(
        publish_mod, "list_issue_comments",
        lambda *_a: [IssueComment(42, body, author="k4bench-bot")],
    )
    assert publish_mod.publish(object(), [posted]).unchanged == ["key4hep/k4Bench#106"]


# ── A preview is a preview of *production* ────────────────────────────────────

def test_no_llm_configured_is_an_error_not_a_different_comment(
    preview, wired, monkeypatch, capsys
):
    from k4bench.blame import attribute as attribute_mod

    monkeypatch.setattr(attribute_mod, "attributor_from_env", lambda: None)
    assert preview.main([]) == 1
    assert not wired
    assert "--ranker-only" in capsys.readouterr().err


def test_ranker_only_asks_for_the_fallback_on_purpose(preview, wired, monkeypatch):
    from k4bench.blame import attribute as attribute_mod

    monkeypatch.setattr(
        attribute_mod, "attributor_from_env",
        lambda: pytest.fail("--ranker-only must not consult the environment"),
    )
    assert preview.main(["--ranker-only"]) == 0
    assert [call["dry_run"] for call in wired] == [True]


# ── References the tool must refuse rather than guess through ─────────────────

@pytest.mark.parametrize(
    "ref",
    ["k4geo#607", "key4hep/k4geo", "key4hep/k4geo#seven", "/k4geo#607", "a/b/c#1"],
)
def test_a_malformed_pull_request_reference_is_refused(preview, ref):
    with pytest.raises(SystemExit) as exc:
        preview.main(["--only", ref])
    assert exc.value.code == 2


def test_a_repo_outside_the_allowlist_renders_nothing(preview, wired, llm, capsys):
    assert preview.main(["--only", "someone/elses-repo#1"]) == 1
    assert not wired
    assert "allowlist" in capsys.readouterr().err
