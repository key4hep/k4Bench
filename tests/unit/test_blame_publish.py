"""Unit tests for :mod:`k4bench.blame.publish` — the one code path that writes
into a repository k4Bench does not own. Every rule that keeps it from spamming a
pull request is asserted here against a recording fake: upsert instead of
append, no edit without a change, and never post into a thread it could not
read."""

from __future__ import annotations

import pytest

from k4bench.blame.comment import PRComment
from k4bench.blame.github import GitHubClient, IssueComment, RateLimitError
from k4bench.blame.publish import publish

_MARKER = "<!-- k4bench-blame-comment:v1 window=2026-07-03..2026-07-04 -->"


def _comment(body: str = f"{_MARKER}\nbody text", number: int = 7) -> PRComment:
    return PRComment(
        repo="key4hep/k4geo", number=number, marker=_MARKER, body=body, score=91.0,
    )


class _FakeGitHub:
    """Records every call, and answers reads from ``threads``.

    ``threads`` maps a PR number to the comments already on it, or to ``None``
    for a thread that cannot be read. ``write_fails`` makes writes return
    ``None``, and ``raises`` makes the read blow up.
    """

    def __init__(self, threads=None, *, write_fails=False, raises=None):
        self.threads = threads or {}
        self.write_fails = write_fails
        self.raises = raises
        self.created: list[tuple[int, str]] = []
        self.updated: list[tuple[int, str]] = []
        self.reads: list[int] = []


@pytest.fixture(autouse=True)
def _patch_github(monkeypatch):
    """Route the module's three GitHub calls to whichever fake a test builds."""
    import k4bench.blame.publish as pub

    def _list(client, slug, number):
        client.reads.append(number)
        if client.raises is not None:
            raise client.raises
        return client.threads.get(number, [])

    def _create(client, slug, number, body):
        client.created.append((number, body))
        return None if client.write_fails else "https://x/new"

    def _update(client, slug, comment_id, body):
        client.updated.append((comment_id, body))
        return None if client.write_fails else "https://x/edited"

    monkeypatch.setattr(pub, "list_issue_comments", _list)
    monkeypatch.setattr(pub, "create_issue_comment", _create)
    monkeypatch.setattr(pub, "update_issue_comment", _update)


def test_posts_when_the_pr_has_no_comment_of_ours():
    gh = _FakeGitHub({7: [IssueComment(1, "an unrelated review comment")]})
    result = publish(gh, [_comment()])
    assert result.created == ["key4hep/k4geo#7"]
    assert gh.created and not gh.updated


def test_edits_in_place_when_the_body_changed():
    # A regression that still stands with a refreshed likelihood is one comment
    # edited, never a second one appended to the thread.
    gh = _FakeGitHub({7: [IssueComment(42, f"{_MARKER}\nyesterday's body")]})
    result = publish(gh, [_comment()])
    assert result.updated == ["key4hep/k4geo#7"]
    assert gh.updated == [(42, f"{_MARKER}\nbody text")]
    assert not gh.created


def test_unchanged_body_performs_no_write_at_all():
    # An edit re-surfaces the comment for everyone watching the PR, so an
    # identical body must not produce one.
    body = f"{_MARKER}\nbody text"
    gh = _FakeGitHub({7: [IssueComment(42, body)]})
    result = publish(gh, [_comment(body)])
    assert result.unchanged == ["key4hep/k4geo#7"]
    assert not gh.created and not gh.updated


def test_unreadable_thread_is_skipped_rather_than_duplicated():
    gh = _FakeGitHub({7: None})
    result = publish(gh, [_comment()])
    assert result.failed == ["key4hep/k4geo#7"]
    assert not gh.created


def test_a_failed_write_is_recorded_not_raised():
    gh = _FakeGitHub({7: []}, write_fails=True)
    result = publish(gh, [_comment()])
    assert result.failed == ["key4hep/k4geo#7"]
    assert result.created == []


def test_one_bad_pr_does_not_stop_the_others():
    gh = _FakeGitHub({7: None, 8: []})
    result = publish(gh, [_comment(number=7), _comment(number=8)])
    assert result.failed == ["key4hep/k4geo#7"]
    assert result.created == ["key4hep/k4geo#8"]


def test_rate_limit_aborts_the_run():
    # Past a rate limit nothing else will succeed; stop rather than hammer.
    gh = _FakeGitHub({7: []}, raises=RateLimitError("throttled"))
    with pytest.raises(RateLimitError):
        publish(gh, [_comment()])


def test_dry_run_writes_nothing_and_says_so():
    gh = _FakeGitHub({7: []})
    result = publish(gh, [_comment()], dry_run=True)
    assert result.planned == ["key4hep/k4geo#7"]
    assert (result.created, result.updated, result.unchanged) == ([], [], [])
    assert not gh.reads  # not even a read: a dry run touches GitHub not at all
    assert "dry run" in result.summary


def test_client_type_is_the_shared_github_client():
    # The fake stands in for a real client; keep the seam honest.
    assert publish(GitHubClient(token=None), [], dry_run=True).planned == []
