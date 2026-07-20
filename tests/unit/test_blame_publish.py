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
_BOT = "k4bench-bot"


def _comment(body: str = f"{_MARKER}\nbody text", number: int = 7) -> PRComment:
    return PRComment(
        repo="key4hep/k4geo", number=number, marker=_MARKER, body=body, score=91.0,
    )


def _mine(comment_id: int, body: str) -> IssueComment:
    """A comment the bot itself wrote — carries the marker *and* its login."""
    return IssueComment(comment_id, body, author=_BOT)


class _FakeGitHub:
    """Records every call, and answers reads from ``threads``.

    ``threads`` maps a PR number to the comments already on it, or to ``None``
    for a thread that cannot be read. ``write_fails`` makes writes return
    ``None``, and ``raises`` makes the read blow up. ``login`` is the identity
    ``authenticated_login`` reports for this token (``None`` = could not read it).
    """

    def __init__(self, threads=None, *, write_fails=False, raises=None, login=_BOT):
        self.threads = threads or {}
        self.write_fails = write_fails
        self.raises = raises
        self.login = login
        self.created: list[tuple[int, str]] = []
        self.updated: list[tuple[int, str]] = []
        self.reads: list[int] = []
        self.logins = 0


@pytest.fixture(autouse=True)
def _patch_github(monkeypatch):
    """Route the module's GitHub calls to whichever fake a test builds."""
    import k4bench.blame.publish as pub

    def _login(client):
        client.logins += 1
        return client.login

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

    monkeypatch.setattr(pub, "authenticated_login", _login)
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
    gh = _FakeGitHub({7: [_mine(42, f"{_MARKER}\nyesterday's body")]})
    result = publish(gh, [_comment()])
    assert result.updated == ["key4hep/k4geo#7"]
    assert gh.updated == [(42, f"{_MARKER}\nbody text")]
    assert not gh.created


def test_unchanged_body_performs_no_write_at_all():
    # An edit re-surfaces the comment for everyone watching the PR, so an
    # identical body must not produce one.
    body = f"{_MARKER}\nbody text"
    gh = _FakeGitHub({7: [_mine(42, body)]})
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


def test_a_quoted_marker_from_another_author_is_not_edited():
    # Someone pasting the hidden marker into their own comment must not divert
    # the edit: with no comment of the bot's own on the thread, it posts a fresh
    # one rather than trying (and failing) to PATCH a comment it does not own.
    quoted = IssueComment(9, f"look what the bot said: {_MARKER}", author="mallory")
    gh = _FakeGitHub({7: [quoted]})
    result = publish(gh, [_comment()])
    assert result.created == ["key4hep/k4geo#7"]
    assert not gh.updated


def test_the_login_is_resolved_once_for_the_whole_run():
    gh = _FakeGitHub({7: [], 8: []})
    publish(gh, [_comment(number=7), _comment(number=8)])
    assert gh.logins == 1


def test_an_unreadable_login_fails_closed_and_posts_nothing():
    # An off-repository write must never guess at ownership: if the bot cannot
    # establish its own login, it edits nothing and reads no thread at all.
    gh = _FakeGitHub({7: [_mine(42, f"{_MARKER}\nyesterday")]}, login=None)
    result = publish(gh, [_comment()])
    assert result.failed == ["key4hep/k4geo#7"]
    assert not gh.created and not gh.updated and not gh.reads


def test_a_marker_only_in_the_body_not_the_first_line_is_not_ours():
    # The marker identifies our comment only as its first line; the same string
    # quoted deeper in a comment (even one authored by the bot) is not a match.
    quoted = IssueComment(9, f"as noted:\n{_MARKER}\ntext", author=_BOT)
    gh = _FakeGitHub({7: [quoted]})
    result = publish(gh, [_comment()])
    assert result.created == ["key4hep/k4geo#7"]
    assert not gh.updated


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


# ── The facts digest ──────────────────────────────────────────────────────────
# Part of a comment is model prose, regenerated every night and never repeating
# itself word for word. What decides an edit is the hidden digest of the
# *benchmark facts* underneath it, so a rephrased summary of the same
# regressions notifies nobody.

def _digested(digest: str, text: str, *, number: int = 7) -> PRComment:
    body = f"{_MARKER}\n<!-- k4bench-blame-facts:{digest} -->\n{text}"
    return PRComment(
        repo="key4hep/k4geo", number=number, marker=_MARKER, body=body,
        score=91.0, facts_digest=digest,
    )


def test_the_same_facts_worded_differently_are_not_edited():
    posted = _digested("abc123", "Only ALLEGRO moved.")
    gh = _FakeGitHub({7: [_mine(42, posted.body)]})
    result = publish(gh, [_digested("abc123", "ALLEGRO alone shows the step.")])
    assert result.unchanged == ["key4hep/k4geo#7"]
    assert not gh.updated


def test_changed_facts_are_edited():
    gh = _FakeGitHub({7: [_mine(42, _digested("abc123", "Only ALLEGRO moved.").body)]})
    result = publish(gh, [_digested("def456", "Only ALLEGRO moved.")])
    assert result.updated == ["key4hep/k4geo#7"]


def test_a_comment_from_before_digests_existed_takes_one_upgrade_edit():
    # Nothing to compare against, so the old whole-body rule applies and the
    # standing comment is rewritten once — into a body that does carry a digest.
    gh = _FakeGitHub({7: [_mine(42, f"{_MARKER}\nan older body")]})
    result = publish(gh, [_digested("abc123", "Only ALLEGRO moved.")])
    assert result.updated == ["key4hep/k4geo#7"]
    assert "k4bench-blame-facts:abc123" in gh.updated[0][1]
