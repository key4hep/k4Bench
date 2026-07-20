"""Put a night's pull-request comments on GitHub — the only writing code path.

:mod:`k4bench.blame.comment` decides *what* is said and to whom; this module
performs the write, and nothing else in the pipeline does. Keeping the write in
one small module means every rule about not spamming someone else's repository
lives in one readable place:

* **Upsert, never append.** A comment is identified by its hidden marker, so a
  regression that stands for a week is one comment edited nightly, not seven.
* **No pointless edits.** An unchanged body performs *no* request at all — an
  edit re-surfaces the comment for everyone watching the PR, so it must mean
  something changed.
* **Never post blind.** If the existing comments could not be read
  (:func:`~k4bench.blame.github.list_issue_comments` returning ``None``), the PR
  is skipped: a duplicate comment is worse than a missing one.
* **Edit only our own comment.** The marker recognises the comment, but a comment
  is claimed as the bot's own only when its author is the token's login too, so
  someone quoting the hidden marker cannot make the bot try to PATCH a comment it
  does not own. The login is read once per run; if it cannot be read the check is
  skipped and the marker alone identifies the comment, as before.
* **One failure is one PR's failure.** Every per-comment error is caught and
  counted, so a repo the token cannot write to does not silence the others. The
  one exception is :class:`~k4bench.blame.github.RateLimitError`, which stops
  the run — past that point nothing will succeed anyway.

``dry_run`` short-circuits every write and logs the exact body instead, which is
how the bot is verified before a repository is added to the allowlist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from k4bench.blame.comment import PRComment
from k4bench.blame.github import (
    GitHubClient,
    RateLimitError,
    authenticated_login,
    create_issue_comment,
    list_issue_comments,
    update_issue_comment,
)

_log = logging.getLogger(__name__)


@dataclass
class PublishResult:
    """What the run did, in the four outcomes worth telling apart.

    ``unchanged`` is a success, not a no-op to be fixed: it is the steady state
    of a regression that has been standing for days.
    """

    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    #: Targets a dry run *would* have written to — kept apart from the three
    #: real outcomes so a dry run can never be read as having posted anything.
    planned: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.planned:
            return f"{len(self.planned)} comment(s) planned (dry run, nothing written)"
        return (
            f"{len(self.created)} created, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.failed)} failed"
        )


def publish(
    client: GitHubClient, comments: list[PRComment], *, dry_run: bool = False
) -> PublishResult:
    """Upsert every comment in *comments*, returning what happened to each.

    Raises :class:`RateLimitError` — and only that — to abort the run; every
    other failure is recorded against its own pull request and the rest
    continue.
    """
    result = PublishResult()
    # Resolved once for the whole run: it identifies the bot across every repo,
    # so the per-comment upsert edits only a comment this token itself wrote.
    login = None if dry_run else authenticated_login(client)
    for comment in comments:
        if dry_run:
            _log.info(
                "publish: [dry run] would comment on %s (likelihood %d%%):\n%s",
                comment.target, round(comment.score), comment.body,
            )
            result.planned.append(comment.target)
            continue
        try:
            _upsert(client, comment, result, login=login)
        except RateLimitError:
            raise
        except Exception as exc:  # noqa: BLE001 — one PR must not stop the rest
            _log.warning("publish: %s failed — %s", comment.target, exc)
            result.failed.append(comment.target)
    _log.info("publish: %s", result.summary)
    return result


def _upsert(
    client: GitHubClient,
    comment: PRComment,
    result: PublishResult,
    *,
    login: str | None,
) -> None:
    """Create, edit, or leave alone the one comment carrying *comment*'s marker.

    When *login* is known, a comment counts as the bot's own only if it carries
    the marker *and* was written by that login, so a quoted marker in someone
    else's comment cannot divert the edit."""
    existing = list_issue_comments(client, comment.repo, comment.number)
    if existing is None:
        _log.warning(
            "publish: could not read %s's comments — skipping rather than "
            "risking a duplicate", comment.target,
        )
        result.failed.append(comment.target)
        return

    mine = next(
        (
            c for c in existing
            if comment.marker in c.body and (login is None or c.author == login)
        ),
        None,
    )
    if mine is None:
        url = create_issue_comment(client, comment.repo, comment.number, comment.body)
        if url is None:
            result.failed.append(comment.target)
            return
        _log.info("publish: commented on %s — %s", comment.target, url)
        result.created.append(comment.target)
        return

    if mine.body == comment.body:
        _log.info("publish: %s already says this — no edit", comment.target)
        result.unchanged.append(comment.target)
        return

    url = update_issue_comment(client, comment.repo, mine.id, comment.body)
    if url is None:
        result.failed.append(comment.target)
        return
    _log.info("publish: updated the comment on %s — %s", comment.target, url)
    result.updated.append(comment.target)
