"""Put a night's pull-request comments on GitHub — the only writing code path.

:mod:`k4bench.blame.comment` decides *what* is said and to whom; this module
performs the write, and nothing else in the pipeline does. Keeping the write in
one small module means every rule about not spamming someone else's repository
lives in one readable place:

* **Upsert, never append.** A comment is identified by its hidden window marker.
  An exact window edits in place, and a strictly expanding or contracting window
  adopts its nearest comparable version, so a changing finding is still one
  comment. When formerly separate lineages converge into one containing window,
  the most recently updated one becomes the survivor.
* **No pointless edits.** An unchanged comment performs *no* request at all — an
  edit re-surfaces the comment for everyone watching the PR, so it must mean
  something changed. "Unchanged" is judged on the hidden *facts* digest, not on
  the body: part of the body is model prose regenerated every night, and a
  reworded sentence about the same regression is not a change anyone wants a
  notification for.
* **Keep the observations an edit replaces.** A comment is rendered only here,
  against the comparable comments already posted
  (:meth:`~k4bench.blame.comment.CommentDraft.render`), so the new body carries
  forward a compact history of earlier versions, each linked to its archived
  report, and the retained-row state that lets a row confirmed in an earlier
  version stay visible after it stops being confirmed.
* **Never post blind.** If the existing comments could not be read
  (:func:`~k4bench.blame.github.list_issue_comments` returning ``None``), the PR
  is skipped: a duplicate comment is worse than a missing one.
* **Edit only our own comment.** A comment is the bot's own only when its *first
  line* is the marker **and** its author is the token's login — a marker quoted
  somewhere inside a human's comment cannot divert the edit. The login is read
  once per run; if it cannot be established the run **fails closed** and posts
  nothing, because an off-repository write must never fall back to editing a
  comment whose ownership it could not prove.
* **One failure is one PR's failure.** Every per-comment error is caught and
  counted, so a repo the token cannot write to does not silence the others. The
  one exception is :class:`~k4bench.blame.github.RateLimitError`, which stops
  the run — past that point nothing will succeed anyway.

:func:`supersede` is the one write to a comment the night did not select: it
withdraws claims tonight's report has moved out of the pull request's reach
(:mod:`k4bench.blame.comment.supersede`), under the same ownership and
fail-closed rules, and only ever edits.

``dry_run`` short-circuits every read and write and logs the standalone body
instead. Existing observation history is necessarily absent because the thread
is deliberately untouched; everything newly rendered remains reviewable before
a repository is added to the allowlist.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from k4bench.blame.comment import (
    CommentDraft,
    facts_digest_of,
    parse_marker,
    window_contains,
    window_from_marker,
)
from k4bench.blame.comment.supersede import Placement, superseded_body
from k4bench.blame.github import (
    GitHubClient,
    RateLimitError,
    authenticated_login,
    create_issue_comment,
    list_issue_comments,
    update_issue_comment,
)

_log = logging.getLogger(__name__)

#: GitHub's hard limit on an issue-comment body, in bytes. Over it the write is
#: rejected outright rather than truncated, so the final body is measured here —
#: the renderer's row caps are sized to stay well clear, and a body that reaches
#: this is a bug worth naming rather than a 422 to decode after the fact.
_MAX_COMMENT_BYTES = 65536


@dataclass
class PublishResult:
    """What the run did, in the four outcomes worth telling apart.

    ``unchanged`` is a success, not a no-op to be fixed: it is the steady state
    of a regression that has been standing for days — the benchmark facts are
    the same tonight, whatever words the model chose for them this time.
    """

    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    #: Targets a dry run *would* have written to — kept apart from the three
    #: real outcomes so a dry run can never be read as having posted anything.
    planned: list[str] = field(default_factory=list)
    #: Comments :func:`supersede` edited to withdraw moved claims.
    superseded: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.planned:
            return f"{len(self.planned)} comment(s) planned (dry run, nothing written)"
        text = (
            f"{len(self.created)} created, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.failed)} failed"
        )
        if self.superseded:
            text += f", {len(self.superseded)} superseded"
        return text


def publish(
    client: GitHubClient, comments: list[CommentDraft], *, dry_run: bool = False
) -> PublishResult:
    """Upsert every comment in *comments*, returning what happened to each.

    Raises :class:`RateLimitError` — and only that — to abort the run; every
    other failure is recorded against its own pull request and the rest
    continue.
    """
    result = PublishResult()
    login = None
    if not dry_run:
        login = _own_login(client)
        if login is None:
            result.failed.extend(c.target for c in comments)
            return result
    for comment in comments:
        if dry_run:
            rendered = comment.render()
            _log.info(
                "publish: [dry run] would comment on %s (likelihood %d%%):\n%s",
                rendered.target, round(rendered.score), rendered.body,
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


def _own_login(client: GitHubClient) -> str | None:
    """The token's own login, or ``None`` when it cannot be established.

    Resolved once per run: it identifies the bot across every repo, so a write
    edits only a comment this token itself wrote. Failing to read it is
    fail-closed — an off-repo write must not guess at ownership — but it is a
    soft failure, logged, never raised; only a rate limit propagates."""
    try:
        login = authenticated_login(client)
    except RateLimitError:
        # The one failure that aborts the run, here as everywhere else:
        # nothing after this point can succeed.
        raise
    except Exception as exc:  # noqa: BLE001 — identity failures must fail closed
        # A timeout, a connection reset, a proxy's HTML error page: the
        # function returns ``None`` for a *read* it understood, and raises for
        # one it never completed. Both mean the same thing here — the bot
        # cannot prove what it owns — and neither may escape a publisher whose
        # contract is that only a rate limit stops the run.
        _log.error(
            "publish: could not establish the bot's own login (GET /user "
            "raised %s: %s) — refusing to edit comments it cannot prove it "
            "owns; posting nothing", type(exc).__name__, exc,
        )
        return None
    if login is None:
        _log.error(
            "publish: could not establish the bot's own login (GET /user) — "
            "refusing to edit comments it cannot prove it owns; posting nothing"
        )
    return login


def supersede(
    client: GitHubClient,
    targets: list[tuple[str, int]],
    *,
    suite: str,
    tonight: Mapping[tuple, Placement],
    min_score: float,
    report_night: str,
    max_edits: int,
    dry_run: bool = False,
) -> PublishResult:
    """Withdraw the claims tonight's report moved out of each target pull
    request's reach, in the bot's own *suite* comments there.

    Each comment is judged on its own hidden state
    (:func:`~k4bench.blame.comment.supersede.superseded_body`); one that makes
    no newly superseded claim is not touched. Only edits — a pull request with
    no such comment gets nothing. Every thread is read before anything is
    written, and more edits than *max_edits* write none of them: a night that
    would withdraw that much is a sign the re-dating itself is wrong. A thread
    that cannot be read, or an edit that fails, is counted as failed; the next
    night's pass finds the same claims again, since nothing about them changed.
    Ownership is proven exactly as :func:`publish` proves it, and a dry run
    reads and writes nothing."""
    result = PublishResult()
    if dry_run:
        for repo, number in targets:
            _log.info(
                "supersede: [dry run] would check %s#%s for claims moved out "
                "of its window", repo, number,
            )
            result.planned.append(f"{repo}#{number}")
        return result
    login = _own_login(client)
    if login is None:
        result.failed.extend(f"{repo}#{number}" for repo, number in targets)
        return result

    edits: list[tuple[str, str, int, str]] = []
    for repo, number in targets:
        target = f"{repo}#{number}"
        try:
            existing = list_issue_comments(client, repo, number)
        except RateLimitError:
            raise
        except Exception as exc:  # noqa: BLE001 — one PR must not stop the rest
            _log.warning("supersede: could not read %s — %s", target, exc)
            result.failed.append(target)
            continue
        if existing is None:
            _log.warning("supersede: could not read %s's comments — skipping", target)
            result.failed.append(target)
            continue
        for comment in existing:
            if comment.author != login:
                continue
            body = superseded_body(
                comment.body, pr=(repo, number), suite=suite, tonight=tonight,
                min_score=min_score, report_night=report_night,
            )
            if body is None:
                continue
            if len(body.encode()) > _MAX_COMMENT_BYTES:
                _log.warning(
                    "supersede: %s comment %s would be %d bytes — not written",
                    target, comment.id, len(body.encode()),
                )
                result.failed.append(target)
                continue
            edits.append((target, repo, comment.id, body))

    if len(edits) > max_edits:
        _log.error(
            "supersede: %d withdrawals exceed the cap of %d — writing none of "
            "them (%s)", len(edits), max_edits,
            ", ".join(sorted({target for target, *_ in edits})),
        )
        result.failed.extend(sorted({target for target, *_ in edits}))
        return result
    for target, repo, comment_id, body in edits:
        try:
            url = update_issue_comment(client, repo, comment_id, body)
        except RateLimitError:
            raise
        except Exception as exc:  # noqa: BLE001 — one PR must not stop the rest
            _log.warning("supersede: %s failed — %s", target, exc)
            result.failed.append(target)
            continue
        if url is None:
            result.failed.append(target)
            continue
        _log.info("supersede: withdrew moved claims on %s — %s", target, url)
        result.superseded.append(target)
    _log.info("supersede: %s", result.summary)
    return result


def _upsert(
    client: GitHubClient,
    draft: CommentDraft,
    result: PublishResult,
    *,
    login: str,
) -> None:
    """Create, edit, or leave alone the bot's comment for this window lineage.

    A comment counts as the bot's own only when its *first line* is the marker
    (the shape :func:`~k4bench.blame.comment.render._render` always produces) **and** it
    was written by *login* — a marker quoted inside someone else's comment
    matches neither test, so it cannot divert the edit. An exact marker wins;
    otherwise the nearest containing or contained window of the same suite is
    migrated by replacing its body, including its first-line marker — another
    suite's comment is never adopted, however its window compares. If formerly separate lineages are
    equally near, the most recently updated is the deterministic survivor; the
    others remain as dated historical comments. Duplicate exact markers still
    fail closed because they already violate the marker's uniqueness invariant,
    as do duplicates *at the nearest lineage window* — but duplicates at a more
    distant comparable window, with one unique nearest candidate standing
    between them and the current window, are logged and ignored rather than
    allowed to hold the pull request unwritable.

    Whether it needs rewriting is decided on the hidden facts digest
    (:func:`~k4bench.blame.comment.facts_digest_of`) rather than the body, so a
    freshly-worded summary of the same regressions is left alone. Before that
    comparison, the body is rendered
    (:meth:`~k4bench.blame.comment.CommentDraft.render`) against the comparable
    prior bodies, whose observations and retained rows it carries forward; the
    digest it computes still wins when available, so an unchanged new report
    date alone performs no edit. Every comment this module
    writes carries a digest; a standing comment whose digest line cannot be read
    falls back to comparing whole bodies, which errs towards an edit rather than
    towards leaving a stale body in place."""
    existing = list_issue_comments(client, draft.repo, draft.number)
    if existing is None:
        _log.warning(
            "publish: could not read %s's comments — skipping rather than "
            "risking a duplicate", draft.target,
        )
        result.failed.append(draft.target)
        return

    owned = [c for c in existing if c.author == login]
    exact = [
        c for c in owned if c.body.partition("\n")[0] == draft.marker
    ]
    if len(exact) > 1:
        # Two comments the bot owns for one window: the upsert's identity
        # assumption is broken, and editing an arbitrary one leaves the other
        # standing with stale reasoning. Neither writing nor guessing is safe,
        # so the pull request is skipped and the duplicates are named for a
        # human to remove.
        _log.error(
            "publish: %s carries %d comments with this window's marker "
            "(ids %s) — editing an arbitrary one would leave the others "
            "stale; skipping",
            draft.target, len(exact), ", ".join(str(c.id) for c in exact),
        )
        result.failed.append(draft.target)
        return

    existing_comment = exact[0] if exact else None
    migrating = False
    lineage_bodies: list[str] = []
    # Only the draft's own suite's comments form its lineage: another suite's
    # marker parses as no window here.
    parsed = parse_marker(draft.marker)
    if existing_comment is None and parsed is not None:
        suite, current_window = parsed
        lineage = [
            (c, window)
            for c in owned
            if (window := window_from_marker(c.body.partition("\n")[0], suite)) is not None
            and window != current_window
            and (
                window_contains(current_window, window)
                or window_contains(window, current_window)
            )
        ]
        nearest = [
            (candidate, window)
            for candidate, window in lineage
            if not any(
                other_window != window
                and (
                    (
                        window_contains(current_window, other_window)
                        and window_contains(other_window, window)
                    )
                    or (
                        window_contains(window, other_window)
                        and window_contains(other_window, current_window)
                    )
                )
                for _, other_window in lineage
            )
        ]
        # The duplicate-window invariant is enforced on the *nearest* set, not
        # the whole lineage: two comments at the nearest window make the
        # migration target genuinely ambiguous, while duplicates at a more
        # distant comparable window — already superseded by a unique nearer
        # comment — are an anomaly for a human to clean up, not a reason to
        # leave this pull request permanently unwritable.
        duplicate_nearest = [
            candidate
            for candidate, window in nearest
            if sum(other_window == window for _, other_window in nearest) > 1
        ]
        if duplicate_nearest:
            _log.error(
                "publish: %s carries duplicate comments at its nearest lineage "
                "window (ids %s) — editing an arbitrary one would leave the "
                "others stale; skipping",
                draft.target,
                ", ".join(str(candidate.id) for candidate in duplicate_nearest),
            )
            result.failed.append(draft.target)
            return
        duplicated_windows = {
            window
            for _, window in lineage
            if sum(other_window == window for _, other_window in lineage) > 1
        }
        if duplicated_windows:
            _log.warning(
                "publish: %s carries duplicate comments at distant lineage "
                "window(s) (ids %s) — ignoring them and migrating the unique "
                "nearest comment; the duplicates need a human to remove",
                draft.target,
                ", ".join(
                    str(candidate.id)
                    for candidate, window in lineage
                    if window in duplicated_windows
                ),
            )
        if nearest:
            existing_comment = max(
                (candidate for candidate, _ in nearest),
                key=lambda candidate: (candidate.updated_at, candidate.id),
            )
            migrating = True
            if len(nearest) > 1:
                _log.warning(
                    "publish: %s has %d equally near lineage comments (ids %s); "
                    "migrating most recently updated id %s and leaving the "
                    "others as historical comments",
                    draft.target,
                    len(nearest),
                    ", ".join(str(candidate.id) for candidate, _ in nearest),
                    existing_comment.id,
                )
            # Converging lineages hand their retained rows to the survivor: the
            # survivor's body first, then every other comparable body whose
            # window is unique (the flagged duplicates stay out — which copy of
            # an anomaly to trust is not a call worth guessing). Only the new
            # body is written; the other comments stand untouched, so no marker
            # is ever duplicated by the merge.
            lineage_bodies = [existing_comment.body] + [
                candidate.body
                for candidate, window in lineage
                if candidate.id != existing_comment.id
                and window not in duplicated_windows
            ]

    if not lineage_bodies and existing_comment is not None:
        lineage_bodies = [existing_comment.body]
    comment = draft.render(lineage_bodies)
    if len(comment.body.encode()) > _MAX_COMMENT_BYTES:
        # Measure only after carrying the existing observation history forward:
        # that is the actual body GitHub would receive.
        _log.warning(
            "publish: %s's body is %d bytes, over GitHub's %d-byte comment "
            "limit — not written", comment.target,
            len(comment.body.encode()), _MAX_COMMENT_BYTES,
        )
        result.failed.append(comment.target)
        return

    if existing_comment is None:
        url = create_issue_comment(client, comment.repo, comment.number, comment.body)
        if url is None:
            result.failed.append(comment.target)
            return
        _log.info("publish: commented on %s — %s", comment.target, url)
        result.created.append(comment.target)
        return

    posted_digest = facts_digest_of(existing_comment.body)
    if not migrating and (
        posted_digest == comment.facts_digest
        if posted_digest and comment.facts_digest
        else existing_comment.body == comment.body
    ):
        _log.info("publish: %s already says this — no edit", comment.target)
        result.unchanged.append(comment.target)
        return

    url = update_issue_comment(
        client, comment.repo, existing_comment.id, comment.body
    )
    if url is None:
        result.failed.append(comment.target)
        return
    _log.info("publish: updated the comment on %s — %s", comment.target, url)
    result.updated.append(comment.target)
