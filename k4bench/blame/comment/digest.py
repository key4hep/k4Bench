"""What counts as a change worth re-notifying a pull request.

The publisher edits a standing comment only when the digest over the comment's
*benchmark facts* changes (:func:`_facts_payload`), and an edit re-notifies
everyone subscribed to the pull request. So the rule is two-sided: a fact that
changes what the comment claims must be in, and a number that moves on its own
every night must be out. Model prose and model scores are never hashed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from k4bench.blame.attribute import AttributionRequest
from k4bench.blame.comment.markdown import (
    _MAX_DESCRIPTION_CHARS,
    _one_line,
    _pct,
)
from k4bench.blame.comment.selection import (
    CommentPlan,
    _row_identity,
)
from k4bench.blame.comment.state import (
    RetainedRow,
    _retained_identity,
    _retained_window,
)
from k4bench.blame.reproduce import ReproducerFacts


@dataclass(frozen=True)
class Reproducers:
    """The recipes published for one comment, keyed by row identity.

    Two sets, because rendering and hashing want different things.

    ``urls`` is every recipe published, and it covers **both** the rows the
    table shows — a reader following a line wants the commands for *that*
    configuration — and the deterministic set below. ``hashed`` names the
    deterministic subset alone (:func:`_recipe_rows`), and ``facts`` holds
    those recipes' commands; only those two reach the facts digest.

    The split is what lets both properties hold at once. The table is ranked by
    model likelihood, so hashing every published recipe would let two scores
    swapping places edit a standing comment and re-notify the pull request on
    model drift — the one thing the digest exists to prevent. Hashing only the
    deterministic subset keeps the digest a statement about benchmark facts,
    while the rendered table still links a recipe on every row it shows. A link
    that appears because the ranking moved therefore shows up on the next edit
    some real change earns, exactly as a row that entered the table the same
    way already does."""

    urls: dict[tuple, str]
    hashed: tuple[tuple, ...]
    facts: tuple[ReproducerFacts, ...]


def _digest_from(
    payload: dict[str, Any], retained: list[dict[str, Any]]
) -> str:
    """The facts digest over *payload* plus the rendered retained rows.

    The retained half is known once the prior bodies say which historical rows
    actually render: their frozen facts and their standing in tonight's report
    are deterministic inputs a reader can see, so a change in them must be able
    to edit a standing comment — while a retained row's snapshot never rescores,
    so no model drift comes back in through this door. A comment with no
    surviving history hashes an empty list."""
    data = dict(payload)
    data["retained"] = retained
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _facts_payload(
    plan: CommentPlan, request: AttributionRequest | None = None,
    reproduce: Reproducers | None = None,
) -> dict[str, Any]:
    """The *benchmark facts* behind a comment, as the digest's canonical payload.

    The publisher edits a standing comment only when the digest over this
    changes (:func:`k4bench.blame.publish._upsert`), and an edit re-notifies
    everyone subscribed to the pull request. So the rule is two-sided, and both
    sides matter: a fact that changes what the comment claims must be in, and a
    number that moves on its own every night must be out.

    **In**: the window; every regression row's identity — platform included —
    how far it moved, and its own change window (the Change window column and
    each row's deep link route through it, and a row's window can move while
    the plan's outer window stands still); what the first pass knew about this pull request
    in each of those scopes; the clean and watch outcomes with their watched
    metrics and unjudged counts; the per-platform package diff and unchanged
    counts; which pull requests were in the field and whether each was judged;
    which older-boundary pull requests the first pass read before scoring
    (:attr:`CommentPlan.historical_refs`); and whether the review's evidence —
    the subject's and competitors' diffs — could actually be fetched.

    The outcomes especially. A comment posted while IDEA had no reliable result
    reads differently once IDEA delivers a clean measurement of the same window
    — that control weakens the attribution and the review is shown it — and a
    digest covering only the positive rows would leave the old reasoning
    standing on the pull request forever, because nothing it hashed had moved.
    Diff availability is the same argument: a night where GitHub refused the
    patch produces a review made from paths and titles, and the night the fetch
    succeeds is a materially better-evidenced comment, not a reworded one.

    **Out**: the narrative, and every model score — the review's likelihoods and
    the ranker's scoring alike. Those drift between nights without anything
    having happened, and a competitor sliding from 84.4 to 84.6 is not worth
    re-notifying everyone watching a pull request. (Whether a candidate was
    scored *at all* is a different thing, and is in: it changes the table cell
    and the prompt from a percentage to "not scored".)

    **Also out, and less obviously so**: ``value``, ``baseline_median`` and
    ``z_score``. They are deterministic and they do reach the review's prompt,
    but they are *tonight's* measurement of a standing regression — the engine
    re-derives them from the latest run every night, so they move whenever the
    benchmark is re-run, which is nightly. Hashing them would edit every
    standing comment every night, which is the exact harm this digest exists to
    prevent. ``pct_change`` is the same kind of number and is included only at
    the precision the comment *displays* it (:func:`_canonical_pct`), so the
    digest changes when the visible table does and not before.

    Serialized as canonical JSON rather than joined strings so a field's value
    can never migrate into its neighbour's — ``a|b`` and ``a`` + ``|b`` hash
    alike, and identities here are user-supplied names.
    """
    payload = {
        # Repair existing comments once after a change to what the rendered
        # body says. Report dates remain excluded, so later reconfirmations do
        # not edit.
        "render_version": 4,
        "window": [plan.base_release or "", plan.onset_release],
        "rows": [
            {
                "id": list(_row_identity(row)),
                "moved": _canonical_pct(row.verdict.pct_change),
                "state": row.scope_state,
                # Both halves of the onset identity: the date is the grouping
                # key and the right-hand end of the visible Change window cell,
                # the run id distinguishes separate steps inside one release.
                # Fixed once a change is confirmed, so neither is nightly churn.
                "onset": [
                    row.verdict.onset_run_date or "",
                    row.verdict.onset_run_id or "",
                ],
                # The other end of that cell. A row's own base moves when the
                # metric's last accepted release does, which the comment's
                # window marker cannot see.
                "base": row.verdict.last_accepted_run_date or "",
            }
            for row in sorted(plan.rows, key=_row_identity)
        ],
        "outcomes": [
            {
                "scope": [o.detector, o.platform, o.sample, o.label],
                "status": o.status,
                "watched": list(o.watched),
                "unjudged": o.unjudged,
            }
            for o in plan.outcomes
        ],
        "packages": {
            platform: [[p.package, p.status] for p in packages]
            for platform, packages in plan.packages_by_platform.items()
        },
        "unchanged": dict(sorted(plan.unchanged.items())),
        "packages_unavailable_on": list(plan.packages_unavailable_on),
        # Listed in *identity* order, never :func:`_sorted_others`' strength
        # order. That helper ranks by score, so two competitors trading places
        # — 85/80 one night, 78/82 the next — would reorder this list and move
        # the hash, smuggling back in exactly the model-score drift the payload
        # is careful never to name. Which pull requests were in the field is the
        # fact; where the ranker put them is not.
        "competitors": [
            {
                "pr": f"{other.repo}#{other.number}",
                "ranked": other.ranked,
                # Rendered verbatim in the "other pull requests" table, so a
                # retitled candidate is a changed comment — and lives here
                # rather than in ``evidence`` because that table is drawn
                # whether or not a review ran.
                "title": _one_line(other.title, _MAX_DESCRIPTION_CHARS),
            }
            for other in sorted(
                (candidate for candidate, _scope in plan.others.values()),
                key=lambda c: (c.repo.lower(), c.number),
            )
        ],
        # Which older-boundary pull requests the first pass read before it
        # scored this window, and which the review is therefore shown too. A
        # different set is a materially different basis for both passes — the
        # review is calibrated against different code — so a standing comment
        # must be able to be replaced when it changes. Identities only: no
        # score, no patch, nothing that drifts on its own.
        "historical": [
            {
                "pr": f"{ref.repo}#{ref.pr}",
                "package": ref.package,
                "boundary": [ref.base_release, ref.onset_release],
            }
            for ref in plan.historical_refs
        ],
        "evidence": _evidence_facts(request),
        # Both halves of every recipe: the commands, so a changed workload
        # edits the comment, and the URL, so a link that appeared (or moved)
        # does too. A run of the same night that failed to publish renders no
        # link, and must not share a digest with one that did. Restricted to
        # the model-independent subset (:class:`Reproducers`) and taken in
        # identity order, so neither the ranking nor the publish order can move
        # this.
        "reproduce": None if reproduce is None else {
            "recipes": [facts.payload() for facts in reproduce.facts],
            "urls": [
                [list(identity), reproduce.urls[identity]]
                for identity in reproduce.hashed
            ],
        },
    }
    if plan.external_changes:
        payload["external_changes"] = [
            [c.scope, c.name, c.before, c.after, c.kind] for c in sorted(plan.external_changes)
        ]
    return payload


def _evidence_facts(request: AttributionRequest | None) -> dict[str, Any] | None:
    """Which of the review's inputs this night could actually assemble.

    Diff *availability*, not diff content: a merged pull request's diff does not
    change, so hashing the text would add nothing a boolean does not already say
    while making the digest sensitive to incidental churn in how the patch was
    clipped. What genuinely varies is whether GitHub answered — and that decides
    whether the model reasoned about code or about file names.

    Titles, paths and sizes ride along because they are shown to the model too
    and are fixed for a merged pull request, so they cost nothing and catch a
    candidate whose metadata was read differently — a retitled pull request is
    a different prompt, and the reviewer's account of it can legitimately
    change. ``None`` when no review was assembled at all: a comment rendered
    from the first pass alone rests on no such evidence, and configuring a model
    later is itself a change of basis."""
    if request is None:
        return None
    return {
        "subject": {
            "title": _one_line(request.title, _MAX_DESCRIPTION_CHARS),
            "files": list(request.files),
            "size": [request.additions, request.deletions],
            "patch": bool(request.patch),
        },
        "competitors": [
            {
                "pr": f"{c.repo}#{c.number}",
                "title": _one_line(c.title, _MAX_DESCRIPTION_CHARS),
                "files": list(c.files),
                "size": [c.additions, c.deletions],
                "patch": bool(c.patch),
            }
            for c in sorted(request.competitors, key=lambda c: (c.repo, c.number))
        ],
    }


def _canonical_pct(pct: float | None) -> str:
    """A step size at the precision the comment shows it.

    ``pct_change`` is re-measured every night, so a standing regression's step
    wobbles a little from one run to the next. Quantizing to what
    :func:`_change_cell` actually prints ties the digest to the rendered table:
    a wobble too small to change a single character of the comment produces no
    edit, and a real move produces one. Hashing the raw float instead — even to
    four decimals, ten times finer than the display — edits a comment whose
    visible body is byte-for-byte identical, which is the most pointless
    notification this bot can send."""
    if pct is None or not math.isfinite(pct):
        return "-"
    return f"{pct:+.1%}"


def _retained_fact(row: RetainedRow) -> dict[str, Any]:
    """One rendered historical row as the digest hashes it: its frozen snapshot
    — identity, movement, window, link-routing fields, confirmation night, and
    the recorded likelihood at the precision the table displays it."""
    return {
        "id": list(_retained_identity(row)),
        "moved": _canonical_pct(row.pct),
        "direction": row.direction,
        "onset": row.onset,
        "shown_window": list(_retained_window(row)),
        "recipe": row.recipe,
        "window": [row.base_release or "", row.onset_release],
        "runs": [row.base_run, row.onset_run],
        "stack": row.stack,
        "night": row.last_reported,
        "likelihood": None if row.likelihood is None else _pct(row.likelihood),
        "source": row.source,
        "state": row.state,
    }
