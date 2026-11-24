"""Claims a posted comment makes that a later report has moved out of the pull
request's reach.

A comment attributes regressions to a pull request in one change window. The
regression engine can later re-date a step: a late confirmation moves its window
back onto an earlier release (:mod:`k4bench.regression`), and the step is then
measured *before* the window the comment was written for. The pull request
shipped inside that window, so it cannot have caused the step, and the complete
candidate search for the step confirms it by not listing the pull request.
Everything here is deterministic — no model is asked.

A claim is **superseded** when both hold:

* tonight's report places the step's onset at or before the comment window's
  base release — a step placed *after* the window is a new step, and the old
  claim is left as the record of what was measured then;
* tonight's sidecar entry for the step has complete candidate discovery, and the
  pull request is not among its candidates.

A withdrawal is permanent state of the comment's lineage. Every withdrawn
identity is recorded, with the night it was withdrawn on, in a hidden marker
(:func:`~k4bench.blame.comment.state._withdrawn_marker`), and one note
regenerated from that whole record opens the visible text
(:func:`withdrawal_block`), so withdrawals on different nights add up rather
than replace each other.

Two places act on it. A comment rendered tonight carries its lineage's record
forward, adds tonight's withdrawals and leaves their scores out
(:func:`~k4bench.blame.comment.render._render`). A comment *not* rendered
tonight — its pull request no longer selected — is found from the recent
sidecars (:func:`supersede_targets`) and edited in place
(:func:`superseded_body`): the note above the posted text, and the scores
cleared from its hidden state, so no later version resurfaces them. The rest of
the posted body is kept as the record it is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace

from k4bench.blame.comment.markdown import REGRESSION, _count
from k4bench.blame.comment.state import (
    _CUMULATIVE_PREFIX,
    _RETAINED_PREFIX,
    _WITHDRAWN_PREFIX,
    ChangeWindow,
    CumulativeEntry,
    _cumulative_marker,
    _decoded_cumulative,
    _decoded_retained,
    _decoded_withdrawn,
    _retained_identity,
    _retained_marker,
    _verdict_identity,
    _withdrawn_marker,
    window_from_marker,
)
from k4bench.blame.models import BlameReport, CandidatePR
from k4bench.regression.models import NightlyReport

#: The first visible line of the withdrawal note — how :func:`superseded_body`
#: finds the note it replaces.
_NOTE_LEAD = "> **Withdrawn"


@dataclass(frozen=True)
class Placement:
    """Where tonight's report places one confirmed step: its onset release, and
    the pull requests tonight's candidate search found for it — ``None`` when
    the sidecar has no complete entry, from which nothing follows."""

    onset: str | None
    candidates: frozenset[tuple[str, int]] | None


def placements(report: NightlyReport, blame: BlameReport) -> dict[tuple, Placement]:
    """Every confirmed step of tonight's report, keyed by row identity."""
    out: dict[tuple, Placement] = {}
    for group in report.groups:
        for verdict in group.regressions:
            entry = blame.entry_for(verdict)
            candidates = (
                None if entry is None or entry.discovery_incomplete
                else frozenset(
                    (candidate.repo.lower(), candidate.number)
                    for candidate in entry.candidates
                )
            )
            out[_verdict_identity(verdict)] = Placement(
                onset=verdict.onset_run_date, candidates=candidates,
            )
    return out


def moved_before(
    placement: Placement | None, window: ChangeWindow, pr: tuple[str, int],
) -> bool:
    """Whether *placement* supersedes a claim that pull request *pr* caused a
    step in *window* (see the module docstring for the two conditions)."""
    base, _onset = window
    return (
        placement is not None
        and base is not None
        and placement.onset is not None
        and placement.onset <= base
        and placement.candidates is not None
        and (pr[0].lower(), pr[1]) not in placement.candidates
    )


def withdrawal_block(
    withdrawn: Mapping[tuple, str],
    window: ChangeWindow,
    *,
    noun: str = REGRESSION,
    kept_as_posted: bool = False,
) -> list[str]:
    """The hidden record and the visible note for every claim *withdrawn*
    from a comment on *window*, as the lines that open its text — or nothing
    when none was. *kept_as_posted* says the text below still shows the
    claims, which is true of a body edited in place and not of one rendered
    afresh."""
    marker = _withdrawn_marker(dict(withdrawn))
    if not withdrawn or not marker:
        return []
    nights = sorted(set(withdrawn.values()))
    when = (
        f"on {nights[0]}" if len(nights) == 1
        else f"between {nights[0]} and {nights[-1]}"
    )
    base, onset = window
    count = len(withdrawn)
    one = count == 1
    text = (
        f"> **Withdrawn {when}.** {_count(count, noun)} this comment attributed "
        f"to this PR {'is' if one else 'are'} now placed at or before release "
        f"`{base}`, before this PR's change window (`{base}` → `{onset}`), and "
        f"the candidate search for {'it' if one else 'them'} does not include "
        f"this PR; {'it is' if one else 'they are'} no longer attributed to it."
    )
    if kept_as_posted:
        text += " The rest of this comment is kept as it was posted."
    return [marker, "> [!NOTE]", text, ""]


def superseded_body(
    body: str,
    *,
    pr: tuple[str, int],
    suite: str,
    tonight: Mapping[tuple, Placement],
    min_score: float,
    report_night: str,
) -> str | None:
    """*body* rewritten to withdraw its newly superseded claims, or ``None``
    when it has none.

    Only the bot's own *suite* comment is considered — its first line must be
    that suite's window marker. Newly superseded identities join the body's
    withdrawal record at *report_night*; the cumulative and retained hidden
    states lose their scores; and the note is regenerated from the whole
    record. Everything else is kept as posted. A claim withdrawn once has no
    score left to withdraw, so the same condition edits only once."""
    lines = body.splitlines()
    if not lines:
        return None
    window = window_from_marker(lines[0], suite)
    if window is None:
        return None
    cumulative = _decoded_cumulative(body)
    moved = {
        identity for identity, entry in cumulative.items()
        if entry.likelihood is not None and entry.likelihood >= min_score
        and moved_before(tonight.get(identity), window, pr)
    }
    if not moved:
        return None
    record = dict(_decoded_withdrawn(body))
    for identity in moved:
        record.setdefault(identity, report_night)
    block = withdrawal_block(record, window, kept_as_posted=True)
    if not block:
        return None

    cleared = withdrawn(cumulative, moved)
    retained = [
        replace(row, likelihood=None, source="", state="not_candidate")
        if _retained_identity(row) in moved else row
        for row in _decoded_retained(body)
    ]
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.startswith(_WITHDRAWN_PREFIX):
            continue
        if line == "> [!NOTE]" and index < len(lines) and lines[index].startswith(_NOTE_LEAD):
            # The previous note and the blank line closing it.
            index += 1
            if index < len(lines) and not lines[index].strip():
                index += 1
            continue
        if line.startswith(_CUMULATIVE_PREFIX):
            marker = _cumulative_marker(cleared)
            if marker:
                out.append(marker)
            continue
        if line.startswith(_RETAINED_PREFIX):
            marker = _retained_marker(retained)
            if marker:
                out.append(marker)
            continue
        if line.startswith("### ") and block:
            out += block
            block = []
        out.append(line)
    if block:
        return None
    return "\n".join(out)


def withdrawn(
    cumulative: Mapping[tuple, CumulativeEntry], identities: Iterable[tuple],
) -> dict[tuple, CumulativeEntry]:
    """*cumulative* with *identities* carrying no attribution."""
    identities = set(identities)
    return {
        identity: entry._replace(likelihood=None, source="")
        if identity in identities else entry
        for identity, entry in cumulative.items()
    }


#: How many earlier sidecars :func:`supersede_targets` is given. A claim's old
#: window stays in every night's sidecar while the step is confirmed there, so
#: the night before a move holds it; the margin is what lets a night whose
#: edit failed — or whose pass did not run — be caught up on the next ones.
HISTORY_NIGHTS = 7


def supersede_targets(
    history: Iterable[BlameReport],
    tonight: Mapping[tuple, Placement],
    may_write: Callable[[CandidatePR], bool],
) -> list[tuple[str, int]]:
    """The pull requests whose comments may carry a superseded claim tonight.

    Every candidate *may_write* accepts (the comment policy's repository and
    merged gate) for a step an earlier sidecar in *history* attributed, in a
    window tonight's placement of that step supersedes. No score enters: what a
    comment actually published — a first-pass score, a reviewer's, or one from
    a night that has since dropped below the threshold — is read from the
    comment itself (:func:`superseded_body`). This only bounds which threads
    are read."""
    found: dict[tuple[str, int], tuple[str, int]] = {}
    for blame in history:
        for entry in blame.entries:
            if entry.discovery_incomplete or entry.base_release == entry.onset_release:
                continue
            window = (entry.base_release, entry.onset_release)
            identity = (*entry.key[:5], entry.key[5] or "")
            placement = tonight.get(identity)
            for candidate in entry.candidates:
                key = (candidate.repo.lower(), candidate.number)
                if key not in found and may_write(candidate) \
                        and moved_before(placement, window, key):
                    found[key] = (candidate.repo, candidate.number)
    return [found[key] for key in sorted(found)]
