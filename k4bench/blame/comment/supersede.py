"""Claims a posted comment makes that tonight's report has moved out of the pull
request's reach.

A comment attributes regressions to a pull request in one change window. The
regression engine can later re-date a step: a late confirmation moves its window
back onto an earlier release (:mod:`k4bench.regression`), and the step is then
measured *before* the window the comment was written for. The pull request
shipped inside that window, so it cannot have caused the step, and tonight's
complete candidate search for the step confirms it by not listing the pull
request. Everything here is deterministic — no model is asked.

A claim is **superseded** when both hold:

* tonight's report places the step's onset at or before the comment window's
  base release — a step placed *after* the window is a new step, and the old
  claim is left as the record of what was measured then;
* tonight's sidecar entry for the step has complete candidate discovery, and the
  pull request is not among its candidates.

Two places act on it. A comment rendered tonight drops the superseded
identities' scores from its lineage state and says so under the table
(:func:`~k4bench.blame.comment.render._render`). A comment *not* rendered
tonight — its pull request no longer selected — is found from the previous
report's attributions (:func:`supersede_targets`) and edited in place
(:func:`superseded_body`): a dated note above the posted text, and the same
scores cleared from its hidden state so no later version resurfaces them. The
rest of the posted body is kept as the record it is.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from k4bench.blame.comment.markdown import _count
from k4bench.blame.comment.state import (
    _CUMULATIVE_PREFIX,
    _MARKER_SUFFIX,
    _RETAINED_PREFIX,
    ChangeWindow,
    CumulativeEntry,
    _cumulative_marker,
    _decoded_cumulative,
    _decoded_retained,
    _retained_identity,
    _retained_marker,
    _verdict_identity,
    window_from_marker,
)
from k4bench.blame.models import BlameReport, CandidatePR
from k4bench.regression.models import NightlyReport

#: The hidden line a superseded body carries, ahead of its note. The payload is
#: a digest of the superseded identities, so the same set never edits twice.
_SUPERSEDED_PREFIX = "<!-- k4bench-blame-superseded:v1 "


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


def superseded_note(count: int, window: ChangeWindow, noun: str = "regression") -> str:
    """One sentence naming how many attributed steps were moved out of
    *window*, in the comment's own wording."""
    base, onset = window
    verb = "is" if count == 1 else "are"
    return (
        f"{_count(count, noun)} this comment attributed to this PR {verb} now "
        f"placed at or before release `{base}`, before this PR's change window "
        f"(`{base}` → `{onset}`), and tonight's candidate search for "
        f"{'it' if count == 1 else 'them'} does not include this PR; "
        f"{'it is' if count == 1 else 'they are'} no longer attributed to it."
    )


def superseded_body(
    body: str,
    *,
    pr: tuple[str, int],
    suite: str,
    tonight: Mapping[tuple, Placement],
    min_score: float,
    report_night: str,
) -> str | None:
    """*body* rewritten to withdraw its superseded claims, or ``None`` when it
    makes none (or already says so).

    Only the bot's own *suite* comment is considered — its first line must be
    that suite's window marker. The cumulative and retained hidden states lose
    the superseded identities' scores, and a dated note opens the visible text;
    everything else is kept as posted. A body already carrying a note for the
    same identities is left alone, so a standing condition edits once."""
    lines = body.splitlines()
    if not lines:
        return None
    window = window_from_marker(lines[0], suite)
    if window is None:
        return None
    cumulative = _decoded_cumulative(body)
    moved = sorted(
        identity for identity, entry in cumulative.items()
        if entry.likelihood is not None and entry.likelihood >= min_score
        and moved_before(tonight.get(identity), window, pr)
    )
    if not moved:
        return None
    digest = _identities_digest(moved)
    superseded_line = f"{_SUPERSEDED_PREFIX}{digest}{_MARKER_SUFFIX}"
    if superseded_line in lines:
        return None

    moved_set = set(moved)
    cleared = withdrawn(cumulative, moved_set)
    retained = [
        replace(row, likelihood=None, source="", state="not_candidate")
        if _retained_identity(row) in moved_set else row
        for row in _decoded_retained(body)
    ]
    note = [
        superseded_line,
        "> [!NOTE]",
        f"> **Superseded on {report_night}.** "
        + superseded_note(len(moved), window)
        + " The rest of this comment is kept as it was posted.",
        "",
    ]
    out: list[str] = []
    skipping = False
    for line in lines:
        if skipping:
            skipping = bool(line.strip())
            continue
        if line.startswith(_SUPERSEDED_PREFIX):
            skipping = True
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
        if line.startswith("### ") and note:
            out += note
            note = []
        out.append(line)
    if note:
        return None
    return "\n".join(out)


def withdrawn(
    cumulative: Mapping[tuple, CumulativeEntry], identities: set[tuple],
) -> dict[tuple, CumulativeEntry]:
    """*cumulative* with *identities* carrying no attribution."""
    return {
        identity: entry._replace(likelihood=None, source="")
        if identity in identities else entry
        for identity, entry in cumulative.items()
    }


def supersede_targets(
    previous_report: NightlyReport,
    previous_blame: BlameReport,
    tonight: Mapping[tuple, Placement],
    allows: Callable[[CandidatePR], bool],
) -> list[tuple[str, int]]:
    """The pull requests whose comments may carry a superseded claim tonight:
    each candidate *allows* accepts (the comment policy's gate) for a step the
    previous report confirmed, where tonight's placement of that step
    supersedes the previous window. Their threads are then read and each
    comment checked on its own state (:func:`superseded_body`); this only
    bounds which threads are read."""
    found: set[tuple[str, int]] = set()
    named: dict[tuple[str, int], tuple[str, int]] = {}
    for group in previous_report.groups:
        for verdict in group.regressions:
            entry = previous_blame.entry_for(verdict)
            if entry is None or entry.discovery_incomplete:
                continue
            window = (entry.base_release, entry.onset_release)
            placement = tonight.get(_verdict_identity(verdict))
            for candidate in entry.candidates:
                key = (candidate.repo.lower(), candidate.number)
                if allows(candidate) and moved_before(placement, window, key):
                    found.add(key)
                    named.setdefault(key, (candidate.repo, candidate.number))
    return [named[key] for key in sorted(found)]


def _identities_digest(identities: list[tuple]) -> str:
    canonical = json.dumps(identities, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
