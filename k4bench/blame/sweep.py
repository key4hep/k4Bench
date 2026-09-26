"""Every configuration of one run scope, side by side, across one change window.

A detector-removal sweep is the sharpest instrument this suite has: the same
detector, sample, platform and night, run once in full and once without each
sub-detector. Read as one picture it says where a cost lives — the step is in
``baseline`` and absent in ``no_X``, so it sits in X — and it says when there is
no single cost at all: two configurations stepping in opposite directions on a
baseline that did not move. Read as a list of confirmed rows here and a list of
non-confirming ones somewhere else, it says neither, because the pattern is in
the comparison and nobody made it.

This module makes it: a :class:`ScopeSweep` holds every configuration of one
``(detector, platform, sample)`` that measured the window, with each metric's
move and status, and :meth:`ScopeSweep.reading` derives what the pattern says
for one metric family. Nothing here renders anything
(:mod:`k4bench.blame.prompt` does) and nothing talks to a model.

Two readings are specific to timing and carry most of the weight:

* whether a time step is in the **typical event** or carried by a **few long
  events** — the mean against the median and upper-trimmed mean of the same
  configuration, and, where the report recorded them, the per-event wall times
  at both ends of the window (:class:`~k4bench.regression.models.EventProfile`);
* how much of a mean step happened inside Geant4 stepping, from the same
  per-event records.

Unknown stays unknown throughout: a metric that was not judged, a run on an
unreliable host, a configuration that failed or is still settling onto a step
of its own is represented as exactly that and never counted as flat.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from k4bench.blame.evidence import steps_in_window
from k4bench.regression.models import (
    EventProfile,
    MetricVerdict,
    NightlyReport,
    RegionDelta,
    RunGroupReport,
    Severity,
    Unjudged,
    unjudged_cause,
)

#: The metrics a sweep shows per family, in column order — the order that
#: breaks a tie for which metric leads the family's reading
#: (:meth:`ScopeSweep.lead_metric`).
SWEEP_METRICS: dict[str, tuple[str, ...]] = {
    "memory": ("peak_vmem_mb", "mean_rss_anon_mb"),
    "time": ("mean_time_s", "median_time_s", "trimmed_mean_time_s", "wall_time_s"),
}

#: The views of the typical event a time step is checked against. Both are
#: judged metrics of every configuration (``report_builder.EVENT_METRICS``).
TYPICAL_TIME_METRICS = ("median_time_s", "trimmed_mean_time_s")

#: Metrics that measure a whole job rather than an event: their step includes
#: whatever happens outside the event loop (initialisation above all).
PER_JOB_METRICS = frozenset({"wall_time_s", "peak_vmem_mb"})

#: A time step is **carried by a few long events** when the typical event moved
#: less than this fraction of it, and **in the typical event** when both typical
#: views moved the same way by at least :data:`_TYPICAL_FRACTION` of it.
_TAIL_FRACTION = 1 / 3
_TYPICAL_FRACTION = 1 / 2

#: A configuration that did not step is **without the step** when it moved less
#: than this fraction of the stepped configurations' median move.
_ABSENT_FRACTION = 1 / 3

#: At most this share of the judged configurations may step, with the baseline
#: flat, for the steps to count as **isolated**.
_ISOLATED_SHARE = 0.1

#: A configuration's move is **smaller** than the baseline's step below this
#: fraction of it, and **larger** above its inverse.
_SMALLER_FRACTION = 2 / 3


# ── What one configuration did ────────────────────────────────────────────────

#: A cell's status, from what the engine made of the metric.
STEPPED = "stepped"  # confirmed, onset inside this window
WATCH = "watch"  # moved past the gates once, not confirmed
OK = "ok"  # judged, inside its normal variation
ELSEWHERE = "stepped_elsewhere"  # confirmed, but in another window
SETTLING = "settling"  # re-anchoring after a step of its own
UNJUDGED = "unjudged"  # recorded, not judged (see ``cause``)
FAILED = "failed"  # the configuration failed

#: Statuses whose percentage is a judged, current move against a settled
#: baseline — the only cells a pattern may be read from.
_COMPARABLE = frozenset({STEPPED, WATCH, OK})

#: Statuses of a judged metric that is not comparable in this window.
_EXCLUDED = (SETTLING, ELSEWHERE)


@dataclass(frozen=True)
class SweepCell:
    """One metric of one configuration: its status and how far it moved.

    ``pct`` is the signed fraction against the metric's own baseline, and
    ``delta`` the same move in the metric's unit; both ``None`` when unknown.
    ``cause`` says why an ``unjudged`` cell was not judged. ``fact_id`` is set
    by a caller that asks a model to score this cell (the review's row id)."""

    metric: str
    status: str
    pct: float | None = None
    delta: float | None = None
    direction: str = "NONE"
    cause: str = ""
    fact_id: str = ""

    @property
    def comparable(self) -> bool:
        return self.status in _COMPARABLE and self.pct is not None


@dataclass(frozen=True)
class SweepRow:
    """One benchmark configuration of a scope."""

    label: str
    cells: tuple[SweepCell, ...] = ()
    #: The per-event wall times at both ends of the window, from this
    #: configuration's stepped time verdict; ``None`` when it did not step in
    #: time or the report predates the field.
    profile: EventProfile | None = None
    #: How the typical event's time moved per detector region, from the same
    #: verdict, largest movement first; empty when unknown.
    regions: tuple[RegionDelta, ...] = ()

    def cell(self, metric: str) -> SweepCell | None:
        return next((c for c in self.cells if c.metric == metric), None)

    def stepped(self, family: str) -> bool:
        """Whether any of *family*'s metrics stepped in this window."""
        return any(
            c.status == STEPPED for c in self.cells if c.metric in SWEEP_METRICS[family]
        )

    def stepped_in(self, metric: str) -> bool:
        """Whether *metric* itself stepped in this window."""
        cell = self.cell(metric)
        return cell is not None and cell.status == STEPPED


# ── How a time step is shaped ─────────────────────────────────────────────────

@dataclass(frozen=True)
class TimeShape:
    """Whether one configuration's time step is in the typical event.

    ``kind`` is one of:

    * ``"typical"`` — the median and trimmed mean moved with it;
    * ``"tail"`` — they did not: a few long events carry it;
    * ``"outside"`` — a wall-time step the mean event time did not follow: it
      happened outside the event loop, in initialisation or teardown;
    * ``"mixed"`` — none of these cleanly.

    ``lead`` is the stepped metric the shape was read for, with its move;
    ``typical`` the moves that decided it. The profile figures are ``None``
    when the report carries no event profile for the row."""

    kind: str
    lead: str
    lead_pct: float
    typical: tuple[tuple[str, float], ...] = ()
    profile: EventProfile | None = None

    @property
    def mean_move(self) -> float | None:
        """The profile's mean move, as a fraction of the base mean."""
        p = self.profile
        if p is None or not p.base.mean:
            return None
        return (p.onset.mean - p.base.mean) / p.base.mean

    @property
    def move_without_longest(self) -> float | None:
        """The profile's mean move once each end's longest event is left out."""
        p = self.profile
        if p is None:
            return None
        base, onset = p.base.mean_without_longest, p.onset.mean_without_longest
        if base is None or onset is None or not base:
            return None
        return (onset - base) / base

    @property
    def median_move(self) -> float | None:
        p = self.profile
        if p is None or not p.base.median:
            return None
        return (p.onset.median - p.base.median) / p.base.median

    @property
    def stepping_share(self) -> float | None:
        """How much of the profile's mean move happened in Geant4 stepping, or
        ``None`` when unknown or when the mean did not move."""
        p = self.profile
        if p is None or p.base.stepping_mean is None or p.onset.stepping_mean is None:
            return None
        moved = p.onset.mean - p.base.mean
        if not moved or not math.isfinite(moved):
            return None
        return (p.onset.stepping_mean - p.base.stepping_mean) / moved


def time_shape(row: SweepRow, lead: str | None = None) -> TimeShape | None:
    """How *row*'s step in the *lead* metric is shaped, or ``None`` when that
    metric did not step or no typical view of it was judged.

    Without a *lead*, it is the mean event time when that stepped, else the wall
    time — the per-job figure, which also carries initialisation, so a wall step
    is first checked against the mean event time. A step in a typical view
    itself (the median or trimmed mean) is a step of the typical event. Every
    other shape is read from both typical views: with either one not judged,
    the typical event is only half seen and the step is not called any shape."""
    candidates = (lead,) if lead is not None else ("mean_time_s", "wall_time_s")
    lead = next(
        (
            c for metric in candidates
            if (c := row.cell(metric)) is not None and c.status == STEPPED
            and c.pct is not None and math.isfinite(c.pct) and c.pct != 0
        ),
        None,
    )
    if lead is None:
        return None
    if lead.metric in TYPICAL_TIME_METRICS:
        return TimeShape(
            kind="typical", lead=lead.metric, lead_pct=lead.pct,
            typical=((lead.metric, lead.pct),), profile=row.profile,
        )
    step = abs(lead.pct)
    mean = row.cell("mean_time_s")
    if lead.metric == "wall_time_s" and mean is not None and mean.comparable:
        if abs(mean.pct) <= step * _TAIL_FRACTION:
            return TimeShape(
                kind="outside", lead=lead.metric, lead_pct=lead.pct,
                typical=(("mean_time_s", mean.pct),), profile=row.profile,
            )
    typical = tuple(
        (metric, c.pct)
        for metric in TYPICAL_TIME_METRICS
        if (c := row.cell(metric)) is not None and c.comparable
    )
    if len(typical) < len(TYPICAL_TIME_METRICS):
        return None
    largest = max(abs(pct) for _metric, pct in typical)
    if largest <= step * _TAIL_FRACTION:
        kind = "tail"
    elif all(
        pct * lead.pct > 0 and abs(pct) >= step * _TYPICAL_FRACTION
        for _metric, pct in typical
    ):
        kind = "typical"
    else:
        kind = "mixed"
    return TimeShape(
        kind=kind, lead=lead.metric, lead_pct=lead.pct, typical=typical,
        profile=row.profile,
    )


# ── One scope ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FamilyReading:
    """What the sweep's pattern says for one metric family, read on one metric.

    Everything is read from the **lead** metric's own cells: a configuration
    counts as stepped when the lead stepped there, as judged when the lead was
    judged there, and every direction, size and comparison below is the lead's.
    Another metric of the family that stepped where the lead did not is kept
    apart in ``also``, never counted as a step of the lead.

    Every list names configurations by label and is ordered largest move first;
    ``judged`` counts the configurations whose lead metric was judged at all,
    which is the denominator every other count is read against."""

    family: str
    lead: str
    judged: int
    stepped: tuple[str, ...]
    up: int
    down: int
    baseline: SweepCell | None
    #: Stepped against the direction most stepped configurations took — or,
    #: when neither direction is a majority, every stepped configuration.
    opposite: tuple[str, ...]
    #: The stepped configurations' median move, unsigned; ``None`` when none
    #: has a known size.
    typical_step: float | None
    #: Judged, not stepped, and moved less than a third of the typical step.
    absent: tuple[str, ...]
    #: Judged, not stepped, but moved as much as a stepped configuration would.
    unconfirmed: tuple[str, ...]
    #: The lead metric could not be judged here, with the causes.
    unjudged: tuple[tuple[str, str], ...]
    #: Judged, but not comparable in this window — still settling after a step
    #: of its own, or stepped in another window — with that status.
    excluded: tuple[tuple[str, str], ...]
    #: Few configurations stepped, and the baseline is among those that
    #: moved less than a third of their step: isolated rows on a flat baseline.
    isolated: bool
    #: When the baseline stepped: configurations not already ``absent`` that
    #: moved the same way by less (``smaller``) or more (``larger``) than it
    #: did, with their move as a fraction of the baseline's — in the metric's
    #: own unit where both are known, since configurations of different size
    #: make percentages unequal.
    smaller: tuple[tuple[str, float], ...] = ()
    larger: tuple[tuple[str, float], ...] = ()
    #: For time: the stepped configurations' shapes, by label.
    shapes: tuple[tuple[str, TimeShape], ...] = ()
    #: ``(metric, labels)`` for each other metric of the family that stepped
    #: on configurations where the lead did not, largest move first.
    also: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def tail(self) -> tuple[str, ...]:
        return tuple(label for label, shape in self.shapes if shape.kind == "tail")

    @property
    def typical(self) -> tuple[str, ...]:
        return tuple(label for label, shape in self.shapes if shape.kind == "typical")


@dataclass(frozen=True)
class ScopeSweep:
    """Every configuration of one ``(detector, platform, sample)`` that measured
    a change window.

    ``unread`` is why nothing here can be read — the night's run was on an
    unreliable host — and is empty for a sweep whose rows are evidence. A scope
    whose run could not be read is kept, because "this detector was reached by
    the change and could not be measured" is a different statement from "this
    detector did not move". ``failures`` are the group's job failures (a
    configuration that produced nothing): they cost those configurations, which
    have no row, and never the rest of the sweep."""

    detector: str
    platform: str
    sample: str
    rows: tuple[SweepRow, ...] = ()
    unread: str = ""
    failures: tuple[str, ...] = ()

    @property
    def scope(self) -> tuple[str, str, str]:
        return (self.detector, self.platform, self.sample)

    def row(self, label: str) -> SweepRow | None:
        return next((r for r in self.rows if r.label == label), None)

    @property
    def stepped_families(self) -> tuple[str, ...]:
        """The families any configuration stepped in, most stepped first."""
        counts = {
            family: sum(1 for r in self.rows if r.stepped(family))
            for family in SWEEP_METRICS
        }
        return tuple(sorted(
            (f for f, n in counts.items() if n),
            key=lambda f: (-counts[f], f),
        ))

    def lead_metric(self, family: str) -> str:
        """The family's metric the reading is taken on: one the baseline stepped
        in, else the one that stepped on the most configurations, else the first
        comparable anywhere, else the first judged anywhere (settling or stepped
        in another window), else the first — column order breaking every tie, so
        the reading follows the pattern most of the sweep shows."""
        metrics = SWEEP_METRICS[family]
        baseline = self.row("baseline")
        if baseline is not None:
            for metric in metrics:
                if baseline.stepped_in(metric):
                    return metric
        steps = {m: sum(r.stepped_in(m) for r in self.rows) for m in metrics}
        most = max(metrics, key=lambda m: steps[m])
        if steps[most]:
            return most
        for wanted in (lambda c: c.comparable, lambda c: c.status in _EXCLUDED):
            for metric in metrics:
                if any((c := r.cell(metric)) is not None and wanted(c) for r in self.rows):
                    return metric
        return metrics[0]

    def reading(self, family: str) -> FamilyReading:
        """What the pattern says for *family* (see :class:`FamilyReading`)."""
        lead = self.lead_metric(family)
        cells = [(r, r.cell(lead)) for r in self.rows]
        # A confirmed step whose size is unknown is still a step: it counts as
        # stepped and judged, and adds nothing to any direction or size.
        judged = [
            (r, c) for r, c in cells
            if c is not None and (c.comparable or c.status == STEPPED)
        ]
        stepped = [(r, c) for r, c in judged if c.status == STEPPED]
        sized = [(r, c) for r, c in stepped if c.pct is not None]

        def move(cell: SweepCell | None) -> float:
            if cell is None or cell.pct is None or not math.isfinite(cell.pct):
                return 0.0
            return abs(cell.pct)

        stepped.sort(key=lambda rc: (-move(rc[1]), rc[0].label))
        up = sum(1 for _r, c in sized if c.pct > 0)
        down = sum(1 for _r, c in sized if c.pct < 0)
        if up > down:
            opposite = [r.label for r, c in stepped if c.pct is not None and c.pct < 0]
        elif down > up:
            opposite = [r.label for r, c in stepped if c.pct is not None and c.pct > 0]
        else:
            opposite = [r.label for r, _c in stepped] if up and down else []

        typical_step = (
            statistics.median(move(c) for _r, c in sized) if sized else None
        )
        quiet = sorted(
            ((r, c) for r, c in judged if c.status != STEPPED),
            key=lambda rc: (-move(rc[1]), rc[0].label),
        )
        absent, unconfirmed = [], []
        if typical_step:
            for r, c in quiet:
                if move(c) < typical_step * _ABSENT_FRACTION:
                    absent.append(r.label)
                else:
                    unconfirmed.append(r.label)

        unjudged = tuple(
            (r.label, c.cause or c.status)
            for r, c in cells
            if c is not None and not c.comparable
            and c.status not in (STEPPED, *_EXCLUDED)
        ) + tuple((r.label, "missing") for r, c in cells if c is None)
        excluded = tuple(
            (r.label, c.status) for r, c in cells
            if c is not None and c.status in _EXCLUDED
        )

        baseline_row = self.row("baseline")
        baseline = baseline_row.cell(lead) if baseline_row is not None else None
        baseline_stepped = any(r.label == "baseline" for r, _c in stepped)
        isolated = (
            bool(stepped) and baseline is not None and "baseline" in absent
            and len(stepped) <= max(2, _ISOLATED_SHARE * len(judged))
        )
        smaller, larger = [], []
        if baseline_stepped and baseline is not None:
            for r, c in judged:
                if r.label == "baseline" or r.label in absent:
                    continue
                ratio = _ratio(c, baseline)
                if ratio is None or ratio <= 0:
                    continue
                if ratio < _SMALLER_FRACTION:
                    smaller.append((r.label, ratio))
                elif ratio > 1 / _SMALLER_FRACTION:
                    larger.append((r.label, ratio))
            smaller.sort(key=lambda lr: (lr[1], lr[0]))
            larger.sort(key=lambda lr: (-lr[1], lr[0]))
        shapes = ()
        if family == "time":
            shapes = tuple(
                (r.label, shape) for r, _c in stepped
                if (shape := time_shape(r, lead)) is not None
            )
        also = []
        for metric in SWEEP_METRICS[family]:
            if metric == lead:
                continue
            extra = sorted(
                (
                    (r, c) for r in self.rows
                    if (c := r.cell(metric)) is not None and c.status == STEPPED
                    and not r.stepped_in(lead)
                ),
                key=lambda rc: (-move(rc[1]), rc[0].label),
            )
            if extra:
                also.append((metric, tuple(r.label for r, _c in extra)))
        return FamilyReading(
            family=family, lead=lead, judged=len(judged),
            stepped=tuple(r.label for r, _c in stepped), up=up, down=down,
            baseline=baseline, opposite=tuple(opposite),
            typical_step=typical_step, absent=tuple(absent), unconfirmed=tuple(unconfirmed),
            unjudged=unjudged, excluded=excluded, isolated=isolated,
            smaller=tuple(smaller), larger=tuple(larger), shapes=shapes,
            also=tuple(also),
        )


def _ratio(cell: SweepCell, baseline: SweepCell) -> float | None:
    """*cell*'s move as a fraction of *baseline*'s, in the metric's unit when
    both are known and as percentages otherwise."""
    if cell.delta is not None and baseline.delta:
        return cell.delta / baseline.delta
    if cell.pct is not None and baseline.pct:
        return cell.pct / baseline.pct
    return None


# ── Building sweeps from a report ─────────────────────────────────────────────

def _cell(verdict: MetricVerdict, window: tuple[str | None, str]) -> SweepCell:
    """What the engine made of one metric of one configuration, for *window*."""
    direction = str(getattr(verdict.direction, "value", verdict.direction))
    pct = verdict.pct_change
    if pct is not None and not math.isfinite(pct):
        pct = None
    delta = None
    if verdict.value is not None and verdict.baseline_median is not None:
        delta = verdict.value - verdict.baseline_median
        if not math.isfinite(delta):
            delta = None
    if verdict.severity is Severity.FAILURE:
        return SweepCell(verdict.metric, FAILED)
    if verdict.severity is Severity.UNKNOWN:
        cause = unjudged_cause(verdict)
        return SweepCell(
            verdict.metric, UNJUDGED,
            cause=cause.value if isinstance(cause, Unjudged) else "not judged",
        )
    if verdict.severity is Severity.CONFIRMED:
        status = STEPPED if steps_in_window(verdict, window) else ELSEWHERE
    elif verdict.reanchor_run_date is not None:
        status = SETTLING
    elif verdict.severity is Severity.WATCH:
        status = WATCH
    else:
        status = OK
    return SweepCell(verdict.metric, status, pct=pct, delta=delta, direction=direction)


def _time_evidence(
    verdicts: list[MetricVerdict], window: tuple[str | None, str],
) -> tuple[EventProfile | None, tuple[RegionDelta, ...]]:
    """The event profile and region movements of a configuration's stepped time
    verdict — the mean event time's first, since that is the statistic the
    profile decomposes. Both come from one read of the window's region files, so
    they are taken from the same verdict."""
    for metric in ("mean_time_s", "wall_time_s", "median_time_s", "trimmed_mean_time_s"):
        for v in verdicts:
            if (
                v.metric == metric and steps_in_window(v, window)
                and (v.event_profile is not None or v.region_deltas)
            ):
                return v.event_profile, tuple(v.region_deltas)
    return None, ()


def scope_sweep(
    group: RunGroupReport, *, base_release: str | None, onset_release: str,
) -> ScopeSweep:
    """*group*'s configurations across ``(base_release, onset_release]``."""
    window = (base_release, onset_release)
    unread = ""
    if group.reliable is not True:
        unread = (
            "tonight's run failed the host reliability check, so nothing was judged"
            if group.reliable is False
            else "tonight's run has no host reliability verdict, so nothing was judged"
        )
    by_label: dict[str, list[MetricVerdict]] = {}
    for verdict in group.verdicts:
        by_label.setdefault(verdict.label, []).append(verdict)
    shown = {m for metrics in SWEEP_METRICS.values() for m in metrics} | set(TYPICAL_TIME_METRICS)
    rows = []
    for label, verdicts in sorted(by_label.items()):
        if any(v.severity is Severity.FAILURE for v in verdicts):
            cells = tuple(SweepCell(m, FAILED) for m in sorted(shown))
        else:
            cells = tuple(
                _cell(v, window)
                for v in sorted(verdicts, key=lambda v: v.metric)
                if v.metric in shown and v.sub_detector is None
            )
        profile, regions = _time_evidence(verdicts, window)
        rows.append(SweepRow(label=label, cells=cells, profile=profile, regions=regions))
    return ScopeSweep(
        detector=group.detector, platform=group.platform, sample=group.sample,
        rows=tuple(rows), unread=unread, failures=tuple(group.job_failures),
    )


def window_sweeps(
    report: NightlyReport,
    *,
    base_release: str | None,
    onset_release: str,
    stacks: set[str],
) -> tuple[ScopeSweep, ...]:
    """Every scope in *report* that measured this window: ran one of *stacks*,
    the releases the window's regressed rows were measured on. Ordered by
    identity. A group that ran another release measured a different window and
    is not here; a group whose run could not be read is here, marked as such
    (:attr:`ScopeSweep.unread`)."""
    return tuple(
        scope_sweep(group, base_release=base_release, onset_release=onset_release)
        for group in sorted(
            report.groups, key=lambda g: (g.detector, g.sample, g.platform)
        )
        if group.k4h_release in stacks and not group.missing_run
    )
