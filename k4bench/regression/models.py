"""Dataclasses and enums for the nightly regression report.

The severity and direction axes are kept separate rather than cross-producted
into one enum: severity says *how much attention* a metric deserves, direction
says *which way* it moved. Neither carries a good/bad judgment — direction is
a plain sign, not an evaluation, since a step in either direction can equally
be an optimization, a deliberate change, or a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How much attention a metric verdict deserves.

    ``OK``        — inside the baseline's normal variation.
    ``WATCH``     — first night crossing both detection gates; shown in the
                    report but not alerted on (see the two-strike rule in
                    :mod:`k4bench.regression.engine`).
    ``CONFIRMED`` — crossed both gates on two consecutive reliable nights.
    ``FAILURE``   — hard job failure (non-zero returncode / missing run);
                    bypasses confirmation and always alerts immediately.
    ``UNKNOWN``   — not enough reliable history to judge; never a flag,
                    mirroring ``reliability.py``'s "no evidence ⇒ no verdict".
    """

    OK = "OK"
    WATCH = "WATCH"
    CONFIRMED = "CONFIRMED"
    FAILURE = "FAILURE"
    UNKNOWN = "UNKNOWN"


class Direction(str, Enum):
    """Which way a flagged metric moved. ``NONE`` for OK/UNKNOWN/FAILURE.

    Purely the mechanical sign of the change — ``UP``/``DOWN`` make no claim
    about whether the move is desirable. A metric going down is not an
    "improvement" any more than one going up is a "regression" in the
    colloquial sense: either can be a deliberate change, an optimization, or
    a bug, and the report leaves that call to a human.
    """

    NONE = "NONE"
    UP = "UP"
    DOWN = "DOWN"


@dataclass(frozen=True)
class SeriesId:
    """Identity of one metric history: the axes that must never be pooled.

    A ``(detector, platform, sample)`` triple is one independent run group
    (different platforms/samples have independent baselines); ``label`` is the
    benchmark config within the run, ``metric`` the column evaluated, and
    ``sub_detector`` the region name for region-level metrics only.
    ``metric_family`` selects the practical-effect floor in the engine.
    """

    detector: str
    platform: str
    sample: str
    label: str
    metric_family: str
    metric: str
    sub_detector: str | None = None


@dataclass(frozen=True)
class MetricVerdict:
    """The engine's judgement of one metric on one night.

    ``run_id`` is the nightly run directory name; ``run_date`` is the *Key4hep
    release* the run measured, not the date it ran. The two differ often — the
    nightly build does not publish every day, so consecutive runs frequently
    re-measure one release. Anything correlating a verdict with the software
    stack must key on ``run_date``.

    ``onset_*`` and ``last_accepted_*`` bound the window a ``CONFIRMED`` change
    entered in, and are ``None`` on every other severity. Confirmation is a
    two-strike rule, so the night a change is *reported* is one reliable night
    after the night it first *appeared*: ``onset_*`` identifies that first
    night, and ``last_accepted_*`` the newest night before it observed at the
    then-accepted level. The change therefore landed in
    ``(last_accepted, onset]`` — the interval to search for a cause.
    ``last_accepted_*`` is ``None`` when no such night exists (a change
    confirmed before the series ever settled), which makes the window
    open-ended rather than empty.

    ``first_confirmed_run_id`` names the night a ``CONFIRMED`` change was
    first confirmed for its release (``None`` on every other severity). It
    equals ``run_id`` on that first night; on a later night of the same
    release re-confirming the change it points back — letting the report and
    email render a repeat as a repeat rather than fresh news.
    """

    detector: str
    platform: str
    sample: str
    label: str
    metric_family: str
    metric: str
    sub_detector: str | None
    run_id: str
    run_date: str
    value: float | None
    baseline_median: float | None
    baseline_mad: float | None
    pct_change: float | None
    z_score: float | None
    severity: Severity
    direction: Direction
    reason: str
    onset_run_id: str | None = None
    onset_run_date: str | None = None
    last_accepted_run_id: str | None = None
    last_accepted_run_date: str | None = None
    first_confirmed_run_id: str | None = None

    @property
    def flagged(self) -> bool:
        """True for anything worth a row in the report (not OK/UNKNOWN)."""
        return self.severity in (Severity.WATCH, Severity.CONFIRMED, Severity.FAILURE)


@dataclass
class RunGroupReport:
    """All verdicts for one ``(detector, platform, sample)`` triple for the night.

    ``job_failures`` carries hard, group-level problems that have no metric
    series to attach to (e.g. no run uploaded for tonight at all, or a config
    that produced no results). ``notes`` carries non-alertable context (e.g.
    tonight's run failed the reliability check, so metrics were not judged).
    ``reliable`` is tonight's host-reliability tri-state (the same per-run
    verdict as :func:`k4bench.results.reliability_evidence.run_reliability_map`;
    ``None`` = no evidence), persisted so report consumers — e.g. the
    dashboard's Overview tab — can apply the standard unreliable-run filter
    without re-downloading run data.
    """

    detector: str
    platform: str
    sample: str
    k4h_release: str
    run_date: str
    run_id: str
    verdicts: list[MetricVerdict] = field(default_factory=list)
    job_failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reliable: bool | None = None

    def _select(self, severity: Severity) -> list[MetricVerdict]:
        return [v for v in self.verdicts if v.severity is severity]

    @property
    def regressions(self) -> list[MetricVerdict]:
        """Confirmed regressions — a "regression" here means any confirmed
        step beyond the baseline, either direction; nothing is judged good or
        bad, only that it moved beyond the baseline twice in a row."""
        return self._select(Severity.CONFIRMED)

    @property
    def watches(self) -> list[MetricVerdict]:
        return self._select(Severity.WATCH)

    @property
    def failures(self) -> list[MetricVerdict]:
        return self._select(Severity.FAILURE)


@dataclass
class NightlyReport:
    """One night's verdicts across every run group found on EOS."""

    generated_at: str
    groups: list[RunGroupReport] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricVerdict]:
        """Confirmed regressions across all groups, either direction."""
        return [v for g in self.groups for v in g.regressions]

    @property
    def watches(self) -> list[MetricVerdict]:
        return [v for g in self.groups for v in g.watches]

    @property
    def failures(self) -> list[MetricVerdict]:
        """Per-config hard failures across all groups."""
        return [v for g in self.groups for v in g.failures]

    @property
    def job_failures(self) -> list[tuple[RunGroupReport, str]]:
        """Group-level hard failures (e.g. missing run), with their group."""
        return [(g, msg) for g in self.groups for msg in g.job_failures]

    @property
    def has_alertable(self) -> bool:
        """True when the night warrants an alert email: any confirmed
        regression or any hard failure. WATCHes never alert."""
        return bool(self.regressions or self.failures or self.job_failures)

    @property
    def report_night(self) -> str:
        """The nightly date this report covers (newest run across groups)."""
        return max((g.run_date for g in self.groups), default="")

    def by_detector(self) -> dict[str, list[RunGroupReport]]:
        """Group the run groups by detector (a detector can have several
        ``(platform, sample)`` groups), preserving insertion order."""
        out: dict[str, list[RunGroupReport]] = {}
        for g in self.groups:
            out.setdefault(g.detector, []).append(g)
        return out
