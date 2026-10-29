"""What each suite's nightly regression report judges, and where it reads it.

A run group's history reaches the engine as trend frames, one per kind of
record its runs hold (:mod:`k4bench.analysis.trend`): their results, their
per-event records and, for a reco job, the costs of its components. A
:class:`MetricSource` names one such frame, how it is built from the run
directories, which of its columns are judged — and in which family, which sets
their floors (:data:`~k4bench.regression.engine.EFFECT_FLOOR`) — and which are
recorded only. A suite's report reads its sources (:func:`sources`) and nothing
else, so what a suite is judged on is said here once, for the report and for
every view that draws a verdict's history.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial

import pandas as pd

from k4bench.analysis.trend import (
    build_component_cost_trend,
    build_event_size_metrics_trend,
    build_event_timing_trend,
    build_results_trend,
)
from k4bench.artifacts import COMPONENTS_SUFFIX, EVENT_SIZE_SUFFIX, EVENTS_SUFFIX, RESULTS_SUFFIX
from k4bench.labels import BASELINE_LABEL
from k4bench.layout import RECO, SIM

#: Run-level metrics evaluated per config: ``{metric: family}``. Deliberately
#: narrow: ``events_per_sec`` is dropped as it is exactly
#: ``n_events / wall_time_s`` (see ``k4bench/runner/result.py``) — tracking
#: it alongside ``wall_time_s`` would flag the same measurement twice with
#: the sign flipped. ``cpu_efficiency`` is host-reliability evidence (see
#: :mod:`k4bench.results.reliability`) rather than a benchmark regression
#: metric, so it is neither judged nor recorded in the nightly report.
RUN_METRICS: dict[str, str] = {
    "wall_time_s": "time",
    "peak_vmem_mb": "memory",
}

#: Per-event summary metrics evaluated per config. ``p95_time_s``,
#: ``p95_rss_mb`` and ``max_rss_mb`` are dropped: noisy tail order-statistics
#: over a few hundred events that add detection overhead without much signal
#: beyond ``mean``/``median`` (time). Total RSS remains a residency diagnostic;
#: anonymous RSS tracks event memory without CVMFS file-cache evictions.
#:
#: ``trimmed_mean_time_s`` is the mean of the events left after the slowest 5%
#: are dropped (:func:`k4bench.analysis.trend.upper_trimmed_mean`) — a view of
#: the typical event, judged *alongside* the untrimmed mean rather than
#: replacing it, because the tail it drops is where a tail-confined regression
#: would appear first.
EVENT_METRICS: dict[str, str] = {
    "mean_time_s": "time",
    "median_time_s": "time",
    "trimmed_mean_time_s": "time",
    "mean_rss_anon_mb": "memory",
}

#: Metrics recorded in the report but never judged. ``user_cpu_s`` tracks
#: ``wall_time_s`` almost exactly — CPU efficiency on these runs is close
#: enough to 1 that the two flag together on the same events — so judging both
#: is one test counted twice. It stays in the report because a night where the
#: two *stop* agreeing is worth being able to look up.
REPORTED_ONLY_METRICS: dict[str, str] = {
    "user_cpu_s": "time",
    "peak_rss_mb": "memory",
}

#: Event metrics recorded without entering the regression engine. Total and
#: file-backed RSS are residency diagnostics: CVMFS publishes can evict the
#: file-backed pages mid-run. ``rss_anon_slope_mb_per_event`` is a memory-growth
#: indicator collecting history before a judgement policy is chosen for it.
EVENT_REPORTED_ONLY_METRICS: dict[str, str] = {
    "mean_rss_mb": "memory",
    "mean_rss_file_mb": "memory",
    "rss_anon_slope_mb_per_event": "memory",
}

#: What one event of a job's output takes on disk, recorded without entering
#: the regression engine (:func:`~k4bench.analysis.trend.build_event_size_metrics_trend`).
#: A fixed-seed sample writes the same bytes on every night of a release, so
#: a step is a change in what the job writes, not noise; it is recorded for the
#: views comparing detectors before a judgement policy is chosen for it. Only
#: for the baseline config, the one those views compare: a sim sweep's
#: hundreds of ablation configs would each add a row per metric to every
#: night's report.
EVENT_SIZE_METRICS: dict[str, str] = {
    "event_size_kb": "size",
    "uncompressed_event_size_kb": "size",
    "file_size_per_event_kb": "size",
    "compression_ratio": "size",
}

#: Every run-level metric whose value is recorded, judged or not.
RUN_VALUE_METRICS: dict[str, str] = {**RUN_METRICS, **REPORTED_ONLY_METRICS}

#: Every event-level metric whose value is recorded, judged or not.
EVENT_VALUE_METRICS: dict[str, str] = {**EVENT_METRICS, **EVENT_REPORTED_ONLY_METRICS}

#: Why each reported-only metric with a reason of its own is not judged; the
#: others share :data:`~k4bench.regression.models.REPORTED_ONLY_REASON`.
REPORTED_ONLY_REASONS: dict[str, str] = {
    "peak_rss_mb": (
        "recorded but not judged — includes file-backed pages that CVMFS publishes can evict"
    ),
    "mean_rss_mb": (
        "recorded but not judged — includes file-backed pages that CVMFS publishes can evict"
    ),
    "mean_rss_file_mb": (
        "recorded but not judged — CVMFS publishes can evict file-backed pages mid-run"
    ),
    "rss_anon_slope_mb_per_event": (
        "recorded but not judged — growth indicator gathering history before a "
        "judgement policy is set"
    ),
    **dict.fromkeys(
        EVENT_SIZE_METRICS,
        "recorded but not judged — event size is displayed before a judgement policy is set",
    ),
}

#: A k4run job's metrics read from its components' costs, judged per config.
#: ``initialize_s`` is the job's initialize phase: the geometry, the conditions
#: and every component's own set-up. It is too small a share of the job's wall
#: time for ``wall_time_s`` to show a step in it — 8% of CLD's, 13% of
#: ALLEGRO's.
JOB_COST_METRICS: dict[str, str] = {
    "initialize_s": "time",
}

#: A k4run job's metrics judged per component, one series for each.
#: ``seconds_per_event`` is what an algorithm adds to an event, so a step in
#: one algorithm is judged where the job's event time is too large to show
#: it; ``initialize_s`` is an algorithm's or a service's own initialize. Both
#: are self costs, so they add up to the job's. The components' instructions,
#: heap allocations and held heap are not judged: they say what a component
#: did, not what it cost, and would judge one step twice beside its time.
COMPONENT_METRICS: dict[str, str] = {
    "seconds_per_event": "component_time",
    "initialize_s": "component_init",
}

#: The column naming a frame's component, ``NaN`` on its rows for a whole job.
COMPONENT = "component"


@dataclass(frozen=True)
class MetricSource:
    """One trend frame of a run group's history, and what is judged off it."""

    #: Names the frame among its suite's sources.
    name: str
    #: Builds the frame from a group's run directories.
    build: Callable[[tuple[str, ...]], pd.DataFrame | None]
    #: The per-config file (:mod:`k4bench.artifacts`) the frame is built from,
    #: which a reader fetching one config's history downloads.
    suffix: str
    #: ``{metric: family}`` judged per config, on the frame's rows for a whole
    #: job: every row, unless the frame has a :data:`COMPONENT` column.
    judged: Mapping[str, str]
    #: ``{metric: family}`` recorded per config, and never judged.
    reported_only: Mapping[str, str] = field(default_factory=dict)
    #: ``{metric: family}`` judged per config and component, on the rows
    #: naming one.
    component_judged: Mapping[str, str] = field(default_factory=dict)
    #: The configs whose rows are read, by label; ``None`` for every config.
    configs: frozenset[str] | None = None

    def _configs_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame if self.configs is None else frame[frame["label"].isin(self.configs)]

    def job_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        """*frame*'s rows for a whole job, of the configs read."""
        frame = self._configs_rows(frame)
        return frame[frame[COMPONENT].isna()] if COMPONENT in frame.columns else frame

    def component_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        """*frame*'s rows naming a component, of the configs read."""
        if COMPONENT not in frame.columns:
            return frame.iloc[0:0]
        frame = self._configs_rows(frame)
        return frame[frame[COMPONENT].notna()]


#: Every source's name for a job's results, which besides their metrics say
#: which configs ran and how they ended, which release they measured and on
#: which host — so every suite reads it.
RESULTS = "results"

_RESULTS = MetricSource(
    RESULTS, build_results_trend, RESULTS_SUFFIX, RUN_METRICS, REPORTED_ONLY_METRICS,
)
_EVENTS = MetricSource(
    "events", build_event_timing_trend, EVENTS_SUFFIX, EVENT_METRICS, EVENT_REPORTED_ONLY_METRICS,
)
_COMPONENTS = MetricSource(
    "components", build_component_cost_trend, COMPONENTS_SUFFIX, JOB_COST_METRICS,
    component_judged=COMPONENT_METRICS,
)
#: The configs :data:`EVENT_SIZE_METRICS` are recorded for, and whose records
#: alone are read.
_EVENT_SIZE_CONFIGS = frozenset({BASELINE_LABEL})

_EVENT_SIZE = MetricSource(
    "event_size", partial(build_event_size_metrics_trend, labels=_EVENT_SIZE_CONFIGS),
    EVENT_SIZE_SUFFIX, {}, EVENT_SIZE_METRICS, configs=_EVENT_SIZE_CONFIGS,
)

#: Each suite's sources, in the order its report lists the verdicts.
CATALOGS: dict[str, tuple[MetricSource, ...]] = {
    SIM: (_RESULTS, _EVENTS, _EVENT_SIZE),
    RECO: (_RESULTS, _EVENTS, _EVENT_SIZE, _COMPONENTS),
}


def sources(suite: str) -> tuple[MetricSource, ...]:
    """The sources *suite*'s report judges."""
    try:
        return CATALOGS[suite]
    except KeyError:
        raise ValueError(f"unknown suite {suite!r}") from None


def source_of(suite: str, metric: str, component: str | None) -> MetricSource | None:
    """The source of *suite* *metric* is read from — of the whole job, or of
    one *component* — or ``None`` for a metric no source records."""
    for source in sources(suite):
        recorded = (
            source.component_judged if component is not None
            else {**source.judged, **source.reported_only}
        )
        if metric in recorded:
            return source
    return None
