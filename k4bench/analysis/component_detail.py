"""One component of a k4run job, everything the run recorded of it: the
numbers a component's detail view shows, gathered from the files the other
views read, each computed once here.

The timed run (``_components.json``, :class:`~k4bench.analysis.components.ComponentTiming`)
gives its own (self) time in each phase and in each event it ran in, the
thread counters of those calls and what they asked of the allocator; the
memory pass (``_heap.json``, :class:`~k4bench.analysis.heap.ComponentHeap`)
gives the heap it held. The two are separate runs of the job, so they are set
side by side only as summaries: an event of one is not an event of the other.

:func:`component_summary` gives every component's event-loop and lifecycle
numbers at once, :func:`component_detail` one component's, with its calls
event by event, and :func:`component_history` one component's across the
runs of a trend window, a run it is missing from left without values, never
zero.

What the numbers do not say:

- when a call started or ended: the auditor records each component's costs
  per event and per phase, not a timeline of its calls;
- why a counter moved: instructions, cycles, faults and context switches
  describe the calls, they do not by themselves name a cause;
- whether memory leaked: net allocation is what a call allocated and did not
  free, which an ownership transfer (the reader's event data, freed by the
  writer) does too. Only the memory pass's growth kept after finalize
  (:meth:`~k4bench.analysis.heap.ComponentHeap.summary`) is evidence of that.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from k4bench.analysis.components import (
    HARDWARE_METRICS,
    INCLUSIVE_METRICS,
    THREAD_METRICS,
    WARM_UP_EVENT,
    ComponentTiming,
    event_loop_time,
    without_warm_up,
)
from k4bench.analysis.heap import MB

#: The phases outside the event loop, in the order a job runs them.
LIFECYCLE_PHASES = ("initialize", "start", "stop", "finalize")

#: The per-call metrics that add up over a component's events: its thread's
#: CPU time and counters, and what it asked of the allocator.
SUMMED_METRICS = (
    "cpu_s",
    *HARDWARE_METRICS,
    *THREAD_METRICS,
    "allocations",
    "allocated_bytes",
    "net_allocated_bytes",
)

#: The categories a component can be inspected in: those doing work or
#: serving it, not what the auditor could not place.
INSPECTED_CATEGORIES = ("algorithm", "service")

#: :func:`component_history`'s ``status``: the configuration's timed run
#: recorded the component, recorded the configuration without it, or did not
#: record the configuration (it failed, its file could not be read).
TIMED = "timed"
ABSENT = "absent"
NOT_TIMED = "not timed"

#: :func:`component_history`'s ``heap_status``: the configuration's memory
#: pass charged heap to the component at some moment, charged it none (or did
#: not run it), or did not run.
HELD = "held"
HELD_NONE = "held none"
NO_MEMORY_PASS = "no memory pass"

#: The run columns of every trend frame.
_RUN_KEYS = ["run_id", "run_date", "k4h_release_date", "k4h_release", "x_date"]

#: What :func:`component_history` draws of the timed run, from
#: :func:`component_summary`'s columns.
TIMED_HISTORY_COLUMNS = (
    "ms_when_run",
    "p95_ms",
    "coverage_pct",
    "initialize_s",
    "finalize_s",
    "instructions_g",
    "ipc",
    "minor_page_faults_per_event",
    "major_page_faults_per_event",
    "voluntary_context_switches_per_event",
    "involuntary_context_switches_per_event",
    "allocations_per_event",
    "allocated_mb_per_event",
    "net_allocated_mb_per_event",
    "largest_allocation_mb",
)

#: What :func:`component_history` draws of the memory pass, from
#: :meth:`~k4bench.analysis.heap.ComponentHeap.summary`'s columns.
HEAP_HISTORY_COLUMNS = (
    "before_first_mb",
    "typical_mb",
    "working_set_median_mb",
    "growth_mb_per_event",
)


def _metric(timing: ComponentTiming, metric: str, warm_up: int | None) -> pd.DataFrame:
    """*metric* in each event a component ran in, the *warm_up* event left
    out; ``NaN`` where it did not run, or the job did not record *metric*."""
    wall = without_warm_up(timing.execute["wall_s"], warm_up)
    if metric not in timing.execute:
        return pd.DataFrame(np.nan, index=wall.index, columns=wall.columns)
    return without_warm_up(timing.execute[metric], warm_up).where(wall.notna())


def component_summary(timing: ComponentTiming, *, warm_up: int | None = WARM_UP_EVENT) -> pd.DataFrame:
    """Every component of *timing*'s job, indexed by name in the auditor's
    order, the *warm_up* event left out of every number over events.

    Columns:

    - the :data:`~k4bench.analysis.components.EVENT_LOOP_TIME_COLUMNS` of its
      own wall time (:func:`~k4bench.analysis.components.event_loop_time`),
      and ``p50_s`` and ``p95_s`` of it over the events it ran in;
    - for each of :data:`SUMMED_METRICS`, its sum over the events it ran in
      (named as the metric), ``{metric}_events``, how many of those events
      have a value, and ``{metric}_per_event``, the sum over those. A metric
      the job did not record is ``NaN`` with no events;
    - ``ipc``: instructions over cycles, each summed over the events both
      were counted in (a ratio of sums, never a mean of ratios), and
      ``ipc_events``, how many;
    - ``allocations_per_s``: its allocations per second of its own time over
      the events they were counted in;
    - for each of :data:`~k4bench.analysis.components.INCLUSIVE_METRICS`,
      ``{metric}_max`` and ``{metric}_median`` over the events it ran in, and
      ``{metric}_events``. They include the nested calls, so they do not add
      up across components;
    - ``{phase}_s`` and ``{phase}_cpu_s`` for each of
      :data:`LIFECYCLE_PHASES`: its own wall and CPU time there, ``NaN`` when
      it was not called in that phase or the job did not record it.

    A component that ran in no event has no time (``NaN``), never zero, and
    the counts of a component that ran stay as recorded, zero included.
    """
    wall = without_warm_up(timing.execute["wall_s"], warm_up)
    out = event_loop_time(timing.execute["wall_s"], warm_up=warm_up)
    columns: dict[str, pd.Series] = {
        "p50_s": wall.quantile(0.5),
        "p95_s": wall.quantile(0.95),
    }
    for metric in SUMMED_METRICS:
        values = _metric(timing, metric, warm_up)
        counted = values.notna().sum()
        total = values.sum(min_count=1)
        columns[metric] = total
        columns[f"{metric}_events"] = counted
        columns[f"{metric}_per_event"] = total / counted.where(counted > 0)

    instructions, cycles = (_metric(timing, metric, warm_up) for metric in HARDWARE_METRICS)
    both = instructions.notna() & cycles.notna()
    retired = instructions.where(both).sum(min_count=1)
    counted_cycles = cycles.where(both).sum(min_count=1)
    columns["ipc"] = retired / counted_cycles.where(counted_cycles > 0)
    columns["ipc_events"] = both.sum()

    allocations = _metric(timing, "allocations", warm_up)
    own = wall.where(allocations.notna()).sum(min_count=1)
    columns["allocations_per_s"] = allocations.sum(min_count=1) / own.where(own > 0)

    for metric in INCLUSIVE_METRICS:
        values = _metric(timing, metric, warm_up)
        columns[f"{metric}_max"] = values.max()
        columns[f"{metric}_median"] = values.median()
        columns[f"{metric}_events"] = values.notna().sum()

    for phase in LIFECYCLE_PHASES:
        table = timing.lifecycle.get(phase)
        for column, metric in ((f"{phase}_s", "wall_s"), (f"{phase}_cpu_s", "cpu_s")):
            columns[column] = (
                table[metric] if table is not None and metric in table
                else pd.Series(np.nan, index=wall.columns)
            )
    return pd.concat([out, pd.DataFrame(columns, index=wall.columns)], axis=1)


def component_choices(timing: ComponentTiming) -> pd.DataFrame:
    """The components of *timing*'s job that can be inspected, indexed by
    name in the auditor's order: its algorithms and services
    (:data:`INSPECTED_CATEGORIES`), whether or not they ran, with columns
    ``category`` and ``impl``. The memory pass and the data flow name the
    same components, by the same names."""
    listed = timing.components[timing.components["category"].isin(INSPECTED_CATEGORIES)]
    return listed[["category", "impl"]].copy()


@dataclass(frozen=True)
class ComponentDetail:
    """What one configuration's run recorded of component :attr:`name`
    (:func:`component_detail`)."""

    name: str
    #: Its row of :attr:`~k4bench.analysis.components.ComponentTiming.components`
    #: (``category``, ``type``, ``impl``, ``library``, ``impl_library``,
    #: ``parent``, ``execute_calls``); ``None`` when the timed run did not
    #: record it.
    identity: pd.Series | None
    #: Its row of :func:`component_summary`; ``None`` likewise.
    summary: pd.Series | None
    #: The phases outside the event loop it was called in × the metrics the
    #: job recorded there, in the job's order.
    lifecycle: pd.DataFrame
    #: The events it ran in, the warm-up left out, × the metrics the job
    #: recorded per event.
    events: pd.DataFrame
    #: Its row of the memory pass's
    #: :meth:`~k4bench.analysis.heap.ComponentHeap.summary`; ``None`` without
    #: a memory pass, or when the pass charged it no heap at any moment.
    heap: pd.Series | None
    #: Why the job has no hardware counters, if it said
    #: (:attr:`~k4bench.analysis.components.ComponentTiming.hardware_counters_error`).
    hardware_counters_error: str | None = None
    #: The threads the timed job ran calls on; above one, wall times of
    #: overlapping calls add up to more than the elapsed time.
    threads: int = 1

    @property
    def timed(self) -> bool:
        """Whether the timed run recorded it."""
        return self.summary is not None


def component_detail(
    name: str,
    timing: ComponentTiming | None,
    *,
    summary: pd.DataFrame | None = None,
    heap_summary: pd.DataFrame | None = None,
    heap_holders: Iterable[str] | None = None,
    warm_up: int | None = WARM_UP_EVENT,
) -> ComponentDetail:
    """What the run recorded of component *name* in one configuration: from
    the timed run's *timing* (whose :func:`component_summary` may be passed
    as *summary*, computed once for every component) and from the memory
    pass's *heap_summary*
    (:meth:`~k4bench.analysis.heap.ComponentHeap.summary`), of which only
    *heap_holders* (:meth:`~k4bench.analysis.heap.ComponentHeap.holders`)
    count, when given. Either may be ``None``: the detail then lacks that
    run's part."""
    identity = row = None
    lifecycle = pd.DataFrame()
    events = pd.DataFrame()
    if timing is not None and name in timing.components.index:
        identity = timing.components.loc[name]
        if summary is None:
            summary = component_summary(timing, warm_up=warm_up)
        row = summary.loc[name]
        lifecycle = pd.DataFrame.from_dict({
            phase: table.loc[name] for phase, table in timing.lifecycle.items()
            if name in table.index and pd.notna(table.at[name, "wall_s"])
        }, orient="index")
        wall = without_warm_up(timing.execute["wall_s"], warm_up)[name]
        ran = wall.index[wall.notna()]
        events = pd.DataFrame({
            metric: without_warm_up(frame, warm_up)[name].reindex(ran)
            for metric, frame in timing.execute.items()
        }, index=ran)
    heap = None
    if heap_summary is not None and name in heap_summary.index:
        if heap_holders is None or name in set(heap_holders):
            heap = heap_summary.loc[name]
    return ComponentDetail(
        name=name, identity=identity, summary=row, lifecycle=lifecycle, events=events, heap=heap,
        hardware_counters_error=timing.hardware_counters_error if timing is not None else None,
        threads=timing.threads if timing is not None else 1,
    )


def _timed_history(rows: pd.DataFrame) -> pd.DataFrame:
    """:data:`TIMED_HISTORY_COLUMNS` of :func:`component_summary` *rows*."""
    return pd.DataFrame({
        "ms_when_run": rows["seconds_when_run"] * 1e3,
        "p95_ms": rows["p95_s"] * 1e3,
        "coverage_pct": rows["coverage"] * 100,
        "initialize_s": rows["initialize_s"],
        "finalize_s": rows["finalize_s"],
        "instructions_g": rows["instructions_per_event"] / 1e9,
        "ipc": rows["ipc"],
        **{f"{metric}_per_event": rows[f"{metric}_per_event"] for metric in THREAD_METRICS},
        "allocations_per_event": rows["allocations_per_event"],
        "allocated_mb_per_event": rows["allocated_bytes_per_event"] / MB,
        "net_allocated_mb_per_event": rows["net_allocated_bytes_per_event"] / MB,
        "largest_allocation_mb": rows["largest_allocation_bytes_max"] / MB,
    }, index=rows.index, columns=list(TIMED_HISTORY_COLUMNS))


def _run_index(rows: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(rows[["x_date", "run_id"]])


def component_history(
    summary_trend: pd.DataFrame | None,
    heap_trend: pd.DataFrame | None,
    label: str,
    name: str,
) -> pd.DataFrame:
    """Component *name* of configuration *label* in each run of the window:
    one row per run either trend holds (any configuration's), in the order
    measured.

    *summary_trend* is :func:`~k4bench.analysis.trend.build_component_summary_trend`'s,
    *heap_trend* :func:`~k4bench.analysis.trend.build_heap_trend`'s; either
    may be ``None``. Columns: the runs' ``run_id``, ``run_date``,
    ``k4h_release_date``, ``k4h_release`` and ``x_date``; ``label``, *name*,
    as the trend views name a series; ``status`` (:data:`TIMED`,
    :data:`ABSENT` or :data:`NOT_TIMED`) and ``heap_status`` (:data:`HELD`,
    :data:`HELD_NONE` or :data:`NO_MEMORY_PASS`); the
    :data:`TIMED_HISTORY_COLUMNS` and :data:`HEAP_HISTORY_COLUMNS`. A run the
    component has no row in leaves them ``NaN``, a gap in a line rather than
    a zero.
    """
    frames = [f for f in (summary_trend, heap_trend) if f is not None and not f.empty]
    columns = [*_RUN_KEYS, "label", "status", "heap_status",
               *TIMED_HISTORY_COLUMNS, *HEAP_HISTORY_COLUMNS]
    if not frames:
        return pd.DataFrame(columns=columns)
    runs = (
        pd.concat([f[_RUN_KEYS] for f in frames])
        .drop_duplicates(["x_date", "run_id"])
        .sort_values(["run_date", "x_date"], na_position="first", kind="stable")
        .reset_index(drop=True)
    )
    keys = _run_index(runs)
    out = runs.assign(label=name)

    def of(trend: pd.DataFrame | None) -> tuple[pd.Index, pd.DataFrame]:
        """The runs that recorded *label* in *trend*, and *name*'s rows there
        by run."""
        none = pd.MultiIndex.from_arrays([[], []], names=["x_date", "run_id"])
        if trend is None or trend.empty:
            return none, pd.DataFrame(index=none)
        rows = trend[trend["label"] == label]
        recorded = _run_index(rows[rows["component"].isna()])
        mine = rows[rows["component"] == name]
        mine = mine.set_index(_run_index(mine))
        return recorded, mine[~mine.index.duplicated(keep="last")]

    timed_runs, timed = of(summary_trend)
    held_runs, held = of(heap_trend)
    in_timed, in_held = keys.isin(timed.index), keys.isin(held.index)
    out["status"] = np.where(in_timed, TIMED, np.where(keys.isin(timed_runs), ABSENT, NOT_TIMED))
    out["heap_status"] = np.where(
        in_held, HELD, np.where(keys.isin(held_runs), HELD_NONE, NO_MEMORY_PASS),
    )
    timed_values = _timed_history(timed) if len(timed) else pd.DataFrame(
        columns=list(TIMED_HISTORY_COLUMNS), dtype=float,
    )
    heap_values = (
        held.reindex(columns=list(HEAP_HISTORY_COLUMNS)) if len(held)
        else pd.DataFrame(columns=list(HEAP_HISTORY_COLUMNS), dtype=float)
    )
    for values in (timed_values, heap_values):
        aligned = values.reindex(keys)
        for column in values.columns:
            out[column] = aligned[column].to_numpy(dtype=float)
    return out[columns]
