"""Which components of a reco job a confirmed step moved.

The reco counterpart of :mod:`k4bench.regression.regions`. A confirmed step is
one number — the job's mean event time, its initialize, the anonymous memory it
held — and one number names no mechanism. The k4Bench auditor records what each
algorithm and service spent of its own (``_components.json``), and the memory
pass the heap each one holds (``_heap.json``), so the step can be taken apart
into the components whose own cost moved: "the IDEA job got 15% faster" becomes
"SimulateSiPMwithEdep went from 3.9 to 1.5 s per event and retired 66% fewer
instructions", which can be matched against a diff.

This module judges nothing (the engine has decided *that* the metric stepped)
and adds no thresholds; it reports the decomposition, largest movement first.
It reads the files of one configuration at the two ends of one window, and only
for a verdict that confirmed. A component one end did not record is *absent*,
never zero: an algorithm added to or removed from the job is a real change and
must stay distinguishable from one that stood still.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from k4bench.analysis.components import (
    ComponentTiming,
    event_loop_time,
    without_warm_up,
)
from k4bench.analysis.loader import load_component_heap, load_component_timing
from k4bench.analysis.trend import EXPECTED_LOAD_ERRORS, MIN_TRIM_EVENTS, TRIM_FRACTION
from k4bench.artifacts import COMPONENTS_SUFFIX, HEAP_SUFFIX, label_path
from k4bench.provenance.sources import load_run_packages, package_of
from k4bench.regression.models import ComponentDelta, MetricVerdict
from k4bench.regression.regions import window_end_dirs

_log = logging.getLogger(__name__)

#: Components carried on a job's verdict. A job runs dozens; the largest few
#: are where a step went, and the rest would say "nothing happened here" many
#: times over.
MAX_COMPONENTS = 6

#: The component costs a step is taken apart into: each component's own time
#: per event of the event loop, the same over its typical events, its own time
#: in initialize, in each phase of the whole job, and the heap it holds.
EVENT, TYPICAL_EVENT, INITIALIZE, JOB, HEAP = "event", "typical_event", "initialize", "job", "heap"

#: What each judged metric is taken apart into. The mean event time into each
#: algorithm's own time per event, which add up to the event. The trimmed mean
#: and the median, which the slow tail does not move, into the same over the
#: events left once the slowest are dropped (:func:`_typical_events`): a median
#: does not decompose, and this is the closest view of the typical event that
#: does. The job's wall time into each component's own time in each phase,
#: kept apart (:func:`_job_phases`). The anonymous RSS into the heap each
#: component holds at an event's end, typically. The peak virtual memory is not
#: taken apart: no component cost measures the address space, so a breakdown of
#: the heap would name components that did not cause its step.
DECOMPOSED_AS: dict[str, str] = {
    "mean_time_s": EVENT,
    "median_time_s": TYPICAL_EVENT,
    "trimmed_mean_time_s": TYPICAL_EVENT,
    "seconds_per_event": EVENT,
    "initialize_s": INITIALIZE,
    "wall_time_s": JOB,
    "mean_rss_anon_mb": HEAP,
}

#: Joins a component and a phase into the name of one of :func:`_job_phases`'
#: entries: ``"GeoSvc in initialize"``.
_IN_PHASE = " in "


def decomposed_as(verdict: MetricVerdict) -> str | None:
    """The component cost *verdict*'s step is taken apart into, or ``None``
    for a metric that has none."""
    return DECOMPOSED_AS.get(verdict.metric)


@dataclass(frozen=True)
class _Night:
    """One run's cost of each component for one configuration, and the
    instructions each retired over the same span where they were counted."""

    cost: pd.Series
    instructions: pd.Series | None = None
    packages: dict[str, str] = field(default_factory=dict)


def _working(timing: ComponentTiming) -> list[str]:
    """The components doing work of their own: algorithms and services, not
    the sequencers, whose own cost is the bookkeeping of the calls they run."""
    services = timing.components.index[timing.components["category"] == "service"]
    return [*timing.algorithms(), *services]


def _packages(run_dir: Path, timing: ComponentTiming) -> dict[str, str]:
    """The stack package each of *timing*'s components is defined in, where
    it could be placed."""
    stack = load_run_packages(run_dir)
    out = {}
    for name, entry in timing.components.iterrows():
        source = package_of(entry["impl_library"] or entry["library"], stack)
        if source is not None:
            out[str(name)] = source.package
    return out


def _per_event(frame: pd.DataFrame) -> pd.Series:
    """Each component's sum of *frame* over the event loop, the warm-up event
    left out, per event of it: the events it was skipped in counted, as
    :func:`~k4bench.analysis.components.event_loop_time` counts them."""
    frame = without_warm_up(frame)
    return frame.sum(min_count=1) / len(frame) if len(frame) else frame.sum(min_count=1)


def _typical_events(wall: pd.DataFrame) -> pd.Index:
    """The events of *wall* left once the slowest are dropped, as
    :func:`~k4bench.analysis.trend.upper_trimmed_mean` drops them, ranked by
    the components' summed own time; every event where too few are left to
    trim."""
    totals = wall.sum(axis=1)
    n_drop = int(len(totals) * TRIM_FRACTION) if len(totals) >= MIN_TRIM_EVENTS else 0
    return totals.sort_values(kind="stable").index[: len(totals) - n_drop]


def _job_phases(timing: ComponentTiming, metric: str) -> pd.Series | None:
    """Each working component's own *metric* in each phase of the job, one
    entry per component and phase (``"GeoSvc in initialize"``), the event loop
    as ``execute``. Kept apart rather than summed per component: a service
    initialized during an event is listed under initialize and also stays in
    that event's algorithm's execute cost, and the auditor does not record
    which calls were made so, so one component's sum would count them twice."""
    parts = {phase: table.get(metric) for phase, table in timing.lifecycle.items()}
    parts["execute"] = timing.execute[metric].sum(min_count=1) if metric in timing.execute else None
    if any(part is None for part in parts.values()):
        return None
    working = _working(timing)
    return pd.Series({
        f"{component}{_IN_PHASE}{phase}": value
        for phase, part in parts.items()
        for component, value in part.reindex(working).items()
        if pd.notna(value)
    }, dtype=float)


def _timed(timing: ComponentTiming, quantity: str) -> tuple[pd.Series, pd.Series | None]:
    """Each working component's own time in *quantity*, and the instructions
    it retired there where the job counted them."""
    if quantity in (EVENT, TYPICAL_EVENT):
        algorithms = timing.algorithms()
        wall = without_warm_up(timing.execute["wall_s"])
        if quantity == TYPICAL_EVENT:
            wall = wall.loc[_typical_events(wall)]
        cost = event_loop_time(wall, warm_up=None)["seconds_per_event"].reindex(algorithms)
        work = (
            _per_event(timing.execute["instructions"].loc[wall.index]).reindex(algorithms)
            if "instructions" in timing.execute else None
        )
        return cost, work
    if quantity == JOB:
        return _job_phases(timing, "wall_s"), _job_phases(timing, "instructions")
    tables = [timing.lifecycle["initialize"]] if "initialize" in timing.lifecycle else []

    def total(metric: str) -> pd.Series | None:
        parts = [table.get(metric) for table in tables]
        if not parts or any(part is None for part in parts):
            return None
        summed = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
        return summed.reindex(_working(timing))

    return total("wall_s"), total("instructions")


def _read_night(run_dir: Path, label: str, quantity: str) -> _Night | None:
    """One run's component costs for one configuration, or ``None`` when it
    recorded none that can be read — the ordinary case for a run whose job
    crashed, or for a memory step on a run without a memory pass, which is why
    the caller treats it as unknown rather than as zero."""
    if not label_path(run_dir, label, COMPONENTS_SUFFIX).is_file():
        return None
    if quantity == HEAP and not label_path(run_dir, label, HEAP_SUFFIX).is_file():
        return None
    try:
        timing = load_component_timing(run_dir, labels=[label])[label]
        packages = _packages(run_dir, timing)
        if quantity == HEAP:
            heap = load_component_heap(run_dir, labels=[label])[label]
            typical = heap.summary()["typical_mb"]
            # A working component the file lists that held nothing is a zero,
            # not absent: it ran, and starting to hold memory is a movement.
            kept = heap.holders().union(typical.index.intersection(_working(timing)), sort=False)
            return _Night(cost=typical.loc[kept], packages=packages)
        cost, work = _timed(timing, quantity)
        if quantity == JOB and cost is not None:
            packages = {
                entry: packages[owner]
                for entry in cost.index
                if (owner := entry.rsplit(_IN_PHASE, 1)[0]) in packages
            }
    except (*EXPECTED_LOAD_ERRORS, KeyError) as exc:
        _log.debug("components: unreadable component costs in %s — %s", run_dir, exc)
        return None
    except Exception:
        _log.exception("components: failed reading component costs in %s", run_dir)
        return None
    return _Night(cost=cost, instructions=work, packages=packages)


def _read_nights(
    run_dirs: Sequence[Path],
    label: str,
    quantity: str,
    judgeable_configs: set[tuple[str, str]] | None,
) -> list[_Night]:
    """The readable nights of one window end, oldest first. Config-nights absent
    from *judgeable_configs* (failed or orphaned) are gaps and are skipped."""
    nights = []
    for run_dir in sorted(run_dirs, key=lambda d: d.name):
        if (
            judgeable_configs is not None
            and (str(run_dir.name), str(label)) not in judgeable_configs
        ):
            continue
        night = _read_night(run_dir, label, quantity)
        if night is not None:
            nights.append(night)
    return nights


def _median_of(nights: Sequence[_Night], read) -> dict[str, float]:
    """Each component's median over *nights* of what *read* gives for a night
    — a release measured on several nights is several measurements of one
    software state, combined as the engine combines them."""
    per_night: dict[str, list[float]] = {}
    for night in nights:
        values = read(night)
        if values is None:
            continue
        for component, value in values.items():
            if pd.notna(value) and math.isfinite(float(value)):
                per_night.setdefault(str(component), []).append(float(value))
    return {name: float(np.median(values)) for name, values in per_night.items()}


def _change(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return after / before - 1.0


def component_deltas(
    run_dirs: Sequence[str],
    *,
    label: str,
    quantity: str,
    base_release: str,
    onset_release: str,
    base_run_id: str | None = None,
    onset_run_id: str | None = None,
    judgeable_configs: set[tuple[str, str]] | None = None,
) -> tuple[ComponentDelta, ...]:
    """How each component's *quantity* (:data:`DECOMPOSED_AS`) moved across
    ``(base, onset]`` for configuration *label*, largest movement first.

    Empty when either end recorded no component costs at all: with one side
    measured there is no comparison, and treating the missing side as zero
    would report every component as newly added. The ends are those of
    :func:`~k4bench.regression.regions.window_end_dirs`, and when
    *judgeable_configs* is supplied a failed or orphaned config-night absent
    from it enters neither.
    """
    ends = window_end_dirs(
        run_dirs, base_release=base_release, onset_release=onset_release,
        base_run_id=base_run_id, onset_run_id=onset_run_id,
    )
    if ends is None:
        return ()
    base_nights, onset_nights = (
        _read_nights(dirs, label, quantity, judgeable_configs) for dirs in ends
    )
    base = _median_of(base_nights, lambda night: night.cost)
    onset = _median_of(onset_nights, lambda night: night.cost)
    if not base or not onset:
        return ()
    base_work = _median_of(base_nights, lambda night: night.instructions)
    onset_work = _median_of(onset_nights, lambda night: night.instructions)
    packages = {
        name: package
        for night in (*base_nights, *onset_nights)
        for name, package in night.packages.items()
    }

    deltas = []
    for component in sorted(set(base) | set(onset)):
        before, after = base.get(component), onset.get(component)
        # A component one end did not run was added or removed; its whole cost
        # is the movement, and saying so is the point.
        delta = (after or 0.0) - (before or 0.0)
        if not math.isfinite(delta):
            continue
        deltas.append(ComponentDelta(
            component=component, base=before, onset=after, delta=delta,
            package=packages.get(component, ""),
            instructions_change=_change(base_work.get(component), onset_work.get(component)),
        ))
    deltas.sort(key=lambda d: (-abs(d.delta), d.component))
    return tuple(deltas)
