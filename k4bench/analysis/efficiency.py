"""How efficiently a k4run job's algorithms run in the event loop, per event:
their time, the instructions they execute and how many of those the CPU gets
through per cycle, and how hard they lean on the heap allocator.

Instructions follow the work an algorithm does, not the host it ran on, so
next to its time they tell a change in the software (instructions moved) from
a slower or busier machine (instructions held, instructions per cycle fell).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from k4bench.analysis.components import (
    WARM_UP_EVENT,
    ComponentTiming,
    event_loop_time,
    without_warm_up,
)

#: The columns of :func:`component_efficiency`.
EFFICIENCY_COLUMNS = [
    "ms_per_event",
    "instructions_per_event",
    "ipc",
    "allocations_per_event",
    "allocations_per_s",
    "allocated_mb_per_event",
    "events",
    "executed_events",
]


def component_efficiency(
    execute: Mapping[str, pd.DataFrame], *, warm_up: int | None = WARM_UP_EVENT,
) -> pd.DataFrame:
    """Each component's event-loop cost in the events it ran in, the
    *warm_up* event left out (:func:`~k4bench.analysis.components.event_loop_time`):
    ``ms_per_event``, its mean self wall time in those events (its
    ``seconds_when_run``, not its share of every event);
    ``instructions_per_event``; ``ipc``, its instructions over its cycles,
    both summed over those events; ``allocations_per_event``,
    ``allocations_per_s`` (per second of its own time: how hard it leans on
    the allocator, however long it runs) and ``allocated_mb_per_event``,
    ``NaN`` without the allocation counter; ``events``, the events left, and
    ``executed_events``, those it ran in.
    *execute* is :attr:`~k4bench.analysis.components.ComponentTiming.execute`.
    Indexed by component, longest first; a component that spent no time, or
    has no counts for an event it ran in, is left out, so the frame is empty
    for a job without hardware counters.
    """
    if not {"instructions", "cycles"} <= execute.keys():
        return pd.DataFrame(columns=EFFICIENCY_COLUMNS, dtype=float)
    time = event_loop_time(execute["wall_s"], warm_up=warm_up)
    ran = without_warm_up(execute["wall_s"], warm_up).notna()
    per_event = time["executed_events"].where(time["executed_events"] > 0)

    def per_component(metric: str) -> pd.Series:
        # A component that did not run in an event spent nothing there; one
        # that ran without a count leaves its sum unknown.
        if metric not in execute:
            return pd.Series(np.nan, index=ran.columns)
        return execute[metric].reindex(ran.index).where(ran, 0.0).sum(skipna=False)

    seconds = time["seconds"]
    instructions = per_component("instructions")
    cycles = per_component("cycles")
    allocations = per_component("allocations")
    table = pd.DataFrame({
        "ms_per_event": time["seconds_when_run"] * 1e3,
        "instructions_per_event": instructions / per_event,
        "ipc": instructions / cycles.where(cycles > 0),
        "allocations_per_event": allocations / per_event,
        "allocations_per_s": allocations / seconds.where(seconds > 0),
        "allocated_mb_per_event": per_component("allocated_bytes") / per_event / 1024**2,
        "events": time["events"],
        "executed_events": time["executed_events"],
    })
    table = table[(table["ms_per_event"] > 0) & table["ipc"].notna()]
    return table.sort_values("ms_per_event", ascending=False)


def algorithm_efficiency(
    timing: ComponentTiming, *, warm_up: int | None = WARM_UP_EVENT,
) -> pd.DataFrame:
    """:func:`component_efficiency` of *timing*'s algorithms that do work of
    their own: services and sequencers, whose own time is bookkeeping, are
    left out."""
    table = component_efficiency(timing.execute, warm_up=warm_up)
    return table[table.index.isin(timing.algorithms())]
