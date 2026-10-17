"""Per-component statistics of one k4run job's event loop.

Everything here reads a :class:`~k4bench.analysis.components.ComponentTiming`
and leaves out event :data:`WARM_UP_EVENT`: the first event is where the job's
memory is first touched and lazily initialized state is built (the CLD
reconstruction takes ~40× the page faults there), so its costs belong to the
job's start-up rather than to processing an event. Each statistic is of the
events that remain.

Costs are self costs (see :mod:`~k4bench.analysis.components`), so those of the
:data:`~k4bench.analysis.components.NON_ADDITIVE_METRICS` aside, they add up
across components: a component's share of an event is meaningful, and the
components' costs sum to the event's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from k4bench.analysis.components import NON_ADDITIVE_METRICS, ComponentTiming

#: The event number every statistic leaves out (see the module docstring).
WARM_UP_EVENT = 0


def measured_events(frame: pd.DataFrame) -> pd.DataFrame:
    """*frame*, an events × components ``execute`` frame, without the warm-up event."""
    return frame[frame.index != WARM_UP_EVENT]


def contributions(timing: ComponentTiming, metric: str) -> pd.DataFrame:
    """What each component added to each measured event's *metric*: its value
    where it ran, zero where it did not, and ``NaN`` where it ran without a
    value (hardware counters that could not be read for that call)."""
    frame = measured_events(timing.execute[metric])
    ran = measured_events(timing.execute["wall_s"]).notna()
    return frame.where(ran | frame.notna(), 0.0)


def component_summary(timing: ComponentTiming, metric: str = "wall_s") -> pd.DataFrame:
    """One row per component that ran in a measured event, in the job's order.

    Columns:

    - ``n_events`` — measured events the component ran in;
    - ``per_event`` — what it adds to an average measured event: its total over
      the job's measured events divided by their number, a filtered component's
      events without it counting as zero. ``NaN`` when one of its calls has no
      value, since the total is then unknown;
    - ``mean``, ``median``, ``p90``, ``max``, ``std``, ``sem``, ``cv`` — over
      the events it ran in. ``std`` and ``sem`` need two events, ``cv``
      (``std / mean``) a nonzero mean;
    - ``share`` — ``per_event`` as a fraction of the components' summed
      ``per_event``: of the event's total. ``NaN`` for a metric that does not
      add up (:data:`~k4bench.analysis.components.NON_ADDITIVE_METRICS`), or
      when any component's ``per_event`` is unknown.
    """
    if metric not in timing.execute:
        raise ValueError(f"the job has no {metric} recorded")
    frame = measured_events(timing.execute[metric])
    ran = measured_events(timing.execute["wall_s"]).notna()
    present = ran.any() | frame.notna().any()
    frame = frame.loc[:, present]

    n = frame.count()
    total = contributions(timing, metric).loc[:, present].sum(skipna=False)
    per_event = total / len(frame) if len(frame) else total * np.nan
    std = frame.std()
    mean = frame.mean()
    out = pd.DataFrame({
        "n_events": n.astype(int),
        "per_event": per_event,
        "mean": mean,
        "median": frame.median(),
        "p90": frame.quantile(0.9),
        "max": frame.max(),
        "std": std,
        "sem": std / np.sqrt(n),
        "cv": std / mean.where(mean != 0),
    })
    if metric in NON_ADDITIVE_METRICS or out["per_event"].isna().any():
        out["share"] = np.nan
    else:
        out["share"] = out["per_event"] / out["per_event"].sum()
    out.index.name = "component"
    return out


def event_totals(timing: ComponentTiming, metric: str = "wall_s") -> pd.Series:
    """Each measured event's *metric* summed over its components; ``NaN`` for
    an event where a component ran without a value. Refused for a metric that
    does not add up."""
    if metric in NON_ADDITIVE_METRICS:
        raise ValueError(f"{metric} does not add up across components")
    if metric not in timing.execute:
        raise ValueError(f"the job has no {metric} recorded")
    return contributions(timing, metric).sum(axis=1, skipna=False)


@dataclass(frozen=True)
class WorkEfficiency:
    """How much work an average measured event took, and how fast it ran."""

    #: User-space instructions per event: the work, independent of the host.
    instructions_per_event: float
    #: User-space cycles per event.
    cycles_per_event: float
    #: Instructions per cycle over the measured events: lower for the same
    #: work means the host ran it slower (contention, a cold cache).
    ipc: float


def work_efficiency(timing: ComponentTiming) -> WorkEfficiency | None:
    """The job's :class:`WorkEfficiency`, from the events whose instructions
    and cycles were both counted; ``None`` without hardware counters."""
    if not {"instructions", "cycles"} <= set(timing.execute):
        return None
    instructions = event_totals(timing, "instructions")
    cycles = event_totals(timing, "cycles")
    counted = instructions.notna() & cycles.notna() & (cycles > 0)
    if not counted.any():
        return None
    instructions, cycles = instructions[counted], cycles[counted]
    return WorkEfficiency(
        instructions_per_event=float(instructions.mean()),
        cycles_per_event=float(cycles.mean()),
        ipc=float(instructions.sum() / cycles.sum()),
    )
