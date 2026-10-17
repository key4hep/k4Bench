"""Paired comparison of two k4run measurements, component by component.

A reco benchmark reconstructs a frozen input, so two of its runs process the
same events and their costs can be compared event by event: the difference of
each event's cost is free of the events' own spread, which for the CLD
reconstruction is ~30% per event while the paired difference is ~0.2%. Each
delta here is the mean of those per-event differences, with a bootstrap
interval over the events.

A component missing from one side counts as costing nothing there, so the
deltas of every component add up to the change of the event's total: a removed
component's delta is minus its cost, an added one's is its cost.

Whether a change is real is judged in two steps. :func:`significant_changes`
asks whether it is larger than both a floor and the run-to-run noise
(:func:`run_to_run_noise`) and whether its interval excludes zero.
:func:`classify_changes` then asks why the time moved: instructions count the
work done independently of the host, so a time change with unchanged
instructions is the host running the same work slower or faster.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from k4bench.analysis.component_stats import WARM_UP_EVENT, event_totals
from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.loader import load_component_timing
from k4bench.artifacts import MACHINE_INFO, RUN_INFO

#: The row of a comparison or a noise estimate that stands for the whole event.
EVENT_TOTAL = "(event total)"

#: Paired events below which no change is called significant.
MIN_PAIRED_EVENTS = 10

#: Run-to-run ratios below which the window gives no noise estimate.
MIN_NOISE_RATIOS = 3

#: How many run-to-run noise widths a change has to exceed.
NOISE_SIGMAS = 3.0

#: The fraction of a time change the instructions' change has to account for
#: to call it a change of the work. When the work changes, time follows it
#: (the CLD truth-tracking variant: 0.7–1.1 of each algorithm's change); when
#: the host does, instructions stay within ~0.1% of a ~40% time change.
WORK_EXPLAINS = 0.5

#: Where the input store kept frozen inputs before it moved them, and where
#: they are now.
INPUT_RELOCATIONS = {"/_inputs/CLD_": "/inputs/reco/CLD/CLD_"}

#: Bootstrap resamples behind each interval, and its coverage.
RESAMPLES = 2000
CONFIDENCE = 0.95


@dataclass(frozen=True)
class Measurement:
    """One label of one run: the costs a comparison reads, and what is known
    about how they were measured."""

    run_dir: Path
    label: str
    timing: ComponentTiming
    #: The run's ``run_info.json``; empty for a run made with the CLI.
    run_info: Mapping
    #: The run's ``machine_info.json``; empty when it has none.
    machine_info: Mapping


def _read_json(path: Path) -> Mapping:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_measurement(run_dir: str | Path, label: str) -> Measurement:
    """*label*'s :class:`Measurement` in *run_dir*; raises ``ValueError`` when
    the label has no component file."""
    run_dir = Path(run_dir)
    timing = load_component_timing(run_dir, [label])[label]
    return Measurement(
        run_dir=run_dir,
        label=label,
        timing=timing,
        run_info=_read_json(run_dir / RUN_INFO),
        machine_info=_read_json(run_dir / MACHINE_INFO),
    )


def paired_events(a: ComponentTiming, b: ComponentTiming) -> pd.Index:
    """The measured events both jobs processed, in *a*'s order."""
    common = a.execute["wall_s"].index.intersection(b.execute["wall_s"].index, sort=False)
    return common[common != WARM_UP_EVENT]


@dataclass(frozen=True)
class Check:
    """One condition two measurements are compared under."""

    #: What was compared: ``input``, ``events``, ``hardware counters``,
    #: ``host``, ``CPU``.
    name: str
    ok: bool
    #: Whether failing it makes the two incomparable, rather than a caveat.
    blocking: bool
    detail: str


def comparability(a: Measurement, b: Measurement) -> list[Check]:
    """The conditions *b* is compared with *a* under, blocking ones first."""
    checks = []

    inputs_a, inputs_b = a.run_info.get("input_files"), b.run_info.get("input_files")
    if not inputs_a or not inputs_b:
        checks.append(Check("input", False, False, "input files not recorded"))
    elif inputs_a == inputs_b:
        checks.append(Check("input", True, True, "same input"))
    elif [_canonical_input(f) for f in inputs_a] == [_canonical_input(f) for f in inputs_b]:
        checks.append(Check("input", True, True, "same input, read from another location"))
    else:
        checks.append(Check("input", False, True, "the two reconstructed different inputs"))

    events = paired_events(a.timing, b.timing)
    measured_a = (a.timing.execute["wall_s"].index != WARM_UP_EVENT).sum()
    measured_b = (b.timing.execute["wall_s"].index != WARM_UP_EVENT).sum()
    if len(events) == 0:
        checks.append(Check("events", False, True, "no event in common"))
    elif len(events) < max(measured_a, measured_b):
        checks.append(Check(
            "events", False, False,
            f"{len(events)} of {max(measured_a, measured_b)} events in common",
        ))
    else:
        checks.append(Check("events", True, True, f"{len(events)} events paired"))

    counted = [
        "instructions" in m.timing.execute
        and bool(m.timing.execute["instructions"].notna().to_numpy().any())
        for m in (a, b)
    ]
    checks.append(Check(
        "hardware counters", all(counted), False,
        "instructions counted in both" if all(counted) else
        "instructions not counted in both: a time change cannot be told apart "
        "into work and host speed",
    ))

    for name, key in (("host", "hostname"), ("CPU", "cpu_model")):
        value_a, value_b = a.machine_info.get(key), b.machine_info.get(key)
        if value_a and value_b and value_a != value_b:
            checks.append(Check(name, False, False, f"{value_a} → {value_b}"))
        elif value_a and value_b:
            checks.append(Check(name, True, False, str(value_a)))
    return sorted(checks, key=lambda check: not check.blocking)


def _canonical_input(location: str) -> str:
    """*location* as the store names it now. A file name alone does not tell
    inputs apart (the detector they were simulated in is a directory), so only
    the store's known moves are undone."""
    for old, new in INPUT_RELOCATIONS.items():
        location = location.replace(old, new)
    return location


def comparable(checks: list[Check]) -> bool:
    """Whether no blocking check failed."""
    return all(check.ok or not check.blocking for check in checks)


def _side(timing: ComponentTiming | None, metric: str, events: pd.Index,
          columns: pd.Index) -> pd.DataFrame:
    """Per-event contributions of *columns* over *events*: zero where a
    component did not run or does not exist, ``NaN`` where it ran without a
    value."""
    if timing is None:
        return pd.DataFrame(0.0, index=events, columns=columns)
    frame = timing.execute[metric].reindex(index=events, columns=columns)
    ran = timing.execute["wall_s"].reindex(index=events, columns=columns).notna()
    return frame.where(ran | frame.notna(), 0.0)


def _paired(a: pd.DataFrame, b: pd.DataFrame, status: pd.Series, seed: int) -> pd.DataFrame:
    """Deltas of *b* over *a*, two events × columns frames with the same index."""
    diff = b - a
    n = len(diff)
    out = pd.DataFrame({
        "status": status,
        "a_per_event": a.mean(skipna=False),
        "b_per_event": b.mean(skipna=False),
        "delta": diff.mean(skipna=False),
        "n_events": n,
    })
    out["rel"] = out["delta"] / out["a_per_event"].where(out["a_per_event"] != 0)
    low = high = np.full(len(out), np.nan)
    if n:
        # Resampled event weights: one row per resample, summing to one.
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=RESAMPLES) / n
        values = diff.to_numpy(dtype=float)
        finite = np.isfinite(values).all(axis=0)
        means = weights @ np.where(np.isfinite(values), values, 0.0)
        tail = (1 - CONFIDENCE) / 2 * 100
        low = np.where(finite, np.percentile(means, tail, axis=0), np.nan)
        high = np.where(finite, np.percentile(means, 100 - tail, axis=0), np.nan)
    out["ci_low"], out["ci_high"] = low, high
    return out


def paired_deltas(
    a: ComponentTiming, b: ComponentTiming, metric: str = "wall_s", *, seed: int = 0,
) -> pd.DataFrame:
    """How *metric* moved from *a* to *b*, per component, over their
    :func:`paired_events`.

    One row per component that ran in either, *a*'s order first, with columns
    ``status`` (``common``, ``added`` in *b* or ``removed`` from it),
    ``a_per_event`` and ``b_per_event`` (what it added to an average paired
    event; zero on the side it is missing from), ``delta`` (``b − a``),
    ``rel`` (``delta`` over ``a_per_event``), ``ci_low`` and ``ci_high`` (the
    bootstrap interval of ``delta``) and ``n_events``. A component that ran
    without a value has ``NaN`` statistics. Refused for a metric either job
    did not record.
    """
    for side, timing in (("a", a), ("b", b)):
        if metric not in timing.execute:
            raise ValueError(f"{side} has no {metric} recorded")
    events = paired_events(a, b)
    ran_a = a.execute["wall_s"].columns[a.execute["wall_s"].notna().any()]
    ran_b = b.execute["wall_s"].columns[b.execute["wall_s"].notna().any()]
    columns = ran_a.append(ran_b.difference(ran_a, sort=False))
    status = pd.Series("common", index=columns)
    status[columns.difference(ran_b)] = "removed"
    status[columns.difference(ran_a)] = "added"
    out = _paired(
        _side(a, metric, events, columns), _side(b, metric, events, columns), status, seed,
    )
    out.index.name = "component"
    return out


def paired_total_delta(
    a: ComponentTiming, b: ComponentTiming, metric: str = "wall_s", *, seed: int = 0,
) -> pd.DataFrame:
    """:func:`paired_deltas` of the whole event, as one :data:`EVENT_TOTAL` row."""
    events = paired_events(a, b)
    total_a = event_totals(a, metric).reindex(events).to_frame(EVENT_TOTAL)
    total_b = event_totals(b, metric).reindex(events).to_frame(EVENT_TOTAL)
    return _paired(total_a, total_b, pd.Series("common", index=[EVENT_TOTAL]), seed)


def run_to_run_noise(trend: pd.DataFrame, label: str, metric: str = "wall_s") -> pd.Series:
    """The relative change of each component's ``per_event`` *metric* from one
    run to the next that the window's runs of *label* show without anything
    changing: a robust spread (1.4826 × the median absolute deviation) of the
    log ratios of consecutive runs, so that one real step among them does not
    widen it. Indexed by component, plus :data:`EVENT_TOTAL`; ``NaN`` with
    fewer than :data:`MIN_NOISE_RATIOS` ratios.

    *trend* is :func:`k4bench.analysis.trend.build_component_trend`'s frame.
    """
    rows = trend[(trend["label"] == label) & (trend["metric"] == metric)]
    if rows.empty:
        return pd.Series(dtype=float)
    order = rows.drop_duplicates("run_dir").sort_values(["run_date", "run_dir"])["run_dir"]
    table = rows.pivot(index="run_dir", columns="component", values="per_event").reindex(order)
    # A component a run lacks costs nothing there; one without a value leaves
    # the run's total unknown.
    by_run = rows.groupby("run_dir")["per_event"]
    table[EVENT_TOTAL] = by_run.sum().where(by_run.count() == by_run.size()).reindex(order)
    ratios = np.log(table.where(table > 0)).diff().iloc[1:]
    spread = 1.4826 * (ratios - ratios.median()).abs().median()
    return spread.where(ratios.count() >= MIN_NOISE_RATIOS)


@dataclass(frozen=True)
class Floors:
    """The smallest change worth calling one, whatever the noise."""

    #: Per event, in the metric's unit.
    absolute: float
    #: As a fraction of the earlier value.
    relative: float


#: 1 ms per event and 1%.
TIME_FLOORS = Floors(absolute=1e-3, relative=0.01)
#: A million instructions per event and 0.1%: instructions repeat to ~0.001%.
INSTRUCTION_FLOORS = Floors(absolute=1e6, relative=0.001)
#: A hundred allocations per event and 0.1%: allocation counts repeat exactly
#: but for the odd event.
ALLOCATION_FLOORS = Floors(absolute=100, relative=0.001)


def significant_changes(
    deltas: pd.DataFrame, floors: Floors, noise: pd.Series | None = None,
) -> pd.DataFrame:
    """Which of *deltas* (:func:`paired_deltas` or :func:`paired_total_delta`)
    are real changes, and why not when not.

    A change is significant when its interval excludes zero and it exceeds
    *floors* and :data:`NOISE_SIGMAS` × the component's *noise*; a component
    without a noise estimate is held to the floors alone, and one that is
    added, removed or cost nothing in the earlier run to the absolute floor
    alone. Columns:
    ``significant``, ``threshold`` (the relative change it had to exceed),
    ``noise_known`` and ``reason``: ``significant``, ``below_floor`` (smaller
    than the absolute floor), ``within_noise`` (smaller than the threshold),
    ``uncertain`` (its interval includes zero), ``too_few_events``, or
    ``unknown`` (no value to judge).
    """
    noise = (noise if noise is not None else pd.Series(dtype=float)).reindex(deltas.index)
    threshold = np.maximum(floors.relative, NOISE_SIGMAS * noise.fillna(0.0))
    # A component one side lacks, or one that cost nothing before, has no
    # earlier value to be relative to: it is judged by its size.
    by_size = (deltas["status"] != "common") | deltas["a_per_event"].eq(0)
    reason = pd.Series("significant", index=deltas.index)
    reason[deltas["ci_low"].le(0) & deltas["ci_high"].ge(0)] = "uncertain"
    reason[~by_size & deltas["rel"].abs().lt(threshold)] = "within_noise"
    reason[deltas["delta"].abs().lt(floors.absolute)] = "below_floor"
    reason[deltas["n_events"].lt(MIN_PAIRED_EVENTS)] = "too_few_events"
    reason[deltas["delta"].isna() | (~by_size & deltas["rel"].isna())] = "unknown"
    return pd.DataFrame({
        "significant": reason == "significant",
        "threshold": threshold,
        "noise_known": noise.notna(),
        "reason": reason,
    })


def classify_changes(
    time: pd.DataFrame,
    *,
    instructions: pd.DataFrame | None = None,
    allocations: pd.DataFrame | None = None,
    time_noise: pd.Series | None = None,
    instruction_noise: pd.Series | None = None,
) -> pd.DataFrame:
    """Which components' time changed, and what the work counts say about why.

    *time* holds :func:`paired_deltas` (or :func:`paired_total_delta`) of
    ``wall_s``; *instructions* and *allocations* the same rows' deltas of
    those metrics, where they were recorded. The :func:`significant_changes`
    columns of *time* come with:

    - ``direction`` — ``slower`` or ``faster``, for a significant change;
    - ``work_changed`` — whether the instructions, or without them the
      allocation counts, changed significantly; ``NaN`` when neither was
      recorded for the component;
    - ``work_fraction`` — the instructions' relative change over the time's:
      about 1 when the time followed the work, about 0 when the host ran the
      same work slower or faster;
    - ``cause`` — for a significant change: ``added`` or ``removed`` for a
      component one side lacks; ``work`` when the instructions changed and
      account for at least :data:`WORK_EXPLAINS` of it, or, without them, the
      allocation counts changed; ``host`` when the instructions stayed within
      their floors and noise, or account for less; ``unknown`` when only
      allocation counts were recorded and they did not change, which does not
      show that the work stayed the same, or when the instructions' change is
      uncertain yet large enough to account for it.
    """
    out = significant_changes(time, TIME_FLOORS, time_noise)
    out["direction"] = pd.Series(np.where(time["delta"] > 0, "slower", "faster"),
                                 index=time.index).where(out["significant"])

    work_changed = pd.Series(np.nan, index=time.index, dtype=object)
    work_fraction = pd.Series(np.nan, index=time.index)
    counted = pd.Series(False, index=time.index)
    unchanged = pd.Series(False, index=time.index)
    if allocations is not None:
        judged = significant_changes(allocations.reindex(time.index), ALLOCATION_FLOORS)
        known = judged["reason"] != "unknown"
        work_changed[known] = judged["significant"][known]
    if instructions is not None:
        instructions = instructions.reindex(time.index)
        judged = significant_changes(instructions, INSTRUCTION_FLOORS, instruction_noise)
        counted = judged["reason"] != "unknown"
        # An uncertain change shows neither that the work changed nor that
        # it stayed the same.
        unchanged = judged["reason"].isin(["below_floor", "within_noise"])
        work_changed[counted] = judged["significant"][counted]
        work_fraction = instructions["rel"] / time["rel"].where(time["rel"] != 0)
    out["work_changed"] = work_changed
    out["work_fraction"] = work_fraction

    cause = pd.Series("unknown", index=time.index)
    cause[work_changed.eq(True)] = "work"
    cause[counted & (unchanged | work_fraction.lt(WORK_EXPLAINS))] = "host"
    cause[time["status"] != "common"] = time["status"]
    out["cause"] = cause.where(out["significant"])
    return out
