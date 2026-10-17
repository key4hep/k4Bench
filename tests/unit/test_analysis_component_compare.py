"""Unit tests for :mod:`k4bench.analysis.component_compare` — paired
comparison of two k4run measurements."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from k4bench.analysis.component_compare import (
    EVENT_TOTAL,
    MIN_PAIRED_EVENTS,
    Measurement,
    classify_changes,
    comparability,
    comparable,
    load_measurement,
    paired_deltas,
    paired_events,
    paired_total_delta,
    run_to_run_noise,
    significant_changes,
    TIME_FLOORS,
)
from k4bench.analysis.components import ComponentTiming

NIGHTS = Path(__file__).resolve().parents[1] / "fixtures" / "reco_nights"

EVENTS = 40


def _event_costs(seed: int, mean: float) -> np.ndarray:
    """Per-event costs around *mean*: events differ by ~30%, as CLD's do."""
    rng = np.random.default_rng(seed)
    return mean * np.clip(1 + 0.3 * rng.standard_normal(EVENTS), 0.2, None)


#: What each component costs per event in the reference job.
TRACK = _event_costs(1, 2.0)
VERTEX = _event_costs(2, 0.5)
TINY = _event_costs(3, 0.0005)


def job(costs: dict[str, np.ndarray], *, instructions: dict | None = None,
        allocations: dict | None = None, events=None) -> ComponentTiming:
    """A job of a sequencer ``Top`` running the components of *costs*, each
    an array of per-event wall times (``NaN`` where it did not run)."""
    names = ["Top", *costs]
    events = list(range(EVENTS)) if events is None else list(events)
    n = len(events)

    def matrix(values: dict[str, np.ndarray], top: float) -> list[list]:
        columns = [np.full(n, top), *(np.asarray(values[c], dtype=float)[:n] for c in costs)]
        return [[None if math.isnan(v) else float(v) for v in row] for row in zip(*columns)]

    zeros = {c: np.where(np.isnan(np.asarray(v, float)[:n]), np.nan, 0.0) for c, v in costs.items()}
    execute = {
        "wall_s": matrix(costs, 1e-4),
        "cpu_s": matrix(costs, 1e-4),
        "peak_rss_increase_mb": matrix(zeros, 0.0),
    }
    if instructions is not None:
        execute["instructions"] = matrix(instructions, 1e5)
    if allocations is not None:
        execute["allocations"] = matrix(allocations, 0.0)
    zero_per_component = [0.0] * len(names)
    return ComponentTiming.from_json({
        "phases": {},
        "components": [
            {"name": name, "category": "algorithm", "type": "T", "impl": name,
             "parent": None if name == "Top" else 0}
            for name in names
        ],
        "lifecycle": {"initialize": {
            "wall_s": zero_per_component, "cpu_s": zero_per_component,
            "peak_rss_increase_mb": zero_per_component,
        }},
        "event_numbers": events,
        "execute": execute,
    })


def _scaled(values: dict[str, np.ndarray], by: dict[str, float]) -> dict[str, np.ndarray]:
    return {c: v * by.get(c, 1.0) for c, v in values.items()}


COSTS = {"Track": TRACK, "Vertex": VERTEX, "Tiny": TINY}
#: Instructions per second of wall time: one host's speed.
RATE = 3e9


def _instructions(costs, by=None):
    return {c: v * RATE * (by or {}).get(c, 1.0) for c, v in costs.items()}


# ── paired deltas ────────────────────────────────────────────────────────────

def test_identical_jobs_have_no_delta():
    deltas = paired_deltas(job(COSTS), job(COSTS))
    assert (deltas["status"] == "common").all()
    assert deltas["delta"].abs().max() == pytest.approx(0.0)
    assert (deltas["ci_low"] == 0).all() and (deltas["ci_high"] == 0).all()


def test_the_warm_up_event_is_not_paired():
    deltas = paired_deltas(job(COSTS), job(COSTS))
    assert (deltas["n_events"] == EVENTS - 1).all()
    assert 0 not in paired_events(job(COSTS), job(COSTS))


def test_a_change_is_measured_event_by_event():
    slower = job(_scaled(COSTS, {"Track": 1.10}))
    track = paired_deltas(job(COSTS), slower).loc["Track"]
    assert track["rel"] == pytest.approx(0.10)
    assert track["delta"] == pytest.approx(0.10 * TRACK[1:].mean())
    # Paired, the events' own 30% spread cancels: the interval is the change.
    assert track["ci_low"] == pytest.approx(track["delta"], rel=0.2)
    assert track["ci_low"] > 0


def test_removed_and_added_components_add_up_to_the_events_change():
    a = job(COSTS)
    b = job({"Track": TRACK * 1.05, "Truth": VERTEX * 3, "Tiny": TINY})
    deltas = paired_deltas(a, b)
    assert deltas["status"].to_dict() == {
        "Top": "common", "Track": "common", "Vertex": "removed", "Tiny": "common",
        "Truth": "added",
    }
    assert deltas.loc["Vertex", "delta"] == pytest.approx(-VERTEX[1:].mean())
    assert deltas.loc["Vertex", "rel"] == pytest.approx(-1.0)
    assert deltas.loc["Truth", "delta"] == pytest.approx(3 * VERTEX[1:].mean())
    assert math.isnan(deltas.loc["Truth", "rel"])
    total = paired_total_delta(a, b)
    assert total.index.tolist() == [EVENT_TOTAL]
    assert deltas["delta"].sum() == pytest.approx(total.loc[EVENT_TOTAL, "delta"])


def test_only_the_events_both_processed_are_paired():
    a, b = job(COSTS), job(COSTS, events=range(EVENTS // 2))
    assert paired_deltas(a, b)["n_events"].iloc[0] == EVENTS // 2 - 1


def test_a_metric_one_side_did_not_record_is_refused():
    with pytest.raises(ValueError, match="b has no instructions"):
        paired_deltas(job(COSTS, instructions=_instructions(COSTS)), job(COSTS), "instructions")


# ── comparability ────────────────────────────────────────────────────────────

URL = "https://store/inputs/reco/CLD/CLD_o2_v09/zbb_100ev_seed42.edm4hep.root"


def measurement(timing=None, **run_info) -> Measurement:
    machine = run_info.pop("machine", {"hostname": "host-1", "cpu_model": "EPYC"})
    return Measurement(
        run_dir=Path("/runs/x"), label="baseline", timing=timing or job(COSTS),
        run_info={"input_files": [URL], **run_info}, machine_info=machine,
    )


def _check(checks, name):
    return next(check for check in checks if check.name == name)


def test_two_runs_of_the_same_input_on_one_host_compare():
    a = measurement(job(COSTS, instructions=_instructions(COSTS)))
    checks = comparability(a, a)
    assert comparable(checks)
    assert all(check.ok for check in checks)
    assert checks[0].blocking


def test_different_inputs_do_not_compare():
    checks = comparability(measurement(), measurement(input_files=[URL.replace("100ev", "10ev")]))
    assert not _check(checks, "input").ok
    assert not comparable(checks)


def test_an_input_moved_in_the_store_is_the_same_input():
    moved = URL.replace("inputs/reco/CLD/CLD_o2_v09", "_inputs/CLD_o2_v09")
    check = _check(comparability(measurement(), measurement(input_files=[moved])), "input")
    assert check.ok and "another location" in check.detail


def test_caveats_do_not_block_a_comparison():
    other = measurement(machine={"hostname": "host-2", "cpu_model": "EPYC"}, input_files=None)
    checks = comparability(measurement(), other)
    assert comparable(checks)
    failed = {check.name for check in checks if not check.ok}
    assert failed == {"input", "host", "hardware counters"}


def test_runs_without_a_common_event_do_not_compare():
    disjoint = job(COSTS, events=range(EVENTS, 2 * EVENTS))
    checks = comparability(measurement(), measurement(disjoint))
    assert not _check(checks, "events").ok
    assert not comparable(checks)


# ── significance ─────────────────────────────────────────────────────────────

def _reasons(a, b, noise=None):
    deltas = pd.concat([paired_deltas(a, b), paired_total_delta(a, b)])
    return significant_changes(deltas, TIME_FLOORS, noise)["reason"]


def test_a_large_relative_change_of_a_tiny_component_is_below_the_floor():
    reasons = _reasons(job(COSTS), job(_scaled(COSTS, {"Tiny": 2.0})))
    assert reasons["Tiny"] == "below_floor"


def test_a_change_smaller_than_the_noise_is_within_it():
    b = job(_scaled(COSTS, {"Track": 1.03}))
    assert _reasons(job(COSTS), b)["Track"] == "significant"
    noisy = pd.Series({"Track": 0.02})
    assert _reasons(job(COSTS), b, noisy)["Track"] == "within_noise"


def test_a_change_whose_interval_includes_zero_is_uncertain():
    # Half the events slower, half faster: large per-event changes, none on average.
    signs = np.where(np.arange(EVENTS) % 2, 1.2, 0.8)
    signs[-1] = 1.4  # leave a mean change above the floors
    b = job({**COSTS, "Track": TRACK * signs})
    assert _reasons(job(COSTS), b)["Track"] == "uncertain"


def test_too_few_events_are_not_judged():
    few = range(MIN_PAIRED_EVENTS)  # the warm-up leaves one too few
    reasons = _reasons(job(COSTS, events=few), job(_scaled(COSTS, {"Track": 2.0}), events=few))
    assert reasons["Track"] == "too_few_events"


# ── causes ───────────────────────────────────────────────────────────────────

def _classify(a, b, **metrics):
    def both(metric):
        return pd.concat([paired_deltas(a, b, metric), paired_total_delta(a, b, metric)])

    return classify_changes(both("wall_s"), **{name: both(m) for name, m in metrics.items()})


def test_the_same_work_run_slower_is_the_host():
    slow_host = {"Track": 1.4, "Vertex": 1.4}
    a = job(COSTS, instructions=_instructions(COSTS))
    b = job(_scaled(COSTS, slow_host), instructions=_instructions(COSTS))
    verdict = _classify(a, b, instructions="instructions")
    assert verdict.loc["Track", "cause"] == "host"
    assert verdict.loc["Track", "direction"] == "slower"
    assert verdict.loc[EVENT_TOTAL, "cause"] == "host"
    assert verdict.loc["Track", "work_fraction"] == pytest.approx(0.0, abs=1e-9)


def test_more_work_taking_more_time_is_the_work():
    more = {"Vertex": 1.5}
    a = job(COSTS, instructions=_instructions(COSTS))
    b = job(_scaled(COSTS, more), instructions=_instructions(COSTS, more))
    verdict = _classify(a, b, instructions="instructions")
    assert verdict.loc["Vertex", "cause"] == "work"
    assert verdict.loc["Vertex", "work_fraction"] == pytest.approx(1.0)
    assert verdict.loc["Track", "significant"] == False  # noqa: E712


def test_a_work_change_too_small_to_explain_the_time_is_the_host():
    a = job(COSTS, instructions=_instructions(COSTS))
    b = job(_scaled(COSTS, {"Track": 1.2}), instructions=_instructions(COSTS, {"Track": 1.01}))
    verdict = _classify(a, b, instructions="instructions")
    assert verdict.loc["Track", "work_changed"] == True  # noqa: E712
    assert verdict.loc["Track", "cause"] == "host"


def test_without_instructions_only_changed_allocations_say_anything():
    allocations = {c: np.where(np.isnan(v), np.nan, 1e5) for c, v in COSTS.items()}
    a = job(COSTS, allocations=allocations)
    b = job(_scaled(COSTS, {"Track": 1.2, "Vertex": 1.2}),
            allocations={**allocations, "Vertex": allocations["Vertex"] * 2})
    verdict = _classify(a, b, allocations="allocations")
    assert verdict.loc["Vertex", "cause"] == "work"
    # Unchanged allocations do not show the work stayed the same.
    assert verdict.loc["Track", "cause"] == "unknown"


def test_added_and_removed_components_are_their_own_cause():
    verdict = _classify(job(COSTS), job({"Track": TRACK, "Truth": VERTEX, "Tiny": TINY}))
    assert verdict.loc["Vertex", "cause"] == "removed"
    assert verdict.loc["Truth", "cause"] == "added"
    assert verdict.loc["Vertex", "direction"] == "faster"


def test_unchanged_components_have_no_cause():
    verdict = _classify(job(COSTS), job(COSTS))
    assert not verdict["significant"].any()
    assert verdict["cause"].isna().all() and verdict["direction"].isna().all()


# ── run-to-run noise ─────────────────────────────────────────────────────────

def _trend(per_event: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {"run_dir": f"/runs/{night}", "run_id": f"2026-10-{night:02d}",
         "run_date": pd.Timestamp(f"2026-10-{night:02d}"), "label": "baseline",
         "metric": "wall_s", "component": component, "per_event": value}
        for component, values in per_event.items()
        for night, value in enumerate(values, start=10)
    ]
    return pd.DataFrame(rows)


def test_noise_is_the_robust_spread_of_run_to_run_ratios():
    # ±1% alternating, plus one 40% step that must not widen it.
    track = [2.0, 2.02, 2.0, 2.02, 2.0, 2.8, 2.8]
    noise = run_to_run_noise(_trend({"Track": track, "Vertex": [0.5] * 7}), "baseline")
    assert noise["Track"] == pytest.approx(1.4826 * math.log(1.01), rel=0.05)
    assert noise["Vertex"] == 0.0
    assert EVENT_TOTAL in noise.index


def test_too_short_a_history_gives_no_noise():
    noise = run_to_run_noise(_trend({"Track": [2.0, 2.1, 2.0]}), "baseline")
    assert math.isnan(noise["Track"])


def test_a_component_one_run_lacks_counts_as_nothing_in_its_total():
    trend = _trend({"Track": [2.0] * 5, "Truth": [1.0] * 5})
    trend = trend[~((trend["component"] == "Truth") & (trend["run_id"] == "2026-10-12"))]
    noise = run_to_run_noise(trend, "baseline")
    assert noise[EVENT_TOTAL] > 0


# ── the CLD nights ───────────────────────────────────────────────────────────

def _night(night: str, label: str = "baseline") -> Measurement:
    return load_measurement(NIGHTS / night, label)


def _verdict(a: Measurement, b: Measurement) -> pd.DataFrame:
    return _classify(a.timing, b.timing, instructions="instructions", allocations="allocations")


def test_cld_the_same_software_twice_is_unchanged():
    a, b = _night("2026-10-16"), _night("2026-10-17")
    assert comparable(comparability(a, b))
    assert not _verdict(a, b)["significant"].any()


def test_cld_the_night_the_host_ran_slower():
    a, b = _night("2026-10-16"), _night("2026-10-14")
    checks = comparability(a, b)
    assert comparable(checks)
    assert {c.name for c in checks if not c.ok} == {"host"}
    verdict = _verdict(a, b)
    significant = verdict[verdict["significant"]]
    assert {"MyConformalTracking", "Refit", EVENT_TOTAL} <= set(significant.index)
    assert (significant["cause"] == "host").all()
    deltas = paired_deltas(a.timing, b.timing)
    assert deltas.loc["MyConformalTracking", "rel"] == pytest.approx(0.42, abs=0.03)


def test_cld_truth_tracking_replaces_the_track_finding():
    a, b = _night("2026-10-17"), _night("2026-10-17", "variant_truth_tracking")
    verdict = _verdict(a, b)
    assert verdict.loc["MyConformalTracking", "cause"] == "removed"
    assert verdict.loc["MyTruthTrackFinder", "cause"] == "added"
    assert verdict.loc["VertexFinder", "cause"] == "work"
    assert verdict.loc[EVENT_TOTAL, "direction"] == "slower"
