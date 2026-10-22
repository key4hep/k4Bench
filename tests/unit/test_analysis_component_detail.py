"""One component's detail, assembled from the timed run and the memory pass:
every number computed once, a metric the run lacks left unknown rather than
zero, and a run the component is missing from a gap in its history."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from k4bench.analysis.component_detail import (
    ABSENT,
    HELD,
    HELD_NONE,
    NO_MEMORY_PASS,
    NOT_TIMED,
    TIMED,
    component_choices,
    component_detail,
    component_history,
    component_summary,
)
from k4bench.analysis.components import ComponentTiming, event_loop_time
from k4bench.analysis.loader import load_each_component_timing
from k4bench.analysis.plots import Measure, plot_standing
from k4bench.analysis.trend import build_component_summary_trend, build_heap_trend

RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"
_MB = 1024**2
NaN = np.nan


def _timing(execute: dict[str, dict[str, list]], *, lifecycle=None, categories=None,
            error: str | None = None) -> ComponentTiming:
    """A job of events 0–3 (0 the warm-up) whose components' per-event
    *execute* values are given by metric, then by component."""
    events = pd.Index([0, 1, 2, 3], name="event_number")
    names = list(execute["wall_s"])
    categories = categories or {}
    components = pd.DataFrame({
        "category": [categories.get(n, "algorithm") for n in names],
        "type": [f"{n}Type" for n in names],
        "impl": [f"{n}Impl" for n in names],
        "library": "", "impl_library": "", "execute_calls": 3,
    }, index=pd.Index(names, name="component"))
    components.insert(5, "parent", pd.Series([None] * len(names), index=components.index, dtype=object))
    return ComponentTiming(
        components=components,
        execute={m: pd.DataFrame(v, index=events, dtype=float) for m, v in execute.items()},
        lifecycle={
            phase: pd.DataFrame(table, index=components.index, dtype=float)
            for phase, table in (lifecycle or {}).items()
        },
        phases={}, process_start_epoch_ns=None, measurement_overhead_ns=None, threads=1,
        hardware_counters_error=error,
    )


def _full() -> ComponentTiming:
    """Always runs in every event; Filtered only in event 1 after the warm-up."""
    return _timing({
        "wall_s": {"Always": [9.0, 0.004, 0.002, 0.003], "Filtered": [1.0, 0.006, NaN, NaN]},
        "cpu_s": {"Always": [8.0, 0.003, 0.002, 0.001], "Filtered": [1.0, 0.005, NaN, NaN]},
        "instructions": {"Always": [1e12, 4e9, 2e9, 3e9], "Filtered": [1e9, 6e9, NaN, NaN]},
        "cycles": {"Always": [1e12, 2e9, 2e9, 2e9], "Filtered": [1e9, 3e9, NaN, NaN]},
        "minor_page_faults": {"Always": [9e5, 10, 20, 30], "Filtered": [1, 4, NaN, NaN]},
        "major_page_faults": {"Always": [50, 0, 0, 0], "Filtered": [0, 0, NaN, NaN]},
        "voluntary_context_switches": {"Always": [5, 1, 0, 2], "Filtered": [0, 0, NaN, NaN]},
        "involuntary_context_switches": {"Always": [5, 0, 3, 0], "Filtered": [0, 1, NaN, NaN]},
        "allocations": {"Always": [1e6, 100, 300, 200], "Filtered": [5, 60, NaN, NaN]},
        "allocated_bytes": {"Always": [1e9, 1 * _MB, 3 * _MB, 2 * _MB], "Filtered": [1, 2 * _MB, NaN, NaN]},
        # The writer frees what the reader allocated: more freed than allocated.
        "net_allocated_bytes": {"Always": [5e8, -2 * _MB, -4 * _MB, -3 * _MB],
                                "Filtered": [0, 1 * _MB, NaN, NaN]},
        "peak_heap_bytes": {"Always": [9e9, 4 * _MB, 8 * _MB, 6 * _MB], "Filtered": [0, _MB, NaN, NaN]},
        "largest_allocation_bytes": {"Always": [9e9, _MB, 3 * _MB, 2 * _MB], "Filtered": [0, _MB, NaN, NaN]},
    }, lifecycle={
        "initialize": {"wall_s": [0.5, 0.25], "cpu_s": [0.4, 0.2], "allocations": [10, 20]},
        "finalize": {"wall_s": [0.1, NaN], "cpu_s": [0.1, NaN], "allocations": [1, NaN]},
    })


def test_every_number_of_a_component_run_in_every_event():
    s = component_summary(_full()).loc["Always"]
    # The warm-up event 0 is left out of everything over events.
    assert s["events"] == s["executed_events"] == 3 and s["coverage"] == 1
    assert s["seconds"] == pytest.approx(0.009)
    assert s["seconds_when_run"] == s["seconds_per_event"] == pytest.approx(0.003)
    assert s["p50_s"] == pytest.approx(0.003)
    assert s["p95_s"] == pytest.approx(0.0039)
    assert s["cpu_s_per_event"] == pytest.approx(0.002)
    # A ratio of sums: 9e9 / 6e9, not the mean of 2, 1 and 1.5.
    assert s["ipc"] == pytest.approx(1.5) and s["ipc_events"] == 3
    assert s["instructions_per_event"] == pytest.approx(3e9)
    assert s["minor_page_faults"] == 60 and s["minor_page_faults_per_event"] == 20
    # A counter that stayed at zero is zero, counted in every event.
    assert s["major_page_faults"] == 0 and s["major_page_faults_events"] == 3
    assert s["allocations_per_s"] == pytest.approx(600 / 0.009)
    assert s["allocated_bytes_per_event"] == pytest.approx(2 * _MB)
    # Net allocation keeps its sign.
    assert s["net_allocated_bytes"] == pytest.approx(-9 * _MB)
    assert s["net_allocated_bytes_per_event"] == pytest.approx(-3 * _MB)
    # Inclusive metrics are not summed: their largest and their median.
    assert s["peak_heap_bytes_max"] == 8 * _MB and s["peak_heap_bytes_median"] == 6 * _MB
    assert s["largest_allocation_bytes_max"] == 3 * _MB
    assert s["initialize_s"] == 0.5 and s["finalize_s"] == 0.1
    # A phase the job did not record has no time.
    assert np.isnan(s["start_s"])


def test_a_filtered_component_is_timed_in_the_events_it_ran_in():
    s = component_summary(_full()).loc["Filtered"]
    assert s["executed_events"] == 1 and s["coverage"] == pytest.approx(1 / 3)
    assert s["seconds_when_run"] == pytest.approx(0.006)
    assert s["seconds_per_event"] == pytest.approx(0.002)
    assert s["p50_s"] == s["p95_s"] == pytest.approx(0.006)
    # Counts are over the event it ran in, not every event.
    assert s["instructions_per_event"] == 6e9 and s["instructions_events"] == 1
    assert s["ipc"] == 2.0
    assert s["major_page_faults"] == 0 and s["major_page_faults_events"] == 1
    # Not called in finalize: no time there, not zero.
    assert np.isnan(s["finalize_s"])


def test_the_summary_agrees_with_the_recorded_events_and_the_event_loop_time():
    timings, _ = load_each_component_timing(RECO_RUN)
    timing = timings["baseline"]
    summary = component_summary(timing)
    # The tracking ran in every event: its sums over events 1-11, by hand.
    events = slice(1, None)
    instructions = timing.execute["instructions"]["MyConformalTracking"].iloc[events].sum()
    cycles = timing.execute["cycles"]["MyConformalTracking"].iloc[events].sum()
    assert summary.at["MyConformalTracking", "ipc"] == pytest.approx(instructions / cycles)
    wall = timing.execute["wall_s"]["MyConformalTracking"].iloc[events]
    assert summary.at["MyConformalTracking", "seconds_when_run"] == pytest.approx(wall.mean())
    loop = event_loop_time(timing.execute["wall_s"])
    pd.testing.assert_frame_equal(summary[loop.columns], loop)


def test_without_hardware_counters_the_time_and_allocations_stay():
    full = _full()
    execute = {m: f for m, f in full.execute.items() if m not in ("instructions", "cycles")}
    timing = ComponentTiming(**{**full.__dict__, "execute": execute,
                                "hardware_counters_error": "perf_event_open: EACCES"})
    s = component_summary(timing).loc["Always"]
    assert np.isnan(s["ipc"]) and s["ipc_events"] == 0
    assert np.isnan(s["instructions"]) and s["instructions_events"] == 0
    assert s["seconds_when_run"] == pytest.approx(0.003)
    assert s["allocations_per_event"] == 200
    detail = component_detail("Always", timing)
    assert detail.hardware_counters_error == "perf_event_open: EACCES"
    assert "instructions" not in detail.events


def test_counters_missing_in_some_events_say_how_many_they_cover():
    timing = _full()
    timing.execute["cycles"].loc[2, "Always"] = NaN
    s = component_summary(timing).loc["Always"]
    assert s["instructions_events"] == 3 and s["cycles_events"] == 2
    # Instructions and cycles are set against each other only where both were counted.
    assert s["ipc_events"] == 2
    assert s["ipc"] == pytest.approx((4e9 + 3e9) / (2e9 + 2e9))


def test_a_component_run_in_no_event_has_no_time_and_no_counts():
    timing = _timing({
        "wall_s": {"Ran": [1.0, 0.1, 0.1, 0.1], "Never": [1.0, NaN, NaN, NaN]},
        "allocations": {"Ran": [1, 2, 2, 2], "Never": [1, NaN, NaN, NaN]},
    })
    s = component_summary(timing).loc["Never"]
    assert s["executed_events"] == 0
    for column in ("seconds", "seconds_when_run", "seconds_per_event", "p50_s",
                   "allocations", "allocations_per_event", "allocations_per_s"):
        assert np.isnan(s[column]), column
    detail = component_detail("Never", timing)
    assert detail.timed and detail.events.empty


def test_the_detail_lists_its_phases_and_the_events_it_ran_in():
    timing = _full()
    detail = component_detail("Filtered", timing)
    assert detail.identity["impl"] == "FilteredImpl"
    assert detail.lifecycle.index.tolist() == ["initialize"]
    assert detail.lifecycle.at["initialize", "wall_s"] == 0.25
    # Event 1 only: the warm-up left out, the events it was skipped in too.
    assert detail.events.index.tolist() == [1]
    assert detail.events.at[1, "wall_s"] == 0.006
    assert component_detail("Unknown", timing).identity is None
    assert not component_detail("Unknown", timing).timed


def test_the_memory_pass_is_its_own_summary_beside_the_timed_run():
    timing = _full()
    heap = pd.DataFrame({"typical_mb": [12.0, 0.0], "leak": [True, False]},
                        index=["Always", "Filtered"])
    with_heap = component_detail("Always", timing, heap_summary=heap, heap_holders=["Always"])
    assert with_heap.heap["typical_mb"] == 12.0
    # The timed run's events are its own: the pass's are never joined to them.
    assert with_heap.events.index.tolist() == [1, 2, 3]
    assert component_detail("Filtered", timing, heap_summary=heap, heap_holders=["Always"]).heap is None
    assert component_detail("Always", timing).heap is None
    # Without the timed run, the pass alone.
    alone = component_detail("Always", None, heap_summary=heap)
    assert alone.heap is not None and not alone.timed


def test_the_choices_are_the_algorithms_and_services_in_the_jobs_order():
    timing = _timing(
        {"wall_s": {"Svc": [0.0] * 4, "Alg": [1.0] * 4, "Tool": [0.0] * 4}},
        categories={"Svc": "service", "Tool": "other"},
    )
    choices = component_choices(timing)
    assert choices.index.tolist() == ["Svc", "Alg"]
    assert choices.at["Alg", "impl"] == "AlgImpl"


def _summary_rows(run_id: str, label: str, components: dict[str, float]) -> list[dict]:
    """A :func:`build_component_summary_trend` run of *label*, with each of
    *components*' time in the events it ran in."""
    run = {"run_id": run_id, "run_date": pd.Timestamp(run_id), "k4h_release_date": pd.Timestamp(run_id),
           "k4h_release": f"key4hep-{run_id}", "x_date": pd.Timestamp(run_id), "label": label}
    return [{**run, "component": None}] + [
        {**run, "component": name, "seconds_when_run": s, "p95_s": s, "coverage": 1.0,
         **{c: NaN for c in ("initialize_s", "finalize_s", "instructions_per_event", "ipc",
                             "minor_page_faults_per_event", "major_page_faults_per_event",
                             "voluntary_context_switches_per_event",
                             "involuntary_context_switches_per_event", "allocations_per_event",
                             "allocated_bytes_per_event", "net_allocated_bytes_per_event",
                             "largest_allocation_bytes_max")}}
        for name, s in components.items()
    ]


def test_the_history_leaves_a_gap_where_the_component_is_missing():
    trend = pd.DataFrame([
        *_summary_rows("2026-10-18", "baseline", {"Tracking": 2.0, "Vertex": 0.5}),
        # Not in the job that night.
        *_summary_rows("2026-10-19", "baseline", {"Vertex": 0.6}),
        # The baseline was not timed that night; another configuration was.
        *_summary_rows("2026-10-20", "variant", {"Tracking": 9.0}),
        # Timed, and genuinely free.
        *_summary_rows("2026-10-21", "baseline", {"Tracking": 0.0}),
    ])
    history = component_history(trend, None, "baseline", "Tracking")
    assert history["status"].tolist() == [TIMED, ABSENT, NOT_TIMED, TIMED]
    assert history["label"].unique().tolist() == ["Tracking"]
    ms = history["ms_when_run"].tolist()
    assert ms[0] == 2000.0 and np.isnan(ms[1]) and np.isnan(ms[2]) and ms[3] == 0.0
    assert (history["heap_status"] == NO_MEMORY_PASS).all()
    assert component_history(None, None, "baseline", "Tracking").empty


def test_the_heap_history_tells_no_heap_held_from_no_memory_pass(tmp_path, reco_run_with_heap):
    held = reco_run_with_heap("held", "2026-10-18")
    without = tmp_path / "without"
    shutil.copytree(held, without)
    (without / "baseline_heap.json").unlink()
    info = json.loads((without / "run_info.json").read_text())
    info.update(date="2026-10-19", k4h_release="key4hep-2026-10-19", k4h_release_date="2026-10-19")
    (without / "run_info.json").write_text(json.dumps(info))
    runs = (str(held), str(without))
    summary, heap = build_component_summary_trend(runs), build_heap_trend(runs)

    lumical = component_history(summary, heap, "baseline", "LumiCalReco")
    assert lumical["heap_status"].tolist() == [HELD, NO_MEMORY_PASS]
    assert lumical["typical_mb"].iloc[0] > 0 and np.isnan(lumical["typical_mb"].iloc[1])
    # Timed both nights, charged no heap the night it had a memory pass.
    refit = component_history(summary, heap, "baseline", "Refit")
    assert refit["status"].tolist() == [TIMED, TIMED]
    assert refit["heap_status"].tolist() == [HELD_NONE, NO_MEMORY_PASS]
    assert refit["typical_mb"].isna().all()


def test_the_summary_trend_lists_every_component_and_each_configuration_read():
    trend = build_component_summary_trend((str(RECO_RUN),))
    markers = trend[trend["component"].isna()]
    assert sorted(markers["label"]) == ["baseline", "variant_truth_tracking"]
    timings, _ = load_each_component_timing(RECO_RUN)
    baseline = trend[(trend["label"] == "baseline") & trend["component"].notna()]
    assert baseline["component"].tolist() == timings["baseline"].components.index.tolist()
    assert set(baseline["category"]) == {"algorithm", "service"}


def test_the_standing_ranks_a_component_among_the_others():
    values = pd.DataFrame({
        "time_ms": [100.0, 5.0, 50.0, NaN],
        "heap_mb": [0.0, 2.0, 30.0, 400.0],
    }, index=["Tracking", "Digi", "Vertex", "GeoSvc"])
    measures = [Measure("time_ms", "Time", "ms", lambda v: f"{v:g} ms", "total"),
                Measure("heap_mb", "Heap", "MB", lambda v: f"{v:g} MB", log=False)]
    fig = plot_standing(values, "Vertex", measures)
    texts = [a.text for a in fig.layout.annotations]
    # Ranked from the largest, its share where the values add up; a component
    # without a value, or with none of it, is not in the panel.
    assert "Rank 2 of 3 · 32% of total" in texts
    assert "Rank 2 of 3" in texts
    assert "50 ms" in texts and "30 MB" in texts
    time, heap = fig.data
    assert [row[0] for row in time.customdata] == ["Tracking", "Vertex", "Digi"]
    assert [row[0] for row in heap.customdata] == ["GeoSvc", "Vertex", "Digi"]
    # It alone is in colour.
    assert len(set(time.marker.color)) == 2 and time.marker.color[1] != time.marker.color[0]
    # A measure it has none of is left out; none at all draws nothing.
    assert len(plot_standing(values, "Tracking", measures).data) == 1
    assert plot_standing(values.assign(heap_mb=NaN), "GeoSvc", measures) is None


def test_the_neighbourhood_is_what_an_algorithm_reads_and_what_reads_it():
    from k4bench.analysis.dataflow import load_dataflow
    from k4bench.analysis.plots.dataflow import dataflow_view, layered_positions, neighbourhood

    flow = load_dataflow(RECO_RUN)["baseline"]
    view = dataflow_view(flow, expanded={a.impl for a in flow.algorithms})
    around = neighbourhood(view, "LumiCalReco")
    assert {n.id for n in around.nodes} == {"k4FWCore__Reader", "LumiCalReco", "k4FWCore__Writer"}
    assert set(around.edges) == {
        ("k4FWCore__Reader", "LumiCalReco"), ("LumiCalReco", "k4FWCore__Writer"),
    }
    # Its edges carry what they carry in the whole graph.
    assert around.edges[("k4FWCore__Reader", "LumiCalReco")] == view.edges[
        ("k4FWCore__Reader", "LumiCalReco")]
    # What it reads, it, what reads it, left to right.
    x = {node: position[0] for node, position in layered_positions(around).items()}
    assert x["k4FWCore__Reader"] < x["LumiCalReco"] < x["k4FWCore__Writer"]
