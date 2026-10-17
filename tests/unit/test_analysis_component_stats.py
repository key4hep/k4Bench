"""Unit tests for :mod:`k4bench.analysis.component_stats` — per-component
statistics of a k4run job's event loop."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from k4bench.analysis.component_stats import (
    component_summary,
    contributions,
    event_totals,
    work_efficiency,
)
from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.loader import load_component_timing

NIGHTS = Path(__file__).resolve().parents[1] / "fixtures" / "reco_nights"

#: A sequencer ``Top`` running ``Track`` and ``Vertex``, and a service ``Geo``
#: that only initializes. Event 0 is the warm-up; ``Vertex`` is filtered out of
#: event 3.
NAMES = ["Geo", "Top", "Track", "Vertex"]


def job(execute: dict[str, list[list]], events=(0, 1, 2, 3), **overrides) -> ComponentTiming:
    """A job over *events* whose ``execute`` rows hold one value per :data:`NAMES`."""
    zeros = [0.0] * len(NAMES)
    raw = {
        "schema_version": 1,
        "phases": {"execute": {"begin_epoch_ns": 0, "end_epoch_ns": 1}},
        "components": [
            {"name": "Geo", "category": "service", "parent": None},
            {"name": "Top", "category": "algorithm", "parent": None},
            {"name": "Track", "category": "algorithm", "parent": 1},
            {"name": "Vertex", "category": "algorithm", "parent": 1},
        ],
        "lifecycle": {"initialize": {
            "wall_s": [2.0, 0.0, 0.5, 0.1], "cpu_s": zeros, "peak_rss_increase_mb": zeros,
        }},
        "event_numbers": list(events),
        "execute": {
            "cpu_s": [[None if v is None else 0.0 for v in row] for row in execute["wall_s"]],
            "peak_rss_increase_mb": [
                [None if v is None else 0.0 for v in row] for row in execute["wall_s"]
            ],
            **execute,
        },
    }
    raw.update(overrides)
    return ComponentTiming.from_json(raw)


WALL = [
    # Geo, Top,  Track, Vertex
    [None, 0.01, 9.0, 9.0],   # warm-up, never counted
    [None, 0.01, 2.0, 1.0],
    [None, 0.01, 4.0, 3.0],
    [None, 0.01, 3.0, None],  # Vertex filtered out
]


def test_the_warm_up_event_is_left_out():
    summary = component_summary(job({"wall_s": WALL}))
    assert summary.loc["Track", "n_events"] == 3
    assert summary.loc["Track", "mean"] == pytest.approx(3.0)
    assert summary.loc["Track", "max"] == pytest.approx(4.0)


def test_components_that_never_ran_have_no_row():
    summary = component_summary(job({"wall_s": WALL}))
    assert summary.index.tolist() == ["Top", "Track", "Vertex"]


def test_a_filtered_component_adds_less_per_event_than_when_it_runs():
    vertex = component_summary(job({"wall_s": WALL})).loc["Vertex"]
    assert vertex["n_events"] == 2
    assert vertex["mean"] == pytest.approx(2.0)
    # 4 s over the 3 measured events, one of which it skipped.
    assert vertex["per_event"] == pytest.approx(4.0 / 3)


def test_shares_are_of_the_events_total():
    summary = component_summary(job({"wall_s": WALL}))
    total = event_totals(job({"wall_s": WALL})).mean()
    assert summary["share"].sum() == pytest.approx(1.0)
    assert summary.loc["Track", "share"] == pytest.approx(3.0 / total)


def test_spread_statistics():
    track = component_summary(job({"wall_s": WALL})).loc["Track"]
    assert track["std"] == pytest.approx(1.0)
    assert track["sem"] == pytest.approx(1.0 / math.sqrt(3))
    assert track["cv"] == pytest.approx(1.0 / 3)
    assert track["p90"] == pytest.approx(3.8)


def test_a_constant_cost_has_no_spread_and_one_event_none_to_estimate():
    top = component_summary(job({"wall_s": WALL})).loc["Top"]
    assert top["std"] == pytest.approx(0.0)
    alone = component_summary(job({"wall_s": WALL[:2]}, events=(0, 1))).loc["Track"]
    assert math.isnan(alone["std"]) and math.isnan(alone["sem"])


def test_a_call_without_a_value_leaves_the_total_unknown():
    instructions = [
        [None, 10, 900, 900],
        [None, 10, 200, 100],
        [None, 10, None, 300],  # Track ran, but its counters could not be read
        [None, 10, 300, None],
    ]
    timing = job({"wall_s": WALL, "instructions": instructions})
    summary = component_summary(timing, "instructions")
    assert math.isnan(summary.loc["Track", "per_event"])
    assert summary.loc["Track", "mean"] == pytest.approx(250)
    assert summary.loc["Track", "n_events"] == 3
    assert summary.loc["Track", "n_values"] == 2
    assert summary.loc["Track", "sem"] == pytest.approx(summary.loc["Track", "std"] / math.sqrt(2))
    assert summary["share"].isna().all()
    totals = event_totals(timing, "instructions")
    assert math.isnan(totals[2])
    assert totals[3] == pytest.approx(310)
    assert contributions(timing, "instructions").loc[3, "Vertex"] == 0.0


@pytest.mark.parametrize("metric", ["peak_heap_bytes", "peak_rss_increase_mb"])
def test_a_metric_that_does_not_add_up_has_no_share_and_no_total(metric):
    values = [[None if v is None else 1.0 for v in row] for row in WALL]
    timing = job({"wall_s": WALL, metric: values})
    assert component_summary(timing, metric)["share"].isna().all()
    with pytest.raises(ValueError, match="does not add up"):
        event_totals(timing, metric)


def test_an_unrecorded_metric_is_refused():
    with pytest.raises(ValueError, match="no instructions recorded"):
        component_summary(job({"wall_s": WALL}), "instructions")


def test_work_efficiency_needs_hardware_counters():
    assert work_efficiency(job({"wall_s": WALL})) is None
    counts = [[None if v is None else v * 1e9 for v in row] for row in WALL]
    cycles = [[None if v is None else v * 5e8 for v in row] for row in WALL]
    efficiency = work_efficiency(job({"wall_s": WALL, "instructions": counts, "cycles": cycles}))
    assert efficiency.ipc == pytest.approx(2.0)
    assert efficiency.instructions_per_event == pytest.approx(event_totals(
        job({"wall_s": WALL})).mean() * 1e9)


def test_sequencers_are_the_components_others_are_nested_in():
    assert job({"wall_s": WALL}).sequencers() == ["Top"]


def test_cld_reconstruction():
    """The 2026-10-17 CLD night, events 1–11: tracking dominates."""
    timing = load_component_timing(NIGHTS / "2026-10-17", ["baseline"])["baseline"]
    summary = component_summary(timing).sort_values("per_event", ascending=False)
    assert summary.index[:3].tolist() == ["MyConformalTracking", "Refit", "VertexFinder"]
    assert summary["share"].iloc[:2].sum() > 0.8
    assert summary["share"].sum() == pytest.approx(1.0)
    assert timing.sequencers() == ["k4FWCore__Sequencer", "k4FWCore__Algs"]
    assert work_efficiency(timing).ipc == pytest.approx(1.40, abs=0.01)
