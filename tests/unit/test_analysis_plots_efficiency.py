"""The Efficiency figure: each algorithm's time per event against its
instructions per cycle, sized by its allocations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from k4bench.analysis.efficiency import component_efficiency  # noqa: E402
from k4bench.analysis.plots.efficiency import plot_component_efficiency  # noqa: E402

EVENTS = pd.Index([0, 1, 2], name="event_number")


def _frame(columns: dict[str, list]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=EVENTS, dtype=float)


#: Track runs every event; Vertex skips event 2; Idle never runs; Blind ran
#: in event 1 without a count. Event 0 warms up, ten times slower.
EXECUTE = {
    "wall_s": _frame({"Track": [10.0, 1.0, 3.0], "Vertex": [5.0, 0.5, None],
                      "Idle": [None] * 3, "Blind": [1.0, 1.0, 1.0]}),
    "instructions": _frame({"Track": [9e9, 4e9, 8e9], "Vertex": [1e9, 3e8, None],
                            "Idle": [None] * 3, "Blind": [1e9, None, 1e9]}),
    "cycles": _frame({"Track": [9e9, 2e9, 4e9], "Vertex": [1e9, 6e8, None],
                      "Idle": [None] * 3, "Blind": [1e9, None, 1e9]}),
    "allocations": _frame({"Track": [500, 100, 300], "Vertex": [50, 10, None],
                           "Idle": [None] * 3, "Blind": [1, 1, 1]}),
}


def test_efficiency_is_per_event_over_the_events_each_algorithm_ran_in():
    table = component_efficiency(EXECUTE)
    assert table.index.tolist() == ["Track", "Vertex"]
    track, vertex = table.loc["Track"], table.loc["Vertex"]
    assert track["ms_per_event"] == pytest.approx(2000.0)
    assert track["ipc"] == pytest.approx(2.0)
    assert track["allocations_per_event"] == pytest.approx(200.0)
    assert track["instructions_per_event"] == pytest.approx(6e9)
    # Per second of its own time: 400 allocations in 4 s.
    assert track["allocations_per_s"] == pytest.approx(100.0)
    # Vertex ran in one event after the warm-up.
    assert vertex["ms_per_event"] == pytest.approx(500.0)
    assert vertex["ipc"] == pytest.approx(0.5)


def test_allocated_memory_is_in_binary_megabytes():
    allocated = _frame({"Track": [0.0, 1024**2, 3 * 1024**2], "Vertex": [0.0, 0.0, None],
                        "Idle": [None] * 3, "Blind": [0.0] * 3})
    table = component_efficiency({**EXECUTE, "allocated_bytes": allocated})
    # 4 MiB over two events, as everywhere else an MB is 1024 kB.
    assert table.loc["Track", "allocated_mb_per_event"] == pytest.approx(2.0)


def test_the_warm_up_event_counts_when_asked_to():
    table = component_efficiency(EXECUTE, warm_up=None)
    assert table.loc["Track", "ms_per_event"] == pytest.approx(14000.0 / 3)


def test_efficiency_needs_hardware_counters():
    assert component_efficiency({"wall_s": EXECUTE["wall_s"]}).empty
    without_allocations = {k: v for k, v in EXECUTE.items() if k != "allocations"}
    table = component_efficiency(without_allocations)
    assert table["allocations_per_event"].isna().all()
    assert plot_component_efficiency(table) is not None


def test_the_chart_puts_time_on_a_log_axis_and_names_the_longest():
    fig = plot_component_efficiency(component_efficiency(EXECUTE))
    (points,) = fig.data
    assert fig.layout.xaxis.type == "log"
    assert list(points.text) == ["Track", "Vertex"]
    assert list(points.y) == pytest.approx([2.0, 0.5])
    # The most allocations per event, the largest point.
    sizes = np.asarray(points.marker.size, dtype=float)
    assert sizes[0] > sizes[1]
    assert "200 allocations per event" in points.customdata[0][1]
    # Or the allocation rate: Vertex allocates 20 a second.
    (points,) = plot_component_efficiency(
        component_efficiency(EXECUTE), size="allocations_per_s",
    ).data
    assert "100 allocations per second" in points.customdata[0][1]
    assert plot_component_efficiency(component_efficiency({"wall_s": EXECUTE["wall_s"]})) is None
    assert not math.isnan(fig.layout.yaxis.range[1])


def test_points_coloured_by_group_are_one_trace_each_the_costliest_first():
    table = component_efficiency(EXECUTE)
    fig = plot_component_efficiency(
        table, groups=pd.Series({"Track": "tracking", "Vertex": "vertexing"}),
        palette=["#111111", "#222222"],
    )
    assert [t.name for t in fig.data] == ["tracking", "vertexing"]
    assert [t.marker.color for t in fig.data] == ["#111111", "#222222"]
    assert fig.layout.showlegend


def test_points_can_be_one_size():
    table = component_efficiency(EXECUTE)
    (points,) = plot_component_efficiency(table, size=None).data
    assert len(set(np.asarray(points.marker.size, dtype=float))) == 1
    # Without allocated bytes recorded, the points are all one size too.
    (points,) = plot_component_efficiency(table, size="allocated_mb_per_event").data
    assert len(set(np.asarray(points.marker.size, dtype=float))) == 1


def test_as_many_of_the_slowest_algorithms_are_named_as_asked():
    (points,) = plot_component_efficiency(component_efficiency(EXECUTE), labelled=1).data
    assert list(points.text) == ["Track", ""]
