"""The Memory figures: every component's heap from initialization through
each event to finalization, and one component event by event."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from k4bench.analysis.heap import NO_COMPONENT  # noqa: E402
from k4bench.analysis.plots import PALETTE  # noqa: E402
from k4bench.analysis.plots.heap import (  # noqa: E402
    FINALIZED,
    INITIALIZED,
    NO_COMPONENT_LABEL,
    format_mb,
    plot_component_heap_events,
    plot_heap_timeline,
)

EVENTS = pd.Index([0, 1, 2, 3], name="event_number")

#: In MB at each event's end: the geometry holds still, the tracker grows,
#: a cache steps once; two small components and what no component allocated.
HELD = pd.DataFrame({
    NO_COMPONENT: [50.0, 50.0, 50.0, 50.0],
    "Geo": [300.0, 300.0, 300.0, 300.0],
    "Tracker": [10.0, 20.0, 30.0, 40.0],
    "Cache": [5.0, 5.0, 25.0, 25.0],
    "Tiny": [1.0, 1.0, 1.0, 1.0],
    "Small": [2.0, 2.0, 2.0, 3.0],
}, index=EVENTS)
BEFORE = pd.Series({NO_COMPONENT: 50.0, "Geo": 300.0, "Tracker": 0.0, "Cache": 0.0, "Tiny": 1.0, "Small": 2.0})
AFTER = pd.Series({NO_COMPONENT: 40.0, "Geo": 0.0, "Tracker": 40.0, "Cache": 0.0, "Tiny": 0.0, "Small": 0.0})
NAMED = ["Tracker", "Geo", "Cache"]


def _named(fig) -> list[str]:
    return [trace.name for trace in fig.data]


def test_held_stacks_no_component_then_the_rest_then_the_named_most_stable_first():
    fig = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED)
    stacked = [t for t in fig.data if t.stackgroup == "heap"]
    assert [t.name for t in stacked] == [NO_COMPONENT_LABEL, "Other (2)", "Geo", "Cache", "Tracker"]
    # Initialization, the four events and finalization, in that order.
    assert list(stacked[0].x) == [INITIALIZED, "0", "1", "2", "3", FINALIZED]
    other = stacked[1]
    assert np.asarray(other.y).tolist() == [3.0, 3.0, 3.0, 3.0, 4.0, 0.0]


def test_held_carries_the_whole_heap_in_the_hover_and_the_rss_as_a_line():
    rss = pd.Series([380.0, 390.0, 400.0, 420.0], index=EVENTS)
    fig = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED, rss=rss)
    total = next(t for t in fig.data if t.stackgroup is None and t.name is None)
    assert np.asarray(total.y).tolist() == [353.0, 368.0, 378.0, 408.0, 419.0, 80.0]
    line = fig.data[-1]
    assert line.name == "Anonymous RSS (timed run)" and line.line.dash == "dash"
    assert list(line.x) == ["0", "1", "2", "3"]


def test_a_heap_reaching_two_gigabytes_is_drawn_in_gigabytes():
    fig = plot_heap_timeline(HELD * 10, BEFORE * 10, AFTER * 10, NAMED)
    assert fig.layout.yaxis.title.text == "Heap held (GB)"
    geo = next(t for t in fig.data if t.name == "Geo")
    assert np.asarray(geo.y)[1] == pytest.approx(3000 / 1024)


def test_change_draws_each_named_series_beyond_its_first_event():
    fig = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED, change=True)
    assert _named(fig) == [*NAMED, "Other (2)", NO_COMPONENT_LABEL]
    tracker = fig.data[0]
    assert list(tracker.x) == ["0", "1", "2", "3"]
    assert np.asarray(tracker.y).tolist() == [0.0, 10.0, 20.0, 30.0]
    assert fig.layout.yaxis.title.text == "Change since the first event (MB)"


def test_change_shows_growth_in_a_series_not_named():
    # Only the geometry is named; the tracker's and the cache's growth is in Other.
    fig = plot_heap_timeline(HELD, BEFORE, AFTER, ["Geo"], change=True)
    lines = {t.name: np.asarray(t.y).tolist() for t in fig.data}
    assert lines["Geo"] == [0.0, 0.0, 0.0, 0.0]
    assert lines["Other (4)"] == [0.0, 10.0, 40.0, 51.0]
    assert lines[NO_COMPONENT_LABEL] == [0.0, 0.0, 0.0, 0.0]
    # Together the lines are the whole heap's change.
    total = HELD.sum(axis=1) - HELD.sum(axis=1).iloc[0]
    assert np.sum([lines[name] for name in lines], axis=0).tolist() == total.tolist()


#: The cache's 20 MB step at event 2.
STEPS = pd.DataFrame({"series": ["Cache"], "event_number": [2], "step_mb": [20.0]})


def test_held_marks_each_step_on_the_top_of_the_stack():
    colors = {"Tracker": "#123456", "Geo": "#654321", "Cache": "#abcdef"}
    fig = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED, colors=colors, steps=STEPS)
    (marks,) = [t for t in fig.data if t.name == "Step"]
    assert list(marks.x) == ["2"] and np.asarray(marks.y).tolist() == [408.0]
    assert list(marks.marker.color) == ["#abcdef"]
    assert list(marks.text) == ["Cache +20 MB"]


def test_held_marks_the_steps_of_one_event_once_naming_each():
    colors = {"Tracker": "#123456", "Geo": "#654321", "Cache": "#abcdef"}
    steps = pd.DataFrame({
        "series": ["Cache", "Tracker"], "event_number": [2, 2], "step_mb": [20.0, 30.0],
    })
    fig = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED, colors=colors, steps=steps)
    (marks,) = [t for t in fig.data if t.name == "Step"]
    assert list(marks.x) == ["2"] and np.asarray(marks.y).tolist() == [408.0]
    # In the largest step's colour, every step in the hover.
    assert list(marks.marker.color) == ["#123456"]
    assert list(marks.text) == ["Tracker +30 MB, Cache +20 MB"]


def test_change_marks_each_step_on_its_series_line_and_not_one_not_named():
    fig = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED, change=True, steps=STEPS)
    (marks,) = [t for t in fig.data if t.name == "Step"]
    assert np.asarray(marks.y).tolist() == [20.0]
    unnamed = plot_heap_timeline(HELD, BEFORE, AFTER, ["Geo"], change=True, steps=STEPS)
    assert "Step" not in _named(unnamed)
    assert "Step" not in _named(plot_heap_timeline(HELD, BEFORE, AFTER, NAMED, steps=STEPS.iloc[:0]))


def test_colours_follow_the_mapping_given():
    colors = {"Tracker": "#123456", "Geo": "#654321", "Cache": "#abcdef"}
    held = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED, colors=colors)
    change = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED, colors=colors, change=True)
    assert next(t for t in held.data if t.name == "Tracker").fillcolor == "#123456"
    assert next(t for t in change.data if t.name == "Tracker").line.color == "#123456"


def test_bands_are_opaque_and_set_apart_by_the_surface():
    fig = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED)
    stacked = [t for t in fig.data if t.stackgroup == "heap"]
    assert all(t.line.color == "#ffffff" for t in stacked)
    assert all(t.fillcolor.startswith("#") for t in stacked)
    # Unmapped series take the default palette in order.
    geo = next(t for t in stacked if t.name == "Geo")
    assert geo.fillcolor == PALETTE[1]


def test_a_band_wide_enough_carries_its_name_in_ink_that_contrasts():
    colors = {"Tracker": "#2a78d6", "Geo": "#eda100", "Cache": "#e34948"}
    fig = plot_heap_timeline(HELD, BEFORE, AFTER, NAMED, colors=colors)
    labels = {a.text: a for a in fig.layout.annotations if a.text != "warm-up"}
    # At event 3 the stack is 50, 4 (Other), 300, 25, 40 of 419 MB: the
    # thinnest, under 6% of it, are named by the legend alone.
    assert set(labels) == {"Geo", "No component", "Tracker"}
    assert labels["Geo"].font.color == "#0b0b0b"
    assert labels["Tracker"].font.color == "#ffffff"
    assert labels["Geo"].x == "3" and labels["Geo"].y == pytest.approx(54 + 150)


def test_nothing_to_draw_is_no_figure():
    assert plot_heap_timeline(HELD.iloc[:0], BEFORE, AFTER, NAMED) is None
    empty = pd.Series(dtype=float, index=pd.Index([], name="event_number"))
    assert plot_component_heap_events(empty, empty, empty) is None


def test_a_components_events_run_from_its_start_to_its_peak_with_its_end_marked():
    start = pd.Series([0.0, 10.0, 20.0], index=EVENTS[:3])
    peak = pd.Series([15.0, 24.0, 33.0], index=EVENTS[:3])
    end = pd.Series([10.0, 20.0, 30.0], index=EVENTS[:3])
    fig = plot_component_heap_events(start, peak, end, before_first=0.0, after_finalize=30.0)
    bars, marks = fig.data
    assert np.asarray(bars.base).tolist() == [0.0, 10.0, 20.0]
    assert np.asarray(bars.y).tolist() == [15.0, 14.0, 13.0]
    assert np.asarray(marks.y).tolist() == [10.0, 20.0, 30.0]
    assert fig.layout.title.text == f"{INITIALIZED}: 0 MB · {FINALIZED}: 30 MB"
    # The axis spans the events, not down to what it held before them.
    low, high = fig.layout.yaxis.range
    assert low < 0 and high > 33


def test_a_component_without_peaks_is_drawn_by_its_ends_alone():
    start = pd.Series([10.0, 20.0], index=EVENTS[:2])
    peak = pd.Series([np.nan, np.nan], index=EVENTS[:2])
    end = pd.Series([20.0, 30.0], index=EVENTS[:2])
    fig = plot_component_heap_events(start, peak, end)
    low, high = fig.layout.yaxis.range
    assert math.isfinite(low) and math.isfinite(high) and low < 10 and high > 30
    nothing = pd.Series([np.nan, np.nan], index=EVENTS[:2])
    assert plot_component_heap_events(nothing, nothing, nothing) is None


@pytest.mark.parametrize(("value", "text"), [
    (2048.0, "2 GB"),
    (280.04, "280 MB"),
    (0.5, "0.5 MB"),
    (0.000504, "0.516 kB"),
    (0.0, "0 MB"),
    (math.nan, "—"),
])
def test_format_mb_writes_a_size_as_it_is_read(value, text):
    assert format_mb(value) == text


def test_format_mb_signs_growth_on_request():
    assert format_mb(2.85, signed=True) == "+2.85 MB"
    assert format_mb(-0.019, signed=True) == "-19.5 kB"
