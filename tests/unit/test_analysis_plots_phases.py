"""The Breakdown figures: a phase's time by component, in a run and across tags."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("plotly")

from k4bench.analysis.plots import (  # noqa: E402
    plot_event_composition,
    plot_phase_component_pie,
    plot_phase_history,
)
from k4bench.analysis.plots import PALETTE  # noqa: E402
from k4bench.analysis.plots.phases import (  # noqa: E402
    UNATTRIBUTED,
    phase_component_series,
    top_components,
)

INIT = pd.Series({
    "GeoSvc": 1.474, "MyConformalTracking": 0.853, "MyAIDAProcessor": 0.586,
    "IOSvc": 0.545, "Reader": 0.046, "Writer": 0.002, "Idle": 0.0, "Absent": None,
})


def test_the_donut_names_the_longest_components_and_folds_the_rest():
    pie = plot_phase_component_pie(INIT, top_n=3).data[0]
    assert list(pie.labels) == ["GeoSvc", "MyConformalTracking", "MyAIDAProcessor", "Other (3)"]
    assert pie.values[-1] == pytest.approx(0.545 + 0.046 + 0.002)
    assert pie.text[0] == "42%"
    # The phase's total in the middle.
    assert plot_phase_component_pie(INIT).layout.annotations[0].text == "<b>3.51 s</b>"


def test_the_donut_keeps_a_components_given_colour_and_greys_other():
    pie = plot_phase_component_pie(INIT, top_n=2, colors={"MyConformalTracking": "#123456"}).data[0]
    assert pie.marker.colors[1] == "#123456"
    assert pie.marker.colors[-1] != pie.marker.colors[0]
    assert top_components(INIT, 2) == ["GeoSvc", "MyConformalTracking"]


def test_the_donut_gives_the_phase_time_no_component_was_charged_for():
    pie = plot_phase_component_pie(INIT, top_n=3, duration=5.0)
    assert pie.data[0].labels[-1] == UNATTRIBUTED
    assert pie.data[0].values[-1] == pytest.approx(5.0 - 3.506)
    # The middle is the phase's duration, as its card writes it.
    assert pie.layout.annotations[0].text == "<b>5.00 s</b>"


def test_the_donut_marks_component_time_beyond_the_phase():
    # Components that ran in parallel are charged more than the phase lasted.
    pie = plot_phase_component_pie(INIT, top_n=3, duration=2.0)
    assert UNATTRIBUTED not in pie.data[0].labels
    assert pie.layout.annotations[0].text == "<b>3.51 s</b><br>component time"


def test_small_slices_carry_no_percentage():
    pie = plot_phase_component_pie(pd.Series({"Big": 1.0, "Tiny": 0.001})).data[0]
    assert list(pie.text) == ["100%", ""]


def _history(seconds_by_tag: dict[str, float], component: str) -> pd.DataFrame:
    tags = list(seconds_by_tag)
    return pd.DataFrame({
        "run_id": tags, "x_date": pd.to_datetime(tags), "run_date": pd.to_datetime(tags),
        "k4h_release": [f"key4hep-{t}" for t in tags], "phase": "initialize",
        "component": component, "seconds": list(seconds_by_tag.values()),
    })


INIT_HISTORY = pd.concat([
    _history({"2026-10-11": 1.5, "2026-10-12": 1.47}, "GeoSvc"),
    _history({"2026-10-11": 22.6, "2026-10-12": 0.59}, "MyAIDAProcessor"),
    _history({"2026-10-11": 0.5, "2026-10-12": 0.55}, "IOSvc"),
    _history({"2026-10-11": 0.01}, "Tiny"),
], ignore_index=True)


def test_series_are_the_longest_components_and_the_rest_as_other():
    series = phase_component_series(INIT_HISTORY, top_n=2)
    # Ranked by the median over the window, which two tags make the mean.
    assert list(dict.fromkeys(series["label"])) == ["MyAIDAProcessor", "GeoSvc", "Other (2)"]
    other = series[series["label"] == "Other (2)"].set_index("run_id")["seconds"]
    assert other.to_dict() == pytest.approx({"2026-10-11": 0.51, "2026-10-12": 0.55})
    # Each series' share of its run's phase.
    shares = series.groupby("run_id")["share_pct"].sum()
    assert shares.to_list() == pytest.approx([100, 100])


def test_phase_history_stacks_the_series_per_tag_on_a_date_axis():
    fig = plot_phase_history(phase_component_series(INIT_HISTORY, top_n=2))
    bars = [t for t in fig.data if t.type == "bar"]
    assert [b.name for b in bars] == ["MyAIDAProcessor", "GeoSvc", "Other (2)"]
    assert fig.layout.barmode == "stack"
    assert fig.layout.xaxis.type == "date"
    assert list(fig.layout.xaxis.ticktext) == ["2026-10-11", "2026-10-12"]
    # Each column carries its total: the phase's duration.
    totals = next(t for t in fig.data if t.type == "scatter")
    assert list(totals.text) == ["24.6 s", "2.61 s"]


def test_phase_history_keeps_the_series_order_whatever_the_runs_dates():
    # The longest component only ran in the later run: ordered by date it would
    # come last, and take another series' colour.
    rows = pd.concat([
        _history({"2026-10-11": 22.6, "2026-10-12": 0.59}, "MyAIDAProcessor"),
        _history({"2026-10-12": 30.0}, "NewSvc"),
        _history({"2026-10-11": 0.5, "2026-10-12": 0.55}, "IOSvc"),
    ], ignore_index=True)
    bars = [t for t in plot_phase_history(phase_component_series(rows, top_n=2)).data
            if t.type == "bar"]
    assert [b.name for b in bars] == ["NewSvc", "MyAIDAProcessor", "Other (1)"]
    assert [b.marker.color for b in bars[:2]] == PALETTE[:2]


def test_phase_history_as_shares_fills_every_column():
    series = phase_component_series(INIT_HISTORY, top_n=2)
    bars = [t for t in plot_phase_history(series, relative=True).data if t.type == "bar"]
    assert [sum(col) for col in zip(*(b.y for b in bars))] == pytest.approx([100, 100])
    # No totals over columns that are all the same height.
    assert all(t.type == "bar" for t in plot_phase_history(series, relative=True).data)


def test_series_carry_the_phase_time_no_component_was_charged_for():
    durations = _history({"2026-10-11": 30.0, "2026-10-12": 2.6}, None)
    series = phase_component_series(INIT_HISTORY, top_n=2, durations=durations)
    assert list(dict.fromkeys(series["label"])) == [
        "MyAIDAProcessor", "GeoSvc", "Other (2)", UNATTRIBUTED,
    ]
    # Only the run whose phase outlasted its components has an unattributed part.
    gap = series[series["label"] == UNATTRIBUTED].set_index("run_id")["seconds"]
    assert gap.to_dict() == pytest.approx({"2026-10-11": 30.0 - 24.61})
    # Shares are of the phase's duration.
    aida = series[series["label"] == "MyAIDAProcessor"].set_index("run_id")["share_pct"]
    assert aida["2026-10-11"] == pytest.approx(22.6 / 30.0 * 100)
    totals = next(t for t in plot_phase_history(series).data if t.type == "scatter")
    assert list(totals.text) == ["30 s", "2.61 s"]


def test_a_tag_run_twice_shows_only_its_latest_run():
    def run(run_date, seconds):
        return pd.DataFrame({
            "run_id": run_date, "x_date": pd.Timestamp("2026-10-11"),
            "run_date": pd.Timestamp(run_date), "k4h_release": "key4hep-2026-10-11",
            "phase": "execute", "component": list(seconds), "seconds": list(seconds.values()),
        })

    # The rerun no longer has A, which the first run spent 8 s in.
    rows = pd.concat([
        run("2026-10-11", {"A": 8.0, "B": 1.0}), run("2026-10-12", {"B": 3.0}),
    ], ignore_index=True)
    series = phase_component_series(rows, top_n=6)
    assert series[["run_id", "label", "seconds"]].to_dict("records") == [
        {"run_id": "2026-10-12", "label": "B", "seconds": 3.0},
    ]
    totals = next(t for t in plot_phase_history(series).data if t.type == "scatter")
    assert list(totals.y) == [3.0]


def test_a_phase_timed_as_a_whole_is_one_series():
    series = phase_component_series(_history({"2026-10-11": 5.9, "2026-10-12": 1.1}, "Configure"), 6)
    assert series["label"].unique().tolist() == ["Configure"]
    assert series["seconds"].tolist() == [5.9, 1.1]


EVENTS = pd.DataFrame(
    {"Track": [9.0, 2.0, 4.0], "Vertex": [1.0, 3.0, None], "Tiny": [0.1, 0.1, 0.1]},
    index=pd.Index([0, 1, 2], name="event_number"),
)


def test_event_composition_stacks_each_event_and_marks_the_warm_up():
    fig = plot_event_composition(EVENTS, ["Track", "Vertex"], colors={"Track": "#123456"})
    assert [t.name for t in fig.data] == ["Track", "Vertex", "Other (1)"]
    assert list(fig.data[0].x) == ["0", "1", "2"]
    # The warm-up is paler.
    assert fig.data[0].marker.opacity[0] < fig.data[0].marker.opacity[1]
    # An algorithm that did not run in an event spent nothing there.
    assert list(fig.data[1].y) == [1.0, 3.0, 0.0]
    assert fig.data[0].marker.color == "#123456"
    assert fig.layout.barmode == "stack"


def test_event_composition_can_put_the_longest_events_first():
    events = EVENTS.assign(Vertex=[1.0, 9.0, None])
    fig = plot_event_composition(events, ["Track"], sort=True)
    # Event 1 (11.1 s), the warm-up (10.1 s), event 2 (4.1 s).
    assert list(fig.data[0].x) == ["1", "0", "2"]
