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
    history_rows,
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


def test_phase_history_draws_another_quantity_in_its_own_unit():
    rows = _history({"2026-10-11": 120.0, "2026-10-12": 4.0}, "ECALBarrel").rename(
        columns={"seconds": "compressed_kb"},
    )
    series = phase_component_series(rows, top_n=2, value="compressed_kb")
    fig = plot_phase_history(
        series, value="compressed_kb", unit="kB", y_title="Event size (kB)", whole="event",
    )
    (bar,) = (t for t in fig.data if t.type == "bar")
    assert list(bar.y) == pytest.approx([120.0, 4.0])
    assert " kB · " in bar.hovertemplate
    totals = next(t for t in fig.data if t.type == "scatter")
    assert list(totals.text) == ["120 kB", "4 kB"]
    assert fig.layout.yaxis.title.text == "Event size (kB)"
    relative = plot_phase_history(series, value="compressed_kb", relative=True, whole="event")
    assert relative.layout.yaxis.title.text == "Share of the event (%)"


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


def test_a_run_no_component_was_charged_in_is_all_unattributed():
    rows = _history({"2026-10-11": 2.0}, "GeoSvc")
    durations = _history({"2026-10-11": 3.0, "2026-10-12": 5.0}, None)
    series = phase_component_series(rows, top_n=6, durations=durations)
    by_run = series.set_index(["run_id", "label"])["seconds"].to_dict()
    assert by_run == pytest.approx({
        ("2026-10-11", "GeoSvc"): 2.0, ("2026-10-11", UNATTRIBUTED): 1.0,
        ("2026-10-12", UNATTRIBUTED): 5.0,
    })
    totals = next(t for t in plot_phase_history(series).data if t.type == "scatter")
    assert list(totals.text) == ["3 s", "5 s"]


def test_a_rerun_no_component_was_charged_in_is_still_the_tags_latest():
    tag = pd.Timestamp("2026-10-11")
    rows = _history({"2026-10-11": 2.0}, "GeoSvc").assign(x_date=tag)
    durations = _history({"2026-10-11": 3.0, "2026-10-12": 5.0}, None).assign(x_date=tag)
    series = phase_component_series(rows, top_n=6, durations=durations)
    assert series[["run_id", "label", "seconds"]].to_dict("records") == [
        {"run_id": "2026-10-12", "label": UNATTRIBUTED, "seconds": 5.0},
    ]


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


def _run(release: str, run_date: str, seconds: dict[str, float]) -> pd.DataFrame:
    """One run of *release*, measured on *run_date*."""
    return pd.DataFrame({
        "run_id": run_date, "x_date": pd.Timestamp(release),
        "run_date": pd.Timestamp(run_date), "k4h_release": f"key4hep-{release}",
        "phase": "execute", "component": list(seconds), "seconds": list(seconds.values()),
    })


#: The 10-11 release measured on 10-11 and re-run on 10-12, the night the
#: 10-12 release was first measured too.
REPEATS = pd.concat([
    _run("2026-10-11", "2026-10-11", {"A": 1.0, "B": 2.0}),
    _run("2026-10-11", "2026-10-12", {"A": 1.5, "B": 2.5}),
    _run("2026-10-12", "2026-10-12", {"A": 4.0, "B": 5.0}),
], ignore_index=True)


def _runs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted(set(zip(frame["k4h_release"], frame["run_id"])))


def test_history_rows_keep_each_tags_latest_run_or_every_run():
    assert _runs(history_rows(REPEATS)) == [
        ("key4hep-2026-10-11", "2026-10-12"), ("key4hep-2026-10-12", "2026-10-12"),
    ]
    assert _runs(history_rows(REPEATS, all_runs=True)) == _runs(REPEATS)


def test_latest_per_release_draws_one_column_per_tag():
    series = phase_component_series(REPEATS, top_n=6)
    totals = next(t for t in plot_phase_history(series).data if t.type == "scatter")
    assert list(totals.y) == [4.0, 9.0]
    assert plot_phase_history(series).layout.xaxis.type == "date"


def test_all_runs_draws_every_repeat_as_its_own_column():
    series = phase_component_series(REPEATS, top_n=6, all_runs=True)
    assert _runs(series) == _runs(REPEATS)
    fig = plot_phase_history(series, all_runs=True)
    totals = next(t for t in fig.data if t.type == "scatter")
    # Two runs of one night, of different releases, are two columns, never
    # summed; ordered by the night measured, the older release first.
    assert list(totals.y) == [3.0, 4.0, 9.0]
    assert list(totals.x) == [
        "2026-10-11 · Key4hep 2026-10-11",
        "2026-10-12 · Key4hep 2026-10-11",
        "2026-10-12 · Key4hep 2026-10-12",
    ]
    assert fig.layout.xaxis.type == "category"
    assert list(fig.layout.xaxis.ticktext) == ["2026-10-11", "2026-10-12", "2026-10-12"]
    assert fig.layout.xaxis.title.text == "Night measured"


def test_all_runs_shares_are_each_runs_own():
    series = phase_component_series(REPEATS, top_n=6, all_runs=True)
    shares = series.groupby(["k4h_release", "run_id"])["share_pct"].sum()
    assert shares.tolist() == pytest.approx([100.0, 100.0, 100.0])


def test_a_phase_timed_as_a_whole_is_one_series():
    series = phase_component_series(_history({"2026-10-11": 5.9, "2026-10-12": 1.1}, "Configure"), 6)
    assert series["label"].unique().tolist() == ["Configure"]
    assert series["seconds"].tolist() == [5.9, 1.1]


def test_series_name_the_components_always_named_ahead_of_the_largest():
    series = phase_component_series(INIT_HISTORY, top_n=1, always=["Tiny", "Absent"])
    assert list(dict.fromkeys(series["label"])) == ["Tiny", "MyAIDAProcessor", "Other (2)"]
    # One always named that is among the largest takes no second place.
    series = phase_component_series(INIT_HISTORY, top_n=2, always=["GeoSvc"])
    assert list(dict.fromkeys(series["label"])) == ["GeoSvc", "MyAIDAProcessor", "Other (2)"]


def test_series_keep_a_component_apart_after_other_whatever_its_size():
    rows = pd.concat([
        INIT_HISTORY, _history({"2026-10-11": 99.0, "2026-10-12": 99.0}, "(no component)"),
    ], ignore_index=True)
    series = phase_component_series(rows, top_n=1, apart={"(no component)": "No component"})
    assert list(dict.fromkeys(series["label"])) == ["MyAIDAProcessor", "Other (3)", "No component"]
    # Its share is of everything the run's series hold.
    shares = series.groupby("run_id")["share_pct"].sum()
    assert shares.to_list() == pytest.approx([100, 100])


def test_series_sum_any_value_column():
    rows = INIT_HISTORY.rename(columns={"seconds": "held_mb"})
    series = phase_component_series(rows, top_n=2, value="held_mb")
    assert list(series.columns) == [
        "run_id", "run_date", "x_date", "k4h_release", "label", "held_mb", "share_pct",
    ]
    other = series[series["label"] == "Other (2)"].set_index("run_id")["held_mb"]
    assert other.to_dict() == pytest.approx({"2026-10-11": 0.51, "2026-10-12": 0.55})


def test_a_series_without_a_value_in_a_run_is_unknown_there_not_zero():
    # The second run recorded no value for any component: a gap, not a drop to 0.
    rows = pd.concat([
        _history({"2026-10-11": 1.5, "2026-10-12": None}, "GeoSvc"),
        _history({"2026-10-11": 0.5, "2026-10-12": None}, "IOSvc"),
        _history({"2026-10-11": 0.01, "2026-10-12": None}, "Tiny"),
    ], ignore_index=True).rename(columns={"seconds": "held_mb"})
    series = phase_component_series(rows, top_n=1, value="held_mb")
    by_run = series.set_index(["label", "run_id"])["held_mb"]
    assert by_run[("GeoSvc", "2026-10-11")] == pytest.approx(1.5)
    assert by_run[("Other (2)", "2026-10-11")] == pytest.approx(0.51)
    assert by_run.xs("2026-10-12", level="run_id").isna().all()


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
