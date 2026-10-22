"""The Reconstruction page's Components section, rendered: every algorithm on
one map and the one selected in charts, reached by a link, a search, a click
on the section's own charts or a click in another section, and back; its
numbers never drawn from another run or configuration than the one on
screen, and its history leaving a run it was missing from as a named gap."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"


def _page_app(dashboard_dir) -> None:
    """The Reconstruction page, whose charts can be clicked: a
    ``session_state["fake_click"]`` of ``(chart key, name)`` calls that chart's
    selection callback before the page is drawn, as Streamlit calls it after
    a click, with the click on the point carrying *name*."""
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st
    from reco import _inspect, page

    if (click := st.session_state.pop("fake_click", None)) is not None:
        chart, name = click
        callback, fig = st.session_state["callbacks"][chart]
        names = {str(row[0]) for trace in fig.data for row in (trace.customdata or ()) if row[0]}
        real_clicked = _inspect.clicked_name
        _inspect.clicked_name = lambda _event, _fig: name if name in names else None
        try:
            callback()
        finally:
            _inspect.clicked_name = real_clicked

    real_chart = st.plotly_chart

    def plotly_chart(fig, *args, **kwargs):
        if callable(kwargs.get("on_select")):
            chart = kwargs["key"].rsplit("_", 1)[0]
            st.session_state.setdefault("callbacks", {})[chart] = (kwargs["on_select"], fig)
            st.session_state.setdefault("chart_keys", []).append(kwargs["key"])
        return real_chart(fig, *args, **kwargs)

    st.plotly_chart = plotly_chart
    try:
        page.main()
    finally:
        st.plotly_chart = real_chart


def _page(monkeypatch, tmp_path, run_dir: Path, tab: str, params=None) -> AppTest:
    monkeypatch.delenv("K4BENCH_DATA_URL", raising=False)
    monkeypatch.setenv("K4BENCH_DATA_DIR", str(run_dir))
    monkeypatch.setenv("K4BENCH_CACHE_DIR", str(tmp_path / "cache"))
    at = AppTest.from_function(_page_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params["tab"] = tab
    for key, value in (params or {}).items():
        at.query_params[key] = value
    return _run(at)


def _run(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, at.exception
    return at


def _state(at: AppTest, key: str, default=None):
    try:
        return at.session_state[key]
    except KeyError:
        return default


def _shown(at: AppTest) -> tuple[str, str, str | None]:
    """The section, configuration and component on screen."""
    return (
        at.session_state["reco_active_section"],
        at.selectbox(key="reco_component_config").value,
        _state(at, "reco_component_name"),
    )


def _headings(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown]


def _tiles(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def _choose(at: AppTest, name: str) -> AppTest:
    """Select *name*'s row of the components table."""
    row = [c.removesuffix(" ⚠") for c in at.dataframe[0].value["Component"]].index(name)
    config = at.selectbox(key="reco_component_config").value
    nonce = _state(at, "reco_component_table_nonce", 0)
    at.session_state[f"reco_component_table_{os.environ['K4BENCH_DATA_DIR']}_{config}_{nonce}"] = {
        "selection": {"rows": [row], "columns": [], "cells": []},
    }
    return _run(at)


def _click(at: AppTest, chart: str, name: str) -> AppTest:
    at.session_state["fake_click"] = (chart, name)
    return _run(at)


def test_a_link_opens_the_component_it_names(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Components", {"component": "VertexFinder"})
    assert _shown(at) == ("Components", "baseline", "VertexFinder")
    assert "#### VertexFinder" in _headings(at)
    identity = next(c.value for c in at.caption if "configuration **baseline**" in c.value)
    assert "**LcfiplusProcessor**" in identity and "run by MarlinProcessorWrapper" in identity
    assert "run directory `reco_run`" in identity
    tiles = _tiles(at)
    assert list(tiles) == ["Time per event", "Initialize", "Finalize", "IPC", "Heap held"]
    assert tiles["Heap held"] == "—"
    # The fixture keeps no finalize phase: no time, not zero.
    assert tiles["Finalize"] == "—"
    assert any("ran in 11 of 11 events" in c.value for c in at.caption)
    # Where it stands, its time event by event and its data flow, in charts.
    assert {"##### Ranking", "##### Time per event", "##### Data flow"} <= set(_headings(at))
    # The time card's sparkline, drawn from zero as the table's are, and the charts.
    charts = {c.proto.id.split("-")[-1]: json.loads(c.proto.spec) for c in at.get("plotly_chart")}
    assert len(charts) == 4
    spark = next(spec for key, spec in charts.items() if "spark" in key)
    assert spark["layout"]["yaxis"]["range"][0] == 0
    # The time chart marks its mean, unlabelled.
    timing = next(spec for key, spec in charts.items() if "event_time" in key)
    (line,) = timing["layout"]["shapes"]
    # Its only text is the axis title: the mean carries no label.
    assert [a["text"] for a in timing["layout"].get("annotations", [])] == ["Time (s)"]
    # The first table is the components', its row selected.
    table = at.dataframe[0]
    selected = json.loads(table.proto.selection_default)["selection"]["rows"]
    assert table.value.iloc[selected]["Component"].tolist() == ["VertexFinder"]
    assert not at.expander


def test_the_section_opens_on_the_algorithm_that_adds_the_most(monkeypatch, tmp_path):
    from k4bench.analysis.component_detail import component_summary
    from k4bench.analysis.loader import load_each_component_timing

    timing = load_each_component_timing(RECO_RUN)[0]["baseline"]
    slowest = component_summary(timing).loc[timing.algorithms(), "seconds_per_event"].idxmax()
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Components")
    assert _shown(at) == ("Components", "baseline", slowest)


def test_every_algorithm_and_service_can_be_searched(monkeypatch, tmp_path):
    from k4bench.analysis.loader import load_each_component_timing

    at = _page(monkeypatch, tmp_path, RECO_RUN, "Components")
    timing = load_each_component_timing(RECO_RUN)[0]["baseline"]
    table = at.dataframe[0].value
    assert set(table["Component"]) == set(timing.components.index)
    # Each with its time and its heap event by event.
    assert {"Time per event", "Package"} <= set(table.columns)
    # A sub-millisecond algorithm, which the map names only on hover.
    _choose(at, "EventNumber")
    assert "#### EventNumber" in _headings(at)
    assert at.query_params["component"] in ("EventNumber", ["EventNumber"])
    # A service, run in no event: its phases, and no event tiles or chart.
    _choose(at, "GeoSvc")
    assert list(_tiles(at)) == ["Initialize", "Finalize", "Heap held"]
    assert any(c.value == "Not run in the event loop." for c in at.caption)
    assert any("reads and writes no event data" in c.value for c in at.caption)


def test_a_component_the_run_lacks_is_said_so_and_nothing_of_it_is_drawn(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Components", {"component": "NoSuchProcessor"})
    assert _shown(at)[2] is None
    (warning,) = at.warning
    assert "**NoSuchProcessor** is not a component of **baseline**" in warning.value
    assert not at.metric and not at.get("plotly_chart")
    _choose(at, "Refit")
    assert not at.warning and "#### Refit" in _headings(at)


def test_another_configuration_never_shows_the_component_it_lacks(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Components", {"component": "MyConformalTracking"})
    at.selectbox(key="reco_component_config").set_value("variant_truth_tracking")
    _run(at)
    # The truth-tracking job has no conformal tracking: said so, no values.
    assert _shown(at)[2] is None
    assert "**MyConformalTracking** is not a component of **variant_truth_tracking**" in at.warning[0].value
    assert not at.metric
    # Back on a configuration that has it, it is shown again.
    at.selectbox(key="reco_component_config").set_value("baseline")
    _run(at)
    assert _shown(at)[2] == "MyConformalTracking"
    assert not at.warning


@pytest.mark.parametrize("chart", ["reco_component_standing", "reco_component_flow"])
def test_a_click_in_the_sections_own_charts_shows_that_component(monkeypatch, tmp_path, chart):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Components", {"component": "VertexFinder"})
    # Both charts reach the Reader: as a column of where it stands, and as
    # what VertexFinder reads from.
    _click(at, chart, "k4FWCore__Reader")
    assert _shown(at) == ("Components", "baseline", "k4FWCore__Reader")
    with pytest.raises(KeyError):
        at.button(key="reco_component_back")


def _inspect_from_time(at: AppTest) -> None:
    _click(at, "reco_breakdown_events_chart", "VertexFinder")


def _inspect_from_memory(at: AppTest) -> None:
    at.button(key="reco_memory_inspect_VertexFinder").click()
    _run(at)


def _inspect_from_data_flow(at: AppTest) -> None:
    at.session_state["reco_dataflow_selected"] = "VertexFinder"
    _run(at)
    at.button(key="reco_dataflow_inspect").click()
    _run(at)


@pytest.mark.parametrize(("section", "inspect"), [
    ("Time", _inspect_from_time),
    ("Memory", _inspect_from_memory),
    ("Data Flow", _inspect_from_data_flow),
])
def test_every_section_opens_the_same_component_and_goes_back(
    monkeypatch, tmp_path, reco_run_with_heap, section, inspect,
):
    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), section)
    # Left on its history: inspecting from a section of the run on screen
    # shows that run.
    at.session_state["reco_component_view"] = "Historical Trends"
    inspect(at)
    assert _shown(at) == ("Components", "baseline", "VertexFinder")
    assert _state(at, "reco_component_view") == "Current Run"
    assert "#### VertexFinder" in _headings(at)
    assert at.button(key="reco_component_back").label == f"Back to {section}"
    at.button(key="reco_component_back").click()
    _run(at)
    assert at.session_state["reco_active_section"] == section


def test_a_configuration_without_component_costs_is_never_swapped_silently(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    run = reco_run_with_heap()
    (run / "variant_truth_tracking_heap.json").write_text(
        (run / "baseline_heap.json").read_text())
    (run / "baseline_components.json").unlink()
    at = _page(monkeypatch, tmp_path, run, "Memory")
    at.session_state["reco_memory_config"] = "baseline"
    _run(at)
    _inspect_from_memory(at)
    assert at.session_state["reco_active_section"] == "Components"
    (warning,) = at.warning
    assert warning.value.startswith(
        "**baseline** has no component costs in run directory `run`, so "
        "**variant_truth_tracking** is shown instead"
    )
    # It stays while the one shown does, through reruns and other sections.
    _run(at)
    assert len(at.warning) == 1
    at.session_state["reco_active_section"] = "Data Flow"
    _run(at)
    at.session_state["reco_active_section"] = "Components"
    _run(at)
    assert len(at.warning) == 1


def test_a_click_on_what_is_no_component_opens_nothing(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Time")
    _click(at, "reco_breakdown_events_chart", "Other (40)")
    assert at.session_state["reco_active_section"] == "Time"


def test_going_back_finds_the_section_as_it_was_left(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Time")
    at.selectbox(key="reco_breakdown_config").set_value("variant_truth_tracking")
    _run(at)
    at.button_group(key="reco_breakdown_event_order").set_value("Longest first")
    _run(at)
    _click(at, "reco_breakdown_events_chart", "MyTruthTrackFinder")
    assert _shown(at) == ("Components", "variant_truth_tracking", "MyTruthTrackFinder")
    at.button(key="reco_component_back").click()
    _run(at)
    assert at.button_group(key="reco_time_lens").value == "Phases and components"
    assert at.selectbox(key="reco_breakdown_config").value == "variant_truth_tracking"
    assert at.button_group(key="reco_breakdown_event_order").value == "Longest first"


def test_the_same_point_clicked_again_opens_it_again(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Time")
    first = at.session_state["chart_keys"][-1]
    _inspect_from_time(at)
    at.button(key="reco_component_back").click()
    _run(at)
    # The chart came back as a new widget, holding no selection.
    assert at.session_state["chart_keys"][-1] != first
    _inspect_from_time(at)
    assert _shown(at) == ("Components", "baseline", "VertexFinder")


def test_without_a_memory_pass_the_heap_says_so(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Components", {"component": "LumiCalReco"})
    assert _tiles(at)["Heap held"] == "—"
    assert any(c.value == "No memory pass for this configuration in this run." for c in at.caption)


def _without_hardware_counters(tmp_path) -> Path:
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    for path in run.glob("*_components.json"):
        raw = json.loads(path.read_text())
        for table in (raw["execute"], *raw["lifecycle"].values()):
            table.pop("instructions"), table.pop("cycles")
        raw["hardware_counters_error"] = "perf_event_open: Function not implemented"
        path.write_text(json.dumps(raw))
    return run


def test_without_hardware_counters_the_time_and_allocations_stay(monkeypatch, tmp_path):
    run = _without_hardware_counters(tmp_path)
    at = _page(monkeypatch, tmp_path, run, "Components", {"component": "VertexFinder"})
    tiles = _tiles(at)
    assert tiles["Time per event"] != "—" and tiles["IPC"] == "—"
    ipc = next(m for m in at.metric if m.label == "IPC")
    assert "perf_event_open: Function not implemented" in ipc.help
    # Where it stands still ranks its time and allocations.
    standing = next(json.loads(c.proto.spec) for c in at.get("plotly_chart")
                    if "reco_component_standing" in c.proto.id)
    titles = " ".join(a["text"] for a in standing["layout"]["annotations"])
    assert "<b>Event loop</b>" in titles and "<b>Allocations</b>" in titles


def test_its_data_flow_is_what_it_reads_and_what_reads_it(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Components", {"component": "LumiCalReco"})
    chart = next(
        json.loads(c.proto.spec)["data"] for c in at.get("plotly_chart")
        if "reco_component_flow" in c.proto.id
    )
    nodes = [row[0] for row in chart[-1]["customdata"]]
    assert set(nodes) == {"k4FWCore__Reader", "LumiCalReco", "k4FWCore__Writer"}
    # It is ringed in the red of the section's charts.
    from k4bench.analysis.plots.component import ACCENT

    rings = dict(zip(nodes, chart[-1]["marker"]["line"]["color"]))
    assert rings["LumiCalReco"] == ACCENT
    # Under the graph, what it reads and writes, compactly.
    reads = next(d.value for d in at.dataframe if "Written by" in d.value.columns)
    assert list(reads.columns) == ["Collection", "Written by", "Type"]
    assert reads.set_index("Collection").at["LumiCalCollection", "Written by"] == "k4FWCore__Reader"
    writes = next(d.value for d in at.dataframe if "Read by" in d.value.columns)
    assert "LumiCalClusters" in set(writes["Collection"])
    # A Gaudi handle has no type: no empty Type column.
    _choose(at, "k4FWCore__Reader")
    writes = next(d.value for d in at.dataframe if "Read by" in d.value.columns)
    assert list(writes.columns) == ["Collection", "Read by"]


def test_a_clicked_point_is_named_by_its_customdata_not_its_place_or_label():
    """A click is read back through the point's own customdata, whatever
    panel it is drawn in and wherever its rank puts it, and "Other" carries
    none."""
    from k4bench.analysis.component_detail import component_summary
    from k4bench.analysis.loader import load_each_component_timing
    from k4bench.analysis.plots import Measure, plot_event_composition, plot_standing
    from reco._inspect import clicked_name

    def click(curve, index):
        return SimpleNamespace(selection={"points": [{"curve_number": curve, "point_index": index}]})

    timing = load_each_component_timing(RECO_RUN)[0]["baseline"]
    summary = component_summary(timing)
    fig = plot_standing(summary, "Refit", [
        Measure("seconds_per_event", "Time", "s", str), Measure("initialize_s", "Init", "s", str),
    ])
    for curve, (column, trace) in enumerate(zip(("seconds_per_event", "initialize_s"), fig.data)):
        ranked = summary[column][summary[column] > 0].sort_values(ascending=False, kind="stable")
        for index, name in enumerate(ranked.index):
            assert clicked_name(click(curve, index), fig) == name
    assert clicked_name(SimpleNamespace(selection={"points": []}), fig) is None
    assert clicked_name(None, fig) is None
    assert clicked_name(click(99, 0), fig) is None

    bars = plot_event_composition(timing.execute["wall_s"], ["Refit", "VertexFinder"])
    assert [clicked_name(click(c, 3), bars) for c in range(len(bars.data))] == [
        "Refit", "VertexFinder", None,
    ]


def test_the_reco_page_section_key_is_the_one_inspect_sets():
    from reco._inspect import SECTION, SECTION_KEY
    from reco.page import RECO_PAGE

    assert SECTION_KEY == f"{RECO_PAGE.key_prefix}active_section"
    assert SECTION in RECO_PAGE.sections


def _history_app(dashboard_dir, run_dirs) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from config import Config
    from data import load_run
    from reco import component
    from section_dispatch import PageContext
    from sidebar import TrendWindow

    ctx = PageContext(
        config=Config(run_dirs[-1], "https://store.example", "/tmp", "https://dash.example"),
        selection=None, data_dir=run_dirs[-1], run_meta=None,
        run=load_run(run_dirs[-1], components=True),
        window=TrendWindow(None, download=lambda _artifacts: run_dirs),
    )
    component.render(ctx)


_NIGHTS = ("2026-10-18", "2026-10-19", "2026-10-20")


def _nights(reco_run_with_heap, *, drop_on: str, name: str = "LumiCalReco") -> list[Path]:
    """Three nights of the heap fixture; on *drop_on*, *name* is not in the
    baseline's job, and the night has no memory pass."""
    runs = [reco_run_with_heap(night, night) for night in _NIGHTS]
    run = runs[_NIGHTS.index(drop_on)]
    path = run / "baseline_components.json"
    raw = json.loads(path.read_text())
    i = next(i for i, c in enumerate(raw["components"]) if c["name"] == name)
    raw["components"] = [
        {**c, "parent": None if c["parent"] is None else c["parent"] - (c["parent"] > i)}
        for j, c in enumerate(raw["components"]) if j != i
    ]
    for table in raw["execute"].values():
        for row in table:
            del row[i]
    for phase in raw["lifecycle"].values():
        for values in phase.values():
            del values[i]
    path.write_text(json.dumps(raw))
    (run / "baseline_heap.json").unlink()
    return runs


def _traces(at: AppTest) -> list[dict]:
    return json.loads(at.get("plotly_chart")[0].proto.spec)["data"]


def _values(array) -> np.ndarray:
    import base64

    if isinstance(array, dict):
        return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"])
    return np.asarray([np.nan if v is None else v for v in array], dtype=float)


def test_the_history_leaves_a_night_without_the_component_as_a_named_gap(reco_run_with_heap):
    runs = _nights(reco_run_with_heap, drop_on="2026-10-19")
    at = AppTest.from_function(
        _history_app, args=(str(_DASHBOARD_DIR), tuple(map(str, runs))), default_timeout=60,
    )
    at.query_params["component"] = "LumiCalReco"
    _run(at)
    at.radio(key="reco_component_view").set_value("Historical Trends")
    _run(at)
    lines = _traces(at)
    assert {t["name"] for t in lines} == {"LumiCalReco"}
    values = _values(next(t for t in lines if t["yaxis"] == "y")["y"])
    assert np.isfinite(values[0]) and np.isnan(values[1]) and np.isfinite(values[2])
    assert any("not in the job in 1 run: 2026-10-19" in c.value for c in at.caption)
    # The memory pass's numbers have gaps of their own.
    at.segmented_control(key="reco_component_hist_group").set_value("Memory")
    _run(at)
    held = next(t for t in _traces(at) if t["yaxis"] == "y3")
    assert np.isnan(_values(held["y"])[1])
    assert any("no memory pass in 1 run: 2026-10-19" in c.value for c in at.caption)


def _history(runs: list[Path], component: str | None = None) -> AppTest:
    """The Components section over *runs*, the last on screen, opened on its
    history."""
    at = AppTest.from_function(
        _history_app, args=(str(_DASHBOARD_DIR), tuple(map(str, runs))), default_timeout=60,
    )
    if component is not None:
        at.query_params["component"] = component
    _run(at)
    at.radio(key="reco_component_view").set_value("Historical Trends")
    return _run(at)


def test_the_history_does_not_need_the_run_on_screen_to_have_components(reco_run_with_heap):
    runs = [reco_run_with_heap(night, night) for night in _NIGHTS]
    for path in runs[-1].glob("*_components.json"):
        path.unlink()
    at = AppTest.from_function(
        _history_app, args=(str(_DASHBOARD_DIR), tuple(map(str, runs))), default_timeout=60,
    )
    _run(at)
    assert any("No component costs in this run" in i.value for i in at.info)
    at = _history(runs, "LumiCalReco")
    assert not at.info and not at.warning
    assert at.selectbox(key="reco_component_hist_component_baseline_0").value == "LumiCalReco"
    values = _values(next(t for t in _traces(at) if t["yaxis"] == "y")["y"])
    assert np.isfinite(values[:2]).all()


def test_the_history_draws_a_component_the_run_on_screen_no_longer_has(reco_run_with_heap):
    runs = _nights(reco_run_with_heap, drop_on="2026-10-20")
    at = AppTest.from_function(
        _history_app, args=(str(_DASHBOARD_DIR), tuple(map(str, runs))), default_timeout=60,
    )
    at.query_params["component"] = "LumiCalReco"
    _run(at)
    assert any("**LumiCalReco** is not a component" in w.value for w in at.warning)
    at.radio(key="reco_component_view").set_value("Historical Trends")
    _run(at)
    assert not at.warning
    values = _values(next(t for t in _traces(at) if t["yaxis"] == "y")["y"])
    assert np.isfinite(values[:2]).all() and np.isnan(values[2])
    assert any("not in the job in 1 run: 2026-10-20" in c.value for c in at.caption)


def test_the_history_leaves_a_night_without_component_costs_as_a_named_gap(reco_run_with_heap):
    runs = [reco_run_with_heap(night, night) for night in _NIGHTS]
    for path in [*runs[1].glob("*_components.json"), *runs[1].glob("*_heap.json")]:
        path.unlink()
    at = _history(runs, "LumiCalReco")
    values = _values(next(t for t in _traces(at) if t["yaxis"] == "y")["y"])
    assert np.isfinite(values[0]) and np.isnan(values[1]) and np.isfinite(values[2])
    assert any("the configuration was not timed in 1 run: 2026-10-19" in c.value
               for c in at.caption)


def test_the_history_says_so_of_a_component_the_window_never_had(reco_run_with_heap):
    runs = [reco_run_with_heap(night, night) for night in _NIGHTS]
    at = _history(runs, "NoSuchAlgorithm")
    assert any("**NoSuchAlgorithm** is not a component of **baseline** in any run"
               in w.value for w in at.warning)
    assert not at.get("plotly_chart")
    at.selectbox(key="reco_component_hist_component_baseline_0").set_value("LumiCalReco")
    _run(at)
    assert not at.warning and _traces(at)


def test_a_component_memory_warns_about_carries_a_warning_sign(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), "Components")
    names = at.dataframe[0].value["Component"].tolist()
    assert {"LumiCalReco ⚠", "VertexFinder ⚠"} <= set(names)
    assert "GeoSvc" in names and "Refit" in names
    # The slowest per event first, then those run in no event.
    per_event = at.dataframe[0].value["Per event"]
    timed = per_event.dropna()
    assert list(timed) == sorted(timed, reverse=True)
    assert per_event.isna().sum() == 0 or per_event.iloc[len(timed):].isna().all()


def test_its_time_and_heap_event_by_event_share_one_layout(monkeypatch, tmp_path, reco_run_with_heap):
    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), "Components", {"component": "VertexFinder"})
    charts = {c.proto.id.split("-")[-1]: json.loads(c.proto.spec)["layout"] for c in at.get("plotly_chart")}
    time = next(layout for key, layout in charts.items() if "event_time" in key)
    heap = next(layout for key, layout in charts.items() if "heap_events" in key)
    for prop in ("height", "margin", "showlegend"):
        assert time[prop] == heap[prop], prop
    assert time["xaxis"]["range"] == heap["xaxis"]["range"]
    assert time["xaxis"]["type"] == heap["xaxis"]["type"] == "linear"
    assert time["xaxis"]["title"]["text"] == heap["xaxis"]["title"]["text"] == "Event number"
    # Plot areas start at the same place, whatever the width of the ticks.
    assert time["margin"]["l"] == heap["margin"]["l"] == 100
    # Each marks its mean, and draws its events as one series of columns.
    assert len(time["shapes"]) == len(heap["shapes"]) == 1
    specs = {c.proto.id.split("-")[-1]: json.loads(c.proto.spec)["data"] for c in at.get("plotly_chart")}
    heap_traces = next(data for key, data in specs.items() if "heap_events" in key)
    # Translucent columns, their tops solid caps: no markers on them.
    assert [t["type"] for t in heap_traces][0] == "bar"
    assert "rgba" in heap_traces[0]["marker"]["color"][1]
    caps = heap_traces[1:]
    assert caps and all(t["type"] == "scatter" and t["mode"] == "lines" for t in caps)
