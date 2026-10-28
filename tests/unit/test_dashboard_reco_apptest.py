"""The Reconstruction page's own sections, rendered: the Breakdown of the
10-17 nightly run (``reco_run``) by phase and component, and the Overview's
detector comparison drawn from runs rather than from nightly reports."""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))


RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"


def _local_env(monkeypatch, run_dir: Path, cache_dir: Path) -> None:
    monkeypatch.delenv("K4BENCH_DATA_URL", raising=False)
    monkeypatch.setenv("K4BENCH_DATA_DIR", str(run_dir))
    monkeypatch.setenv("K4BENCH_CACHE_DIR", str(cache_dir))


def _page_app(dashboard_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from reco import page

    page.main()


def _page(
    monkeypatch, tmp_path, run_dir: Path, tab: str | None = None, params=None,
) -> AppTest:
    _local_env(monkeypatch, run_dir, tmp_path / "cache")
    at = AppTest.from_function(_page_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    if tab:
        at.query_params["tab"] = tab
    for key, value in (params or {}).items():
        at.query_params[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _choose_component(at: AppTest, name: str) -> None:
    """Select *name*'s row of the Components table."""
    row = [c.removesuffix(" ⚠") for c in at.dataframe[0].value["Component"]].index(name)
    config = at.selectbox(key="reco_component_config").value
    try:
        nonce = at.session_state["reco_component_table_nonce"]
    except KeyError:
        nonce = 0
    at.session_state[f"reco_component_table_{os.environ['K4BENCH_DATA_DIR']}_{config}_{nonce}"] = {
        "selection": {"rows": [row], "columns": [], "cells": []},
    }
    at.run()
    assert not at.exception, at.exception


def _traces(at: AppTest) -> list[dict]:
    return json.loads(at.get("plotly_chart")[0].proto.spec)["data"]


def test_a_local_run_opens_on_the_time(monkeypatch, tmp_path):
    # The Overview compares detectors across the window, which a local run lacks.
    at = _page(monkeypatch, tmp_path, RECO_RUN)
    assert at.session_state["reco_active_section"] == "Time"
    assert at.button_group[0].options == [
        "Time", "Memory", "Components", "Data Flow", "Correlations", "Event Size",
        "Machine Info", "Logs",
    ]


@pytest.mark.parametrize(("tab", "section", "lens"), [
    ("Breakdown", "Time", ("reco_time_lens", "Phases and components")),
    ("event timing", "Time", ("reco_time_lens", "Per event")),
    ("Event Memory", "Memory", ("reco_memory_lens", "Per event")),
    ("Efficiency", "Components", None),
    ("Events", "Correlations", None),
])
def test_a_link_to_a_former_section_opens_where_it_now_is(monkeypatch, tmp_path, tab, section, lens):
    at = _page(monkeypatch, tmp_path, RECO_RUN, tab)
    assert at.session_state["reco_active_section"] == section
    if lens is not None:
        key, value = lens
        assert at.button_group(key=key).value == value


def test_a_link_to_the_former_efficiency_opens_its_configuration(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Efficiency", {
        "reco_efficiency_view": "Current Run",
        "reco_efficiency_config": "variant_truth_tracking",
        "reco_efficiency_hist_config": "baseline",
    })
    assert at.session_state["reco_active_section"] == "Components"
    assert at.selectbox(key="reco_component_config").value == "variant_truth_tracking"


@pytest.mark.parametrize(("params", "values"), [
    ({"reco_efficiency_view": "Historical Trends", "reco_efficiency_config": "baseline",
      "reco_efficiency_hist_config": "variant_truth_tracking"},
     {"reco_component_view": "Historical Trends", "reco_component_config": "variant_truth_tracking"}),
    ({"reco_efficiency_view": "Current Run", "reco_efficiency_config": "baseline",
      "reco_efficiency_hist_config": "variant_truth_tracking"},
     {"reco_component_view": "Current Run", "reco_component_config": "baseline"}),
    ({"reco_efficiency_config": "baseline"}, {"reco_component_config": "baseline"}),
    ({}, {}),
])
def test_the_former_efficiency_view_and_configuration_carry_over(params, values):
    from reco.page import RECO_PAGE

    _section, link = RECO_PAGE.aliases["Efficiency"]
    assert link(params) == values


def test_each_lens_is_in_the_url(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Time", {"reco_time_lens": "Per event"})
    assert at.button_group(key="reco_time_lens").value == "Per event"
    # Event Timing's own controls, with the algorithms on offer.
    assert at.multiselect(key="evt_timing_timings")


def _values(array) -> np.ndarray:
    """A chart spec's numeric array, which Plotly writes as typed binary."""
    if isinstance(array, dict):
        return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"])
    return np.asarray(array, dtype=float)


def _charts(at: AppTest) -> list[list[dict]]:
    return [json.loads(c.proto.spec)["data"] for c in at.get("plotly_chart")]


def test_the_breakdown_gives_each_phase_and_its_components(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Breakdown")
    assert at.selectbox(key="reco_breakdown_config").options == [
        "baseline", "variant_truth_tracking",
    ]
    # The fixture keeps the initialize and execute spans.
    assert [m.label for m in at.metric] == ["Configure", "Initialize", "Event loop"]
    assert [c.value for c in at.caption if c.value.endswith("of the job")] == [
        "0.2% of the job", "0.8% of the job", "99.0% of the job",
    ]
    # A donut per phase in job order (the fixture keeps no finalize phase),
    # then the event loop event by event.
    (init,), (loop,), events = _charts(at)
    assert init["labels"][0] == "GeoSvc"
    assert loop["labels"][0] == "MyConformalTracking"
    assert any("No component time recorded for finalization" in c.value for c in at.caption)
    # A component keeps one colour across the donuts and the event chart.
    colour = dict(zip(init["labels"], init["marker"]["colors"]))
    for name, c in zip(loop["labels"], loop["marker"]["colors"]):
        if name in colour and not name.startswith("Other"):
            assert colour[name] == c
    by_event = {t["name"]: t for t in events}
    loop_colour = dict(zip(loop["labels"], loop["marker"]["colors"]))
    assert by_event["MyConformalTracking"]["marker"]["color"] == loop_colour["MyConformalTracking"]
    # Every event of the event loop, the warm-up included.
    assert by_event["Refit"]["x"][0] == "0"
    assert len(by_event["Refit"]["x"]) == 12
    # Local mode has no trend window to draw history from: one view, no choice.
    with pytest.raises(KeyError):
        at.radio(key="reco_breakdown_view")
    # Every component's table is the Components section's.
    assert not at.dataframe


def test_a_run_without_a_phase_to_show_still_gives_its_components(monkeypatch, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    for path in run.glob("*_components.json"):
        raw = json.loads(path.read_text())
        path.write_text(json.dumps({**raw, "phases": {}}))
    at = _page(monkeypatch, tmp_path, run, "Breakdown")
    assert "No phase of **baseline** lasted 50 ms or more" in at.info[0].value
    assert not at.metric
    (init,), (loop,), _ = _charts(at)
    assert init["labels"][0] == "GeoSvc"


def _run_with_tracking_package(tmp_path) -> Path:
    """The fixture run with ConformalTracking's library in the stack's
    conformaltracking package, which the run's manifest names."""
    import gzip

    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    install = ("/cvmfs/sft-nightlies.cern.ch/lcg/nightlies/devkey-head/Sat/"
               "conformaltracking/HEAD/x86_64-el9-gcc16-opt")
    path = run / "baseline_components.json"
    raw = json.loads(path.read_text())
    for component in raw["components"]:
        if component["name"] == "MyConformalTracking":
            component["impl_library"] = f"{install}/lib/libConformalTracking.so"
    path.write_text(json.dumps(raw))
    info = json.loads((run / "run_info.json").read_text())
    info["k4h_packages"] = {"conformaltracking": {
        "commit": "38735aa", "version": "HEAD",
        "repo_url": "https://github.com/iLCSoft/ConformalTracking.git",
    }}
    (run / "run_info.json").write_text(json.dumps(info))
    (run / "LCG_externals.txt.gz").write_bytes(
        gzip.compress(f"conformaltracking; 38735; HEAD; {install}; ROOT-5be27\n".encode())
    )
    return run


def test_a_component_names_its_package_and_links_to_its_code(monkeypatch, tmp_path):
    run = _run_with_tracking_package(tmp_path)
    at = _page(monkeypatch, tmp_path, run, "Components", {"component": "MyConformalTracking"})
    identity = next(c.value for c in at.caption if "configuration **baseline**" in c.value)
    assert "**ConformalTrackingV2**" in identity
    assert "package conformaltracking @ 38735aa" in identity
    links = {b.proto.label: b.proto.url for b in at.get("link_button")}
    assert links["Repository"] == "https://github.com/iLCSoft/ConformalTracking/tree/38735aa"
    assert "symbol%3AConformalTrackingV2" in links["Code"]
    # A processor whose library was not recorded has no package, and no links.
    _choose_component(at, "Refit")
    assert not at.exception, at.exception
    assert "package unknown" in next(c.value for c in at.caption if "configuration **" in c.value)
    assert not at.get("link_button")


def test_the_event_chart_can_put_the_longest_events_first(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Breakdown")
    at.button_group(key="reco_breakdown_event_order").set_value("Longest first").run()
    events = _charts(at)[-1]
    totals = np.sum([_values(t["y"]) for t in events], axis=0)
    assert list(totals) == sorted(totals, reverse=True)


def test_the_breakdown_follows_the_configuration(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Breakdown")
    at.selectbox(key="reco_breakdown_config").set_value("variant_truth_tracking").run()
    assert not at.exception, at.exception
    (loop,) = _charts(at)[1]
    assert "MyTruthTrackFinder" in loop["labels"]
    assert "MyConformalTracking" not in loop["labels"]


def _phase_history_app(dashboard_dir, run_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from config import Config
    from data import RunData
    from reco import breakdown
    from section_dispatch import PageContext
    from sidebar import TrendWindow

    ctx = PageContext(
        config=Config(run_dir, "https://store.example", "/tmp", "https://dash.example"),
        selection=None, data_dir=run_dir, run_meta=None,
        run=RunData(None, lambda _artifacts: run_dir),
        window=TrendWindow(None, download=lambda _artifacts: (run_dir,)),
    )
    breakdown.render(ctx)


def test_the_phase_history_is_one_chart_for_the_chosen_phase():
    at = AppTest.from_function(
        _phase_history_app, args=(str(_DASHBOARD_DIR), str(RECO_RUN)), default_timeout=60,
    )
    at.run()
    at.radio(key="reco_breakdown_view").set_value("Historical Trends").run()
    assert not at.exception, at.exception
    phase = at.button_group(key="reco_breakdown_hist_phase")
    # The fixture keeps the initialize and execute spans; configure is derived.
    assert phase.options == ["Configure", "Initialize", "Event loop"]
    assert phase.value == "initialize"
    assert len(at.get("plotly_chart")) == 1
    bars = [t for t in _traces(at) if t["type"] == "bar"]
    assert bars[0]["name"] == "GeoSvc"
    assert bars[-2]["name"].startswith("Other (")
    # The phase's time no component was charged for tops each column.
    assert bars[-1]["name"] == "Unattributed"
    # The same components as lines.
    at.button_group(key="reco_breakdown_hist_chart_type").set_value("Lines").run()
    lines = [t for t in _traces(at) if t["type"] == "scatter"]
    assert [t["name"] for t in lines] == [b["name"] for b in bars]
    # The event loop splits by algorithm too.
    phase.set_value("execute").run()
    assert _traces(at)[0]["name"] == "MyConformalTracking"
    # Configure is timed only as a whole: one line, no chart or scale choice.
    at.button_group(key="reco_breakdown_hist_phase").set_value("configure").run()
    (line,) = _traces(at)
    assert line["type"] == "scatter" and line["name"] == "Configure"
    for key in ("reco_breakdown_hist_chart_type", "reco_breakdown_hist_scale"):
        with pytest.raises(KeyError):
            at.button_group(key=key)
    # No explanatory text under the chart.
    assert not at.caption


def test_a_run_timed_in_the_warm_up_alone_shows_no_event_loop_numbers(monkeypatch, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    for path in run.glob("*_components.json"):
        raw = json.loads(path.read_text())
        # Only the warm-up event, which every summary leaves out.
        raw["event_numbers"] = raw["event_numbers"][:1]
        raw["execute"] = {metric: rows[:1] for metric, rows in raw["execute"].items()}
        path.write_text(json.dumps(raw))
    at = _page(monkeypatch, tmp_path, run, "Components", {"component": "VertexFinder"})
    assert "Time per event" not in {m.label for m in at.metric}
    assert any(c.value == "Not run in the event loop." for c in at.caption)


def _axis_titles(at: AppTest) -> set[str]:
    layout = json.loads(at.get("plotly_chart")[0].proto.spec)["layout"]
    return {
        axis["title"]["text"] for name, axis in layout.items()
        if name.startswith(("xaxis", "yaxis")) and "text" in axis.get("title", {})
    }


def test_event_timing_sets_algorithms_beside_the_whole_event(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Event Timing")
    timing_of = at.multiselect(key="evt_timing_timings")
    # At first: the whole event beside the baseline's slowest algorithm, for
    # the baseline alone.
    assert timing_of.value == ["Whole event", "MyConformalTracking"]
    assert at.multiselect(key="evt_timing_configs").value == []
    assert {t["name"] for t in _traces(at) if t.get("name")} == {
        "Whole event", "MyConformalTracking",
    }
    assert "Time per event (s)" in _axis_titles(at)
    # Algorithms are offered slowest first, with their time; services are not.
    assert timing_of.options[0] == "Whole event"
    assert any(o.startswith("MyConformalTracking  ·  ") for o in timing_of.options)
    assert not any(o.startswith("GeoSvc") for o in timing_of.options)

    # Several timings of several configurations: each pair is one series.
    at.multiselect(key="evt_timing_configs").set_value(["variant_truth_tracking"]).run()
    at.multiselect(key="evt_timing_timings").set_value(["Whole event", "VertexFinder"]).run()
    assert not at.exception, at.exception
    assert {t["name"] for t in _traces(at) if t.get("name")} == {
        "Whole event · baseline", "VertexFinder · baseline",
        "Whole event · variant_truth_tracking", "VertexFinder · variant_truth_tracking",
    }
    assert len(at.dataframe[0].value) == 4

    # The whole event alone, as on the Simulation page.
    at.multiselect(key="evt_timing_timings").set_value(["Whole event"]).run()
    assert "Event time (s)" in _axis_titles(at)

    # One algorithm: the configurations that run it, named as they are.
    at.multiselect(key="evt_timing_timings").set_value(["MyConformalTracking"]).run()
    assert not at.exception, at.exception
    assert "MyConformalTracking time (s)" in _axis_titles(at)
    assert at.selectbox(key="evt_timing_baseline").options == ["baseline"]
    assert at.dataframe[0].value.index.tolist() == ["baseline"]


def _timing_values(at: AppTest) -> list[str]:
    values = at.query_params["timing"]
    return values if isinstance(values, list) else [values]


def test_event_timing_links_its_timings_by_name_not_by_label(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Event Timing")
    # The labels carry this run's times; the link carries the names alone.
    assert _timing_values(at) == ["Whole event", "MyConformalTracking"]
    at.multiselect(key="evt_timing_timings").set_value(["Whole event", "VertexFinder"]).run()
    assert _timing_values(at) == ["Whole event", "VertexFinder"]

    # A copied link reopens them; one that did not run here is named.
    linked = _page(monkeypatch, tmp_path, RECO_RUN, "Event Timing",
                   params={"timing": ["VertexFinder", "NoSuchAlgorithm"]})
    assert linked.multiselect(key="evt_timing_timings").value == ["VertexFinder"]
    assert any(
        "Not timed in this run: `NoSuchAlgorithm`" in c.value for c in linked.caption
    )


def _timing_selector_app(dashboard_dir, run_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    from data import cached_load_component_timing
    from tabs import event_timing

    components = cached_load_component_timing(run_dir)[0]
    # A test opens another run, one timed by the variant alone, through "variant".
    if st.session_state.get("variant"):
        components = {"variant_truth_tracking": components["variant_truth_tracking"]}
    st.session_state["chosen"] = event_timing._timing_selector(components, "baseline")


def test_event_timing_names_an_algorithm_the_next_run_did_not_time():
    at = AppTest.from_function(
        _timing_selector_app, args=(str(_DASHBOARD_DIR), str(RECO_RUN)), default_timeout=60,
    )
    at.run()
    at.multiselect(key="evt_timing_timings").set_value(["Whole event", "MyConformalTracking"]).run()
    # Only the baseline runs MyConformalTracking.
    at.session_state["variant"] = True
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["chosen"] == ["Whole event"]
    assert any("Not timed in this run: `MyConformalTracking`" in c.value for c in at.caption)
    assert _timing_values(at) == ["Whole event"]
    # The choices it kept are kept, and nothing is named again.
    at.run()
    assert at.session_state["chosen"] == ["Whole event"]
    assert not at.caption


def _event_timing_history_app(dashboard_dir, run_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from config import Config
    from data import RunData, cached_load_results
    from reco import page
    from section_dispatch import PageContext
    from sidebar import TrendWindow

    ctx = PageContext(
        config=Config(run_dir, "https://store.example", "/tmp", "https://dash.example"),
        selection=None, data_dir=run_dir, run_meta=None,
        run=RunData(cached_load_results(run_dir), lambda _artifacts: run_dir,
                    reads_components=True),
        window=TrendWindow(None, download=lambda _artifacts: (run_dir,)),
    )
    page._event_timing(ctx)


def test_event_timing_history_draws_the_slowest_algorithms_beside_the_event():
    at = AppTest.from_function(
        _event_timing_history_app, args=(str(_DASHBOARD_DIR), str(RECO_RUN)),
        default_timeout=60,
    )
    at.run()
    assert not at.exception, at.exception
    at.radio(key="evt_timing_view_mode").set_value("Historical Trends").run()
    assert not at.exception, at.exception
    # The baseline's whole event, then its five slowest algorithms.
    assert at.selectbox(key="evt_timing_hist_config").value == "baseline"
    names = list(dict.fromkeys(t["name"] for t in _traces(at)))
    assert names[:2] == ["Whole event", "MyConformalTracking"]
    assert len(names) == 6
    medians = [t for t in _traces(at) if t["yaxis"] == "y"]
    assert all(0 < y < 10 for t in medians for y in _values(t["y"]))
    # On log y axes at first, and linear ones on request.
    def y_types():
        layout = json.loads(at.get("plotly_chart")[0].proto.spec)["layout"]
        return {a.get("type", "linear") for n, a in layout.items() if n.startswith("yaxis")}
    assert y_types() == {"log"}
    at.toggle(key="evt_timing_hist_log_y").set_value(False).run()
    assert y_types() == {"linear"}
    # As many as asked, up to all of them.
    slider = at.slider(key="evt_timing_hist_slowest")
    slider.set_value(1).run()
    assert list(dict.fromkeys(t["name"] for t in _traces(at))) == [
        "Whole event", "MyConformalTracking",
    ]
    # Another configuration, its own slowest: the truth tracker.
    at.selectbox(key="evt_timing_hist_config").set_value("variant_truth_tracking").run()
    assert not at.exception, at.exception
    assert list(dict.fromkeys(t["name"] for t in _traces(at)))[1] == "MyTruthTrackFinder"


def test_event_timing_ratios_set_each_timing_against_its_own(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Event Timing")
    # The baseline alone: its timings are not compared with one another.
    assert not any(t.get("xaxis") in ("x3", "x4") for t in _traces(at))
    assert at.dataframe[0].value["Ratio to baseline"].tolist() == [1.0, 1.0]

    at.multiselect(key="evt_timing_configs").set_value(["variant_truth_tracking"]).run()
    at.multiselect(key="evt_timing_timings").set_value(["Whole event", "VertexFinder"]).run()
    assert not at.exception, at.exception
    ratios = {t["name"] for t in _traces(at) if t.get("xaxis") in ("x3", "x4")}
    assert ratios == {
        "Whole event · variant_truth_tracking", "VertexFinder · variant_truth_tracking",
    }
    # Each timing's baseline is its own reference.
    ratio = at.dataframe[0].value["Ratio to baseline"]
    assert ratio["Whole event · baseline"] == pytest.approx(1.0)
    assert ratio["VertexFinder · baseline"] == pytest.approx(1.0)


def _history_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three runs of one configuration: two of the 10-16 tag, the later one
    kept, and an unreliable one of 10-17. Over every run B is the slower,
    over the reliable latest-of-tag one A."""
    runs = pd.DataFrame({
        "run_id": ["r0", "r1", "r2"],
        "run_date": pd.to_datetime(["2026-10-16 01:00", "2026-10-16 05:00", "2026-10-17 01:00"]),
        "x_date": pd.to_datetime(["2026-10-16", "2026-10-16", "2026-10-17"]),
        "k4h_release": ["key4hep-10-16", "key4hep-10-16", "key4hep-10-17"],
        "label": "baseline",
        "n_events": 10,
    })
    event = runs.assign(mean_time_s=60.0, median_time_s=60.0, std_time_s=1.0)
    times = {"A": [3.0, 3.0, 3.0], "B": [50.0, 0.2, 20.0]}
    algorithms = pd.concat(
        runs.assign(component=name, mean_time_s=t, median_time_s=t, std_time_s=0.1)
        for name, t in times.items()
    )
    return event, algorithms


def _synthetic_history_app(dashboard_dir, event, algorithms) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from tabs import event_timing

    # The selected run has no algorithm timing; the window has.
    event_timing.render(
        None, lambda: (event, {"r0": True, "r1": True, "r2": False}), True,
        components=None, load_algorithm_trend=lambda: algorithms,
    )


def test_event_timing_history_ranks_the_runs_it_draws():
    at = AppTest.from_function(
        _synthetic_history_app, args=(str(_DASHBOARD_DIR), *_history_frames()),
        default_timeout=60,
    )
    at.run()
    at.radio(key="evt_timing_view_mode").set_value("Historical Trends").run()
    assert not at.exception, at.exception
    at.slider(key="evt_timing_hist_slowest").set_value(1).run()
    # Reliable runs, each tag's latest: A is the slower.
    assert at.segmented_control(key="evt_timing_hist_exclude_unreliable").value == "Reliable only"
    assert list(dict.fromkeys(t["name"] for t in _traces(at))) == ["Whole event", "A"]
    # Every run, each tag's latest: B, at 0.2 s and 20 s.
    at.segmented_control(key="evt_timing_hist_exclude_unreliable").set_value("All runs").run()
    assert not at.exception, at.exception
    assert list(dict.fromkeys(t["name"] for t in _traces(at))) == ["Whole event", "B"]


def _components_only_app(dashboard_dir, run_dir, trends_enabled) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from data import cached_load_component_timing
    from tabs import event_timing

    # The run's algorithms were timed, its events were not read.
    event_timing.render(
        None, None, trends_enabled, components=cached_load_component_timing(run_dir)[0],
    )


@pytest.mark.parametrize("trends_enabled", [False, True], ids=["local", "remote"])
def test_event_timing_draws_algorithms_of_a_run_without_events(trends_enabled):
    at = AppTest.from_function(
        _components_only_app, args=(str(_DASHBOARD_DIR), str(RECO_RUN), trends_enabled),
        default_timeout=60,
    )
    at.run()
    assert not at.exception, at.exception
    assert at.multiselect(key="evt_timing_timings").value == ["Whole event", "MyConformalTracking"]
    # The whole event, unmeasured, leaves the algorithm drawn on its own.
    assert not at.info
    assert {t["name"] for t in _traces(at) if t.get("name")} == {"baseline"}
    assert "MyConformalTracking time (s)" in _axis_titles(at)
    # The whole event alone: nothing to draw, and the view says so.
    at.multiselect(key="evt_timing_timings").set_value(["Whole event"]).run()
    assert not at.exception, at.exception
    assert "No event timing data" in at.info[0].value


def _warm_up_only_app(dashboard_dir, run_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from data import cached_load_component_timing, cached_load_event_timing
    from tabs import event_timing

    components = cached_load_component_timing(run_dir)[0]
    # The variant's VertexFinder ran in the warm-up event alone.
    wall = components["variant_truth_tracking"].execute["wall_s"]
    wall.loc[wall.index != 0, "VertexFinder"] = float("nan")
    event_timing.render(cached_load_event_timing(run_dir), None, False, components=components)


def test_event_timing_leaves_out_a_configuration_timed_in_the_warm_up_alone():
    at = AppTest.from_function(
        _warm_up_only_app, args=(str(_DASHBOARD_DIR), str(RECO_RUN)), default_timeout=60,
    )
    at.run()
    assert not at.exception, at.exception
    at.multiselect(key="evt_timing_configs").set_value(["variant_truth_tracking"]).run()
    at.multiselect(key="evt_timing_timings").set_value(["Whole event", "VertexFinder"]).run()
    assert not at.exception, at.exception
    assert {t["name"] for t in _traces(at) if t.get("name")} == {
        "Whole event · baseline", "VertexFinder · baseline",
        "Whole event · variant_truth_tracking",
    }
    # Alone, VertexFinder offers the configuration it ran in after the warm-up.
    at.multiselect(key="evt_timing_timings").set_value(["VertexFinder"]).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="evt_timing_baseline").options == ["baseline"]


def test_a_run_without_algorithm_timing_says_so(monkeypatch, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    for path in run.glob("*_components.json"):
        path.unlink()
    (run / "run_info.json").write_text(json.dumps(
        {**json.loads((RECO_RUN / "run_info.json").read_text()), "suite": "reco"}
    ))
    at = _page(monkeypatch, tmp_path, run, "Breakdown")
    assert "No phase timing in this run" in at.info[0].value



# ── Memory, from the memory pass ─────────────────────────────────────────────

_MIB = 1024**2


def test_memory_gives_the_jobs_heap_and_names_what_grows_and_is_kept(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), "Memory")
    # Only the baseline ran its memory pass: nothing to choose.
    with pytest.raises(KeyError):
        at.selectbox(key="reco_memory_config")
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Before the first event"] == "596 MB"
    assert tiles["Growth per event"] == "+4.7 MB"
    # After the four tiles, a card each, the fastest-growing first, its
    # sentence in the badge's tooltip.
    assert [(m.label, m.value, m.proto.delta) for m in at.metric[4:]] == [
        ("LumiCalReco", "+20 MB", "+2 MB per event"),
        ("VertexFinder", "+12 MB", "+1.2 MB per event"),
    ]
    assert [m.proto.help for m in at.markdown if "-badge[" in m.value] == [
        "**LumiCalReco** grows 2 MB per event (20 MB over the run); after finalize it "
        "still holds 20 MB more than after event 1.",
        "**VertexFinder** grows 1.2 MB per event (12 MB over the run); after finalize it "
        "still holds 12 MB more than after event 1.",
    ]
    assert "20 MB still held after finalize" in [c.value for c in at.caption]
    traces = _traces(at)
    named = [t["name"] for t in traces if t.get("stackgroup") == "heap"]
    assert named[0] == "No component" and named[1].startswith("Other (")
    assert {"GeoSvc", "LumiCalReco", "VertexFinder", "k4FWCore__Writer"} <= set(named)
    assert traces[-1]["name"] == "Anonymous RSS (timed run)"


def test_the_suspects_take_the_first_colours_and_keep_them_as_fewer_are_shown(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    from k4bench.analysis.plots import PALETTE

    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), "Memory")

    def fills():
        return {t["name"]: t["fillcolor"] for t in _traces(at) if t.get("stackgroup") == "heap"}

    first = fills()
    assert (first["LumiCalReco"], first["VertexFinder"], first["GeoSvc"]) == tuple(PALETTE[:3])
    at.slider(key="reco_memory_topn").set_value(3).run()
    narrowed = fills()
    kept = {name: color for name, color in narrowed.items() if name in first}
    assert kept == {name: first[name] for name in kept}
    assert set(kept) >= {"LumiCalReco", "VertexFinder", "GeoSvc"}
    assert "k4FWCore__Writer" not in narrowed


def test_a_warned_component_opens_in_components(monkeypatch, tmp_path, reco_run_with_heap):
    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), "Memory")
    assert [b.label for b in at.button if (b.key or "").startswith("reco_memory_inspect_")] == [
        "LumiCalReco", "VertexFinder",
    ]
    # Not the one Components opens on by itself.
    at.button(key="reco_memory_inspect_VertexFinder").click().run()
    assert not at.exception
    assert at.session_state["reco_active_section"] == "Components"
    assert at.session_state["reco_component_name"] == "VertexFinder"
    assert "#### VertexFinder" in [m.value for m in at.markdown]


def test_memory_draws_each_series_change_since_the_first_event(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), "Memory")
    at.segmented_control(key="reco_memory_mode").set_value("Change since the first event").run()
    assert not at.exception
    lines = {t["name"]: _values(t["y"]) for t in _traces(at)}
    assert lines["LumiCalReco"][-1] == pytest.approx(22.0)
    assert lines["k4FWCore__Writer"][-1] == pytest.approx(16.5)
    assert "No component" in lines and any(name.startswith("Other (") for name in lines)


def test_a_smaller_palette_repeats_colours_rather_than_hide_a_suspect(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    def twelve_leaks(raw: dict) -> None:
        # Twelve algorithms, each growing steadily and keeping it after finalize.
        growing = [n for n in raw["components"] if n.startswith("My")][:12]
        for j, name in enumerate(growing):
            i = raw["components"].index(name)
            for e, row in enumerate(raw["execute"]["held_bytes"]):
                row[i] = round((j + 1) * 1.5 * (e + 1) * _MIB)
            raw["at_end"]["held_bytes"][i] = raw["execute"]["held_bytes"][-1][i]

    run = reco_run_with_heap()
    _rewrite_heap(run, twelve_leaks)
    at = _page(monkeypatch, tmp_path, run, "Memory")
    flagged = [b.label for b in at.button if (b.key or "").startswith("reco_memory_inspect_")]
    assert len(flagged) >= 12
    at.selectbox(key="reco_memory_palette").set_value("Matplotlib").run()
    assert not at.exception
    bands = {t["name"] for t in _traces(at) if t.get("stackgroup") == "heap"}
    assert set(flagged) <= bands


def test_components_set_the_heap_beside_the_time(monkeypatch, tmp_path, reco_run_with_heap):
    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), "Components", {"component": "LumiCalReco"})
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Heap held"] == "14 MB" and tiles["Growth per event"] == "+2 MB"
    assert [m.value for m in at.markdown if "-badge[" in m.value] == [
        ":orange-badge[:material/trending_up: Grows]",
    ]
    assert "##### Heap per event" in [m.value for m in at.markdown]
    # The table links every component to its code; a flag is a tint, not a column.
    table = at.dataframe[0].value
    assert {"Package", "Code", "Repository"} <= set(table.columns)
    assert "Flag" not in table.columns
    # Where it stands on the heap: the geometry and the reader hold more.
    standing = next(json.loads(c.proto.spec) for c in at.get("plotly_chart")
                    if "reco_component_standing" in c.proto.id)
    heap = standing["data"][-1]
    assert [row[0] for row in heap["customdata"]][:3] == ["GeoSvc", "k4FWCore__Reader", "LumiCalReco"]
    # A component that held no heap held none: zero, not unknown.
    _choose_component(at, "Refit")
    assert {m.label: m.value for m in at.metric}["Heap held"] == "0 MB"


def _pool(raw: dict) -> None:
    """The first Marlin processor takes on 300 MB in event 4 and keeps it,
    after finalize too."""
    i = next(i for i, name in enumerate(raw["components"]) if name.startswith("My"))
    for e, (start, end, peak) in enumerate(zip(*(raw["execute"][key] for key in (
        "start_held_bytes", "held_bytes", "peak_held_bytes",
    )))):
        end[i] = peak[i] = 300 * _MIB if e >= 4 else 0
        start[i] = 300 * _MIB if e >= 5 else 0
    raw["at_end"]["held_bytes"][i] = 300 * _MIB


def test_memory_names_what_stepped_up_and_marks_its_step(monkeypatch, tmp_path, reco_run_with_heap):
    run = reco_run_with_heap()
    _rewrite_heap(run, _pool)
    name = next(n for n in json.loads((run / "baseline_heap.json").read_text())["components"]
                if n.startswith("My"))
    at = _page(monkeypatch, tmp_path, run, "Memory")
    # Its card comes first, ahead of what grows.
    card = next(m for m in at.metric if m.label == name)
    assert (card.value, card.proto.delta) == ("+300 MB", "1 step, at event 4")
    assert next(m for m in at.markdown if "-badge[" in m.value).proto.help == (
        f"**{name}** stepped up by 300 MB at event 4; it held 300 MB at the last "
        "event's end and 300 MB after finalize."
    )
    (marks,) = [t for t in _traces(at) if t.get("name") == "Step"]
    assert marks["x"] == ["4"] and marks["text"] == [f"{name} +300 MB"]
    assert name in {t["name"] for t in _traces(at) if t.get("stackgroup") == "heap"}

    at.button(key=f"reco_memory_inspect_step_{name}").click().run()
    assert not at.exception
    assert at.session_state["reco_active_section"] == "Components"
    assert any("-badge[" in m.value and "Steps up" in m.value for m in at.markdown)
    assert "##### Heap per event" in [m.value for m in at.markdown]


def _rewrite_heap(run: Path, change) -> None:
    path = run / "baseline_heap.json"
    raw = json.loads(path.read_text())
    change(raw)
    path.write_text(json.dumps(raw))


def _no_events(raw: dict) -> None:
    raw["before_first_event"] = None
    raw["event_numbers"] = []
    raw["execute"] = {key: [] for key in raw["execute"]}


def _warm_up_only(raw: dict) -> None:
    raw["event_numbers"] = raw["event_numbers"][:1]
    raw["execute"] = {key: values[:1] for key, values in raw["execute"].items()}


def _events_at_once(raw: dict) -> None:
    # No peaks per event, and nothing kept after finalize, so nothing is suspected.
    n = len(raw["components"])
    raw["execute"]["peak_held_bytes"] = [[None] * n for _ in raw["event_numbers"]]
    raw["at_end"]["held_bytes"] = [0] * n


def test_every_suspect_gets_a_band_however_many_there_are(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    # Ten suspects and nothing else holding heap: ten series, which the
    # default palette must colour besides its grey.
    run = reco_run_with_heap()
    suspects = ["LumiCalReco", "VertexFinder"]
    emptied = ["GeoSvc", "k4FWCore__Reader", "k4FWCore__Writer"]

    def eight_more(raw: dict) -> None:
        names = raw["components"]
        idle = [n for n in names[1:] if n not in {*suspects, *emptied}]
        suspects.extend(idle[:8])
        for name in emptied:
            i = names.index(name)
            for key in ("start_held_bytes", "held_bytes", "peak_held_bytes"):
                for row in raw["execute"][key]:
                    row[i] = 0
            for key in ("held_bytes", "peak_held_bytes"):
                raw["before_first_event"][key][i] = 0
        for name in idle[:8]:
            i = names.index(name)
            for row, event in zip(raw["execute"]["held_bytes"], raw["event_numbers"]):
                row[i] = 2 * (event + 1) * _MIB
            raw["at_end"]["held_bytes"][i] = raw["execute"]["held_bytes"][-1][i]

    _rewrite_heap(run, eight_more)
    assert len(suspects) == 10
    at = _page(monkeypatch, tmp_path, run, "Memory")
    assert {b.label for b in at.button if (b.key or "").startswith("reco_memory_inspect_")} == set(suspects)
    for mode in ("Held", "Change since the first event"):
        at.segmented_control(key="reco_memory_mode").set_value(mode).run()
        assert not at.exception
        assert set(suspects) <= {t.get("name") for t in _traces(at)}


@pytest.mark.parametrize(
    "change", [_no_events, _warm_up_only, _events_at_once],
    ids=["no events", "the warm-up alone", "events at once"],
)
def test_memory_draws_a_run_whatever_statistics_it_lacks(
    monkeypatch, tmp_path, change, reco_run_with_heap,
):
    run = reco_run_with_heap()
    _rewrite_heap(run, change)
    at = _page(monkeypatch, tmp_path, run, "Memory")
    assert not at.exception
    at = _page(monkeypatch, tmp_path, run, "Components", {"component": "LumiCalReco"})
    assert "#### LumiCalReco" in [m.value for m in at.markdown]
    assert "Heap held" in {m.label for m in at.metric}


def test_a_component_stays_chosen_in_another_run_that_has_it(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    first, second = reco_run_with_heap("first"), reco_run_with_heap("second")
    at = _page(monkeypatch, tmp_path, first, "Components", {"component": "k4FWCore__Writer"})
    assert "#### k4FWCore__Writer" in [m.value for m in at.markdown]
    at.text_input(key="reco_data_dir").set_value(str(second)).run()
    assert not at.exception
    assert "#### k4FWCore__Writer" in [m.value for m in at.markdown]
    assert "run directory `second`" in " ".join(c.value for c in at.caption)


def test_too_few_events_to_tell_steady_growth_is_said_so(monkeypatch, tmp_path, reco_run_with_heap):
    def nine_events(raw: dict) -> None:
        raw["event_numbers"] = raw["event_numbers"][:9]
        raw["execute"] = {key: values[:9] for key, values in raw["execute"].items()}

    run = reco_run_with_heap()
    _rewrite_heap(run, nine_events)
    at = _page(monkeypatch, tmp_path, run, "Memory")
    assert not at.warning
    assert at.info[0].value.startswith("Only 8 events after the warm-up: too few to tell")


def test_a_sparkline_keeps_a_spike_of_one_event():
    from reco.memory import _SPARK_POINTS, spark as _spark

    held = pd.Series(100.0, index=range(1200))
    held[501] = 600.0
    spark = _spark(held)
    assert len(spark) <= _SPARK_POINTS and max(spark) == 600.0
    # A rise stays a rise: each run's least comes before its most.
    rising = _spark(pd.Series(np.arange(1200, dtype=float)))
    assert rising == sorted(rising) and rising[0] == 0 and rising[-1] == 1199
    short = pd.Series([1.0, 2.0, 3.0])
    assert _spark(short) == [1.0, 2.0, 3.0]


def test_a_malformed_heap_file_hides_no_other_configuration(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    run = reco_run_with_heap()
    (run / "variant_truth_tracking_heap.json").write_text("{")
    at = _page(monkeypatch, tmp_path, run, "Memory")
    with pytest.raises(KeyError):
        at.selectbox(key="reco_memory_config")
    assert [w.value for w in at.warning][0].startswith(
        "The heap file of **variant_truth_tracking** could not be read, so it is not offered:"
    )
    assert any(m.label == "Before the first event" for m in at.metric)


def _run_with_broken_components(tmp_path, *labels: str) -> Path:
    """The fixture run with each of *labels*' ``_components.json`` malformed."""
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    for label in labels:
        (run / f"{label}_components.json").write_text("{")
    return run


_UNREADABLE_VARIANT = (
    "- **variant_truth_tracking**: variant_truth_tracking_components.json: JSONDecodeError:"
)


@pytest.mark.parametrize("section", [
    "Time", "Memory", "Components", "Data Flow", "Event Timing", "Correlations",
])
def test_a_malformed_component_file_is_named_in_every_section_reading_one(
    monkeypatch, tmp_path, section,
):
    run = _run_with_broken_components(tmp_path, "variant_truth_tracking")
    at = _page(monkeypatch, tmp_path, run, section)
    (warning,) = [w.value for w in at.warning if "component costs" in w.value]
    assert _UNREADABLE_VARIANT in warning


@pytest.mark.parametrize("section, key", [
    ("Time", "reco_breakdown_config"), ("Components", "reco_component_config"),
])
def test_a_malformed_component_file_hides_no_other_configuration(
    monkeypatch, tmp_path, section, key,
):
    run = _run_with_broken_components(tmp_path, "variant_truth_tracking")
    at = _page(monkeypatch, tmp_path, run, section)
    # The baseline alone is left: nothing to choose.
    with pytest.raises(KeyError):
        at.selectbox(key=key)
    assert not at.info


def test_a_malformed_component_file_keeps_its_configurations_events(monkeypatch, tmp_path):
    run = _run_with_broken_components(tmp_path, "variant_truth_tracking")
    at = _page(monkeypatch, tmp_path, run, "Event Timing")
    assert "variant_truth_tracking" in at.multiselect(key="evt_timing_configs").options


def test_a_run_whose_component_files_are_all_unreadable_says_so(monkeypatch, tmp_path):
    run = _run_with_broken_components(tmp_path, "baseline", "variant_truth_tracking")
    at = _page(monkeypatch, tmp_path, run, "Breakdown")
    assert "left a `_components.json` that could be read" in at.info[0].value


def test_a_run_without_component_files_says_so_without_a_warning(monkeypatch, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    for path in run.glob("*_components.json"):
        path.unlink()
    at = _page(monkeypatch, tmp_path, run, "Components")
    assert at.info[0].value.endswith("left a `_components.json`.")
    assert not at.warning


def test_a_failed_configurations_malformed_component_file_is_not_news(monkeypatch, tmp_path):
    run = _run_with_broken_components(tmp_path, "variant_truth_tracking")
    results = run / "variant_truth_tracking_results.csv"
    header, row = results.read_text().splitlines()
    results.write_text(f"{header}\n{row.replace(',0,', ',1,', 1)}\n")
    at = _page(monkeypatch, tmp_path, run, "Breakdown")
    with pytest.raises(KeyError):
        at.selectbox(key="reco_breakdown_config")
    assert not at.warning


def test_a_repaired_component_file_is_offered_after_a_refresh(monkeypatch, tmp_path):
    run = _run_with_broken_components(tmp_path, "variant_truth_tracking")
    at = _page(monkeypatch, tmp_path, run, "Breakdown")
    shutil.copy(RECO_RUN / "variant_truth_tracking_components.json", run)
    next(b for b in at.button if b.label == "Refresh Data").click().run()
    assert not at.exception, at.exception
    assert at.selectbox(key="reco_breakdown_config").options == [
        "baseline", "variant_truth_tracking",
    ]
    assert not at.warning


def test_a_run_without_a_memory_pass_says_so(monkeypatch, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    (run / "run_info.json").write_text(json.dumps(
        {**json.loads((RECO_RUN / "run_info.json").read_text()), "suite": "reco"}
    ))
    at = _page(monkeypatch, tmp_path, run, "Memory")
    assert "No heap breakdown in this run" in at.info[0].value


def _memory_history_app(dashboard_dir, run_dirs) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from config import Config
    from data import RunData
    from reco import memory
    from section_dispatch import PageContext
    from sidebar import TrendWindow

    ctx = PageContext(
        config=Config(run_dirs[-1], "https://store.example", "/tmp", "https://dash.example"),
        selection=None, data_dir=run_dirs[-1], run_meta=None,
        run=RunData(None, lambda _artifacts: run_dirs[-1]),
        window=TrendWindow(None, download=lambda _artifacts: run_dirs),
    )
    memory.render(ctx)


_NIGHTS = ("2026-10-18", "2026-10-19", "2026-10-20")


def _memory_history(runs: list[Path]) -> AppTest:
    at = AppTest.from_function(
        _memory_history_app, args=(str(_DASHBOARD_DIR), tuple(map(str, runs))),
        default_timeout=60,
    )
    at.run()
    at.radio(key="reco_memory_view").set_value("Historical Trends").run()
    assert not at.exception, at.exception
    return at


def _nights(reco_run_with_heap) -> list[Path]:
    return [reco_run_with_heap(night, night) for night in _NIGHTS]


def _line_colors(at: AppTest) -> dict[str, str]:
    return {t["name"]: t["marker"]["line"]["color"] for t in _traces(at)}


def test_the_memory_history_draws_each_components_heap_night_by_night(reco_run_with_heap):
    from k4bench.analysis.plots import PALETTE
    from k4bench.analysis.plots.heap import _INK, _NO_COMPONENT_LINE

    at = _memory_history(_nights(reco_run_with_heap))
    assert at.segmented_control(key="reco_memory_hist_metric").value == "Typical event"
    assert "##### Heap at an event's end, night by night" in [m.value for m in at.markdown]
    # Lines only.
    with pytest.raises(KeyError):
        at.segmented_control(key="reco_memory_hist_chart_type")
    at.slider(key="reco_memory_hist_topn").set_value(2).run()
    traces = _traces(at)
    assert {t["type"] for t in traces} == {"scatter"}
    # The whole job's median in ink, the suspects, the fastest-growing first,
    # then the two largest, and what no component allocated last, in grey;
    # the rest are not drawn.
    assert _line_colors(at) == {
        "Whole-job median": _INK, "LumiCalReco": PALETTE[0], "VertexFinder": PALETTE[1],
        "GeoSvc": PALETTE[2], "k4FWCore__Reader": PALETTE[3], "No component": _NO_COMPONENT_LINE,
    }
    geo = next(t for t in traces if t["name"] == "GeoSvc")
    assert _values(geo["y"]) == pytest.approx([397.0] * len(_NIGHTS))
    assert geo["customdata"][0][2] == "397 MB"
    # Before the first event, the lines add up to the job's heap.
    at.segmented_control(key="reco_memory_hist_metric").set_value("Before the first event").run()
    assert not at.exception, at.exception
    lines = np.sum([_values(t["y"]) for t in _traces(at)], axis=0)
    assert lines == pytest.approx([596.0] * len(_NIGHTS))
    # No explanatory text under the chart.
    assert not at.caption


def test_the_memory_history_draws_working_set_and_growth_as_lines(reco_run_with_heap):
    at = _memory_history(_nights(reco_run_with_heap))
    at.segmented_control(key="reco_memory_hist_metric").set_value("Working set").run()
    assert not at.exception, at.exception
    # Two panels: the median and the 95th percentile.
    traces = _traces(at)
    assert {t["yaxis"] for t in traces} == {"y", "y2"}
    reader = [t for t in traces if t["name"] == "k4FWCore__Reader"]
    assert [_values(t["y"]) for t in reader] == [pytest.approx([5.0] * 3)] * 2
    assert "No component" not in {t["name"] for t in traces}
    at.segmented_control(key="reco_memory_hist_metric").set_value("Growth per event").run()
    assert not at.exception, at.exception
    traces = _traces(at)
    assert {t["yaxis"] for t in traces} == {"y"}
    # The fastest growing first, the suspects among them.
    assert [t["name"] for t in traces][:3] == ["LumiCalReco", "VertexFinder", "k4FWCore__Writer"]
    assert traces[0]["customdata"][0][2] == "+2 MB per event"
    assert not at.caption


def test_the_memory_history_leaves_a_night_without_a_typical_event_out(reco_run_with_heap):
    # The middle night ran only its warm-up event: no typical heap, not 0 MB.
    runs = _nights(reco_run_with_heap)
    _rewrite_heap(runs[1], _warm_up_only)
    at = _memory_history(runs)
    for trace in _traces(at):
        values = _values(trace["y"])
        assert np.isnan(values[1]), trace["name"]
        assert not np.isnan(values[0]), trace["name"]


def _rotating(raw: dict) -> None:
    # Events 1–9 pass 100 MB from one of three components to the next, so
    # each holds it at a third of the events' ends and its median is 0 MB,
    # while the whole job's is 100 MB beside what no component holds: 7 MB
    # at five events' ends, nothing at four. The warm-up event holds nothing:
    # counted, it would take the job's median to 103.5 MB and no component's
    # to 3.5 MB.
    names = raw["components"]
    rotating = names[1:4]
    events = list(range(10))

    def held(event: int) -> list[int]:
        values = [0] * len(names)
        if event:
            values[names.index(rotating[event % 3])] = 100 * _MIB
            values[0] = 7 * _MIB if event <= 5 else 0
        return values

    raw["before_first_event"]["held_bytes"] = [0] * len(names)
    raw["before_first_event"]["peak_held_bytes"] = [0] * len(names)
    raw["event_numbers"] = events
    starts = [held(e - 1) if e else [0] * len(names) for e in events]
    raw["execute"] = {
        "start_held_bytes": starts,
        "held_bytes": [held(e) for e in events],
        # A component peaks at no less than it held as the event started.
        "peak_held_bytes": [list(map(max, start, held(e))) for e, start in zip(events, starts)],
        "malloc_in_use_bytes": [sum(held(e)) for e in events],
        "rss_anon_mb": [700.0] * len(events),
    }
    raw["at_end"]["held_bytes"] = [0] * len(names)


def test_the_memory_history_draws_the_whole_jobs_median_not_the_sum_of_medians(
    reco_run_with_heap,
):
    from k4bench.analysis.plots.heap import _INK, _NO_COMPONENT_LINE

    runs = _nights(reco_run_with_heap)
    for run in runs:
        _rewrite_heap(run, _rotating)
    # The middle night has no memory pass: no point, not 0 MB.
    (runs[1] / "baseline_heap.json").unlink()
    rotating = json.loads((runs[0] / "baseline_heap.json").read_text())["components"][1:4]
    at = _memory_history(runs)
    assert at.segmented_control(key="reco_memory_hist_metric").value == "Typical event"
    traces = {t["name"]: t for t in _traces(at)}
    # The components' medians as they are, no Other summing the rest.
    assert set(traces) == {"Whole-job median", *rotating, "No component"}
    assert {name: list(_values(traces[name]["y"])) for name in traces} == {
        "Whole-job median": [107.0, 107.0], **{name: [0.0, 0.0] for name in rotating},
        "No component": [7.0, 7.0],
    }
    assert [x[:10] for x in traces["Whole-job median"]["x"]] == [_NIGHTS[0], _NIGHTS[2]]
    assert _line_colors(at)["Whole-job median"] == _INK
    assert _line_colors(at)["No component"] == _NO_COMPONENT_LINE
    # Fewer components named leave the whole job's median as it is.
    at.slider(key="reco_memory_hist_topn").set_value(1).run()
    traces = {t["name"]: t for t in _traces(at)}
    assert len(set(traces) & set(rotating)) == 1
    assert list(_values(traces["Whole-job median"]["y"])) == [107.0, 107.0]


def test_the_memory_history_names_the_heaviest_events_needs(reco_run_with_heap):
    def spiky(raw: dict) -> None:
        # IOSvc needs nothing in a typical event, and 400 MB in one of them.
        i = raw["components"].index("IOSvc")
        raw["execute"]["peak_held_bytes"][6][i] += 400 * _MIB

    runs = _nights(reco_run_with_heap)
    for run in runs:
        _rewrite_heap(run, spiky)
    at = _memory_history(runs)
    at.slider(key="reco_memory_hist_topn").set_value(1).run()
    at.segmented_control(key="reco_memory_hist_metric").set_value("Working set").run()
    assert not at.exception, at.exception
    names = {t["name"] for t in _traces(at)}
    # The largest typical need, and the largest of the heavier events.
    assert {"k4FWCore__Reader", "IOSvc"} <= names


def test_the_memory_history_names_a_component_flagged_on_any_night(reco_run_with_heap):
    def released(raw: dict) -> None:
        # VertexFinder frees what it gained at the end: not suspected that night.
        raw["at_end"]["held_bytes"][raw["components"].index("VertexFinder")] = 0

    runs = _nights(reco_run_with_heap)
    for run in runs[1:]:
        _rewrite_heap(run, released)
    at = _memory_history(runs)
    at.slider(key="reco_memory_hist_topn").set_value(1).run()
    for metric in ("Typical event", "Growth per event"):
        at.segmented_control(key="reco_memory_hist_metric").set_value(metric).run()
        assert not at.exception, at.exception
        assert "VertexFinder" in {t.get("name") for t in _traces(at)}


def test_the_memory_historys_colours_hold_as_fewer_are_shown(reco_run_with_heap):
    at = _memory_history(_nights(reco_run_with_heap))

    every = _line_colors(at)
    at.slider(key="reco_memory_hist_topn").set_value(2).run()
    fewer = _line_colors(at)
    assert "k4FWCore__Writer" in every and "k4FWCore__Writer" not in fewer
    named = [name for name in fewer if not name.startswith("Other (")]
    assert {name: fewer[name] for name in named} == {name: every[name] for name in named}


def test_the_memory_history_says_when_the_window_has_no_heap_data(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    at = _memory_history([run])
    assert at.info[0].value.startswith("No heap data in the selected window.")
    assert "2026-10-19" in at.info[0].value
    assert not at.get("plotly_chart")


def _node_ids(at: AppTest) -> list[str]:
    return [row[0] for row in _charts(at)[0][-1]["customdata"]]


def test_the_data_flow_groups_the_instances_of_a_class(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Data Flow")
    assert at.selectbox(key="reco_dataflow_config").options == ["baseline", "variant_truth_tracking"]
    ids = _node_ids(at)
    assert "group:DDPlanarDigiProcessor" in ids and "group:CLICPfoSelector" in ids
    assert "VXDBarrelDigitiser" not in ids
    assert ids[0] == "k4FWCore__Reader" and ids[-1] == "k4FWCore__Writer"
    # What the run could not record is said, not drawn.
    assert "MyAIDAProcessor" not in ids and "EventNumber" not in ids
    assert any(c.value.startswith("5 algorithms without recorded links not drawn")
               for c in at.caption)
    assert any(c.value.startswith("Dot size: time per event, the warm-up left out")
               for c in at.caption)
    (limits,) = at.expander
    text = " ".join(m.value for m in limits.markdown)
    # The Reader's outputs were backfilled and the Writer reads by its rules,
    # so no Gaudi algorithm's collections are unknown.
    assert "Not recorded" not in text
    assert ("Not drawn**, no collection linking them to another algorithm (none declared, or "
            "none with a recorded writer or reader): MyAIDAProcessor, OverlayFalse, "
            "RenameCollection, EventNumber, CollPatcherREC") in text
    assert "PFOsFromJets (VertexFinder)" in text


def test_expanding_shows_every_instance_and_collapsing_regroups(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Data Flow")
    at.button[0].click().run()  # Expand all
    ids = _node_ids(at)
    assert "VXDBarrelDigitiser" in ids and "MyCLICPfoSelectorTight" in ids
    assert not any(i.startswith("group:") for i in ids)
    at.button[1].click().run()  # Collapse all
    assert "group:DDPlanarDigiProcessor" in _node_ids(at)


def test_the_writers_reads_are_drawn_on_request(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, RECO_RUN, "Data Flow")
    without = len(_charts(at)[0])
    at.toggle(key="reco_dataflow_writer").set_value(True).run()
    assert len(_charts(at)[0]) > without


def test_a_selected_algorithm_lists_what_it_reads_and_writes(monkeypatch, tmp_path):
    _local_env(monkeypatch, RECO_RUN, tmp_path / "cache")
    at = AppTest.from_function(_page_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params["tab"] = "Data Flow"
    at.session_state["reco_dataflow_selected"] = "Refit"
    at.run()
    assert not at.exception, at.exception
    reads, writes = (frame.value for frame in at.dataframe)
    tracks = reads.set_index("Collection").loc["SiTracks"]
    assert tracks["Written by"] == "ClonesAndSplitTracksFinder"
    assert tracks["Type"] == "Track" and tracks["Parameter"] == "InputTrackCollectionName (set)"
    assert reads.set_index("Collection").loc["SiTrackRelations", "Written by"] == "unknown"
    refitted = writes.set_index("Collection").loc["SiTracks_Refitted", "Read by"]
    assert "MyDDMarlinPandora_10ns" in refitted


def test_the_selected_writer_lists_what_its_rules_keep(monkeypatch, tmp_path):
    _local_env(monkeypatch, RECO_RUN, tmp_path / "cache")
    at = AppTest.from_function(_page_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params["tab"] = "Data Flow"
    at.session_state["reco_dataflow_selected"] = "k4FWCore__Writer"
    at.run()
    assert not at.exception, at.exception
    assert not any("Not recorded for this run" in i.value for i in at.info)
    # Its rule is ``keep *``: what the Reader read is saved again.
    assert any("MCParticles" in c.value.split(", ") for c in at.caption)


def test_a_run_without_a_data_flow_says_so(monkeypatch, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    for path in run.glob("*_dataflow.json"):
        path.unlink()
    at = _page(monkeypatch, tmp_path, run, "Data Flow")
    assert "No data flow recorded for this run" in at.info[0].value


def test_a_click_on_a_node_names_it_and_other_clicks_are_ignored():
    """The point Streamlit reports for a click on the graph (its documented
    ``curve_number`` and ``point_index``) is mapped back to the node."""
    from types import SimpleNamespace

    from k4bench.analysis.dataflow import load_dataflow
    from k4bench.analysis.plots.dataflow import dataflow_view, plot_dataflow
    from reco.dataflow import _clicked_node

    fig = plot_dataflow(dataflow_view(load_dataflow(RECO_RUN)["baseline"]))
    nodes = len(fig.data) - 1

    def click(curve, index):
        return SimpleNamespace(selection={"points": [{"curve_number": curve, "point_index": index}]})

    assert _clicked_node(click(nodes, 0), fig) == "k4FWCore__Reader"
    assert _clicked_node(click(nodes - 1, 0), fig) is None, "a point on an edge is not a node"
    assert _clicked_node(SimpleNamespace(selection={"points": []}), fig) is None
    assert _clicked_node(None, fig) is None


def test_an_unlinked_instance_in_a_drawn_group_is_not_called_hidden():
    from k4bench.analysis.dataflow import DataFlow
    from k4bench.analysis.plots.dataflow import dataflow_view
    from reco.dataflow import _unlinked

    def algorithm(name, impl, reads=(), writes=()):
        return {"name": name, "type": "MarlinProcessorWrapper", "impl": impl, "framework": "marlin",
                "inputs": [{"name": c, "store": "lcio"} for c in reads],
                "outputs": [{"name": c, "store": "lcio"} for c in writes]}

    flow = DataFlow.from_json({
        "schema_version": 1, "handles_recorded": True, "marlin_recorded": True,
        "algorithms": [algorithm("DigiA", "Digi", writes=["Hits"]), algorithm("DigiB", "Digi"),
                       algorithm("Tracking", "Tracking", reads=["Hits"]), algorithm("Status", "Status")],
    })
    view = dataflow_view(flow, writer_edges=False, isolated=False)
    assert _unlinked(flow, view) == ["Status"]


def test_a_component_timed_in_no_event_has_no_time():
    from types import SimpleNamespace

    from reco.dataflow import seconds_per_event

    # Run only in the warm-up event 0 is not run either.
    wall = pd.DataFrame({"Timed": [9.0, 1.0, np.nan], "Untimed": [np.nan] * 3,
                         "WarmUpOnly": [1.0, np.nan, np.nan]})
    timing = SimpleNamespace(execute={"wall_s": wall})
    ctx = SimpleNamespace(run=SimpleNamespace(components={"baseline": timing}))
    assert seconds_per_event(ctx, "baseline") == {"Timed": 0.5}


def test_data_flow_and_components_agree_on_an_algorithm_run_in_every_event():
    from types import SimpleNamespace

    from reco.dataflow import seconds_per_event

    from k4bench.analysis.component_detail import component_summary

    events = pd.Index([0, 1, 2, 3], name="event_number")
    execute = {
        "wall_s": pd.DataFrame({"Always": [1.0, 0.004, 0.002, 0.003],
                                "Filtered": [1.0, 0.006, np.nan, np.nan]}, index=events),
        "instructions": pd.DataFrame({"Always": [1e9] * 4, "Filtered": [1e9, 1e9, np.nan, np.nan]},
                                     index=events),
        "cycles": pd.DataFrame({"Always": [1e9] * 4, "Filtered": [1e9, 1e9, np.nan, np.nan]},
                               index=events),
    }
    timing = SimpleNamespace(execute=execute, lifecycle={})
    ctx = SimpleNamespace(run=SimpleNamespace(components={"baseline": timing}))
    flow = seconds_per_event(ctx, "baseline")
    summary = component_summary(timing)
    assert flow["Always"] == pytest.approx(summary.at["Always", "seconds_when_run"]) == pytest.approx(0.003)
    # Filtered: its share of every event in Data Flow, as Components' standing
    # counts it, beside its time in the events it ran in.
    assert flow["Filtered"] == pytest.approx(summary.at["Filtered", "seconds_per_event"]) == pytest.approx(0.002)
    assert summary.at["Filtered", "seconds_when_run"] == pytest.approx(0.006)


def test_gaudi_algorithms_without_recorded_handles_are_named(monkeypatch, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    path = run / "baseline_dataflow.json"
    raw = json.loads(path.read_text())
    reader = next(a for a in raw["algorithms"] if a["type"] == "Reader")
    reader["inputs"] = reader["outputs"] = None
    path.write_text(json.dumps(raw))
    at = _page(monkeypatch, tmp_path, run, "Data Flow")
    text = " ".join(m.value for m in at.expander[0].markdown)
    assert "Not recorded** (auditor too old): k4FWCore__Reader" in text

