"""The Reconstruction page's own sections, rendered: the Breakdown of the
10-17 nightly run (``reco_run``) by phase and component, and the Overview's
detector comparison drawn from runs rather than from nightly reports."""

from __future__ import annotations

import base64
import json
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

from k4bench.regression.models import Severity  # noqa: E402
from tabs import _overview_data  # noqa: E402

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


def _page(monkeypatch, tmp_path, run_dir: Path, tab: str | None = None) -> AppTest:
    _local_env(monkeypatch, run_dir, tmp_path / "cache")
    at = AppTest.from_function(_page_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    if tab:
        at.query_params["tab"] = tab
    at.run()
    assert not at.exception, at.exception
    return at


def _traces(at: AppTest) -> list[dict]:
    return json.loads(at.get("plotly_chart")[0].proto.spec)["data"]


def test_a_local_run_opens_on_the_breakdown(monkeypatch, tmp_path):
    # The Overview compares detectors across the window, which a local run lacks.
    at = _page(monkeypatch, tmp_path, RECO_RUN)
    assert at.session_state["reco_active_section"] == "Breakdown"
    assert at.button_group[0].options == [
        "Breakdown", "Efficiency", "Memory", "Data Flow", "Event Timing", "Event Memory",
        "Event Size", "Machine Info", "Logs",
    ]


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
    # Local mode has no trend window to draw history from.
    assert at.radio(key="reco_breakdown_view").options == ["Current Run", "Components"]
    # The component table is a view of its own.
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


def test_the_component_table_links_each_class_to_its_package(monkeypatch, tmp_path):
    run = _run_with_tracking_package(tmp_path)
    at = _page(monkeypatch, tmp_path, run, "Breakdown")
    at.radio(key="reco_breakdown_view").set_value("Components").run()
    assert not at.exception, at.exception
    assert not at.get("plotly_chart")
    (table,) = at.dataframe
    df = table.value.set_index("Component")
    row = df.loc["MyConformalTracking"]
    assert row["Class"] == "ConformalTrackingV2"
    assert row["Package"] == "conformaltracking @ 38735aa"
    assert row["Repository"] == "https://github.com/iLCSoft/ConformalTracking/tree/38735aa"
    assert "symbol%3AConformalTrackingV2" in row["Code"]
    # Algorithms and services, most expensive first.
    assert set(df["Kind"]) == {"Algorithm", "Service"}
    totals = df[["Initialization", "Event loop", "Finalization"]].sum(axis=1)
    assert list(totals) == sorted(totals, reverse=True)
    # The other processors' libraries were not recorded.
    assert not df.loc["Refit", "Package"].startswith("marlintrk")
    assert any("Marlin processor(s) have no package" in c.value for c in at.caption)


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
    # The Components view keeps the chosen configuration.
    at.radio(key="reco_breakdown_view").set_value("Components").run()
    assert at.selectbox(key="reco_breakdown_config").value == "variant_truth_tracking"
    components = at.dataframe[0].value["Component"].tolist()
    assert "MyTruthTrackFinder" in components
    assert "MyConformalTracking" not in components


def _phase_history_app(dashboard_dir, run_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from config import Config
    from data import RunData
    from k4bench.analysis.trend import build_results_trend
    from reco import breakdown
    from section_dispatch import PageContext
    from sidebar import TrendWindow

    results = build_results_trend((run_dir,))
    ctx = PageContext(
        config=Config(run_dir, "https://store.example", "/tmp", "https://dash.example"),
        selection=None, data_dir=run_dir, run_meta=None,
        run=RunData(None, None, None, results, sorted(results["label"].unique())),
        window=TrendWindow(None, (run_dir,)), trend_machine=None, reliability={},
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


def test_efficiency_draws_each_algorithm_of_the_event_loop(monkeypatch, tmp_path):
    run = _run_with_tracking_package(tmp_path)
    at = _page(monkeypatch, tmp_path, run, "Efficiency")
    # Coloured by package: one trace each, the most time first. The fixture's
    # manifest names only the tracking's package.
    traces = _traces(at)
    assert [t["name"] for t in traces] == ["Unknown", "conformaltracking @ 38735aa"]
    names = [name for t in traces for name, _ in t["customdata"]]
    assert traces[1]["customdata"][0][0] == "MyConformalTracking"
    # Algorithms only: no services, no sequencers.
    assert "GeoSvc" not in names
    assert all(0 < ipc < 8 for t in traces for ipc in _values(t["y"]))
    # The points can be one size.
    at.selectbox(key="reco_efficiency_size").set_value("Same size").run()
    assert not at.exception, at.exception
    assert len({s for t in _traces(at) for s in _values(t["marker"]["size"])}) == 1
    # And carry no names, or all of them.
    slider = at.slider(key="reco_efficiency_labelled")
    assert slider.max == len(names)
    slider.set_value(0).run()
    assert not any(name for t in _traces(at) for name in t["text"])
    at.slider(key="reco_efficiency_labelled").set_value(len(names)).run()
    assert all(name for t in _traces(at) for name in t["text"])


def _efficiency_history_app(dashboard_dir, run_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from config import Config
    from data import RunData
    from k4bench.analysis.trend import build_results_trend
    from reco import efficiency
    from section_dispatch import PageContext
    from sidebar import TrendWindow

    results = build_results_trend((run_dir,))
    ctx = PageContext(
        config=Config(run_dir, "https://store.example", "/tmp", "https://dash.example"),
        selection=None, data_dir=run_dir, run_meta=None,
        run=RunData(None, None, None, results, sorted(results["label"].unique())),
        window=TrendWindow(None, (run_dir,)), trend_machine=None, reliability={},
    )
    efficiency.render(ctx)


def test_efficiency_history_draws_time_instructions_and_ipc_of_the_slowest():
    at = AppTest.from_function(
        _efficiency_history_app, args=(str(_DASHBOARD_DIR), str(RECO_RUN)), default_timeout=60,
    )
    at.run()
    at.radio(key="reco_efficiency_view").set_value("Historical Trends").run()
    assert not at.exception, at.exception
    lines = _traces(at)
    # Five algorithms, slowest first, in each of three panels.
    names = list(dict.fromkeys(t["name"] for t in lines))
    assert len(names) == 5 and names[0] == "MyConformalTracking"
    assert len(lines) == 15
    ipc = [t for t in lines if t["yaxis"] == "y3"]
    assert all(0 < y < 8 for t in ipc for y in _values(t["y"]))


def test_efficiency_says_why_a_run_has_no_instruction_counts(monkeypatch, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    for path in run.glob("*_components.json"):
        raw = json.loads(path.read_text())
        for table in (raw["execute"], *raw["lifecycle"].values()):
            table.pop("instructions"), table.pop("cycles")
        raw["hardware_counters_error"] = "perf_event_open: Function not implemented"
        path.write_text(json.dumps(raw))
    at = _page(monkeypatch, tmp_path, run, "Efficiency")
    assert "perf_event_open: Function not implemented" in at.info[0].value
    assert not at.get("plotly_chart")


def test_efficiency_tells_counted_runs_without_events_from_runs_without_counts(
    monkeypatch, tmp_path,
):
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    for path in run.glob("*_components.json"):
        raw = json.loads(path.read_text())
        # Only the warm-up event, which the tab leaves out.
        raw["event_numbers"] = raw["event_numbers"][:1]
        raw["execute"] = {metric: rows[:1] for metric, rows in raw["execute"].items()}
        path.write_text(json.dumps(raw))
    at = _page(monkeypatch, tmp_path, run, "Efficiency")
    assert "after the warm-up event" in at.info[0].value
    assert "did not count" not in at.info[0].value
    assert not at.get("plotly_chart")


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


def _event_timing_history_app(dashboard_dir, run_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from config import Config
    from data import RunData, cached_load_component_timing
    from k4bench.analysis.trend import build_results_trend
    from reco import page
    from section_dispatch import PageContext
    from sidebar import TrendWindow

    results = build_results_trend((run_dir,))
    ctx = PageContext(
        config=Config(run_dir, "https://store.example", "/tmp", "https://dash.example"),
        selection=None, data_dir=run_dir, run_meta=None,
        run=RunData(None, None, None, results, sorted(results["label"].unique()),
                    components=cached_load_component_timing(run_dir)),
        window=TrendWindow(None, (run_dir,)), trend_machine=None, reliability={},
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
        None, event, True, {"r0": True, "r1": True, "r2": False},
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
        None, None, trends_enabled, {},
        components=cached_load_component_timing(run_dir),
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

    components = cached_load_component_timing(run_dir)
    # The variant's VertexFinder ran in the warm-up event alone.
    wall = components["variant_truth_tracking"].execute["wall_s"]
    wall.loc[wall.index != 0, "VertexFinder"] = float("nan")
    event_timing.render(cached_load_event_timing(run_dir), None, False, {}, components=components)


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
    # Only the baseline ran its memory pass.
    assert at.selectbox(key="reco_memory_config").options == ["baseline"]
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Before the first event"] == "596 MB"
    assert tiles["Growth per event"] == "+4.7 MB"
    assert [w.value for w in at.warning] == [
        "- **LumiCalReco** grows 2 MB per event (20 MB over the run); after finalize it "
        "still holds 20 MB more than after event 1.\n"
        "- **VertexFinder** grows 1.2 MB per event (12 MB over the run); after finalize it "
        "still holds 12 MB more than after event 1."
    ]
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


def test_a_warned_component_opens_in_the_components_view(monkeypatch, tmp_path, reco_run_with_heap):
    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), "Memory")
    assert [b.label for b in at.button if (b.key or "").startswith("reco_memory_inspect_")] == [
        "LumiCalReco", "VertexFinder",
    ]
    # Not the one Components opens on by itself: the fastest-growing one.
    at.button(key="reco_memory_inspect_VertexFinder").click().run()
    assert not at.exception
    assert at.radio(key="reco_memory_view").value == "Components"
    assert "##### VertexFinder, event by event" in [m.value for m in at.markdown]
    table = at.dataframe[0]
    assert table.value.iloc[json.loads(table.proto.selection_default)["selection"]["rows"][0]][
        "Component"
    ] == "VertexFinder"


def test_a_link_opens_the_view_it_names(monkeypatch, tmp_path, reco_run_with_heap):
    _local_env(monkeypatch, reco_run_with_heap(), tmp_path / "cache")
    at = AppTest.from_function(_page_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params["tab"] = "Memory"
    at.query_params["reco_memory_view"] = "Components"
    at.run()
    assert not at.exception, at.exception
    assert at.radio(key="reco_memory_view").value == "Components"
    assert "##### Components" in [m.value for m in at.markdown]


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


def test_memory_lists_the_components_and_draws_the_suspect_event_by_event(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    at = _page(monkeypatch, tmp_path, reco_run_with_heap(), "Memory")
    at.radio(key="reco_memory_view").set_value("Components").run()
    assert not at.exception
    table = at.dataframe[0].value
    assert table["Component"].tolist()[:2] == ["GeoSvc", "k4FWCore__Reader"]
    flagged = table[table["Flag"] != ""]["Component"].tolist()
    assert flagged == ["LumiCalReco", "VertexFinder"]
    reader = table[table["Component"] == "k4FWCore__Reader"].iloc[0]
    assert reader["Working set"] == pytest.approx(5.0)
    assert len(reader["Held per event"]) == 12
    assert "##### LumiCalReco, event by event" in [m.value for m in at.markdown]


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
    at.radio(key="reco_memory_view").set_value("Components").run()
    assert not at.exception
    assert len(at.dataframe[0].value)


def test_a_component_that_needed_memory_in_the_warm_up_alone_is_listed(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    def spike(raw: dict) -> None:
        # MyTrackChecker allocates and frees 500 MB in event 0 and holds nothing else.
        raw["execute"]["peak_held_bytes"][0][raw["components"].index("MyTrackChecker")] = 500 * _MIB

    run = reco_run_with_heap()
    _rewrite_heap(run, spike)
    at = _page(monkeypatch, tmp_path, run, "Memory")
    at.radio(key="reco_memory_view").set_value("Components").run()
    assert not at.exception
    components = at.dataframe[0].value["Component"].tolist()
    assert "MyTrackChecker" in components
    # Components that never held anything stay out.
    assert "MyCLICPfoSelectorTight" not in components


def test_a_selection_past_the_tables_end_draws_the_default(
    monkeypatch, tmp_path, reco_run_with_heap,
):
    from reco.memory import _table_key

    run = reco_run_with_heap()
    at = _page(monkeypatch, tmp_path, run, "Memory")
    at.radio(key="reco_memory_view").set_value("Components").run()
    at.session_state[_table_key(str(run), "baseline")] = {
        "selection": {"rows": [99], "columns": [], "cells": []},
    }
    at.run()
    assert not at.exception
    assert "##### LumiCalReco, event by event" in [m.value for m in at.markdown]


def test_a_selection_stays_with_its_run(monkeypatch, tmp_path, reco_run_with_heap):
    from reco.memory import _table_key

    first, second = reco_run_with_heap("first"), reco_run_with_heap("second")
    at = _page(monkeypatch, tmp_path, first, "Memory")
    at.radio(key="reco_memory_view").set_value("Components").run()
    writer = at.dataframe[0].value["Component"].tolist().index("k4FWCore__Writer")
    at.session_state[_table_key(str(first), "baseline")] = {
        "selection": {"rows": [writer], "columns": [], "cells": []},
    }
    at.run()
    assert "##### k4FWCore__Writer, event by event" in [m.value for m in at.markdown]
    # The same configuration of another run opens on its own default.
    at.text_input(key="reco_data_dir").set_value(str(second)).run()
    assert not at.exception
    assert "##### LumiCalReco, event by event" in [m.value for m in at.markdown]


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
    from reco.memory import _SPARK_POINTS, _spark

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
    assert at.selectbox(key="reco_memory_config").options == ["baseline"]
    assert [w.value for w in at.warning][0].startswith(
        "The heap file of **variant_truth_tracking** could not be read, so it is not offered:"
    )
    assert any(m.label == "Before the first event" for m in at.metric)


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
    from k4bench.analysis.trend import build_results_trend
    from reco import memory
    from section_dispatch import PageContext
    from sidebar import TrendWindow

    results = build_results_trend(run_dirs)
    ctx = PageContext(
        config=Config(run_dirs[-1], "https://store.example", "/tmp", "https://dash.example"),
        selection=None, data_dir=run_dirs[-1], run_meta=None,
        run=RunData(None, None, None, results, sorted(results["label"].unique())),
        window=TrendWindow(None, run_dirs), trend_machine=None, reliability={},
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
    from k4bench.analysis.plots.heap import _NO_COMPONENT_LINE, _OTHER_LINE

    at = _memory_history(_nights(reco_run_with_heap))
    assert at.segmented_control(key="reco_memory_hist_metric").value == "Typical event"
    assert "##### Heap at an event's end, night by night" in [m.value for m in at.markdown]
    # Lines only.
    with pytest.raises(KeyError):
        at.segmented_control(key="reco_memory_hist_chart_type")
    at.slider(key="reco_memory_hist_topn").set_value(2).run()
    traces = _traces(at)
    assert {t["type"] for t in traces} == {"scatter"}
    # The suspects, the fastest-growing first, then the two largest; the rest
    # and what no component allocated last, in grey.
    assert _line_colors(at) == {
        "LumiCalReco": PALETTE[0], "VertexFinder": PALETTE[1], "GeoSvc": PALETTE[2],
        "k4FWCore__Reader": PALETTE[3], "Other (1)": _OTHER_LINE, "No component": _NO_COMPONENT_LINE,
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

# ── The Overview, from runs ──────────────────────────────────────────────────

def _results(run_ids, wall, returncode=0):
    return pd.DataFrame({
        "run_id": run_ids,
        "run_date": pd.to_datetime(run_ids),
        "k4h_release": [f"key4hep-{r}" for r in run_ids],
        "label": "baseline",
        "returncode": returncode,
        "wall_time_s": wall,
        "peak_vmem_mb": [3441.0] * len(run_ids),
    })


def _events(run_ids, mean):
    return pd.DataFrame({
        "run_id": run_ids,
        "run_date": pd.to_datetime(run_ids),
        "k4h_release": [f"key4hep-{r}" for r in run_ids],
        "label": "baseline",
        "mean_time_s": mean,
        "mean_rss_anon_mb": [760.0] * len(run_ids),
    })


def test_run_history_rows_carry_no_verdict_but_mark_failures():
    days = ["2026-10-13", "2026-10-14"]
    results = pd.concat([
        _results(days, [480.0, 660.0]),
        _results(["2026-10-15"], [12.0], returncode=1),
    ], ignore_index=True)
    rows = _overview_data.run_history_rows(
        "CLD_o2_v09", results, _events(days, [4.8, 6.6]), {"2026-10-14": False},
    )
    assert set(rows["metric"]) == {
        "wall_time_s", "peak_vmem_mb", "mean_time_s", "mean_rss_anon_mb",
    }
    wall = rows[rows["metric"] == "wall_time_s"].set_index("run_night")
    assert wall.loc["2026-10-13", "severity"] == Severity.UNKNOWN.value
    assert wall.loc["2026-10-15", "severity"] == Severity.FAILURE.value
    assert wall.loc["2026-10-14", "reliable"] is False
    assert wall.loc["2026-10-14", "night"] == "2026-10-14"


def test_run_history_rows_drop_events_no_result_backs():
    # The 10-14 baseline crashed before writing its results, but left a
    # partial events file; a failed run that did write them is kept.
    results = pd.concat([
        _results(["2026-10-13"], [480.0]),
        _results(["2026-10-15"], [12.0], returncode=1),
    ], ignore_index=True)
    events = _events(["2026-10-13", "2026-10-14", "2026-10-15"], [4.8, 0.1, 0.2])
    rows = _overview_data.run_history_rows("CLD_o2_v09", results, events, {})
    mean = rows[rows["metric"] == "mean_time_s"].set_index("run_night")
    assert sorted(mean.index) == ["2026-10-13", "2026-10-15"]
    assert mean.loc["2026-10-15", "severity"] == Severity.FAILURE.value


def test_run_overview_inputs_cover_every_detector_with_runs():
    days = ["2026-10-16", "2026-10-17"]
    hist = pd.concat([
        _overview_data.run_history_rows(d, _results(days, [480.0, 481.0]),
                                        _events(days, [4.8, 4.81]), {})
        for d in ("CLD_o2_v09", "ILD_l5_v11")
    ], ignore_index=True)
    rel = pd.concat([
        _overview_data.run_reliability_rows(d, _results(days, [480.0, 481.0]), {})
        for d in ("CLD_o2_v09", "ILD_l5_v11")
    ], ignore_index=True)
    inputs = _overview_data.run_overview_inputs(hist, rel, platform="p", sample="s")
    assert inputs.detectors == ["CLD_o2_v09", "ILD_l5_v11"]
    assert inputs.latest_night == "2026-10-17"
    assert inputs.last_run == {"CLD_o2_v09": "2026-10-17", "ILD_l5_v11": "2026-10-17"}
    assert not inputs.scope_empty
    wide, as_of = _overview_data.latest_snapshot(inputs.snap_rows)
    assert wide.loc["CLD_o2_v09", "wall_time_s"] == 481.0
    assert as_of["CLD_o2_v09"] == "2026-10-17"


def _figures_app(dashboard_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import pandas as _pd

    from tabs import _overview_data as data
    from tabs import detectors_overview

    days = ["2026-10-16", "2026-10-17"]
    results = _pd.DataFrame({
        "run_id": days, "run_date": _pd.to_datetime(days),
        "k4h_release": [f"key4hep-{d}" for d in days], "label": "baseline",
        "returncode": 0, "wall_time_s": [480.0, 481.0], "peak_vmem_mb": [3441.0, 3442.0],
    })
    events = results[["run_id", "run_date", "k4h_release", "label"]].assign(
        mean_time_s=[4.8, 4.81], mean_rss_anon_mb=[760.0, 761.0],
    )
    detectors_overview.render_figures(
        data.run_overview_inputs(
            data.run_history_rows("CLD_o2_v09", results, events, {}),
            data.run_reliability_rows("CLD_o2_v09", results, {}),
            platform="x86_64-el9-gcc16-opt", sample="p8_ee_Zbb_ecm91",
        ),
        key_prefix="reco_",
    )


def _overview_app(dashboard_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from datetime import date
    from types import SimpleNamespace

    from reco import overview

    overview.render(SimpleNamespace(
        config=SimpleNamespace(data_url="https://store.example", cache_dir="/tmp"),
        remote=SimpleNamespace(platform="x86_64-el9-gcc16-opt", sample="p8_ee_Zbb_ecm91"),
        window=SimpleNamespace(dates=(date(2026, 10, 1), date(2026, 10, 17))),
    ))


def test_the_overview_names_a_detector_whose_runs_could_not_be_listed(monkeypatch):
    import requests
    from reco import overview

    def list_run_dates(base_url, suite, detector, platform, sample):
        if detector == "ILD_l5_v11":
            raise requests.ConnectionError("timed out")
        # Never run on this platform: an answer, not a failure.
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError("404", response=response)

    monkeypatch.setattr(
        overview, "cached_list_detectors", lambda base_url, suite: ["CLD_o2_v09", "ILD_l5_v11"],
    )
    monkeypatch.setattr(overview, "cached_list_run_dates", list_run_dates)
    at = AppTest.from_function(_overview_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    (warning,) = at.warning
    assert "**ILD_l5_v11**" in warning.value
    assert "CLD_o2_v09" not in warning.value


def test_the_overview_draws_the_simulation_figures_without_regression_flags():
    at = AppTest.from_function(_figures_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.radio(key="reco_det_ov_view_mode").options == [
        "Performance Trends", "Performance Landscape",
    ]
    traces = _traces(at)
    # Legend entries are grouped by detector family, each trace by its detector.
    assert {t.get("legendgroup") for t in traces} >= {"CLD_o2_v09"}
    # Unjudged values: no Confirmed/Watch pills to toggle.
    assert not [p for p in at.get("button_group") if "Regressions" in str(p.proto)]
    at.radio(key="reco_det_ov_view_mode").set_value("Performance Landscape").run()
    assert not at.exception, at.exception
    assert at.get("plotly_chart")


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

    from reco.dataflow import _seconds_per_event

    wall = pd.DataFrame({"Timed": [1.0, np.nan], "Untimed": [np.nan, np.nan]})
    timing = SimpleNamespace(execute={"wall_s": wall})
    ctx = SimpleNamespace(run=SimpleNamespace(components={"baseline": timing}))
    assert _seconds_per_event(ctx, "baseline") == {"Timed": 0.5}


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
