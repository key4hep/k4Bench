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
        "Breakdown", "Data Flow", "Event Timing", "Event Memory", "Machine Info", "Logs",
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


def test_the_component_table_links_each_class_to_its_package(monkeypatch, tmp_path):
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
