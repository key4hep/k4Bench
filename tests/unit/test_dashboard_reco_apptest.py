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
        "Breakdown", "Event Timing", "Event Memory", "Machine Info", "Logs",
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
    assert not at.radio


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
