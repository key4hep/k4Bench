"""A copied link reopens the same measurement: the Reconstruction page in
remote mode, driven through ``AppTest`` against a stubbed run store whose
releases are re-run on later nights.

The first visit pins the run it opened on, a link restores the run, the
section, its view and the trend window, moving between sections keeps the run,
and a history draws each release's latest run or every run as the link says.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

#: A nightly CLD reconstruction run (see its README).
RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"

DET, PLAT, SAMPLE = "CLD_o2_v09", "x86_64-el9-gcc16-opt", "p8_ee_Zbb_ecm91"
OLD, NEW = "key4hep-2026-10-15", "key4hep-2026-10-17"

#: {stack: [run dates]}: the 10-15 release re-run on 10-17, the night the
#: 10-17 release was first measured, which was itself re-run twice.
RUNS = {
    OLD: ["2026-10-15", "2026-10-17"],
    NEW: ["2026-10-17", "2026-10-18", "2026-10-19"],
}


@pytest.fixture
def store(tmp_path, monkeypatch) -> dict[str, str]:
    """``{"stack/date": run dir}``: a copy of the fixture run for every run of
    :data:`RUNS`, its ``run_info.json`` naming its own release and night."""
    dirs = {}
    for stack, days in RUNS.items():
        for day in days:
            run_dir = tmp_path / "cache" / stack / day
            shutil.copytree(RECO_RUN, run_dir)
            info = json.loads((run_dir / "run_info.json").read_text())
            info.update(date=day, k4h_release=stack,
                        k4h_release_date=stack.removeprefix("key4hep-"))
            (run_dir / "run_info.json").write_text(json.dumps(info))
            dirs[f"{stack}/{day}"] = str(run_dir)
    monkeypatch.setenv("K4BENCH_DATA_URL", "https://store.invalid")
    monkeypatch.setenv("K4BENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("K4BENCH_DATA_DIR", raising=False)
    return dirs


def _remote_reco_app(dashboard_dir, dirs, runs, det, plat, sample) -> None:
    # No annotations: AppTest.from_function execs this body as a bare script.
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    import sidebar
    from reco import page

    # A test lands a run mid-session through "new_runs".
    listed = {
        stack: sorted([*days, *st.session_state.get("new_runs", {}).get(stack, [])])
        for stack, days in runs.items()
    }
    sidebar.cached_list_detectors = lambda url, suite: [det]
    sidebar.cached_list_platforms = lambda url, suite, d: [plat]
    sidebar.cached_scan_stack_samples = lambda url, suite, d, p: {s: [sample] for s in runs}
    sidebar.cached_list_runs = (
        lambda url, suite, d, p, stack, s: sorted(listed[stack], reverse=True)
    )
    sidebar.cached_fetch_run = lambda url, suite, d, p, stack, s, day, cache: dirs[f"{stack}/{day}"]
    sidebar.read_again = lambda cached, *args: cached(*args)
    sidebar.cached_list_run_dates = lambda url, suite, d, p, s: listed
    sidebar.list_platform_stacks = lambda url, suite, p: sorted(runs, reverse=True)
    sidebar.cached_fetch_runs_windowed = lambda url, suite, d, p, s, cache, items: tuple(
        dirs[f"{stack}/{day}"] for stack, days in items for day in days
        if f"{stack}/{day}" in dirs
    )
    page.main()


def _open(store, params) -> AppTest:
    at = AppTest.from_function(
        _remote_reco_app, args=(str(_DASHBOARD_DIR), store, RUNS, DET, PLAT, SAMPLE),
        default_timeout=120,
    )
    for key, value in params.items():
        at.query_params[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _query(at: AppTest) -> dict[str, str]:
    values = {k: at.query_params[k] for k in at.query_params.keys()}
    return {k: v[0] if isinstance(v, list) else v for k, v in values.items()}


def _scope_note(at: AppTest) -> str:
    return next(c.value for c in at.caption if c.value.startswith("Showing "))


def test_a_first_visit_pins_the_run_it_opened_on(store):
    at = _open(store, {"tab": "Logs"})
    query = _query(at)
    assert (query["stack"], query["run"]) == (NEW, "2026-10-19")
    assert query["releases"] == f"{OLD}..{NEW}"
    assert _scope_note(at).endswith("Key4hep 2026-10-17, measured 2026-10-19")
    # Run B lands: the session keeps showing run A, as its link does.
    at.session_state["new_runs"] = {NEW: ["2026-10-20"]}
    at.run()
    assert _query(at)["run"] == "2026-10-19"
    assert at.selectbox(key="reco_sb_run").options[0] == "2026-10-20 · latest"


def test_a_link_restores_the_run_section_view_and_window(store):
    at = _open(store, {
        "tab": "Breakdown", "reco_breakdown_view": "Historical Trends",
        "stack": OLD, "run": "2026-10-15", "releases": f"{OLD}..{OLD}",
    })
    assert at.session_state["reco_active_section"] == "Breakdown"
    assert at.radio(key="reco_breakdown_view").value == "Historical Trends"
    assert at.selectbox(key="reco_sb_stack").value == OLD
    assert at.selectbox(key="reco_sb_run").value == "2026-10-15"
    assert at.select_slider(key="reco_trend_releases").value == (OLD, OLD)
    query = _query(at)
    assert (query["stack"], query["run"], query["releases"]) == (OLD, "2026-10-15", f"{OLD}..{OLD}")


def test_moving_between_sections_keeps_the_run(store):
    at = _open(store, {"tab": "Logs", "stack": OLD, "run": "2026-10-17"})
    for section in ("Efficiency", "Breakdown", "Logs"):
        at.button_group[0].set_value(section).run()
        assert not at.exception, at.exception
        assert _query(at)["tab"] == section
        assert (_query(at)["stack"], _query(at)["run"]) == (OLD, "2026-10-17")
        assert at.selectbox(key="reco_sb_run").value == "2026-10-17"


def test_the_history_mode_is_offered_only_where_a_section_honours_it(store):
    at = _open(store, {"tab": "Logs", "history": "all"})
    assert not [r for r in at.radio if r.key == "reco_trend_history"]
    at.button_group[0].set_value("Breakdown").run()
    assert at.radio(key="reco_trend_history").value == "all"


def _breakdown_columns(at: AppTest) -> list[str]:
    """The columns of the Breakdown's history chart, one per drawn run."""
    chart = next(
        c for c in at.get("plotly_chart")
        if json.loads(c.proto.spec)["layout"].get("barmode") == "stack"
    )
    bars = [t for t in json.loads(chart.proto.spec)["data"] if t["type"] == "bar"]
    return list(bars[0]["x"])


def test_a_history_draws_the_latest_run_per_release_or_every_run(store):
    params = {"tab": "Breakdown", "reco_breakdown_view": "Historical Trends"}
    latest = _open(store, params)
    assert len(_breakdown_columns(latest)) == 2
    every = _open(store, params | {"history": "all"})
    columns = _breakdown_columns(every)
    # Five runs, two of them measured on 10-17 with different releases.
    assert columns == [
        "2026-10-15 · Key4hep 2026-10-15",
        "2026-10-17 · Key4hep 2026-10-15",
        "2026-10-17 · Key4hep 2026-10-17",
        "2026-10-18 · Key4hep 2026-10-17",
        "2026-10-19 · Key4hep 2026-10-17",
    ]


def test_every_run_keeps_runs_of_one_night_apart_on_a_line_too(store):
    at = _open(store, {
        "tab": "Breakdown", "reco_breakdown_view": "Historical Trends",
        "reco_breakdown_hist_chart_type": "Lines", "history": "all",
    })
    chart = json.loads(at.get("plotly_chart")[0].proto.spec)
    line = next(t for t in chart["data"] if t["type"] == "scatter")
    # 10-17 is two runs, of two releases: two places on the axis.
    assert line["x"] == [
        "2026-10-15 · Key4hep 2026-10-15",
        "2026-10-17 · Key4hep 2026-10-15",
        "2026-10-17 · Key4hep 2026-10-17",
        "2026-10-18 · Key4hep 2026-10-17",
        "2026-10-19 · Key4hep 2026-10-17",
    ]
    axis = chart["layout"]["xaxis"]
    assert axis["type"] == "category"
    assert axis["ticktext"] == [
        "2026-10-15", "2026-10-17", "2026-10-17", "2026-10-18", "2026-10-19",
    ]


def test_a_stale_run_link_on_the_page_offers_the_latest_and_draws_nothing_else(store):
    at = _open(store, {"tab": "Logs", "stack": NEW, "run": "2026-10-01"})
    assert "The linked run `2026-10-01` is not in the store" in at.warning[0].value
    assert not at.button_group
    next(b for b in at.button if b.label.startswith("Open the latest run")).click().run()
    assert _query(at)["run"] == "2026-10-19"
    assert at.button_group
