"""The sidebar's remote-mode pickers, driven through ``AppTest`` against a stubbed
run store: which run the Detector → Platform → Sample → Stack chain lands on,
how it follows and writes the URL's deep link, where it stops, and what the
Trend window covers.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import sidebar  # noqa: E402

PLAT = "x86_64-el9-gcc16-opt"
NEW, OLD = "key4hep-2026-10-10", "key4hep-2026-10-09"

#: {detector: {stack: [samples]}} — CLD has no ``p8_ee_Zbb_ecm91``.
STORE = {
    "ALLEGRO_o1_v04": {NEW: ["p8_ee_Zbb_ecm91", "single_e-_10GeV"], OLD: ["single_e-_10GeV"]},
    "CLD_o2_v09": {NEW: ["single_e-_10GeV"], OLD: ["single_e-_10GeV"]},
}


def _pick_run_app(dashboard_dir, store, plat, fail) -> None:
    # No annotations: AppTest.from_function execs this body as a bare script.
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    import sidebar

    def _list_detectors(url):
        if fail == "detectors":
            raise OSError("listing timed out")
        return sorted(store)

    sidebar.cached_list_detectors = _list_detectors
    sidebar.cached_list_platforms = lambda url, det: [plat]
    sidebar.cached_scan_stack_samples = lambda url, det, p: store[det]
    sidebar.cached_fetch_latest_run = (
        lambda url, det, p, stack, sample, cache: None if fail == "run"
        else f"/cache/{det}/{stack}/{sample}"
    )
    st.session_state["picked"] = sidebar.pick_run("https://store.invalid", "/cache")


def _run_pick(params=None, fail=None) -> AppTest:
    at = AppTest.from_function(
        _pick_run_app, args=(str(_DASHBOARD_DIR), STORE, PLAT, fail), default_timeout=30
    )
    for key, value in (params or {}).items():
        at.query_params[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _query(at: AppTest) -> dict[str, str]:
    return {k: at.query_params[k][0] for k in at.query_params.keys()}


def test_pick_run_follows_the_deep_link_and_defaults_to_the_newest_release():
    at = _run_pick({"detector": "CLD_o2_v09", "sample": "single_e-_10GeV"})
    picked = at.session_state["picked"]
    assert (picked.detector, picked.platform, picked.sample, picked.stack) == (
        "CLD_o2_v09", PLAT, "single_e-_10GeV", NEW,
    )
    assert picked.data_dir == f"/cache/CLD_o2_v09/{NEW}/single_e-_10GeV"
    # The address bar is written back to what is on screen.
    assert _query(at) == {
        "detector": "CLD_o2_v09", "platform": PLAT,
        "sample": "single_e-_10GeV", "stack": NEW,
    }
    assert "Available in 2 release(s)" in at.caption[0].value


def test_pick_run_offers_only_the_releases_that_hold_the_sample():
    at = _run_pick({"detector": "ALLEGRO_o1_v04", "sample": "p8_ee_Zbb_ecm91"})
    assert at.selectbox(key="sb_stack").options == [NEW]
    assert at.session_state["picked"].stack == NEW


def test_pick_run_names_a_sample_that_switching_detector_dropped():
    at = _run_pick({"detector": "ALLEGRO_o1_v04", "sample": "p8_ee_Zbb_ecm91"})
    at.selectbox(key="sb_detector").set_value("CLD_o2_v09").run()
    assert at.session_state["picked"].sample == "single_e-_10GeV"
    notes = [c.value for c in at.caption]
    assert any(
        "`p8_ee_Zbb_ecm91` isn't benchmarked for CLD_o2_v09 — showing `single_e-_10GeV`" in n
        for n in notes
    ), notes


def test_pick_run_stops_where_the_store_cannot_be_read():
    at = _run_pick(fail="detectors")
    assert at.session_state["picked"] is None
    assert "Failed to list detectors: listing timed out" in at.error[0].value
    assert not at.selectbox


def test_pick_run_stops_when_the_release_has_no_run():
    at = _run_pick(fail="run")
    assert at.session_state["picked"] is None
    assert at.warning[0].value == "No runs found for the selected combination."


def _window_app(dashboard_dir, run_dates, report_nights) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    import sidebar

    sidebar.cached_list_run_dates = lambda url, det, plat, sample: run_dates
    sidebar.cached_list_report_dates = lambda url: report_nights
    sidebar.cached_fetch_runs_windowed = (
        lambda url, det, plat, sample, cache, items: tuple(
            f"/cache/{stack}/{d}" for stack, dates in items for d in dates
        )
    )
    selection = sidebar.RunSelection(
        "CLD_o2_v09", "PLAT", "single_e-_10GeV", "key4hep-2026-10-10", "/cache/run"
    )
    st.session_state["window"] = sidebar.pick_trend_window(
        "https://store.invalid", selection, "/cache"
    )


def _run_window(run_dates, report_nights, params=None) -> AppTest:
    at = AppTest.from_function(
        _window_app, args=(str(_DASHBOARD_DIR), run_dates, report_nights),
        default_timeout=30,
    )
    for key, value in (params or {}).items():
        at.query_params[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def test_trend_window_opens_on_the_last_14_report_nights():
    # The 10-12 run was uploaded ahead of its report: the window reaches it,
    # but still counts its 14 days back from the newest report night.
    run_dates = {NEW: ["2026-10-12", "2026-10-10"], OLD: ["2026-09-20", "2026-10-01"]}
    at = _run_window(run_dates, ["2026-10-10", "2026-10-11"])
    window = at.session_state["window"]
    assert window.dates == (date(2026, 9, 28), date(2026, 10, 12))
    assert window.run_dirs == (
        f"/cache/{OLD}/2026-10-01", f"/cache/{NEW}/2026-10-10", f"/cache/{NEW}/2026-10-12",
    )
    assert at.selectbox(key="sb_trend_preset").value == "Last 14 days"
    assert _query(at) == {"range": "Last 14 days"}
    assert at.caption[0].value == (
        "2026-09-28 → 2026-10-12 · 3 run(s) across 2 release(s)"
    )


def test_trend_window_names_the_last_run_of_a_detector_outside_it():
    at = _run_window({OLD: ["2026-08-01"]}, ["2026-10-11"])
    assert at.session_state["window"].run_dirs == ()
    assert at.caption[0].value.endswith("· CLD_o2_v09 last ran 2026-08-01")


def test_trend_window_is_absent_until_something_has_run():
    at = _run_window({}, [])
    assert at.session_state["window"] == sidebar.TrendWindow()
    assert not at.header and not at.selectbox


def test_custom_range_waits_for_both_dates():
    at = _run_window({NEW: ["2026-10-11"]}, ["2026-10-11"], {"range": "Custom…"})
    assert at.session_state["window"].dates is not None
    at.date_input[0].set_value((date(2026, 10, 11),)).run()
    assert at.session_state["window"] == sidebar.TrendWindow()
    assert at.info[0].value == "Pick both a start and end date."
