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

    def _list_detectors(url, suite):
        if fail == "detectors":
            raise OSError("listing timed out")
        return sorted(store)

    sidebar.cached_list_detectors = _list_detectors
    sidebar.cached_list_platforms = lambda url, suite, det: [plat]
    sidebar.cached_scan_stack_samples = lambda url, suite, det, p: store[det]
    sidebar.cached_fetch_latest_run = (
        lambda url, suite, det, p, stack, sample, cache: None if fail == "run"
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


def _window_app(dashboard_dir, run_dates, report_nights, platform_stacks, across) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    import sidebar

    def _platform_stacks(url, suite, plat):
        if platform_stacks is None:
            raise OSError("listing timed out")
        # A test publishes a release mid-session through "new_stacks".
        return [*st.session_state.get("new_stacks", []), *platform_stacks]

    sidebar.cached_list_run_dates = lambda url, suite, det, plat, sample: run_dates
    # A test lands a report mid-session through "new_nights".
    sidebar.cached_list_report_dates = (
        lambda url, suite: [*report_nights, *st.session_state.get("new_nights", [])]
    )
    sidebar.list_platform_stacks = _platform_stacks
    sidebar.cached_fetch_runs_windowed = (
        lambda url, suite, det, plat, sample, cache, items: tuple(
            f"/cache/{stack}/{d}" for stack, dates in items for d in dates
        )
    )
    selection = sidebar.RunSelection(
        st.session_state.get("detector", "CLD_o2_v09"),
        "PLAT", "single_e-_10GeV", "key4hep-2026-10-10", "/cache/run",
    )
    # A test changes section mid-session through "across".
    st.session_state["window"] = sidebar.pick_trend_window(
        "https://store.invalid", selection, "/cache",
        st.session_state.get("across", across),
    )


def _rel(day: str) -> str:
    return f"key4hep-2026-{day}"


#: Every release on the platform, newest first as the listing returns them:
#: one every other day, the last one ahead of its report.
PLATFORM_STACKS = [_rel(d) for d in (
    "10-12", "10-10", "10-08", "10-06", "10-04", "10-02",
    "09-30", "09-28", "09-26", "09-24", "09-22", "09-20",
)]
REPORT_NIGHTS = ["2026-10-10", "2026-10-11"]

#: A detector benchmarked for the three newest releases only, as after a
#: detector version bump.
NEW_DETECTOR_RUNS = {
    _rel("10-08"): ["2026-10-08"],
    _rel("10-10"): ["2026-10-10", "2026-10-11"],
    _rel("10-12"): ["2026-10-12"],
}

#: A detector with a run in every other release from 09-24 on.
LONG_RUNS = {_rel(d): [f"2026-{d}"] for d in ("09-24", "09-26", "09-28", "10-10")}


def _run_window(
    run_dates, report_nights=REPORT_NIGHTS, platform_stacks=PLATFORM_STACKS,
    across=True,
) -> AppTest:
    at = AppTest.from_function(
        _window_app,
        args=(str(_DASHBOARD_DIR), run_dates, report_nights, platform_stacks, across),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    return at


def test_across_detectors_the_window_opens_on_the_platforms_last_10_releases():
    at = _run_window(NEW_DETECTOR_RUNS)
    window = at.session_state["window"]
    # 10-12 was uploaded ahead of its report, so the Overview has nothing of
    # it to draw yet: it is not offered, even though this detector ran it.
    assert at.select_slider[0].options == [
        p.removeprefix("key4hep-") for p in reversed(PLATFORM_STACKS[1:])
    ]
    assert at.select_slider[0].value == (_rel("09-22"), _rel("10-10"))
    # The 10-10 release was re-run on the newest report night.
    assert window.dates == (date(2026, 9, 22), date(2026, 10, 11))
    # The window is not part of the deep link.
    assert _query(at) == {}
    # The selected detector's runs are no part of what the Overview draws.
    assert at.caption[0].value == "2026-09-22 → 2026-10-10 · 10 release(s)"


def test_for_one_detector_the_window_ranges_over_its_own_releases():
    # The new detector's three releases are the whole slider, so no range can
    # reach back past its history and show the same runs as a shorter one.
    # 10-12 is offered although no report has judged it: its run is plotted.
    at = _run_window(NEW_DETECTOR_RUNS, platform_stacks=None, across=False)
    slider = at.select_slider[0]
    assert slider.options == ["2026-10-08", "2026-10-10", "2026-10-12"]
    assert slider.value == (_rel("10-08"), _rel("10-12"))
    window = at.session_state["window"]
    assert window.run_dirs == (
        f"/cache/{_rel('10-08')}/2026-10-08",
        f"/cache/{_rel('10-10')}/2026-10-10",
        f"/cache/{_rel('10-10')}/2026-10-11",
        f"/cache/{_rel('10-12')}/2026-10-12",
    )
    assert at.caption[0].value == "2026-10-08 → 2026-10-12 · 3 release(s) · 4 run(s)"
    # The platform's releases are not listed at all: the failing stub would
    # have warned.
    assert not at.warning


def test_trend_window_is_absent_until_something_has_run():
    at = _run_window({}, [], [])
    assert at.session_state["window"] == sidebar.TrendWindow()
    assert not at.header and not at.select_slider


def test_trend_window_falls_back_to_the_detectors_releases():
    at = _run_window({_rel("10-10"): ["2026-10-10"]}, platform_stacks=None)
    assert "Could not list the platform's releases: listing timed out" in at.warning[0].value
    window = at.session_state["window"]
    assert window.dates == (date(2026, 10, 10), date(2026, 10, 11))
    assert window.run_dirs == (f"/cache/{_rel('10-10')}/2026-10-10",)


def test_a_picked_range_spans_the_releases_between_its_two_ends():
    at = _run_window(LONG_RUNS)
    at.select_slider[0].set_value((_rel("09-24"), _rel("09-28"))).run()
    window = at.session_state["window"]
    # The 09-28 release was benchmarked until 09-30 was tagged.
    assert window.dates == (date(2026, 9, 24), date(2026, 9, 29))
    assert window.run_dirs == tuple(
        f"/cache/{_rel(d)}/2026-{d}" for d in ("09-24", "09-26", "09-28")
    )


def test_across_detectors_a_picked_range_survives_switching_detector():
    at = _run_window(LONG_RUNS)
    at.select_slider[0].set_value((_rel("09-24"), _rel("09-28"))).run()
    at.session_state["detector"] = "ALLEGRO_o1_v04"
    at.run()
    assert at.select_slider[0].value == (_rel("09-24"), _rel("09-28"))
    assert at.session_state["window"].dates == (date(2026, 9, 24), date(2026, 9, 29))


def test_across_detectors_a_new_report_reopens_the_window_on_the_newest_releases():
    # A range picked yesterday would otherwise stop short of tonight's release.
    at = _run_window(LONG_RUNS)
    at.select_slider[0].set_value((_rel("09-24"), _rel("09-28"))).run()
    at.session_state["new_stacks"] = [_rel("10-14")]
    at.run()
    # Uploaded but not yet reported: nothing changes.
    assert at.select_slider[0].value == (_rel("09-24"), _rel("09-28"))
    at.session_state["new_nights"] = ["2026-10-14"]
    at.run()
    assert at.select_slider[0].value == (_rel("09-26"), _rel("10-14"))


def test_changing_scope_reopens_the_window_on_the_newest_releases():
    at = _run_window(LONG_RUNS)
    at.select_slider[0].set_value((_rel("09-24"), _rel("09-26"))).run()
    at.session_state["across"] = False
    at.run()
    assert at.select_slider[0].value == (_rel("09-24"), _rel("10-10"))
