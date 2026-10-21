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


#: {(detector, stack): [run dates]} — every release re-run on later nights;
#: the two releases share the night 2026-10-10.
RUNS = {
    ("ALLEGRO_o1_v04", NEW): ["2026-10-10", "2026-10-11"],
    ("ALLEGRO_o1_v04", OLD): ["2026-10-09", "2026-10-10"],
    ("CLD_o2_v09", NEW): ["2026-10-10", "2026-10-11"],
    ("CLD_o2_v09", OLD): ["2026-10-09"],
}


def _pick_run_app(dashboard_dir, store, runs, plat, fail, pick_kwargs) -> None:
    # No annotations: AppTest.from_function execs this body as a bare script.
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    import sidebar

    # Every suite the listings were asked for, in call order.
    st.session_state["suites"] = suites = []
    # A test changes how the store answers mid-session through "fail".
    failing = st.session_state.get("fail", fail)

    def _list_detectors(url, suite):
        suites.append(suite)
        if failing == "detectors":
            raise OSError("listing timed out")
        return sorted(store)

    def _list_platforms(url, suite, det):
        suites.append(suite)
        return [plat]

    def _scan_stack_samples(url, suite, det, p):
        suites.append(suite)
        return store[det]

    def _list_runs(url, suite, det, p, stack, sample):
        suites.append(suite)
        if failing == "runs":
            raise OSError("listing timed out")
        if failing == "no runs":
            return []
        # A test lands a run mid-session through "new_runs".
        new = st.session_state.get("new_runs", {}).get((det, stack), [])
        return sorted([*runs[(det, stack)], *new], reverse=True)

    def _fetch_run(url, suite, det, p, stack, sample, day, cache):
        suites.append(suite)
        return f"/cache/{suite}/{det}/{stack}/{sample}/{day}"

    sidebar.cached_list_detectors = _list_detectors
    sidebar.cached_list_platforms = _list_platforms
    sidebar.cached_scan_stack_samples = _scan_stack_samples
    sidebar.cached_list_runs = _list_runs
    sidebar.cached_fetch_run = _fetch_run
    # "away" stands for the reader being on another page: nothing is picked,
    # and the page switch cleared the URL the pickers mirror themselves in.
    if st.session_state.get("away"):
        st.query_params.clear()
        return
    sidebar.render_refresh_button()
    st.session_state["picked"] = sidebar.pick_run(
        "https://store.invalid", "/cache", **pick_kwargs
    )


def _run_pick(params=None, fail=None, **pick_kwargs) -> AppTest:
    at = AppTest.from_function(
        _pick_run_app, args=(str(_DASHBOARD_DIR), STORE, RUNS, PLAT, fail, pick_kwargs),
        default_timeout=30,
    )
    for key, value in (params or {}).items():
        at.query_params[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _query(at: AppTest) -> dict[str, str]:
    """The URL's parameters, normalising AppTest's list-or-scalar values."""
    values = {k: at.query_params[k] for k in at.query_params.keys()}
    return {k: v[0] if isinstance(v, list) else v for k, v in values.items()}


def _picked(at: AppTest) -> tuple[str, ...]:
    """The picked run's identity, every level of it."""
    ref = at.session_state["picked"].ref
    return (ref.suite, ref.detector, ref.platform, ref.stack, ref.sample, ref.date)


def _captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


def test_pick_run_follows_the_deep_link_and_defaults_to_the_newest_run():
    at = _run_pick({"detector": "CLD_o2_v09", "sample": "single_e-_10GeV"})
    picked = at.session_state["picked"]
    assert _picked(at) == ("sim", "CLD_o2_v09", PLAT, NEW, "single_e-_10GeV", "2026-10-11")
    assert picked.data_dir == f"/cache/sim/CLD_o2_v09/{NEW}/single_e-_10GeV/2026-10-11"
    # The address bar is written back to what is on screen, the run exactly.
    assert _query(at) == {
        "detector": "CLD_o2_v09", "platform": PLAT,
        "sample": "single_e-_10GeV", "stack": NEW, "run": "2026-10-11",
    }
    # The release and the night it was measured are named apart.
    assert at.caption[0].value == "Available in 2 releases"
    assert at.caption[1].value == "Measured **2026-10-11** · run 2 of 2"
    assert at.selectbox(key="sb_run").options == ["2026-10-11 · latest", "2026-10-10"]


def test_a_pinned_link_keeps_its_run_after_the_release_is_run_again():
    link = {"detector": "CLD_o2_v09", "sample": "single_e-_10GeV", "stack": NEW,
            "run": "2026-10-10"}
    at = _run_pick(link)
    assert _picked(at)[-1] == "2026-10-10"
    # Run B lands under the same release: the session and a fresh visit to
    # the same link both keep run A.
    at.session_state["new_runs"] = {("CLD_o2_v09", NEW): ["2026-10-12"]}
    at.button[0].click().run()  # Refresh Data
    assert _picked(at)[-1] == "2026-10-10"
    assert at.selectbox(key="sb_run").options[0] == "2026-10-12 · latest"
    assert _query(at)["run"] == "2026-10-10"
    fresh = AppTest.from_function(
        _pick_run_app, args=(str(_DASHBOARD_DIR), STORE, RUNS, PLAT, None, {}),
        default_timeout=30,
    )
    fresh.session_state["new_runs"] = {("CLD_o2_v09", NEW): ["2026-10-12"]}
    for key, value in link.items():
        fresh.query_params[key] = value
    fresh.run()
    assert _picked(fresh)[-1] == "2026-10-10"


def test_a_fresh_visit_pins_the_latest_run_it_opened_on():
    at = _run_pick({"detector": "CLD_o2_v09"})
    assert _picked(at)[-1] == "2026-10-11"
    at.session_state["new_runs"] = {("CLD_o2_v09", NEW): ["2026-10-12"]}
    at.run()
    assert _picked(at)[-1] == "2026-10-11"
    assert _query(at)["run"] == "2026-10-11"


def test_runs_of_one_night_stay_apart_across_releases_and_detectors():
    at = _run_pick({"detector": "ALLEGRO_o1_v04", "sample": "single_e-_10GeV",
                    "stack": NEW, "run": "2026-10-10"})
    first = at.session_state["picked"]
    # The same night, measured with the older release: kept, as a valid choice
    # one level down is, but a different run.
    at.selectbox(key="sb_stack").set_value(OLD).run()
    second = at.session_state["picked"]
    assert (second.stack, second.run) == (OLD, "2026-10-10")
    # And by another detector.
    at.selectbox(key="sb_stack").set_value(NEW).run()
    at.selectbox(key="sb_detector").set_value("CLD_o2_v09").run()
    third = at.session_state["picked"]
    assert (third.detector, third.stack, third.run) == ("CLD_o2_v09", NEW, "2026-10-10")
    refs = {first.ref, second.ref, third.ref}
    assert len(refs) == 3
    assert len({sel.data_dir for sel in (first, second, third)}) == 3


def test_changing_release_resets_a_run_it_does_not_have_and_says_so():
    at = _run_pick({"detector": "CLD_o2_v09", "stack": NEW, "run": "2026-10-11"})
    at.selectbox(key="sb_stack").set_value(OLD).run()
    assert _picked(at)[-2:] == ("single_e-_10GeV", "2026-10-09")
    assert any(
        f"`2026-10-11` isn't a run of {OLD} — showing `2026-10-09`" in c for c in _captions(at)
    ), _captions(at)
    assert _query(at)["run"] == "2026-10-09"


def test_a_linked_run_missing_from_the_store_is_named_with_an_explicit_fallback():
    at = _run_pick({"detector": "CLD_o2_v09", "stack": NEW, "run": "2026-10-05"})
    # Nothing is drawn in its place until the reader asks for the latest.
    assert at.session_state["picked"] is None
    assert "The linked run `2026-10-05` is not in the store" in at.warning[0].value
    assert f"CLD_o2_v09 on {PLAT} with {NEW}" in at.warning[0].value
    assert not [box for box in at.selectbox if box.key == "sb_run"]
    at.button[1].click().run()
    assert _picked(at)[-1] == "2026-10-11"
    assert _query(at)["run"] == "2026-10-11"
    assert not at.warning


def test_a_malformed_run_link_is_named_and_not_guessed():
    at = _run_pick({"detector": "CLD_o2_v09", "run": "last-tuesday"})
    assert at.session_state["picked"] is None
    assert "`last-tuesday` does not name a run" in at.warning[0].value


def test_a_stalled_run_listing_keeps_the_pinned_run():
    at = _run_pick({"detector": "CLD_o2_v09", "stack": NEW, "run": "2026-10-10"}, fail="runs")
    # Not replaced by a guess at the latest: the pinned run is still drawn.
    assert _picked(at)[-1] == "2026-10-10"
    assert "Could not list the runs of" in at.warning[0].value
    assert "pinned run measured 2026-10-10" in at.warning[0].value
    assert at.selectbox(key="sb_run").options == ["2026-10-10"]
    # Its place among the release's runs is unknown until the listing answers.
    assert "Measured **2026-10-10**" in _captions(at)
    # Once the listing answers, the others are offered, the pin kept.
    at.session_state["fail"] = None
    at.run()
    assert _picked(at)[-1] == "2026-10-10"
    assert at.selectbox(key="sb_run").options == ["2026-10-11 · latest", "2026-10-10"]


def test_a_linked_run_shown_through_a_stalled_listing_is_checked_once_it_answers():
    at = _run_pick({"detector": "CLD_o2_v09", "stack": NEW, "run": "2026-10-05"}, fail="runs")
    assert "pinned run measured 2026-10-05" in at.warning[0].value
    at.session_state["fail"] = None
    at.run()
    # Not quietly replaced by the latest: named, with the fallback on offer.
    assert at.session_state["picked"] is None
    assert "The linked run `2026-10-05` is not in the store" in at.warning[0].value
    assert _query(at)["run"] == "2026-10-05"
    next(b for b in at.button if b.label.startswith("Open the latest run")).click().run()
    assert _picked(at)[-1] == "2026-10-11"


def test_a_stalled_run_listing_without_a_pin_stops():
    at = _run_pick({"detector": "CLD_o2_v09"}, fail="runs")
    assert at.session_state["picked"] is None
    assert "Failed to list the runs of" in at.error[0].value


def test_a_stale_linked_release_is_named():
    at = _run_pick({"detector": "CLD_o2_v09", "stack": "key4hep-2026-01-01"})
    assert _picked(at)[3] == NEW
    assert any(
        "`key4hep-2026-01-01` doesn't hold single_e-_10GeV for CLD_o2_v09 — showing" in c
        for c in _captions(at)
    ), _captions(at)


def test_pick_run_offers_only_the_releases_that_hold_the_sample():
    at = _run_pick({"detector": "ALLEGRO_o1_v04", "sample": "p8_ee_Zbb_ecm91"})
    assert at.selectbox(key="sb_stack").options == [NEW]
    assert at.caption[0].value == "Available in 1 release"
    assert at.session_state["picked"].stack == NEW


def test_pick_run_names_a_sample_that_switching_detector_dropped():
    at = _run_pick({"detector": "ALLEGRO_o1_v04", "sample": "p8_ee_Zbb_ecm91"})
    at.selectbox(key="sb_detector").set_value("CLD_o2_v09").run()
    assert at.session_state["picked"].sample == "single_e-_10GeV"
    notes = _captions(at)
    assert any(
        "`p8_ee_Zbb_ecm91` isn't benchmarked for CLD_o2_v09 — showing `single_e-_10GeV`" in n
        for n in notes
    ), notes


def test_pick_run_walks_the_requested_suite_under_its_own_keys():
    at = _run_pick(
        {"detector": "CLD_o2_v09", "sample": "single_e-_10GeV"},
        suite="reco", key_prefix="reco_",
    )
    assert set(at.session_state["suites"]) == {"reco"}
    assert at.session_state["picked"].data_dir == (
        f"/cache/reco/CLD_o2_v09/{NEW}/single_e-_10GeV/2026-10-11"
    )
    assert [box.key for box in at.selectbox] == [
        "reco_sb_detector", "reco_sb_platform", "reco_sb_sample", "reco_sb_stack", "reco_sb_run",
    ]
    # Every page shares the URL's parameter names.
    assert _query(at) == {
        "detector": "CLD_o2_v09", "platform": PLAT,
        "sample": "single_e-_10GeV", "stack": NEW, "run": "2026-10-11",
    }


def test_pick_run_returns_to_the_run_a_reader_left_on_another_page():
    # Neither is a default: ALLEGRO_o1_v04 is listed first, NEW is the newest.
    at = _run_pick({"detector": "CLD_o2_v09"})
    at.selectbox(key="sb_stack").set_value(OLD).run()
    at.session_state["away"] = True
    at.run()
    assert _query(at) == {}
    at.session_state["away"] = False
    at.run()
    picked = at.session_state["picked"]
    assert (picked.detector, picked.stack, picked.run) == ("CLD_o2_v09", OLD, "2026-10-09")
    # The address bar is a deep link to it again.
    assert _query(at)["detector"] == "CLD_o2_v09"
    assert _query(at)["stack"] == OLD
    assert _query(at)["run"] == "2026-10-09"


def test_pick_run_stops_where_the_store_cannot_be_read():
    at = _run_pick(fail="detectors")
    assert at.session_state["picked"] is None
    assert "Failed to list detectors: listing timed out" in at.error[0].value
    assert not at.selectbox


def test_pick_run_stops_when_the_release_has_no_run():
    at = _run_pick(fail="no runs")
    assert at.session_state["picked"] is None
    assert at.warning[0].value == "No runs found for the selected combination."


def _window_app(
    dashboard_dir, run_dates, report_nights, platform_stacks, across, window_kwargs,
) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    import sidebar

    def _platform_stacks(url, suite, plat):
        if platform_stacks is None or st.session_state.get("stall"):
            raise OSError("listing timed out")
        # A test publishes a release mid-session through "new_stacks".
        return [*st.session_state.get("new_stacks", []), *platform_stacks]

    # Every suite the run listings were asked for.
    st.session_state["suites"] = suites = []

    def _run_dates(url, suite, det, plat, sample):
        suites.append(suite)
        return run_dates

    sidebar.cached_list_run_dates = _run_dates
    # A test lands a report mid-session through "new_nights".
    def _report_dates(url, suite):
        if report_nights is None:
            raise OSError("no reports/ for this suite")
        # A test lands a report mid-session through "new_nights".
        return [*report_nights, *st.session_state.get("new_nights", [])]

    sidebar.cached_list_report_dates = _report_dates
    sidebar.list_platform_stacks = _platform_stacks
    def _fetch_runs_windowed(url, suite, det, plat, sample, cache, items):
        suites.append(suite)
        return tuple(f"/cache/{stack}/{d}" for stack, dates in items for d in dates)

    sidebar.cached_fetch_runs_windowed = _fetch_runs_windowed
    from k4bench.layout import RunRef

    selection = sidebar.RunSelection(RunRef(
        window_kwargs.get("suite", "sim"), st.session_state.get("detector", "CLD_o2_v09"),
        "PLAT", "key4hep-2026-10-10", "single_e-_10GeV", "2026-10-10",
    ), "/cache/run")
    # A test changes section mid-session through "across".
    st.session_state["window"] = sidebar.pick_trend_window(
        "https://store.invalid", selection, "/cache",
        st.session_state.get("across", across), **window_kwargs,
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
    across=True, params=None, **window_kwargs,
) -> AppTest:
    at = AppTest.from_function(
        _window_app,
        args=(
            str(_DASHBOARD_DIR), run_dates, report_nights, platform_stacks, across,
            window_kwargs,
        ),
        default_timeout=30,
    )
    for key, value in (params or {}).items():
        at.query_params[key] = value
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
    # The window is part of the deep link, by its releases' directories.
    assert _query(at) == {"releases": f"{_rel('09-22')}..{_rel('10-10')}"}
    # The selected detector's runs are no part of what the Overview draws.
    assert at.caption[0].value == "2026-09-22 → 2026-10-10 · 10 releases"


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
    assert at.caption[0].value == "2026-10-08 → 2026-10-12 · 3 releases · 4 runs"
    # The platform's releases are not listed at all: the failing stub would
    # have warned.
    assert not at.warning


def test_a_suite_without_reports_bounds_the_window_by_its_runs():
    # The failing report listing would have warned had it been read.
    at = _run_window(
        NEW_DETECTOR_RUNS, report_nights=None, platform_stacks=None, across=False,
        suite="reco", reports=False,
    )
    assert not at.warning
    assert at.session_state["suites"] == ["reco", "reco"]
    window = at.session_state["window"]
    assert window.dates == (date(2026, 10, 8), date(2026, 10, 12))
    assert len(window.run_dirs) == 4


def test_without_reports_the_window_runs_to_the_newest_rerun_of_its_release():
    # 10-10 was the newest release and kept being re-run until 10-16.
    runs = {_rel("10-08"): ["2026-10-08"], _rel("10-10"): ["2026-10-10", "2026-10-16"]}
    at = _run_window(runs, report_nights=None, platform_stacks=None, across=False,
                     suite="reco", reports=False)
    assert at.session_state["window"].dates == (date(2026, 10, 8), date(2026, 10, 16))


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


def test_a_linked_window_reopens_on_its_releases():
    at = _run_window(LONG_RUNS, params={"releases": f"{_rel('09-24')}..{_rel('09-28')}"})
    assert at.select_slider[0].value == (_rel("09-24"), _rel("09-28"))
    assert at.session_state["window"].dates == (date(2026, 9, 24), date(2026, 9, 29))
    # A link holds the window even though newer releases were published since.
    assert _query(at)["releases"] == f"{_rel('09-24')}..{_rel('09-28')}"


def test_a_linked_window_the_store_does_not_offer_is_named():
    at = _run_window(LONG_RUNS, params={"releases": f"{_rel('01-01')}..{_rel('09-28')}"})
    assert at.select_slider[0].value == (_rel("09-22"), _rel("10-10"))
    assert any("The linked window" in c and "showing the newest" in c for c in _captions(at))
    assert _query(at)["releases"] == f"{_rel('09-22')}..{_rel('10-10')}"


def test_a_stalled_listing_leaves_a_linked_window_in_the_url():
    linked = f"{_rel('09-24')}..{_rel('09-28')}"
    at = _run_window(LONG_RUNS, platform_stacks=None, params={"releases": linked})
    # Judged against a listing that did not answer, the link could only be
    # dropped: it is kept for the next run, and not overwritten.
    assert _query(at)["releases"] == linked
    assert not any("The linked window" in c for c in _captions(at))


def test_the_runs_per_release_choice_is_offered_where_a_section_honours_it():
    at = _run_window(NEW_DETECTOR_RUNS, across=False, offer_all_runs=True)
    assert at.radio[0].value == "latest"
    assert at.session_state["window"].all_runs is False
    at.radio[0].set_value("all").run()
    assert at.session_state["window"].all_runs is True
    assert _query(at)["history"] == "all"
    # The window is the same either way: what changes is what a history draws.
    assert len(at.session_state["window"].run_dirs) == 4


def test_a_linked_runs_per_release_choice_is_restored_or_named():
    at = _run_window(NEW_DETECTOR_RUNS, across=False, offer_all_runs=True,
                     params={"history": "all"})
    assert at.session_state["window"].all_runs is True
    at = _run_window(NEW_DETECTOR_RUNS, across=False, offer_all_runs=True,
                     params={"history": "every"})
    assert at.session_state["window"].all_runs is False
    assert "`every` is not a choice of runs" in " ".join(_captions(at))


def test_a_section_without_the_choice_draws_the_latest_and_keeps_the_link():
    at = _run_window(NEW_DETECTOR_RUNS, across=False, params={"history": "all"})
    assert not at.radio
    assert at.session_state["window"].all_runs is False
    # Kept for a section that honours it.
    assert _query(at)["history"] == "all"


def test_a_stalled_listing_does_not_move_a_picked_window():
    # 09-22 is a platform release this detector never ran: while the
    # platform's listing stalls, the slider offers only the detector's own.
    linked = f"{_rel('09-22')}..{_rel('09-26')}"
    at = _run_window(LONG_RUNS, params={"releases": linked})
    assert at.select_slider[0].value == (_rel("09-22"), _rel("09-26"))
    at.session_state["stall"] = True
    at.run()
    assert "Could not list the platform's releases" in at.warning[0].value
    assert _query(at)["releases"] == linked
    at.session_state["stall"] = False
    at.run()
    # The listing answered again: the window is the one picked before the stall.
    assert at.select_slider[0].value == (_rel("09-22"), _rel("09-26"))
    assert _query(at)["releases"] == linked
