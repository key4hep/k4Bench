"""Tests for the dashboard's cached WebEOS wrappers (``dashboard/remote_cache.py``).

``st.cache_data`` keeps whatever a function returns, for every viewer, until its
TTL runs out. These pin down that what a stalled server *failed* to return is
never kept: content is handed back once as far as it was read, discovery raises
so its callers can fall back, and both are read again on the next call.

All remote calls are stubbed; nothing touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402

from k4bench import remote  # noqa: E402
from k4bench.layout import SIM  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import remote_cache  # noqa: E402

BASE = "https://eos.example/data"


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _timeout(what: str) -> remote.IncompleteFetch:
    return remote.IncompleteFetch(None, [f"{what}: read timed out"])


def _nights(reports) -> dict[str, str]:
    """The fetched reports as ``{date: the night each parsed report names}``."""
    return {date: report.night for date, report in reports.items()}


def test_report_window_with_a_stalled_night_is_read_again(monkeypatch):
    calls: list[str] = []
    stalled = {"2026-09-23"}

    def fetch_report(base_url, suite, date, *, strict=False):
        calls.append(date)
        if date in stalled:
            raise _timeout(date)
        return {"night": date}

    monkeypatch.setattr(remote, "fetch_report", fetch_report)
    nights = ("2026-09-24", "2026-09-23")

    # The night that answered is shown; the stalled one is simply absent…
    assert _nights(remote_cache.cached_fetch_reports(BASE, SIM, nights)) == {
        "2026-09-24": "2026-09-24",
    }
    # …and not remembered as absent: the next call asks again, and only for it.
    stalled.clear()
    calls.clear()
    assert _nights(remote_cache.cached_fetch_reports(BASE, SIM, nights)) == {
        "2026-09-24": "2026-09-24",
        "2026-09-23": "2026-09-23",
    }
    assert calls == ["2026-09-23"]

    # A complete window is cached as before, and so is every night of it on
    # its own: a window sharing nights with it reads only its new ones.
    calls.clear()
    remote_cache.cached_fetch_reports(BASE, SIM, nights)
    assert calls == []
    remote_cache.cached_fetch_reports(BASE, SIM, ("2026-09-25", "2026-09-24"))
    assert calls == ["2026-09-25"]


def test_report_window_keeps_a_night_that_is_absent(monkeypatch):
    calls: list[str] = []

    def fetch_report(base_url, suite, date, *, strict=False):
        calls.append(date)
        if date == "2026-09-22":
            return {"groups": [{"detector": "CLD"}]}  # half-written: no platform
        return None if date == "2026-09-23" else {"night": date}

    monkeypatch.setattr(remote, "fetch_report", fetch_report)
    nights = ("2026-09-24", "2026-09-23", "2026-09-22")
    remote_cache.cached_fetch_reports(BASE, SIM, nights)
    calls.clear()
    # A 404 or an unparseable report is an answer: cached like any other, and
    # costing only its own night.
    assert _nights(remote_cache.cached_fetch_reports(BASE, SIM, nights)) == {
        "2026-09-24": "2026-09-24",
    }
    assert calls == []


def test_report_listing_that_did_not_answer_is_not_cached(monkeypatch):
    answers = iter([
        remote.requests.ReadTimeout("read timed out"),
        ["2026-09-24", "2026-09-23"],
    ])

    def list_report_dates(base_url, suite):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(remote, "list_report_dates", list_report_dates)
    with pytest.raises(remote.requests.ReadTimeout):
        remote_cache.cached_list_report_dates(BASE, SIM)
    assert remote_cache.cached_list_report_dates(BASE, SIM) == ["2026-09-24", "2026-09-23"]


@pytest.mark.parametrize("name, wrapper, args", [
    ("list_run_dates_all_stacks", "cached_list_run_dates", (BASE, SIM, "DET", "PLAT", "single_e")),
    ("scan_stack_samples", "cached_scan_stack_samples", (BASE, SIM, "DET", "PLAT")),
])
def test_discovery_that_stopped_short_raises_and_is_read_again(monkeypatch, name, wrapper, args):
    # A listing missing the newest release would make an older one look newest,
    # and a scan missing one would drop a valid sidebar selection: the caller
    # must see that it is incomplete, not a smaller answer.
    complete = {"key4hep-2026-09-24": ["x"], "key4hep-2026-09-23": ["x"]}
    calls = []

    def listing(*a, strict=False):
        assert strict, "the cached listing must run strict to see a stall"
        calls.append(a)
        if len(calls) == 1:
            raise remote.IncompleteFetch(
                {"key4hep-2026-09-23": ["x"]}, ["key4hep-2026-09-24: read timed out"],
            )
        return complete

    monkeypatch.setattr(remote, name, listing)
    cached = getattr(remote_cache, wrapper)
    with pytest.raises(remote.IncompleteFetch):
        cached(*args)
    assert cached(*args) == complete
    assert cached(*args) == complete
    assert len(calls) == 2  # the complete answer was kept, the short one not


def test_partial_trend_download_returns_the_runs_that_landed(monkeypatch):
    landed = [{"stack": "S", "date": "2026-09-24", "run_dir": "/cache/S/2026-09-24"}]

    def fetch_runs_windowed(*args, cache_root=None, strict=False, artifacts=None):
        assert strict
        assert artifacts == ("run_info.json",)
        raise remote.IncompleteFetch(landed, ["S/2026-09-23: read timed out"])

    monkeypatch.setattr(remote, "fetch_runs_windowed", fetch_runs_windowed)
    got = remote_cache.cached_fetch_runs_windowed(
        BASE, SIM, "DET", "PLAT", "single_e", "/cache",
        (("S", ("2026-09-23", "2026-09-24")),), ("run_info.json",),
    )
    assert got == ("/cache/S/2026-09-24",)


def test_a_run_files_download_that_failed_is_tried_again(monkeypatch):
    calls = []

    def ensure_run_artifacts(*args, cache_root=None):
        calls.append(args[-1])
        if len(calls) == 1:
            raise remote.requests.ReadTimeout("read timed out")
        return Path("/cache/run")

    monkeypatch.setattr(remote, "ensure_run_artifacts", ensure_run_artifacts)
    args = (BASE, SIM, "DET", "PLAT", "S", "single_e", "2026-09-24", "/cache")
    with pytest.raises(remote.requests.ReadTimeout):
        remote_cache.cached_fetch_run_files(*args, ("*_events.json",))
    assert remote_cache.cached_fetch_run_files(*args, ("*_events.json",)) == "/cache/run"
    assert remote_cache.cached_fetch_run_files(*args, ("*_events.json",)) == "/cache/run"
    # Other files of the same run are a download of their own.
    remote_cache.cached_fetch_run_files(*args, ("*_heap.json",))
    assert calls == [("*_events.json",), ("*_events.json",), ("*_heap.json",)]


@pytest.mark.parametrize("name, wrapper, args", [
    ("fetch_blame", "cached_fetch_blame", (BASE, SIM, "2026-09-24")),
    ("fetch_stack_packages", "cached_fetch_stack_packages", (BASE, SIM, "DET", "PLAT", "S")),
])
def test_other_wrappers_do_not_keep_a_stall(monkeypatch, name, wrapper, args):
    calls = []

    def fetch(*a, strict=False):
        assert strict
        calls.append(a)
        if len(calls) == 1:
            raise remote.IncompleteFetch(None, ["read timed out"])
        return {"answered": True}

    monkeypatch.setattr(remote, name, fetch)
    cached = getattr(remote_cache, wrapper)
    assert cached(*args) is None
    assert cached(*args) == {"answered": True}
    assert cached(*args) == {"answered": True}
    assert len(calls) == 2


def _drill_down_verdict():
    from k4bench.regression.models import Direction, MetricVerdict, Severity

    return MetricVerdict(
        detector="ALLEGRO_o2_v01", platform="PLAT", sample="single_e", label="baseline",
        metric_family="memory", metric="peak_vmem_mb", sub_detector=None,
        run_id="2026-09-24", run_date="2026-09-24", value=5850.0,
        baseline_median=6623.5, baseline_mad=0.3, pct_change=-0.117, z_score=-5216.0,
        severity=Severity.WATCH, direction=Direction.DOWN, reason="step",
    )


def test_metric_drill_down_warns_when_its_run_history_stopped_short(monkeypatch):
    # The chart's window is placed around the flagged run from this listing, so
    # it must not be drawn from part of the releases.
    from tabs.reports import _metric_trend as trend

    verdict = _drill_down_verdict()

    def listing(*_args):
        raise remote.IncompleteFetch({}, ["key4hep-2026-09-24: read timed out"])

    def downloads(*_args):
        raise AssertionError("nothing may be downloaded from a partial listing")

    warnings = []
    monkeypatch.setattr(trend.st, "warning", lambda text, **_kw: warnings.append(text))
    trend.render_metric_trend(
        verdict, BASE, "sim", "/cache",
        list_run_dates=listing, fetch_runs_windowed=downloads, widget_namespace="t",
    )
    assert len(warnings) == 1
    assert "Could not list this metric's run history" in warnings[0]


def test_metric_drill_down_warns_when_the_local_cache_fails(monkeypatch):
    # A full or unwritable run cache propagates from the strict download as
    # itself; the drill-down must say so rather than raise into the page, and
    # must not blame EOS for it.
    import errno

    from tabs.reports import _metric_trend as trend

    def listing(*_args):
        return {"key4hep-2026-09-24": ["2026-09-24"]}

    def downloads(*_args):
        raise OSError(errno.ENOSPC, "No space left on device")

    warnings = []
    monkeypatch.setattr(trend.st, "warning", lambda text, **_kw: warnings.append(text))
    trend.render_metric_trend(
        _drill_down_verdict(), BASE, "sim", "/cache",
        list_run_dates=listing, fetch_runs_windowed=downloads, widget_namespace="t",
    )
    assert len(warnings) == 1
    assert "local cache" in warnings[0]
    assert "No space left on device" in warnings[0]


def test_stacks_are_unioned_across_detectors(monkeypatch):
    # Detectors join and leave the matrix, so no single detector's history is
    # the full set of releases.
    per_detector = {"IDEA": ["key4hep-2026-07-10"], "SiD": ["key4hep-2026-07-09"]}
    monkeypatch.setattr(remote_cache, "cached_list_detectors", lambda url, suite: ["IDEA", "SiD"])
    monkeypatch.setattr(
        remote_cache, "cached_list_stacks",
        lambda url, suite, detector, platform: per_detector[detector],
    )
    assert remote_cache.list_platform_stacks("u", SIM, "PLAT") == [
        "key4hep-2026-07-10", "key4hep-2026-07-09",
    ]


def _http_error(status: int):
    import requests

    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status}", response=response)


def test_a_detector_without_the_platform_is_skipped(monkeypatch):
    def _stacks(url, suite, detector, platform):
        if detector == "SiD":
            raise _http_error(404)  # never ran on this platform
        return ["key4hep-2026-07-10"]

    monkeypatch.setattr(remote_cache, "cached_list_detectors", lambda url, suite: ["IDEA", "SiD"])
    monkeypatch.setattr(remote_cache, "cached_list_stacks", _stacks)
    # One detector missing a platform must not blank the whole list.
    assert remote_cache.list_platform_stacks("u", SIM, "PLAT") == ["key4hep-2026-07-10"]


@pytest.mark.parametrize("error", [
    remote.requests.ReadTimeout("read timed out"), _http_error(504),
], ids=["timeout", "gateway"])
def test_a_detector_listing_that_did_not_answer_is_not_skipped(monkeypatch, error):
    def _stacks(url, suite, detector, platform):
        if detector == "SiD":
            raise error
        return ["key4hep-2026-07-10"]

    monkeypatch.setattr(remote_cache, "cached_list_detectors", lambda url, suite: ["IDEA", "SiD"])
    monkeypatch.setattr(remote_cache, "cached_list_stacks", _stacks)
    # A stall says nothing about SiD's releases: the union would be short, and
    # the sidebar would judge a linked trend window against it as complete.
    with pytest.raises(type(error)):
        remote_cache.list_platform_stacks("u", SIM, "PLAT")


def test_an_unexpected_listing_error_is_not_swallowed(monkeypatch):
    def _stacks(url, suite, detector, platform):
        raise TypeError("a bug in the listing code")

    monkeypatch.setattr(remote_cache, "cached_list_detectors", lambda url, suite: ["IDEA"])
    monkeypatch.setattr(remote_cache, "cached_list_stacks", _stacks)
    # Only an absent directory is expected here. Anything else is a bug, and
    # must surface rather than read as "this platform has no releases".
    with pytest.raises(TypeError):
        remote_cache.list_platform_stacks("u", SIM, "PLAT")


def test_read_again_reads_one_listing_past_the_cache(monkeypatch):
    # A run published after its release's listing was cached: the cached
    # listing lacks it until it is read again.
    answers = iter([["2026-10-11"], ["2026-10-12", "2026-10-11"]])
    monkeypatch.setattr(remote, "list_runs", lambda *args: next(answers))
    args = (BASE, SIM, "DET", "PLAT", "key4hep-2026-10-10", "single_e")
    assert remote_cache.cached_list_runs(*args) == ["2026-10-11"]
    assert remote_cache.cached_list_runs(*args) == ["2026-10-11"]
    assert remote_cache.read_again(remote_cache.cached_list_runs, *args) == [
        "2026-10-12", "2026-10-11",
    ]
    # And what it read is what the cache holds from then on.
    assert remote_cache.cached_list_runs(*args) == ["2026-10-12", "2026-10-11"]
