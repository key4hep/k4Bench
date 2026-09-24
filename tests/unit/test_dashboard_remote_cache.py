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

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import remote_cache  # noqa: E402

BASE = "https://eos.example/data"


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _timeout(what: str) -> remote.IncompleteFetch:
    return remote.IncompleteFetch(None, [f"{what}: read timed out"])


def test_report_window_with_a_stalled_night_is_read_again(monkeypatch):
    calls: list[str] = []
    stalled = {"2026-09-23"}

    def fetch_report(base_url, date, *, strict=False):
        calls.append(date)
        if date in stalled:
            raise _timeout(date)
        return {"night": date}

    monkeypatch.setattr(remote, "fetch_report", fetch_report)
    nights = ("2026-09-24", "2026-09-23")

    # The night that answered is shown; the stalled one is simply absent…
    assert remote_cache._cached_fetch_reports(BASE, nights) == {
        "2026-09-24": {"night": "2026-09-24"},
    }
    # …and not remembered as absent: the next call asks again.
    stalled.clear()
    calls.clear()
    assert remote_cache._cached_fetch_reports(BASE, nights) == {
        "2026-09-24": {"night": "2026-09-24"},
        "2026-09-23": {"night": "2026-09-23"},
    }
    assert sorted(calls) == sorted(nights)

    # A complete window is cached as before.
    calls.clear()
    remote_cache._cached_fetch_reports(BASE, nights)
    assert calls == []


def test_report_window_keeps_a_night_that_is_absent(monkeypatch):
    calls: list[str] = []

    def fetch_report(base_url, date, *, strict=False):
        calls.append(date)
        return None if date == "2026-09-23" else {"night": date}

    monkeypatch.setattr(remote, "fetch_report", fetch_report)
    nights = ("2026-09-24", "2026-09-23")
    remote_cache._cached_fetch_reports(BASE, nights)
    calls.clear()
    # A 404 or an unparseable report is an answer: cached like any other.
    assert remote_cache._cached_fetch_reports(BASE, nights) == {
        "2026-09-24": {"night": "2026-09-24"},
    }
    assert calls == []


def test_report_listing_that_did_not_answer_is_not_cached(monkeypatch):
    answers = iter([
        remote.requests.ReadTimeout("read timed out"),
        ["2026-09-24", "2026-09-23"],
    ])

    def list_report_dates(base_url):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(remote, "list_report_dates", list_report_dates)
    with pytest.raises(remote.requests.ReadTimeout):
        remote_cache._cached_list_report_dates(BASE)
    assert remote_cache._cached_list_report_dates(BASE) == ["2026-09-24", "2026-09-23"]


@pytest.mark.parametrize("name, wrapper, args", [
    ("list_run_dates_all_stacks", "_cached_list_run_dates", (BASE, "DET", "PLAT", "single_e")),
    ("scan_stack_samples", "_cached_scan_stack_samples", (BASE, "DET", "PLAT")),
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

    def fetch_runs_windowed(*args, cache_root=None, strict=False):
        assert strict
        raise remote.IncompleteFetch(landed, ["S/2026-09-23: read timed out"])

    monkeypatch.setattr(remote, "fetch_runs_windowed", fetch_runs_windowed)
    got = remote_cache._cached_fetch_runs_windowed(
        BASE, "DET", "PLAT", "single_e", "/cache",
        (("S", ("2026-09-23", "2026-09-24")),),
    )
    assert got == ("/cache/S/2026-09-24",)


@pytest.mark.parametrize("name, wrapper, args", [
    ("fetch_blame", "_cached_fetch_blame", (BASE, "2026-09-24")),
    ("fetch_stack_packages", "_cached_fetch_stack_packages", (BASE, "DET", "PLAT", "S")),
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


def test_metric_drill_down_warns_when_its_run_history_stopped_short(monkeypatch):
    # The chart's window is placed around the flagged run from this listing, so
    # it must not be drawn from part of the releases.
    from tabs import _regression_trend as trend

    from k4bench.regression.models import Direction, MetricVerdict, Severity

    verdict = MetricVerdict(
        detector="ALLEGRO_o2_v01", platform="PLAT", sample="single_e", label="baseline",
        metric_family="memory", metric="peak_vmem_mb", sub_detector=None,
        run_id="2026-09-24", run_date="2026-09-24", value=5850.0,
        baseline_median=6623.5, baseline_mad=0.3, pct_change=-0.117, z_score=-5216.0,
        severity=Severity.WATCH, direction=Direction.DOWN, reason="step",
    )

    def listing(*_args):
        raise remote.IncompleteFetch({}, ["key4hep-2026-09-24: read timed out"])

    def downloads(*_args):
        raise AssertionError("nothing may be downloaded from a partial listing")

    warnings = []
    monkeypatch.setattr(trend.st, "warning", lambda text, **_kw: warnings.append(text))
    trend.render_metric_trend(
        verdict, BASE, "/cache",
        list_run_dates=listing, fetch_runs_windowed=downloads, widget_namespace="t",
    )
    assert len(warnings) == 1
    assert "Could not list this metric's run history" in warnings[0]
