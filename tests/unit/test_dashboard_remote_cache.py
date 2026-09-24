"""Tests for the dashboard's cached WebEOS wrappers (``dashboard/remote_cache.py``).

``st.cache_data`` keeps whatever a function returns, for every viewer, until its
TTL runs out. These pin down that what a stalled server *failed* to return is
never kept: the wrappers hand it back once and read it again on the next call.

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


def test_partial_run_date_scan_is_read_again(monkeypatch):
    complete = {"key4hep-2026-09-24": ["2026-09-24"], "key4hep-2026-09-23": ["2026-09-23"]}
    partial = {"key4hep-2026-09-24": ["2026-09-24"]}
    answers = iter([
        remote.IncompleteFetch(partial, ["key4hep-2026-09-23: read timed out"]),
        complete,
    ])

    def list_run_dates_all_stacks(base_url, detector, platform, sample, *, strict=False):
        assert strict, "the cached scan must run strict to see a stall"
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(remote, "list_run_dates_all_stacks", list_run_dates_all_stacks)
    args = (BASE, "DET", "PLAT", "single_e")
    assert remote_cache._cached_list_run_dates(*args) == partial
    assert remote_cache._cached_list_run_dates(*args) == complete
    assert remote_cache._cached_list_run_dates(*args) == complete  # cached now


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
    ("scan_stack_samples", "_cached_scan_stack_samples", (BASE, "DET", "PLAT")),
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
