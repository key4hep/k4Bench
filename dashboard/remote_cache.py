"""Streamlit-cached wrappers around :mod:`k4bench.remote`'s network calls.

These are thin ``@st.cache_data`` shims kept out of ``app.py`` so the entry
point stays focused on layout and data-source resolution. Each wrapper imports
its underlying ``k4bench.remote`` function lazily so importing this module is
cheap.
"""
from __future__ import annotations

import logging

import streamlit as st

_log = logging.getLogger(__name__)


@st.cache_data(show_spinner="Fetching detectors...", ttl=3600)
def _cached_list_detectors(base_url: str) -> list[str]:
    from k4bench.remote import list_detectors
    return list_detectors(base_url)


@st.cache_data(show_spinner="Fetching platforms...", ttl=3600)
def _cached_list_platforms(base_url: str, detector: str) -> list[str]:
    from k4bench.remote import list_platforms
    return list_platforms(base_url, detector)


@st.cache_data(show_spinner="Scanning releases...", ttl=3600)
def _cached_scan_stack_samples(
    base_url: str, detector: str, platform: str
) -> dict[str, list[str]]:
    from k4bench.remote import scan_stack_samples
    return scan_stack_samples(base_url, detector, platform)


@st.cache_data(show_spinner="Scanning run dates...", ttl=600)
def _cached_list_run_dates(
    base_url: str, detector: str, platform: str, sample: str
) -> dict[str, list[str]]:
    from k4bench.remote import list_run_dates_all_stacks
    return list_run_dates_all_stacks(base_url, detector, platform, sample)


@st.cache_data(show_spinner="Scanning Key4hep releases...", ttl=600)
def _cached_list_stacks(base_url: str, detector: str, platform: str) -> list[str]:
    from k4bench.remote import list_stacks
    return list_stacks(base_url, detector, platform)


@st.cache_data(show_spinner="Fetching stack provenance...", ttl=3600)
def _cached_fetch_stack_packages(
    base_url: str, detector: str, platform: str, stack: str
) -> dict | None:
    """Cached :func:`k4bench.remote.fetch_stack_packages`.

    A release is immutable once published, so this is cached for the full hour
    — the same stack is re-read every time it is an endpoint of a comparison.
    """
    from k4bench.remote import fetch_stack_packages
    return fetch_stack_packages(base_url, detector, platform, stack)


@st.cache_data(show_spinner="Listing regression reports...", ttl=600)
def _cached_list_report_dates(base_url: str) -> list[str]:
    from k4bench.remote import list_report_dates
    return list_report_dates(base_url)


@st.cache_data(show_spinner="Fetching regression report...", ttl=3600)
def _cached_fetch_report(base_url: str, date: str) -> dict | None:
    from k4bench.remote import fetch_report
    return fetch_report(base_url, date)


@st.cache_data(show_spinner="Fetching nightly reports...", ttl=3600)
def _cached_fetch_reports(base_url: str, dates: tuple[str, ...]) -> dict[str, dict]:
    """Fetch a whole window of nightly reports in parallel, keyed by date.

    One report per night and each is a small JSON, so a cold window is
    dominated by request latency — fanning out cuts a 30-night first load
    from ~30 sequential round-trips to a few. Nights that fail to fetch are
    simply absent from the result. Cached on the *dates* tuple: growing the
    window refetches it in one parallel burst rather than serially.

    Worker count is deliberately modest (4): this only ever runs on a cache
    miss (a Streamlit rerun that hits the cache spawns no threads at all),
    and keeping concurrent SSL connections low reduces the chance of a rerun
    landing mid-fetch and racing the pool's shutdown.
    """
    from concurrent.futures import ThreadPoolExecutor

    from k4bench.remote import fetch_report

    def _one(date: str) -> tuple[str, dict | None]:
        # fetch_report already handles network/parse failures (logs and returns
        # None); this guards only unexpected errors so one bad night can't abort
        # the whole window — logged (dashboard convention) but not surfaced as a
        # UI warning, since the night is simply absent from the result.
        try:
            return date, fetch_report(base_url, date)
        except Exception:
            _log.exception("fetch_report: unexpected error for %s", date)
            return date, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        return {date: raw for date, raw in pool.map(_one, dates) if raw}


@st.cache_data(show_spinner="Downloading latest run...", ttl=3600)
def _cached_fetch_latest_run(
    base_url: str, detector: str, platform: str, stack: str, sample: str, cache_dir: str
) -> str | None:
    from k4bench.remote import ensure_latest_run_cached
    p = ensure_latest_run_cached(
        base_url, detector, platform, stack, sample, cache_root=cache_dir
    )
    return str(p) if p else None


@st.cache_data(show_spinner="Downloading trend data...", ttl=3600)
def _cached_fetch_runs_windowed(
    base_url: str,
    detector: str,
    platform: str,
    sample: str,
    cache_dir: str,
    stacks_dates_items: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[str, ...]:
    """Download the windowed ``(stack, date)`` set and return cached run dirs.

    *stacks_dates_items* is a hashable ``((stack, (date, ...)), ...)`` so the cache
    key varies with the selected window — widening reuses already-cached runs and
    only the new runs are fetched.
    """
    from k4bench.remote import fetch_runs_windowed
    stacks_dates = {stack: list(dates) for stack, dates in stacks_dates_items}
    runs = fetch_runs_windowed(
        base_url, detector, platform, sample, stacks_dates, cache_root=cache_dir
    )
    return tuple(sorted(r["run_dir"] for r in runs))
