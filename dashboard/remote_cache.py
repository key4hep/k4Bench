"""Streamlit-cached wrappers around :mod:`k4bench.remote`'s network calls.

These are thin ``@st.cache_data`` shims kept out of ``app.py`` so the entry
point stays focused on layout and data-source resolution. Each wrapper imports
its underlying ``k4bench.remote`` function lazily so importing this module is
cheap.
"""
from __future__ import annotations

import streamlit as st


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


@st.cache_data(show_spinner="Listing regression reports...", ttl=600)
def _cached_list_report_dates(base_url: str) -> list[str]:
    from k4bench.remote import list_report_dates
    return list_report_dates(base_url)


@st.cache_data(show_spinner="Fetching regression report...", ttl=3600)
def _cached_fetch_report(base_url: str, date: str) -> dict | None:
    from k4bench.remote import fetch_report
    return fetch_report(base_url, date)


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
