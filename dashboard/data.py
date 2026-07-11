from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from k4bench.analysis.loader import load_event_timing, load_region_timing, load_results  # noqa: F401 (re-exported)
from k4bench.analysis import trend as _trend
from k4bench.analysis.trend import EXPECTED_LOAD_ERRORS as _EXPECTED_ERRORS
from k4bench.analysis.trend import parse_run_dir as _parse_run_dir  # noqa: F401 (re-exported)

_log = logging.getLogger(__name__)


@st.cache_data(show_spinner="Loading benchmark results...")
def cached_load_results(data_dir: str) -> pd.DataFrame | None:
    try:
        df = load_results(data_dir)
        return df if not df.empty else None
    except _EXPECTED_ERRORS:
        return None
    except Exception:
        _log.exception("cached_load_results: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading benchmark results from '{data_dir}'. Check logs for details.")
        return None


@st.cache_data(show_spinner="Loading event timing...")
def cached_load_event_timing(data_dir: str) -> dict | None:
    try:
        d = load_event_timing(data_dir)
        return d if d else None
    except _EXPECTED_ERRORS:
        return None
    except Exception:
        _log.exception("cached_load_event_timing: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading event timing from '{data_dir}'. Check logs for details.")
        return None


@st.cache_data(show_spinner="Loading region timing...")
def cached_load_region_timing(data_dir: str) -> dict | None:
    try:
        d = load_region_timing(data_dir)
        return d if d else None
    except _EXPECTED_ERRORS:
        return None
    except Exception:
        _log.exception("cached_load_region_timing: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading region timing from '{data_dir}'. Check logs for details.")
        return None


def run_metadata(run_dir: str) -> dict:
    """Return metadata for a single run (date) directory.

    Thin public wrapper over :func:`k4bench.analysis.trend.parse_run_dir` for
    callers that already hold one concrete run directory (e.g. the cached
    latest run for single-run tabs).
    """
    return _parse_run_dir(Path(run_dir))


# The trend loaders below are thin ``@st.cache_data`` shims over the pure
# builders in ``k4bench.analysis.trend`` (shared with the nightly regression
# report, which must run without Streamlit). See that module for behaviour docs.

@st.cache_data(show_spinner="Loading trend data...", ttl=3600)
def cached_load_trend_results(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_results_trend` for the Trends tab."""
    return _trend.build_results_trend(run_dirs)


@st.cache_data(show_spinner="Loading region timing trends...", ttl=3600)
def cached_load_trend_region_timing(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_region_timing_trend`."""
    return _trend.build_region_timing_trend(run_dirs)


@st.cache_data(show_spinner="Loading event timing trends...", ttl=3600)
def cached_load_trend_event_timing(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_event_timing_trend`."""
    return _trend.build_event_timing_trend(run_dirs)


@st.cache_data(show_spinner="Loading machine info trends...", ttl=3600)
def cached_load_trend_machine_info(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_machine_info_trend`."""
    return _trend.build_machine_info_trend(run_dirs)


@st.cache_data(show_spinner=False, ttl=60)
def load_machine_info(run_dir: str) -> dict | None:
    """Load ``machine_info.json`` from a run directory, or return ``None`` if absent."""
    return _trend.load_machine_info(run_dir)


def collect_labels(
    results: pd.DataFrame | None,
    event_data: dict | None,
    region_data: dict | None,
) -> list[str]:
    labels: set[str] = set()
    if results is not None and "label" in results.columns:
        labels.update(results["label"].unique())
    if event_data is not None:
        labels.update(event_data.keys())
    if region_data is not None:
        labels.update(region_data.keys())
    return sorted(labels)
