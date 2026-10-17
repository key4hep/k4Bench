from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.loader import judgeable_config_data
from k4bench.analysis.loader import load_component_timing, load_event_timing, load_region_timing, load_results  # noqa: F401 (re-exported)
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


# Each run's matrices are ~1 MB pickled; the bound keeps a long session's
# comparisons from piling up in the pod's memory.
@st.cache_data(show_spinner="Loading component costs...", max_entries=16)
def cached_load_component_timing(data_dir: str) -> dict[str, ComponentTiming]:
    """Every label's ``_components.json`` in *data_dir*; empty when it has none."""
    try:
        return load_component_timing(data_dir)
    except _EXPECTED_ERRORS:
        return {}
    except Exception:
        _log.exception("cached_load_component_timing: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading component costs from '{data_dir}'. Check logs for details.")
        return {}


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


@st.cache_data(show_spinner="Loading job phase trends...", ttl=3600)
def cached_load_trend_phases(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_phase_trend`."""
    return _trend.build_phase_trend(run_dirs)


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


@dataclass(frozen=True)
class RunData:
    """The selected run's data, plus the window's trend results its labels also draw on."""

    results: pd.DataFrame | None
    event_data: dict | None
    region_data: dict | None
    #: The windowed runs' results; ``None`` when the window holds none, as in local mode.
    trend_results: pd.DataFrame | None
    #: Every configuration with data on screen, sorted.
    labels: list[str]
    #: Each successful configuration's per-component costs; loaded only for a
    #: page that reads them (k4run runs have them, ddsim runs none).
    components: dict[str, ComponentTiming] = field(default_factory=dict)


def load_run(
    data_dir: str, run_dirs: tuple[str, ...], *, components: bool = False,
) -> RunData:
    """Load the run in *data_dir* and the results of the windowed *run_dirs*,
    with the run's per-component costs when *components*."""
    results = cached_load_results(data_dir)
    event_data = cached_load_event_timing(data_dir)
    region_data = cached_load_region_timing(data_dir)
    component_data = cached_load_component_timing(data_dir) if components else {}
    # Current-run event/region files can be partial leftovers from a
    # crashed config, or orphans with no result row at all. These views
    # present their inputs as ordinary evidence (unlike regression
    # plots, which can mark a failed point explicitly), so only
    # successful result-backed labels belong here.
    event_data = judgeable_config_data(event_data, results)
    region_data = judgeable_config_data(region_data, results)
    component_data = judgeable_config_data(component_data, results) or {}

    labels: set[str] = set()
    if results is not None and "label" in results.columns:
        labels.update(results["label"].unique())
    labels.update(event_data or ())
    labels.update(region_data or ())
    # The latest run alone can be missing every config (a failed/timed-out
    # job, or ones retired from the newest release) while the trend window
    # still has history for them — union those labels in too, so the
    # "no benchmark data" guards don't blank out a page that Run
    # Trends does have data for.
    trend_results = cached_load_trend_results(run_dirs) if run_dirs else None
    if trend_results is not None and "label" in trend_results.columns:
        labels.update(trend_results["label"])
    return RunData(
        results, event_data, region_data, trend_results, sorted(labels), component_data,
    )
