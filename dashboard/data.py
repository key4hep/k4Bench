from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from dd4bench.analysis.loader import load_event_timing, load_region_timing, load_results

_log = logging.getLogger(__name__)

_EXPECTED_ERRORS = (FileNotFoundError, ValueError, pd.errors.ParserError)


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
