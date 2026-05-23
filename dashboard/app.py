from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import Config
from data import (
    cached_load_event_timing,
    cached_load_region_timing,
    cached_load_results,
    cached_load_trend_results,
    collect_labels,
    list_run_metadata,
)
from tabs import event_memory, event_timing, overview, region_timing, trends


@st.cache_data(show_spinner="Fetching available detectors...", ttl=3600)
def _cached_list_detectors(base_url: str) -> list[str]:
    from remote import list_detectors
    return list_detectors(base_url)


@st.cache_data(show_spinner="Downloading all runs...", ttl=3600)
def _cached_download_all_runs(base_url: str, detector: str) -> str:
    from remote import download_all_runs
    return str(download_all_runs(base_url, detector))


def main() -> None:
    st.set_page_config(
        page_title="DD4bench Dashboard",
        page_icon="⚡",
        layout="wide",
    )

    config = Config.from_env()
    detector_dir: str | None = None

    with st.sidebar:
        st.header("Data Source")
        if st.button("Refresh Data", width='content'):
            st.cache_data.clear()
            st.rerun()

        if config.data_url:
            # ── Remote mode: fetch all runs from WebEOS ────────────────────
            st.caption(f"WebEOS: `{config.data_url}`")
            try:
                detectors = _cached_list_detectors(config.data_url)
            except Exception as err:
                st.error(f"Failed to list detectors: {err}")
                return
            if not detectors:
                st.error("No detectors found at the configured WebEOS URL.")
                return
            detector = st.selectbox("Detector", detectors)
            if not detector:
                return
            try:
                detector_dir = _cached_download_all_runs(config.data_url, detector)
            except Exception as err:
                st.error(f"Failed to download runs for detector {detector}: {err}")
                return
            run_meta = list_run_metadata(detector_dir)
            if not run_meta:
                st.warning(f"No runs found for detector '{detector}'.")
                return

            # ── Platform selector ──────────────────────────────────────────
            platforms = sorted({m["platform"] for m in run_meta if m["platform"]})
            if not platforms:
                st.warning("No valid platforms found in run metadata.")
                return
            selected_platform = st.selectbox("Platform", platforms)

            # ── Release selector (filtered by platform, newest first) ──────
            releases = sorted(
                {m["k4h_release"] for m in run_meta
                 if m["platform"] == selected_platform and m["k4h_release"]},
                reverse=True,
            )
            if not releases:
                st.warning(f"No releases found for platform '{selected_platform}'.")
                return
            selected_release = st.selectbox("Release", releases)

            # Pick the most recent run dir matching (platform, release)
            matching = [
                m for m in run_meta
                if m["platform"] == selected_platform
                and m["k4h_release"] == selected_release
            ]
            matching.sort(key=lambda m: m["run_date"] if pd.notna(m["run_date"]) else pd.Timestamp.min)
            if not matching:
                st.warning(
                    f"No runs found for platform '{selected_platform}' "
                    f"and release '{selected_release}'. Try refreshing the data."
                )
                return
            data_dir = matching[-1]["run_dir"]
        else:
            # ── Local mode: manual path ────────────────────────────────────
            data_dir = st.text_input("Data directory", value=config.data_dir)

        _data_path = Path(data_dir) if data_dir else None
        _path_valid = bool(_data_path and _data_path.exists() and _data_path.is_dir())
        if not _path_valid:
            st.error(f"Data directory not found: '{data_dir}'. Check for typos or a missing directory.")
            results = event_data = region_data = None
            available_labels: list[str] = []
        else:
            results = cached_load_results(data_dir)
            event_data = cached_load_event_timing(data_dir)
            region_data = cached_load_region_timing(data_dir)
            available_labels = collect_labels(results, event_data, region_data)

        if _path_valid and not available_labels:
            st.warning("No benchmark data found in the specified directory.")
        if not available_labels:
            selected_labels: list[str] = []
            baseline_label: str | None = None
        else:
            st.header("Filters")
            selected_labels = st.multiselect(
                "Configurations", available_labels, default=available_labels
            )
            baseline_label = (
                st.selectbox("Baseline", options=selected_labels, index=0)
                if selected_labels
                else None
            )

    st.title("Benchmark Dashboard")

    if not available_labels:
        st.info(
            f"No benchmark results found in **{data_dir}**. "
            "Set the `DD4BENCH_DATA_DIR` environment variable or update the path in the sidebar."
        )
        return

    tab_names = ["Trends", "Region Timing", "Run Overview", "Event Timing", "Event Memory"]
    if detector_dir is None:
        tab_names = tab_names[1:]  # Trends only makes sense with multi-run data

    tabs = st.tabs(tab_names)
    tab_idx = 0

    if detector_dir is not None:
        with tabs[tab_idx]:
            trend_data = cached_load_trend_results(detector_dir)
            trends.render(trend_data, selected_labels)
        tab_idx += 1

    with tabs[tab_idx]:
        region_timing.render(region_data, selected_labels)
    with tabs[tab_idx + 1]:
        overview.render(results, selected_labels, baseline_label)
    with tabs[tab_idx + 2]:
        event_timing.render(event_data, selected_labels, baseline_label)
    with tabs[tab_idx + 3]:
        event_memory.render(event_data, selected_labels, baseline_label)


main()
