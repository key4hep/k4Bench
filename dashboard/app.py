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
    load_machine_info,
)
from tabs import event_memory, event_timing, machine_info, overview, region_timing, trends


# ── Cached remote helpers ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching detectors...", ttl=3600)
def _cached_list_detectors(base_url: str) -> list[str]:
    from remote import list_detectors
    return list_detectors(base_url)


@st.cache_data(show_spinner="Fetching platforms...", ttl=3600)
def _cached_list_platforms(base_url: str, detector: str) -> list[str]:
    from remote import list_platforms
    return list_platforms(base_url, detector)


@st.cache_data(show_spinner="Fetching stacks...", ttl=3600)
def _cached_list_stacks(base_url: str, detector: str, platform: str) -> list[str]:
    from remote import list_stacks
    return list_stacks(base_url, detector, platform)


@st.cache_data(show_spinner="Fetching samples...", ttl=3600)
def _cached_list_samples(base_url: str, detector: str, platform: str, stack: str) -> list[str]:
    from remote import list_samples
    return list_samples(base_url, detector, platform, stack)


@st.cache_data(show_spinner="Downloading runs...", ttl=3600)
def _cached_download_all_runs(
    base_url: str, detector: str, platform: str, stack: str, sample: str
) -> str:
    from remote import download_all_runs
    return str(download_all_runs(base_url, detector, platform, stack, sample))


@st.cache_data(show_spinner="Downloading trend data (all stacks)...", ttl=3600)
def _cached_download_all_stacks(
    base_url: str, detector: str, platform: str, sample: str
) -> str:
    from remote import download_all_stacks_for_sample
    return str(download_all_stacks_for_sample(base_url, detector, platform, sample))


def main() -> None:
    st.set_page_config(
        page_title="DD4bench Dashboard",
        page_icon="⚡",
        layout="wide",
    )

    config = Config.from_env()

    # ``trends_dir``  — flat temp dir with ALL stacks' runs for this (detector, platform, sample).
    #                   Used exclusively by the Trends tab.
    # ``sample_dir``  — temp dir with runs for the selected (detector, platform, stack, sample).
    #                   Used to pick the most recent run for single-run tabs.
    # ``data_dir``    — path to the selected date-level run dir (single-run tabs).
    trends_dir: str | None = None
    sample_dir: str | None = None
    data_dir:   str | None = None
    # Metadata for the selected run (used by Machine Info tab)
    selected_run_meta: dict | None = None

    with st.sidebar:
        st.header("Data Source")
        if st.button("Refresh Data", width="content"):
            st.cache_data.clear()
            st.rerun()

        if config.data_url:
            # ── Remote mode ────────────────────────────────────────────────────
            st.caption(f"WebEOS: `{config.data_url}`")

            # Detector
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

            # Platform
            try:
                platforms = _cached_list_platforms(config.data_url, detector)
            except Exception as err:
                st.error(f"Failed to list platforms: {err}")
                return
            if not platforms:
                st.warning(f"No platforms found for detector '{detector}'.")
                return
            platform = st.selectbox("Platform", platforms)
            if not platform:
                return

            # Stack
            try:
                stacks = _cached_list_stacks(config.data_url, detector, platform)
            except Exception as err:
                st.error(f"Failed to list stacks: {err}")
                return
            if not stacks:
                st.warning(f"No stacks found for '{detector} / {platform}'.")
                return
            stack = st.selectbox("Stack", stacks)
            if not stack:
                return

            # Sample
            try:
                samples = _cached_list_samples(config.data_url, detector, platform, stack)
            except Exception as err:
                st.error(f"Failed to list samples: {err}")
                return
            if not samples:
                st.warning(f"No samples found for '{detector} / {platform} / {stack}'.")
                return
            sample = st.selectbox("Sample", samples)
            if not sample:
                return

            # Download runs for the selected stack (single-run tabs)
            try:
                sample_dir = _cached_download_all_runs(
                    config.data_url, detector, platform, stack, sample
                )
            except Exception as err:
                st.error(f"Failed to download runs: {err}")
                return

            # Download runs across ALL stacks for this sample (Trends tab)
            try:
                trends_dir = _cached_download_all_stacks(
                    config.data_url, detector, platform, sample
                )
            except Exception as err:
                st.warning(f"Could not load cross-stack trend data: {err}")
                trends_dir = None

            run_meta = list_run_metadata(sample_dir)
            if not run_meta:
                st.warning("No runs found for the selected combination.")
                return

            # Pick the most recent run date
            run_meta_sorted = sorted(
                run_meta,
                key=lambda m: m["run_date"] if pd.notna(m["run_date"]) else pd.Timestamp.min,
            )
            selected_run_meta = run_meta_sorted[-1]
            data_dir = selected_run_meta["run_dir"]

        else:
            # ── Local mode ─────────────────────────────────────────────────────
            data_dir = st.text_input("Data directory", value=config.data_dir)

        # ── Validate data_dir & load data ──────────────────────────────────────
        _data_path  = Path(data_dir) if data_dir else None
        _path_valid = bool(_data_path and _data_path.exists() and _data_path.is_dir())

        if not _path_valid:
            st.error(
                f"Data directory not found: '{data_dir}'. "
                "Check for typos or a missing directory."
            )
            results = event_data = region_data = None
            available_labels: list[str] = []
        else:
            results      = cached_load_results(data_dir)
            event_data   = cached_load_event_timing(data_dir)
            region_data  = cached_load_region_timing(data_dir)
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
        path_hint = f" in **{data_dir}**" if data_dir else ""
        st.info(
            f"No benchmark results found{path_hint}. "
            "Set the `DD4BENCH_DATA_DIR` environment variable or update the path in the sidebar."
        )
        return

    # ── Build tab list ─────────────────────────────────────────────────────────
    tab_names = ["Trends", "Region Timing", "Run Overview", "Event Timing", "Event Memory", "Machine Info"]
    if trends_dir is None:
        # Trends only makes sense with multi-run (remote) data
        tab_names = tab_names[1:]

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # Trends (remote only) — uses all stacks so history is complete
    if trends_dir is not None:
        with tabs[tab_idx]:
            trend_data = cached_load_trend_results(trends_dir)
            trends.render(trend_data, selected_labels)
        tab_idx += 1

    # Region Timing
    with tabs[tab_idx]:
        region_timing.render(region_data, selected_labels)
    tab_idx += 1

    # Run Overview
    with tabs[tab_idx]:
        overview.render(results, selected_labels, baseline_label)
    tab_idx += 1

    # Event Timing
    with tabs[tab_idx]:
        event_timing.render(event_data, selected_labels, baseline_label)
    tab_idx += 1

    # Event Memory
    with tabs[tab_idx]:
        event_memory.render(event_data, selected_labels, baseline_label)
    tab_idx += 1

    # Machine Info
    with tabs[tab_idx]:
        minfo = load_machine_info(data_dir) if _path_valid else None
        machine_info.render(minfo, run_meta=selected_run_meta)


main()
