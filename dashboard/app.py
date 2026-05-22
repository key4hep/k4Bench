from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import Config
from data import (
    cached_load_event_timing,
    cached_load_region_timing,
    cached_load_results,
    collect_labels,
)
from tabs import event_memory, event_timing, overview, region_timing


def main() -> None:
    st.set_page_config(
        page_title="DD4bench Dashboard",
        page_icon="⚡",
        layout="wide",
    )

    config = Config.from_env()

    with st.sidebar:
        st.header("Data Source")
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
                "Runs", available_labels, default=available_labels
            )
            baseline_label = (
                st.selectbox("Baseline run", options=selected_labels, index=0)
                if selected_labels
                else None
            )

    det_name = Path(data_dir).name if data_dir else "DD4bench"
    st.title(f"{det_name} — Benchmark Dashboard")

    if not available_labels:
        st.info(
            f"No benchmark results found in **{data_dir}**. "
            "Set the `DD4BENCH_DATA_DIR` environment variable or update the path in the sidebar."
        )
        return

    tab_region, tab_overview, tab_evt_timing, tab_evt_memory = st.tabs(
        ["Region Timing", "Run Overview", "Event Timing", "Event Memory"]
    )

    with tab_region:
        region_timing.render(region_data, selected_labels)
    with tab_overview:
        overview.render(results, selected_labels, baseline_label)
    with tab_evt_timing:
        event_timing.render(event_data, selected_labels, baseline_label)
    with tab_evt_memory:
        event_memory.render(event_data, selected_labels, baseline_label)


main()
