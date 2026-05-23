from __future__ import annotations

import streamlit as st

from dd4bench.analysis.plots import plot_region_timing


def render(
    region_data: dict | None,
    selected_labels: list[str],
) -> None:
    if region_data is None:
        st.info("No region timing data available in the selected directory.")
        return
    if not selected_labels:
        st.info("Select at least one run in the sidebar.")
        return

    # Only offer labels that actually have region data so the user never
    # selects a config that immediately triggers the "no data" warning.
    filtered_labels = [lbl for lbl in selected_labels if lbl in region_data and region_data[lbl]]
    if not filtered_labels:
        st.info("No region timing data available for any of the selected configurations.")
        return

    col_cfg, col_attr, col_topn, col_warmup = st.columns([2, 2, 2, 1])
    with col_cfg:
        config = st.selectbox("Configuration", filtered_labels, key="region_config")
    with col_attr:
        attribution = st.radio(
            "Attribution",
            options=["at_location", "by_birth"],
            format_func=lambda x: "At location" if x == "at_location" else "By birth",
            horizontal=True,
            key="region_attr",
        )
    with col_topn:
        top_n = st.slider("Top N detectors", min_value=3, max_value=15, value=8, key="region_topn")
    with col_warmup:
        exclude_warmup = st.toggle("Exclude event 0 (warmup)", value=True, key="region_warmup")

    fig = plot_region_timing(
        region_data,
        labels=[config],
        show="both",
        attribution=attribution,
        top_n=top_n,
        exclude_events=[0] if exclude_warmup else None,
    )
    st.plotly_chart(fig, width="stretch")
