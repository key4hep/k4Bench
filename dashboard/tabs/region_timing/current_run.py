"""Current-run view — single-run region timing bar chart."""
from __future__ import annotations

import streamlit as st

from k4bench.analysis.plots import plot_region_timing
from ui_utils import _PALETTES, _PALETTE_NAMES, _auto_palette_index

from ._common import _ATTRIBUTION_HELP


def _render_current_run(region_data: dict, selected_labels: list[str]) -> None:
    """Render the current-run region timing view (existing behaviour)."""
    filtered_labels = [lbl for lbl in selected_labels if lbl in region_data and region_data[lbl]]
    if not filtered_labels:
        st.info("No region timing data available for any of the selected configurations.")
        return

    col_cfg, col_attr = st.columns([2, 2])
    with col_cfg:
        config = st.selectbox("Configuration", filtered_labels, key="region_config")
    with col_attr:
        attribution = st.selectbox(
            "Attribution",
            options=["at_location", "by_birth"],
            format_func=lambda x: "At location" if x == "at_location" else "By birth",
            key="region_attr",
            help=_ATTRIBUTION_HELP,
        )

    col_topn, col_pal = st.columns([2, 2])
    with col_topn:
        top_n = st.slider("Top N detectors", min_value=3, max_value=15, value=8, key="region_topn")
    with col_pal:
        palette_name = st.selectbox(
            "Colour palette",
            options=_PALETTE_NAMES,
            index=_auto_palette_index(top_n),
            key="region_cur_palette",
        )

    fig = plot_region_timing(
        region_data,
        labels=[config],
        show="both",
        attribution=attribution,
        top_n=top_n,
        exclude_events=[0],
        palette=_PALETTES[palette_name],
    )
    st.plotly_chart(fig, width="stretch", key="region_current_chart")
