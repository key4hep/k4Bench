"""Region Timing tab.

Four independent views — Current Run, Attribution Analysis, Step Analysis and
Historical Trends — each living in its own module. ``render`` is the dispatcher
that builds the view selector and delegates to the chosen view.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .attribution import _render_attribution_analysis
from .current_run import _render_current_run
from .historical import _render_historical
from .step_analysis import _render_step_analysis


def render(
    region_data: dict | None,
    trend_region_df: pd.DataFrame | None,
    selected_labels: list[str],
    trends_enabled: bool = False,
    reliability: dict[str, bool | None] | None = None,
) -> None:
    if region_data is None and not trends_enabled:
        st.info("No region timing data available in the selected directory.")
        return
    if not selected_labels:
        st.info("Select at least one run in the sidebar.")
        return

    # Build view options dynamically based on available data
    # Order: current-run analyses first, then historical trends.
    # "Historical Trends" is gated on remote mode (not on the current window's
    # data) so the selector stays stable when the trend window changes.
    view_options: list[str] = ["Current Run"]
    if region_data is not None:
        view_options.append("Attribution Analysis")
        # Step Analysis requires interval_counts — check any label has it
        has_steps = any(
            region_data[lbl].get("steps") is not None
            for lbl in selected_labels
            if lbl in region_data
        )
        if has_steps:
            view_options.append("Step Analysis")
    if trends_enabled:
        view_options.append("Historical Trends")

    view = (
        st.radio("View", options=view_options, horizontal=True, key="region_view_mode")
        if len(view_options) > 1
        else view_options[0]
    )

    if view == "Current Run":
        if region_data is None:
            st.info("No region timing data available in the selected directory.")
        else:
            _render_current_run(region_data, selected_labels)
    elif view == "Historical Trends":
        _render_historical(trend_region_df, selected_labels, reliability)
    elif view == "Attribution Analysis":
        _render_attribution_analysis(region_data, selected_labels)
    elif view == "Step Analysis":
        _render_step_analysis(region_data, selected_labels)
