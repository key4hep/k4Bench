from __future__ import annotations

import pandas as pd
import streamlit as st

from k4bench.analysis.plots import plot_event_timing
from stats import build_event_stats_table, select_top_n_by_ratio, style_stats_table
from ui_utils import _is_valid_df, _PALETTES, _PALETTE_NAMES, _auto_palette_index, _render_historical_trends


_STAT_COLS = {
    "Mean":   "mean_time_s",
    "Median": "median_time_s",
    "P95":    "p95_time_s",
}

_HIST_STATS = [
    ("median_time_s", "Median time (s)"),
    ("mean_time_s",   "Mean time (s)"),
    ("std_time_s",    "Std dev (s)"),
]


def _render_current_run(
    event_data: dict,
    selected_labels: list[str],
) -> None:
    """Render the current-run per-event timing view."""
    col_bl, col_topn, col_pal = st.columns([2, 2, 2])
    with col_bl:
        baseline_label = (
            st.selectbox(
                "Baseline",
                options=selected_labels,
                index=0,
                key="evt_timing_baseline",
                help=(
                    "The configuration used as the reference. "
                    "The lower panel shows every other config's timing "
                    "as a ratio relative to this one — values above 1 are slower, "
                    "below 1 are faster."
                ),
            )
            if selected_labels
            else None
        )
    with col_topn:
        max_n = len(selected_labels)
        if max_n > 2:
            if st.session_state.get("_evt_timing_max_n") != max_n:
                st.session_state["evt_timing_topn"] = min(5, max_n)
                st.session_state["_evt_timing_max_n"] = max_n
            top_n = st.slider(
                "Top N runs by timing ratio",
                min_value=2,
                max_value=max_n,
                key="evt_timing_topn",
                help=(
                    "When many configurations are selected, shows only the N "
                    "with the largest absolute deviation from the baseline. "
                    "Keeps the plot readable when dozens of configs are loaded."
                ),
            )
        else:
            top_n = max_n
            st.session_state["_evt_timing_max_n"] = max_n
    with col_pal:
        palette_name = st.selectbox(
            "Colour palette",
            options=_PALETTE_NAMES,
            index=_auto_palette_index(top_n),
            key="evt_timing_palette",
        )

    display_labels = select_top_n_by_ratio(
        event_data, selected_labels, "event_time_s", "s", baseline_label, True, top_n
    )

    fig = plot_event_timing(
        event_data,
        labels=display_labels,
        baseline_label=baseline_label,
        show="both",
        exclude_events=[0],
        palette=_PALETTES[palette_name],
    )
    st.plotly_chart(fig, width="stretch", key="evt_timing_current_chart")

    st.subheader("Statistics")
    stats = build_event_stats_table(
        event_data, display_labels, "event_time_s", "s", baseline_label, True
    )
    if not stats.empty:
        st.dataframe(style_stats_table(stats), width="stretch")
    else:
        st.info("No valid statistics available (missing or empty data).")


def _render_historical(
    trend_event_df: pd.DataFrame,
    selected_labels: list[str],
) -> None:
    """Render the historical event timing trends view (3-panel: Median | Mean | Std)."""
    if not _is_valid_df(trend_event_df):
        st.info(
            "No event timing trend data in the selected window. "
            "Widen the trend window in the sidebar."
        )
        return
    avail_labels = sorted(trend_event_df["label"].unique())
    filtered_labels = [lbl for lbl in selected_labels if lbl in avail_labels]
    if not filtered_labels:
        st.info("No historical event timing data available for the selected configurations.")
        return
    present_stats = [(col, lbl) for col, lbl in _HIST_STATS if col in trend_event_df.columns]
    if not present_stats:
        st.info("No historical event timing statistics available.")
        return

    _render_historical_trends(
        trend_event_df, filtered_labels, present_stats,
        std_col="std_time_s",
        n_col_candidates=["n_events"],
        unit="s",
        key_prefix="evt_timing_hist",
        no_data_msg="No event timing trend data for the selected configurations.",
    )


def render(
    event_data: dict | None,
    trend_event_df: pd.DataFrame | None,
    selected_labels: list[str],
    trends_enabled: bool = False,
) -> None:
    if event_data is None and not trends_enabled:
        st.info("No event timing data available in the selected directory.")
        return
    if not selected_labels:
        st.info("Select at least one run in the sidebar.")
        return

    # The "Historical Trends" option is gated on remote mode (not on the current
    # window's data) so the view selector stays put when the trend window changes.
    if trends_enabled:
        view = st.radio(
            "View",
            options=["Current Run", "Historical Trends"],
            horizontal=True,
            key="evt_timing_view_mode",
        )
    else:
        view = "Current Run"

    if view == "Current Run":
        if event_data is None:
            st.info("No event timing data available in the selected directory.")
        else:
            _render_current_run(event_data, selected_labels)
    else:
        _render_historical(trend_event_df, selected_labels)
