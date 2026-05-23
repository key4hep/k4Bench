from __future__ import annotations

import streamlit as st

from dd4bench.analysis.plots import plot_event_timing
from stats import build_event_stats_table, select_top_n_by_ratio, style_stats_table


def render(
    event_data: dict | None,
    selected_labels: list[str],
    baseline_label: str | None,
) -> None:
    if event_data is None:
        st.info("No event timing data available in the selected directory.")
        return
    if not selected_labels:
        st.info("Select at least one run in the sidebar.")
        return

    col_topn, _, col_warmup = st.columns([2, 1, 1])
    with col_warmup:
        exclude_warmup = st.toggle("Exclude event 0 (warmup)", value=True, key="evt_timing_warmup")
    with col_topn:
        max_n = len(selected_labels)
        # min_value must be ≤ session-state value at all times; when only one
        # label is selected the minimum is 1 (slider is disabled anyway).
        min_n = min(2, max_n)
        # Reset the slider to show all configs whenever the selection size changes
        # so a stale session-state value never silently hides newly added configs.
        if st.session_state.get("_evt_timing_max_n") != max_n:
            st.session_state["evt_timing_topn"] = max(min_n, min(5, max_n))
            st.session_state["_evt_timing_max_n"] = max_n
        top_n = st.slider(
            "Top N runs by timing ratio",
            min_value=min_n, max_value=max(3, max_n),
            value=min(5, max_n),
            key="evt_timing_topn",
            disabled=(max_n <= 2),
        )

    display_labels = select_top_n_by_ratio(
        event_data, selected_labels, "event_time_s", "s", baseline_label, exclude_warmup, top_n
    )

    fig = plot_event_timing(
        event_data,
        labels=display_labels,
        baseline_label=baseline_label,
        show="both",
        exclude_events=[0] if exclude_warmup else None,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Statistics")
    stats = build_event_stats_table(
        event_data, display_labels, "event_time_s", "s", baseline_label, exclude_warmup
    )
    if not stats.empty:
        st.dataframe(style_stats_table(stats), width="stretch")
    else:
        st.info("No valid statistics available (missing or empty data).")
