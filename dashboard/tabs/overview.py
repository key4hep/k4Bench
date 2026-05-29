from __future__ import annotations

import pandas as pd
import streamlit as st

from dd4bench.analysis.plots import plot_run_overview

_ALL_METRICS = [
    # Performance
    ("wall_time_s",              "Wall Time (s)"),
    ("events_per_sec",           "Throughput (ev/s)"),
    ("cpu_efficiency",           "CPU Efficiency"),
    # Resources
    ("user_cpu_s",               "User CPU (s)"),
    ("peak_rss_mb",              "Peak RSS (MB)"),
    ("involuntary_ctx_switches", "Involuntary Ctx Switches"),
]


def render(
    results: pd.DataFrame | None,
    selected_labels: list[str],
    baseline_label: str | None,
) -> None:
    if results is None:
        st.info("No results data available in the selected directory.")
        return
    if not selected_labels:
        st.info("Select at least one run in the sidebar.")
        return

    col_toggle, col_metrics = st.columns([1, 3])
    with col_toggle:
        relative = st.toggle("Relative to baseline", value=False)
    with col_metrics:
        metric_labels = [label for _, label in _ALL_METRICS]
        chosen_labels = st.multiselect("Metrics", options=metric_labels, default=metric_labels)

    chosen_metrics = [(col, label) for col, label in _ALL_METRICS if label in chosen_labels]
    metrics = chosen_metrics if chosen_metrics else None

    # Compute derived columns before handing the DataFrame to the plot function
    results = results.copy()
    if "user_cpu_s" in results.columns and "wall_time_s" in results.columns:
        results["cpu_efficiency"] = (
            results["user_cpu_s"] / results["wall_time_s"].replace(0, float("nan"))
        )

    fig = plot_run_overview(
        results,
        labels=selected_labels,
        metrics=metrics,
        relative=relative,
        baseline_label=baseline_label,
    )
    st.plotly_chart(fig, width="stretch", key="overview_chart")
