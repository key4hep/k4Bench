from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dd4bench.analysis.plots._theme import PALETTE, _TEMPLATE

_METRICS = [
    ("wall_time_s",    "Wall Time (s)"),
    ("peak_rss_mb",    "Peak RSS (MB)"),
    ("events_per_sec", "Throughput (ev/s)"),
]


def render(trend_df: pd.DataFrame | None, selected_labels: list[str]) -> None:
    if trend_df is None:
        st.info("No trend data available. Run the nightly benchmark at least once.")
        return
    if not selected_labels:
        st.info("Select at least one configuration in the sidebar.")
        return

    df = trend_df[trend_df["label"].isin(selected_labels)].dropna(subset=["run_date"])
    if df.empty:
        st.warning("No trend data for the selected configurations.")
        return

    present_metrics = [(col, label) for col, label in _METRICS if col in df.columns]
    n = len(present_metrics)

    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=[label for _, label in present_metrics],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    for cfg_idx, cfg_label in enumerate(selected_labels):
        cfg_df = df[df["label"] == cfg_label].sort_values("run_date")
        if cfg_df.empty:
            continue
        color = PALETTE[cfg_idx % len(PALETTE)]
        for row_idx, (col, metric_label) in enumerate(present_metrics):
            fig.add_trace(
                go.Scatter(
                    x=cfg_df["run_date"],
                    y=cfg_df[col],
                    mode="lines+markers",
                    name=cfg_label,
                    legendgroup=cfg_label,
                    showlegend=(row_idx == 0),
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    customdata=cfg_df["k4h_release"],
                    hovertemplate=(
                        f"<b>{cfg_label}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        f"{metric_label}: %{{y:.4g}}<br>"
                        "Release: %{customdata}<extra></extra>"
                    ),
                ),
                row=row_idx + 1,
                col=1,
            )

    fig.update_layout(
        title_text="Performance Trends Over Time",
        title_font=dict(size=16, color="#222222"),
        template=_TEMPLATE,
        height=280 * n + 100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=100, b=40),
    )
    fig.update_xaxes(tickformat="%Y-%m-%d", row=n, col=1)

    st.plotly_chart(fig, use_container_width=True)
