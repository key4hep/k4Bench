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

    df = trend_df[trend_df["label"].isin(selected_labels)].copy()
    df["k4h_release_date"] = pd.to_datetime(df["k4h_release_date"]).dt.normalize()
    df["run_date"] = pd.to_datetime(df["run_date"]).dt.normalize()
    # Use the release date as x-axis so multiple CI runs of the same release
    # map to the same x-position.  Fall back to run_date for old runs that
    # predate the k4h_release_date metadata field — better than dropping them.
    df["x_date"] = df["k4h_release_date"].fillna(df["run_date"])
    df = df.dropna(subset=["x_date"])
    # Pre-format run_date as string for clean hover display.
    df["run_date_str"] = df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
    if df.empty:
        st.warning("No trend data for the selected configurations.")
        return

    present_metrics = [(col, label) for col, label in _METRICS if col in df.columns]
    n = len(present_metrics)
    if n == 0:
        st.warning("No supported metrics found for the current dataframe.")
        return

    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=[label for _, label in present_metrics],
        shared_xaxes=True,
        vertical_spacing=0.14,
    )

    for cfg_idx, cfg_label in enumerate(selected_labels):
        cfg_df = df[df["label"] == cfg_label].sort_values("x_date")
        if cfg_df.empty:
            continue
        color = PALETTE[cfg_idx % len(PALETTE)]
        # customdata columns: [run_date_str, k4h_release]
        custom = cfg_df[["run_date_str", "k4h_release"]].values
        for row_idx, (col, metric_label) in enumerate(present_metrics):
            fig.add_trace(
                go.Scatter(
                    x=cfg_df["x_date"],
                    y=cfg_df[col],
                    mode="lines+markers",
                    name=cfg_label,
                    legendgroup=cfg_label,
                    showlegend=(row_idx == 0),
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{cfg_label}</b><br>"
                        "Release: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                        f"{metric_label}: %{{y:.4g}}<br>"
                        "CI run: %{customdata[0]}<extra></extra>"
                    ),
                ),
                row=row_idx + 1,
                col=1,
            )

    # ── Legend sizing ──────────────────────────────────────────────────────
    n_cfg = len(selected_labels)
    # How many legend entries fit per row (generous: 3 wide)
    legend_rows = max(1, -(-n_cfg // 3))
    legend_px = 30 + legend_rows * 24   # px height of the legend block

    # ── Margins ────────────────────────────────────────────────────────────
    t_margin = 40
    # x-tick labels are rotated; allow 80 px below the bottom axis, then the
    # legend, then a small buffer.
    x_tick_gap_px = 80
    b_margin = x_tick_gap_px + legend_px + 30
    fig_height = 280 * n + t_margin + b_margin

    # ── Legend y position in "paper" coords ────────────────────────────────
    # paper y=0 is the bottom edge of the plot area; negative values are inside
    # the bottom margin.  We want the top of the legend to start just below the
    # x-tick labels, i.e. x_tick_gap_px into the margin.
    plot_area_h = fig_height - t_margin - b_margin   # ≈ 280 * n
    y_legend = -(x_tick_gap_px / plot_area_h)

    fig.update_layout(
        template=_TEMPLATE,
        height=fig_height,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=y_legend,
            xanchor="center",
            x=0.5,
            tracegroupgap=4,
        ),
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
    )

    # ── X-axis: one tick per unique x-date, no repeats ─────────────────────
    unique_dates = sorted(df["x_date"].dropna().unique())
    tick_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]
    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=unique_dates,
        ticktext=tick_labels,
        tickangle=-30,
    )

    st.plotly_chart(fig, width='stretch')
