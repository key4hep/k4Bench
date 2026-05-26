from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative as _ql
from plotly.subplots import make_subplots

from dd4bench.analysis.plots._theme import PALETTE, _TEMPLATE


def _to_rgba(color: str, alpha: float) -> str:
    """Convert a hex *or* rgb() colour string to an rgba() string."""
    color = color.strip()
    if color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    elif color.startswith("rgb("):
        r, g, b = [int(x.strip()) for x in color[4:-1].split(",")]
    else:
        return color  # unknown format — pass through unchanged
    return f"rgba({r},{g},{b},{alpha})"


_METRICS = [
    ("wall_time_s",    "Wall Time (s)"),
    ("peak_rss_mb",    "Peak RSS (MB)"),
    ("events_per_sec", "Throughput (ev/s)"),
]

_PALETTES = {
    "Matplotlib": PALETTE,          # default — matches the rest of the dashboard
    "Dark24":     _ql.Dark24,       # 24 distinct colours, good for many configs
    "Vivid":      _ql.Vivid,        # high-contrast, punchy
    "Safe":       _ql.Safe,         # colourblind-friendly
}


def render(trend_df: pd.DataFrame | None, selected_labels: list[str]) -> None:
    if trend_df is None:
        st.info("No trend data available. Run the nightly benchmark at least once.")
        return
    if not selected_labels:
        st.info("Select at least one configuration in the sidebar.")
        return

    # ── Display controls ───────────────────────────────────────────────────────
    ctrl_l, ctrl_r, _ = st.columns([1.2, 1.2, 2.6])
    with ctrl_l:
        palette_name = st.selectbox(
            "Colour palette",
            options=list(_PALETTES.keys()),
            index=0,
        )
    with ctrl_r:
        line_style = st.radio(
            "Line style",
            options=["Linear", "Spline"],
            horizontal=True,
            index=0,
        )

    palette    = _PALETTES[palette_name]
    line_shape = "spline" if line_style == "Spline" else "linear"

    # ── Data prep ─────────────────────────────────────────────────────────────
    df = trend_df[trend_df["label"].isin(selected_labels)].copy()
    df["k4h_release_date"] = pd.to_datetime(df["k4h_release_date"]).dt.normalize()
    df["run_date"]         = pd.to_datetime(df["run_date"]).dt.normalize()
    df["x_date"] = df["k4h_release_date"].fillna(df["run_date"])
    df = df.dropna(subset=["x_date"])
    # When multiple CI runs share the same nightly tag, keep only the latest run.
    df = df.sort_values("run_date").groupby(["label", "x_date"], as_index=False).last()
    df["run_date_str"] = df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
    if df.empty:
        st.warning("No trend data for the selected configurations.")
        return

    present_metrics = [(col, label) for col, label in _METRICS if col in df.columns]
    if not present_metrics:
        st.warning("No supported metrics found for the current dataframe.")
        return

    unique_dates = sorted(df["x_date"].dropna().unique())
    tick_labels  = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]
    n = len(present_metrics)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[f"<b>{label}</b>" for _, label in present_metrics],
        horizontal_spacing=0.15,
    )

    for cfg_idx, cfg_label in enumerate(selected_labels):
        cfg_df = df[df["label"] == cfg_label].sort_values("x_date")
        if cfg_df.empty:
            continue
        color        = palette[cfg_idx % len(palette)]
        line_color   = _to_rgba(color, 0.75)
        marker_color = _to_rgba(color, 0.55)
        custom = cfg_df[["run_date_str", "k4h_release"]].values

        for col_idx, (metric_col, metric_label) in enumerate(present_metrics):
            fig.add_trace(
                go.Scatter(
                    x=cfg_df["x_date"],
                    y=cfg_df[metric_col],
                    mode="lines+markers",
                    name=cfg_label,
                    legendgroup=cfg_label,
                    showlegend=(col_idx == 0),
                    line=dict(color=line_color, width=2, shape=line_shape),
                    marker=dict(size=7, color=marker_color, line=dict(color=color, width=1.5)),
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{cfg_label}</b><br>"
                        "Tag: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                        f"{metric_label}: %{{y:.4g}}<br>"
                        "CI run: %{customdata[0]}<extra></extra>"
                    ),
                ),
                row=1, col=col_idx + 1,
            )

    # ── Axes ──────────────────────────────────────────────────────────────────
    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=unique_dates,
        ticktext=tick_labels,
        tickangle=-30,
        title_text="Key4hep Nightly Tag",
    )
    for col_idx, (_, metric_label) in enumerate(present_metrics):
        ykey = "yaxis" if col_idx == 0 else f"yaxis{col_idx + 1}"
        fig.update_layout({ykey: {"title": {"text": metric_label}}})

    # ── Layout & legend ───────────────────────────────────────────────────────
    n_cfg       = len(selected_labels)
    legend_rows = max(1, -(-n_cfg // 4))
    legend_px   = 24 + legend_rows * 32
    t_margin    = 40
    x_tick_gap  = 140
    b_margin    = x_tick_gap + legend_px + 20
    fig_height  = 380 + t_margin + b_margin
    y_legend    = -(x_tick_gap / 380)

    fig.update_layout(
        template=_TEMPLATE,
        height=fig_height,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=y_legend,
            xanchor="center",
            x=0.5,
            entrywidth=220,
            entrywidthmode="pixels",
            tracegroupgap=0,
            font=dict(size=13),
        ),
    )

    st.plotly_chart(fig, width='stretch')
