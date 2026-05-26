from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dd4bench.analysis.plots._theme import _TEMPLATE
from ui_utils import _DASHES, _PALETTES, _SYMBOLS, _to_rgba


_METRICS = [
    ("wall_time_s",    "Wall Time (s)"),
    ("user_cpu_s",     "User CPU (s)"),
    ("peak_rss_mb",    "Peak RSS (MB)"),
    ("events_per_sec", "Throughput (ev/s)"),
]


def _render_timeseries(
    df: pd.DataFrame,
    selected_labels: list[str],
    palette: list[str],
    line_shape: str,
    line_alpha: float,
    use_dash: bool,
    use_marker: bool,
) -> None:
    """Render the main time-series subplot figure."""
    marker_alpha = max(0.1, line_alpha - 0.2)

    present_metrics = [(col, label) for col, label in _METRICS if col in df.columns]
    if not present_metrics:
        st.warning("No supported metrics found for the current dataframe.")
        return

    unique_dates = sorted(pd.to_datetime(df["x_date"].dropna().unique()))
    tick_labels  = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]
    n = len(present_metrics)

    # Lay out metrics in a 2-column grid (ceil(n/2) rows × 2 cols)
    n_cols = min(n, 2)
    n_rows = -(-n // n_cols)   # ceiling division

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes="all",
        horizontal_spacing=0.10,
        vertical_spacing=0.08,
    )

    for cfg_idx, cfg_label in enumerate(selected_labels):
        cfg_df = df[df["label"] == cfg_label].sort_values("x_date")
        if cfg_df.empty:
            continue
        n_colors     = len(palette)
        cycle        = cfg_idx // n_colors
        color        = palette[cfg_idx % n_colors]
        line_color   = _to_rgba(color, line_alpha)
        marker_color = _to_rgba(color, marker_alpha)
        dash         = _DASHES [cycle % len(_DASHES) ] if use_dash   else "solid"
        symbol       = _SYMBOLS[cycle % len(_SYMBOLS)] if use_marker else "circle"
        custom = cfg_df[["run_date_str", "k4h_release"]].values

        for plot_idx, (metric_col, metric_label) in enumerate(present_metrics):
            row = plot_idx // n_cols + 1
            col = plot_idx %  n_cols + 1
            fig.add_trace(
                go.Scatter(
                    x=cfg_df["x_date"],
                    y=cfg_df[metric_col],
                    mode="lines+markers",
                    name=cfg_label,
                    legendgroup=cfg_label,
                    showlegend=(plot_idx == 0),
                    line=dict(color=line_color, width=2, shape=line_shape, dash=dash),
                    marker=dict(size=7, color=marker_color, symbol=symbol,
                                line=dict(color=color, width=1.5)),
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{cfg_label}</b><br>"
                        "Tag: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                        f"{metric_label}: %{{y:.4g}}<br>"
                        "CI run: %{customdata[0]}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

    # Apply ticks to all axes first, then hide them on the top row
    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=unique_dates,
        ticktext=tick_labels,
        tickangle=-30,
        title_text="Key4hep Nightly Tag",
    )
    if n_rows > 1:
        for col in range(1, n_cols + 1):
            fig.update_xaxes(
                showticklabels=False,
                title_text="",
                row=1, col=col,
            )
    for plot_idx, (_, metric_label) in enumerate(present_metrics):
        ykey = "yaxis" if plot_idx == 0 else f"yaxis{plot_idx + 1}"
        fig.update_layout({ykey: {"title": {"text": f"<b>{metric_label}</b>"}}})

    t_margin   = 40
    b_margin   = 100   # room for bottom-row x-tick labels only
    fig_height = n_rows * 380 + t_margin + b_margin

    fig.update_layout(
        template=_TEMPLATE,
        height=fig_height,
        margin=dict(l=20, r=200, t=t_margin, b=b_margin),
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=0.5,
            yanchor="middle",
            tracegroupgap=4,
            font=dict(size=13),
        ),
    )

    st.plotly_chart(fig, width="stretch")


def render(trend_df: pd.DataFrame | None, selected_labels: list[str]) -> None:
    if trend_df is None:
        st.info("No trend data available. Run the nightly benchmark at least once.")
        return
    if not selected_labels:
        st.info("Select at least one configuration in the sidebar.")
        return

    # ── Display controls ───────────────────────────────────────────────────────
    ctrl_l, ctrl_m, ctrl_r, ctrl_s = st.columns(4, vertical_alignment="bottom")
    with ctrl_l:
        palette_name = st.selectbox(
            "Colour palette",
            options=list(_PALETTES.keys()),
            index=0,
        )
    with ctrl_m:
        style_cycling = st.selectbox(
            "Style cycling",
            options=["Colour only", "Colour + Dash", "Colour + Marker", "Colour + Dash + Marker"],
            index=0,
            help=(
                "When the number of configurations exceeds the palette size, "
                "additional visual cues are layered on top of colour — "
                "dash pattern and/or marker shape — so every line stays "
                "distinguishable even with 20+ configs."
            ),
        )
    with ctrl_r:
        alpha = st.slider(
            "Opacity",
            min_value=0.1, max_value=1.0,
            value=0.75, step=0.05,
        )
    with ctrl_s:
        smooth = st.toggle("Smooth lines", value=False)

    palette    = _PALETTES[palette_name]
    line_shape = "spline" if smooth else "linear"
    use_dash   = style_cycling in ("Colour + Dash",   "Colour + Dash + Marker")
    use_marker = style_cycling in ("Colour + Marker", "Colour + Dash + Marker")

    # ── Data prep ─────────────────────────────────────────────────────────────
    # Dates and x_date are already normalised by cached_load_trend_results.
    df = trend_df[trend_df["label"].isin(selected_labels)].copy()
    df["x_date"]   = pd.to_datetime(df["x_date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.dropna(subset=["x_date"])
    # When multiple CI runs share the same nightly tag, keep only the latest run.
    df = df.loc[df.groupby(["label", "x_date"])["run_date"].idxmax()].reset_index(drop=True)
    df["run_date_str"] = df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
    if df.empty:
        st.warning("No trend data for the selected configurations.")
        return

    # ── Data freshness ────────────────────────────────────────────────────────
    earliest = df["x_date"].min()
    latest   = df["x_date"].max()
    if pd.notna(earliest) and pd.notna(latest):
        st.caption(
            f"Data range: **{earliest.strftime('%Y-%m-%d')}** → "
            f"**{latest.strftime('%Y-%m-%d')}** "
            f"({df['x_date'].nunique()} nightly tags)"
        )

    # ── Time-series plots ──────────────────────────────────────────────────────
    _render_timeseries(df, selected_labels, palette, line_shape, alpha, use_dash, use_marker)
