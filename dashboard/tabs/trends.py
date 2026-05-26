from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.colors as _pc
from plotly.colors import qualitative as _ql
from plotly.subplots import make_subplots

from dd4bench.analysis.plots._theme import PALETTE, _TEMPLATE


def _to_rgba(color: str, alpha: float) -> str:
    """Apply alpha to any Plotly colour string, returning rgba()."""
    color = color.strip()
    if color.startswith("rgba("):
        return color                                   # already has alpha
    try:
        r, g, b = (
            _pc.hex_to_rgb(color)
            if color.startswith("#")
            else _pc.unlabel_rgb(color)                # handles rgb(...)
        )
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return color                                   # unknown format, pass through


_METRICS = [
    ("wall_time_s",    "Wall Time (s)"),
    ("peak_rss_mb",    "Peak RSS (MB)"),
    ("events_per_sec", "Throughput (ev/s)"),
]

_PALETTES = {
    "Matplotlib": PALETTE,
    "Plotly":     _ql.Plotly,
    "D3":         _ql.D3,
    "G10":        _ql.G10,
    "Dark24":     _ql.Dark24,
    "Light24":    _ql.Light24,
    "Alphabet":   _ql.Alphabet,
    "Safe":       _ql.Safe,
    "Bold":       _ql.Bold,
}

# Secondary differentiators for when colour alone doesn't scale (15+ configs).
# Dash and symbol cycle once per full palette sweep.
_DASHES  = ["solid", "dash", "dot", "dashdot"]
_SYMBOLS = ["circle", "square", "diamond", "cross",
            "triangle-up", "star", "pentagon", "hexagon"]


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
        )
    with ctrl_r:
        alpha = st.slider(
            "Opacity",
            min_value=0.1, max_value=1.0,
            value=0.75, step=0.05,
        )
    with ctrl_s:
        smooth = st.toggle("Smooth lines", value=False)

    palette      = _PALETTES[palette_name]
    line_shape   = "spline" if smooth else "linear"
    line_alpha   = alpha
    marker_alpha = max(0.1, alpha - 0.2)
    use_dash     = style_cycling in ("Colour + Dash",   "Colour + Dash + Marker")
    use_marker   = style_cycling in ("Colour + Marker", "Colour + Dash + Marker")

    # ── Data prep ─────────────────────────────────────────────────────────────
    df = trend_df[trend_df["label"].isin(selected_labels)].copy()
    df["k4h_release_date"] = pd.to_datetime(df["k4h_release_date"]).dt.normalize()
    df["run_date"]         = pd.to_datetime(df["run_date"]).dt.normalize()
    df["x_date"] = df["k4h_release_date"].fillna(df["run_date"])
    df = df.dropna(subset=["x_date"])
    # When multiple CI runs share the same nightly tag, keep only the latest run.
    # idxmax skips NaN and doesn't rely on sort order, so the full row is always
    # from the run with the highest (most recent) run_date.
    df = df.loc[df.groupby(["label", "x_date"])["run_date"].idxmax()].reset_index(drop=True)
    df["run_date_str"] = df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
    if df.empty:
        st.warning("No trend data for the selected configurations.")
        return

    present_metrics = [(col, label) for col, label in _METRICS if col in df.columns]
    if not present_metrics:
        st.warning("No supported metrics found for the current dataframe.")
        return

    unique_dates = sorted(pd.to_datetime(df["x_date"].dropna().unique()))
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
        n_colors     = len(palette)
        cycle        = cfg_idx // n_colors      # how many full palette sweeps so far
        color        = palette[cfg_idx % n_colors]
        line_color   = _to_rgba(color, line_alpha)
        marker_color = _to_rgba(color, marker_alpha)
        dash         = _DASHES [cycle % len(_DASHES) ] if use_dash   else "solid"
        symbol       = _SYMBOLS[cycle % len(_SYMBOLS)] if use_marker else "circle"
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
    plot_h      = fig_height - t_margin - b_margin   # actual chart area height
    y_legend    = -(x_tick_gap / plot_h)

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
