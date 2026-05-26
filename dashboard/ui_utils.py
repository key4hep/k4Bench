"""Shared UI utilities for dashboard tabs.

All palette/style constants and reusable Plotly helpers live here so each tab
imports from one place rather than duplicating the same definitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.colors as _pc
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative as _ql
from plotly.subplots import make_subplots

from dd4bench.analysis.plots._theme import PALETTE, _TEMPLATE


# ── Data-validation helpers ────────────────────────────────────────────────────

def _is_valid_df(df: "pd.DataFrame | None") -> bool:
    """Return *True* iff *df* is a non-``None``, non-empty :class:`~pandas.DataFrame`."""
    return df is not None and not df.empty


# ── Colour helper ──────────────────────────────────────────────────────────────

def _to_rgba(color: str, alpha: float) -> str:
    """Convert any Plotly colour string to ``rgba(…)`` with the given alpha."""
    color = color.strip()
    if color.startswith("rgba("):
        return color
    try:
        r, g, b = (
            _pc.hex_to_rgb(color)
            if color.startswith("#")
            else _pc.unlabel_rgb(color)
        )
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return color


# ── Style constants ────────────────────────────────────────────────────────────

_PALETTES: dict[str, list[str]] = {
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

_DASHES  = ["solid", "dash", "dot", "dashdot"]
_SYMBOLS = ["circle", "square", "diamond", "cross",
            "triangle-up", "star", "pentagon", "hexagon"]


# ── Legend layout helper ───────────────────────────────────────────────────────

def _bottom_legend_params(
    n_items: int,
    plot_h: int,
    *,
    x_tick_gap: int = 120,
    n_cols: int = 4,
    entry_width: int = 220,
    font_size: int = 13,
) -> tuple[int, dict]:
    """Compute bottom margin and legend dict for a fixed-width column-grid legend.

    Parameters
    ----------
    n_items : int
        Number of distinct legend entries.
    plot_h : int
        Height of the *data* area in pixels (used to compute the ``y`` anchor).
    x_tick_gap : int
        Vertical gap in pixels between the plot bottom edge and the legend top.
    n_cols : int
        Maximum number of legend entries per row.
    entry_width : int
        Width in pixels allocated to each entry.
    font_size : int
        Legend font size in points.

    Returns
    -------
    b_margin : int
        Bottom figure margin (pixels) to pass to ``fig.update_layout``.
    legend_dict : dict
        Keyword arguments suitable for ``fig.update_layout(legend=…)``.
    """
    n_rows    = max(1, -(-n_items // n_cols))   # ceiling division
    legend_px = 24 + n_rows * 32
    b_margin  = x_tick_gap + legend_px + 20
    y_legend  = -(x_tick_gap / plot_h)
    return b_margin, dict(
        orientation="h",
        yanchor="top",
        y=y_legend,
        xanchor="center",
        x=0.5,
        entrywidth=entry_width,
        entrywidthmode="pixels",
        tracegroupgap=0,
        font=dict(size=font_size),
    )


# ── Shared historical-trends renderer ─────────────────────────────────────────

def _render_historical_trends(
    trend_df: pd.DataFrame,
    filtered_labels: list[str],
    stats_spec: list[tuple[str, str]],
    *,
    std_col: str,
    n_col_candidates: list[str],
    unit: str,
    key_prefix: str,
    no_data_msg: str = "",
) -> None:
    """Render a multi-panel (Median | Mean | Std) historical trend figure.

    Shared implementation for the Event Timing and Event Memory historical
    sub-views.  Both tabs have an identical figure structure; only the column
    names, units, and Streamlit widget keys differ.

    Parameters
    ----------
    trend_df : pd.DataFrame
        Full long-form trend DataFrame (will be filtered to ``filtered_labels``).
    filtered_labels : list[str]
        Config labels to plot (already validated against ``trend_df``).
    stats_spec : list of (col, panel_title)
        Statistic columns to show and their subplot headings.
    std_col : str
        Name of the standard-deviation column used for error bars.
    n_col_candidates : list[str]
        Column names tried in order to find the event count (first hit wins).
    unit : str
        Physical unit string shown in hover-tips (e.g. ``"s"``, ``"MB"``).
    key_prefix : str
        Prefix for all Streamlit widget keys (must be unique per tab).
    no_data_msg : str
        Warning shown when the filtered DataFrame is empty after deduplication.
    """
    # ── Style controls ────────────────────────────────────────────────────────
    ctrl_l, ctrl_m, ctrl_r = st.columns(3, vertical_alignment="bottom")
    with ctrl_l:
        palette_name = st.selectbox(
            "Colour palette",
            options=list(_PALETTES.keys()),
            index=0,
            key=f"{key_prefix}_palette",
        )
    with ctrl_m:
        style_cycling = st.selectbox(
            "Style cycling",
            options=["Colour only", "Colour + Dash", "Colour + Marker", "Colour + Dash + Marker"],
            index=0,
            key=f"{key_prefix}_style",
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
            value=0.85, step=0.05,
            key=f"{key_prefix}_alpha",
        )

    palette    = _PALETTES[palette_name]
    use_dash   = style_cycling in ("Colour + Dash",   "Colour + Dash + Marker")
    use_marker = style_cycling in ("Colour + Marker", "Colour + Dash + Marker")

    # ── Data prep ─────────────────────────────────────────────────────────────
    df = trend_df[trend_df["label"].isin(filtered_labels)].copy()
    df["x_date"]  = pd.to_datetime(df["x_date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.loc[df.groupby(["label", "x_date"])["run_date"].idxmax()].reset_index(drop=True)
    if df.empty:
        st.warning(no_data_msg or "No trend data for the selected configurations.")
        return

    unique_dates = sorted(df["x_date"].dropna().unique())
    tick_labels  = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1,
        cols=len(stats_spec),
        shared_xaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=[lbl for _, lbl in stats_spec],
    )

    marker_alpha = max(0.1, alpha - 0.2)
    for cfg_idx, cfg_label in enumerate(filtered_labels):
        cfg_df = df[df["label"] == cfg_label].sort_values("x_date")
        if cfg_df.empty:
            continue
        n_colors     = len(palette)
        cycle        = cfg_idx // n_colors
        color        = palette[cfg_idx % n_colors]
        line_color   = _to_rgba(color, alpha)
        marker_color = _to_rgba(color, marker_alpha)
        dash         = _DASHES [cycle % len(_DASHES) ] if use_dash   else "solid"
        symbol       = _SYMBOLS[cycle % len(_SYMBOLS)] if use_marker else "circle"
        run_date_str = cfg_df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
        k4h_release  = cfg_df.get("k4h_release", pd.Series(["unknown"] * len(cfg_df))).fillna("unknown")
        custom       = list(zip(run_date_str, k4h_release))

        # Error bars — SEM for each panel
        n_col   = next((c for c in n_col_candidates if c in cfg_df.columns), None)
        has_err = std_col in cfg_df.columns and n_col is not None
        if has_err:
            std  = cfg_df[std_col].to_numpy()
            n    = cfg_df[n_col].to_numpy()
            # n=1  → SEM of mean/median is undefined (need ≥2 events)
            # n≤2  → SEM of std is undefined  (need ≥3 events for unbiased estimate)
            valid_mean   = n > 1
            valid_std    = n > 2
            sem_mean     = np.where(valid_mean, std / np.sqrt(n), np.nan)
            sem_median   = np.where(valid_mean, std * np.sqrt(np.pi / 2) / np.sqrt(n), np.nan)
            sem_std      = np.where(valid_std,  std / np.sqrt(2 * (n - 1)), np.nan)
            sem_by_panel = [sem_median.tolist(), sem_mean.tolist(), sem_std.tolist()]
        else:
            sem_by_panel = [None, None, None]

        for col_idx, (stat_col, stat_label) in enumerate(stats_spec):
            if stat_col not in cfg_df.columns:
                continue
            sem   = sem_by_panel[col_idx] if col_idx < len(sem_by_panel) else None
            err_y = None
            if sem is not None:
                err_y = dict(
                    type="data",
                    array=sem,
                    arrayminus=sem,
                    visible=True,
                    color=_to_rgba(color, 0.3),
                    thickness=1.5,
                    width=4,
                )
            fig.add_trace(
                go.Scatter(
                    x=cfg_df["x_date"],
                    y=cfg_df[stat_col],
                    mode="lines+markers",
                    name=cfg_label,
                    legendgroup=cfg_label,
                    showlegend=(col_idx == 0),
                    line=dict(color=line_color, width=2, dash=dash),
                    marker=dict(size=7, color=marker_color, symbol=symbol,
                                line=dict(color=color, width=1.5)),
                    error_y=err_y,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{cfg_label}</b><br>"
                        "Tag: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                        f"{stat_label}: %{{y:.4g}} {unit}<br>"
                        "CI run: %{customdata[0]}<extra></extra>"
                    ),
                ),
                row=1, col=col_idx + 1,
            )

    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=unique_dates,
        ticktext=tick_labels,
        tickangle=-30,
        title_text="Key4hep Nightly Tag",
    )

    # ── Legend & margins ──────────────────────────────────────────────────────
    _PLOT_H   = 380
    _T_MARGIN = 40
    b_margin, legend_dict = _bottom_legend_params(len(filtered_labels), _PLOT_H)
    fig_height = _PLOT_H + _T_MARGIN + b_margin

    fig.update_layout(
        template=_TEMPLATE,
        height=fig_height,
        margin=dict(l=20, r=20, t=_T_MARGIN, b=b_margin),
        legend=legend_dict,
    )

    st.plotly_chart(fig, width="stretch")
