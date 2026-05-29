"""Historical-trends view — per-detector region timing over CI runs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dd4bench.analysis.plots._theme import _TEMPLATE
from ui_utils import _DASHES, _LEGEND_B_MARGIN, _PALETTES, _PALETTE_NAMES, _SYMBOLS, _auto_palette_index, _is_valid_df, _legend_below, _to_rgba

from ._common import _ATTRIBUTION_HELP, _palette_placeholder


def _render_historical(
    trend_region_df: pd.DataFrame,
    selected_labels: list[str],
) -> None:
    """Render the historical region timing trends view."""
    if not _is_valid_df(trend_region_df):
        st.info(
            "No region timing trend data in the selected window. "
            "Widen the trend window in the sidebar."
        )
        return
    avail_labels   = sorted(trend_region_df["label"].unique())
    filtered_labels = [lbl for lbl in selected_labels if lbl in avail_labels]
    if not filtered_labels:
        st.info("No historical region timing data available for the selected configurations.")
        return

    col_cfg, col_attr = st.columns([2, 2])
    with col_cfg:
        config = st.selectbox("Configuration", filtered_labels, key="region_hist_config")
    with col_attr:
        attribution = st.radio(
            "Attribution",
            options=["at_location", "by_birth"],
            format_func=lambda x: "At location" if x == "at_location" else "By birth",
            horizontal=True,
            key="region_hist_attr",
            help=_ATTRIBUTION_HELP,
        )

    # Style controls — palette selectbox is rendered last (after top_detectors is
    # known) so its default index can auto-select the right Matplotlib tab-N.
    ctrl_l, ctrl_m, ctrl_r = st.columns(3, vertical_alignment="bottom")
    with ctrl_m:
        style_cycling = st.selectbox(
            "Style cycling",
            options=["Colour only", "Colour + Dash", "Colour + Marker", "Colour + Dash + Marker"],
            index=0,
            key="region_hist_style",
        )
    with ctrl_r:
        alpha = st.slider(
            "Opacity", min_value=0.1, max_value=1.0, value=0.85, step=0.05,
            key="region_hist_alpha",
        )

    use_dash   = style_cycling in ("Colour + Dash",   "Colour + Dash + Marker")
    use_marker = style_cycling in ("Colour + Marker", "Colour + Dash + Marker")

    sub = trend_region_df[
        (trend_region_df["label"] == config)
        & (trend_region_df["attribution"] == attribution)
    ].copy()

    if sub.empty:
        _palette_placeholder(ctrl_l, "region_hist_palette")
        st.info(
            f"No historical region timing data for **{config}** "
            f"({attribution.replace('_', ' ')})."
        )
        return

    sub["x_date"]   = pd.to_datetime(sub["x_date"])
    sub["run_date"] = pd.to_datetime(sub["run_date"])

    # Deduplicate: keep the latest CI run per (detector, nightly tag).
    # Drop rows where run_date is NaT first — idxmax() raises on all-NaT groups.
    sub = sub.dropna(subset=["run_date"])
    sub = sub.loc[
        sub.groupby(["detector", "x_date"])["run_date"].idxmax()
    ].reset_index(drop=True)

    detector_rank = (
        sub.groupby("detector")["median_time_s"].median().sort_values(ascending=False)
    )
    top_detectors = detector_rank.index.tolist()

    with ctrl_l:
        palette_name = st.selectbox(
            "Colour palette", options=_PALETTE_NAMES,
            index=_auto_palette_index(len(top_detectors)), key="region_hist_palette",
        )
    palette = _PALETTES[palette_name]

    unique_dates = sorted(sub["x_date"].dropna().unique())
    tick_labels  = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]

    _STATS = [
        ("median_time_s", "Median time (s)"),
        ("mean_time_s",   "Mean time (s)"),
        ("std_time_s",    "Std dev (s)"),
    ]
    present_stats = [(col, lbl) for col, lbl in _STATS if col in sub.columns]

    fig = make_subplots(
        rows=1,
        cols=len(present_stats),
        shared_xaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=[lbl for _, lbl in present_stats],
    )

    marker_alpha = max(0.1, alpha - 0.2)
    for det_idx, detector in enumerate(top_detectors):
        det_df = sub[sub["detector"] == detector].sort_values("x_date")
        if det_df.empty:
            continue
        n_colors     = len(palette)
        cycle        = det_idx // n_colors
        color        = palette[det_idx % n_colors]
        line_color   = _to_rgba(color, alpha)
        marker_color = _to_rgba(color, marker_alpha)
        dash         = _DASHES [cycle % len(_DASHES) ] if use_dash   else "solid"
        symbol       = _SYMBOLS[cycle % len(_SYMBOLS)] if use_marker else "circle"
        run_date_str = det_df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
        k4h_release  = det_df["k4h_release"].fillna("unknown")
        custom       = list(zip(run_date_str, k4h_release))

        has_err = "std_time_s" in det_df.columns and "n_events" in det_df.columns
        if has_err:
            std          = det_df["std_time_s"].to_numpy()
            n            = det_df["n_events"].to_numpy()
            valid_mean   = n > 1
            valid_std    = n > 2
            sem_mean     = np.where(valid_mean, std / np.sqrt(n), np.nan).tolist()
            sem_median   = np.where(valid_mean, std * np.sqrt(np.pi / 2) / np.sqrt(n), np.nan).tolist()
            sem_std      = np.where(valid_std,  std / np.sqrt(2 * (n - 1)), np.nan).tolist()
            sem_by_panel = [sem_median, sem_mean, sem_std]
        else:
            sem_by_panel = [None, None, None]

        for col_idx, (stat_col, stat_label) in enumerate(present_stats):
            sem   = sem_by_panel[col_idx] if col_idx < len(sem_by_panel) else None
            err_y = None
            if sem is not None:
                err_y = dict(
                    type="data", array=sem, arrayminus=sem,
                    visible=True, color=_to_rgba(color, 0.3),
                    thickness=1.5, width=4,
                )
            fig.add_trace(
                go.Scatter(
                    x=det_df["x_date"],
                    y=det_df[stat_col],
                    mode="lines+markers",
                    name=detector,
                    legendgroup=detector,
                    showlegend=(col_idx == 0),
                    line=dict(color=line_color, width=2, dash=dash),
                    marker=dict(size=7, color=marker_color, symbol=symbol,
                                line=dict(color=color, width=1.5)),
                    error_y=err_y,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{detector}</b><br>"
                        "Tag: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                        f"{stat_label}: %{{y:.4g}} s<br>"
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

    # Extra 40 px so the "Key4hep Nightly Tag" x-axis title has breathing room
    # before the horizontal legend — same treatment as trends.py.
    _b_margin = _LEGEND_B_MARGIN + 40
    fig.update_layout(
        template=_TEMPLATE,
        height=380 + 40 + _b_margin,
        margin=dict(l=20, r=20, t=40, b=_b_margin),
        legend=_legend_below(380, entry_width=180, font_size=12, y_offset=110),
    )

    st.plotly_chart(fig, width="stretch", key="region_historical_chart")
