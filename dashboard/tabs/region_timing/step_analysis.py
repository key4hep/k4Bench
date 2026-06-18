"""Step Analysis view — step-count decomposition: scatter + ranked bar panels."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import _TEMPLATE
from ui_utils import _PALETTES, _PALETTE_NAMES, _SYMBOLS, _auto_palette_index, _legend_below, _to_rgba

from ._common import _palette_placeholder


def _render_step_analysis(region_data: dict, selected_labels: list[str]) -> None:
    """Step count decomposition: scatter (steps vs µs/step) + ranked bar panels.

    Answers: *why* is a region expensive?
      - Many cheap steps  → geometry-dominated  (geometry simplification / step limits)
      - Few expensive steps → physics-dominated (physics list / production cuts)

    Uses ``interval_counts`` from the regions JSON, which the loader exposes
    as ``region_data[label]["steps"]`` (a DataFrame indexed by event_number).
    """
    filtered_labels = [lbl for lbl in selected_labels if lbl in region_data and region_data[lbl]]
    if not filtered_labels:
        st.info("No region timing data available for any of the selected configurations.")
        return

    # ── Controls — config first, then data, then palette (needs n) ───────────
    col_cfg, col_pal = st.columns([3, 1])
    with col_cfg:
        config = st.selectbox("Configuration", filtered_labels, key="sa_config")

    data     = region_data.get(config, {})
    al_df    = data.get("at_location")
    steps_df = data.get("steps")

    if al_df is None:
        _palette_placeholder(col_pal, "sa_palette")
        st.info("No timing data available for this configuration.")
        return
    if steps_df is None:
        _palette_placeholder(col_pal, "sa_palette")
        st.info(
            "Step count data (`interval_counts`) is not available in this run's regions JSON. "
            "Regenerate the benchmark output with a newer k4bench version."
        )
        return

    # Exclude warm-up event 0
    al_df    = al_df.drop(index=0, errors="ignore")
    steps_df = steps_df.drop(index=0, errors="ignore")

    # Detectors present in both timing and step data (exclude "unattributed" steps)
    dets = [d for d in al_df.columns if d in steps_df.columns and d != "unattributed"]

    al_means    = al_df[dets].mean()
    # Explicit float cast before mean/clip to avoid nullable-integer edge cases.
    steps_means = steps_df[dets].fillna(0).astype(float).mean().clip(lower=0)

    # Sort by total mean time (most expensive first)
    ranked = al_means[al_means > 1e-9].sort_values(ascending=False)
    det_list = ranked.index.tolist()
    if not det_list:
        _palette_placeholder(col_pal, "sa_palette")
        st.info("No detector data to show.")
        return

    n = len(det_list)

    # ── Log Y-axis toggle ─────────────────────────────────────────────────────
    _, ctrl_logy = st.columns([3, 1])
    with ctrl_logy:
        log_y = st.toggle("Log Y-axis", value=False, key="sa_logy",
                          help="Switch the cost-per-step axis to log scale — "
                               "useful when a few detectors dominate by orders of magnitude.")

    with col_pal:
        palette_name = st.selectbox(
            "Colour palette", options=_PALETTE_NAMES,
            index=_auto_palette_index(n), key="sa_palette",
        )
    palette    = _PALETTES[palette_name]
    total_time    = np.array([float(al_means[d])             for d in det_list])  # s
    raw_step_cnt  = np.array([float(steps_means.get(d, 0.0)) for d in det_list])  # steps
    plot_step_cnt = np.maximum(raw_step_cnt, 0.1)            # floored for the log x-axis only
    with np.errstate(divide="ignore", invalid="ignore"):     # 0-step detectors → NaN tps
        tps_us = np.where(raw_step_cnt > 0, total_time / raw_step_cnt * 1e6, np.nan)  # µs per step

    n_colors   = len(palette)
    pt_colors  = [palette[i % n_colors] for i in range(n)]
    # Which palette cycle each detector falls into (0 = first 10, 1 = next 10, …)
    pt_cycles  = [i // n_colors for i in range(n)]

    # ── Figure: scatter (left) + two bar panels (right) ───────────────────────
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.44, 0.28, 0.28],
        horizontal_spacing=0.06,
        subplot_titles=[
            "Steps vs cost per step",
            "Mean steps / event",
            "Time per step (µs)",
        ],
    )

    # ── Scatter: one trace per detector for individual legend entries ──────────
    # Bubble size ∝ √(total_time), scaled to [10, 38] px.
    # When all values are identical (single detector or uniform cost), use the
    # midpoint size so bubbles don't all collapse to the minimum.
    sqrt_t     = np.sqrt(total_time)
    size_range = (10.0, 38.0)
    t_range    = float(sqrt_t.max() - sqrt_t.min())
    if np.isclose(t_range, 0.0):
        msize = np.full_like(sqrt_t, float(np.mean(size_range)))
    else:
        t_norm = (sqrt_t - sqrt_t.min()) / t_range
        msize  = size_range[0] + t_norm * (size_range[1] - size_range[0])

    for i, det in enumerate(det_list):
        cycle  = pt_cycles[i]
        # Change marker symbol and border for each palette cycle so repeated
        # colours are still visually distinguishable in the legend and scatter.
        symbol       = _SYMBOLS[cycle % len(_SYMBOLS)]
        border_color = pt_colors[i] if cycle > 0 else "white"
        fig.add_trace(
            go.Scatter(
                x=[plot_step_cnt[i]], y=[tps_us[i]],
                mode="markers",
                name=det,
                legendgroup=det,
                marker=dict(
                    size=float(msize[i]),
                    color=pt_colors[i],
                    symbol=symbol,
                    line=dict(color=border_color, width=1.5),
                    opacity=0.88,
                ),
                showlegend=True,
                customdata=[(det, float(raw_step_cnt[i]), float(tps_us[i]),
                             float(total_time[i]),
                             float(total_time[i] / total_time.sum() * 100))],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Mean steps / event: %{customdata[1]:,.0f}<br>"
                    "Time per step: %{customdata[2]:.2f} µs<br>"
                    "Total mean time: %{customdata[3]:.4g} s<br>"
                    "Share of event: %{customdata[4]:.1f}%<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # Quadrant zone labels (corner annotations in normalised subplot coords)
    for ann in [
        dict(x=0.02, y=0.97, text="<b>Physics-dominated</b><br><i>few, costly steps</i>",
             color="#C05820", xanchor="left",  yanchor="top"),
        dict(x=0.98, y=0.97, text="<b>Both</b><br><i>many costly steps</i>",
             color="#880000", xanchor="right", yanchor="top"),
        dict(x=0.98, y=0.03, text="<b>Geometry-dominated</b><br><i>many cheap steps</i>",
             color="#1A7A1A", xanchor="right", yanchor="bottom"),
        dict(x=0.02, y=0.03, text="<b>Negligible</b>",
             color="#888888", xanchor="left",  yanchor="bottom"),
    ]:
        fig.add_annotation(
            x=ann["x"], y=ann["y"],
            xref="x domain", yref="y domain",
            text=ann["text"],
            showarrow=False,
            font=dict(size=9, color=ann["color"]),
            align="left" if ann["xanchor"] == "left" else "right",
            xanchor=ann["xanchor"], yanchor=ann["yanchor"],
            bgcolor="rgba(255,255,255,0.55)",
            borderpad=3,
        )

    fig.update_xaxes(
        title_text="Mean steps per event",
        type="log",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Time per step (µs)",
        type="log" if log_y else "linear",
        row=1, col=1,
    )

    # ── Bar panels: one trace per detector so legend clicks hide/show bars too ──
    # Each bar shares legendgroup with its scatter point → clicking a legend
    # entry toggles visibility across all three panels simultaneously.
    #
    # Col 2 (mean steps/event): ordered by total mean time desc (matches scatter)
    # Col 3 (time per step):    ordered independently by tps desc (highest at top)
    # Y-tick labels are suppressed on both bars; the legend carries the names.

    # Col 2 — add detectors in ascending total-time order so the most expensive
    # ends up at the top (Plotly places the last-added category at the top).
    # Second-palette-cycle detectors get a coloured border to stay distinguishable.
    for i in reversed(range(n)):                    # cheapest first → most expensive last → top
        det   = det_list[i]
        cycle = pt_cycles[i]
        fig.add_trace(
            go.Bar(
                y=[det],
                x=[float(raw_step_cnt[i])],
                orientation="h",
                name=det,
                legendgroup=det,
                showlegend=False,
                marker_color=_to_rgba(pt_colors[i], 0.80),
                marker_line_color=pt_colors[i] if cycle > 0 else "rgba(0,0,0,0)",
                marker_line_width=2.0 if cycle > 0 else 0,
                customdata=[(det, float(raw_step_cnt[i]))],
                hovertemplate="<b>%{customdata[0]}</b><br>Steps: %{customdata[1]:,.0f}<extra></extra>",
            ),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="Mean steps / event", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # Col 3 — sorted independently by tps ascending so the highest tps lands at top.
    tps_order = sorted(range(n), key=lambda i: tps_us[i])   # lowest tps first → highest last → top
    for i in tps_order:
        det   = det_list[i]
        cycle = pt_cycles[i]
        fig.add_trace(
            go.Bar(
                y=[det],
                x=[float(tps_us[i])],
                orientation="h",
                name=det,
                legendgroup=det,
                showlegend=False,
                marker_color=_to_rgba(pt_colors[i], 0.80),
                marker_line_color=pt_colors[i] if cycle > 0 else "rgba(0,0,0,0)",
                marker_line_width=2.0 if cycle > 0 else 0,
                customdata=[(det, float(tps_us[i]))],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.2f} µs / step<extra></extra>",
            ),
            row=1, col=3,
        )
    fig.update_xaxes(title_text="Time per step (µs)", row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=3)

    # ── Legend at bottom ───────────────────────────────────────────────────────
    fig_h = max(420, 70 + n * 35)
    # Bubble-size legend note
    st.caption(
        "Bubble area is proportional to total mean simulation time per event. "
        "Log x-axis — step counts span several orders of magnitude."
    )
    legend, b_margin = _legend_below(
        fig_h, n, t_margin=45, tick_clearance=50, entry_width=200, font_size=12,
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=fig_h + 45 + b_margin,
        margin=dict(l=20, r=20, t=45, b=b_margin),
        legend=legend,
    )

    st.plotly_chart(fig, width="stretch", key="region_step_chart")
