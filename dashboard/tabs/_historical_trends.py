"""The historical-trends section shared by the per-event tabs.

Event Timing and Event Memory draw their history of per-run statistics with
:func:`render_historical_trends`; they and Run Trends add the collapsed
run-to-run variability table with :func:`render_stability_expander`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import _TEMPLATE
from k4bench.metrics import metric_unit
from k4bench.regression.engine import BASELINE_WINDOW_RUNS, NOISE_WINDOW_RUNS
from stats import build_stability_table
from ui_utils import (
    DASHES,
    PALETTES,
    SYMBOLS,
    display_options,
    is_valid_df,
    legend_below,
    opacity_control,
    palette_control,
    style_cycling_control,
    style_cycling_flags,
    to_rgba,
)


def render_stability_expander(
    df: pd.DataFrame | None,
    metrics: list[str],
    reliability: dict[str, bool | None] | None,
    *,
    key: str,
) -> None:
    """Collapsed per-config table of how much *metrics* move between runs.

    *df* must still hold every run (before any one-point-per-tag collapse):
    the same-release repeats the headline column is built from are exactly the
    runs such a collapse drops. Nothing renders when none of *metrics* is in
    *df*, so old data without them shows no empty box. Display-only; nothing
    here is written back to results or reports.
    """
    if not is_valid_df(df):
        return
    table = build_stability_table(
        df, {m: metric_unit(m) for m in metrics}, reliability,
    )
    if table.empty:
        return
    with st.expander("Run-to-run variability", expanded=False, key=key):
        st.caption(
            "**Same-release spread**: typical change between consecutive reliable "
            "runs of the *same* Key4hep release, over the last "
            f"{BASELINE_WINDOW_RUNS} such pairs in the trend window. Stack changes "
            "are excluded, but not k4Bench-side changes: runs of one release can "
            "come from different k4Bench commits, so it can contain harness, "
            "plugin or configuration changes as well as measurement noise. "
            "**Recent movement**: the "
            f"same statistic over the last {NOISE_WINDOW_RUNS} reliable runs "
            "regardless of release, so it also contains stack changes. "
            "Runs are ordered by release date, then run, as the regression engine "
            "orders them: a later rerun of an old release counts with that release. "
            "Neither is an uncertainty or a confidence interval."
        )
        st.dataframe(table, width="stretch")


def render_historical_trends(
    trend_df: pd.DataFrame,
    filtered_labels: list[str],
    stats_spec: list[tuple[str, str]],
    *,
    std_col: str,
    n_col_candidates: list[str],
    unit: str,
    key_prefix: str,
    no_data_msg: str = "",
    display_options_slot=None,
    error_sources: dict[str, tuple[str, str]] | None = None,
    units: dict[str, str] | None = None,
) -> None:
    """Render historical statistics in a grid of up to three columns.

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
    display_options_slot :
        Placeholder for the Display options trigger, reserved by the tab on the
        row that carries its View selector, so the popover sits in the same
        place in this view as in the tab's current-run view.
    error_sources :
        Optional mapping from statistic column to (standard deviation, event
        count) columns. When supplied, unlisted statistics have no error bars.
        Otherwise all statistics use ``std_col`` and ``n_col_candidates``.
    units :
        Optional per-statistic hover unit, for a panel whose unit is not
        ``unit`` (e.g. a slope in MB/event on a memory tab).
    """
    # ── Display options ───────────────────────────────────────────────────────
    # The opacity slider is keyed ``…_line_alpha`` rather than ``…_alpha``: the
    # tab passes *key_prefix* ``evt_timing_hist``, and the current-run view's
    # "Bar opacity" slider is ``evt_timing`` + ``_hist_alpha`` — the same string.
    # Sharing one slot would carry each view's value into the other on a view
    # switch, where the two sliders mean different things and start at different
    # defaults.
    values = display_options(
        palette_control(f"{key_prefix}_palette", len(filtered_labels)),
        style_cycling_control(f"{key_prefix}_style"),
        opacity_control(f"{key_prefix}_line_alpha"),
        key_prefix=f"{key_prefix}_display",
        slot=display_options_slot,
    )
    palette = PALETTES[values["palette"]]
    alpha = values["alpha"]
    use_dash, use_marker = style_cycling_flags(values["style"])

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
    n_cols = min(3, len(stats_spec))
    n_rows = math.ceil(len(stats_spec) / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes="all",
        horizontal_spacing=0.06,
        vertical_spacing=0.12 if n_rows > 1 else 0.0,
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
        line_color   = to_rgba(color, alpha)
        marker_color = to_rgba(color, marker_alpha)
        dash         = DASHES [cycle % len(DASHES) ] if use_dash   else "solid"
        symbol       = SYMBOLS[cycle % len(SYMBOLS)] if use_marker else "circle"
        run_date_str = cfg_df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
        k4h_release  = cfg_df.get("k4h_release", pd.Series(["unknown"] * len(cfg_df))).fillna("unknown")
        custom       = list(zip(run_date_str, k4h_release))

        for col_idx, (stat_col, stat_label) in enumerate(stats_spec):
            if stat_col not in cfg_df.columns:
                continue
            n_col = next((c for c in n_col_candidates if c in cfg_df.columns), None)
            source = error_sources.get(stat_col) if error_sources is not None else (std_col, n_col)
            sem = None
            if source is not None and all(c in cfg_df.columns for c in source):
                std = cfg_df[source[0]].to_numpy(dtype=float)
                n = cfg_df[source[1]].to_numpy(dtype=float)
                statistic = stat_col.partition("_")[0]
                valid = n > (2 if statistic == "std" else 1)
                sem = np.full(len(n), np.nan)
                if statistic == "std":
                    sem[valid] = std[valid] / np.sqrt(2 * (n[valid] - 1))
                elif statistic in ("mean", "median"):
                    sem[valid] = std[valid] / np.sqrt(n[valid])
                    if statistic == "median":
                        sem *= np.sqrt(np.pi / 2)
                else:
                    sem = None
            err_y = None
            if sem is not None:
                err_y = dict(
                    type="data",
                    array=sem,
                    arrayminus=sem,
                    visible=True,
                    color=to_rgba(color, 0.3),
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
                    marker=dict(
                        size=7, color=marker_color, symbol=symbol, line=dict(color=color, width=1.5)
                    ),
                    error_y=err_y,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{cfg_label}</b><br>"
                        "Tag: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                        f"{stat_label}: %{{y:.4g}} {(units or {}).get(stat_col, unit)}<br>"
                        "CI run: %{customdata[0]}<extra></extra>"
                    ),
                ),
                row=col_idx // n_cols + 1,
                col=col_idx % n_cols + 1,
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
    # tick_clearance=75: rotated (-30°) date ticks + "Key4hep Nightly Tag" title.
    _PLOT_H = 380 * n_rows
    _T_MARGIN = 40
    _legend, _B_MARGIN = legend_below(
        _PLOT_H, len(filtered_labels), t_margin=_T_MARGIN, tick_clearance=75,
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=_PLOT_H + _T_MARGIN + _B_MARGIN,
        margin=dict(l=20, r=20, t=_T_MARGIN, b=_B_MARGIN),
        legend=_legend,
    )

    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_chart")
