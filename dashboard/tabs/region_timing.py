from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dd4bench.analysis.plots import plot_region_timing
from dd4bench.analysis.plots._theme import _TEMPLATE
from ui_utils import _DASHES, _PALETTES, _PALETTE_NAMES, _SYMBOLS, _auto_palette_index, _bottom_legend_params, _is_valid_df, _to_rgba

# Fixed colours for source / sink — independent of user palette
_SINK_COLOR   = "#3FA5C8"   # teal-blue  — absorbs secondaries
_SOURCE_COLOR = "#E07A5D"   # warm orange — emits secondaries


# ── Attribution Analysis view ──────────────────────────────────────────────────

def _attribution_explainer() -> None:
    """Two coloured cards explaining source / sink in plain language."""
    col_src, col_snk = st.columns(2)
    with col_src:
        st.markdown(
            f'<div style="background:rgba(224,122,93,0.10);border-left:4px solid {_SOURCE_COLOR};'
            'padding:10px 14px;border-radius:4px">'
            f'<b style="color:{_SOURCE_COLOR}">&#x1F7E0; Source region</b><br>'
            "Particles <em>created</em> here spend most of their simulation tracking time in other regions.<br>"
            '<small style="color:#888">by birth &gt; at location &mdash; bar extends left</small>'
            "</div>",
            unsafe_allow_html=True,
        )
    with col_snk:
        st.markdown(
            f'<div style="background:rgba(63,165,200,0.10);border-left:4px solid {_SINK_COLOR};'
            'padding:10px 14px;border-radius:4px">'
            f'<b style="color:{_SINK_COLOR}">&#x1F535; Sink region</b><br>'
            "Simulation tracking time here is dominated by particles created in other regions.<br>"
            '<small style="color:#888">at location &gt; by birth &mdash; bar extends right</small>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.write("")


def _render_attribution_analysis(region_data: dict, selected_labels: list[str]) -> None:
    """Attribution analysis: scatter (at location vs by birth) + diverging asymmetry bar.

    Key implementation note
    -----------------------
    The zero-line for the diverging bar is drawn with ``fig.add_shape`` rather
    than a ``go.Scatter`` trace.  Adding a numeric-y Scatter to the same
    subplot *before* the categorical-y Bar would lock the y-axis to linear
    mode, silently dropping all bar labels — that was the root cause of the
    previously empty bar panel.
    """
    filtered_labels = [lbl for lbl in selected_labels if lbl in region_data and region_data[lbl]]
    if not filtered_labels:
        st.info("No region timing data available for any of the selected configurations.")
        return

    # ── Controls — no Top N slider; all detectors are shown ───────────────────
    col_cfg, col_pal = st.columns([3, 1])
    with col_cfg:
        config = st.selectbox("Configuration", filtered_labels, key="ss_config")

    # ── Data (computed before palette selectbox so n is known for auto-index) ──
    data  = region_data.get(config, {})
    al_df = data.get("at_location")
    bb_df = data.get("by_birth")
    if al_df is None or bb_df is None:
        with col_pal:
            st.selectbox("Colour palette", options=_PALETTE_NAMES, index=0, key="ss_palette")
        st.info("Both attributions are required for this view.")
        return

    al_df = al_df.drop(index=0, errors="ignore")   # exclude warm-up event 0
    bb_df = bb_df.drop(index=0, errors="ignore")

    al_means = al_df.mean()
    bb_means = bb_df.mean()

    # All detectors with non-trivial signal, ranked by max of both attributions
    all_dets  = sorted(set(al_means.index) | set(bb_means.index))
    union_max = pd.Series({
        d: max(float(al_means.get(d, 0.0)), float(bb_means.get(d, 0.0)))
        for d in all_dets
    })
    det_list = union_max[union_max > 1e-9].sort_values(ascending=False).index.tolist()
    if not det_list:
        with col_pal:
            st.selectbox("Colour palette", options=_PALETTE_NAMES, index=0, key="ss_palette")
        st.info("No detector data to show.")
        return

    n = len(det_list)
    with col_pal:
        palette_name = st.selectbox(
            "Colour palette", options=_PALETTE_NAMES,
            index=_auto_palette_index(n), key="ss_palette",
        )
    palette = _PALETTES[palette_name]

    st.divider()
    _attribution_explainer()

    al_vals = np.array([float(al_means.get(d, 0.0)) for d in det_list])
    bb_vals = np.array([float(bb_means.get(d, 0.0)) for d in det_list])
    delta   = al_vals - bb_vals

    # % asymmetry = (al − bb) / avg(al, bb) × 100
    # Detectors whose combined signal is too small (< 1e-6 s) are masked to NaN
    # to avoid noise-dominated asymmetries exploding the axis scale.
    total     = al_vals + bb_vals
    valid     = total > 1e-6
    pct_asymm = np.full_like(delta, np.nan)
    pct_asymm[valid] = delta[valid] / (total[valid] / 2.0) * 100.0

    # Bar ordering: descending pct → positive (at_location > by_birth, i.e. sink-like)
    # at top.  Convention: positive asymmetry ⇒ at_location dominates ⇒ sink-like.
    # NaN (unmeasurable signal) is treated as zero for ordering so it sinks to the
    # middle; Plotly places the first category at the top for horizontal bars.
    bar_ord   = np.argsort(np.nan_to_num(pct_asymm, nan=0.0))[::-1]
    bar_dets  = [det_list[i]  for i in bar_ord]
    bar_pct   = pct_asymm[bar_ord]
    bar_delta = delta[bar_ord]
    bar_al    = al_vals[bar_ord]
    bar_bb    = bb_vals[bar_ord]
    bar_colors = []
    for p in bar_pct:
        if np.isnan(p):
            bar_colors.append("rgba(160,160,160,0.55)")  # grey — signal too small
        elif p >= 0:
            bar_colors.append(_to_rgba(_SINK_COLOR,   0.82))
        else:
            bar_colors.append(_to_rgba(_SOURCE_COLOR, 0.82))

    pt_colors = [palette[i % len(palette)] for i in range(n)]

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.46, 0.54],
        horizontal_spacing=0.14,
        subplot_titles=[
            "At location vs By birth",
            "Asymmetry  (at location − by birth) / avg  [%]",
        ],
    )

    # ── Left panel: scatter ────────────────────────────────────────────────────
    ax_max = max(float(al_vals.max()), float(bb_vals.max())) * 1.18
    ax_max = max(ax_max, 1e-9)

    # y = x diagonal (no legend entry)
    fig.add_trace(
        go.Scatter(
            x=[0, ax_max], y=[0, ax_max],
            mode="lines",
            line=dict(color="rgba(140,140,140,0.45)", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip", name="",
        ),
        row=1, col=1,
    )

    # One trace per detector → individual named legend entries
    for i, det in enumerate(det_list):
        fig.add_trace(
            go.Scatter(
                x=[bb_vals[i]], y=[al_vals[i]],
                mode="markers",
                name=det,
                legendgroup=det,
                marker=dict(size=12, color=pt_colors[i], line=dict(color="white", width=1.5)),
                showlegend=True,
                customdata=[(det, float(bb_vals[i]), float(al_vals[i]),
                             float(delta[i]), float(pct_asymm[i]))],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "By birth: %{customdata[1]:.4g} s<br>"
                    "At location: %{customdata[2]:.4g} s<br>"
                    "Δ: %{customdata[3]:+.4g} s  (%{customdata[4]:+.1f}%)<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # Zone badges
    fig.add_annotation(
        x=ax_max * 0.07, y=ax_max * 0.91,
        xref="x1", yref="y1",
        text="<b>Sink</b>",
        showarrow=False,
        font=dict(size=10, color=_SINK_COLOR),
        bgcolor="rgba(63,165,200,0.10)",
        bordercolor="rgba(63,165,200,0.35)",
        borderwidth=1, borderpad=5,
    )
    fig.add_annotation(
        x=ax_max * 0.93, y=ax_max * 0.09,
        xref="x1", yref="y1",
        text="<b>Source</b>",
        showarrow=False,
        font=dict(size=10, color=_SOURCE_COLOR),
        bgcolor="rgba(224,122,93,0.10)",
        bordercolor="rgba(224,122,93,0.35)",
        borderwidth=1, borderpad=5,
    )

    fig.update_xaxes(title_text="By birth — mean time per event (s)",    range=[0, ax_max], row=1, col=1)
    fig.update_yaxes(title_text="At location — mean time per event (s)", range=[0, ax_max], row=1, col=1)

    # ── Right panel: diverging bar ─────────────────────────────────────────────
    # Zero line via add_shape — NOT a go.Scatter.
    # A Scatter with numeric y placed before the Bar would force the shared
    # y-axis to linear mode, silently rendering all categorical bar labels as
    # NaN and making the bars invisible.
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=0, y1=1,
        xref="x2", yref="y2 domain",
        line=dict(color="rgba(100,100,100,0.55)", width=1),
    )

    fig.add_trace(
        go.Bar(
            y=bar_dets,
            x=bar_pct.tolist(),
            orientation="h",
            marker_color=bar_colors,
            marker_line_width=0,
            customdata=list(zip(
                bar_dets,
                bar_pct.tolist(),
                bar_delta.tolist(),
                bar_al.tolist(),
                bar_bb.tolist(),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Asymmetry: %{customdata[1]:+.1f}%<br>"
                "Δ = %{customdata[2]:+.4g} s<br>"
                "At location: %{customdata[3]:.4g} s<br>"
                "By birth: %{customdata[4]:.4g} s<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Use nan-safe min/max since bar_pct may contain NaN for tiny-signal detectors.
    x_abs = float(max(abs(np.nanmin(bar_pct)), abs(np.nanmax(bar_pct)))) * 1.20
    x_abs = max(x_abs, 5.0)
    fig.update_xaxes(
        title_text="← source  |  asymmetry (%)  |  sink →",
        range=[-x_abs, x_abs],
        row=1, col=2,
    )

    # ── Legend at bottom ───────────────────────────────────────────────────────
    fig_h    = min(max(420, 70 + n * 35), 1400)  # cap at 1400 px to stay sane in Streamlit
    b_margin, legend_dict = _bottom_legend_params(
        n_items=n, plot_h=fig_h, x_tick_gap=80, entry_width=200, font_size=12
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=fig_h + b_margin,
        margin=dict(l=20, r=20, t=45, b=b_margin),
        legend=legend_dict,
    )

    st.plotly_chart(fig, width="stretch")


# ── Current-run view ───────────────────────────────────────────────────────────

def _render_current_run(region_data: dict, selected_labels: list[str]) -> None:
    """Render the current-run region timing view (existing behaviour)."""
    filtered_labels = [lbl for lbl in selected_labels if lbl in region_data and region_data[lbl]]
    if not filtered_labels:
        st.info("No region timing data available for any of the selected configurations.")
        return

    col_cfg, col_attr = st.columns([2, 2])
    with col_cfg:
        config = st.selectbox("Configuration", filtered_labels, key="region_config")
    with col_attr:
        attribution = st.selectbox(
            "Attribution",
            options=["at_location", "by_birth"],
            format_func=lambda x: "At location" if x == "at_location" else "By birth",
            key="region_attr",
            help=(
                "**At location** — time is charged to the detector region where the "
                "particle *deposited* its energy. Shows which regions are most "
                "expensive to simulate.\n\n"
                "**By birth** — time is charged to the detector region where the "
                "particle was *created*. Shows which regions produce the costliest "
                "secondary particles."
            ),
        )

    col_topn, col_pal = st.columns([2, 2])
    with col_topn:
        top_n = st.slider("Top N detectors", min_value=3, max_value=15, value=8, key="region_topn")
    with col_pal:
        palette_name = st.selectbox(
            "Colour palette",
            options=_PALETTE_NAMES,
            index=_auto_palette_index(top_n),
            key="region_cur_palette",
        )

    fig = plot_region_timing(
        region_data,
        labels=[config],
        show="both",
        attribution=attribution,
        top_n=top_n,
        exclude_events=[0],
        palette=_PALETTES[palette_name],
    )
    st.plotly_chart(fig, width="stretch")


# ── Historical-trends view ─────────────────────────────────────────────────────

def _render_historical(
    trend_region_df: pd.DataFrame,
    selected_labels: list[str],
) -> None:
    """Render the historical region timing trends view."""
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
            help=(
                "**At location** — time is charged to the detector region where the "
                "particle *deposited* its energy. Shows which regions are most "
                "expensive to simulate.\n\n"
                "**By birth** — time is charged to the detector region where the "
                "particle was *created*. Shows which regions produce the costliest "
                "secondary particles."
            ),
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
        with ctrl_l:
            st.selectbox("Colour palette", options=_PALETTE_NAMES, index=0, key="region_hist_palette")
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

    t_margin = 40
    b_margin, legend_dict = _bottom_legend_params(
        len(top_detectors), 380, entry_width=180, font_size=12
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=380 + t_margin + b_margin,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        legend=legend_dict,
    )

    st.plotly_chart(fig, width="stretch")


# ── Step Analysis view ────────────────────────────────────────────────────────

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
        with col_pal:
            st.selectbox("Colour palette", options=_PALETTE_NAMES, index=0, key="sa_palette")
        st.info("No timing data available for this configuration.")
        return
    if steps_df is None:
        with col_pal:
            st.selectbox("Colour palette", options=_PALETTE_NAMES, index=0, key="sa_palette")
        st.info(
            "Step count data (`interval_counts`) is not available in this run's regions JSON. "
            "Regenerate the benchmark output with a newer dd4bench version."
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
    all_det_list = ranked.index.tolist()
    if not all_det_list:
        with col_pal:
            st.selectbox("Colour palette", options=_PALETTE_NAMES, index=0, key="sa_palette")
        st.info("No detector data to show.")
        return

    # ── Secondary controls (need data to set slider max) ──────────────────────
    n_total = len(all_det_list)
    ctrl_topn, ctrl_logy, _ = st.columns([2, 1, 1])
    with ctrl_topn:
        top_n = st.slider(
            "Top N detectors", min_value=3, max_value=n_total,
            value=min(n_total, 15), key="sa_topn",
        )
    with ctrl_logy:
        log_y = st.toggle("Log Y-axis", value=False, key="sa_logy",
                          help="Switch the cost-per-step axis to log scale — "
                               "useful when a few detectors dominate by orders of magnitude.")

    det_list = all_det_list[:top_n]
    n = len(det_list)

    with col_pal:
        palette_name = st.selectbox(
            "Colour palette", options=_PALETTE_NAMES,
            index=_auto_palette_index(n), key="sa_palette",
        )
    palette    = _PALETTES[palette_name]
    total_time = np.array([float(al_means[d])    for d in det_list])   # s
    step_cnt   = np.array([max(float(steps_means.get(d, 0.0)), 0.1) for d in det_list])  # steps (floor for log)
    tps_us     = total_time / step_cnt * 1e6   # µs per step

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
                x=[step_cnt[i]], y=[tps_us[i]],
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
                customdata=[(det, float(step_cnt[i]), float(tps_us[i]),
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
                x=[float(step_cnt[i])],
                orientation="h",
                name=det,
                legendgroup=det,
                showlegend=False,
                marker_color=_to_rgba(pt_colors[i], 0.80),
                marker_line_color=pt_colors[i] if cycle > 0 else "rgba(0,0,0,0)",
                marker_line_width=2.0 if cycle > 0 else 0,
                customdata=[(det, float(step_cnt[i]))],
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
    fig_h    = max(420, 70 + n * 35)
    b_margin, legend_dict = _bottom_legend_params(
        n_items=n, plot_h=fig_h, x_tick_gap=80, entry_width=200, font_size=12
    )
    # Bubble-size legend note
    st.caption(
        "Bubble area is proportional to total mean simulation time per event. "
        "Log x-axis — step counts span several orders of magnitude."
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=fig_h + b_margin,
        margin=dict(l=20, r=20, t=45, b=b_margin),
        legend=legend_dict,
    )

    st.plotly_chart(fig, width="stretch")


# ── Tab entry point ────────────────────────────────────────────────────────────

def render(
    region_data: dict | None,
    trend_region_df: pd.DataFrame | None,
    selected_labels: list[str],
) -> None:
    if region_data is None and trend_region_df is None:
        st.info("No region timing data available in the selected directory.")
        return
    if not selected_labels:
        st.info("Select at least one run in the sidebar.")
        return

    # Build view options dynamically based on available data
    # Order: current-run analyses first, then historical trends
    view_options: list[str] = ["Current Run"]
    if region_data is not None:
        view_options.append("Attribution Analysis")
        # Step Analysis requires interval_counts — check any label has it
        has_steps = any(
            region_data[lbl].get("steps") is not None
            for lbl in selected_labels
            if lbl in region_data
        )
        if has_steps:
            view_options.append("Step Analysis")
    if _is_valid_df(trend_region_df):
        view_options.append("Historical Trends")

    view = (
        st.radio("View", options=view_options, horizontal=True, key="region_view_mode")
        if len(view_options) > 1
        else view_options[0]
    )

    if view == "Current Run":
        if region_data is None:
            st.info("No region timing data available in the selected directory.")
        else:
            _render_current_run(region_data, selected_labels)
    elif view == "Historical Trends":
        _render_historical(trend_region_df, selected_labels)
    elif view == "Attribution Analysis":
        _render_attribution_analysis(region_data, selected_labels)
    elif view == "Step Analysis":
        _render_step_analysis(region_data, selected_labels)
