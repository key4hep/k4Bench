"""plot_region_timing: per-detector breakdown and per-event stacked-area sequence."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._theme import _OTHER_COLOR, _PALETTE, _TEMPLATE, _UNACCOUNTED_COLOR, _hex_to_rgba
from ._utils import (
    _DEFAULT_EXCLUDE_EVENTS,
    _build_stacked_arrays,
    _ensure_region_data,
    _region_top_n,
)


def plot_region_timing(
    source: dict[str, dict] | str | Path | list[str | Path],
    *,
    labels: list[str] | None = None,
    show: str = "both",
    attribution: str = "at_location",
    top_n: int = 8,
    figsize: tuple[float, float] | None = None,
    exclude_events: list[int] | None = None,
    palette: list[str] | None = None,
    alpha: float = 0.85,
) -> go.Figure:
    """Plot per-detector timing breakdown and/or per-event sequence for one or more runs.

    Single run: a donut chart + sorted horizontal bar chart (breakdown panel) and/or
    a stacked-area sequence chart.  Multiple runs: a grouped horizontal bar chart
    (breakdown) and/or a vertical stack of per-run stacked-area sequence charts.

    Parameters
    ----------
    source : dict[str, dict], str/Path, or list of str/Path
        Pre-loaded dict from :func:`~k4bench.analysis.loader.load_region_timing`,
        a single log-dir path, or a list of log-dir paths.
    labels : list[str] or None
        Restrict to these run labels.
    show : {"both", "breakdown", "sequence"}
        Which panels to display.
    attribution : {"at_location", "by_birth"}
        Which attribution to use.
    top_n : int
        Show the top *n* detectors by mean time; remaining are grouped into ``"Other"``.
    figsize : (width, height) or None
        Figure size in inches (converted to pixels at 96 dpi).
    exclude_events : list[int] or None
        Event numbers to exclude.  Defaults to ``[0]``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if show not in ("both", "breakdown", "sequence"):
        raise ValueError(f"show must be 'both', 'breakdown', or 'sequence', got {show!r}")
    if attribution not in ("at_location", "by_birth"):
        raise ValueError(f"attribution must be 'at_location' or 'by_birth', got {attribution!r}")

    if exclude_events is None:
        exclude_events = list(_DEFAULT_EXCLUDE_EVENTS)

    region_data = _ensure_region_data(source, labels=labels)
    if not region_data:
        raise ValueError(f"No region data found for labels={labels}.")

    label_list = list(region_data.keys())
    n = len(label_list)
    _pal = palette if palette is not None else _PALETTE

    # ------------------------------------------------------------------
    # Filter events and align DataFrames
    # ------------------------------------------------------------------
    filtered: dict[str, dict] = {}
    for lbl, data in region_data.items():
        ev_df = data["events"]
        mask  = ~ev_df["event_number"].isin(exclude_events)
        ev_filt = ev_df[mask].copy().reset_index(drop=True)
        if ev_filt.empty:
            raise ValueError(f"No events left after applying exclude_events for '{lbl}'.")
        time_df   = data[attribution]
        time_filt = (
            time_df.loc[time_df.index.isin(ev_filt["event_number"])]
            .reindex(ev_filt["event_number"].values)
        )
        missing_ev = time_filt.index[time_filt.isnull().any(axis=1)].tolist()
        if missing_ev:
            warnings.warn(
                f"'{lbl}': {len(missing_ev)} event(s) missing from time_df "
                f"(e.g. {missing_ev[:3]}); filling with 0.0 — means will be biased downward.",
                stacklevel=2,
            )
        time_filt = time_filt.fillna(0.0)
        filtered[lbl] = {"events": ev_filt, "time": time_filt}

    # ------------------------------------------------------------------
    # Top-N detectors — ranked by max-of-means across all runs so that
    # detectors dominant in any run are not folded into "Other".
    # ------------------------------------------------------------------
    means_per_run = [filtered[lbl]["time"].mean() for lbl in label_list]
    all_cols = sorted({col for s in means_per_run for col in s.index})
    union_max = pd.Series(
        {col: max(float(s.get(col, 0.0)) for s in means_per_run) for col in all_cols}
    )
    top_dets, all_dets_sorted = _region_top_n(union_max.to_frame().T, top_n)

    needs_other = len(all_dets_sorted) > top_n
    det_display = top_dets + (["Other"] if needs_other else [])
    det_colors: dict[str, str] = {
        det: _pal[i % len(_pal)] for i, det in enumerate(top_dets)
    }
    if "Other" in det_display:
        det_colors["Other"] = _OTHER_COLOR
    det_colors["Unaccounted"] = _UNACCOUNTED_COLOR

    show_breakdown = show in ("both", "breakdown")
    show_seq       = show in ("both", "sequence")

    # ------------------------------------------------------------------
    # Build figure layout
    # ------------------------------------------------------------------
    px_w = int(figsize[0] * 96) if figsize else 1200

    if n == 1:
        if show == "both":
            # Named constants so annotation positions can be derived analytically.
            _v_sp, _h_sp = 0.16, 0.18
            _r1_h, _r2_h = 0.40, 0.60
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "domain"}, {"type": "xy"}],
                    [{"type": "xy", "colspan": 2}, None],
                ],
                row_heights=[_r1_h, _r2_h],
                vertical_spacing=_v_sp,
                horizontal_spacing=_h_sp,
            )
            px_h = figsize[1] * 96 if figsize else 1000
        elif show == "breakdown":
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "domain"}, {"type": "xy"}]],
                horizontal_spacing=0.08,
            )
            px_h = figsize[1] * 96 if figsize else 450
        else:
            fig = make_subplots(rows=1, cols=1)
            px_h = figsize[1] * 96 if figsize else 450
    else:
        if show == "both":
            seq_h_frac = 0.25
            bar_h_frac = max(0.15, 1.0 - seq_h_frac * n)
            row_heights = [bar_h_frac] + [seq_h_frac] * n
            n_rows_both = 1 + n
            v_spacing_both = min(0.06, 1.0 / (n_rows_both - 1)) if n_rows_both > 1 else 0.06
            fig = make_subplots(
                rows=n_rows_both, cols=1,
                row_heights=row_heights,
                vertical_spacing=v_spacing_both,
            )
            px_h = figsize[1] * 96 if figsize else (500 + 350 * n)
        elif show == "breakdown":
            fig = make_subplots(rows=1, cols=1)
            px_h = figsize[1] * 96 if figsize else 500
        else:
            v_spacing_seq = min(0.06, 1.0 / (n - 1)) if n > 1 else 0.06
            fig = make_subplots(
                rows=n, cols=1,
                shared_xaxes=True,
                vertical_spacing=v_spacing_seq,
            )
            px_h = figsize[1] * 96 if figsize else 350 * n

    # ------------------------------------------------------------------
    # Breakdown panel
    # ------------------------------------------------------------------
    if show_breakdown:
        lbl0     = label_list[0]
        time_df0 = filtered[lbl0]["time"]
        ev_df0   = filtered[lbl0]["events"]
        stacked0 = _build_stacked_arrays(time_df0, top_dets, all_dets_sorted)
        means0   = {det: float(arr.mean()) for det, arr in stacked0.items()}
        sems0    = {
            det: (float(arr.std(ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)
            for det, arr in stacked0.items()
        }
        mean_unacc = float(ev_df0["event_unaccounted_s"].mean())
        unacc_arr  = ev_df0["event_unaccounted_s"].to_numpy()
        sem_unacc  = float(unacc_arr.std(ddof=1)) / np.sqrt(len(unacc_arr)) if len(unacc_arr) > 1 else 0.0
        total_wall0 = float(ev_df0["event_wall_s"].mean())

        if n == 1 and show != "sequence":
            # Donut chart
            donut_cats = det_display + ["Unaccounted"]
            donut_vals = [means0.get(d, 0.0) for d in det_display] + [max(0.0, mean_unacc)]
            donut_clrs = [det_colors[c] for c in donut_cats]

            valid_idx = [i for i, v in enumerate(donut_vals) if v > 0]
            vv = [donut_vals[i] for i in valid_idx]
            vc = [donut_clrs[i] for i in valid_idx]
            vl = [donut_cats[i] for i in valid_idx]

            if vv:
                fig.add_trace(
                    go.Pie(
                        labels=vl, values=vv, hole=0.55,
                        marker=dict(colors=vc, line=dict(color="white", width=1.5)),
                        textinfo="percent",
                        textfont=dict(size=10),
                        hovertemplate="<b>%{label}</b><br>%{value:.3g} s<br>%{percent}<extra></extra>",
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
                if show == "both":
                    # Centre of col-1 / row-1 in paper coordinates, derived from
                    # the make_subplots constants defined above.
                    # Col-1 occupies x ∈ [0, (1-_h_sp)/2]; centre = (1-_h_sp)/4
                    # Row-1 occupies y ∈ [_r2_h*(1-_v_sp)+_v_sp, 1.0]; centre = mid-point
                    _ann_x = (1 - _h_sp) / 4
                    _r1_bottom = _r2_h * (1 - _v_sp) + _v_sp
                    _ann_y = (_r1_bottom + 1.0) / 2
                else:
                    _ann_x, _ann_y = 0.23, 0.5
                fig.add_annotation(
                    text=f"μ = {total_wall0:.3g} s<br>per event",
                    x=_ann_x, y=_ann_y,
                    xref="paper", yref="paper",
                    xanchor="center", yanchor="middle",
                    showarrow=False,
                    font=dict(size=14, color="#333333"),
                    align="center",
                )

            # Horizontal bar (single run)
            bar_cats  = det_display + ["Unaccounted"]
            bar_means = [means0.get(d, 0.0) for d in det_display] + [max(0.0, mean_unacc)]
            bar_sems  = [sems0.get(d, 0.0)  for d in det_display] + [sem_unacc]
            bar_clrs  = [det_colors.get(d, _UNACCOUNTED_COLOR) for d in bar_cats]

            n_det_cats = len(det_display)
            order = sorted(range(n_det_cats), key=lambda i: bar_means[i], reverse=True)
            order.append(n_det_cats)
            s_cats  = [bar_cats[i]  for i in order]
            s_means = [bar_means[i] for i in order]
            s_sems  = [bar_sems[i]  for i in order]
            s_clrs  = [bar_clrs[i]  for i in order]

            fig.add_trace(
                go.Bar(
                    y=s_cats, x=s_means,
                    orientation="h",
                    marker_color=s_clrs,
                    marker_line_width=0,
                    error_x=dict(type="data", array=s_sems, visible=True,
                                 thickness=0.8, color="#555555"),
                    showlegend=False,
                    hovertemplate="%{y}<br><b>%{x:.3g} s</b> ± %{error_x.array:.2g} s<extra></extra>",
                ),
                row=1, col=2,
            )
            x_right = max(v + s for v, s in zip(s_means, s_sems)) if s_means else 1.0
            fig.add_trace(
                go.Scatter(
                    x=[v + s + x_right * 0.03 for v, s in zip(s_means, s_sems)],
                    y=s_cats,
                    mode="text",
                    text=[f"{v:.3g} s" for v in s_means],
                    textposition="middle right",
                    textfont=dict(size=10, color="#333333"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=2,
            )
            fig.update_xaxes(title_text="Mean time per event (s)",
                             range=[0, x_right * 1.45], row=1, col=2)
            fig.update_yaxes(autorange="reversed", row=1, col=2)

        else:
            # Grouped bar (multi-run)
            all_run_means: dict[str, dict[str, float]] = {}
            all_run_unacc: dict[str, float] = {}
            for lbl in label_list:
                td = filtered[lbl]["time"]
                ed = filtered[lbl]["events"]
                st = _build_stacked_arrays(td, top_dets, all_dets_sorted)
                all_run_means[lbl] = {det: float(arr.mean()) for det, arr in st.items()}
                all_run_unacc[lbl] = max(0.0, float(ed["event_unaccounted_s"].mean()))

            all_bar_dets = det_display + ["Unaccounted"]
            for run_i, lbl in enumerate(label_list):
                run_vals = [all_run_means[lbl].get(d, 0.0) for d in det_display]
                run_vals.append(all_run_unacc[lbl])
                fig.add_trace(
                    go.Bar(
                        y=all_bar_dets, x=run_vals,
                        orientation="h",
                        name=lbl,
                        marker_color=_pal[run_i % len(_pal)],
                        opacity=alpha,
                        marker_line_width=0,
                        hovertemplate=f"<b>{lbl}</b><br>%{{y}}: %{{x:.3g}} s<extra></extra>",
                    ),
                    row=1, col=1,
                )
            fig.update_xaxes(title_text="Mean time per event (s)", row=1, col=1)
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            fig.update_layout(barmode="group")

    # ------------------------------------------------------------------
    # Sequence panel(s)
    # ------------------------------------------------------------------
    if show_seq:
        for run_i, lbl in enumerate(label_list):
            time_df = filtered[lbl]["time"]
            ev_df   = filtered[lbl]["events"]

            event_nums = time_df.index.to_numpy()
            ev_idx     = ev_df.set_index("event_number").reindex(event_nums)
            wall_times = ev_idx["event_wall_s"].to_numpy()
            _unacc_raw = ev_idx["event_unaccounted_s"].to_numpy()
            _neg_frac  = (_unacc_raw < -1e-6).mean()
            if _neg_frac > 0.05:
                warnings.warn(
                    f"'{lbl}': {_neg_frac:.0%} of events have negative unaccounted time "
                    f"(min={_unacc_raw.min():.3g} s). Region timing sum may exceed wall time.",
                    stacklevel=2,
                )
            unaccounted = np.maximum(0.0, _unacc_raw)

            stacked     = _build_stacked_arrays(time_df, top_dets, all_dets_sorted)
            stack_order = sorted(stacked, key=lambda d: stacked[d].mean(), reverse=True)

            if n == 1:
                seq_rc = (2, 1) if show == "both" else (1, 1)
            else:
                seq_row = (run_i + 2) if show == "both" else (run_i + 1)
                seq_rc  = (seq_row, 1)

            for det in stack_order:
                color = det_colors.get(det, _OTHER_COLOR)
                fig.add_trace(
                    go.Scatter(
                        x=event_nums, y=stacked[det],
                        name=det,
                        mode="lines",
                        line=dict(width=0, color=color),
                        fillcolor=_hex_to_rgba(color, alpha),
                        fill="tonexty",
                        stackgroup=f"stack_{lbl}",
                        legendgroup=det,
                        showlegend=(run_i == 0),
                        hovertemplate=f"<b>{det}</b><br>event: %{{x}}<br>%{{y:.4g}} s<extra></extra>",
                    ),
                    row=seq_rc[0], col=seq_rc[1],
                )

            fig.add_trace(
                go.Scatter(
                    x=event_nums, y=unaccounted,
                    name="Unaccounted",
                    mode="lines",
                    line=dict(width=0, color=_UNACCOUNTED_COLOR),
                    fillcolor=_hex_to_rgba(_UNACCOUNTED_COLOR, alpha),
                    fill="tonexty",
                    stackgroup=f"stack_{lbl}",
                    legendgroup="Unaccounted",
                    showlegend=(run_i == 0),
                    hovertemplate="<b>Unaccounted</b><br>event: %{x}<br>%{y:.4g} s<extra></extra>",
                ),
                row=seq_rc[0], col=seq_rc[1],
            )

            fig.add_trace(
                go.Scatter(
                    x=event_nums, y=wall_times,
                    name="Wall time",
                    mode="lines",
                    line=dict(color="black", width=0.9, dash="dash"),
                    opacity=0.6,
                    legendgroup="Wall time",
                    showlegend=(run_i == 0),
                    hovertemplate="<b>Wall time</b><br>event: %{x}<br>%{y:.4g} s<extra></extra>",
                ),
                row=seq_rc[0], col=seq_rc[1],
            )

            fig.update_xaxes(title_text="Event number", row=seq_rc[0], col=seq_rc[1])
            if n > 1:
                fig.update_yaxes(title_text=lbl, title_font=dict(size=9),
                                 row=seq_rc[0], col=seq_rc[1])
            else:
                fig.update_yaxes(title_text="Time (s)", row=seq_rc[0], col=seq_rc[1])

    # ------------------------------------------------------------------
    # Layout
    fig.update_layout(
        template=_TEMPLATE,
        width=px_w,
        height=int(px_h),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.01,
            font=dict(size=9),
        ),
        margin=dict(l=20, r=160, t=30, b=40),
    )

    return fig
