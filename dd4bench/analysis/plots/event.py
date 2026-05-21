"""Per-event timing and memory plots.

Both public functions delegate to the shared ``_plot_event_metric`` implementation,
which differs only in which DataFrame column and axis labels are used.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._theme import _BLUE, _PALETTE, _TEMPLATE
from ._traces import _histogram_traces
from ._utils import (
    _DEFAULT_EXCLUDE_EVENTS,
    _OUTLIER_EXTREME_RATIO,
    _OUTLIER_FRACTION_WARN,
    _compute_core_range,
    _compute_stats,
    _default_baseline,
    _detector_title,
    _ensure_event_data,
)


# ---------------------------------------------------------------------------
# Shared layout builder
# ---------------------------------------------------------------------------

def _build_event_layout(
    show: str,
    n: int,
    px_h: int,
) -> tuple[go.Figure, tuple | None, tuple | None, tuple | None, tuple | None, int]:
    """Build the subplot grid and return (fig, dist_rc, seq_rc, ratio_dist_rc, ratio_seq_rc, total_h)."""
    two_run_ratio = (n == 2) and (show == "both")

    if show == "both" and n == 1:
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
        return fig, (1, 1), (1, 2), None, None, px_h

    if show == "distribution":
        fig = make_subplots(rows=1, cols=1)
        return fig, (1, 1), None, None, None, px_h

    if show == "sequence":
        fig = make_subplots(rows=1, cols=1)
        return fig, None, (1, 1), None, None, px_h

    if two_run_ratio:
        total_h = px_h + int(1.6 * 96)
        ratio_frac = (1.6 * 96) / total_h
        main_frac  = px_h / total_h
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
            row_heights=[main_frac, ratio_frac],
            shared_xaxes="columns",
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )
        fig.update_xaxes(title_standoff=15, row=2, col=1)
        fig.update_xaxes(title_standoff=15, row=2, col=2)
        return fig, (1, 1), (1, 2), (2, 1), (2, 2), total_h

    # n > 2, show == "both"
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
    return fig, (1, 1), (1, 2), None, None, px_h


# ---------------------------------------------------------------------------
# Shared implementation
# ---------------------------------------------------------------------------

def _plot_event_metric(
    source: dict[str, pd.DataFrame] | str | Path | list[str | Path],
    *,
    column: str,
    xlabel: str,
    yseq_label: str,
    stat_prefix: str,
    stat_unit: str,
    warn_name: str,
    warn_unit: str,
    base_title: str,
    labels: list[str] | None = None,
    baseline_label: str | None = None,
    show: str = "both",
    bins: int | str = "auto",
    alpha: float = 0.7,
    figsize: tuple[float, float] | None = None,
    outlier_threshold: float = 3.5,
    exclude_events: list[int] | None = None,
) -> go.Figure:
    if show not in ("both", "distribution", "sequence"):
        raise ValueError(f"show must be 'both', 'distribution', or 'sequence', got {show!r}")

    det_title = _detector_title(source)

    if exclude_events is None:
        exclude_events = list(_DEFAULT_EXCLUDE_EVENTS)

    all_event_data = _ensure_event_data(source)
    available = sorted(all_event_data.keys())
    if labels is None:
        event_data = all_event_data
    else:
        event_data = {
            k: v for k, v in all_event_data.items()
            if k in labels or any(k.endswith(f"/{w}") for w in labels)
        }
    if not event_data:
        raise ValueError(
            f"No *_events.json files found for labels={labels}.\n"
            f"Available: {available}"
        )

    label_list = list(event_data.keys())
    n = len(label_list)

    filtered_data = {
        lbl: df[~df["event_number"].isin(exclude_events)]
        for lbl, df in event_data.items()
    }
    empty_labels = [lbl for lbl, df in filtered_data.items() if df.empty]
    if empty_labels:
        raise ValueError(
            "No events left after applying exclude_events for: "
            + ", ".join(empty_labels)
        )

    required_cols = {column, "event_number"}
    for lbl, df in filtered_data.items():
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"'{lbl}': missing columns {missing}")

    arrays = {lbl: filtered_data[lbl][column].to_numpy() for lbl in label_list}
    all_data = np.concatenate(list(arrays.values()))

    per_run_ranges = [_compute_core_range(arr, threshold=outlier_threshold)[0] for arr in arrays.values()]
    core_range = (min(r[0] for r in per_run_ranges), max(r[1] for r in per_run_ranges))
    clipped_all = all_data[(all_data >= core_range[0]) & (all_data <= core_range[1])]
    if len(clipped_all) == 0:
        clipped_all = all_data
        core_range = (float(all_data.min()), float(all_data.max()))
    _, common_edges = np.histogram(clipped_all, bins=bins)

    two_run_ratio = (n == 2) and (show == "both")

    total_clipped = 0
    pending_warnings: list[str] = []
    for lbl, arr in arrays.items():
        nc = int(np.sum((arr < core_range[0]) | (arr > core_range[1])))
        total_clipped += nc
        if nc > 0:
            frac = nc / len(arr)
            extreme = arr.max() > _OUTLIER_EXTREME_RATIO * core_range[1]
            if frac > _OUTLIER_FRACTION_WARN or extreme:
                pending_warnings.append(
                    f"{warn_name}: {lbl}: {nc} event(s) ({frac:.1%}) outside plotted range. "
                    f"Max value: {arr.max():.4g} {warn_unit}, core upper bound: {core_range[1]:.4g} {warn_unit}. "
                    "Check for simulation anomalies or adjust outlier_threshold."
                )
    for msg in pending_warnings:
        warnings.warn(msg, stacklevel=3)

    clipping_note = ""
    if total_clipped > 0:
        frac = total_clipped / len(all_data)
        clipping_note = f"{total_clipped} event(s) ({frac:.1%}) outside plotted range"

    ref_label = baseline_label if baseline_label is not None else _default_baseline(label_list)
    if n > 1 and ref_label not in arrays:
        raise ValueError(
            f"baseline_label '{ref_label}' not found in loaded runs.\n"
            f"Available: {label_list}"
        )

    px_w = int((figsize[0] if figsize else (12 if show == "both" else 6)) * 96)
    px_h = int((figsize[1] if figsize else 4.5) * 96)

    fig, dist_rc, seq_rc, ratio_dist_rc, ratio_seq_rc, total_h = _build_event_layout(show, n, px_h)

    # ------------------------------------------------------------------
    # Distribution panel
    # ------------------------------------------------------------------
    hist_alpha = alpha if n == 1 else min(alpha, 0.6)
    if dist_rc is not None:
        for tr in _histogram_traces(arrays, common_edges, label_list, hist_alpha, n > 1):
            fig.add_trace(tr, row=dist_rc[0], col=dist_rc[1])

        for i, lbl in enumerate(label_list):
            color = _BLUE if n == 1 else _PALETTE[i % len(_PALETTE)]
            fig.add_vline(
                x=float(arrays[lbl].mean()), line_dash="dash", line_color=color,
                line_width=1.2, opacity=0.8, row=dist_rc[0], col=dist_rc[1],
            )

        fig.update_yaxes(title_text="Count", row=dist_rc[0], col=dist_rc[1])

        if n == 1:
            mean, std, sem, se_std = _compute_stats(arrays[label_list[0]])
            fig.add_annotation(
                xref="x domain", yref="y domain",
                x=0.03, y=0.97,
                text=(
                    f"{stat_prefix} = {mean:.4g} ± {sem:.2g} {stat_unit}<br>"
                    f"σ = {std:.4g} ± {se_std:.2g} {stat_unit}"
                ),
                showarrow=False, align="left",
                bgcolor="white", bordercolor="lightgray", borderwidth=1,
                font=dict(size=11),
                row=dist_rc[0], col=dist_rc[1],
            )
            _xlabel = f"{xlabel}<br><sup>{clipping_note}</sup>" if clipping_note else xlabel
            fig.update_xaxes(title_text=_xlabel, row=dist_rc[0], col=dist_rc[1])
        elif not two_run_ratio:
            _xlabel = f"{xlabel}<br><sup>{clipping_note}</sup>" if clipping_note else xlabel
            fig.update_xaxes(title_text=_xlabel, row=dist_rc[0], col=dist_rc[1])

    # ------------------------------------------------------------------
    # Sequence panel
    # ------------------------------------------------------------------
    if seq_rc is not None:
        for i, lbl in enumerate(label_list):
            color = _BLUE if n == 1 else _PALETTE[i % len(_PALETTE)]
            df_lbl = filtered_data[lbl]
            fig.add_trace(
                go.Scattergl(
                    x=df_lbl["event_number"],
                    y=df_lbl[column],
                    mode="lines",
                    name=lbl,
                    legendgroup=lbl,
                    line=dict(color=color, width=1),
                    opacity=0.7,
                    showlegend=(n > 1 and dist_rc is None),
                    hovertemplate=(
                        f"<b>{lbl}</b><br>event: %{{x}}<br>{yseq_label}: %{{y:.4g}}<extra></extra>"
                    ),
                ),
                row=seq_rc[0], col=seq_rc[1],
            )
        if not two_run_ratio:
            fig.update_xaxes(title_text="Event number", row=seq_rc[0], col=seq_rc[1])
        fig.update_yaxes(title_text=yseq_label, row=seq_rc[0], col=seq_rc[1])

    # ------------------------------------------------------------------
    # Ratio panels (n == 2, show == "both")
    # ------------------------------------------------------------------
    if two_run_ratio:
        other_label = next(lbl for lbl in label_list if lbl != ref_label)

        ref_counts, _   = np.histogram(arrays[ref_label],   bins=common_edges)
        other_counts, _ = np.histogram(arrays[other_label], bins=common_edges)
        ref_counts   = ref_counts.astype(float)
        other_counts = other_counts.astype(float)

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                (ref_counts > 0) & (other_counts > 0),
                other_counts / ref_counts, np.nan,
            )
            ratio_err = np.where(
                (ref_counts > 0) & (other_counts > 0),
                ratio * np.sqrt(1.0 / other_counts + 1.0 / ref_counts), np.nan,
            )

        bin_centers = 0.5 * (common_edges[:-1] + common_edges[1:])
        valid = ~np.isnan(ratio)

        fig.add_trace(
            go.Scatter(
                x=bin_centers[valid], y=ratio[valid],
                error_y=dict(type="data", array=ratio_err[valid], visible=True,
                             thickness=0.8, width=4, color="#444444"),
                mode="markers",
                marker=dict(color="#444444", size=5),
                showlegend=False,
                hovertemplate="bin: %{x:.4g}<br>ratio: %{y:.3f}<extra></extra>",
            ),
            row=ratio_dist_rc[0], col=ratio_dist_rc[1],
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="black", line_width=1.0,
                      opacity=0.5, row=ratio_dist_rc[0], col=ratio_dist_rc[1])
        _xlabel_r = f"{xlabel}<br><sup>{clipping_note}</sup>" if clipping_note else xlabel
        fig.update_xaxes(title_text=_xlabel_r, row=ratio_dist_rc[0], col=ratio_dist_rc[1])
        fig.update_yaxes(title_text="Ratio (other/ref.)", title_font=dict(size=10),
                         row=ratio_dist_rc[0], col=ratio_dist_rc[1])

        df_ref   = event_data[ref_label].set_index("event_number")
        df_other = event_data[other_label].set_index("event_number")
        common_evts = df_ref.index.intersection(df_other.index)
        if len(common_evts) > 0:
            t_ref   = df_ref.loc[common_evts, column].to_numpy()
            t_other = df_other.loc[common_evts, column].to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                evt_ratio = np.where(t_ref > 0, t_other / t_ref, np.nan)
            valid_e = ~np.isnan(evt_ratio)
            fig.add_trace(
                go.Scattergl(
                    x=common_evts[valid_e], y=evt_ratio[valid_e],
                    mode="markers",
                    marker=dict(color="#444444", size=4, opacity=0.5),
                    showlegend=False,
                    hovertemplate="event: %{x}<br>ratio: %{y:.3f}<extra></extra>",
                ),
                row=ratio_seq_rc[0], col=ratio_seq_rc[1],
            )
        fig.add_hline(y=1.0, line_dash="dash", line_color="black", line_width=1.0,
                      opacity=0.5, row=ratio_seq_rc[0], col=ratio_seq_rc[1])
        fig.update_xaxes(title_text="Event number", row=ratio_seq_rc[0], col=ratio_seq_rc[1])
        fig.update_yaxes(title_text="Ratio (other/ref.)", title_font=dict(size=10),
                         row=ratio_seq_rc[0], col=ratio_seq_rc[1])

    if n > 1:
        fig.update_layout(barmode="overlay")

    suptitle_base = base_title if n == 1 else f"{base_title} Comparison"
    suptitle = f"{suptitle_base} — {det_title}" if det_title else suptitle_base

    fig.update_layout(
        title_text=suptitle,
        title_font=dict(size=16, color="#222222"),
        template=_TEMPLATE,
        width=px_w,
        height=total_h,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=10),
    )

    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_event_timing(
    source: dict[str, pd.DataFrame] | str | Path | list[str | Path],
    *,
    labels: list[str] | None = None,
    baseline_label: str | None = None,
    show: str = "both",
    bins: int | str = "auto",
    alpha: float = 0.7,
    figsize: tuple[float, float] | None = None,
    outlier_threshold: float = 3.5,
    exclude_events: list[int] | None = None,
) -> go.Figure:
    """Plot per-event timing distributions for one or more runs.

    Single run: histogram with μ ± SEM and σ ± SE(σ) shown as an annotation.
    Multiple runs: overlaid histograms and, for exactly two runs with
    ``show="both"``, bin-by-bin ratio panels.

    Parameters
    ----------
    source : dict[str, pd.DataFrame], str/Path, or list of str/Path
        Pre-loaded dict from :func:`~dd4bench.analysis.loader.load_event_timing`,
        a single log-dir path, or a list of log-dir paths.
    labels : list[str] or None
        Restrict to these run labels.
    baseline_label : str or None
        Reference run for the ratio panel (multi-run only).
    show : {"both", "distribution", "sequence"}
        Which panels to display.
    bins : int or str
        Bin specification for histograms.
    alpha : float
        Histogram opacity (default: 0.7).
    figsize : (width, height) or None
        Figure size in inches (converted to pixels at 96 dpi).
    outlier_threshold : float
        MAD-based modified Z-score threshold for x-range clipping.
    exclude_events : list[int] or None
        Event numbers to exclude.  Defaults to ``[0]``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    return _plot_event_metric(
        source,
        column="event_time_s",
        xlabel="Event time (s)",
        yseq_label="Event time (s)",
        stat_prefix="μ",
        stat_unit="s",
        warn_name="plot_event_timing",
        warn_unit="s",
        base_title="Per-Event Timing",
        labels=labels,
        baseline_label=baseline_label,
        show=show,
        bins=bins,
        alpha=alpha,
        figsize=figsize,
        outlier_threshold=outlier_threshold,
        exclude_events=exclude_events,
    )


def plot_event_memory(
    source: dict[str, pd.DataFrame] | str | Path | list[str | Path],
    *,
    labels: list[str] | None = None,
    baseline_label: str | None = None,
    show: str = "both",
    bins: int | str = "auto",
    alpha: float = 0.7,
    figsize: tuple[float, float] | None = None,
    outlier_threshold: float = 3.5,
    exclude_events: list[int] | None = None,
) -> go.Figure:
    """Plot per-event memory (RSS) distributions for one or more runs.

    Distribution panel shows a histogram of peak RSS per event.
    Sequence panel shows peak RSS vs event number.

    Parameters
    ----------
    source : dict[str, pd.DataFrame], str/Path, or list of str/Path
        Pre-loaded dict from :func:`~dd4bench.analysis.loader.load_event_timing`,
        a single log-dir path, or a list of log-dir paths.
    labels : list[str] or None
        Restrict to these run labels.
    baseline_label : str or None
        Reference run for the ratio panel (multi-run only).
    show : {"both", "distribution", "sequence"}
        Which panels to display.
    bins : int or str
        Bin specification for histograms.
    alpha : float
        Histogram opacity (default: 0.7).
    figsize : (width, height) or None
        Figure size in inches (converted to pixels at 96 dpi).
    outlier_threshold : float
        MAD-based modified Z-score threshold for x-range clipping.
    exclude_events : list[int] or None
        Event numbers to exclude.  Defaults to ``[0]``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    return _plot_event_metric(
        source,
        column="rss_end_mb",
        xlabel="Peak RSS per event (MB)",
        yseq_label="Peak RSS (MB)",
        stat_prefix="μ<sub>RSS</sub>",
        stat_unit="MB",
        warn_name="plot_event_memory",
        warn_unit="MB",
        base_title="Per-Event Memory",
        labels=labels,
        baseline_label=baseline_label,
        show=show,
        bins=bins,
        alpha=alpha,
        figsize=figsize,
        outlier_threshold=outlier_threshold,
        exclude_events=exclude_events,
    )
