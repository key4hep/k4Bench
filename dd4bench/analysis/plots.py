"""Matplotlib plotting functions for dd4bench results.

All functions return a :class:`~matplotlib.figure.Figure` so the caller
can save, further customise, or let Jupyter display it automatically.
``plt.close`` is called before returning so the inline backend does not
render the figure a second time at end-of-cell.

Typical notebook usage::

    from dd4bench.analysis import load_results, plot_sweep, plot_event_timing

    df = load_results("logs/results.csv")
    plot_sweep(df)

    plot_event_timing("logs/")
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from dd4bench.analysis.loader import load_event_timing, load_region_timing, load_results

# ---------------------------------------------------------------------------
# Style & colours
# ---------------------------------------------------------------------------

_STYLE = "seaborn-v0_8-whitegrid"

_BLUE   = "#1f77b4"
_RED    = "#d62728"

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_METRIC_UNITS: dict[str, str] = {
    "wall_time_s":    "(s)",
    "peak_rss_mb":    "(MB)",
    "user_cpu_s":     "(s)",
    "sys_cpu_s":      "(s)",
    "events_per_sec": "(ev/s)",
    "output_size_mb": "(MB)",
}

_DEFAULT_EXCLUDE_EVENTS = [0]


def _compute_stats(data: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, sem, se_std) with se_std = std / sqrt(2*(n-1))."""
    n = len(data)
    mean = float(data.mean())
    if n > 1:
        std    = float(data.std(ddof=1))
        sem    = std / np.sqrt(n)
        se_std = std / np.sqrt(2 * (n - 1))
    else:
        std = sem = se_std = float("nan")
    return mean, std, sem, se_std


def _use_style() -> None:
    try:
        plt.style.use(_STYLE)
    except OSError:
        pass


def _apply_tick_style(ax: plt.Axes) -> None:
    """Minor ticks + inward ticks on all four sides (scientific style)."""
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=2.5)


def _ensure_df(results: pd.DataFrame | str | Path | list) -> pd.DataFrame:
    """Accept a DataFrame, a single log-dir path, or a list of paths."""
    if isinstance(results, pd.DataFrame):
        return results
    if isinstance(results, list):
        frames = []
        for path in results:
            df = load_results(path).copy()
            prefix = Path(path).name
            df["label"] = df["label"].astype(str).apply(
                lambda lbl, p=prefix: lbl if lbl.startswith(f"{p}/") else f"{p}/{lbl}"
            )
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
    return load_results(results)


def _ensure_event_data(
    source: dict[str, pd.DataFrame] | str | Path | list,
    labels: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Accept a pre-loaded dict, a single log-dir path, or a list of paths."""
    if isinstance(source, dict):
        if labels is not None:
            return {k: v for k, v in source.items() if k in labels}
        return source
    if isinstance(source, list):
        out: dict[str, pd.DataFrame] = {}
        for path in source:
            prefix = Path(path).name
            for lbl, df in load_event_timing(path).items():
                key = lbl if lbl.startswith(f"{prefix}/") else f"{prefix}/{lbl}"
                if key in out:
                    raise ValueError(
                        f"Duplicate label '{key}': two source paths share the directory name "
                        f"'{prefix}'. Rename the directories to disambiguate."
                    )
                out[key] = df
        if labels is not None:
            out = {k: v for k, v in out.items()
                   if k in labels or any(k.endswith(f"/{w}") for w in labels)}
        return out
    return load_event_timing(source, labels=labels)


def _detector_title(source: object) -> str | None:
    """Return a display name from path(s); None for pre-loaded data."""
    if isinstance(source, (str, Path)):
        return Path(source).name
    if isinstance(source, list) and source:
        return " vs ".join(Path(s).name for s in source)
    return None


def _default_baseline(labels: list[str]) -> str:
    """Return the first label as the default baseline reference."""
    return labels[0]


def _matches_baseline(label: str, baseline_label: str | None) -> bool:
    """True if label is the baseline, supporting both plain and prefixed labels."""
    if baseline_label is None:
        return False
    return label == baseline_label or label.endswith(f"/{baseline_label}")


# ---------------------------------------------------------------------------
# plot_run_overview
# ---------------------------------------------------------------------------

_OVERVIEW_METRICS = [
    ("wall_time_s",    f"Wall Time {_METRIC_UNITS['wall_time_s']}"),
    ("peak_rss_mb",    f"Peak RSS {_METRIC_UNITS['peak_rss_mb']}"),
    ("user_cpu_s",     f"User CPU {_METRIC_UNITS['user_cpu_s']}"),
    ("events_per_sec", f"Throughput {_METRIC_UNITS['events_per_sec']}"),
]


def plot_run_overview(
    results: pd.DataFrame | str | Path | list[str | Path],
    *,
    labels: list[str] | None = None,
    metrics: list[tuple[str, str]] | None = None,
    relative: bool = True,
    baseline_label: str | None = None,
) -> plt.Figure:
    """Plot run metrics for all runs in a 2 × 2 panel grid.

    Each metric is drawn as a horizontal bar chart with value annotations.

    Parameters
    ----------
    results : pd.DataFrame, str/Path, or list of str/Path
        Results DataFrame, a single log-dir path, or a list of log-dir paths
        for multi-detector comparisons.  When a list is given, run labels are
        prefixed with the directory name (e.g. ``ALLEGRO_o1_v03/baseline_all``).
    labels : list[str] or None
        Show only these run labels.  Supports plain (``"baseline_all"``) and
        prefixed (``"ALLEGRO_o1_v03/baseline_all"``) labels.
    metrics : list of (column, axis-label) pairs or None
        Which metrics to plot.  Defaults to wall_time_s, peak_rss_mb,
        user_cpu_s, events_per_sec.
    relative : bool
        If ``True``, normalise every metric to the baseline run (= 100 %).
        A reference line is drawn at 100 % and the absolute baseline value
        is shown in the x-axis label.  Defaults to ``True``.
    baseline_label : str or None
        Which run to treat as 100 % when ``relative=True``.
        Defaults to the first run in the data.  Ignored when ``relative=False``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _use_style()
    det_title = _detector_title(results)
    results = _ensure_df(results)

    if metrics is None:
        metrics = _OVERVIEW_METRICS

    if labels is not None:
        results = results[results["label"].apply(
            lambda lbl: lbl in labels or any(lbl.endswith(f"/{w}") for w in labels)
        )]

    metric_cols = [col for col, _ in metrics if col in results.columns]
    df = results.dropna(subset=metric_cols, how="all").copy()

    # Capture label order before display-sort so default baseline is deterministic.
    load_order_labels = df["label"].tolist()

    baseline_vals: dict[str, float] = {}
    if relative:
        _bl = baseline_label if baseline_label is not None else _default_baseline(load_order_labels)
        bl_mask = df["label"].apply(lambda l: _matches_baseline(l, _bl))
        if not bl_mask.any():
            hint = " Pass baseline_label=... to specify the reference run." if baseline_label is None else ""
            raise ValueError(f"baseline_label '{_bl}' not found for relative=True.{hint}")
        for col, _ in metrics:
            if col in df.columns:
                bv = float(df.loc[bl_mask, col].iloc[0])
                baseline_vals[col] = bv
                if bv == 0:
                    raise ValueError(
                        f"Baseline value for metric '{col}' is 0 — cannot normalise to percentage."
                    )
                df[col] = df[col] / bv * 100

    if "wall_time_s" in df.columns:
        df = df.sort_values("wall_time_s", ascending=True)

    run_labels = df["label"].tolist()
    n_runs = len(run_labels)

    prefixes = [lbl.split("/")[0] if "/" in lbl else None for lbl in run_labels]
    unique_prefixes = list(dict.fromkeys(p for p in prefixes if p is not None))
    if unique_prefixes:
        prefix_color = {p: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(unique_prefixes)}
        bar_colors = [prefix_color[p] for p in prefixes]
    else:
        bar_colors = [_BLUE] * n_runs

    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("metrics must contain at least one metric")

    ncols = 2
    nrows = (n_metrics + 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, max(4.0, 0.45 * n_runs + 2.5) * nrows / 2),
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (col, ylabel) in zip(axes_flat, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        values = df[col].tolist()
        valid_v = [v for v in values if pd.notna(v)]
        x_max = max(valid_v) if valid_v else 1.0

        ax.barh(run_labels, values, color=bar_colors,
                edgecolor="white", linewidth=0.6, height=0.55)

        for i, val in enumerate(values):
            if pd.notna(val):
                bar_y = i
                label_str = f"{val:.1f}%" if relative else f"{val:.4g}"
                ax.text(val + 0.01 * x_max, bar_y,
                        label_str, va="center", ha="left", fontsize=8, color="#444444")

        if relative:
            ax.axvline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            bv = baseline_vals.get(col)
            unit = _METRIC_UNITS.get(col, "")
            bv_str = f"{bv:.4g} {unit}".strip() if bv is not None else ""
            xlabel = ylabel.split(" (")[0] + f" % of baseline = {bv_str}"
        else:
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            xlabel = ylabel
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0, right=x_max * 1.18)
        ax.grid(False)
        _apply_tick_style(ax)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    title = f"Run Metrics Overview ({det_title})" if det_title else "Run Metrics Overview"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Outlier-robust range helper
# ---------------------------------------------------------------------------

_OUTLIER_FRACTION_WARN = 0.05   # warn if more than 5 % of events are clipped
_OUTLIER_EXTREME_RATIO  = 5.0   # warn if max value exceeds 5× the core upper bound


def _compute_core_range(
    data: np.ndarray,
    threshold: float = 3.5,
) -> tuple[tuple[float, float], int]:
    """Return an x-range that excludes outliers and the number of clipped points.

    Uses the modified Z-score (Iglewicz & Hoaglin, 1993) with MAD as the
    spread estimate, which is robust for skewed or heavy-tailed distributions.

    Parameters
    ----------
    data : np.ndarray
        1-D array of values.
    threshold : float
        Modified Z-score threshold above which a point is considered an
        outlier (default: 3.5, as recommended by Iglewicz & Hoaglin).

    Returns
    -------
    (x_min, x_max) : tuple[float, float]
        Range that covers the core distribution, with a small margin.
    n_clipped : int
        Number of data points outside ``(x_min, x_max)``.
    """
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))

    if mad > 0:
        # 0.6745 = Phi^{-1}(3/4): normalises MAD to be consistent with sigma
        modified_z = 0.6745 * np.abs(data - median) / mad
        mask = modified_z <= threshold
        core = data[mask]
        n_clipped = int((~mask).sum())
    else:
        core = data  # all values identical — no outliers to remove
        n_clipped = 0

    if len(core) == 0:
        core = data
        n_clipped = 0

    x_min, x_max = float(core.min()), float(core.max())
    margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.01 * abs(x_max)
    x_min = max(0.0, x_min - margin)
    x_max = x_max + margin

    return (x_min, x_max), n_clipped


# ---------------------------------------------------------------------------
# plot_event_timing
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
) -> plt.Figure:
    """Plot per-event timing distributions for one or more runs.

    Single run: histogram with μ ± SEM and σ ± SE(σ) shown directly on the plot.
    Multiple runs: overlaid histograms with a stats comparison table and, for
    exactly two runs with ``show="both"``, bin-by-bin ratio panels.

    Parameters
    ----------
    source : dict[str, pd.DataFrame], str/Path, or list of str/Path
        Pre-loaded dict from :func:`~dd4bench.analysis.loader.load_event_timing`,
        a single log-dir path, or a list of log-dir paths for multi-detector
        comparisons.  When a list is given, run labels are prefixed with the
        directory name (e.g. ``ALLEGRO_o1_v03/baseline_all``).
    labels : list[str] or None
        Restrict to these run labels.  Loads all runs when ``None``.
    baseline_label : str or None
        Reference run for Δμ in the stats table (multi-run only).
        Defaults to the first run.
    show : {"both", "distribution", "sequence"}
        ``"both"`` (default): histogram + event-time-vs-event-number side by side.
        ``"distribution"``: histogram panel only.
        ``"sequence"``: event-time-vs-event-number panel only.
    bins : int or str
        Bin specification for histograms.  ``"auto"`` uses NumPy's estimator.
    alpha : float
        Histogram opacity (default: 0.7).
    figsize : (width, height) or None
        Size of the main plot area in inches.  Defaults to ``(12, 4.5)`` for
        ``show="both"`` and ``(6, 4.5)`` for single-panel modes.
    outlier_threshold : float
        MAD-based modified Z-score threshold for x-range clipping (default: 3.5).
    exclude_events : list[int] or None
        Event numbers to exclude from statistics and histograms.
        Defaults to ``[0]``.  Pass ``[]`` to disable.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if show not in ("both", "distribution", "sequence"):
        raise ValueError(f"show must be 'both', 'distribution', or 'sequence', got {show!r}")

    _use_style()
    det_title = _detector_title(source)

    if exclude_events is None:
        exclude_events = [0]

    all_event_data = _ensure_event_data(source)
    available = sorted(all_event_data.keys())
    event_data = _ensure_event_data(source, labels=labels)
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

    arrays = {lbl: filtered_data[lbl]["event_time_s"].to_numpy() for lbl in label_list}
    all_data = np.concatenate(list(arrays.values()))
    # Per-run ranges unioned so no distribution gets clipped by a differently-scaled run.
    per_run_ranges = [_compute_core_range(arr, threshold=outlier_threshold)[0] for arr in arrays.values()]
    core_range = (min(r[0] for r in per_run_ranges), max(r[1] for r in per_run_ranges))
    clipped_all = all_data[(all_data >= core_range[0]) & (all_data <= core_range[1])]
    if len(clipped_all) == 0:
        clipped_all = all_data
    _, common_edges = np.histogram(clipped_all, bins=bins)

    show_dist = show in ("both", "distribution")
    show_seq  = show in ("both", "sequence")
    ncols = 2 if show == "both" else 1

    if figsize is None:
        figsize = (12, 4.5) if (show == "both" or n > 1) else (6, 4.5)

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    per_row_h   = 0.38
    table_h     = (n + 1.5) * per_row_h
    two_run_ratio = (n == 2) and (show == "both")

    if n == 1 or show != "both":
        fig, ax_arr = plt.subplots(1, ncols, figsize=figsize)
        ax_arr  = np.atleast_1d(ax_arr)
        ax_dist = ax_arr[0] if show_dist else None
        ax_seq  = ax_arr[1] if show == "both" else (ax_arr[0] if show_seq else None)
        ax_rdist = ax_rseq = ax_table = None
    elif two_run_ratio:
        ratio_h = 1.6
        total_h = figsize[1] + ratio_h + table_h + 1.0
        fig = plt.figure(figsize=(figsize[0], total_h))
        gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[figsize[1] + ratio_h, table_h])
        gs_top   = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_outer[0],
            height_ratios=[figsize[1], ratio_h], hspace=0.05,
        )
        ax_dist  = fig.add_subplot(gs_top[0, 0])
        ax_seq   = fig.add_subplot(gs_top[0, 1])
        ax_rdist = fig.add_subplot(gs_top[1, 0], sharex=ax_dist)
        ax_rseq  = fig.add_subplot(gs_top[1, 1])
        ax_table = fig.add_subplot(gs_outer[1])
        ax_table.axis("off")
    else:
        total_h = figsize[1] + table_h + 1.0
        fig = plt.figure(figsize=(figsize[0], total_h))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[figsize[1], table_h])
        ax_dist  = fig.add_subplot(gs[0, 0])
        ax_seq   = fig.add_subplot(gs[0, 1])
        ax_table = fig.add_subplot(gs[1, :])
        ax_rdist = ax_rseq = None
        ax_table.axis("off")

    # -----------------------------------------------------------------------
    # Reference run for Δμ (multi-run only)
    # -----------------------------------------------------------------------
    ref_label = baseline_label if baseline_label is not None else _default_baseline(label_list)
    if n > 1 and ref_label not in arrays:
        raise ValueError(
            f"baseline_label '{ref_label}' not found in loaded runs.\n"
            f"Available: {label_list}"
        )
    ref_mean, _, ref_sem, _ = _compute_stats(arrays[ref_label]) if n > 1 else (None, None, None, None)

    # -----------------------------------------------------------------------
    # Draw histograms and sequence lines
    # -----------------------------------------------------------------------
    hist_alpha  = alpha if n == 1 else min(alpha, 0.6)
    total_clipped = 0
    table_rows  = []
    hist_counts = {}

    for i, lbl in enumerate(label_list):
        data  = arrays[lbl]
        color = _BLUE if n == 1 else _PALETTE[i % len(_PALETTE)]
        mean, std, sem, se_std = _compute_stats(data)

        if ax_dist is not None:
            ax_dist.hist(
                data, bins=common_edges,
                color=color, edgecolor="none", alpha=hist_alpha,
                label=lbl if n > 1 else None,
            )
            ax_dist.axvline(mean, color=color, linestyle="--", linewidth=1.2, alpha=0.8)

        if ax_seq is not None:
            ax_seq.plot(
                filtered_data[lbl]["event_number"], filtered_data[lbl]["event_time_s"],
                color=color, alpha=0.7, linewidth=1.0,
                label=lbl if n > 1 else None,
            )

        n_clipped = int(np.sum((data < core_range[0]) | (data > core_range[1])))
        total_clipped += n_clipped
        if n_clipped > 0:
            frac = n_clipped / len(data)
            extreme = data.max() > _OUTLIER_EXTREME_RATIO * core_range[1]
            if frac > _OUTLIER_FRACTION_WARN or extreme:
                warnings.warn(
                    f"plot_event_timing: {lbl}: {n_clipped} event(s) ({frac:.1%}) outside plotted range. "
                    f"Max value: {data.max():.4g} s, core upper bound: {core_range[1]:.4g} s. "
                    "Check for simulation anomalies or adjust outlier_threshold.",
                    stacklevel=2,
                )

        if n > 1:
            counts, _ = np.histogram(data, bins=common_edges)
            hist_counts[lbl] = counts.astype(float)
            if lbl == ref_label:
                delta_cell = "ref."
            else:
                delta_pct = (mean - ref_mean) / ref_mean * 100
                delta_err = (100.0 / ref_mean) * np.sqrt(
                    sem**2 + (mean / ref_mean)**2 * ref_sem**2
                )
                sign = "+" if delta_pct >= 0 else ""
                delta_cell = f"{sign}{delta_pct:.1f} ± {delta_err:.1f}%"
            table_rows.append([lbl, f"{mean:.4g} ± {sem:.2g}", f"{std:.4g}", delta_cell])

    clipping_note = ""
    if total_clipped > 0:
        frac = total_clipped / len(all_data)
        clipping_note = f"{total_clipped} event(s) ({frac:.1%}) outside plotted range"

    # -----------------------------------------------------------------------
    # Style distribution panel
    # -----------------------------------------------------------------------
    if ax_dist is not None:
        ax_dist.set_xlim(core_range)
        ax_dist.set_ylabel("Count")
        ax_dist.set_title(
            f"Event Time Distribution ({label_list[0]})" if n == 1
            else "Event Time Distribution"
        )
        ax_dist.grid(False)
        _apply_tick_style(ax_dist)

        if n == 1:
            mean, std, sem, se_std = _compute_stats(arrays[label_list[0]])
            ax_dist.text(
                0.03, 0.97,
                f"$\\mu = {mean:.4g} \\pm {sem:.2g}$ s\n$\\sigma = {std:.4g} \\pm {se_std:.2g}$ s",
                transform=ax_dist.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7),
            )
            xlabel = "Event time (s)"
            if clipping_note:
                xlabel += f"\n\n{clipping_note}"
            ax_dist.set_xlabel(xlabel)
        else:
            ax_dist.legend(loc="upper right", fontsize="small")
            if not two_run_ratio:
                xlabel = "Event time (s)"
                if clipping_note:
                    xlabel += f"\n\n{clipping_note}"
                ax_dist.set_xlabel(xlabel)
            else:
                plt.setp(ax_dist.get_xticklabels(), visible=False)

    # -----------------------------------------------------------------------
    # Style sequence panel
    # -----------------------------------------------------------------------
    if ax_seq is not None:
        ax_seq.set_xlabel("Event number")
        ax_seq.set_ylabel("Event time (s)")
        ax_seq.set_title("Event Time vs Event Number")
        ax_seq.grid(False)
        _apply_tick_style(ax_seq)
        if n > 1:
            ax_seq.legend(loc="upper right", fontsize="small")
        if two_run_ratio:
            plt.setp(ax_seq.get_xticklabels(), visible=False)
            ax_seq.set_xlabel("")

    # -----------------------------------------------------------------------
    # Ratio panels (n == 2, show == "both")
    # -----------------------------------------------------------------------
    if two_run_ratio:
        other_label = next(lbl for lbl in label_list if lbl != ref_label)
        other_color = _PALETTE[label_list.index(other_label) % len(_PALETTE)]

        ref_counts_arr   = hist_counts[ref_label]
        other_counts_arr = hist_counts[other_label]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                (ref_counts_arr > 0) & (other_counts_arr > 0),
                other_counts_arr / ref_counts_arr, np.nan,
            )
            ratio_err = np.where(
                (ref_counts_arr > 0) & (other_counts_arr > 0),
                ratio * np.sqrt(1.0 / other_counts_arr + 1.0 / ref_counts_arr), np.nan,
            )
        bin_centers = 0.5 * (common_edges[:-1] + common_edges[1:])
        valid = ~np.isnan(ratio)
        ax_rdist.errorbar(
            bin_centers[valid], ratio[valid], yerr=ratio_err[valid],
            fmt="o", color="#444444",
            markersize=4, capsize=3, linewidth=0, elinewidth=0.8, capthick=0.8, alpha=0.75,
        )
        ax_rdist.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax_rdist.set_xlim(core_range)
        ax_rdist.set_ylabel("Ratio (other / ref.)", fontsize=8)
        ax_rdist.grid(False)
        _apply_tick_style(ax_rdist)
        xlabel_bottom = "Event time (s)"
        if clipping_note:
            xlabel_bottom += f"\n\n{clipping_note}"
        ax_rdist.set_xlabel(xlabel_bottom)

        df_ref   = event_data[ref_label].set_index("event_number")
        df_other = event_data[other_label].set_index("event_number")
        common_evts = df_ref.index.intersection(df_other.index)
        if len(common_evts) > 0:
            t_ref   = df_ref.loc[common_evts, "event_time_s"].to_numpy()
            t_other = df_other.loc[common_evts, "event_time_s"].to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                evt_ratio = np.where(t_ref > 0, t_other / t_ref, np.nan)
            ax_rseq.scatter(common_evts, evt_ratio, color="#444444", s=12, alpha=0.5, linewidths=0)
        ax_rseq.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax_rseq.set_xlabel("Event number")
        ax_rseq.set_ylabel("Ratio (other / ref.)", fontsize=8)
        ax_rseq.grid(False)
        _apply_tick_style(ax_rseq)

    # -----------------------------------------------------------------------
    # Stats table (multi-run)
    # -----------------------------------------------------------------------
    if ax_table is not None:
        n_cols      = 4
        col_headers = ["Run", "μ ± SEM (s)", "σ (s)", f"Δμ ± δ(Δμ)  [ref: {ref_label}]"]
        tbl = ax_table.table(
            cellText=table_rows, colLabels=col_headers,
            bbox=[0, 0, 1, 1], cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(col=list(range(n_cols)))
        for j in range(n_cols):
            cell = tbl[(0, j)]
            cell.set_facecolor("#d4d4d4")
            cell.set_text_props(fontweight="bold")
        for i in range(n):
            r, g, b = mcolors.to_rgb(_PALETTE[i % len(_PALETTE)])
            tint = (r, g, b, 0.12)
            for j in range(n_cols):
                tbl[(i + 1, j)].set_facecolor(tint)

    base_title = "Per-Event Timing" if n == 1 else "Per-Event Timing Comparison"
    suptitle = f"{base_title} ({det_title})" if det_title else base_title
    fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# plot_event_memory
# ---------------------------------------------------------------------------


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
) -> plt.Figure:
    """Plot per-event memory (RSS) distributions for one or more runs.

    Distribution panel shows a histogram of peak RSS per event (retained memory
    after each event).  Sequence panel shows the same values vs event number
    (memory profile over the run).

    Single run: histogram with μ ± SEM and σ ± SE(σ) shown directly on the plot.
    Multiple runs: overlaid histograms with a stats comparison table and, for
    exactly two runs with ``show="both"``, bin-by-bin ratio panels.

    Parameters
    ----------
    source : dict[str, pd.DataFrame], str/Path, or list of str/Path
        Pre-loaded dict from :func:`~dd4bench.analysis.loader.load_event_timing`,
        a single log-dir path, or a list of log-dir paths for multi-detector
        comparisons.  When a list is given, run labels are prefixed with the
        directory name (e.g. ``ALLEGRO_o1_v03/baseline_all``).
    labels : list[str] or None
        Restrict to these run labels.  Loads all runs when ``None``.
    baseline_label : str or None
        Reference run for Δμ in the stats table (multi-run only).
        Defaults to the first run.
    show : {"both", "distribution", "sequence"}
        ``"both"`` (default): RSS Δ histogram + peak RSS vs event number.
        ``"distribution"``: histogram panel only.
        ``"sequence"``: peak RSS vs event number panel only.
    bins : int or str
        Bin specification for histograms.  ``"auto"`` uses NumPy's estimator.
    alpha : float
        Histogram opacity (default: 0.7).
    figsize : (width, height) or None
        Size of the main plot area in inches.  Defaults to ``(12, 4.5)`` for
        ``show="both"`` and ``(6, 4.5)`` for single-panel modes.
    outlier_threshold : float
        MAD-based modified Z-score threshold for x-range clipping (default: 3.5).
    exclude_events : list[int] or None
        Event numbers to exclude from statistics and histograms.
        Defaults to ``[0]``.  Pass ``[]`` to disable.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if show not in ("both", "distribution", "sequence"):
        raise ValueError(f"show must be 'both', 'distribution', or 'sequence', got {show!r}")

    _use_style()
    det_title = _detector_title(source)

    if exclude_events is None:
        exclude_events = [0]

    all_event_data = _ensure_event_data(source)
    available = sorted(all_event_data.keys())
    event_data = _ensure_event_data(source, labels=labels)
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

    arrays = {lbl: filtered_data[lbl]["rss_end_mb"].to_numpy() for lbl in label_list}
    all_data = np.concatenate(list(arrays.values()))
    per_run_ranges = [_compute_core_range(arr, threshold=outlier_threshold)[0] for arr in arrays.values()]
    core_range = (min(r[0] for r in per_run_ranges), max(r[1] for r in per_run_ranges))
    clipped_all = all_data[(all_data >= core_range[0]) & (all_data <= core_range[1])]
    if len(clipped_all) == 0:
        clipped_all = all_data
    _, common_edges = np.histogram(clipped_all, bins=bins)

    show_dist = show in ("both", "distribution")
    show_seq  = show in ("both", "sequence")
    ncols = 2 if show == "both" else 1

    if figsize is None:
        figsize = (12, 4.5) if (show == "both" or n > 1) else (6, 4.5)

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    per_row_h     = 0.38
    table_h       = (n + 1.5) * per_row_h
    two_run_ratio = (n == 2) and (show == "both")

    if n == 1 or show != "both":
        fig, ax_arr = plt.subplots(1, ncols, figsize=figsize)
        ax_arr  = np.atleast_1d(ax_arr)
        ax_dist = ax_arr[0] if show_dist else None
        ax_seq  = ax_arr[1] if show == "both" else (ax_arr[0] if show_seq else None)
        ax_rdist = ax_rseq = ax_table = None
    elif two_run_ratio:
        ratio_h = 1.6
        total_h = figsize[1] + ratio_h + table_h + 1.0
        fig = plt.figure(figsize=(figsize[0], total_h))
        gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[figsize[1] + ratio_h, table_h])
        gs_top   = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_outer[0],
            height_ratios=[figsize[1], ratio_h], hspace=0.05,
        )
        ax_dist  = fig.add_subplot(gs_top[0, 0])
        ax_seq   = fig.add_subplot(gs_top[0, 1])
        ax_rdist = fig.add_subplot(gs_top[1, 0], sharex=ax_dist)
        ax_rseq  = fig.add_subplot(gs_top[1, 1])
        ax_table = fig.add_subplot(gs_outer[1])
        ax_table.axis("off")
    else:
        total_h = figsize[1] + table_h + 1.0
        fig = plt.figure(figsize=(figsize[0], total_h))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[figsize[1], table_h])
        ax_dist  = fig.add_subplot(gs[0, 0])
        ax_seq   = fig.add_subplot(gs[0, 1])
        ax_table = fig.add_subplot(gs[1, :])
        ax_rdist = ax_rseq = None
        ax_table.axis("off")

    # -----------------------------------------------------------------------
    # Reference run for Δμ (multi-run only)
    # -----------------------------------------------------------------------
    ref_label = baseline_label if baseline_label is not None else _default_baseline(label_list)
    if n > 1 and ref_label not in arrays:
        raise ValueError(
            f"baseline_label '{ref_label}' not found in loaded runs.\n"
            f"Available: {label_list}"
        )
    ref_mean, _, ref_sem, _ = _compute_stats(arrays[ref_label]) if n > 1 else (None, None, None, None)

    # -----------------------------------------------------------------------
    # Draw histograms and sequence lines
    # -----------------------------------------------------------------------
    hist_alpha  = alpha if n == 1 else min(alpha, 0.6)
    total_clipped = 0
    table_rows  = []
    hist_counts = {}

    for i, lbl in enumerate(label_list):
        data  = arrays[lbl]
        color = _BLUE if n == 1 else _PALETTE[i % len(_PALETTE)]
        mean, std, sem, se_std = _compute_stats(data)

        if ax_dist is not None:
            ax_dist.hist(
                data, bins=common_edges,
                color=color, edgecolor="none", alpha=hist_alpha,
                label=lbl if n > 1 else None,
            )
            ax_dist.axvline(mean, color=color, linestyle="--", linewidth=1.2, alpha=0.8)

        if ax_seq is not None:
            ax_seq.plot(
                filtered_data[lbl]["event_number"], filtered_data[lbl]["rss_end_mb"],
                color=color, alpha=0.7, linewidth=1.0,
                label=lbl if n > 1 else None,
            )

        n_clipped = int(np.sum((data < core_range[0]) | (data > core_range[1])))
        total_clipped += n_clipped
        if n_clipped > 0:
            frac = n_clipped / len(data)
            extreme = data.max() > _OUTLIER_EXTREME_RATIO * core_range[1]
            if frac > _OUTLIER_FRACTION_WARN or extreme:
                warnings.warn(
                    f"plot_event_memory: {lbl}: {n_clipped} event(s) ({frac:.1%}) outside plotted range. "
                    f"Max RSS: {data.max():.4g} MB, core upper bound: {core_range[1]:.4g} MB. "
                    "Check for simulation anomalies or adjust outlier_threshold.",
                    stacklevel=2,
                )

        if n > 1:
            counts, _ = np.histogram(data, bins=common_edges)
            hist_counts[lbl] = counts.astype(float)
            if lbl == ref_label:
                delta_cell = "ref."
            else:
                delta_pct = (mean - ref_mean) / ref_mean * 100
                delta_err = (100.0 / ref_mean) * np.sqrt(
                    sem**2 + (mean / ref_mean)**2 * ref_sem**2
                )
                sign = "+" if delta_pct >= 0 else ""
                delta_cell = f"{sign}{delta_pct:.1f} ± {delta_err:.1f}%"
            table_rows.append([lbl, f"{mean:.4g} ± {sem:.2g}", f"{std:.4g}", delta_cell])

    clipping_note = ""
    if total_clipped > 0:
        frac = total_clipped / len(all_data)
        clipping_note = f"{total_clipped} event(s) ({frac:.1%}) outside plotted range"

    # -----------------------------------------------------------------------
    # Style distribution panel
    # -----------------------------------------------------------------------
    if ax_dist is not None:
        ax_dist.set_xlim(core_range)
        ax_dist.set_ylabel("Count")
        ax_dist.grid(False)
        _apply_tick_style(ax_dist)

        if n == 1:
            mean, std, sem, se_std = _compute_stats(arrays[label_list[0]])
            ax_dist.text(
                0.03, 0.97,
                f"$\\mu_{{\\rm RSS}} = {mean:.4g} \\pm {sem:.2g}$ MB\n$\\sigma = {std:.4g} \\pm {se_std:.2g}$ MB",
                transform=ax_dist.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7),
            )
            xlabel = "Peak RSS per event (MB)"
            if clipping_note:
                xlabel += f"\n\n{clipping_note}"
            ax_dist.set_xlabel(xlabel)
        else:
            ax_dist.legend(loc="upper right", fontsize="small")
            if not two_run_ratio:
                xlabel = "Peak RSS per event (MB)"
                if clipping_note:
                    xlabel += f"\n\n{clipping_note}"
                ax_dist.set_xlabel(xlabel)
            else:
                plt.setp(ax_dist.get_xticklabels(), visible=False)

    # -----------------------------------------------------------------------
    # Style sequence panel
    # -----------------------------------------------------------------------
    if ax_seq is not None:
        ax_seq.set_xlabel("Event number")
        ax_seq.set_ylabel("Peak RSS (MB)")
        ax_seq.grid(False)
        _apply_tick_style(ax_seq)
        if n > 1:
            ax_seq.legend(loc="upper right", fontsize="small")
        if two_run_ratio:
            plt.setp(ax_seq.get_xticklabels(), visible=False)
            ax_seq.set_xlabel("")

    # -----------------------------------------------------------------------
    # Ratio panels (n == 2, show == "both")
    # -----------------------------------------------------------------------
    if two_run_ratio:
        other_label = next(lbl for lbl in label_list if lbl != ref_label)

        ref_counts_arr   = hist_counts[ref_label]
        other_counts_arr = hist_counts[other_label]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                (ref_counts_arr > 0) & (other_counts_arr > 0),
                other_counts_arr / ref_counts_arr, np.nan,
            )
            ratio_err = np.where(
                (ref_counts_arr > 0) & (other_counts_arr > 0),
                ratio * np.sqrt(1.0 / other_counts_arr + 1.0 / ref_counts_arr), np.nan,
            )
        bin_centers = 0.5 * (common_edges[:-1] + common_edges[1:])
        valid = ~np.isnan(ratio)
        ax_rdist.errorbar(
            bin_centers[valid], ratio[valid], yerr=ratio_err[valid],
            fmt="o", color="#444444",
            markersize=4, capsize=3, linewidth=0, elinewidth=0.8, capthick=0.8, alpha=0.75,
        )
        ax_rdist.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax_rdist.set_xlim(core_range)
        ax_rdist.set_ylabel("Ratio (other / ref.)", fontsize=8)
        ax_rdist.grid(False)
        _apply_tick_style(ax_rdist)
        xlabel_bottom = "Peak RSS per event (MB)"
        if clipping_note:
            xlabel_bottom += f"\n\n{clipping_note}"
        ax_rdist.set_xlabel(xlabel_bottom)

        df_ref   = event_data[ref_label].set_index("event_number")
        df_other = event_data[other_label].set_index("event_number")
        common_evts = df_ref.index.intersection(df_other.index)
        if len(common_evts) > 0:
            rss_ref   = df_ref.loc[common_evts, "rss_end_mb"].to_numpy()
            rss_other = df_other.loc[common_evts, "rss_end_mb"].to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                evt_ratio = np.where(rss_ref > 0, rss_other / rss_ref, np.nan)
            ax_rseq.scatter(common_evts, evt_ratio, color="#444444", s=12, alpha=0.5, linewidths=0)
        ax_rseq.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax_rseq.set_xlabel("Event number")
        ax_rseq.set_ylabel("Ratio (other / ref.)", fontsize=8)
        ax_rseq.grid(False)
        _apply_tick_style(ax_rseq)

    # -----------------------------------------------------------------------
    # Stats table (multi-run)
    # -----------------------------------------------------------------------
    if ax_table is not None:
        n_cols      = 4
        col_headers = ["Run", "μ ± SEM (MB)", "σ (MB)", f"Δμ ± δ(Δμ)  [ref: {ref_label}]"]
        tbl = ax_table.table(
            cellText=table_rows, colLabels=col_headers,
            bbox=[0, 0, 1, 1], cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(col=list(range(n_cols)))
        for j in range(n_cols):
            cell = tbl[(0, j)]
            cell.set_facecolor("#d4d4d4")
            cell.set_text_props(fontweight="bold")
        for i in range(n):
            r, g, b = mcolors.to_rgb(_PALETTE[i % len(_PALETTE)])
            tint = (r, g, b, 0.12)
            for j in range(n_cols):
                tbl[(i + 1, j)].set_facecolor(tint)

    base_title = "Per-Event Memory" if n == 1 else "Per-Event Memory Comparison"
    suptitle = f"{base_title} ({det_title})" if det_title else base_title
    fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Region timing helpers
# ---------------------------------------------------------------------------

_UNACCOUNTED_COLOR = "#999999"
_OTHER_COLOR       = "#d0d0d0"


def _ensure_region_data(
    source: dict[str, dict] | str | Path | list[str | Path],
    labels: list[str] | None = None,
) -> dict[str, dict]:
    """Accept a pre-loaded dict, a single log-dir path, or a list of paths."""
    if isinstance(source, dict):
        if labels is not None:
            return {k: v for k, v in source.items() if k in labels}
        return source
    if isinstance(source, list):
        out: dict[str, dict] = {}
        for path in source:
            prefix = Path(path).name
            for lbl, data in load_region_timing(path).items():
                key = lbl if lbl.startswith(f"{prefix}/") else f"{prefix}/{lbl}"
                if key in out:
                    raise ValueError(
                        f"Duplicate label '{key}': two source paths share the directory name "
                        f"'{prefix}'. Rename the directories to disambiguate."
                    )
                out[key] = data
        if labels is not None:
            out = {k: v for k, v in out.items()
                   if k in labels or any(k.endswith(f"/{w}") for w in labels)}
        return out
    return load_region_timing(source, labels=labels)


def _region_top_n(
    time_df: pd.DataFrame,
    top_n: int,
) -> tuple[list[str], list[str]]:
    """Return (top_dets, all_dets_sorted) by mean time descending, skipping zero-time columns."""
    means = time_df.mean()
    active = means[means > 0].sort_values(ascending=False)
    all_sorted = active.index.tolist()
    return all_sorted[:top_n], all_sorted


def _build_stacked_arrays(
    time_df: pd.DataFrame,
    top_dets: list[str],
    all_dets_sorted: list[str],
) -> dict[str, np.ndarray]:
    """Build per-detector time arrays; detectors outside top_dets are grouped as 'Other'.

    Any columns in *time_df* not covered by *all_dets_sorted* are also folded into 'Other'
    so no data is silently dropped when runs have different detector sets.
    """
    n = len(time_df)
    arrays: dict[str, np.ndarray] = {}
    for det in top_dets:
        arrays[det] = time_df[det].to_numpy() if det in time_df.columns else np.zeros(n)

    other_dets = [d for d in all_dets_sorted if d not in top_dets]
    extra_dets = [d for d in time_df.columns if d not in top_dets and d not in all_dets_sorted]
    if other_dets or extra_dets:
        other_arr = np.zeros(n)
        for det in other_dets + extra_dets:
            if det in time_df.columns:
                other_arr = other_arr + time_df[det].to_numpy()
        arrays["Other"] = other_arr
    return arrays


# ---------------------------------------------------------------------------
# plot_region_timing
# ---------------------------------------------------------------------------


def plot_region_timing(
    source: dict[str, dict] | str | Path | list[str | Path],
    *,
    labels: list[str] | None = None,
    show: str = "both",
    attribution: str = "at_location",
    top_n: int = 8,
    figsize: tuple[float, float] | None = None,
    exclude_events: list[int] | None = None,
) -> plt.Figure:
    """Plot per-detector timing breakdown and/or per-event sequence for one or more runs.

    Single run: a donut chart + sorted horizontal bar chart (breakdown panel) and/or
    a stacked-area sequence chart.  Multiple runs: a grouped horizontal bar chart
    (breakdown) and/or a vertical stack of per-run stacked-area sequence charts.

    Parameters
    ----------
    source : dict[str, dict], str/Path, or list of str/Path
        Pre-loaded dict from :func:`~dd4bench.analysis.loader.load_region_timing`,
        a single log-dir path, or a list of log-dir paths.  When a list is given,
        run labels are prefixed with the directory name.
    labels : list[str] or None
        Restrict to these run labels.  Loads all runs when ``None``.
    show : {"both", "breakdown", "sequence"}
        ``"both"`` (default): breakdown + stacked-area sequence.
        ``"breakdown"``: detector time breakdown panel only.
        ``"sequence"``: per-event stacked area chart only.
    attribution : {"at_location", "by_birth"}
        Which attribution to use.  ``"at_location"`` (default) charges time to
        the detector where the Geant4 step physically occurred.  ``"by_birth"``
        charges time to the detector where the primary track was created.
    top_n : int
        Show the top *n* detectors by mean time; remaining detectors are grouped
        into ``"Other"``.  Default: 8.
    figsize : (width, height) or None
        Figure size in inches.
    exclude_events : list[int] or None
        Event numbers to exclude.  Defaults to ``[0]`` (first event is typically
        an initialisation outlier).  Pass ``[]`` to disable.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if show not in ("both", "breakdown", "sequence"):
        raise ValueError(f"show must be 'both', 'breakdown', or 'sequence', got {show!r}")
    if attribution not in ("at_location", "by_birth"):
        raise ValueError(f"attribution must be 'at_location' or 'by_birth', got {attribution!r}")

    _use_style()
    det_title = _detector_title(source)

    if exclude_events is None:
        exclude_events = _DEFAULT_EXCLUDE_EVENTS

    region_data = _ensure_region_data(source, labels=labels)
    if not region_data:
        raise ValueError(f"No region data found for labels={labels}.")

    label_list = list(region_data.keys())
    n = len(label_list)

    # -----------------------------------------------------------------------
    # Filter events and align timing / event DataFrames
    # -----------------------------------------------------------------------
    filtered: dict[str, dict] = {}
    for lbl, data in region_data.items():
        ev_df = data["events"]
        mask  = ~ev_df["event_number"].isin(exclude_events)
        ev_filt = ev_df[mask].copy().reset_index(drop=True)
        if ev_filt.empty:
            raise ValueError(f"No events left after applying exclude_events for '{lbl}'.")
        time_df  = data[attribution]
        time_filt = (
            time_df.loc[time_df.index.isin(ev_filt["event_number"])]
            .reindex(ev_filt["event_number"].values)
            .fillna(0.0)
        )
        filtered[lbl] = {"events": ev_filt, "time": time_filt}

    # -----------------------------------------------------------------------
    # Determine top-N detectors (ordered by first run; other runs folded correctly
    # via _build_stacked_arrays's extra_dets logic)
    # -----------------------------------------------------------------------
    top_dets, all_dets_sorted = _region_top_n(filtered[label_list[0]]["time"], top_n)

    det_display = top_dets + (["Other"] if len(all_dets_sorted) > top_n else [])
    det_colors: dict[str, str] = {
        det: _PALETTE[i % len(_PALETTE)] for i, det in enumerate(top_dets)
    }
    if "Other" in det_display:
        det_colors["Other"] = _OTHER_COLOR
    det_colors["Unaccounted"] = _UNACCOUNTED_COLOR

    attr_str      = "at location" if attribution == "at_location" else "by birth"
    show_breakdown = show in ("both", "breakdown")
    show_seq       = show in ("both", "sequence")

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    ax_donut: plt.Axes | None = None
    ax_bar:   plt.Axes | None = None
    ax_seq:   plt.Axes | None = None
    ax_seq_list: list[plt.Axes] = []

    if n == 1:
        if show == "both":
            fw, fh = figsize or (13.0, 8.5)
            fig = plt.figure(figsize=(fw, fh))
            gs = GridSpec(2, 2, figure=fig, height_ratios=[5, 3.5], hspace=0.4, wspace=0.3)
            ax_donut = fig.add_subplot(gs[0, 0])
            ax_bar   = fig.add_subplot(gs[0, 1])
            ax_seq   = fig.add_subplot(gs[1, :])
        elif show == "breakdown":
            fw, fh = figsize or (11.0, 4.5)
            fig, axes = plt.subplots(1, 2, figsize=(fw, fh))
            ax_donut, ax_bar = axes[0], axes[1]
        else:
            fw, fh = figsize or (10.0, 4.5)
            fig, ax_seq = plt.subplots(figsize=(fw, fh))
    else:
        bar_h = max(4.5, 0.5 * (len(det_display) + 2) + 1.5)
        if show == "both":
            seq_h = 3.5
            fw, fh = figsize or (11.0, bar_h + seq_h * n)
            fig = plt.figure(figsize=(fw, fh))
            gs = GridSpec(1 + n, 1, figure=fig,
                          height_ratios=[bar_h] + [seq_h] * n, hspace=0.45)
            ax_bar      = fig.add_subplot(gs[0, 0])
            ax_seq_list = [fig.add_subplot(gs[1 + i, 0]) for i in range(n)]
        elif show == "breakdown":
            fw, fh = figsize or (10.0, bar_h)
            fig, ax_bar = plt.subplots(figsize=(fw, fh))
        else:
            seq_h = 3.5
            fw, fh = figsize or (10.0, seq_h * n)
            fig, axes_arr = plt.subplots(n, 1, figsize=(fw, fh), sharex=True)
            ax_seq_list = list(np.atleast_1d(axes_arr))

    # -----------------------------------------------------------------------
    # Breakdown panel
    # -----------------------------------------------------------------------
    if show_breakdown:
        lbl0        = label_list[0]
        time_df0    = filtered[lbl0]["time"]
        ev_df0      = filtered[lbl0]["events"]
        stacked0    = _build_stacked_arrays(time_df0, top_dets, all_dets_sorted)
        means0      = {det: float(arr.mean()) for det, arr in stacked0.items()}
        sems0       = {
            det: (float(arr.std(ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)
            for det, arr in stacked0.items()
        }
        mean_unacc  = float(ev_df0["event_unaccounted_s"].mean())
        unacc_arr   = ev_df0["event_unaccounted_s"].to_numpy()
        sem_unacc   = (float(unacc_arr.std(ddof=1)) / np.sqrt(len(unacc_arr))) if len(unacc_arr) > 1 else 0.0
        total_wall0 = float(ev_df0["event_wall_s"].mean())

        if n == 1 and ax_donut is not None:
            donut_cats   = det_display + ["Unaccounted"]
            donut_vals   = [means0.get(d, 0.0) for d in det_display]
            donut_vals.append(max(0.0, mean_unacc))
            donut_clrs   = [det_colors[c] for c in donut_cats]

            valid_idx = [i for i, v in enumerate(donut_vals) if v > 0]
            vv = [donut_vals[i] for i in valid_idx]
            vc = [donut_clrs[i] for i in valid_idx]

            if vv:
                wedges, _, autotexts = ax_donut.pie(
                    vv,
                    colors=vc,
                    autopct=lambda pct: f"{pct:.1f}%" if pct >= 3.0 else "",
                    pctdistance=0.75,
                    wedgeprops=dict(width=0.5, edgecolor="white", linewidth=1.5),
                    startangle=90,
                    counterclock=False,
                )
                for at in autotexts:
                    at.set_fontsize(8)
                ax_donut.text(
                    0, 0, f"μ = {total_wall0:.3g} s\nper event",
                    ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="#333333",
                )
            else:
                ax_donut.text(0.5, 0.5, "No timing data",
                              ha="center", va="center", transform=ax_donut.transAxes)
            ax_donut.set_title(f"Time Breakdown ({attr_str})", pad=10)

        if ax_bar is not None:
            if n == 1:
                bar_cats  = det_display + ["Unaccounted"]
                bar_means = [means0.get(d, 0.0) for d in det_display] + [max(0.0, mean_unacc)]
                bar_stds  = [sems0.get(d, 0.0) for d in det_display] + [sem_unacc]
                bar_clrs  = [det_colors.get(d, _UNACCOUNTED_COLOR) for d in bar_cats]

                n_det_cats = len(det_display)
                order = sorted(range(n_det_cats), key=lambda i: bar_means[i], reverse=True)
                order.append(n_det_cats)  # Unaccounted at bottom

                s_cats  = [bar_cats[i]  for i in order]
                s_means = [bar_means[i] for i in order]
                s_stds  = [bar_stds[i]  for i in order]
                s_clrs  = [bar_clrs[i]  for i in order]

                y_pos = list(range(len(s_cats)))
                ax_bar.barh(
                    y_pos, s_means, xerr=s_stds,
                    color=s_clrs, edgecolor="white", linewidth=0.5, height=0.65,
                    error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor="#555555"),
                )
                ax_bar.set_yticks(y_pos)
                ax_bar.set_yticklabels(s_cats, fontsize=8.5)
                x_max = max(s_means) if s_means else 1.0
                # xlim must clear the error caps, not just the bar ends
                x_right = max((v + s for v, s in zip(s_means, s_stds)), default=x_max)
                for i, (val, std_v) in enumerate(zip(s_means, s_stds)):
                    ax_bar.text(
                        val + std_v + 0.02 * x_right, i,
                        f"{val:.3g} s", va="center", ha="left", fontsize=7.5, color="#444444",
                    )
                ax_bar.set_xlim(0, x_right * 1.35)
                ax_bar.set_xlabel("Mean time per event (s)")
                ax_bar.set_title(f"Mean time per detector ({attr_str})", pad=6)
                ax_bar.grid(False)
                _apply_tick_style(ax_bar)

            else:
                all_run_means: dict[str, dict[str, float]] = {}
                all_run_unacc: dict[str, float] = {}
                for lbl in label_list:
                    td = filtered[lbl]["time"]
                    ed = filtered[lbl]["events"]
                    st = _build_stacked_arrays(td, top_dets, all_dets_sorted)
                    all_run_means[lbl] = {det: float(arr.mean()) for det, arr in st.items()}
                    all_run_unacc[lbl] = max(0.0, float(ed["event_unaccounted_s"].mean()))

                run_palette  = {lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(label_list)}
                all_bar_dets = det_display + ["Unaccounted"]
                n_det_rows   = len(all_bar_dets)
                bar_h_each   = 0.7 / n
                base_pos     = np.arange(n_det_rows, dtype=float)

                for run_i, lbl in enumerate(label_list):
                    run_vals = [all_run_means[lbl].get(d, 0.0) for d in det_display]
                    run_vals.append(all_run_unacc[lbl])
                    offsets = base_pos - 0.35 + (run_i + 0.5) * bar_h_each
                    ax_bar.barh(
                        offsets, run_vals, height=bar_h_each * 0.85,
                        color=run_palette[lbl], alpha=0.85,
                        edgecolor="white", linewidth=0.3, label=lbl,
                    )

                ax_bar.set_yticks(base_pos)
                ax_bar.set_yticklabels(all_bar_dets, fontsize=8.5)
                ax_bar.set_xlabel("Mean time per event (s)")
                ax_bar.set_title(f"Detector time breakdown ({attr_str}) — multi-run", pad=6)
                ax_bar.legend(loc="lower right", fontsize="small")
                ax_bar.grid(False)
                _apply_tick_style(ax_bar)

    # -----------------------------------------------------------------------
    # Sequence panel(s)
    # -----------------------------------------------------------------------
    if show_seq:
        seq_axes = [ax_seq] if n == 1 else ax_seq_list

        for run_i, (lbl, ax_s) in enumerate(zip(label_list, seq_axes)):
            time_df = filtered[lbl]["time"]
            ev_df   = filtered[lbl]["events"]

            event_nums = time_df.index.to_numpy()
            ev_idx     = ev_df.set_index("event_number").reindex(event_nums)
            wall_times = ev_idx["event_wall_s"].to_numpy()
            unaccounted = np.maximum(0.0, ev_idx["event_unaccounted_s"].to_numpy())

            stacked = _build_stacked_arrays(time_df, top_dets, all_dets_sorted)

            # Largest detector at the bottom of the stack (plotted first)
            stack_order = sorted(stacked, key=lambda d: stacked[d].mean(), reverse=True)
            stack_ys    = [stacked[d] for d in stack_order]
            stack_clrs  = [det_colors.get(d, _OTHER_COLOR) for d in stack_order]

            stack_ys.append(unaccounted)
            stack_clrs.append(_UNACCOUNTED_COLOR)
            stack_labels = list(stack_order) + ["Unaccounted"]

            ax_s.stackplot(
                event_nums, *stack_ys,
                colors=stack_clrs, labels=stack_labels, alpha=0.85,
            )
            ax_s.plot(
                event_nums, wall_times,
                color="black", linewidth=0.9, alpha=0.6, linestyle="--", label="Wall time",
            )
            ax_s.set_xlabel("Event number")
            ax_s.set_ylabel("Time (s)")
            ax_s.set_title(
                f"Per-Event Detector Time ({attr_str})" if n == 1 else lbl
            )
            ax_s.grid(False)
            _apply_tick_style(ax_s)

            # Reverse legend so it reads top-of-stack → bottom (Wall time first)
            handles, leg_labels = ax_s.get_legend_handles_labels()
            ax_s.legend(
                handles[::-1], leg_labels[::-1],
                loc="upper right", fontsize=7.5, ncol=2, framealpha=0.85,
            )

    # -----------------------------------------------------------------------
    # Super-title
    # -----------------------------------------------------------------------
    base_title = "Per-Region Timing"
    if n > 1:
        base_title += " — Multi-Run Comparison"
    suptitle = f"{base_title} ({det_title})" if det_title else base_title
    fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig

