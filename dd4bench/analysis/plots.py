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

from dd4bench.analysis.loader import load_event_timing, load_results

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


def _ensure_df(results: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Accept either a DataFrame or a CSV path and return a DataFrame."""
    if isinstance(results, pd.DataFrame):
        return results
    return load_results(results)


def _ensure_event_data(
    source: dict[str, pd.DataFrame] | str | Path,
    labels: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Accept either a pre-loaded dict or a log-dir path and return event data."""
    if isinstance(source, dict):
        if labels is not None:
            return {k: v for k, v in source.items() if k in labels}
        return source
    return load_event_timing(source, labels=labels)


# ---------------------------------------------------------------------------
# plot_sweep
# ---------------------------------------------------------------------------


def plot_sweep(
    results: pd.DataFrame | str | Path,
    *,
    baseline_label: str = "baseline_all",
    top_n: int | None = None,
) -> plt.Figure:
    """Plot a FULL-sweep result: wall-time and RSS delta vs baseline.

    Each bar shows the computational cost of a subdetector — i.e. how much
    wall time and peak RSS are saved when that detector is removed.  Bars
    are sorted so the most expensive detector appears at the top.

    Parameters
    ----------
    results : pd.DataFrame or str or Path
        Results DataFrame or path to a CSV file written by ``dd4bench``.
    baseline_label : str
        Label of the baseline run (default: ``"baseline_all"``).
    top_n : int or None
        If set, show only the *top_n* most expensive detectors (by wall time).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _use_style()
    results = _ensure_df(results)

    baseline = results[results["label"] == baseline_label]
    if baseline.empty:
        raise ValueError(f"Baseline label '{baseline_label}' not found in results.")

    base_wall = float(baseline["wall_time_s"].iloc[0])
    base_rss  = float(baseline["peak_rss_mb"].iloc[0])

    runs = results[results["label"] != baseline_label].copy()

    missing = runs["wall_time_s"].isna() | runs["peak_rss_mb"].isna()
    if missing.any():
        warnings.warn(
            f"Dropping {missing.sum()} run(s) with missing wall_time_s or peak_rss_mb.",
            stacklevel=2,
        )
        runs = runs[~missing]

    runs["wall_delta_s"] = base_wall - runs["wall_time_s"]
    runs["rss_delta_mb"] = base_rss  - runs["peak_rss_mb"]

    # Sort ascending so barh places most expensive at the top
    runs = runs.sort_values("wall_delta_s", ascending=True)

    if top_n is not None:
        runs = runs.tail(top_n)

    names = runs["label"].str.replace(r"^without_", "", regex=True).tolist()
    n = len(names)

    fig, (ax_wall, ax_rss) = plt.subplots(
        1, 2,
        figsize=(14, max(4.0, 0.45 * n + 2.5)),
        sharey=True,
    )

    colors_w = [_BLUE if v >= 0 else _RED for v in runs["wall_delta_s"]]
    ax_wall.barh(names, runs["wall_delta_s"], color=colors_w, edgecolor="white", linewidth=0.5)
    ax_wall.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_wall.set_xlabel("Wall time saved vs baseline (s)")
    ax_wall.set_title(f"Wall Time Impact\n(baseline = {base_wall:.1f} s)")
    ax_wall.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    colors_r = [_BLUE if v >= 0 else _RED for v in runs["rss_delta_mb"]]
    ax_rss.barh(names, runs["rss_delta_mb"], color=colors_r, edgecolor="white", linewidth=0.5)
    ax_rss.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_rss.set_xlabel("Peak RSS saved vs baseline (MB)")
    ax_rss.set_title(f"Memory Impact\n(baseline = {base_rss:.0f} MB)")
    ax_rss.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    fig.suptitle(
        "Detector Sweep — Computational Cost per Subdetector",
        fontweight="bold",
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# plot_run_overview
# ---------------------------------------------------------------------------

_OVERVIEW_METRICS = [
    ("wall_time_s",    "Wall Time (s)"),
    ("peak_rss_mb",    "Peak RSS (MB)"),
    ("user_cpu_s",     "User CPU (s)"),
    ("events_per_sec", "Throughput (ev/s)"),
]


def plot_run_overview(
    results: pd.DataFrame | str | Path,
    *,
    metrics: list[tuple[str, str]] | None = None,
    baseline_label: str | None = "baseline_all",
) -> plt.Figure:
    """Plot absolute run metrics for all runs in a 2 × 2 panel grid.

    Useful as a first look at a sweep or comparison result — shows wall
    time, memory, CPU, and throughput side by side.  The baseline run
    (if present) is highlighted in a distinct colour.

    Parameters
    ----------
    results : pd.DataFrame or str or Path
        Results DataFrame or path to a CSV file written by ``dd4bench``.
    metrics : list of (column, axis-label) pairs or None
        Which metrics to plot and how to label them.  Defaults to
        wall_time_s, peak_rss_mb, user_cpu_s, events_per_sec.
    baseline_label : str or None
        Label of the baseline run; highlighted in a different colour.
        Pass ``None`` to disable highlighting.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _use_style()
    results = _ensure_df(results)

    if metrics is None:
        metrics = _OVERVIEW_METRICS

    # Drop rows where all selected metrics are NaN
    metric_cols = [col for col, _ in metrics if col in results.columns]
    df = results.dropna(subset=metric_cols, how="all").copy()

    # Sort by wall_time_s descending (slowest at top for barh) if available
    if "wall_time_s" in df.columns:
        df = df.sort_values("wall_time_s", ascending=True)

    labels = df["label"].tolist()
    n = len(labels)

    n_metrics = len(metrics)
    ncols = 2
    nrows = (n_metrics + 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, max(4.0, 0.45 * n + 2.5) * nrows / 2),
        sharey=True,
    )
    axes_flat = axes.flatten() if n_metrics > 1 else [axes]

    for ax, (col, ylabel) in zip(axes_flat, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        values = df[col].tolist()
        colors = [
            "#ff7f0e" if lbl == baseline_label else _BLUE
            for lbl in labels
        ]

        ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(ylabel)
        ax.set_title(ylabel.split(" (")[0])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Hide any unused axes
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    if baseline_label is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#ff7f0e", label=f"Baseline ({baseline_label})"),
            Patch(facecolor=_BLUE,     label="Sweep runs"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

    fig.suptitle("Run Metrics Overview", fontweight="bold")
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
        core = data[modified_z <= threshold]
    else:
        core = data  # all values identical — no outliers to remove

    if len(core) == 0:
        core = data

    x_min, x_max = float(core.min()), float(core.max())
    margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.01 * abs(x_max)
    x_min = max(0.0, x_min - margin)
    x_max = x_max + margin

    n_clipped = int(np.sum((data < x_min) | (data > x_max)))
    return (x_min, x_max), n_clipped


# ---------------------------------------------------------------------------
# plot_event_timing
# ---------------------------------------------------------------------------


def plot_event_timing(
    log_dir: str | Path,
    *,
    labels: list[str] | None = None,
    bins: int | str = "auto",
    alpha: float = 0.8,
    figsize: tuple[float, float] | None = None,
    outlier_threshold: float = 3.5,
    exclude_events: list[int] | None = None,
) -> plt.Figure:
    """Plot per-event timing distributions loaded from a log directory.

    Left panel: violin plots (multiple runs) or histogram (single run)
    of per-event wall-time distributions.

    Right panel: event wall time vs event number — useful for detecting
    warm-up effects or outlier events.  Excluded events are shown as open
    red markers so their timing is still visible for reference.

    For the single-run histogram the x-range is automatically focused on
    the core of the distribution using a MAD-based modified Z-score.  A
    note is added to the plot whenever events are clipped, and a Python
    warning is raised if more than 5 % are excluded or an outlier is
    extremely far from the core.

    Parameters
    ----------
    log_dir : str or Path
        Directory containing ``*_events.json`` files written by the
        DD4benchTimingAction plugin.
    labels : list[str] or None
        If set, only load data for these run labels.
    bins : int or str
        Bin specification for the single-run histogram.  Any value
        accepted by :func:`matplotlib.pyplot.hist` is valid; ``"auto"``
        (the default) lets NumPy choose the number of bins via the
        Sturges / Freedman-Diaconis estimator.
    alpha : float
        Opacity of the histogram bars (default: 0.8).
    figsize : (width, height) or None
        Figure size in inches.  Defaults to ``(12, 4.5)``.
    outlier_threshold : float
        Modified Z-score threshold for the automatic range clipping
        (default: 3.5).  Increase to keep more of the tails; decrease
        to clip more aggressively.
    exclude_events : list[int] or None
        Event numbers to exclude from statistics and histograms.
        Defaults to ``[0]`` (the first event typically includes geometry
        initialisation overhead).  Pass ``[]`` to disable exclusion.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _use_style()

    if exclude_events is None:
        exclude_events = [0]

    event_data = load_event_timing(log_dir, labels=labels)
    if not event_data:
        raise ValueError(f"No *_events.json files found in '{log_dir}'.")

    label_list = list(event_data.keys())
    n = len(label_list)

    if figsize is None:
        figsize = (12, 4.5)

    fig, (ax_dist, ax_seq) = plt.subplots(1, 2, figsize=figsize)

    # Filter excluded events for stats/histograms; keep full data for sequential plot.
    filtered_data = {
        lbl: df[~df["event_number"].isin(exclude_events)]
        for lbl, df in event_data.items()
    }

    arrays = [filtered_data[lbl]["event_time_s"].to_numpy() for lbl in label_list]

    if n == 1:
        data = arrays[0]
        core_range, n_clipped = _compute_core_range(data, threshold=outlier_threshold)
        n_data = len(data)
        mean    = float(data.mean())
        std     = float(data.std(ddof=1))
        sem     = std / np.sqrt(n_data)
        se_std  = std / np.sqrt(2 * (n_data - 1))

        ax_dist.hist(
            data, bins=bins, range=core_range,
            color=_BLUE, edgecolor="none", alpha=alpha,
        )
        ax_dist.axvline(mean, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
        ax_dist.set_xlim(core_range)
        ax_dist.set_ylabel("Count")
        ax_dist.set_title(f"Event Time Distribution ({label_list[0]})")
        ax_dist.grid(False)

        # μ ± SEM and σ ± SE(σ) box: top-left, inside plot area
        ax_dist.text(
            0.03, 0.97,
            f"$\\mu = {mean:.4g} \\pm {sem:.2g}$ s\n$\\sigma = {std:.4g} \\pm {se_std:.2g}$ s",
            transform=ax_dist.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7),
        )

        # Clipping note below the x-axis label — never overlaps data or mean line.
        # Extra blank line adds visual separation from the axis label.
        xlabel = "Event time (s)"
        if n_clipped > 0:
            frac = n_clipped / len(data)
            note = f"{n_clipped} event(s) ({frac:.1%}) outside plotted range"
            xlabel += f"\n\n{note}"
            extreme = data.max() > _OUTLIER_EXTREME_RATIO * core_range[1]
            if frac > _OUTLIER_FRACTION_WARN or extreme:
                warnings.warn(
                    f"plot_event_timing: {note}. "
                    f"Max value: {data.max():.4g} s, core upper bound: {core_range[1]:.4g} s. "
                    "Check for simulation anomalies or adjust outlier_threshold.",
                    stacklevel=2,
                )
        ax_dist.set_xlabel(xlabel)

    else:
        parts = ax_dist.violinplot(
            arrays,
            positions=range(n),
            showmedians=True,
            showextrema=True,
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(_PALETTE[i % len(_PALETTE)])
            body.set_alpha(alpha)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax_dist.set_xticks(range(n))
        ax_dist.set_xticklabels(label_list, rotation=30, ha="right")
        ax_dist.set_xlabel("Event time (s)")
        ax_dist.set_ylabel("Count")
        ax_dist.set_title("Event Time Distribution")
        ax_dist.grid(False)

    _apply_tick_style(ax_dist)

    for i, lbl in enumerate(label_list):
        df = filtered_data[lbl]
        color = _PALETTE[i % len(_PALETTE)]
        ax_seq.plot(
            df["event_number"], df["event_time_s"],
            color=color, alpha=0.7, linewidth=1.0, label=lbl,
        )

    ax_seq.set_xlabel("Event number")
    ax_seq.set_ylabel("Event time (s)")
    ax_seq.set_title("Event Time vs Event Number")
    ax_seq.grid(False)
    _apply_tick_style(ax_seq)
    if n > 1:
        ax_seq.legend(loc="upper right", fontsize="small")

    if exclude_events:
        evts_str = ", ".join(str(e) for e in sorted(exclude_events))
        warnings.warn(
            f"plot_event_timing: event(s) {evts_str} excluded from statistics and histograms "
            "(geometry initialisation overhead). Pass exclude_events=[] to disable.",
            UserWarning, stacklevel=2,
        )

    fig.suptitle("Per-Event Timing", fontweight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# plot_event_timing_overlay
# ---------------------------------------------------------------------------


def plot_event_timing_overlay(
    source: dict[str, pd.DataFrame] | str | Path,
    *,
    labels: list[str] | None = None,
    baseline_label: str | None = None,
    bins: int | str = "auto",
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    outlier_threshold: float = 3.5,
    exclude_events: list[int] | None = None,
) -> plt.Figure:
    """Overlay per-event timing histograms for multiple runs.

    All distributions share an x-axis range derived from the combined data
    via MAD-based outlier clipping.  Each run gets its own colour; a dashed
    vertical line marks its mean.  A stats table below the panels shows
    μ ± SEM, σ, and Δμ ± δ(Δμ) relative to the baseline run.

    Parameters
    ----------
    source : dict[str, pd.DataFrame] or str or Path
        Either a pre-loaded dict from :func:`~dd4bench.analysis.loader.load_event_timing`
        or a path to a log directory containing ``*_events.json`` files.
    labels : list[str] or None
        If set, restrict to these run labels.
    baseline_label : str or None
        Label to use as the reference for Δμ.  Defaults to the first run.
    bins : int or str
        Bin specification passed to :func:`matplotlib.pyplot.hist`.
        ``"auto"`` (default) lets NumPy choose via Sturges / FD.
    alpha : float
        Opacity of each histogram (default: 0.5).
    figsize : (width, height) or None
        Size of the *plot area* in inches (default: ``(12, 4.5)``).
        The figure grows taller automatically to fit the stats table.
    outlier_threshold : float
        Modified Z-score threshold for the shared x-range (default: 3.5).
    exclude_events : list[int] or None
        Event numbers to exclude from statistics and histograms.
        Defaults to ``[0]`` (the first event typically includes geometry
        initialisation overhead).  Pass ``[]`` to disable exclusion.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _use_style()

    if exclude_events is None:
        exclude_events = [0]

    # Load all available runs first so we can show them in error messages.
    all_event_data = _ensure_event_data(source)
    available = sorted(all_event_data.keys())

    event_data = _ensure_event_data(source, labels=labels)
    if not event_data:
        raise ValueError(
            f"No event timing data found for labels={labels}.\n"
            f"Available labels in '{source}': {available}"
        )
    if len(event_data) < 2:
        raise ValueError(
            "plot_event_timing_overlay requires at least 2 runs. "
            "Use plot_event_timing for a single run.\n"
            f"Available labels: {available}"
        )

    label_list = list(event_data.keys())
    n_runs     = len(label_list)

    # Filter excluded events for stats/histograms; keep full data for sequential plot.
    filtered_data = {
        lbl: df[~df["event_number"].isin(exclude_events)]
        for lbl, df in event_data.items()
    }

    arrays   = {lbl: filtered_data[lbl]["event_time_s"].to_numpy() for lbl in label_list}
    all_data = np.concatenate(list(arrays.values()))
    core_range, _ = _compute_core_range(all_data, threshold=outlier_threshold)

    ref_label = baseline_label if baseline_label is not None else label_list[0]
    if ref_label not in arrays:
        raise ValueError(
            f"baseline_label '{ref_label}' not found in the loaded runs.\n"
            f"Available labels: {label_list}"
        )
    ref_data = arrays[ref_label]
    ref_mean = float(ref_data.mean())
    ref_sem  = float(ref_data.std(ddof=1) / np.sqrt(len(ref_data)))

    # Compute shared bin edges so every panel uses identical binning.
    clipped_all = all_data[(all_data >= core_range[0]) & (all_data <= core_range[1])]
    _, common_edges = np.histogram(
        clipped_all, bins=bins if not isinstance(bins, str) else "auto"
    )

    # ---------------------------------------------------------------------------
    # Layout
    # Two-run case: nested GridSpec keeps the plot→ratio gap tight (hspace=0.05)
    # while tight_layout adds generous space between the ratio xlabel and table.
    # N-run case: flat 2-row GridSpec + extra figure height for the xlabel note.
    # ---------------------------------------------------------------------------
    if figsize is None:
        figsize = (12, 4.5)
    per_row_h = 0.38
    table_h   = (n_runs + 1.5) * per_row_h
    two_run   = (n_runs == 2)
    ratio_h   = 1.6 if two_run else 0.0
    note_gap  = 1.0   # extra inches for tight_layout to breathe below xlabel
    total_h   = figsize[1] + ratio_h + table_h + note_gap

    fig = plt.figure(figsize=(figsize[0], total_h))
    if two_run:
        # Outer: [plot+ratio block] | [table]
        gs_outer = GridSpec(
            2, 1, figure=fig,
            height_ratios=[figsize[1] + ratio_h, table_h],
        )
        # Inner: [main panels] | [ratio panels] — tight gap between them
        gs_top = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_outer[0],
            height_ratios=[figsize[1], ratio_h],
            hspace=0.05,
        )
        ax_dist  = fig.add_subplot(gs_top[0, 0])
        ax_seq   = fig.add_subplot(gs_top[0, 1])
        ax_rdist = fig.add_subplot(gs_top[1, 0], sharex=ax_dist)
        ax_rseq  = fig.add_subplot(gs_top[1, 1])
        ax_table = fig.add_subplot(gs_outer[1])
    else:
        gs = GridSpec(
            2, 2, figure=fig,
            height_ratios=[figsize[1], table_h],
        )
        ax_dist  = fig.add_subplot(gs[0, 0])
        ax_seq   = fig.add_subplot(gs[0, 1])
        ax_table = fig.add_subplot(gs[1, :])
        ax_rdist = ax_rseq = None
    ax_table.axis("off")

    total_clipped = 0
    table_rows    = []
    hist_counts   = {}

    for i, lbl in enumerate(label_list):
        data   = arrays[lbl]
        color  = _PALETTE[i % len(_PALETTE)]
        n_data = len(data)
        mean   = float(data.mean())
        std    = float(data.std(ddof=1))
        sem    = std / np.sqrt(n_data)

        total_clipped += int(np.sum((data < core_range[0]) | (data > core_range[1])))

        # Δμ with standard error propagation: δ(Δμ) = 100/μ_ref · √(SEM_i² + (μ_i/μ_ref)² SEM_ref²)
        if lbl == ref_label:
            delta_cell = "ref."
        else:
            delta_pct = (mean - ref_mean) / ref_mean * 100
            delta_err = (100.0 / ref_mean) * np.sqrt(
                sem**2 + (mean / ref_mean)**2 * ref_sem**2
            )
            sign       = "+" if delta_pct >= 0 else ""
            delta_cell = f"{sign}{delta_pct:.1f} ± {delta_err:.1f}%"

        table_rows.append([
            lbl,
            f"{mean:.4g} ± {sem:.2g}",
            f"{std:.4g}",
            delta_cell,
        ])

        counts, _ = np.histogram(data, bins=common_edges)
        hist_counts[lbl] = counts.astype(float)

        ax_dist.hist(
            data, bins=common_edges, range=core_range,
            color=color, edgecolor="none", alpha=alpha,
        )
        ax_dist.axvline(mean, color=color, linestyle="--", linewidth=1.2, alpha=0.7)

        ax_seq.plot(
            filtered_data[lbl]["event_number"],
            filtered_data[lbl]["event_time_s"],
            color=color, alpha=0.7, linewidth=1.0,
        )

    # Clipping note and warning
    clipping_note = ""
    if total_clipped > 0:
        frac = total_clipped / len(all_data)
        clipping_note = f"{total_clipped} event(s) ({frac:.1%}) outside plotted range"
        extreme = all_data.max() > _OUTLIER_EXTREME_RATIO * core_range[1]
        if frac > _OUTLIER_FRACTION_WARN or extreme:
            warnings.warn(
                f"plot_event_timing_overlay: {clipping_note}. "
                f"Max value: {all_data.max():.4g} s, core upper bound: {core_range[1]:.4g} s. "
                "Check for simulation anomalies or adjust outlier_threshold.",
                stacklevel=2,
            )

    ax_dist.set_xlim(core_range)
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Event Time Distribution")
    ax_dist.grid(False)
    _apply_tick_style(ax_dist)

    ax_seq.set_xlabel("Event number")
    ax_seq.set_ylabel("Event time (s)")
    ax_seq.set_title("Event Time vs Event Number")
    ax_seq.grid(False)
    _apply_tick_style(ax_seq)

    if exclude_events:
        evts_str = ", ".join(str(e) for e in sorted(exclude_events))
        warnings.warn(
            f"plot_event_timing_overlay: event(s) {evts_str} excluded from statistics and histograms "
            "(geometry initialisation overhead). Pass exclude_events=[] to disable.",
            UserWarning, stacklevel=2,
        )

    if two_run:
        # Hide x-axis labels on main panels — only ratio panels carry them.
        plt.setp(ax_dist.get_xticklabels(), visible=False)
        plt.setp(ax_seq.get_xticklabels(), visible=False)
        ax_seq.set_xlabel("")

        other_label = next(l for l in label_list if l != ref_label)
        other_idx   = label_list.index(other_label)
        other_color = _PALETTE[other_idx % len(_PALETTE)]

        # ---- Left ratio panel: histogram count ratio with Poisson errors -----
        ref_counts_arr   = hist_counts[ref_label]
        other_counts_arr = hist_counts[other_label]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                (ref_counts_arr > 0) & (other_counts_arr > 0),
                other_counts_arr / ref_counts_arr,
                np.nan,
            )
            ratio_err = np.where(
                (ref_counts_arr > 0) & (other_counts_arr > 0),
                ratio * np.sqrt(1.0 / other_counts_arr + 1.0 / ref_counts_arr),
                np.nan,
            )
        bin_centers = 0.5 * (common_edges[:-1] + common_edges[1:])
        valid = ~np.isnan(ratio)
        ax_rdist.errorbar(
            bin_centers[valid], ratio[valid],
            yerr=ratio_err[valid],
            fmt="o", color="#444444",
            markersize=4, capsize=3, linewidth=0, elinewidth=0.8, capthick=0.8,
            alpha=0.75,
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

        # ---- Right ratio panel: per-event time ratio vs event number ---------
        df_ref   = event_data[ref_label].set_index("event_number")
        df_other = event_data[other_label].set_index("event_number")
        common_evts = df_ref.index.intersection(df_other.index)
        if len(common_evts) > 0:
            t_ref   = df_ref.loc[common_evts, "event_time_s"].to_numpy()
            t_other = df_other.loc[common_evts, "event_time_s"].to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                evt_ratio = np.where(t_ref > 0, t_other / t_ref, np.nan)
            ax_rseq.scatter(
                common_evts, evt_ratio,
                color="#444444", s=12, alpha=0.5, linewidths=0,
            )
        ax_rseq.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax_rseq.set_xlabel("Event number")
        ax_rseq.set_ylabel("Ratio (other / ref.)", fontsize=8)
        ax_rseq.grid(False)
        _apply_tick_style(ax_rseq)
    else:
        xlabel = "Event time (s)"
        if clipping_note:
            xlabel += f"\n\n{clipping_note}"
        ax_dist.set_xlabel(xlabel)

    # -----------------------------------------------------------------------
    # Stats table — columns are naturally aligned, values easy to compare
    # -----------------------------------------------------------------------
    n_cols      = 4
    col_headers = ["Run", "μ ± SEM (s)", "σ (s)", f"Δμ ± δ(Δμ)  [ref: {ref_label}]"]

    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=col_headers,
        bbox=[0, 0, 1, 1],
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(n_cols)))

    # Bold grey header row
    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor("#d4d4d4")
        cell.set_text_props(fontweight="bold")

    # Light colour tint per data row — matches each run's histogram colour
    for i in range(n_runs):
        r, g, b = mcolors.to_rgb(_PALETTE[i % len(_PALETTE)])
        tint = (r, g, b, 0.12)
        for j in range(n_cols):
            tbl[(i + 1, j)].set_facecolor(tint)

    fig.suptitle("Per-Event Timing Comparison", fontweight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# plot_compare
# ---------------------------------------------------------------------------


def plot_compare(
    results: pd.DataFrame | str | Path,
    *,
    label_a: str = "geometry_a",
    label_b: str = "geometry_b",
    metrics: list[str] | None = None,
) -> plt.Figure:
    """Plot a head-to-head geometry comparison (COMPARE mode).

    Parameters
    ----------
    results : pd.DataFrame or str or Path
        Results DataFrame or path to a CSV file written by ``dd4bench``.
    label_a : str
        Label for the first geometry (default: ``"geometry_a"``).
    label_b : str
        Label for the second geometry (default: ``"geometry_b"``).
    metrics : list[str] or None
        Column names to compare.  Defaults to
        ``["wall_time_s", "peak_rss_mb", "events_per_sec"]``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _use_style()
    results = _ensure_df(results)

    if metrics is None:
        metrics = ["wall_time_s", "peak_rss_mb", "events_per_sec"]

    row_a = results[results["label"] == label_a]
    row_b = results[results["label"] == label_b]

    if row_a.empty:
        raise ValueError(f"Label '{label_a}' not found in results.")
    if row_b.empty:
        raise ValueError(f"Label '{label_b}' not found in results.")

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        val_a = float(row_a[metric].iloc[0]) if metric in row_a.columns else None
        val_b = float(row_b[metric].iloc[0]) if metric in row_b.columns else None

        values = [val_a or 0.0, val_b or 0.0]
        bars = ax.bar(
            [label_a, label_b],
            values,
            color=[_PALETTE[0], _PALETTE[1]],
            edgecolor="white",
            linewidth=0.5,
            width=0.5,
        )

        for bar, val in zip(bars, [val_a, val_b]):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9,
                )

        unit = _METRIC_UNITS.get(metric, "")
        ax.set_ylabel(f"{metric.replace('_', ' ')} {unit}".strip())
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

    fig.suptitle(f"Geometry Comparison: {label_a} vs {label_b}", fontweight="bold")
    fig.tight_layout()
    plt.close(fig)
    return fig
