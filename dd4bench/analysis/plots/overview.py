"""plot_run_overview: 2×2 panel of run-level metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._theme import _METRIC_UNITS, _TEMPLATE
from ._utils import _default_baseline, _detector_title, _ensure_df, _matches_baseline

_LOWER_IS_BETTER = {"wall_time_s", "peak_rss_mb", "user_cpu_s"}

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
) -> go.Figure:
    """Plot run metrics for all runs in a 2 × 2 panel grid.

    Each metric is drawn as a horizontal bar chart with value annotations.

    Parameters
    ----------
    results : pd.DataFrame, str/Path, or list of str/Path
        Results DataFrame, a single log-dir path, or a list of log-dir paths
        for multi-detector comparisons.
    labels : list[str] or None
        Show only these run labels.
    metrics : list of (column, axis-label) pairs or None
        Which metrics to plot.  Defaults to wall_time_s, peak_rss_mb,
        user_cpu_s, events_per_sec.
    relative : bool
        If ``True``, normalise every metric to the baseline run (= 100 %).
    baseline_label : str or None
        Which run to treat as 100 % when ``relative=True``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
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

    # Capture label order before the wall-time sort so that baseline resolution
    # is deterministic (first loaded row) regardless of display order.
    load_order_labels = df["label"].tolist()

    baseline_vals: dict[str, float] = {}
    if relative:
        _bl = baseline_label if baseline_label is not None else _default_baseline(load_order_labels)
        bl_mask = df["label"].apply(lambda lbl: _matches_baseline(lbl, _bl))
        if not bl_mask.any():
            hint = " Pass baseline_label=... to specify the reference run." if baseline_label is None else ""
            raise ValueError(f"baseline_label '{_bl}' not found for relative=True.{hint}")
        if bl_mask.sum() > 1:
            matched = df.loc[bl_mask, "label"].tolist()
            raise ValueError(
                f"baseline_label '{_bl}' matches multiple runs: {matched}. "
                "Pass the full prefixed label to disambiguate."
            )
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


    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("metrics must contain at least one metric")

    ncols = 2
    nrows = (n_metrics + 1) // ncols

    subplot_titles = []
    for col, ylabel in metrics:
        if relative:
            subplot_titles.append(f"{ylabel.split(' (')[0]} %")
        else:
            subplot_titles.append(ylabel)

    # Keep the inter-row gap fixed at ~80 px regardless of figure height.
    fig_height = max(300, 45 * n_runs + 150) * nrows
    v_spacing = (80 / fig_height) if nrows > 1 else 0.0

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.12,
        vertical_spacing=v_spacing,
    )

    for idx, (col, _ylabel) in enumerate(metrics):
        row = idx // ncols + 1
        col_num = idx % ncols + 1

        if col not in df.columns:
            continue

        values = df[col].tolist()
        valid_v = [v for v in values if pd.notna(v)]
        x_max = max(valid_v) if valid_v else 1.0

        # Gradient: green (best) → yellow → red (worst), ranked per metric.
        col_series = df.set_index("label")[col]
        n_valid = int(col_series.notna().sum())
        ranks = col_series.rank(ascending=(col in _LOWER_IS_BETTER), na_option="keep")
        denom = max(n_valid - 1, 1)
        sample_pts = []
        for lbl in run_labels:
            if lbl not in ranks.index or pd.isna(ranks.at[lbl]):
                sample_pts.append(0.0)
            else:
                sample_pts.append(float(1.0 - (float(ranks.at[lbl]) - 1.0) / denom))
        bar_colors = pc.sample_colorscale("RdYlGn", sample_pts)

        for i, (run_label, val) in enumerate(zip(run_labels, values)):
            if relative:
                text = f"{val:.1f}%" if pd.notna(val) else ""
                hover_tmpl = "%{y}<br><b>%{x:.1f}%</b><extra></extra>"
            else:
                text = f"{val:.4g}" if pd.notna(val) else ""
                hover_tmpl = "%{y}<br><b>%{x:.4g}</b><extra></extra>"

            fig.add_trace(
                go.Bar(
                    x=[val],
                    y=[run_label],
                    orientation="h",
                    marker_color=bar_colors[i],
                    marker_line_width=0,
                    name=run_label,
                    showlegend=False,
                    text=[text],
                    textposition="outside",
                    textfont=dict(size=9, color="#444444"),
                    hovertemplate=hover_tmpl,
                ),
                row=row, col=col_num,
            )

        axis_kw = dict(range=[0, x_max * 1.22], row=row, col=col_num)
        if relative:
            axis_kw["ticksuffix"] = "%"
            fig.add_vline(
                x=100, line_dash="dash", line_color="black",
                line_width=0.8, opacity=0.4, row=row, col=col_num,
            )
        fig.update_xaxes(**axis_kw)

    title = f"Run Metrics Overview — {det_title}" if det_title else "Run Metrics Overview"
    fig.update_layout(
        title_text=title,
        title_font=dict(size=16, color="#222222"),
        template=_TEMPLATE,
        height=fig_height,
        bargap=0.35,
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=40),
    )

    return fig
