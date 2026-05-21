"""Reusable Plotly trace builders shared across plot modules."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ._theme import _BLUE, _PALETTE, _hex_to_rgba


def _histogram_traces(
    arrays: dict[str, np.ndarray],
    common_edges: np.ndarray,
    label_list: list[str],
    alpha: float,
    show_legend: bool,
) -> list[go.Bar]:
    """Return one histogram Bar trace per label using shared bin edges."""
    centers = 0.5 * (common_edges[:-1] + common_edges[1:])
    widths  = common_edges[1:] - common_edges[:-1]
    traces = []
    for i, lbl in enumerate(label_list):
        color = _BLUE if len(label_list) == 1 else _PALETTE[i % len(_PALETTE)]
        counts, _ = np.histogram(arrays[lbl], bins=common_edges)
        traces.append(go.Bar(
            x=centers,
            y=counts,
            width=widths,
            name=lbl,
            legendgroup=lbl,
            marker_color=_hex_to_rgba(color, alpha),
            marker_line_width=0,
            showlegend=show_legend,
            hovertemplate=f"<b>{lbl}</b><br>bin centre: %{{x:.4g}}<br>count: %{{y}}<extra></extra>",
        ))
    return traces


def _format_delta_cell(mean: float, sem: float, ref_mean: float, ref_sem: float) -> str:
    """Return a formatted Δμ ± δ(Δμ) string for use as a stats-table cell.

    Returns ``"—"`` for the reference run (mean == ref_mean) and
    ``"undefined"`` when ref_mean is zero to avoid division by zero.
    """
    if mean == ref_mean:
        return "—"
    if ref_mean == 0:
        return "undefined"
    delta_pct = (mean - ref_mean) / ref_mean * 100
    delta_err = (100.0 / ref_mean) * np.sqrt(sem**2 + (mean / ref_mean * ref_sem) ** 2)
    sign = "+" if delta_pct >= 0 else ""
    return f"{sign}{delta_pct:.2f}% ± {delta_err:.2f}%"


def _stats_table_trace(
    table_rows: list[list[str]],
    ref_label: str,
    unit_label: str,
    label_list: list[str],
) -> go.Table:
    """Build a Plotly Table trace from per-run stats rows."""
    headers = [
        "Run",
        f"μ ± SEM ({unit_label})",
        f"σ ({unit_label})",
        f"Δμ ± δ(Δμ)  [ref: {ref_label}]",
    ]
    n = len(table_rows)
    row_colors = [_hex_to_rgba(_PALETTE[i % len(_PALETTE)], 0.12) for i in range(n)]
    col_data = [list(col) for col in zip(*table_rows)] if table_rows else [[] for _ in headers]

    return go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#d4d4d4",
            align="center",
            font=dict(size=11, color="#222222"),
            line_color="white",
        ),
        cells=dict(
            values=col_data,
            fill_color=[row_colors],
            align="center",
            font=dict(size=10, color="#333333"),
            line_color="white",
            height=24,
        ),
    )
