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
