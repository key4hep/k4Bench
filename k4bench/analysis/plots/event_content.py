"""A per-event cost against what each event holds: one point per event
(:func:`plot_event_content`, from :func:`~k4bench.analysis.event_drivers.paired`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import _BLUE, _RED, _TEMPLATE


def plot_event_content(
    pairs: pd.DataFrame,
    *,
    x_label: str,
    y_label: str,
    line=None,
    above: pd.DataFrame | None = None,
    log_y: bool = False,
) -> go.Figure | None:
    """Each event of *pairs* (``x``, ``y``, indexed by event) as a point. With *line* (a
    :class:`~k4bench.analysis.event_drivers.Trend`), the line is drawn and
    the events of *above* ringed. ``None`` when there is nothing to draw."""
    if pairs.empty:
        return None
    events = pairs.index.to_numpy()
    marker = dict(size=8, opacity=0.8, line=dict(color="white", width=1), color=_BLUE)
    hover = (
        "Event %{customdata[0]}<br>" + f"{x_label}: " + "%{x:.4g}<br>"
        + f"{y_label}: " + "%{y:.4g}"
    )
    customdata = np.column_stack([events])
    fig = go.Figure(go.Scatter(
        x=pairs["x"], y=pairs["y"], mode="markers", marker=marker, customdata=customdata,
        hovertemplate=hover + "<extra></extra>", name="Events", showlegend=False,
    ))
    if line is not None:
        x = np.linspace(float(pairs["x"].min()), float(pairs["x"].max()), 2)
        fig.add_trace(go.Scatter(
            x=x, y=line.at(x), mode="lines", line=dict(color="#555555", width=2, dash="dash"),
            name="Trend", hoverinfo="skip", showlegend=False,
        ))
        if above is not None and not above.empty:
            fig.add_trace(go.Scatter(
                x=above["x"], y=above["y"], mode="markers",
                marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=_RED, width=2)),
                name="Above the trend", hoverinfo="skip", showlegend=False,
            ))
    fig.update_layout(
        template=_TEMPLATE, height=480, margin=dict(l=20, r=20, t=30, b=10),
    )
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label, type="log" if log_y else "linear",
                     rangemode="normal" if log_y else "tozero")
    return fig


#: Negative correlations blue, positive red, none white.
_DIVERGING = [[0.0, "#2166ac"], [0.5, "#f7f7f7"], [1.0, "#b2182b"]]


def plot_association_matrix(
    rho: pd.DataFrame,
    n: pd.DataFrame,
    *,
    row_labels: dict[str, str],
    column_labels: dict[str, str],
    column_titles: dict[str, str] | None = None,
    selected: tuple[str, str] | None = None,
) -> go.Figure | None:
    """Each pair of *rho* (costs down, quantities across) as a square whose
    colour is the correlation and whose size is its strength, the quantities
    named by *column_labels* and described on hover by *column_titles*; the
    *selected* pair ringed. Each point's ``customdata`` is its cost, its
    quantity, ρ and the events paired. ``None`` when there is nothing to draw."""
    column_titles = column_titles or {}
    cells = rho.stack().dropna()
    if cells.empty:
        return None
    rows = [r for r in rho.index if r in cells.index.get_level_values(0)]
    columns = list(rho.columns)
    y = [row_labels.get(r, r) for r, _ in cells.index]
    x = [column_labels.get(c, c) for _, c in cells.index]
    values = cells.to_numpy(dtype=float)
    counts = [int(n.loc[r, c]) for r, c in cells.index]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            symbol="square", size=8 + 22 * np.abs(values), color=values,
            colorscale=_DIVERGING, cmin=-1, cmax=1, line=dict(width=0),
            colorbar=dict(title=dict(text="ρ", side="right"), thickness=12,
                          tickvals=[-1, -0.5, 0, 0.5, 1]),
        ),
        customdata=[
            [r, c, v, k, column_titles.get(c, c)]
            for (r, c), v, k in zip(cells.index, values, counts)
        ],
        hovertemplate=(
            "<b>%{y}</b> against <b>%{x}</b><br>%{customdata[4]}<br>ρ = %{customdata[2]:+.2f} "
            "over %{customdata[3]} events<br>Click to plot<extra></extra>"
        ),
        showlegend=False,
    ))
    if selected is not None and selected in cells.index:
        r, c = selected
        fig.add_trace(go.Scatter(
            x=[column_labels.get(c, c)], y=[row_labels.get(r, r)], mode="markers",
            marker=dict(symbol="square-open", size=34, color="#222222", line=dict(width=2)),
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        template=_TEMPLATE, height=120 + 34 * len(rows),
        margin=dict(l=10, r=10, t=10, b=10), clickmode="event+select", dragmode=False,
    )
    fig.update_xaxes(
        categoryorder="array", categoryarray=[column_labels.get(c, c) for c in columns],
        side="top", tickangle=-35, showgrid=True, gridcolor="#eeeeee",
    )
    fig.update_yaxes(
        categoryorder="array", categoryarray=[row_labels.get(r, r) for r in rows][::-1],
        showgrid=True, gridcolor="#eeeeee",
    )
    return fig
