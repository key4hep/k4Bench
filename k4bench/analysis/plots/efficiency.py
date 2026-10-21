"""How efficiently a k4run job's algorithms run in the event loop: each one's
time per event against its instructions per cycle, sized by its heap
allocations (:func:`plot_component_efficiency`, from
:func:`~k4bench.analysis.efficiency.component_efficiency`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import _PALETTE, _TEMPLATE

def _rate(value: float) -> str:
    for scale, prefix in ((1e6, " M"), (1e3, " k")):
        if value >= scale:
            return f"{value / scale:.3g}{prefix} allocations per second"
    return f"{value:.3g} allocations per second"


#: What a point's area can show: a
#: :func:`~k4bench.analysis.efficiency.component_efficiency` column and how
#: its hover writes one.
SIZE_METRICS = {
    "allocations_per_event": lambda v: f"{v:,.0f} allocations per event it ran in",
    "allocations_per_s": _rate,
    "allocated_mb_per_event": lambda v: f"{v:,.1f} MB allocated per event it ran in",
}

#: The algorithms, longest first, whose names the chart writes by their point.
LABELLED = 8

#: Instructions per cycle below which code mostly waits, on memory or
#: mispredicted branches, rather than executes: the chart shades it.
STALL_IPC = 1.0

#: The shaded band's colour, as ``r,g,b``.
STALL_RGB = "224,122,93"


def _milliseconds(value: float) -> str:
    return f"{value / 1e3:.2f} s" if value >= 1e3 else f"{value:.3g} ms"


def _coverage(ran: float, events: float) -> str:
    """The events an algorithm skipped, when it skipped any."""
    return f" ({ran:.0f} of {events:.0f} events)" if ran < events else ""


def plot_component_efficiency(
    table: pd.DataFrame,
    *,
    size: str | None = "allocations_per_event",
    groups: pd.Series | None = None,
    palette: list[str] = _PALETTE,
    labelled: int = LABELLED,
) -> go.Figure | None:
    """Each algorithm of
    :func:`~k4bench.analysis.efficiency.component_efficiency`'s *table* as a
    point:
    its time per event it ran in across on a log axis, its instructions per cycle up,
    instructions per cycle below :data:`STALL_IPC` shaded.

    *size*, a :data:`SIZE_METRICS` column, sets the points' areas; ``None``,
    or a column without values, draws them all one size. *groups* maps an
    algorithm to the group it is coloured and listed in the legend by (its
    package, say), groups with the most time first, in *palette*'s order;
    without it every point has the first colour. The *labelled* longest
    algorithms are named above or below their point in turn.
    ``None`` when there is nothing to draw.
    """
    if table.empty:
        return None
    values = table[size] if size in SIZE_METRICS else pd.Series(np.nan, index=table.index)
    largest = float(values.max()) if values.notna().any() else 0.0
    sizes = (
        8 + 46 * np.sqrt(values.fillna(0.0) / largest) if largest > 0
        else pd.Series(12.0, index=table.index)
    )
    # Above and below in turn: the longest algorithms often cost alike
    # (pattern recognition and its refit), and their names would overlap.
    positions = {
        name: "top center" if i % 2 == 0 else "bottom center"
        for i, name in enumerate(table.index[:labelled])
    }
    hover = pd.Series([
        f"{_milliseconds(ms)} per event it ran in{_coverage(ran, events)}<br>IPC {ipc:.2f}"
        + (f"<br>{SIZE_METRICS[size](v)}" if size in SIZE_METRICS and pd.notna(v) else "")
        for ms, ipc, v, ran, events in zip(
            table["ms_per_event"], table["ipc"], values,
            table["executed_events"], table["events"],
        )
    ], index=table.index)
    if groups is None:
        members = {None: table.index}
    else:
        groups = groups.reindex(table.index).fillna("Unknown")
        order = table.groupby(groups)["ms_per_event"].sum().sort_values(ascending=False).index
        members = {group: table.index[groups == group] for group in order}

    fig = go.Figure()
    for i, (group, names) in enumerate(members.items()):
        rows = table.loc[names]
        fig.add_trace(go.Scatter(
            x=rows["ms_per_event"], y=rows["ipc"], mode="markers+text", name=group,
            text=[name if name in positions else "" for name in names],
            textposition=[positions.get(name, "top center") for name in names],
            textfont=dict(size=11),
            marker=dict(
                size=sizes[names], color=palette[i % len(palette)], opacity=0.75,
                line=dict(color="white", width=1.5),
            ),
            customdata=list(zip(names, hover[names])),
            hovertemplate=(
                "<b>%{customdata[0]}</b>" + (f" · {group}" if group else "")
                + "<br>%{customdata[1]}<extra></extra>"
            ),
        ))
    fig.add_hrect(
        y0=0, y1=STALL_IPC, fillcolor=f"rgba({STALL_RGB},0.08)", line_width=0, layer="below",
    )
    fig.update_layout(
        template=_TEMPLATE, height=480, margin=dict(l=20, r=20, t=30, b=10),
        showlegend=groups is not None,
    )
    fig.update_xaxes(type="log", title_text="Self time per event it ran in (ms, log scale)")
    fig.update_yaxes(
        title_text="Instructions per cycle", rangemode="tozero",
        range=[0, max(float(table["ipc"].max()) * 1.15, 1.0)],
    )
    return fig
