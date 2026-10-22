"""Where one component stands among the others of its job, measure by measure
(:func:`plot_standing`, from
:func:`~k4bench.analysis.component_detail.component_summary`)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._theme import _TEMPLATE
from .heap import _GRID, _INK, _MUTED

#: The component's own bar, in a strong red, and the others', a light, cool
#: grey that recedes.
ACCENT = "#f70000"
MUTED_FILL = "#dcdde0"

#: The chart's height, in pixels.
_HEIGHT = 284


@dataclass(frozen=True)
class Measure:
    """One measure a :func:`plot_standing` panel draws."""

    #: The column of the values.
    column: str
    title: str
    #: The unit the axis counts in, written after the title.
    unit: str
    #: How the hover and the arrow write a value.
    write: Callable[[float], str]
    #: What the components' values add up to, when they do (``"the event
    #: loop"``), which its share is of; ``None`` for a measure that does not add up.
    total: str | None = None
    #: How the axis writes its ticks: in full by default; ``",~s"`` for counts
    #: (``100M``), where an SI prefix is no unit to misread.
    tickformat: str = ",~g"
    #: On a log axis, a decade a tick: costs that span orders of magnitude,
    #: whose small columns a linear axis would flatten.
    log: bool = True


def plot_standing(
    values: pd.DataFrame, name: str, measures: Sequence[Measure], *, color: str = ACCENT,
) -> go.Figure | None:
    """A panel per one of *measures* that *name* has a value above zero in:
    every component of *values* (indexed by component) with one, as a column,
    the largest first, *name*'s in *color* and pointed at with its value, the
    rest grey; over the panel, its rank and, for a measure whose values add
    up, its share of their sum, under its title and unit. A panel's axis is
    logarithmic unless its measure says otherwise. A panel's columns carry their component's name as
    ``customdata[0]``. ``None`` when *name* has a value in none of them."""
    shown = []
    for measure in measures:
        column = values[measure.column]
        column = column[column.notna() & (column > 0)].sort_values(ascending=False, kind="stable")
        if name in column.index:
            shown.append((measure, column))
    if not shown:
        return None
    fig = make_subplots(rows=1, cols=len(shown), horizontal_spacing=0.07)
    for i, (measure, column) in enumerate(shown, start=1):
        place = column.index.get_loc(name)
        value = float(column[name])
        fig.add_trace(go.Bar(
            x=list(range(len(column))), y=column.to_numpy(),
            marker=dict(
                color=[color if n == name else MUTED_FILL for n in column.index],
                cornerradius=2, line=dict(width=0),
            ),
            customdata=[(n, measure.write(v)) for n, v in column.items()],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
            showlegend=False,
        ), row=1, col=i)
        axis = "" if i == 1 else str(i)
        # A log axis places annotations by the decade.
        fig.add_annotation(
            x=place, y=math.log10(value) if measure.log else value,
            xref=f"x{axis}", yref=f"y{axis}", text=measure.write(value),
            showarrow=True, arrowhead=0, arrowcolor=color, arrowwidth=1.5, ax=0, ay=-26,
            font=dict(color=_INK, size=11), bgcolor="rgba(255,255,255,0.85)",
        )
        rank = f"Rank {place + 1} of {len(column)}"
        if measure.total is not None and column.sum() > 0:
            share = value / column.sum()
            rank += f" · {'<1%' if share < 0.005 else f'{share:.0%}'} of {measure.total}"
        title = (f"<b>{measure.title}</b>"
                 f"<span style='color:{_MUTED}'>  {measure.unit}</span>")
        for text, shift, size, ink in ((title, 32, 13, _INK), (rank, 14, 12, _MUTED)):
            fig.add_annotation(
                text=text, x=0, xref=f"x{axis} domain", xanchor="left",
                y=1, yref="paper", yanchor="bottom", yshift=shift, showarrow=False,
                font=dict(color=ink, size=size),
            )
        fig.update_xaxes(visible=False, range=[-0.6, len(column) - 0.4], row=1, col=i)
        if measure.log:
            # Room above the largest for the arrow, a little below the least.
            low, high = math.log10(column.min()), math.log10(column.max())
            fig.update_yaxes(
                type="log", range=[low - 0.25, high + 0.9], dtick=1, row=1, col=i,
            )
        else:
            fig.update_yaxes(range=[0, float(column.max()) * 1.35], nticks=4, row=1, col=i)
        fig.update_yaxes(
            gridcolor=_GRID, color=_MUTED, tickformat=measure.tickformat,
            tickfont=dict(size=10), row=1, col=i,
        )
    fig.update_layout(
        template=_TEMPLATE, height=_HEIGHT, bargap=0.12,
        margin=dict(l=10, r=10, t=70, b=4),
    )
    return fig
