"""The heap a k4run job's components held, from its memory pass
(:class:`~k4bench.analysis.heap.ComponentHeap`): every component from
initialization through each event to finalization
(:func:`plot_heap_timeline`), and one component event by event
(:func:`plot_component_heap_events`). Values are in MB, as the
:mod:`~k4bench.analysis.heap` summaries give them."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from k4bench.analysis.heap import NO_COMPONENT

from ._theme import _PALETTE, _TEMPLATE, _vivid
from ._utils import bottom_legend

#: What the heap no component allocated is called on a chart.
NO_COMPONENT_LABEL = "No component"

#: The two moments around the events, as the timeline's axis names them.
INITIALIZED = "Initialized"
FINALIZED = "Finalized"

#: Event numbers the timeline's axis names, besides its two moments.
_EVENT_TICKS = 8

#: The chart's surface, which also draws the gaps between stacked bands: a
#: band is set apart by the surface, not by a stroke around it.
_SURFACE = "#ffffff"
#: Recessive chrome beside the data: hairline gridlines, the rules setting
#: the two moments apart from the events, and muted text.
_GRID = "#ecebe6"
_RULE = "#c3c2b7"
_MUTED = "#898781"
#: Ink for text and for the RSS line, and text set on a dark fill.
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
#: The neutral bands: what no component allocated, and the components folded
#: into "Other"; as lines, a step darker so a line shows on the surface.
_NO_COMPONENT_FILL = "#e9e8e3"
_OTHER_FILL = "#cfcdc6"
_NO_COMPONENT_LINE = "#b5b3aa"
_OTHER_LINE = "#8f8d85"
#: The share of the axis a band needs at the last event to carry its name
#: inside it; a thinner one is named by the legend and the hover alone.
_LABEL_SHARE = 0.06


def format_mb(value: float, *, signed: bool = False) -> str:
    """*value* MB as a reader takes it in: GB from 1024 MB, kB below 0.1 MB,
    three significant digits, with its sign when *signed*."""
    if not math.isfinite(value):
        return "—"
    sign = "+" if signed and value > 0 else ""
    if abs(value) >= 1024:
        return f"{sign}{value / 1024:.3g} GB"
    if 0 < abs(value) < 0.1:
        return f"{sign}{value * 1024:.3g} kB"
    return f"{sign}{value:.3g} MB"


def _luminance(color: str) -> float:
    """The relative luminance of a ``#rrggbb`` *color*."""
    channels = [int(color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


#: The contrast with its fill above which a band's name is set in white,
#: as on a saturated hue; below it, in ink.
_WHITE_TEXT_CONTRAST = 4.0


def _text_on(fill: str) -> str:
    """White on a *fill* dark enough for it, otherwise ink."""
    if not fill.startswith("#") or len(fill) != 7:
        return _INK
    return _SURFACE if 1.05 / (_luminance(fill) + 0.05) >= _WHITE_TEXT_CONTRAST else _INK


def _style(fig: go.Figure) -> None:
    """Recessive axes: hairline horizontal gridlines only, muted ticks."""
    fig.update_xaxes(showgrid=False, tickfont=dict(color=_MUTED), title_font=dict(color=_SECONDARY_INK))
    fig.update_yaxes(
        gridcolor=_GRID, gridwidth=1, zeroline=False,
        tickfont=dict(color=_MUTED), title_font=dict(color=_SECONDARY_INK),
    )


def _scale(largest_mb: float) -> tuple[float, str]:
    """The divisor and unit an axis reaching *largest_mb* is drawn in."""
    return (1024.0, "GB") if largest_mb >= 2048 else (1.0, "MB")


def _event_ticks(events: list[int], *, clear_ends: bool = False) -> list[str]:
    """About :data:`_EVENT_TICKS` of *events* at a round step, as category
    labels; with *clear_ends*, none so close to the first or last event that
    it would run into a label beside the events."""
    if not events:
        return []
    raw = max(1.0, len(events) / _EVENT_TICKS)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    clearance = max(2, round(0.05 * len(events))) if clear_ends else 0
    first, last = events[0], events[-1]
    return [
        str(e) for e in events
        if (e - first) % step == 0 and e - first >= clearance and last - e >= clearance
    ]


def _named_bands(held: pd.DataFrame, named: list[str]) -> list[str]:
    """*named*, the most stable first: by how far what each held moved across
    the events, so the bands that move are drawn on top of those that do not."""
    present = [name for name in named if name in held.columns and name != NO_COMPONENT]
    moved = (held[present].max() - held[present].min()).reindex(present)
    return moved.sort_values(kind="stable").index.tolist()


def plot_heap_timeline(
    held: pd.DataFrame,
    before_first: pd.Series,
    after_finalize: pd.Series,
    named: list[str],
    *,
    colors: Mapping[str, str] | None = None,
    rss: pd.Series | None = None,
    change: bool = False,
    warm_up: int | None = 0,
    steps: pd.DataFrame | None = None,
) -> go.Figure | None:
    """The heap each series held, from initialization through each event to
    finalization.

    *held* is events × series, in MB at each event's end, indexed by event
    number: the job's components (``execute["held_bytes"]``) or groups of them,
    :data:`~k4bench.analysis.heap.NO_COMPONENT` among them.
    *before_first* and *after_finalize* are the same series' heap before the
    first event and after the job's finalize.

    Stacked, the *named* series are a band each, the most stable lowest; the
    rest are one "Other (k)" band, and what no component allocated the lowest
    band of all, both in neutral greys. The bands are opaque and set apart by
    a gap in the surface colour; one wide enough at the last event carries its
    name inside it there. A dashed line is *rss*, the anonymous RSS of the job
    at each event's end (MB, indexed by event number) where given; the top of
    the stack, the heap of the whole job, is in the hover. With *change*, each
    of the *named* series is instead a line of what it held at each event's
    end beyond what it held at the first event's end, so growth rises and a
    step jumps, and the rest and what no component allocated are two grey
    lines: together the lines make the whole heap's change, so growth in a
    series not named still shows. *colors* maps a series to its colour; the others take the
    default palette's colours but its grey, in order, which a caller drawing
    more series than it has colours must not rely on. The
    *warm_up* event is shaded. *steps* marks the events where a series
    rose at once, one row each with its ``series``, ``event_number`` and
    ``step_mb`` (:meth:`~k4bench.analysis.heap.ComponentHeap.steps`): a
    marker in the series' colour on the top of the stack, one for all of an
    event's steps (in the largest's colour, every one in the hover), or on
    the series' line with *change* (a step of a series not named is not
    marked there).
    ``None`` when there is nothing to draw.
    """
    if held.empty:
        return None
    held = held.sort_index()
    events = [int(e) for e in held.index]
    colors = colors or {}
    vivid = _vivid(_PALETTE)
    palette = iter(vivid * (len(named) // len(vivid) + 1))
    color_of = {name: colors.get(name) or next(palette) for name in named}
    fig = go.Figure()

    if change:
        grown = held - held.iloc[0]
        lines = [name for name in named if name in grown.columns and name != NO_COMPONENT]
        rest = [c for c in grown.columns if c not in lines and c != NO_COMPONENT]
        series = [(name, grown[name], dict(color=color_of[name], width=2)) for name in lines]
        if rest:
            series.append((f"Other ({len(rest)})", grown[rest].sum(axis=1), dict(color=_OTHER_LINE, width=2)))
        if NO_COMPONENT in grown.columns:
            series.append((NO_COMPONENT_LABEL, grown[NO_COMPONENT], dict(color=_NO_COMPONENT_LINE, width=2, dash="dot")))
        divisor, unit = _scale(max((float(values.abs().max()) for _, values, _ in series), default=0.0))
        for label, values, line in series:
            fig.add_trace(go.Scatter(
                x=[str(e) for e in events], y=values / divisor, name=label, mode="lines", line=line,
                hovertemplate=f"{label}: %{{y:+.3g}} {unit}<extra></extra>",
            ))
        fig.add_hline(y=0, line=dict(color=_RULE, width=1))
        moments = [str(e) for e in events]
        y_title = f"Change since the first event ({unit})"
    else:
        moments = [INITIALIZED, *(str(e) for e in events), FINALIZED]
        table = pd.concat([
            before_first.reindex(held.columns).to_frame(INITIALIZED).T,
            held.set_axis([str(e) for e in events]),
            after_finalize.reindex(held.columns).to_frame(FINALIZED).T,
        ]).fillna(0.0)
        bands = _named_bands(held, named)
        rest = [c for c in held.columns if c not in bands and c != NO_COMPONENT]
        stack: list[tuple[str, pd.Series, str]] = []
        if NO_COMPONENT in table.columns:
            stack.append((NO_COMPONENT_LABEL, table[NO_COMPONENT], _NO_COMPONENT_FILL))
        if rest:
            stack.append((f"Other ({len(rest)})", table[rest].sum(axis=1), _OTHER_FILL))
        stack += [(name, table[name], color_of[name]) for name in bands]
        total = table.sum(axis=1)
        largest = max(float(total.max()), float(rss.max()) if rss is not None and len(rss) else 0.0)
        divisor, unit = _scale(largest)
        last, below = str(events[-1]), 0.0
        for label, values, color in stack:
            fig.add_trace(go.Scatter(
                x=moments, y=values / divisor, name=label, mode="lines", stackgroup="heap",
                line=dict(color=_SURFACE, width=1.5), fillcolor=color,
                hovertemplate=f"{label}: %{{y:.3g}} {unit}<extra></extra>",
            ))
            width = float(values[last])
            if largest and width / largest >= _LABEL_SHARE:
                fig.add_annotation(
                    x=last, y=(below + width / 2) / divisor, text=label, showarrow=False,
                    xanchor="right", xshift=-8, font=dict(size=11, color=_text_on(color)),
                )
            below += width
        fig.add_trace(go.Scatter(
            x=moments, y=total / divisor, mode="lines", line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hovertemplate=f"<b>Heap held: %{{y:.3g}} {unit}</b><extra></extra>",
        ))
        if rss is not None and len(rss):
            rss = rss.reindex(events)
            fig.add_trace(go.Scatter(
                x=[str(e) for e in events], y=rss / divisor, name="Anonymous RSS (timed run)",
                mode="lines", line=dict(color=_INK, width=1.5, dash="dash"),
                hovertemplate=f"Anonymous RSS (timed run): %{{y:.3g}} {unit}<extra></extra>",
            ))
        # The two moments are not events: a hairline sets each apart.
        for x in (0.5, len(moments) - 1.5):
            fig.add_vline(x=x, line=dict(color=_RULE, width=1))
        y_title = f"Heap held ({unit})"

    if steps is not None and len(steps):
        marks = steps[steps["event_number"].isin(events)]
        if change:
            marks = marks[marks["series"].isin(lines)]
        marks = marks.assign(text=[
            f"{NO_COMPONENT_LABEL if name == NO_COMPONENT else name} {format_mb(mb, signed=True)}"
            for name, mb in zip(marks["series"], marks["step_mb"])
        ])
        if change:
            heights = [grown.at[e, name] for name, e in zip(marks["series"], marks["event_number"])]
        elif len(marks):
            # The steps of one event share the top of the stack: one marker
            # names them all, in the colour of the largest.
            marks = (
                marks.sort_values("step_mb", ascending=False, kind="stable")
                .groupby("event_number", sort=True, as_index=False)
                .agg(series=("series", "first"), text=("text", ", ".join))
            )
            heights = [total[str(e)] for e in marks["event_number"]]
        if len(marks):
            fig.add_trace(go.Scatter(
                x=[str(e) for e in marks["event_number"]], y=np.asarray(heights, dtype=float) / divisor,
                name="Step",
                mode="markers",
                marker=dict(
                    symbol="triangle-down", size=11, line=dict(color=_SURFACE, width=1),
                    color=[color_of.get(name, _OTHER_LINE) for name in marks["series"]],
                ),
                text=marks["text"].tolist(),
                hovertemplate="Step: %{text}<extra></extra>",
            ))

    if warm_up is not None and warm_up in events:
        at = moments.index(str(warm_up))
        fig.add_vrect(
            x0=at - 0.5, x1=at + 0.5, fillcolor="rgba(0,0,0,0.05)", line_width=0, layer="below",
            annotation_text="warm-up", annotation_position="top left",
            annotation_font=dict(size=10, color=_MUTED),
        )
    ticks = _event_ticks(events, clear_ends=not change)
    if not change:
        ticks = [INITIALIZED, *ticks, FINALIZED]
    fig.update_layout(
        template=_TEMPLATE, height=480, margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified", legend=bottom_legend(),
    )
    fig.update_xaxes(
        type="category", tickmode="array", tickvals=ticks, tickangle=0, title_text="Event number",
    )
    fig.update_yaxes(title_text=y_title, rangemode="tozero" if not change else "normal")
    _style(fig)
    return fig


def plot_component_heap_events(
    start: pd.Series,
    peak: pd.Series,
    end: pd.Series,
    *,
    before_first: float | None = None,
    after_finalize: float | None = None,
    color: str = _PALETTE[0],
    warm_up: int | None = 0,
) -> go.Figure | None:
    """One component's heap event by event, in MB, each series indexed by
    event number: a bar from what it held as the event started (*start*) up to
    the most it held during it (*peak*), the event's working set, and a marker
    at what it held at the event's end (*end*). What it held before the first
    event and after finalize, where given, head the chart, so its axis spans
    the events alone. Without peaks (a job that ran events at once), the bars
    are left out and the marks drawn alone. The *warm_up* event is paler.
    ``None`` when there is nothing to draw.
    """
    if start.empty:
        return None
    events = start.sort_index().index
    start, peak, end = (s.reindex(events) for s in (start, peak, end))
    values = np.concatenate([start.to_numpy(), peak.to_numpy(), end.to_numpy()]).astype(float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    low, high = float(values.min()), float(values.max())
    divisor, unit = _scale(high)
    opacity = [0.4 if e == warm_up else 1.0 for e in events]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=events, y=(peak - start) / divisor, base=start / divisor,
        name="During the event (start to peak)",
        marker=dict(color=color, opacity=opacity, line=dict(width=0)),
        customdata=np.column_stack([start / divisor, peak / divisor, (peak - start) / divisor]),
        hovertemplate=(
            f"start %{{customdata[0]:.3g}} {unit} · peak %{{customdata[1]:.3g}} {unit}"
            f" · working set %{{customdata[2]:.3g}} {unit}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=events, y=end / divisor, name="At the event's end", mode="markers",
        marker=dict(color=_INK, size=6, symbol="line-ew-open", line=dict(width=2)),
        hovertemplate=f"end %{{y:.3g}} {unit}<extra></extra>",
    ))
    moments = " · ".join(
        f"{text}: {format_mb(value)}"
        for value, text in ((before_first, INITIALIZED), (after_finalize, FINALIZED))
        if value is not None and math.isfinite(value)
    )
    pad = max((high - low) * 0.08, abs(high) * 0.002, 1e-6)
    fig.update_layout(
        template=_TEMPLATE, height=380, margin=dict(l=20, r=20, t=40 if moments else 20, b=20),
        title=dict(text=moments, x=0, xanchor="left", font=dict(size=13, color=_SECONDARY_INK))
        if moments else None,
        hovermode="x unified", bargap=0.1, legend=bottom_legend(),
    )
    fig.update_xaxes(title_text="Event number")
    fig.update_yaxes(title_text=f"Heap held ({unit})", range=[(low - pad) / divisor, (high + pad) / divisor])
    _style(fig)
    return fig
