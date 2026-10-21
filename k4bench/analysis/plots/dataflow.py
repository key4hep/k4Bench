"""The data-flow graph of a k4run job (:mod:`k4bench.analysis.dataflow`).

:func:`dataflow_view` collapses the algorithms that share a class (the six
tracker digitisers, say) into one group node unless the group is expanded,
:func:`layered_positions` lays the nodes out left to right in data-flow order,
and :func:`plot_dataflow` draws them.
"""

from __future__ import annotations

from collections.abc import Collection as Container
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import plotly.graph_objects as go

from k4bench.analysis.dataflow import DataFlow, Link

from ._theme import _BLUE, _TEMPLATE

#: The k4FWCore algorithm that saves the event, drawn last.
WRITER_TYPE = "Writer"

#: Sweeps of the barycentre ordering that untangles each layer.
_SWEEPS = 4

#: Segments each edge's curve is drawn with.
_CURVE_POINTS = 32

_EDGE_COLOR = "rgba(110, 118, 129, 0.45)"
_WRITER_EDGE_COLOR = "rgba(110, 118, 129, 0.18)"
_HOLLOW_COLOR = "#8a919b"
_LABEL_COLOR = "#3c4350"
#: A label's backing, so the edges it crosses do not run through its text.
_LABEL_BACKGROUND = "rgba(255, 255, 255, 0.85)"

#: Height of one row of nodes, in pixels.
_ROW_HEIGHT = 100


@dataclass(frozen=True)
class Node:
    """An algorithm, or a collapsed group of algorithms sharing a class."""

    id: str
    label: str
    #: The algorithms it stands for, in execution order.
    members: tuple[str, ...]
    impl: str
    framework: str
    #: Event-loop seconds per event (each member's ``seconds_per_event`` of
    #: :func:`~k4bench.analysis.components.event_loop_time`), summed over the
    #: members; ``None`` when any member is untimed.
    seconds: float | None

    @property
    def is_group(self) -> bool:
        return len(self.members) > 1


@dataclass(frozen=True)
class FlowView:
    """The graph as drawn: nodes in execution order and the edges between them."""

    nodes: tuple[Node, ...]
    edges: Mapping[tuple[str, str], tuple[Link, ...]]
    #: The Writer's node id, or ``None`` when the job saved nothing.
    sink: str | None
    #: The classes with several instances, expandable or collapsible.
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The nodes linked to another, through the Writer's reads left out too.
    linked: frozenset[str] = frozenset()

    @property
    def connected(self) -> set[str]:
        """The nodes an edge touches, drawn or among the Writer's left-out reads."""
        return {node for edge in self.edges for node in edge} | self.linked


def group_id(impl: str) -> str:
    return f"group:{impl}"


def dataflow_view(
    flow: DataFlow,
    seconds: Mapping[str, float] | None = None,
    expanded: Container[str] = (),
    *,
    writer_edges: bool = True,
    isolated: bool = True,
) -> FlowView:
    """Nodes and edges of *flow*: algorithms sharing a class collapsed into
    one node unless the class is in *expanded* or one instance reads from
    another, each node carrying the event-loop seconds per event of its
    algorithms (from *seconds*), the Writer's reads dropped unless
    *writer_edges*, and the nodes no edge touches, the Writer's reads
    included, dropped (but the Writer) unless *isolated*."""
    seconds = seconds or {}
    graph_edges = flow.graph().edges
    by_impl: dict[str, list[str]] = {}
    for algorithm in flow.algorithms:
        by_impl.setdefault(algorithm.impl, []).append(algorithm.name)
    # A class one of whose instances reads, directly or through other
    # algorithms, from another would read from itself as one node: drawn apart.
    groups = {
        impl: tuple(names) for impl, names in by_impl.items()
        if len(names) > 1 and not _chained(set(names), graph_edges)
    }

    node_of: dict[str, str] = {}
    nodes: list[Node] = []
    sink = None
    for algorithm in flow.algorithms:
        collapsed = algorithm.impl in groups and algorithm.impl not in expanded
        node_id = group_id(algorithm.impl) if collapsed else algorithm.name
        node_of[algorithm.name] = node_id
        if algorithm.type == WRITER_TYPE:
            sink = node_id
        if any(node.id == node_id for node in nodes):
            continue
        members = groups[algorithm.impl] if collapsed else (algorithm.name,)
        known = [seconds[m] for m in members if m in seconds]
        nodes.append(Node(
            id=node_id,
            label=f"{algorithm.impl} ×{len(members)}" if collapsed else algorithm.name,
            members=members, impl=algorithm.impl, framework=algorithm.framework,
            seconds=sum(known) if len(known) == len(members) else None,
        ))

    edges: dict[tuple[str, str], list[Link]] = {}
    linked = {sink}
    for (producer, consumer), links in graph_edges.items():
        pair = (node_of[producer], node_of[consumer])
        if pair[0] == pair[1]:
            continue
        linked.update(pair)
        if not writer_edges and pair[1] == sink:
            continue
        edges.setdefault(pair, []).extend(links)
    if not isolated:
        nodes = [node for node in nodes if node.id in linked]
    return FlowView(tuple(nodes), {p: tuple(ls) for p, ls in edges.items()}, sink, groups,
                    frozenset(linked - {None}))


def _chained(members: set[str], edges: Iterable[tuple[str, str]]) -> bool:
    """Whether one of *members* reads, directly or through algorithms outside
    them, from another."""
    seen, todo = set(), list(members)
    while todo:
        node = todo.pop()
        for producer, consumer in edges:
            if producer == node and consumer not in seen:
                seen.add(consumer)
                todo.append(consumer)
    return bool(seen & members)


def layered_positions(view: FlowView) -> dict[str, tuple[float, float]]:
    """``{node id: (x, y)}``: each node one layer right of the furthest node
    it reads from, the Writer last, and the nodes no edge touches in a row
    underneath. Within a layer, nodes are ordered by the mean height of what
    they read from, which keeps edges short and uncrossed."""
    order = {node.id: index for index, node in enumerate(view.nodes)}
    forward = [(a, b) for a, b in view.edges if order[a] < order[b] and b != view.sink]
    layer = {node.id: 0 for node in view.nodes}
    for node in view.nodes:
        for producer, consumer in forward:
            if consumer == node.id:
                layer[node.id] = max(layer[node.id], layer[producer] + 1)
    connected = view.connected
    placed = [n.id for n in view.nodes if n.id in connected and n.id != view.sink]
    last = max((layer[n] for n in placed), default=0)
    if view.sink is not None:
        layer[view.sink] = last + 1
        placed.append(view.sink)

    layers: dict[int, list[str]] = {}
    for node_id in placed:
        layers.setdefault(layer[node_id], []).append(node_id)
    y = {}
    for members in layers.values():
        for i, node_id in enumerate(members):
            y[node_id] = i - (len(members) - 1) / 2
    for _ in range(_SWEEPS):
        for index in sorted(layers):
            members = layers[index]

            def centre(node_id: str) -> float:
                above = [y[a] for a, b in forward if b == node_id and a in y]
                return sum(above) / len(above) if above else y[node_id]

            members.sort(key=centre)
            for i, node_id in enumerate(members):
                y[node_id] = i - (len(members) - 1) / 2

    positions = {node_id: (float(layer[node_id]), y[node_id]) for node_id in placed}
    loose = [n.id for n in view.nodes if n.id not in positions]
    if loose:
        bottom = min((p[1] for p in positions.values()), default=0.0) - 1.6
        span = max(last + 1, 1)
        for i, node_id in enumerate(loose):
            positions[node_id] = (span * (i + 0.5) / len(loose), bottom)
    return positions


def _describe(links: tuple[Link, ...]) -> str:
    lines = []
    for link in links:
        name = link.name if link.source == link.name else f"{link.source} → {link.name}"
        kind = f" ({link.type})" if link.type else ""
        lines.append(f"{name}{kind}")
    shown = lines[:12]
    if len(lines) > len(shown):
        shown.append(f"… {len(lines) - len(shown)} more")
    return "<br>".join(shown)


def _ms(seconds: float | None) -> str:
    if seconds is None:
        return "time not recorded"
    ms = seconds * 1000
    return f"{ms / 1000:.2f} s per event" if ms >= 1000 else f"{ms:.3g} ms per event"


def plot_dataflow(
    view: FlowView,
    positions: Mapping[str, tuple[float, float]] | None = None,
    *,
    selected: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Draw *view*: a dot per node, its area the node's time per event, a
    curve per edge, its width the number of collections it carries, hovering
    anywhere along it to list them. A group node is a square. The last trace is the nodes, its
    ``customdata[i][0]`` the id of the node at index ``i``."""
    positions = positions or layered_positions(view)
    fig = go.Figure()
    hover_x, hover_y, hover_text = [], [], []
    for (producer, consumer), links in view.edges.items():
        (x0, y0), (x1, y1) = positions[producer], positions[consumer]
        xs, ys = [], []
        for i in range(_CURVE_POINTS + 1):
            s = i / _CURVE_POINTS
            a, b, c, d = (1 - s) ** 3, 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s * s, s ** 3
            cx = (x0 + x1) / 2
            xs.append(a * x0 + (b + c) * cx + d * x1)
            ys.append((a + b) * y0 + (c + d) * y1)
        to_sink = consumer == view.sink
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(width=min(1 + 0.5 * len(links), 5),
                      color=_WRITER_EDGE_COLOR if to_sink else _EDGE_COLOR),
        ))
        # Hover points along the curve, but not near its ends, where hovering
        # should find the node.
        text = f"<b>{_short(producer, view)} → {_short(consumer, view)}</b><br>{_describe(links)}"
        inner = slice(_CURVE_POINTS // 8, _CURVE_POINTS - _CURVE_POINTS // 8 + 1)
        hover_x += xs[inner]
        hover_y += ys[inner]
        hover_text += [text] * len(xs[inner])
    fig.add_trace(go.Scatter(
        x=hover_x, y=hover_y, mode="markers", showlegend=False, text=hover_text,
        marker=dict(size=8, opacity=0), hovertemplate="%{text}<extra></extra>",
    ))

    connected = view.connected
    largest = max((n.seconds or 0.0 for n in view.nodes), default=0.0) or 1.0
    nodes = list(view.nodes)
    hollow = [n.id not in connected and n.id != view.sink for n in nodes]
    sizes = [10 + 32 * ((n.seconds or 0.0) / largest) ** 0.5 for n in nodes]
    fig.add_trace(go.Scatter(
        x=[positions[n.id][0] for n in nodes],
        y=[positions[n.id][1] for n in nodes],
        mode="markers", showlegend=False,
        text=[n.label for n in nodes],
        marker=dict(
            size=sizes,
            symbol=["square" if n.is_group else "circle" for n in nodes],
            color=["rgba(0,0,0,0)" if h else _BLUE for h in hollow],
            line=dict(
                width=[3 if n.id == selected else 1.5 for n in nodes],
                color=["#d62728" if n.id == selected else (_HOLLOW_COLOR if h else "white")
                       for n, h in zip(nodes, hollow)],
            ),
        ),
        customdata=[[n.id, n.impl, n.framework, _ms(n.seconds), len(n.members)] for n in nodes],
        hovertemplate=(
            "<b>%{text}</b><br>%{customdata[1]} · %{customdata[2]}<br>%{customdata[3]}"
            "<extra></extra>"
        ),
    ))
    ys = [p[1] for p in positions.values()] or [0.0]
    xs = [p[0] for p in positions.values()] or [0.0]
    loose_y = [positions[n.id][1] for n, h in zip(nodes, hollow) if h]
    sink_alone = view.sink is not None and not any(view.sink in edge for edge in view.edges)
    # A label goes above its node, but below it when the node one layer to
    # the left at the same height has its label above: side by side, two long
    # names would run into each other.
    above: dict[tuple[float, float], bool] = {}
    for node_id in sorted(positions, key=lambda n: positions[n]):
        x, y = positions[node_id]
        above[(x, y)] = not above.get((x - 1, y), False)
    annotations = []
    for node, size in zip(nodes, sizes):
        x, y = positions[node.id]
        up = above[(x, y)]
        text = node.label
        if node.id == view.sink and sink_alone:
            text += f"<br><span style='font-size:10px;color:{_HOLLOW_COLOR}'>inputs hidden</span>"
        annotations.append(dict(
            x=x, y=y, xref="x", yref="y", showarrow=False, text=text,
            yanchor="bottom" if up else "top", yshift=(size / 2 + 3) * (1 if up else -1),
            font=dict(size=12, color=_LABEL_COLOR), bgcolor=_LABEL_BACKGROUND, borderpad=1,
        ))
    if loose_y:
        annotations.append(dict(
            x=min(xs), y=loose_y[0] - 0.55, xref="x", yref="y", xanchor="left", showarrow=False,
            text="No collections recorded", font=dict(size=11, color=_HOLLOW_COLOR),
        ))
    fig.update_layout(
        template=_TEMPLATE,
        height=height or int(max(480, _ROW_HEIGHT * (max(ys) - min(ys) + 2))),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[min(xs) - 0.6, max(xs) + 0.6]),
        yaxis=dict(visible=False, range=[min(ys) - 0.9, max(ys) + 0.9]),
        annotations=annotations,
        hoverlabel=dict(align="left"),
        clickmode="event+select",
        # Dragging pans and the wheel zooms (with the chart's ``scrollZoom``),
        # leaving a click free to pick a node.
        dragmode="pan",
    )
    return fig


def _short(node_id: str, view: FlowView) -> str:
    return next((n.label for n in view.nodes if n.id == node_id), node_id)
