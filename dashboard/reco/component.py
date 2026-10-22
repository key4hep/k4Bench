"""Reconstruction Components: every algorithm and service of the selected
run's configuration, and the one selected in charts.

Any section that names a component opens it here (:mod:`reco._inspect`), and
a table holds every algorithm and service, which a row selects. The configuration and the
component are in the page URL (``?component=``), by the component's name in
the run; a component the selected run or configuration does not have is said
so, nothing of it drawn, never another run's values.

The table gives each one's time and heap as numbers and, event by event, as
sparklines. For the one selected, Current Run gives its main numbers as tiles
with sparklines, where it stands among the others on each measure, its time
and its heap event by event, and its place in the data flow. Historical Trends draws its numbers across the
trend window, a run it was missing from as a gap that is named, never as a
zero. The numbers are assembled in :mod:`k4bench.analysis.component_detail`.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import (
    COMPONENTS,
    DATAFLOW,
    HEAP,
    PACKAGES,
    cached_load_dataflow,
    cached_load_trend_component_summary,
    cached_load_trend_heap,
)
from k4bench.analysis.component_detail import (
    ABSENT,
    HELD_NONE,
    NOT_TIMED,
    NO_MEMORY_PASS,
    ComponentDetail,
    component_choices,
    component_detail,
    component_history,
    component_summary,
)
from k4bench.analysis.components import WARM_UP_EVENT, ComponentTiming
from k4bench.analysis.heap import MB, ComponentHeap
from k4bench.analysis.loader import judgeable_config_data
from k4bench.analysis.plots import Measure, plot_standing
from k4bench.analysis.plots._theme import _hex_to_rgba
from k4bench.analysis.plots.component import ACCENT
from k4bench.analysis.plots.dataflow import (
    dataflow_view,
    layered_positions,
    neighbourhood,
    plot_dataflow,
)
from k4bench.analysis.plots.heap import _GRID, _INK, _MUTED, format_mb
from k4bench.analysis.plots.phases import history_rows
from k4bench.labels import pretty_release
from reco import memory
from reco._controls import select_config
from reco._inspect import (
    COMPONENT_KEY,
    CONFIG_KEY,
    CURRENT_RUN,
    ORIGIN_KEY,
    TABLE_NONCE_KEY,
    VIEW_KEY,
    clickable_chart,
    go_back,
)
from reco._sources import SOURCE_COLUMN_CONFIG, source_columns
from reco.dataflow import render_collections, seconds_per_event
from section_dispatch import PageContext, judgeable_trend, no_components_reason
from sections import TREND_WINDOW_SCOPE, SectionScope
from tabs._historical_trends import render_historical_trends
from tabs._reliability import render_reliability_filter
from ui_chrome import drop_stale_selection
from ui_utils import view_control_row
from url_params import COMPONENT

_KEY = "reco_component"
#: A component asked for (by a link, or kept from another run or
#: configuration) that the one on screen does not have.
_MISSING_KEY = f"{_KEY}_missing"
_HIST_KEY = f"{_KEY}_hist"


def _time(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "—"
    if seconds >= 1:
        return f"{seconds:.3g} s"
    return f"{seconds * 1e3:.3g} ms" if seconds >= 1e-3 else f"{seconds * 1e6:.3g} µs"


def _count(value: float) -> str:
    """A count as a reader takes it in: k, M or G from a thousand on."""
    if not math.isfinite(value):
        return "—"
    for scale, suffix in ((1e9, " G"), (1e6, " M"), (1e3, " k")):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.3g}"


def _run_name(ctx: PageContext) -> str:
    """The run on screen, as the sidebar names it."""
    if ctx.selection is None:
        return f"run directory `{Path(ctx.data_dir).name}`"
    return f"run {ctx.selection.run} · Key4hep {pretty_release(ctx.selection.stack)}"


def _default(timing: ComponentTiming, summary: pd.DataFrame, names: list[str]) -> str | None:
    """What the section opens on when nothing asked for a component: the
    algorithm that adds the most to an event, else the first listed."""
    per_event = summary.loc[timing.algorithms(), "seconds_per_event"].dropna()
    if len(per_event):
        return str(per_event.idxmax())
    return names[0] if names else None


def _choose(name: str) -> None:
    """Show *name*: a click on one of the section's own charts."""
    st.session_state[COMPONENT_KEY] = name
    st.session_state.pop(_MISSING_KEY, None)
    st.session_state[TABLE_NONCE_KEY] = st.session_state.get(TABLE_NONCE_KEY, 0) + 1


def _event_times(timing: ComponentTiming, name: str) -> pd.Series:
    """Its own time in each event it ran in, the warm-up left out, in ms."""
    wall = timing.execute["wall_s"][name]
    return wall[(wall.index != WARM_UP_EVENT) & wall.notna()] * 1e3


def _component_table(
    ctx: PageContext, timing: ComponentTiming, summary: pd.DataFrame,
    heap: ComponentHeap | None, heap_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    """Every algorithm and service of the job, the slowest per event it ran in
    first, then those run in no event by their initialization, each with its
    time and its heap: as numbers, and event by event as sparklines. Indexed by
    name."""
    choices = component_choices(timing)
    rows = summary.loc[choices.index]
    order = rows.sort_values(
        ["seconds_when_run", "initialize_s"], ascending=False, na_position="last", kind="stable",
    ).index
    rows, choices = rows.loc[order], choices.loc[order]
    flagged = (
        heap_summary.reindex(order)[["leak", "stepped"]].fillna(False).astype(bool).any(axis=1)
        if heap_summary is not None else pd.Series(False, index=order)
    )
    table = pd.DataFrame({
        "Component": [f"{name} ⚠" if flagged[name] else name for name in order],
        "Kind": choices["category"].str.capitalize().to_numpy(),
        "Per event": rows["seconds_when_run"].to_numpy() * 1e3,
        "Time per event": [memory.spark(_event_times(timing, name)) for name in order],
        "Initialize": rows["initialize_s"].to_numpy() * 1e3,
        "Finalize": rows["finalize_s"].to_numpy() * 1e3,
    }, index=order)
    if heap is not None and heap_summary is not None:
        held = heap_summary.reindex(order)
        table["Heap held"] = held["typical_mb"].to_numpy()
        table["Heap per event"] = [
            memory.held_spark(heap, name) if name in heap_summary.index else []
            for name in order
        ]
    sources = source_columns(ctx.run_dir(PACKAGES), timing, order)
    for column in SOURCE_COLUMN_CONFIG:
        table[column] = sources[column].to_numpy()
    return table


#: How the components table draws its columns.
_TABLE_COLUMNS = {
    "Component": st.column_config.TextColumn("Component", width="medium", pinned=True),
    "Kind": st.column_config.TextColumn("Kind", width="small"),
    "Per event": st.column_config.NumberColumn(
        "Per event", format="%.2f ms",
        help="Its own wall time per event it ran in, the warm-up left out.",
    ),
    "Time per event": st.column_config.LineChartColumn(
        "Time per event", width="medium", y_min=0,
        help="Its own time in each event it ran in, the warm-up left out, from zero up.",
    ),
    "Initialize": st.column_config.NumberColumn(
        "Initialize", format="%.2f ms", help="Its own wall time in initialize.",
    ),
    "Finalize": st.column_config.NumberColumn(
        "Finalize", format="%.2f ms", help="Its own wall time in finalize.",
    ),
    "Heap held": st.column_config.NumberColumn(
        "Heap held", format="%.1f MB",
        help="From the memory pass: what it held at the end of an event, the median.",
    ),
    "Heap per event": st.column_config.AreaChartColumn(
        "Heap per event", width="medium", y_min=0,
        help="From the memory pass: what it held at the end of each event, from zero up.",
    ),
    **SOURCE_COLUMN_CONFIG,
}

#: Rows the components table shows before it scrolls.
_TABLE_ROWS = 8


def table_key(data_dir: str, config: str) -> str:
    """The components table's widget key: one per run and configuration, so a
    selection never outlives the table it was made in, and one per component
    chosen elsewhere, which it then starts from."""
    return f"{_KEY}_table_{data_dir}_{config}_{st.session_state.get(TABLE_NONCE_KEY, 0)}"


def _select_component(
    ctx: PageContext, config: str, table: pd.DataFrame, default: str | None,
) -> str | None:
    """The components *table*, the chosen one's row selected: the one
    remembered, else the link's, else *default*; returns it. One the
    configuration on screen lacks is remembered as missing, and nothing is
    chosen until it is offered again or a row is selected; it is never
    swapped for another silently."""
    names = table.index.tolist()
    requested = st.session_state.get(COMPONENT_KEY, st.query_params.get(COMPONENT) or default)
    missing = st.session_state.get(_MISSING_KEY)
    if requested is None and missing in names:
        requested = st.session_state.pop(_MISSING_KEY)
    if requested is not None and requested not in names:
        st.session_state[_MISSING_KEY] = requested
        requested = None
    st.session_state[COMPONENT_KEY] = requested

    st.markdown(
        "##### Components",
        help="The slowest per event first; then the services and the algorithms run in "
             "no event, by their initialization. "
             "The sparklines are its time and its heap in each event. **⚠** marks a "
             "component Memory warns about: its heap grows and it keeps the gain after "
             "finalize, or it stepped up by 100 MB or more in an event and kept it. "
             "Select a row to see it below; search and sort with the table's toolbar "
             "and headers.",
    )
    event = st.dataframe(
        table, hide_index=True, width="stretch",
        key=table_key(ctx.data_dir, config),
        height=35 * (min(len(table), _TABLE_ROWS) + 1) + 3,
        on_select="rerun", selection_mode="single-row",
        selection_default=(
            {"selection": {"rows": [names.index(requested)]}} if requested else None
        ),
        column_config=_TABLE_COLUMNS,
    )
    # Positions in this table; one past its end selects nothing, and
    # deselecting keeps what is shown.
    rows = [row for row in (event.selection.rows if event is not None else []) if 0 <= row < len(names)]
    if rows and names[rows[0]] != requested:
        requested = st.session_state[COMPONENT_KEY] = names[rows[0]]
        st.session_state.pop(_MISSING_KEY, None)
    if requested is not None:
        st.query_params[COMPONENT] = requested
    return requested


def _render_header(ctx: PageContext, config: str, timing: ComponentTiming, detail: ComponentDetail) -> None:
    identity = detail.identity
    st.markdown(f"#### {detail.name}")
    parts = []
    if identity is not None:
        parts.append(f"**{identity['impl']}**")
        if identity["type"] != identity["impl"]:
            parts.append(f"run by {identity['type']}")
        parts.append(identity["category"].capitalize())
        if identity["parent"]:
            parts.append(f"in {identity['parent']}")
    parts += [_run_name(ctx), f"configuration **{config}**"]
    source = source_columns(ctx.run_dir(PACKAGES), timing, [detail.name]).iloc[0]
    package_col, links_col = st.columns([3, 1], vertical_alignment="center")
    with package_col:
        st.caption(" · ".join([*parts, f"package {source['Package']}" if source["Package"] != "—"
                                else "package unknown"]))
    with links_col, st.container(horizontal=True, horizontal_alignment="right"):
        if source["Code"]:
            st.link_button("Code", source["Code"], icon=":material/code:", type="tertiary",
                           help="Where the class is defined: a code search.")
        if source["Repository"]:
            st.link_button("Repository", source["Repository"], icon=":material/open_in_new:",
                           type="tertiary", help="The package's repository at the commit this "
                                                 "run's stack built.")


def _event_spark(timing: ComponentTiming, name: str) -> list[float]:
    """Its own time in each event it ran in, the warm-up left out, in ms."""
    return memory.spark(_event_times(timing, name))


#: A card's sparkline: the grey of Streamlit's own, its area a tint of it.
_SPARK_LINE = "#9a9ca5"
_SPARK_FILL = "rgba(154, 156, 165, 0.2)"


def _sparkline(kind: str, values: list[float], unit: str) -> go.Figure:
    """*values* (in *unit*) drawn from zero, as the components table draws
    them, a line or an area (*kind*) under it, each value in the hover."""
    fig = go.Figure(go.Scatter(
        y=values, mode="lines", line=dict(color=_SPARK_LINE, width=1.5),
        fill="tozeroy" if kind == "area" else None, fillcolor=_SPARK_FILL,
        hovertemplate=f"%{{y:.3g}} {unit}<extra></extra>",
    ))
    fig.update_layout(
        height=56, margin=dict(l=0, r=0, t=4, b=0), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True, range=[0, max(values) * 1.08 or 1.0])
    return fig


def _render_tiles(
    detail: ComponentDetail, timing: ComponentTiming, heap: ComponentHeap | None,
) -> None:
    """Its main numbers, a card each: what it costs an event (an algorithm
    run in the event loop) or the job (a service), and the heap it holds."""
    s, row, name = detail.summary, detail.heap, detail.name
    tiles = []
    if s is not None and s["executed_events"] > 0:
        reason = detail.hardware_counters_error
        tiles += [
            ("Time per event", _time(s["seconds_when_run"]),
             f"median {_time(s['p50_s'])} · ran in {s['executed_events']:.0f} of "
             f"{s['events']:.0f} events",
             "Its own wall time per event it ran in, the warm-up left out, what it spent "
             "outside the components it ran itself. A filter or a sequence that stops "
             "early skips an algorithm in some events. The sparkline is every event, "
             "from zero.",
             ("line", _event_spark(timing, name), "ms")),
            ("Initialize", _time(s["initialize_s"]), "its own time",
             "Its own wall time in initialize.", None),
            ("Finalize", _time(s["finalize_s"]), "its own time",
             "Its own wall time in finalize.", None),
            ("IPC", f"{s['ipc']:.2f}" if math.isfinite(s["ipc"]) else "—",
             "instructions per cycle",
             f"The hardware counters could not be read (`{reason}`)." if reason else
             "All its instructions over all its cycles in the events it ran in, on the "
             "thread that called it. It describes the calls; it does not by itself say "
             "why its time changed.",
             None),
        ]
    elif s is not None:
        tiles += [
            ("Initialize", _time(s["initialize_s"]), "not run in the event loop",
             "Its own wall time in initialize.", None),
            ("Finalize", _time(s["finalize_s"]), "",
             "Its own wall time in finalize.", None),
        ]
    flags = []
    if row is not None:
        tiles += [
            ("Heap held", format_mb(row["typical_mb"]), "at an event's end, median",
             "From the memory pass: what it held at the end of an event. The sparkline is "
             "every event, from zero.", ("area", memory.held_spark(heap, name), "MB")),
            ("Growth per event", format_mb(row["growth_mb_per_event"], signed=True),
             f"{format_mb(row['after_finalize_mb'])} kept after finalize",
             "From the memory pass: how much what it holds grows from one event to the "
             "next (Theil–Sen), and what it still held after the job's own finalize.",
             None),
        ]
        flags = [kind for kind, on in (("Grows", row["leak"]), ("Steps up", row["stepped"])) if on]
    elif heap is None:
        tiles.append(("Heap held", "—", "no memory pass",
                      "The configuration has no memory pass in this run.", None))
    else:
        tiles.append(("Heap held", "0 MB", "none at any moment",
                      "The memory pass charged it no heap at any moment.", None))
    for column, (title, value, caption, help_text, spark) in zip(st.columns(len(tiles)), tiles):
        with column, st.container(border=True, height="stretch"):
            st.metric(title, value, help=help_text)
            if spark is not None and spark[1]:
                st.plotly_chart(
                    _sparkline(*spark), width="stretch", key=f"{_KEY}_spark_{title}",
                    config={"displayModeBar": False},
                )
            if caption:
                st.caption(caption)
            if title == "Growth per event":
                for kind in flags:
                    color, icon = memory.FINDING_KINDS[kind]
                    st.badge(kind, color=color, icon=icon,
                             help="As Memory names it: it keeps heap it took on.")


#: The measures of where a component stands.
_MEASURES = (
    Measure("loop_ms", "Event loop", "ms per event", lambda v: _time(v / 1e3), "total"),
    Measure("initialize_s", "Initialization", "s", _time, "total"),
    Measure("allocations", "Allocations", "event loop", _count, "total", tickformat=",~s"),
    Measure("heap_mb", "Heap held", "MB", format_mb),
)


def _render_standing(
    config: str, summary: pd.DataFrame, heap: pd.DataFrame | None, names: list[str], name: str,
) -> None:
    values = pd.DataFrame({
        "loop_ms": summary["seconds_per_event"] * 1e3,
        "initialize_s": summary["initialize_s"],
        "allocations": summary["allocations"],
    }).reindex(names)
    values["heap_mb"] = (heap["typical_mb"].reindex(names) if heap is not None
                         else float("nan"))
    fig = plot_standing(values, name, _MEASURES)
    if fig is None:
        return
    st.space("medium")
    st.markdown(
        "##### Ranking",
        help="Every algorithm and service that has some of a measure, as a column on a "
             "log axis, the largest first: this one in red, the others grey. Over each panel, its "
             "rank and its share of the total. The event-loop time counts the events "
             "it was skipped in, so it is what it adds to the job's time per event; the "
             "heap held is a median and is not shared out. Click a column to see that "
             "component.",
    )
    clickable_chart(fig, f"{_KEY}_standing", config, "Components", on_click=_choose,
                    width="stretch", config={"displayModeBar": False})


#: The left margin of the two charts event by event, in pixels, the same
#: whatever their tick labels' width, so their plot areas line up, and how far
#: left of the plot area their y-axis title is centred.
_EVENTS_LEFT = 100
_EVENTS_TITLE_SHIFT = 66
#: The gap between their columns, as a share of an event's width, and how
#: opaque a column's body is under its solid cap.
_EVENTS_BARGAP = 0.15
_BODY_ALPHA = 0.3


def _event_bars(
    values: pd.Series, events: pd.Index, mean: float, y_title: str,
    write: Callable[[float], str],
) -> go.Figure:
    """A column per event of *values* (indexed by event number; an event
    without one has none), from zero, translucent under a solid cap at its
    top, the warm-up fainter, and a dotted line at *mean*: the chart each of a
    component's measures event by event is drawn as, across the same
    *events*, so two side by side line up."""
    values = values.dropna()
    warm_up = values.index == WARM_UP_EVENT
    fig = go.Figure(go.Bar(
        x=values.index, y=values.to_numpy(),
        marker=dict(
            color=[_hex_to_rgba(ACCENT, _BODY_ALPHA * (0.5 if w else 1.0)) for w in warm_up],
            line=dict(width=0),
        ),
        customdata=[write(v) for v in values],
        hovertemplate="Event %{x}: %{customdata}<extra></extra>",
    ))
    # A cap as wide as its column on each: line segments, apart.
    half = (1 - _EVENTS_BARGAP) / 2
    for events_here, alpha in ((values[~warm_up], 1.0), (values[warm_up], 0.5)):
        xs, ys = [], []
        for event, value in events_here.items():
            xs += [event - half, event + half, None]
            ys += [value, value, None]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", hoverinfo="skip",
                line=dict(color=_hex_to_rgba(ACCENT, alpha), width=2),
            ))
    if math.isfinite(mean):
        fig.add_hline(y=mean, line=dict(color=_INK, width=1, dash="dot"))
    fig.update_layout(
        template="plotly_white", height=300, bargap=_EVENTS_BARGAP, showlegend=False,
        margin=dict(l=_EVENTS_LEFT, r=10, t=10, b=10),
    )
    fig.update_xaxes(
        type="linear", title_text="Event number", tickformat="d", showgrid=False,
        range=[events.min() - 0.6, events.max() + 0.6],
    )
    fig.update_yaxes(rangemode="tozero", gridcolor=_GRID, automargin=False)
    # The axis title as a label of its own: the dashboard's chart theme sets
    # the axis title's distance from the ticks, too close to read easily.
    fig.add_annotation(
        text=y_title, textangle=-90, x=0, xref="paper", xanchor="center",
        xshift=-_EVENTS_TITLE_SHIFT, y=0.5, yref="paper", showarrow=False,
        font=dict(size=13, color=_MUTED),
    )
    return fig


def _render_event_time(
    timing: ComponentTiming, detail: ComponentDetail, events: pd.Index,
) -> None:
    st.markdown(
        "##### Time per event",
        help="Its own wall time in each event, the warm-up paler; the dotted line is "
             "its mean over the events it ran in, the warm-up left out (the time per "
             "event above). The run records each call's cost per event, not when it "
             "started.",
    )
    s = detail.summary
    if s is None or not s["executed_events"]:
        st.caption("Not run in the event loop.")
        return
    fig = _event_bars(
        timing.execute["wall_s"][detail.name], events, s["seconds_when_run"], "Time (s)", _time,
    )
    st.plotly_chart(fig, width="stretch", key=f"{_KEY}_event_time")


def _render_event_heap(
    heap: ComponentHeap | None, name: str, events: pd.Index,
) -> None:
    st.markdown(
        "##### Heap per event",
        help="From the memory pass: what it held at the end of each event, the warm-up "
             "paler; the dotted line is its mean over the events, the warm-up left out. "
             "The memory pass is a run of its own, so its events are set beside the "
             "timed run's, never joined to them.",
    )
    if heap is None:
        st.caption("No memory pass for this configuration in this run.")
        return
    if name not in heap.holders():
        st.caption("The memory pass charged it no heap at any moment.")
        return
    held = heap.execute["held_bytes"][name] / MB
    fig = _event_bars(
        held, events, float(held[held.index != WARM_UP_EVENT].mean()), "Heap held (MB)",
        format_mb,
    )
    st.plotly_chart(fig, width="stretch", key=f"{_KEY}_heap_events")


def _flow(ctx: PageContext, config: str):
    flows = judgeable_config_data(cached_load_dataflow(ctx.run_dir(DATAFLOW)), ctx.run.results) or {}
    return flows.get(config)


def _render_flow(ctx: PageContext, config: str, name: str) -> None:
    st.markdown(
        "##### Data flow",
        help="What it reads, from the algorithms that wrote it, and the algorithms that "
             "read what it writes, as Data Flow joins them; hover an edge for the "
             "collections, listed under the graph. Click an algorithm to see it.",
    )
    flow = _flow(ctx, config)
    if flow is None:
        st.caption(f"No data flow recorded for **{config}** in this run.")
        return
    if all(a.name != name for a in flow.algorithms):
        st.caption("It reads and writes no event data the run recorded.")
        return
    view = dataflow_view(
        flow, seconds_per_event(ctx, config), {a.impl for a in flow.algorithms},
    )
    around = neighbourhood(view, name)
    if around.edges:
        positions = layered_positions(around)
        # As tall as its fullest layer needs.
        widest = max(Counter(round(x) for x, _ in positions.values()).values())
        fig = plot_dataflow(around, positions, selected=name, height=90 + 55 * widest, ring=ACCENT)
        clickable_chart(fig, f"{_KEY}_flow", config, "Components", on_click=_choose,
                        width="stretch", config={"displayModeBar": False})
    else:
        st.caption("No recorded collection links it to another algorithm.")
    render_collections(flow, flow.graph(), name, compact=True)


def _render_current_run(
    ctx: PageContext, config: str, timing: ComponentTiming, summary: pd.DataFrame,
    heap: ComponentHeap | None, heap_summary: pd.DataFrame | None, names: list[str], name: str,
) -> None:
    detail = component_detail(
        name, timing, summary=summary, heap_summary=heap_summary,
        heap_holders=heap.holders() if heap is not None else None, warm_up=WARM_UP_EVENT,
    )
    _render_header(ctx, config, timing, detail)
    _render_tiles(detail, timing, heap)
    _render_standing(config, summary, heap_summary, names, name)
    st.space("medium")
    # The events either pass recorded: the two charts span the same.
    events = timing.execute["wall_s"].index
    if heap is not None:
        events = events.union(heap.execute["held_bytes"].index)
    time_col, heap_col = st.columns(2, gap="medium")
    with time_col:
        _render_event_time(timing, detail, events)
    with heap_col:
        _render_event_heap(heap, name, events)
    _render_flow(ctx, config, name)


#: The history's panels by what they show: a
#: :func:`~k4bench.analysis.component_detail.component_history` column and
#: its title.
_HIST_GROUPS = {
    "Time": [
        ("ms_when_run", "Time per event it ran in (ms)"),
        ("p95_ms", "95th percentile of it (ms)"),
        ("coverage_pct", "Events it ran in (%)"),
        ("initialize_s", "Initialization (s)"),
    ],
    "CPU": [
        ("instructions_g", "Instructions per event it ran in (×10⁹)"),
        ("ipc", "Instructions per cycle"),
        ("minor_page_faults_per_event", "Minor page faults per event it ran in"),
        ("major_page_faults_per_event", "Major page faults per event it ran in"),
        ("voluntary_context_switches_per_event", "Voluntary switches per event it ran in"),
        ("involuntary_context_switches_per_event", "Involuntary switches per event it ran in"),
    ],
    "Memory": [
        ("allocations_per_event", "Allocations per event it ran in"),
        ("allocated_mb_per_event", "Allocated per event it ran in (MB)"),
        ("typical_mb", "Heap held at an event's end (MB)"),
        ("working_set_median_mb", "Working set, median (MB)"),
        ("growth_mb_per_event", "Heap growth per event (MB)"),
    ],
}
_HIST_UNITS = {
    "ms_when_run": "ms", "p95_ms": "ms", "coverage_pct": "%", "initialize_s": "s",
    "instructions_g": "G", "allocated_mb_per_event": "MB", "typical_mb": "MB",
    "working_set_median_mb": "MB", "growth_mb_per_event": "MB",
}

#: Why a run has no values, by its status.
_GAPS = {
    "status": {ABSENT: "not in the job", NOT_TIMED: "the configuration was not timed"},
    "heap_status": {
        HELD_NONE: "no heap charged to it", NO_MEMORY_PASS: "no memory pass",
    },
}
#: Runs a gap's caption names, the earliest; the rest are counted.
_GAPS_NAMED = 6


def _render_gaps(rows: pd.DataFrame, column: str, config: str) -> None:
    """Name the runs drawn as gaps, and why."""
    parts = []
    for status, why in _GAPS[column].items():
        runs = rows[rows[column] == status]
        if runs.empty:
            continue
        named = [
            f"{pd.Timestamp(d).strftime('%Y-%m-%d')} (Key4hep {pretty_release(str(r))})"
            for d, r in zip(runs["run_date"], runs["k4h_release"])
        ][:_GAPS_NAMED]
        rest = len(runs) - len(named)
        parts.append(f"{why} in {len(runs)} run{'s' if len(runs) > 1 else ''}: "
                     + ", ".join(named) + (f" and {rest} more" if rest > 0 else ""))
    if parts:
        st.caption(f"Gaps for **{config}**: " + "; ".join(parts) + ".")


def _window_components(trend: pd.DataFrame, config: str) -> list[str]:
    """The components configuration *config* had in any run of the window's
    *trend*, the slowest per event it ran in first, then by initialization."""
    rows = trend[(trend["label"] == config) & trend["component"].notna()]
    medians = rows.groupby("component")[["seconds_when_run", "initialize_s"]].median()
    return medians.sort_values(
        ["seconds_when_run", "initialize_s"], ascending=False, na_position="last", kind="stable",
    ).index.tolist()


def _select_in_window(config: str, names: list[str]) -> str | None:
    """The component whose history is drawn, chosen among *names*, those of
    *config* in the window: the one remembered, also when the run on screen
    lacks it, else the link's, else the first; returns it. One the window
    lacks is said so, and nothing is drawn until another is chosen."""
    requested = (
        st.session_state.get(COMPONENT_KEY) or st.session_state.get(_MISSING_KEY)
        or st.query_params.get(COMPONENT) or (names[0] if names else None)
    )
    choice = st.selectbox(
        "Component", names, index=names.index(requested) if requested in names else None,
        key=f"{_HIST_KEY}_component_{config}_{st.session_state.get(TABLE_NONCE_KEY, 0)}",
        placeholder="Choose a component",
    )
    if choice is not None:
        requested = choice
    if requested not in names:
        st.session_state[_MISSING_KEY] = requested
        st.session_state[COMPONENT_KEY] = None
        if requested is not None:
            st.warning(f"**{requested}** is not a component of **{config}** in any run of "
                       "the selected window, so nothing of it is shown.")
        return None
    st.session_state[COMPONENT_KEY] = requested
    st.session_state.pop(_MISSING_KEY, None)
    st.query_params[COMPONENT] = requested
    return requested


def _render_historical(ctx: PageContext, reliability_slot, display_options_slot) -> None:
    """The history of a component of any configuration the window recorded,
    whatever the run on screen holds."""
    summary_trend = judgeable_trend(ctx, cached_load_trend_component_summary, COMPONENTS)
    if summary_trend is None or summary_trend.empty:
        st.info(
            "No component costs in the selected window. Widen the trend window in "
            "the sidebar."
        )
        return
    config = _config_row(sorted(summary_trend["label"].unique()))
    name = _select_in_window(config, _window_components(summary_trend, config))
    if name is None:
        return
    drop_stale_selection(f"{_HIST_KEY}_group", list(_HIST_GROUPS))
    group = st.segmented_control(
        "Show", list(_HIST_GROUPS), default="Time", key=f"{_HIST_KEY}_group",
        bind="query-params",
    ) or "Time"
    # The heap's history reads every run's memory pass: only when asked for.
    heap_trend = judgeable_trend(ctx, cached_load_trend_heap, HEAP) if group == "Memory" else None
    history = component_history(summary_trend, heap_trend, config, name)
    history = render_reliability_filter(
        history, ctx.reliability, key=f"{_HIST_KEY}_exclude_unreliable", slot=reliability_slot,
    )
    if history.empty:
        return
    rows = history_rows(history, all_runs=ctx.window.all_runs)
    _render_gaps(rows, "status", config)
    if group == "Memory":
        _render_gaps(rows, "heap_status", config)
    panels = [(column, title) for column, title in _HIST_GROUPS[group] if rows[column].notna().any()]
    if not panels:
        st.info(f"No {group.lower()} numbers recorded for **{name}** of **{config}** in the "
                "selected window.")
        return
    render_historical_trends(
        rows, [name], panels,
        std_col="", n_col_candidates=[], unit="", units=_HIST_UNITS, error_sources={},
        key_prefix=_HIST_KEY, display_options_slot=display_options_slot,
        all_runs=ctx.window.all_runs,
    )


def _config_row(labels: list[str]) -> str:
    """The configuration chosen among *labels*, beside the way back to the
    section the Components section was opened from."""
    drop_stale_selection(CONFIG_KEY, labels)
    config_col, back_col = st.columns([2, 3], vertical_alignment="bottom")
    with config_col:
        config = select_config(labels, CONFIG_KEY)
    with back_col:
        if origin := st.session_state.get(ORIGIN_KEY):
            st.button(f"Back to {origin}", key=f"{_KEY}_back", icon=":material/arrow_back:",
                      type="tertiary", on_click=go_back)
    return config


def render(ctx: PageContext) -> SectionScope | None:
    views = [CURRENT_RUN] + (["Historical Trends"] if ctx.trends_enabled else [])
    drop_stale_selection(VIEW_KEY, views)
    view, reliability_slot, display_options_slot = view_control_row(views, key=VIEW_KEY)
    if view == "Historical Trends":
        _render_historical(ctx, reliability_slot, display_options_slot)
        return TREND_WINDOW_SCOPE
    labels = sorted(ctx.run.components)
    if not labels:
        st.info(f"No component costs in this run: {no_components_reason(ctx)}")
        return None
    config = _config_row(labels)
    timing = ctx.run.components[config]
    summary = component_summary(timing, warm_up=WARM_UP_EVENT)
    heap = memory.heaps(ctx).get(config)
    heap_summary = memory.summarise(ctx.run_dir(HEAP), config)[0] if heap is not None else None
    table = _component_table(ctx, timing, summary, heap, heap_summary)
    names = table.index.tolist()
    name = _select_component(ctx, config, table, _default(timing, summary, names))
    if name is None:
        missing = st.session_state.get(_MISSING_KEY)
        if missing:
            st.warning(
                f"**{missing}** is not a component of **{config}** in {_run_name(ctx)}, "
                "so nothing of it is shown. Select one of its components above."
            )
        return None
    _render_current_run(ctx, config, timing, summary, heap, heap_summary, names, name)
    return None
