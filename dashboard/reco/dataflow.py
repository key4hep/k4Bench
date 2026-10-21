"""Reconstruction Data Flow: which algorithm feeds which, in the selected run.

The graph joins the algorithms of one configuration's job by the collections
one writes and a later one reads (:mod:`k4bench.analysis.dataflow`), each
algorithm sized by its event-loop time. Algorithms sharing a class are one
square node until clicked, which expands them; clicking an algorithm lists
what it reads and writes underneath. What the recorded data cannot show is
listed under the graph rather than drawn as if it were known.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data import cached_load_dataflow
from k4bench.analysis.components import event_loop_time
from k4bench.analysis.dataflow import Collection, DataFlow, Graph
from k4bench.analysis.loader import judgeable_config_data
from k4bench.analysis.plots.dataflow import (
    FlowView,
    dataflow_view,
    layered_positions,
    plot_dataflow,
)
from section_dispatch import PageContext

_CONFIG_KEY = "reco_dataflow_config"
#: The classes whose instances are drawn one by one.
_EXPANDED_KEY = "reco_dataflow_expanded"
#: The algorithm whose reads and writes are listed.
_SELECTED_KEY = "reco_dataflow_selected"
#: Bumped after each click, so the chart comes back without a selection and
#: the next click on the same node is a click again.
_CHART_KEY = "reco_dataflow_chart"
_WRITER_KEY = "reco_dataflow_writer"

#: Drag to pan, scroll to zoom, and buttons for both plus going back to the
#: whole graph (a double click does that too).
_CHART_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtons": [["zoomIn2d", "zoomOut2d", "resetScale2d"]],
}


def _seconds_per_event(ctx: PageContext, config: str) -> dict[str, float]:
    """Each algorithm's own time per event, the warm-up left out and the
    events it was skipped in counted: what it adds to the workload, which
    adds up over a group's members. One that ran in no such event is left
    out, unknown rather than free."""
    timing = ctx.run.components.get(config)
    if timing is None:
        return {}
    per_event = event_loop_time(timing.execute["wall_s"])["seconds_per_event"].dropna()
    return {str(name): float(seconds) for name, seconds in per_event.items()}


def _clicked_node(event, fig) -> str | None:
    """The id of the node the user clicked, if the chart reports one."""
    nodes = len(fig.data) - 1
    selection = getattr(event, "selection", None) or {}
    for point in selection.get("points", []):
        if point.get("curve_number") == nodes and point.get("point_index") is not None:
            return fig.data[nodes].customdata[point["point_index"]][0]
    return None


def _handle_click(node: str, view: FlowView) -> None:
    expanded: set[str] = st.session_state[_EXPANDED_KEY]
    clicked = next(n for n in view.nodes if n.id == node)
    if clicked.is_group:
        expanded.add(clicked.impl)
        st.session_state[_SELECTED_KEY] = None
    else:
        st.session_state[_SELECTED_KEY] = node


def _io_table(flow: DataFlow, graph: Graph, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What algorithm *name* reads (with who wrote it) and writes (with who
    reads it)."""
    algorithm = next(a for a in flow.algorithms if a.name == name)
    unresolved = {(c, read.store, read.name) for c, read in graph.unresolved}

    def describe(collection: Collection) -> dict:
        set_by = (
            "" if collection.parameter is None
            else f"{collection.parameter} ({'set' if collection.explicit else 'default'})"
        )
        return {"Type": collection.type or "", "Parameter": set_by,
                "Store": "LCIO" if collection.store == "lcio" else "Gaudi"}

    reads = []
    for read in algorithm.inputs or ():
        producers = [
            p if link.source == link.name else f"{p} (as {link.source})"
            for (p, c), links in graph.edges.items() if c == name
            for link in links if link.name == read.name and link.store == read.store
        ]
        missing = (name, read.store, read.name) in unresolved
        written_by = ", ".join(dict.fromkeys(producers)) or ("unknown" if missing else "")
        reads.append({"Collection": read.name, "Written by": written_by, **describe(read)})
    writes = []
    for out in algorithm.outputs or ():
        consumers = [
            c for (p, c), links in graph.edges.items() if p == name
            for link in links if link.source == out.name and link.source_store == out.store
        ]
        read_by = ", ".join(dict.fromkeys(consumers)) or "none"
        writes.append({"Collection": out.name, "Read by": read_by, **describe(out)})
    return pd.DataFrame(reads), pd.DataFrame(writes)


def _render_selected(flow: DataFlow, graph: Graph, view: FlowView) -> None:
    name = st.session_state.get(_SELECTED_KEY)
    if not name or all(a.name != name for a in flow.algorithms):
        return
    algorithm = next(a for a in flow.algorithms if a.name == name)
    title_col, action_col = st.columns([4, 1], vertical_alignment="bottom")
    with title_col:
        st.markdown(f"##### {name}")
        st.caption(f"{algorithm.impl} · {'Marlin' if algorithm.framework == 'marlin' else 'Gaudi'}")
    group = view.groups.get(algorithm.impl)
    if group and algorithm.impl in st.session_state[_EXPANDED_KEY]:
        with action_col:
            if st.button(f"Collapse {algorithm.impl}", key="reco_dataflow_collapse_one"):
                st.session_state[_EXPANDED_KEY].discard(algorithm.impl)
                st.session_state[_SELECTED_KEY] = None
                st.rerun()
    if algorithm.output_commands is not None:
        # The Writer declares no data handles: its reads follow from its
        # keep/drop rules, as the graph links them into it.
        kept = sorted({link.name for (_, c), links in graph.edges.items()
                       if c == name for link in links})
        st.markdown("**Reads** (kept by `IOSvc.outputCommands`)")
        st.caption(", ".join(kept) if kept else "Nothing kept by the output commands.")
        return
    if algorithm.inputs is None and algorithm.outputs is None:
        st.info("Not recorded for this run.")
        return
    reads, writes = _io_table(flow, graph, name)
    read_col, write_col = st.columns(2, gap="medium")
    with read_col:
        st.markdown("**Reads**")
        if reads.empty:
            st.caption("None declared.")
        else:
            st.dataframe(reads, hide_index=True, width="stretch")
    with write_col:
        st.markdown("**Writes**")
        if writes.empty:
            st.caption("None declared.")
        else:
            st.dataframe(writes, hide_index=True, width="stretch")


def _unlinked(flow: DataFlow, view: FlowView) -> list[str]:
    """The algorithms not drawn, as no recorded collection links them to
    another, but the Writer; one in a drawn group node is drawn."""
    drawn = {member for node in view.nodes for member in node.members}
    return [a.name for a in flow.algorithms if a.name not in drawn and a.output_commands is None]


def _render_hidden(flow: DataFlow, view: FlowView) -> None:
    hidden = _unlinked(flow, view)
    if not hidden:
        return
    plural = "s" if len(hidden) != 1 else ""
    st.caption(f"{len(hidden)} algorithm{plural} without recorded links not drawn (see Limitations).")


def _render_limits(flow: DataFlow, graph: Graph, view: FlowView) -> None:
    silent = _unlinked(flow, view)
    with st.expander("Limitations"):
        # The Writer's reads follow from its rules, handles or not.
        unknown = [a.name for a in flow.algorithms if a.framework == "gaudi"
                   and a.output_commands is None and (a.inputs is None or a.outputs is None)]
        if unknown:
            st.markdown("- **Not recorded** (auditor too old): " + ", ".join(unknown))
        if not flow.marlin_recorded and any(a.framework == "marlin" for a in flow.algorithms):
            st.markdown("- **Not recorded:** all Marlin processors (no `Marlin -x` for this run)")
        if silent:
            st.markdown("- **Not drawn**, no collection linking them to another algorithm "
                        "(none declared, or none with a recorded writer or reader): "
                        + ", ".join(silent))
        if graph.unresolved:
            items = ", ".join(
                f"{read.name} ({consumer}{'' if read.explicit is not False else ', default'})"
                for consumer, read in graph.unresolved
            )
            st.markdown(f"- **No recorded writer:** {items}")
        st.markdown("- Declared inputs may go unread; EDM4hep↔LCIO conversions are inferred "
                    "from the job options.")


def render(ctx: PageContext) -> None:
    flows = judgeable_config_data(cached_load_dataflow(ctx.data_dir), ctx.run.results) or {}
    if not flows:
        st.info("No data flow recorded for this run.")
        return
    st.session_state.setdefault(_EXPANDED_KEY, set())
    st.session_state.setdefault(_SELECTED_KEY, None)
    st.session_state.setdefault(_CHART_KEY, 0)

    config_col, writer_col, expand_col, collapse_col = st.columns(
        [3, 2, 1, 1], gap="small", vertical_alignment="bottom",
    )
    with config_col:
        config = st.selectbox(
            "Configuration", sorted(flows), key=_CONFIG_KEY, bind="query-params",
        )
    flow = flows[config]
    with writer_col:
        writer = st.toggle("Show Writer inputs", key=_WRITER_KEY, bind="query-params")
    seconds = _seconds_per_event(ctx, config)
    view = dataflow_view(flow, seconds,
                         st.session_state[_EXPANDED_KEY], writer_edges=writer, isolated=False)
    with expand_col:
        if st.button("Expand all", disabled=not view.groups, width="stretch"):
            st.session_state[_EXPANDED_KEY] = set(view.groups)
            st.rerun()
    with collapse_col:
        if st.button("Collapse all", disabled=not st.session_state[_EXPANDED_KEY], width="stretch"):
            st.session_state[_EXPANDED_KEY] = set()
            st.rerun()

    graph = flow.graph()
    fig = plot_dataflow(view, layered_positions(view), selected=st.session_state[_SELECTED_KEY])
    if seconds:
        timed = "time per event, the warm-up left out"
    elif config in ctx.run.components:
        timed = "one size, no algorithm timed after the warm-up"
    else:
        timed = "one size, no algorithm timing recorded"
    st.caption(f"Dot size: {timed} · Square: algorithms of one class, click to expand "
               "· Click an algorithm for its collections · Hover an edge for what it carries")
    event = st.plotly_chart(
        fig, key=f"{_CHART_KEY}_{st.session_state[_CHART_KEY]}", on_select="rerun",
        selection_mode="points", width="stretch", config=_CHART_CONFIG,
    )
    if (node := _clicked_node(event, fig)) is not None:
        _handle_click(node, view)
        st.session_state[_CHART_KEY] += 1
        st.rerun()
    _render_hidden(flow, view)
    _render_selected(flow, graph, view)
    _render_limits(flow, graph, view)
