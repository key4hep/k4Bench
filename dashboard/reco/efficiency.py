"""Reconstruction Efficiency: each algorithm's time against the instructions
its thread retired and the cycles it took, and how they move from night to night.

Current Run draws one point per algorithm of a configuration's event loop,
coloured by the stack package its code comes from: its time in the events it
ran in against its instructions per cycle, sized by its heap allocations per
event it ran in (by default), its allocation rate or allocated memory, or not at all.
Historical Trends draws the slowest algorithms across the trend window in three
panels, time, instructions and instructions per cycle per event it ran in.
The counters describe a change; they do not by themselves name its cause, so
the help says what to check rather than giving a verdict. The warm-up event is
left out, as everywhere a run's event loop is summarised. A line above each
chart says what it shows; how to read it is behind its help.
"""
from __future__ import annotations

import streamlit as st

from data import COMPONENTS, PACKAGES, cached_load_trend_efficiency
from k4bench.analysis.components import HARDWARE_METRICS, WARM_UP_EVENT
from k4bench.analysis.efficiency import algorithm_efficiency
from k4bench.analysis.plots.efficiency import LABELLED, plot_component_efficiency
from k4bench.analysis.plots.phases import history_rows
from reco._sources import package_labels
from section_dispatch import PageContext, judgeable_trend, no_components_reason
from sections import TREND_WINDOW_SCOPE, SectionScope
from tabs._historical_trends import render_historical_trends
from tabs._reliability import render_reliability_filter
from ui_chrome import drop_stale_selection
from ui_utils import (
    PALETTE_NAMES,
    PALETTES,
    auto_palette_index,
    legend_below,
    top_n_control,
    view_control_row,
)

_KEY = "reco_efficiency"

#: What a point's area shows, by its option's name: a
#: :data:`~k4bench.analysis.plots.efficiency.SIZE_METRICS` column, or nothing.
_SIZES = {
    "Allocations per event it ran in": "allocations_per_event",
    "Allocation rate": "allocations_per_s",
    "Allocated memory per event it ran in": "allocated_mb_per_event",
    "Same size": None,
}

#: The slowest algorithms the history draws, by default.
_HIST_TOP_N = 5
_HIST_TOP_N_KEY = f"{_KEY}_hist_topn"

#: The history's panels: a trend column and its title, and the hover's unit.
_HIST_PANELS = [
    ("ms_per_event", "Time per event it ran in (ms)"),
    ("instructions_g", "Instructions per event it ran in (×10⁹)"),
    ("ipc", "Instructions per cycle"),
]
_HIST_UNITS = {"ms_per_event": "ms", "instructions_g": "G", "ipc": ""}


def _intro() -> None:
    """One line on what a point is; how to read the chart behind its help."""
    st.caption(
        "Each point is an algorithm in the event loop, coloured by the package its "
        "code comes from.",
        help=(
            "**Further right:** the longer it takes in an event it runs in (log "
            "scale); hovering says how many events it skipped, if any.\n\n"
            "**Higher up:** the more instructions it retires per counted CPU "
            "cycle (IPC). Both are counted in user space on the thread that "
            "called it, so work it hands to threads of its own is not.\n\n"
            "IPC depends on the processor and the instruction mix: a vectorised "
            "implementation can finish sooner while retiring fewer instructions "
            "at a lower IPC, so a higher IPC is not better across different "
            "code. Read it with the time and the instructions when comparing "
            "similar workloads; a low IPC alone does not say an algorithm "
            "waits on memory."
        ),
    )


def _labelled_slider(n_algorithms: int) -> int:
    """How many of the slowest of *n_algorithms* the chart names, up to all."""
    key = f"{_KEY}_labelled"
    # A configuration with fewer algorithms than the stored choice names all.
    if st.session_state.get(key, 0) > n_algorithms:
        st.session_state[key] = n_algorithms
    return st.slider(
        "Named algorithms", min_value=0, max_value=n_algorithms,
        value=min(LABELLED, n_algorithms), key=key, bind="query-params",
        help="How many of the slowest algorithms carry their name; hovering "
             "names any of them.",
    )


def _render_current_run(ctx: PageContext) -> None:
    labels = sorted(ctx.run.components)
    if not labels:
        st.info(f"No algorithm timing in this run: {no_components_reason(ctx)}")
        return
    config_col, size_col, named_col = st.columns([4, 2, 2], vertical_alignment="bottom")
    with config_col:
        config = st.selectbox(
            "Configuration", labels, key=f"{_KEY}_config", bind="query-params",
        )
    drop_stale_selection(f"{_KEY}_size", list(_SIZES))
    with size_col:
        size = _SIZES[st.selectbox(
            "Point size", list(_SIZES), key=f"{_KEY}_size", bind="query-params",
            help="What a point's area shows, the largest point being the most. "
                 "Allocations count the heap blocks it allocated; allocation rate "
                 "is those per second of the algorithm's own time: how hard it "
                 "leans on the allocator, however long it runs. Allocated memory "
                 "is the heap memory those blocks take, freed or not: churn, not the heap "
                 "it keeps (that is Memory's).",
        )]
    timing = ctx.run.components[config]
    table = algorithm_efficiency(timing, warm_up=WARM_UP_EVENT)
    if table.empty and set(HARDWARE_METRICS) <= timing.execute.keys():
        st.info(
            f"No algorithm of **{config}**'s event loop has instruction counts "
            "after the warm-up event."
        )
        return
    if table.empty:
        reason = timing.hardware_counters_error
        st.info(
            f"No instruction counts for **{config}**: "
            + (f"the hardware counters could not be read (`{reason}`)." if reason
               else "the auditor did not count them.")
        )
        return
    with named_col:
        labelled = _labelled_slider(len(table))
    packages = package_labels(ctx.run_dir(PACKAGES), timing, table.index)
    n_packages = packages.nunique()
    fig = plot_component_efficiency(
        table, size=size, groups=packages,
        palette=PALETTES[PALETTE_NAMES[auto_palette_index(n_packages)]], labelled=labelled,
    )
    legend, bottom = legend_below(n_packages, tick_clearance=60)
    fig.update_layout(
        height=fig.layout.height + bottom, legend=legend,
        margin=dict(l=20, r=20, t=30, b=bottom),
    )
    _intro()
    st.plotly_chart(fig, width="stretch", key=f"{_KEY}_chart")


def _render_historical(ctx: PageContext, reliability_slot, display_options_slot) -> None:
    trend = judgeable_trend(ctx, cached_load_trend_efficiency, COMPONENTS)
    if trend is None or trend.empty:
        st.info(
            "No instruction counts in the selected window. Widen the trend window "
            "in the sidebar."
        )
        return
    trend = render_reliability_filter(
        trend, ctx.reliability, key=f"{_KEY}_hist_exclude_unreliable", slot=reliability_slot,
    )
    if trend.empty:
        return
    config = st.selectbox(
        "Configuration", sorted(trend["label"].unique()), key=f"{_KEY}_hist_config",
        bind="query-params",
    )
    rows = history_rows(trend[trend["label"] == config], all_runs=ctx.window.all_runs)
    # The Top N chooses the algorithms before the renderer draws its popover,
    # in which it then sits; Streamlit has already restored the slider's value.
    top_n = int(st.session_state.get(_HIST_TOP_N_KEY, _HIST_TOP_N))
    slowest = rows.groupby("component")["ms_per_event"].median().sort_values(ascending=False)
    top = slowest.index[:top_n].tolist()
    series = rows[rows["component"].isin(top)].assign(
        label=lambda df: df["component"],
        instructions_g=lambda df: df["instructions_per_event"] / 1e9,
    )
    st.caption(
        "The slowest algorithms night by night: their time, the instructions they "
        "retire and their instructions per cycle.",
        help=(
            "**Time** is the elapsed (wall) time of its calls; the CPU time of "
            "the thread that called it leaves out what it waited off the CPU. "
            "**Instructions** and cycles are counted in user space on that "
            "thread.\n\n"
            "**Instructions moved:** the work it executed changed, through its "
            "code, its configuration, its input or how it was compiled; the "
            "count alone does not say which.\n\n"
            "**Time moved, instructions held:** worth checking the cycles, the "
            "thread's CPU time, the host's load and scheduling, its clock and "
            "memory effects. The counters narrow the investigation; they do "
            "not establish the cause."
        ),
    )
    render_historical_trends(
        series, top, _HIST_PANELS,
        std_col="", n_col_candidates=[], unit="", units=_HIST_UNITS, error_sources={},
        key_prefix=f"{_KEY}_hist", display_options_slot=display_options_slot,
        extra_controls=(
            top_n_control(_HIST_TOP_N_KEY, "algorithms", "algorithms", default=_HIST_TOP_N),
        ),
        all_runs=ctx.window.all_runs,
    )


def render(ctx: PageContext) -> SectionScope | None:
    views = ["Current Run"] + (["Historical Trends"] if ctx.trends_enabled else [])
    drop_stale_selection(f"{_KEY}_view", views)
    view, reliability_slot, display_options_slot = view_control_row(views, key=f"{_KEY}_view")
    if view == "Historical Trends":
        _render_historical(ctx, reliability_slot, display_options_slot)
        return TREND_WINDOW_SCOPE
    _render_current_run(ctx)
    return None
