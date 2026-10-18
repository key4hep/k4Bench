"""Reconstruction Efficiency: which algorithms run inefficiently, and how that
moves from night to night.

Current Run draws one point per algorithm of a configuration's event loop,
coloured by the stack package its code comes from: its time per event against
its instructions per cycle, sized by its heap allocations per event (by
default), its allocation rate or allocated memory, or not at all. Low
instructions per cycle at a high time points at code waiting on memory.
Historical Trends draws the slowest algorithms across the trend window in three
panels, time, instructions and instructions per cycle per event: time that
moved while the instructions held is the machine, instructions that moved the
software. The warm-up event is left out, as everywhere a run's event loop is
summarised. A line above each chart says what it shows; how to read it is
behind its help.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data import cached_load_run_packages, cached_load_trend_efficiency
from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.efficiency import algorithm_efficiency
from k4bench.analysis.plots.efficiency import (
    LABELLED,
    STALL_IPC,
    plot_component_efficiency,
)
from k4bench.analysis.plots.phases import latest_run_per_tag
from k4bench.provenance.sources import package_of
from section_dispatch import PageContext, judgeable_trend
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

#: The event that warms the job up (``component_stats.WARM_UP_EVENT``).
_WARM_UP_EVENT = 0

_KEY = "reco_efficiency"

#: What a point's area shows, by its option's name: a
#: :data:`~k4bench.analysis.plots.efficiency.SIZE_METRICS` column, or nothing.
_SIZES = {
    "Allocations per event": "allocations_per_event",
    "Allocation rate": "allocations_per_s",
    "Allocated memory per event": "allocated_mb_per_event",
    "Same size": None,
}

#: The slowest algorithms the history draws, by default.
_HIST_TOP_N = 5
_HIST_TOP_N_KEY = f"{_KEY}_hist_topn"

#: The history's panels: a trend column and its title, and the hover's unit.
_HIST_PANELS = [
    ("ms_per_event", "Time per event (ms)"),
    ("instructions_g", "Instructions per event (×10⁹)"),
    ("ipc", "Instructions per cycle"),
]
_HIST_UNITS = {"ms_per_event": "ms", "instructions_g": "G", "ipc": ""}


def _packages(ctx: PageContext, timing: ComponentTiming, names: pd.Index) -> pd.Series:
    """The stack package each of *names* comes from, by its class's library."""
    packages = cached_load_run_packages(ctx.data_dir)
    libraries = timing.components["impl_library"]
    return pd.Series(
        [getattr(package_of(libraries[name], packages), "label", "Unknown") for name in names],
        index=names,
    )


def _intro() -> None:
    """One line on what a point is; how to read the chart behind its help."""
    st.caption(
        "Each point is an algorithm in the event loop, coloured by the package its "
        "code comes from.",
        help=(
            "**Further right:** the longer it takes per event (log scale).\n\n"
            "**Higher up:** the more work the CPU gets done in each clock cycle "
            "(instructions per cycle).\n\n"
            "An algorithm runs efficiently at around 2 or more instructions per "
            f"cycle. In the shaded band below {STALL_IPC:g} it spends most of its "
            "time waiting, usually on memory."
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
        value=min(LABELLED, n_algorithms), key=key,
        help="How many of the slowest algorithms carry their name; hovering "
             "names any of them.",
    )


def _render_current_run(ctx: PageContext) -> None:
    labels = sorted(ctx.run.components)
    if not labels:
        st.info(
            "No algorithm timing in this run: no configuration that succeeded "
            "left a `_components.json`."
        )
        return
    config_col, size_col, named_col = st.columns([4, 2, 2], vertical_alignment="bottom")
    with config_col:
        config = st.selectbox("Configuration", labels, key=f"{_KEY}_config")
    drop_stale_selection(f"{_KEY}_size", list(_SIZES))
    with size_col:
        size = _SIZES[st.selectbox(
            "Point size", list(_SIZES), key=f"{_KEY}_size",
            help="What a point's area shows, the largest point being the most. "
                 "Allocation rate is allocations per second of the algorithm's "
                 "own time: how hard it leans on the allocator, however long it runs.",
        )]
    timing = ctx.run.components[config]
    table = algorithm_efficiency(timing, warm_up=_WARM_UP_EVENT)
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
    packages = _packages(ctx, timing, table.index)
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
    trend = judgeable_trend(ctx, cached_load_trend_efficiency)
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
    )
    rows = latest_run_per_tag(trend[trend["label"] == config])
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
        "The slowest algorithms night by night: their time, the work they do and "
        "how fast the CPU gets through it.",
        help=(
            "**Instructions** follow the work an algorithm does, not the machine it "
            "ran on.\n\n"
            "**Time moved, instructions held:** the machine ran it slower; "
            "instructions per cycle fell with it.\n\n"
            "**Instructions moved:** the software, its configuration or its input "
            "changed what it does."
        ),
    )
    render_historical_trends(
        series, top, _HIST_PANELS,
        std_col="", n_col_candidates=[], unit="", units=_HIST_UNITS, error_sources={},
        key_prefix=f"{_KEY}_hist", display_options_slot=display_options_slot,
        extra_controls=(
            top_n_control(_HIST_TOP_N_KEY, "algorithms", "algorithms", default=_HIST_TOP_N),
        ),
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
