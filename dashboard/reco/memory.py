"""Reconstruction Memory: where a configuration's heap goes, component by
component, from initialization through every event to finalization, and what
grows without being released.

It reads the run's memory pass (``_heap.json``), a second run of the job after
the timed one that charges every heap block to the component that allocated
it. Current Run gives the job's heap in four tiles, names each component that
grows and still holds what it gained after finalize, and draws every
component's heap event by event, as held or as its change since the first
event, with the timed run's anonymous RSS beside it; each component the
warning names opens in Components from a button below it. Components lists every
component that held heap with its numbers and its heap across the events, and
draws the selected one event by event: what it held as each event started,
the most it held during it and what it kept. Historical Trends draws one of a
configuration's numbers across the trend window as a line per component:
what it held before the first event or at a typical event's end, its
working set or its growth per event. The warm-up event is left out of every
statistic over events, as everywhere a run's event loop is summarised.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

from data import cached_load_component_heap, cached_load_trend_heap
from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.heap import LEAK_MIN_EVENTS, MB, NO_COMPONENT, ComponentHeap, HeapTotals
from k4bench.analysis.loader import judgeable_config_data
from k4bench.analysis.plots._theme import _vivid
from k4bench.analysis.plots.heap import (
    _NO_COMPONENT_LINE,
    _OTHER_LINE,
    NO_COMPONENT_LABEL,
    _scale,
    format_mb,
    plot_component_heap_events,
    plot_heap_timeline,
)
from k4bench.analysis.plots.phases import latest_run_per_tag, phase_component_series
from reco._sources import SOURCE_COLUMN_CONFIG, package_labels, source_columns
from section_dispatch import PageContext, judgeable_trend
from sections import TREND_WINDOW_SCOPE, SectionScope
from tabs._historical_trends import render_historical_trends
from tabs._reliability import render_reliability_filter
from ui_chrome import drop_stale_selection
from ui_utils import (
    PALETTE_NAMES,
    PALETTES,
    auto_palette_index,
    display_options,
    legend_below,
    palette_control,
    view_control_row,
)

_KEY = "reco_memory"
_VIEW_KEY = f"{_KEY}_view"

#: The component a warning's button opened the Components view on, and how
#: many times one did: each time, the table is a new widget, which starts from
#: that component's row selected.
_FOCUS_KEY = f"{_KEY}_focus"
_TABLE_NONCE_KEY = f"{_KEY}_table_nonce"

#: The event that warms the job up (``component_stats.WARM_UP_EVENT``).
_WARM_UP_EVENT = 0

#: Largest series the timeline shows by default, besides the suspects, and
#: at most; the rest are one "Other" band.
_TOP_N = 6
_MAX_SHOWN = 8
_TOP_N_KEY = f"{_KEY}_topn"

#: Points a sparkline draws at most, of a run's events.
_SPARK_POINTS = 150

#: Rows the components table shows before it scrolls.
_TABLE_ROWS = 15

#: What the timeline's bands are.
_GROUPS = ["Component", "Package"]

#: What the timeline shows of each band: held, or its change since the
#: first event's end.
_MODES = {"Held": False, "Change since the first event": True}

#: The data-area height and top margin of the timeline, as the trend views
#: size theirs for :func:`ui_utils.legend_below`.
_PLOT_H = 420
_T_MARGIN = 30


def _name(component: str) -> str:
    """How a heap column is written on screen."""
    return NO_COMPONENT_LABEL if component == NO_COMPONENT else component


def _palette_size(wanted: int) -> int:
    """How many series to size the default palette for so that, its greys
    left out for Other, it still has *wanted* colours."""
    n = wanted
    for _ in PALETTE_NAMES:
        palette = PALETTES[PALETTE_NAMES[auto_palette_index(n)]]
        if len(_vivid(palette)) >= wanted:
            break
        n = len(palette) + 1
    return n


def _colors(named: list[str], palette: list[str]) -> dict[str, str]:
    """Each of *named* in the colour of its place, from *palette* but its
    greys, which Other and no component are drawn in. The default palette
    has a colour for each; one chosen smaller repeats its colours rather than
    fold a suspect into Other."""
    vivid = _vivid(palette)
    return {name: vivid[i % len(vivid)] for i, name in enumerate(named)}


def _heaps(ctx: PageContext) -> dict[str, ComponentHeap]:
    """The run's heap files of configurations that succeeded, naming any
    that could not be read."""
    heaps, unreadable = cached_load_component_heap(ctx.data_dir)
    for label, reason in unreadable.items():
        st.warning(f"The heap file of **{label}** could not be read, so it is not offered: {reason}")
    return judgeable_config_data(heaps, ctx.run.results) or {}


@st.cache_data(show_spinner=False, max_entries=32)
def _summarise(data_dir: str, label: str) -> tuple[pd.DataFrame, HeapTotals]:
    """*label*'s per-component summary and job totals, the warm-up left out."""
    heap = cached_load_component_heap(data_dir)[0][label]
    return heap.summary(warm_up=_WARM_UP_EVENT), heap.totals(warm_up=_WARM_UP_EVENT)


def _select_config(ctx: PageContext) -> tuple[str, ComponentHeap] | None:
    """The configuration chosen for both views, and its heap; ``None``, said
    so, when the run has none."""
    heaps = _heaps(ctx)
    if not heaps:
        st.info(
            "No heap breakdown in this run: no configuration that succeeded left "
            "a `_heap.json`. The memory pass runs in the nightlies from 2026-10-19 "
            "on; one that failed leaves its log in the run as `<configuration>_heap.log`."
        )
        return None
    label = st.selectbox(
        "Configuration", sorted(heaps), key=f"{_KEY}_config", bind="query-params",
    )
    return label, heaps[label]


def _spark(values: pd.Series) -> list[float]:
    """At most :data:`_SPARK_POINTS` of *values*: the least and the most of
    each of half as many runs of them, in the order they came, so a spike in
    a single event survives the thinning."""
    if len(values) <= _SPARK_POINTS:
        return values.tolist()
    out: list[float] = []
    for run in np.array_split(values.to_numpy(dtype=float), _SPARK_POINTS // 2):
        finite = np.flatnonzero(np.isfinite(run))
        if len(finite):
            extremes = {finite[np.argmin(run[finite])], finite[np.argmax(run[finite])]}
            out += [float(run[i]) for i in sorted(extremes)]
    return out


def _measured_events(heap: ComponentHeap) -> pd.Index:
    """The events the statistics are over, in order: the warm-up left out."""
    events = heap.execute["held_bytes"].index.sort_values()
    return events[events != _WARM_UP_EVENT]


def _render_tiles(summary: pd.DataFrame, totals: HeapTotals) -> None:
    """The job's heap before the first event, in the event loop, at its
    largest, and how fast it grows."""
    before = summary["before_first_mb"].dropna()
    growth = summary["growth_mb_per_event"].dropna()
    tiles = [
        (
            "Before the first event", format_mb(totals.before_first_mb),
            f"largest: {_name(before.idxmax())}, {before.max() / totals.before_first_mb:.0%}"
            if len(before) and totals.before_first_mb > 0 else "",
            "What initialize and start left on the heap: geometry, conditions, "
            "lookup tables, libraries loaded.",
            None,
        ),
        (
            "Event loop", format_mb(totals.typical_mb),
            "median at an event's end",
            "The heap at the end of each event, the warm-up event left out: its "
            "median here, every event in the sparkline.",
            _spark(totals.per_event_mb),
        ),
        (
            "Largest", format_mb(totals.largest_mb),
            f"at the end of event {totals.largest_event}" if totals.largest_event is not None else "",
            "The most heap held at an event's end, the warm-up event left out. "
            "Within an event it can be higher, for a moment.",
            None,
        ),
        (
            "Growth per event", format_mb(totals.growth_mb_per_event, signed=True),
            f"fastest: {_name(growth.idxmax())}, {format_mb(growth.max(), signed=True)}"
            if len(growth) and growth.max() > 0 else "",
            "How much the heap grows from one event to the next: the median of "
            "the slopes between every two events (Theil–Sen), so a single large "
            "event does not move it.",
            None,
        ),
    ]
    for column, (title, value, caption, help_text, spark) in zip(st.columns(len(tiles)), tiles):
        with column, st.container(border=True, height="stretch"):
            st.metric(
                title, value, help=help_text,
                chart_data=spark or None, chart_type="area" if spark else "line",
            )
            if caption:
                st.caption(caption)


def _render_leaks(summary: pd.DataFrame, heap: ComponentHeap) -> None:
    """One line per component that grows and keeps what it gained, or why
    none could be checked."""
    events = _measured_events(heap)
    if 0 < len(events) < LEAK_MIN_EVENTS:
        st.info(
            f"Only {len(events)} events after the warm-up: too few to tell steady growth "
            f"from a single step ({LEAK_MIN_EVENTS} are needed), so no component is "
            "checked for growth it keeps."
        )
        return
    suspects = summary[summary["leak"]].sort_values("growth_mb_per_event", ascending=False)
    if suspects.empty:
        return
    span = float(events[-1] - events[0])
    lines = [
        f"**{_name(name)}** grows {format_mb(row.growth_mb_per_event)} per event "
        f"({format_mb(row.growth_mb_per_event * span)} over the run); after finalize it "
        f"still holds {format_mb(row.retained_mb)} more than after event {events[0]}."
        for name, row in suspects.iterrows()
    ]
    st.warning(
        lines[0] if len(lines) == 1 else "\n".join(f"- {line}" for line in lines),
        icon="⚠️",
    )
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption("Inspect in Components:")
        for name in suspects.index:
            st.button(
                _name(name), key=f"{_KEY}_inspect_{name}", type="tertiary",
                icon=":material/arrow_forward:", on_click=_inspect, args=(name,),
            )


def _inspect(name: str) -> None:
    """Open *name* in the Components view, its row selected and drawn."""
    st.session_state[_VIEW_KEY] = "Components"
    st.session_state[_FOCUS_KEY] = name
    st.session_state[_TABLE_NONCE_KEY] = st.session_state.get(_TABLE_NONCE_KEY, 0) + 1


def _clear_focus() -> None:
    """A row the user selects, or none, outranks a warning's choice."""
    st.session_state.pop(_FOCUS_KEY, None)


def _groups(ctx: PageContext, label: str, names) -> pd.Series:
    """The package each heap column belongs to: no component stays its own."""
    timing: ComponentTiming | None = ctx.run.components.get(label)
    if timing is None:
        return pd.Series("Unknown", index=names, dtype=object).mask(names == NO_COMPONENT, NO_COMPONENT)
    packages = package_labels(ctx.data_dir, timing, names)
    return packages.mask(packages.index == NO_COMPONENT, NO_COMPONENT)


def _rss(ctx: PageContext, label: str) -> pd.Series | None:
    """The timed run's anonymous RSS at each event's end, in MB."""
    events = (ctx.run.event_data or {}).get(label)
    if events is None or "rss_anon_end_mb" not in events:
        return None
    rss = events.set_index("event_number")["rss_anon_end_mb"]
    # A failed /proc read is written negative.
    return rss.mask(rss < 0)


def _render_current_run(ctx: PageContext, display_options_slot) -> None:
    if (selected := _select_config(ctx)) is None:
        return
    label, heap = selected
    summary, totals = _summarise(ctx.data_dir, label)
    _render_tiles(summary, totals)
    _render_leaks(summary, heap)

    st.markdown(
        "##### Heap by component, event by event",
        help=(
            "What each component held on the heap after initialization, at the "
            "end of every event and after finalization, stacked: the most "
            "stable at the bottom, what moves on top. The dashed line is the "
            "timed run's anonymous RSS, the memory resident in RAM: heap that "
            "was allocated and never touched takes none, and what libraries map "
            "themselves is not heap.\n\n"
            "**Change since the first event** draws what each component holds "
            "beyond what it held at the first event's end: growth rises, a "
            "step jumps."
        ),
    )
    group_col, mode_col, shown_col = st.columns([1.6, 2.6, 2], vertical_alignment="bottom")
    drop_stale_selection(f"{_KEY}_group", _GROUPS)
    with group_col:
        group = st.segmented_control(
            "Group by", _GROUPS, default="Component", key=f"{_KEY}_group", bind="query-params",
        ) or "Component"
    drop_stale_selection(f"{_KEY}_mode", list(_MODES))
    with mode_col:
        change = _MODES[st.segmented_control(
            "Show", list(_MODES), default="Held", key=f"{_KEY}_mode", bind="query-params",
        ) or "Held"]
    with shown_col:
        top_n = st.slider(
            "Largest shown", min_value=1, max_value=_MAX_SHOWN, value=_TOP_N, key=_TOP_N_KEY,
            bind="query-params",
            help="How many of the largest components (or packages) get a band of "
                 "their own; the rest are one Other band. One that grows and keeps "
                 "it always gets a band besides.",
        )

    held = heap.execute["held_bytes"].sort_index() / MB
    before = heap.before_first_event["held_bytes"] / MB
    after = heap.at_end["held_bytes"] / MB
    suspects = summary[summary["leak"]].sort_values("growth_mb_per_event", ascending=False)
    leaks: list[str] = suspects.index.tolist()
    if group == "Package":
        groups = _groups(ctx, label, held.columns)
        held = held.T.groupby(groups).sum().T
        before = before.groupby(groups).sum()
        after = after.groupby(groups).sum()
        leaks = list(dict.fromkeys(groups[leaks]))

    by_size = held.drop(columns=NO_COMPONENT, errors="ignore").max().sort_values(ascending=False)
    by_size = by_size[by_size > 0]
    # The suspects, then the largest: both modes name the same series, each
    # in the colour of its place here, so showing more only adds colours.
    order = list(dict.fromkeys([*leaks, *by_size.index]))
    wanted = len(set(leaks) | set(by_size.index[:top_n]))
    display = display_options(
        palette_control(f"{_KEY}_palette", _palette_size(wanted)),
        key_prefix=f"{_KEY}_display",
        slot=display_options_slot,
    )
    named = order[:wanted]
    colors = _colors(named, PALETTES[display["palette"]])

    fig = plot_heap_timeline(
        held, before, after, named, colors=colors,
        rss=_rss(ctx, label), change=change, warm_up=_WARM_UP_EVENT,
    )
    if fig is None:
        st.info(f"No event of **{label}** has heap recorded.")
        return
    legend, bottom = legend_below(len(fig.data), tick_clearance=60)
    fig.update_layout(
        height=_PLOT_H + _T_MARGIN + bottom, legend=legend,
        margin=dict(l=20, r=20, t=_T_MARGIN, b=bottom),
    )
    st.plotly_chart(fig, width="stretch", key=f"{_KEY}_timeline")


def _component_table(
    ctx: PageContext, label: str, heap: ComponentHeap, summary: pd.DataFrame,
) -> pd.DataFrame:
    """Every component that held heap at any moment, the largest typical
    holder first, with its sources and its heap across the events."""
    held = heap.execute["held_bytes"].sort_index() / MB
    # Whether a component is listed reads every moment, the warm-up event's
    # too: one that only needed memory then is still worth inspecting.
    shown = summary.loc[heap.holders()].sort_values("typical_mb", ascending=False, kind="stable")
    timing: ComponentTiming | None = ctx.run.components.get(label)
    sources = (
        source_columns(ctx.data_dir, timing, shown.index) if timing is not None
        else pd.DataFrame({"Package": "—", "Code": None, "Repository": None}, index=shown.index)
    )
    return pd.DataFrame({
        "Flag": np.where(shown["leak"], "⚠ grows", ""),
        "Component": [_name(name) for name in shown.index],
        "Initialized": shown["before_first_mb"].to_numpy(),
        "Typical": shown["typical_mb"].to_numpy(),
        "Held per event": [_spark(held[name]) for name in shown.index],
        "Working set": shown["working_set_median_mb"].to_numpy(),
        "Working set p95": shown["working_set_p95_mb"].to_numpy(),
        "Growth": shown["growth_mb_per_event"].to_numpy(),
        "Finalized": shown["after_finalize_mb"].to_numpy(),
        **{column: sources[column].to_numpy() for column in SOURCE_COLUMN_CONFIG},
    }, index=shown.index)


def _table_key(data_dir: str, label: str, nonce: int = 0) -> str:
    """The components table's widget key: one per run and configuration, so a
    selection never outlives the table it was made in, and one per warning
    button click (*nonce*), so a click starts from the row it selects."""
    return f"{_KEY}_table_{data_dir}_{label}_{nonce}"


def _default_component(shown: pd.DataFrame) -> str | None:
    """What the Components view draws with no row selected, of the components
    *shown* (their :meth:`~k4bench.analysis.heap.ComponentHeap.summary` rows):
    the one that grows fastest and keeps it, as the warning names it first,
    else the one whose events need the most, else the largest holder.
    Statistics a run lacks (no events but the warm-up, no peaks per event) are
    passed over; ``None`` when nothing is shown."""
    suspects = shown.loc[shown["leak"], "growth_mb_per_event"].dropna()
    if len(suspects):
        return suspects.idxmax()
    for column in ("working_set_median_mb", "typical_mb", "largest_mb",
                   "before_first_mb", "after_finalize_mb"):
        values = shown[column].dropna()
        if len(values) and values.max() > 0:
            return values.idxmax()
    return shown.index[0] if len(shown) else None


def _render_components(ctx: PageContext) -> None:
    if (selected := _select_config(ctx)) is None:
        return
    label, heap = selected
    summary, _ = _summarise(ctx.data_dir, label)
    table = _component_table(ctx, label, heap, summary)
    st.markdown(
        "##### Components",
        help=(
            "Every component that held heap at some moment, in MB: before the "
            "first event (**Initialized**), the median at an event's end "
            "(**Typical**), and after finalize (**Finalized**). The **working "
            "set** is the most it held during an event beyond what it held as the "
            "event started: what the event needed of it. **Growth** is how much "
            "what it holds grows from one event to the next. **⚠ grows** "
            "marks a component that grew by 10 MB or more over the events, "
            "steadily rather than in one step, and after finalize still holds "
            "10 MB or more of it, at least half of what it gained.\n\n"
            "Select a row to draw its events."
        ),
    )
    focus = st.session_state.get(_FOCUS_KEY)
    focus = focus if focus in table.index else None
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        # Up to :data:`_TABLE_ROWS` rows at once, the header's height included.
        height=35 * (min(len(table), _TABLE_ROWS) + 1) + 3,
        on_select=_clear_focus,
        selection_mode="single-row",
        selection_default={"selection": {"rows": [table.index.get_loc(focus)]}} if focus else None,
        key=_table_key(ctx.data_dir, label, st.session_state.get(_TABLE_NONCE_KEY, 0)),
        column_config={
            "Flag": st.column_config.TextColumn("Flag", width="small"),
            "Component": st.column_config.TextColumn("Component", width="medium"),
            "Initialized": st.column_config.NumberColumn("Initialized", format="%.1f MB"),
            "Typical": st.column_config.NumberColumn("Typical", format="%.1f MB"),
            # From zero: on a scale from its own least, a few kB would read
            # as a cliff.
            "Held per event": st.column_config.AreaChartColumn(
                "Held per event", width="medium", y_min=0,
                help="What it held at the end of each event, from zero up to its most.",
            ),
            "Working set": st.column_config.NumberColumn(
                "Working set", format="%.2f MB", help="The median over the events.",
            ),
            "Working set p95": st.column_config.NumberColumn("Working set p95", format="%.2f MB"),
            "Growth": st.column_config.NumberColumn(
                "Growth", format="%+.3f MB", help="Per event.",
            ),
            "Finalized": st.column_config.NumberColumn("Finalized", format="%.1f MB"),
            **SOURCE_COLUMN_CONFIG,
        },
    )
    # Positions in this table; one past its end selects nothing.
    rows = [row for row in (event.selection.rows if event is not None else []) if 0 <= row < len(table)]
    if rows:
        name = table.index[rows[0]]
    elif focus:
        name = focus
    else:
        name = _default_component(summary.loc[table.index])
    if name is not None:
        _render_component_events(heap, summary, name)


def _render_component_events(heap: ComponentHeap, summary: pd.DataFrame, name: str) -> None:
    """*name*'s heap event by event."""
    st.markdown(
        f"##### {_name(name)}, event by event",
        help=(
            "Each bar runs from what it held as the event started to the most it "
            "held during the event, its working set; the mark is what it held at "
            "the event's end. A bar that starts where the last one ended is memory "
            "kept from one event to the next."
        ),
    )
    fig = plot_component_heap_events(
        heap.execute["start_held_bytes"][name] / MB,
        heap.execute["peak_held_bytes"][name] / MB,
        heap.execute["held_bytes"][name] / MB,
        before_first=summary.at[name, "before_first_mb"],
        after_finalize=summary.at[name, "after_finalize_mb"],
        warm_up=_WARM_UP_EVENT,
    )
    if fig is not None:
        st.plotly_chart(fig, width="stretch", key=f"{_KEY}_component_events")


_HIST_KEY = f"{_KEY}_hist"

#: What the history draws of each component, by its option's name: a
#: :func:`~k4bench.analysis.trend.build_heap_trend` column.
_HIST_METRICS = {
    "Before the first event": "before_first_mb",
    "Typical event": "typical_mb",
    "Working set": "working_set_median_mb",
    "Growth per event": "growth_mb_per_event",
}

#: The metrics whose components add up to the job's heap: the rest of them
#: are one Other line, and what no component allocated is a line of its own.
_ADDITIVE = ("before_first_mb", "typical_mb")

#: Each metric's panels: a trend column and its title, before its unit.
_HIST_PANELS = {
    "before_first_mb": [("before_first_mb", "Heap held")],
    "typical_mb": [("typical_mb", "Heap held")],
    "working_set_median_mb": [
        ("working_set_median_mb", "Median over the events"),
        ("working_set_p95_mb", "95th percentile over the events"),
    ],
    "growth_mb_per_event": [("growth_mb_per_event", "Growth per event")],
}

#: The line above the history's chart, and what is behind its help.
_HIST_HEADINGS = {
    "before_first_mb": (
        "Heap before the first event, night by night",
        "What each component held when initialize and start were done: geometry, "
        "conditions, lookup tables, libraries loaded.",
    ),
    "typical_mb": (
        "Heap at an event's end, night by night",
        "What each component held at the end of an event: its median over the run's "
        "events, the warm-up event left out.",
    ),
    "working_set_median_mb": (
        "Working set, night by night",
        "The most each component held during an event beyond what it held as the event "
        "started: what the event needed of it. Its median over the run's events on the "
        "left; on the right its 95th percentile, what the heavier events needed.",
    ),
    "growth_mb_per_event": (
        "Growth per event, night by night",
        "How much what each component holds grows from one event to the next: the "
        "median of the slopes between every two events (Theil–Sen), so a single large "
        "event does not move it.",
    ),
}

#: Which series the history draws, behind the same help.
_HIST_SERIES = (
    "\n\nThe largest components over the window are named (**Largest shown**), and "
    "besides them every component that grows and keeps it on any night of the window. "
    "When a nightly tag was run more than once, only its latest run is drawn."
)
_HIST_REST = (
    " The rest are one *Other* line, and what no component allocated is *No "
    "component*, both in grey."
)


def _hist_colors(named: list[str], palette: list[str], other: str | None) -> dict[str, str]:
    """:func:`_colors`, with *other* and no component in the grey of the
    timeline's lines."""
    colors = _colors(named, palette)
    colors[NO_COMPONENT_LABEL] = _NO_COMPONENT_LINE
    if other is not None:
        colors[other] = _OTHER_LINE
    return colors


def _mb_format(divisor: float) -> Callable[[float], str]:
    """How a value drawn in units of *divisor* MB is written."""
    def write(value: float) -> str:
        return format_mb(value * divisor)
    return write


def _growth_format(value: float) -> str:
    return f"{format_mb(value, signed=True)} per event"


def _render_historical(ctx: PageContext, reliability_slot, display_options_slot) -> None:
    trend = judgeable_trend(ctx, cached_load_trend_heap)
    if trend is None or trend.empty:
        st.info(
            "No heap data in the selected window. Widen the trend window in the "
            "sidebar: heap data starts with the 2026-10-19 nightly."
        )
        return
    trend = render_reliability_filter(
        trend, ctx.reliability, key=f"{_HIST_KEY}_exclude_unreliable", slot=reliability_slot,
    )
    if trend.empty:
        return
    config_col, metric_col, shown_col = st.columns([2, 4.6, 2], vertical_alignment="bottom")
    with config_col:
        config = st.selectbox(
            "Configuration", sorted(trend["label"].unique()), key=f"{_HIST_KEY}_config",
            bind="query-params",
        )
    drop_stale_selection(f"{_HIST_KEY}_metric", list(_HIST_METRICS))
    with metric_col:
        name = st.segmented_control(
            "Metric", list(_HIST_METRICS), default="Typical event", key=f"{_HIST_KEY}_metric",
            bind="query-params",
        ) or "Typical event"
    metric = _HIST_METRICS[name]
    additive = metric in _ADDITIVE
    with shown_col:
        top_n = st.slider(
            "Largest shown", min_value=1, max_value=_MAX_SHOWN, value=_TOP_N,
            key=f"{_HIST_KEY}_topn", bind="query-params",
            help="How many of the largest components over the window get a line "
                 "of their own. One that grows and keeps it on any night always gets "
                 "one besides.",
        )

    rows = latest_run_per_tag(trend[trend["label"] == config])
    components = rows[rows["component"].notna()]
    if not components[metric].notna().any():
        st.info(f"No {name.lower()} recorded for **{config}** in the selected window.")
        return
    # The suspects first, the fastest-growing first, as Current Run names
    # them, then the largest: a component's colour is its place here.
    suspects = (
        components[components["leak"].eq(True)].groupby("component")["growth_mb_per_event"]
        .median().sort_values(ascending=False).index.tolist()
    )
    title, about = _HIST_HEADINGS[metric]
    st.markdown(f"##### {title}", help=about + (_HIST_REST if additive else "") + _HIST_SERIES)

    if additive:
        # What no component allocated is not a component, as the part of a
        # phase no component was charged for is not: it is no rival for a
        # place among the named and is not folded into Other.
        series = phase_component_series(
            components, top_n, value=metric, always=suspects,
            apart={NO_COMPONENT: NO_COMPONENT_LABEL},
        )
        labels = list(dict.fromkeys(series["label"]))
        other = next((label for label in labels if label.startswith("Other (")), None)
        named = [label for label in labels if label not in (other, NO_COMPONENT_LABEL)]
        colors = partial(_hist_colors, named, other=other)
    else:
        # Each panel names its own largest: a component whose heavier events
        # need much but its typical one little is not left out of the 95th
        # percentile's.
        medians = (
            components[components["component"] != NO_COMPONENT]
            .groupby("component")[[column for column, _ in _HIST_PANELS[metric]]].median()
        )
        labels = list(dict.fromkeys([*suspects, *(
            name for column in medians.columns
            for name in medians[column].dropna().sort_values(ascending=False).index[:top_n]
        )]))
        series = components[components["component"].isin(labels)].assign(
            label=lambda df: df["component"],
        )
        # No line here is grey, so each takes the palette's colour of its
        # place, the grey among them.
        colors = None
    panels = _HIST_PANELS[metric]
    if metric == "growth_mb_per_event":
        unit, value_format = "MB", _growth_format
    else:
        columns = [column for column, _ in panels]
        divisor, unit = _scale(series[columns].max().max())
        series = series.assign(**{column: series[column] / divisor for column in columns})
        value_format = _mb_format(divisor)
    render_historical_trends(
        series, labels, [(column, f"{panel} ({unit})") for column, panel in panels],
        std_col="", n_col_candidates=[], unit=unit, error_sources={},
        key_prefix=_HIST_KEY, display_options_slot=display_options_slot,
        colors=colors, value_format=value_format,
    )


def render(ctx: PageContext) -> SectionScope | None:
    views = ["Current Run", "Components"] + (["Historical Trends"] if ctx.trends_enabled else [])
    drop_stale_selection(_VIEW_KEY, views)
    view, reliability_slot, display_options_slot = view_control_row(views, key=_VIEW_KEY)
    if view == "Historical Trends":
        _render_historical(ctx, reliability_slot, display_options_slot)
        return TREND_WINDOW_SCOPE
    if view == "Components":
        _render_components(ctx)
    else:
        _render_current_run(ctx, display_options_slot)
    return None
