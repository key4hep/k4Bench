"""Reconstruction Breakdown: where the job's time goes, phase by phase and
component by component.

Current Run gives each phase of the selected run as a card, then a donut each
of its initialization, event loop and finalization by component, and the event
loop event by event in the donuts' colours. Components lists the selected run's
algorithms and services with their time per phase and links to their code at
the revision the run's stack built. Historical Trends draws one phase
at a time across the trend window, its components as stacked columns or, with
the trend views' shared renderer, as lines; configure, timed only as a whole,
as one line.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data import cached_load_trend_phases
from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.plots import (
    plot_event_composition,
    plot_phase_component_pie,
    plot_phase_history,
)
from k4bench.analysis.plots.phases import (
    MIN_PHASE_S,
    PHASE_NAMES,
    phase_component_series,
    top_components,
)
from reco._sources import SOURCE_COLUMN_CONFIG, source_columns
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
    fit_legend_below,
    palette_control,
    top_n_control,
    view_control_row,
)

#: Components a phase's history names; the rest are one "Other" series.
_TOP_N = 6
_TOP_N_KEY = "reco_breakdown_hist_topn"

#: Components each current-run donut names; the rest are one "Other" slice.
_PIE_TOP_N = 5

#: The phases broken down by component (``trend.COMPONENT_PHASES``), as their
#: chart titles name them.
_PHASE_TITLES = {
    "initialize": "Initialization", "execute": "Event loop", "finalize": "Finalization",
}

#: The event that warms the job up (``component_stats.WARM_UP_EVENT``).
_WARM_UP_EVENT = 0


def _seconds(value: float) -> str:
    return f"{value:.2f} s" if value < 100 else f"{value:,.0f} s"


def _select_config(ctx: PageContext) -> tuple[str, ComponentTiming] | None:
    """The configuration chosen for the run views, shared between them, and
    its component costs; ``None``, said so, when the run has none."""
    labels = sorted(ctx.run.components)
    if not labels:
        st.info(
            "No phase timing in this run: no configuration that succeeded "
            "left a `_components.json`."
        )
        return None
    config = st.selectbox(
        "Configuration", labels, key="reco_breakdown_config", bind="query-params",
    )
    return config, ctx.run.components[config]


def _render_current_run(ctx: PageContext) -> None:
    if (selected := _select_config(ctx)) is None:
        return
    config, timing = selected
    phases = timing.phase_seconds()
    total = sum(phases.values())
    shown = [p for p in PHASE_NAMES if phases.get(p, 0) >= MIN_PHASE_S]
    if not shown:
        st.info(f"No phase of **{config}** lasted {MIN_PHASE_S * 1000:.0f} ms or more.")
    for column, phase in zip(st.columns(len(shown)) if shown else (), shown):
        with column, st.container(border=True, height="stretch"):
            st.metric(PHASE_NAMES[phase], _seconds(phases[phase]))
            st.caption(f"{phases[phase] / total:.1%} of the job")

    # One colour per component across the donuts and the per-event chart: a
    # component several phases show (the tracking initializes, runs and
    # finalizes) keeps it.
    per_phase = {phase: timing.phase_component_seconds(phase) for phase in _PHASE_TITLES}
    named = list(dict.fromkeys(
        name for seconds in per_phase.values() for name in top_components(seconds, _PIE_TOP_N)
    ))
    palette = PALETTES[PALETTE_NAMES[auto_palette_index(len(named))]]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(named)}
    for column, (phase, title) in zip(st.columns(len(_PHASE_TITLES), gap="medium"),
                                      _PHASE_TITLES.items()):
        with column:
            seconds = per_phase[phase]
            if not (seconds > 0).any():
                st.markdown(f"##### {title}")
                st.caption(f"No component time recorded for {title.lower()}.")
                continue
            st.plotly_chart(
                plot_phase_component_pie(
                    seconds, top_n=_PIE_TOP_N, colors=colors, title=title,
                    duration=phases.get(phase),
                ),
                width="stretch", key=f"reco_breakdown_{phase}_chart",
                config={"displayModeBar": False},
            )

    title_col, order_col = st.columns([3, 1], vertical_alignment="bottom")
    with title_col:
        st.markdown("##### Event loop, event by event")
    with order_col:
        order = st.segmented_control(
            "Order", ["Event order", "Longest first"], default="Event order",
            key="reco_breakdown_event_order", label_visibility="collapsed",
            bind="query-params",
        ) or "Event order"
    fig = plot_event_composition(
        timing.execute["wall_s"], top_components(per_phase["execute"], _PIE_TOP_N),
        colors=colors, sort=order == "Longest first", warm_up=_WARM_UP_EVENT,
    )
    fit_legend_below(fig, len(fig.data))
    st.plotly_chart(fig, width="stretch", key="reco_breakdown_events_chart")


def _render_components(ctx: PageContext) -> None:
    """Every algorithm and service, most expensive first, with its own time in
    each phase and links to its code."""
    if (selected := _select_config(ctx)) is None:
        return
    _, timing = selected
    per_phase = {phase: timing.phase_component_seconds(phase) for phase in _PHASE_TITLES}
    components = timing.components[timing.components["category"].isin(("algorithm", "service"))]
    if components.empty:
        return
    sources = source_columns(ctx.data_dir, timing, components.index)
    rows = []
    for name, component in components.iterrows():
        rows.append({
            "Component": name,
            "Class": component["impl"],
            "Kind": component["category"].capitalize(),
            **{title: float(per_phase[phase].get(name, 0.0) or 0.0)
               for phase, title in _PHASE_TITLES.items()},
            **sources.loc[name].to_dict(),
        })
    df = pd.DataFrame(rows)
    df = df.iloc[(-df[list(_PHASE_TITLES.values())].sum(axis=1)).argsort(kind="stable")]

    st.caption(
        "Each algorithm's and service's own wall time per phase, the event loop's "
        "summed over events, and the package its code comes from at the revision "
        "this run's stack built."
    )
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Component": st.column_config.TextColumn("Component", width="medium"),
            "Class": st.column_config.TextColumn(
                "Class", help="The class doing the work: the component's type, or the "
                              "Marlin processor a MarlinProcessorWrapper runs.",
            ),
            "Kind": st.column_config.TextColumn("Kind", width="small"),
            **{title: st.column_config.NumberColumn(title, format="%.2f s")
               for title in _PHASE_TITLES.values()},
            **SOURCE_COLUMN_CONFIG,
        },
    )
    unresolved = components[(components["impl"] != components["type"])
                            & (components["impl_library"] == "")]
    if not unresolved.empty:
        st.caption(
            f"{len(unresolved)} Marlin processor(s) have no package: the run did not "
            "record the library defining them (k4bench 0.0.49 and earlier recorded "
            "only the wrapper's)."
        )


#: What each component of a phase's history is drawn as.
_SCALES = {"Seconds": False, "Share of phase": True}
_CHARTS = {"Bars": False, "Lines": True}


def _render_historical(ctx: PageContext, reliability_slot, display_options_slot) -> None:
    trend = judgeable_trend(ctx, cached_load_trend_phases)
    if trend is None or trend.empty:
        st.info(
            "No phase timing in the selected window. Widen the trend window in "
            "the sidebar."
        )
        return
    trend = render_reliability_filter(
        trend, ctx.reliability, key="reco_breakdown_hist_exclude_unreliable",
        slot=reliability_slot,
    )
    if trend.empty:
        return
    config_col, phase_col, chart_col, scale_col = st.columns(
        [2, 3, 1.4, 1.8], vertical_alignment="bottom",
    )
    with config_col:
        config = st.selectbox(
            "Configuration", sorted(trend["label"].unique()), key="reco_breakdown_hist_config",
            bind="query-params",
        )
    rows = trend[trend["label"] == config]
    totals = rows[rows["component"].isna()]
    longest = totals.groupby("phase")["seconds"].max()
    phases = [p for p in PHASE_NAMES if longest.get(p, 0) >= MIN_PHASE_S]
    if not phases:
        st.info(f"No phase timing for **{config}** in the selected window.")
        return
    drop_stale_selection("reco_breakdown_hist_phase", phases)
    with phase_col:
        phase = st.segmented_control(
            "Phase", phases, default="initialize" if "initialize" in phases else phases[0],
            format_func=PHASE_NAMES.get, key="reco_breakdown_hist_phase", bind="query-params",
        ) or phases[0]
    components = rows[rows["phase"].eq(phase) & rows["component"].notna()]
    by_component = not components.empty
    # The phase's duration, whose part no component was charged for is a
    # series of its own.
    durations = totals[totals["phase"] == phase] if by_component else None
    lines, relative = True, False
    if by_component:
        with chart_col:
            lines = _CHARTS[st.segmented_control(
                "Chart", list(_CHARTS), default="Bars", key="reco_breakdown_hist_chart_type",
                bind="query-params",
            ) or "Bars"]
        with scale_col:
            relative = _SCALES[st.segmented_control(
                "Show", list(_SCALES), default="Seconds", key="reco_breakdown_hist_scale",
                bind="query-params",
            ) or "Seconds"]
    else:
        # A phase timed only as a whole (configure) is one line: its total.
        components = totals[totals["phase"] == phase].assign(component=PHASE_NAMES[phase])

    # The Top N chooses the series before either chart draws its popover, in
    # which it then sits; Streamlit has already restored the slider's value.
    top_n = int(st.session_state.get(_TOP_N_KEY, _TOP_N))
    series = phase_component_series(components, top_n, durations)
    top_n_control_ = top_n_control(_TOP_N_KEY, "components", "components", default=_TOP_N)
    stat, title, unit = (
        ("share_pct", "Share of the phase (%)", "%") if relative else ("seconds", "Time (s)", "s")
    )
    if lines:
        render_historical_trends(
            series, list(dict.fromkeys(series["label"])), [(stat, title)],
            std_col="", n_col_candidates=[], unit=unit, error_sources={},
            key_prefix="reco_breakdown_hist", display_options_slot=display_options_slot,
            extra_controls=(top_n_control_,) if by_component else (),
        )
        return
    display = display_options(
        top_n_control_,
        palette_control("reco_breakdown_hist_palette", top_n + 1),
        key_prefix="reco_breakdown_hist_display",
        slot=display_options_slot,
    )
    fig = plot_phase_history(series, palette=PALETTES[display["palette"]], relative=relative)
    fit_legend_below(fig, series["label"].nunique())
    st.plotly_chart(fig, width="stretch", key="reco_breakdown_hist_chart")


def render(ctx: PageContext) -> SectionScope | None:
    views = ["Current Run", "Components"] + (["Historical Trends"] if ctx.trends_enabled else [])
    drop_stale_selection("reco_breakdown_view", views)
    view, reliability_slot, display_options_slot = view_control_row(
        views, key="reco_breakdown_view",
    )
    if view == "Historical Trends":
        _render_historical(ctx, reliability_slot, display_options_slot)
        return TREND_WINDOW_SCOPE
    if view == "Components":
        _render_components(ctx)
    else:
        _render_current_run(ctx)
    return None
