"""The Reconstruction page: the ``reco`` suite's k4run benchmarks.

Each section answers one question, and where it can be looked at two ways a
lens at its top chooses which (:func:`_lenses`):

- **Overview**: how the reco detectors compare over the window, as the
  Simulation Overview compares the simulated ones;
- **Time**: where the job's time goes, by phase and component (Breakdown), or
  event by event, as the Simulation page's Event Timing draws it, with any
  algorithm's own time beside the event's;
- **Memory**: where the heap goes, component by component, and what grows
  without being released, or the process's memory event by event, as the
  Simulation page's Event Memory draws it;
- **Components**: every algorithm on one map, and everything the run recorded
  of the one selected, which the other sections open by a click;
- **Data Flow**: which algorithm feeds which;
- **Correlations**: what an event's cost goes with, the Simulation page's
  section with each algorithm's costs on offer;
- **Event Size**: what the EDM4hep files the job wrote take per event, the
  Simulation page's section;
- **Machine Info** and **Logs**, the latter reading the log as Gaudi's.

The k4Bench auditor writes each label's ``_events.json`` in the ddsim event
format, so the shared per-event tabs draw a reco run as they draw a sim run.
"""
from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from data import (
    COMPONENTS,
    EVENT_SIZE,
    EVENT_SOURCE_FILES,
    EVENTS,
    cached_load_trend_algorithm_timing,
    cached_load_trend_event_timing,
)
from k4bench.layout import RECO
from reco import _inspect, breakdown, component, dataflow, memory, overview
from section_dispatch import (
    RENDERERS,
    PageContext,
    Renderer,
    history,
    judgeable_trend,
    warn_unreadable_components,
)
from sections import SECTION_SCOPE, SectionScope
from suite_page import SuitePage, render_page
from tabs import correlations, event_timing, logs
from ui_chrome import drop_stale_selection

#: The lenses of the sections with two, by their widget keys, the default first.
TIME_LENS = "reco_time_lens"
TIME_LENSES = ("Phases and components", "Per event")
MEMORY_LENS = "reco_memory_lens"
MEMORY_LENSES = ("Heap by component", "Per event")


def _efficiency_link(params: Mapping[str, str]) -> dict[str, str]:
    """The Components view and configuration a link to the former Efficiency
    section showed: its view, and the configuration of that view."""
    view = params.get("reco_efficiency_view")
    config = params.get(
        "reco_efficiency_hist_config" if view == "Historical Trends" else "reco_efficiency_config"
    )
    return {
        key: value
        for key, value in ((_inspect.VIEW_KEY, view), (_inspect.CONFIG_KEY, config))
        if value
    }


def _event_timing(ctx: PageContext) -> SectionScope | None:
    """The sim tab, with each algorithm's own time in each event on offer."""
    return event_timing.render(
        ctx.run.event_data,
        history(ctx, cached_load_trend_event_timing, EVENTS),
        ctx.trends_enabled,
        components=ctx.run.components,
        load_algorithm_trend=lambda: judgeable_trend(
            ctx, cached_load_trend_algorithm_timing, COMPONENTS,
        ),
        all_runs=ctx.window.all_runs,
    )


def _correlations(ctx: PageContext) -> SectionScope | None:
    """The sim tab, with each algorithm's costs in each event on offer."""
    return correlations.render(
        ctx.run_dir(EVENT_SOURCE_FILES + EVENT_SIZE), ctx.run.results, ctx.run.event_data,
        components=ctx.run.components,
    )


def _warning_unreadable_components(render: Renderer) -> Renderer:
    """*render*, first naming the run's configurations whose component costs
    could not be read."""
    def rendered(ctx: PageContext) -> SectionScope | None:
        warn_unreadable_components(ctx)
        return render(ctx)
    return rendered


def _lenses(key: str, lenses: dict[str, Renderer]) -> Renderer:
    """A section looked at in one of *lenses*' ways, chosen at its top (the
    first by default) and bound to the URL under *key*; each draws as it
    would as a section of its own, its views and their scope included."""
    names = list(lenses)

    def rendered(ctx: PageContext) -> SectionScope | None:
        drop_stale_selection(key, names)
        # An old link's section sets its lens before the widget exists; a
        # default beside it would be a second value.
        lens = st.segmented_control(
            "Show", names, default=None if key in st.session_state else names[0], key=key,
            bind="query-params", label_visibility="collapsed",
        ) or names[0]
        return lenses[lens](ctx)
    return rendered


def _logs(ctx: PageContext) -> None:
    """Per-config status and the log explorer, reading Gaudi's messages."""
    logs.render(ctx.run.results, ctx.run_dir, ctx.run_meta, log_format=logs.GAUDI)


RECO_PAGE = SuitePage(
    suite=RECO,
    sections=(
        "Overview", "Time", "Memory", "Components", "Data Flow", "Correlations",
        "Event Size", "Machine Info", "Logs",
    ),
    # Like the Simulation Overview, it compares detectors across the window.
    remote_only=frozenset({"Overview"}),
    scopes={
        "Overview": SECTION_SCOPE["Overview"],
        **{name: SECTION_SCOPE["Region Timing"]
           for name in ("Time", "Memory", "Components", "Data Flow")},
        **{name: SECTION_SCOPE[name]
           for name in ("Correlations", "Event Size", "Machine Info", "Logs")},
    },
    renderers={
        "Overview": overview.render,
        "Time": _warning_unreadable_components(_lenses(TIME_LENS, dict(zip(TIME_LENSES, (
            breakdown.render, _event_timing,
        ))))),
        "Memory": _lenses(MEMORY_LENS, dict(zip(MEMORY_LENSES, (
            _warning_unreadable_components(memory.render), RENDERERS["Event Memory"],
        )))),
        "Components": _warning_unreadable_components(component.render),
        "Data Flow": _warning_unreadable_components(dataflow.render),
        "Correlations": _warning_unreadable_components(_correlations),
        "Event Size": RENDERERS["Event Size"],
        "Machine Info": RENDERERS["Machine Info"],
        "Logs": _logs,
    },
    key_prefix="reco_",
    # Its sections read no nightly report yet, so its runs alone bound the window.
    reports=False,
    components=True,
    all_runs_sections=frozenset({"Time", "Memory", "Components", "Event Size"}),
    # The sections the page had before its sections were grouped by question,
    # which links may still name.
    aliases={
        "Breakdown": ("Time", {TIME_LENS: TIME_LENSES[0]}),
        "Event Timing": ("Time", {TIME_LENS: TIME_LENSES[1]}),
        "Event Memory": ("Memory", {MEMORY_LENS: MEMORY_LENSES[1]}),
        "Efficiency": ("Components", _efficiency_link),
        "Component": ("Components", {}),
        "Events": ("Correlations", {}),
    },
)


def main() -> None:
    render_page(RECO_PAGE)
