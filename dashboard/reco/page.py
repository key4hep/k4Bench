"""The Reconstruction page: the ``reco`` suite's k4run benchmarks.

The k4Bench auditor writes each label's ``_events.json`` in the ddsim event
format, so the per-event timing and memory tabs and Machine Info draw a reco
run as they draw a sim run, and are reused here, Event Timing with a selector
for one algorithm's own time in each event besides the whole event's. Logs is
reused too, reading the log as Gaudi's. The Overview compares the reco detectors as
the Simulation Overview does, the Breakdown splits the job's time by phase
and by component, Efficiency sets each algorithm's time against how well it
uses the CPU, Memory shows where the heap goes component by component and
what grows without being released, and the Data Flow draws which algorithm
feeds which. Correlations, the sim tab with each algorithm's costs on offer,
sets them against what each event holds. Event Size, shared too, reads the
EDM4hep files the job wrote.
"""
from __future__ import annotations

from data import (
    COMPONENTS,
    EVENT_SIZE,
    EVENT_SOURCE_FILES,
    EVENTS,
    cached_load_trend_algorithm_timing,
    cached_load_trend_event_timing,
)
from k4bench.layout import RECO
from reco import breakdown, dataflow, efficiency, memory, overview
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

#: The sim sections this page shares as they are, which read nothing sim-specific.
_SHARED = ("Event Memory", "Event Size", "Machine Info")


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


def _logs(ctx: PageContext) -> None:
    """Per-config status and the log explorer, reading Gaudi's messages."""
    logs.render(ctx.run.results, ctx.run_dir, ctx.run_meta, log_format=logs.GAUDI)


RECO_PAGE = SuitePage(
    suite=RECO,
    sections=(
        "Overview", "Breakdown", "Efficiency", "Memory", "Data Flow", "Event Timing",
        "Event Memory", "Correlations", "Event Size", "Machine Info", "Logs",
    ),
    # Like the Simulation Overview, it compares detectors across the window.
    remote_only=frozenset({"Overview"}),
    scopes={
        "Overview": SECTION_SCOPE["Overview"],
        "Breakdown": SECTION_SCOPE["Region Timing"],
        "Efficiency": SECTION_SCOPE["Region Timing"],
        "Memory": SECTION_SCOPE["Region Timing"],
        "Data Flow": SECTION_SCOPE["Region Timing"],
        **{name: SECTION_SCOPE[name]
           for name in ("Event Timing", "Correlations", *_SHARED, "Logs")},
    },
    renderers={
        "Overview": overview.render,
        # The sections that read the run's component costs.
        **{name: _warning_unreadable_components(render) for name, render in (
            ("Breakdown", breakdown.render),
            ("Efficiency", efficiency.render),
            ("Memory", memory.render),
            ("Data Flow", dataflow.render),
            ("Event Timing", _event_timing),
            ("Correlations", _correlations),
        )},
        **{name: RENDERERS[name] for name in _SHARED},
        "Logs": _logs,
    },
    key_prefix="reco_",
    # No nightly report judges the reco suite yet.
    reports=False,
    components=True,
    all_runs_sections=frozenset({
        "Breakdown", "Efficiency", "Memory", "Event Timing", "Event Memory", "Event Size",
    }),
)


def main() -> None:
    render_page(RECO_PAGE)
