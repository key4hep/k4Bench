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
feeds which.
"""
from __future__ import annotations

from data import cached_load_trend_algorithm_timing, cached_load_trend_event_timing
from k4bench.layout import RECO
from reco import breakdown, dataflow, efficiency, memory, overview
from section_dispatch import RENDERERS, PageContext, judgeable_trend
from sections import SECTION_SCOPE, SectionScope
from suite_page import SuitePage, render_page
from tabs import event_timing, logs

#: The sim sections this page shares as they are, which read nothing sim-specific.
_SHARED = ("Event Memory", "Machine Info")


def _event_timing(ctx: PageContext) -> SectionScope | None:
    """The sim tab, with each algorithm's own time in each event on offer."""
    return event_timing.render(
        ctx.run.event_data,
        judgeable_trend(ctx, cached_load_trend_event_timing),
        ctx.trends_enabled, ctx.reliability,
        components=ctx.run.components,
        load_algorithm_trend=lambda: judgeable_trend(ctx, cached_load_trend_algorithm_timing),
    )


def _logs(ctx: PageContext) -> None:
    """Per-config status and the log explorer, reading Gaudi's messages."""
    logs.render(ctx.run.results, ctx.data_dir, ctx.run_meta, log_format=logs.GAUDI)


RECO_PAGE = SuitePage(
    suite=RECO,
    sections=(
        "Overview", "Breakdown", "Efficiency", "Memory", "Data Flow", "Event Timing", *_SHARED,
        "Logs",
    ),
    # Like the Simulation Overview, it compares detectors across the window.
    remote_only=frozenset({"Overview"}),
    scopes={
        "Overview": SECTION_SCOPE["Overview"],
        "Breakdown": SECTION_SCOPE["Region Timing"],
        "Efficiency": SECTION_SCOPE["Region Timing"],
        "Memory": SECTION_SCOPE["Region Timing"],
        "Data Flow": SECTION_SCOPE["Region Timing"],
        **{name: SECTION_SCOPE[name] for name in ("Event Timing", *_SHARED, "Logs")},
    },
    renderers={
        "Overview": overview.render,
        "Breakdown": breakdown.render,
        "Efficiency": efficiency.render,
        "Memory": memory.render,
        "Data Flow": dataflow.render,
        "Event Timing": _event_timing,
        **{name: RENDERERS[name] for name in _SHARED},
        "Logs": _logs,
    },
    key_prefix="reco_",
    # No nightly report judges the reco suite yet.
    reports=False,
    components=True,
)


def main() -> None:
    render_page(RECO_PAGE)
