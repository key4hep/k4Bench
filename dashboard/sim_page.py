"""The Simulation page: the ``sim`` suite's ddsim benchmarks.

It is the dashboard's default page, served at ``/``, which is where the
nightly mail's and pull-request comments' ``?tab=`` deep links for the sim
suite point. Its sections run broadest first — across detectors, over time —
then narrow into one run's internals, then the forensics for judging a number,
which is the order the questions are actually asked in.
"""
from __future__ import annotations

from data import (
    EVENT_SIZE,
    EVENT_SOURCE_FILES,
    EVENTS,
    REGIONS,
    cached_load_trend_event_timing,
    cached_load_trend_region_timing,
)
from k4bench.layout import SIM
from section_dispatch import SHARED_RENDERERS, PageContext, event_memory, history
from sections import ONE_RUN, SHARED_SCOPES, TREND_WINDOW_SCOPE, SectionScope
from suite_page import SuitePage, render_page
from tabs import correlations, event_timing, impact, logs, region_timing, trends


def _run_trends(ctx: PageContext) -> None:
    """The sidebar triple's history across every release in the window."""
    sel = ctx.remote
    trends.render(
        ctx.trend_results, ctx.reliability,
        data_url=ctx.config.data_url, suite=sel.ref.suite, detector=sel.detector,
        platform=sel.platform, sample=sel.sample,
    )


def _config_impact(ctx: PageContext) -> None:
    impact.render(ctx.run.results)


def _region_timing(ctx: PageContext) -> SectionScope | None:
    return region_timing.render(
        ctx.run.region_data,
        history(ctx, cached_load_trend_region_timing, REGIONS),
        ctx.trends_enabled,
    )


def _event_timing(ctx: PageContext) -> SectionScope | None:
    return event_timing.render(
        ctx.run.event_data,
        history(ctx, cached_load_trend_event_timing, EVENTS),
        ctx.trends_enabled, all_runs=ctx.window.all_runs,
    )


def _correlations(ctx: PageContext) -> SectionScope | None:
    """The run's per-event costs, its event time and each detector's time and
    steps, against what each event holds."""
    return correlations.render(
        ctx.run_dir(EVENT_SOURCE_FILES + EVENT_SIZE), ctx.run.results, ctx.run.event_data,
        region_data=ctx.run.region_data,
    )


def _logs(ctx: PageContext) -> None:
    """Per-config status and the log explorer."""
    logs.render(ctx.run.results, ctx.run_dir, ctx.run_meta)


SIM_PAGE = SuitePage(
    suite=SIM,
    sections=(
        "Overview", "Run Trends", "Regressions", "Stack Changes", "Config Impact",
        "Region Timing", "Event Timing", "Event Memory", "Correlations", "Event Size",
        "Machine Info", "Logs",
    ),
    # The trend views, the report-backed views and Stack Changes, which compares
    # two Key4hep releases. Config Impact compares the selected run's own
    # configurations, so it works locally too.
    remote_only=frozenset({"Overview", "Run Trends", "Regressions", "Stack Changes"}),
    scopes={
        **SHARED_SCOPES,
        "Run Trends": TREND_WINDOW_SCOPE,
        **dict.fromkeys(
            ("Config Impact", "Region Timing", "Event Timing", "Event Memory",
             "Correlations", "Logs"),
            ONE_RUN,
        ),
    },
    renderers={
        **SHARED_RENDERERS,
        "Run Trends": _run_trends,
        "Config Impact": _config_impact,
        "Region Timing": _region_timing,
        "Event Timing": _event_timing,
        "Event Memory": event_memory,
        "Correlations": _correlations,
        "Logs": _logs,
    },
    all_runs_sections=frozenset({"Event Timing", "Event Memory", "Event Size"}),
)


def main() -> None:
    render_page(SIM_PAGE)
