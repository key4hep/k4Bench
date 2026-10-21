"""The section bar, and how each section is drawn from what the sidebar loaded.

:data:`RENDERERS` maps every :data:`sections.SECTION_NAMES` entry to a function
of one :class:`PageContext`. The tabs keep their own signatures; each entry
only says which part of the context its tab takes. Only the active section is
drawn, so a section's own heavy data (the per-event trends) is loaded inside
its entry rather than up front.

A fourth registry next to the three in :mod:`sections`, and kept apart from
them for the same reason: how a section is drawn is a separate fact from where
it sits in the bar, which data source it needs and what scope it honours.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from config import Config
from data import (
    RunData,
    cached_load_trend_event_size,
    cached_load_trend_event_timing,
    cached_load_trend_region_timing,
    load_machine_info,
)
from k4bench.analysis.loader import judgeable_config_rows
from sections import SectionScope
from sidebar import RunSelection, TrendWindow
from tabs import (
    correlations,
    detectors_overview,
    event_memory,
    event_size,
    event_timing,
    impact,
    logs,
    machine_info,
    region_timing,
    regressions,
    stack_changes,
    trends,
)


@dataclass(frozen=True)
class PageContext:
    """Everything the sidebar resolved and loaded, for the active section."""

    config: Config
    #: The sidebar's selection; ``None`` in local mode, which has no store
    #: hierarchy and offers none of the sections that read it.
    selection: RunSelection | None
    #: The run the single-run tabs show.
    data_dir: str
    run_meta: dict | None
    run: RunData
    window: TrendWindow
    trend_machine: pd.DataFrame | None
    #: Per-run reliability verdict (``{run_id: reliable}``) shared by every
    #: historical view; empty in local mode.
    reliability: dict

    @property
    def trends_enabled(self) -> bool:
        """Whether the trend views exist: remote mode, whatever the window holds."""
        return bool(self.config.data_url)

    @property
    def remote(self) -> RunSelection:
        """:attr:`selection`, for the remote-only sections that always have one."""
        if self.selection is None:
            raise RuntimeError("a remote-only section was drawn in local mode")
        return self.selection


#: Draws one section from the page's context, returning the scope it rendered
#: when a control inside it changed that scope.
Renderer = Callable[[PageContext], SectionScope | None]


def current_section(section_names: list[str], key: str = "active_section") -> str:
    """The active section, readable before the section bar is drawn — the
    sidebar, which is drawn first, scopes its trend window by it.

    ``?tab=...`` (used by the nightly regression email's "view in dashboard"
    links, see ``k4bench.regression.render.dashboard_link``) seeds
    session_state the same way as ``seed_query_param`` (see its docstring) —
    case-insensitively here, since section names are matched by label. A click
    on the bar is in session_state before the rerun it starts, so this already
    names the section being opened.
    """
    if key not in st.session_state:
        requested = (st.query_params.get("tab") or "").strip().lower()
        matched = next((name for name in section_names if name.lower() == requested), None)
        st.session_state[key] = matched or section_names[0]
    # Clicking the active section again deselects it.
    return st.session_state[key] or section_names[0]


def pick_section(section_names: list[str], key: str = "active_section") -> str:
    """The section bar, returning the active section (see :func:`current_section`).

    *key* holds the selection; each page passes its own, and it is kept for
    the whole session, so returning from another page reopens the section it
    left.
    """
    current_section(section_names, key)
    active = st.segmented_control(
        "Section", section_names,
        key=key, label_visibility="collapsed", width="stretch",
        persist_state="session",
    ) or section_names[0]
    st.query_params["tab"] = active
    return active


def _overview(ctx: PageContext) -> None:
    """Cross-detector comparison built from the nightly reports, scoped by the
    sidebar's platform, sample and trend window. Its Nightly Report view
    re-renders the e-group mail, whose deep links must be absolute — hence
    the dashboard's own URL."""
    sel = ctx.remote
    detectors_overview.render(
        ctx.config.data_url, ctx.config.dashboard_url,
        sel.platform, sel.sample, ctx.window.dates,
    )


def _run_trends(ctx: PageContext) -> None:
    """The sidebar triple's history across every release in the window."""
    sel = ctx.remote
    trends.render(
        ctx.run.trend_results, ctx.reliability,
        data_url=ctx.config.data_url, detector=sel.detector,
        platform=sel.platform, sample=sel.sample,
    )


def _regressions(ctx: PageContext) -> SectionScope | None:
    """The precomputed nightly reports for the sidebar triple. The sidebar's
    release picks which report nights are on offer; the default is the most
    attention-worthy night, a picker exposes the rest and ``?report=`` pins one."""
    sel = ctx.remote
    return regressions.render(
        ctx.config.data_url, ctx.config.cache_dir,
        sel.detector, sel.platform, sel.sample, sel.stack,
    )


def _stack_changes(ctx: PageContext) -> SectionScope | None:
    """The platform-wide package diff between two releases, then the
    regressions in that range for the sidebar's detector and sample. The
    sidebar stack seeds the comparison's newer end."""
    sel = ctx.remote
    return stack_changes.render(
        ctx.config.data_url, ctx.config.cache_dir,
        sel.platform, sel.detector, sel.sample, sel.stack,
    )


def _config_impact(ctx: PageContext) -> None:
    impact.render(ctx.run.results)


def judgeable_trend(
    ctx: PageContext, load: Callable[[tuple[str, ...]], pd.DataFrame | None]
) -> pd.DataFrame | None:
    """One per-event trend frame for the window, its failed configs dropped."""
    frame = load(ctx.window.run_dirs) if ctx.window.run_dirs else None
    return judgeable_config_rows(frame, ctx.run.trend_results)


def warn_unreadable_components(ctx: PageContext) -> None:
    """Name each configuration of the run whose ``_components.json`` could
    not be read, and why: it is drawn without its component costs."""
    unreadable = ctx.run.unreadable_components
    if not unreadable:
        return
    reasons = "\n".join(f"- **{label}**: {reason}" for label, reason in sorted(unreadable.items()))
    st.warning(
        "The component costs of these configurations could not be read, so "
        f"they are not offered:\n\n{reasons}"
    )


def no_components_reason(ctx: PageContext) -> str:
    """Why the run offers no configuration's component costs: none that
    succeeded left a ``_components.json``, or none it left could be read."""
    if ctx.run.unreadable_components:
        return ("no configuration that succeeded left a `_components.json` that "
                "could be read (see the warning above).")
    return "no configuration that succeeded left a `_components.json`."


def _region_timing(ctx: PageContext) -> SectionScope | None:
    return region_timing.render(
        ctx.run.region_data,
        judgeable_trend(ctx, cached_load_trend_region_timing),
        ctx.trends_enabled, ctx.reliability,
    )


def _event_timing(ctx: PageContext) -> SectionScope | None:
    return event_timing.render(
        ctx.run.event_data,
        judgeable_trend(ctx, cached_load_trend_event_timing),
        ctx.trends_enabled, ctx.reliability,
    )


def _event_memory(ctx: PageContext) -> SectionScope | None:
    return event_memory.render(
        ctx.run.event_data,
        judgeable_trend(ctx, cached_load_trend_event_timing),
        ctx.trends_enabled, ctx.reliability,
    )


def _correlations(ctx: PageContext) -> SectionScope | None:
    """The run's per-event costs, its event time and each detector's time and
    steps, against what each event holds."""
    return correlations.render(
        ctx.data_dir, ctx.run.results, ctx.run.event_data, region_data=ctx.run.region_data,
    )


def _event_size(ctx: PageContext) -> SectionScope | None:
    """What each event of the run's output files takes on disk, and, in remote
    mode, its history across the window."""
    return event_size.render(
        ctx.data_dir, ctx.run.results,
        (lambda: judgeable_trend(ctx, cached_load_trend_event_size)) if ctx.trends_enabled
        else None,
    )


def _machine_info(ctx: PageContext) -> SectionScope | None:
    return machine_info.render(
        load_machine_info(ctx.data_dir),
        run_meta=ctx.run_meta,
        results=ctx.run.results,
        trend_machine_df=ctx.trend_machine,
        trend_results_df=ctx.run.trend_results,
        trends_enabled=ctx.trends_enabled,
    )


def _logs(ctx: PageContext) -> None:
    """Per-config status and the log explorer."""
    logs.render(ctx.run.results, ctx.data_dir, ctx.run_meta)


#: How each section is drawn. A section that offers a control over its own
#: scope returns the scope it rendered, for the note under the section bar.
RENDERERS: dict[str, Renderer] = {
    "Overview": _overview,
    "Run Trends": _run_trends,
    "Regressions": _regressions,
    "Stack Changes": _stack_changes,
    "Config Impact": _config_impact,
    "Region Timing": _region_timing,
    "Event Timing": _event_timing,
    "Event Memory": _event_memory,
    "Correlations": _correlations,
    "Event Size": _event_size,
    "Machine Info": _machine_info,
    "Logs": _logs,
}
