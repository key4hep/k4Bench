"""The section bar, and how each section is drawn from what the sidebar loaded.

:data:`RENDERERS` maps every :data:`sections.SECTION_NAMES` entry to a function
of one :class:`PageContext`. The tabs keep their own signatures; each entry
only says which part of the context its tab takes. Only the active section is
drawn, and the context loads nothing up front beyond the selected run's
results: each entry asks for the files its tab reads (:meth:`PageContext.run_dir`,
the :class:`data.RunData` attributes), and hands a history view a loader
called only once that view is open, so in remote mode a section downloads
what it draws and no more.

A fourth registry next to the three in :mod:`sections`, and kept apart from
them for the same reason: how a section is drawn is a separate fact from where
it sits in the bar, which data source it needs and what scope it honours.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property

import pandas as pd
import streamlit as st

from config import Config
from data import (
    EVENT_SIZE,
    EVENT_SOURCE_FILES,
    EVENTS,
    REGIONS,
    RUN_BASE,
    Artifacts,
    RunData,
    cached_load_trend_event_size,
    cached_load_trend_event_timing,
    cached_load_trend_machine_info,
    cached_load_trend_region_timing,
    cached_load_trend_results,
    load_machine_info,
)
from k4bench.analysis.loader import judgeable_config_rows
from k4bench.results.reliability_evidence import run_reliability_map
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
    """Everything the sidebar resolved, for the active section: the run and
    the window, whose files are read when the section first asks for them."""

    config: Config
    #: The sidebar's selection; ``None`` in local mode, which has no store
    #: hierarchy and offers none of the sections that read it.
    selection: RunSelection | None
    #: The directory of the run the single-run tabs show, holding its
    #: :data:`data.RUN_BASE` files; any other file is read from
    #: :meth:`run_dir`, which makes sure it is there.
    data_dir: str
    run_meta: dict | None
    run: RunData
    window: TrendWindow

    def run_dir(self, artifacts: Artifacts) -> str:
        """The run's directory once it holds the files *artifacts* match
        (:attr:`data.RunData.directory`)."""
        return self.run.directory(artifacts)

    @cached_property
    def trend_results(self) -> pd.DataFrame | None:
        """The window's results, which every trend view draws on; ``None``
        when the window holds none, as in local mode."""
        run_dirs = self.window.run_dirs(RUN_BASE)
        return cached_load_trend_results(run_dirs) if run_dirs else None

    @cached_property
    def trend_machine(self) -> pd.DataFrame | None:
        """The window's machine info; ``None`` when the window holds none."""
        run_dirs = self.window.run_dirs(RUN_BASE)
        return cached_load_trend_machine_info(run_dirs) if run_dirs else None

    @cached_property
    def reliability(self) -> dict[str, bool | None]:
        """Per-run reliability verdict (``{run_id: reliable}``) shared by every
        historical view, matching the Machine Info tab's per-run verdict;
        empty in local mode or when no machine info is available, in which
        case the filter it feeds is a no-op."""
        return run_reliability_map(self.trend_results, self.trend_machine)

    @property
    def labels(self) -> list[str]:
        """Every configuration with data on screen: the run's, or, when it has
        none — a failed or timed-out job, say, or ones retired from the
        newest release — the window's, whose history the trend views still
        draw."""
        if self.run.labels:
            return self.run.labels
        results = self.trend_results
        if results is None or "label" not in results.columns:
            return []
        return sorted(results["label"].dropna().unique())

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


#: A section's old name, and the section and the widget values it now opens
#: on: ``{old name: (section, {widget key: value})}``; the values may instead
#: be a function of the link's parameters, for a section whose own controls
#: were renamed.
Aliases = Mapping[
    str, tuple[str, Mapping[str, str] | Callable[[Mapping[str, str]], Mapping[str, str]]],
]


def current_section(
    section_names: list[str], key: str = "active_section", aliases: Aliases | None = None,
) -> str:
    """The active section, readable before the section bar is drawn — the
    sidebar, which is drawn first, scopes its trend window by it.

    ``?tab=...`` (used by the nightly regression email's "view in dashboard"
    links, see ``k4bench.regression.render.dashboard_link``) seeds
    session_state the same way as ``seed_query_param`` (see its docstring) —
    case-insensitively here, since section names are matched by label. A click
    on the bar is in session_state before the rerun it starts, so this already
    names the section being opened. A section's old name in *aliases* opens
    the section it is now part of, with the widget values that show what it
    showed (a view, say, or the configuration its own controls chose), so an
    old link still lands where it pointed.
    """
    if key not in st.session_state:
        requested = (st.query_params.get("tab") or "").strip().lower()
        matched = next((name for name in section_names if name.lower() == requested), None)
        alias = next(
            (target for name, target in (aliases or {}).items() if name.lower() == requested),
            None,
        )
        if matched is None and alias is not None:
            matched, values = alias
            if callable(values):
                values = values(st.query_params)
            for widget, value in values.items():
                st.session_state.setdefault(widget, value)
        st.session_state[key] = matched or section_names[0]
    # Clicking the active section again deselects it.
    return st.session_state[key] or section_names[0]


def pick_section(
    section_names: list[str], key: str = "active_section", aliases: Aliases | None = None,
) -> str:
    """The section bar, returning the active section (see :func:`current_section`).

    *key* holds the selection; each page passes its own, and it is kept for
    the whole session, so returning from another page reopens the section it
    left.
    """
    current_section(section_names, key, aliases)
    active = st.segmented_control(
        "Section", section_names,
        key=key, label_visibility="collapsed", width="stretch",
        persist_state="session",
    ) or section_names[0]
    st.query_params["tab"] = active
    return active


def _overview(ctx: PageContext, *, key_prefix: str = "") -> None:
    """Cross-detector comparison built from the nightly reports, scoped by the
    sidebar's platform, sample and trend window. Its Nightly Report view
    re-renders the e-group mail, whose deep links must be absolute — hence
    the dashboard's own URL. *key_prefix* keeps a page's figure choices its
    own."""
    sel = ctx.remote
    detectors_overview.render(
        ctx.config.data_url, sel.ref.suite, ctx.config.dashboard_url,
        sel.platform, sel.sample, ctx.window.dates, key_prefix=key_prefix,
    )


def _run_trends(ctx: PageContext) -> None:
    """The sidebar triple's history across every release in the window."""
    sel = ctx.remote
    trends.render(
        ctx.trend_results, ctx.reliability,
        data_url=ctx.config.data_url, detector=sel.detector,
        platform=sel.platform, sample=sel.sample,
    )


def _regressions(ctx: PageContext) -> SectionScope | None:
    """The precomputed nightly reports for the sidebar triple. The sidebar's
    release picks which report nights are on offer; the default is the most
    attention-worthy night, a picker exposes the rest and ``?report=`` pins one."""
    sel = ctx.remote
    return regressions.render(
        ctx.config.data_url, sel.ref.suite, ctx.config.cache_dir,
        sel.detector, sel.platform, sel.sample, sel.stack,
    )


def _stack_changes(ctx: PageContext) -> SectionScope | None:
    """The platform-wide package diff between two releases, then the
    regressions in that range for the sidebar's detector and sample. The
    sidebar stack seeds the comparison's newer end."""
    sel = ctx.remote
    return stack_changes.render(
        ctx.config.data_url, sel.ref.suite, ctx.config.cache_dir,
        sel.platform, sel.detector, sel.sample, sel.stack,
    )


def _config_impact(ctx: PageContext) -> None:
    impact.render(ctx.run.results)


def judgeable_trend(
    ctx: PageContext,
    load: Callable[[tuple[str, ...]], pd.DataFrame | None],
    artifacts: Artifacts,
) -> pd.DataFrame | None:
    """One trend frame for the window — *load* over the windowed runs once
    they hold the files *artifacts* match — its failed configs dropped."""
    run_dirs = ctx.window.run_dirs(artifacts)
    frame = load(run_dirs) if run_dirs else None
    return judgeable_config_rows(frame, ctx.trend_results)


def history(
    ctx: PageContext,
    load: Callable[[tuple[str, ...]], pd.DataFrame | None],
    artifacts: Artifacts,
) -> Callable[[], tuple[pd.DataFrame | None, dict[str, bool | None]]]:
    """What a tab's history view draws, loaded only once that view is open:
    the :func:`judgeable_trend` of *load* over *artifacts*, and the window's
    per-run reliability."""
    return lambda: (judgeable_trend(ctx, load, artifacts), ctx.reliability)


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
        history(ctx, cached_load_trend_region_timing, REGIONS),
        ctx.trends_enabled,
    )


def _event_timing(ctx: PageContext) -> SectionScope | None:
    return event_timing.render(
        ctx.run.event_data,
        history(ctx, cached_load_trend_event_timing, EVENTS),
        ctx.trends_enabled, all_runs=ctx.window.all_runs,
    )


def _event_memory(ctx: PageContext) -> SectionScope | None:
    return event_memory.render(
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


def _event_size(ctx: PageContext) -> SectionScope | None:
    """What each event of the run's output files takes on disk, and, in remote
    mode, its history across the window."""
    return event_size.render(
        ctx.run_dir(EVENT_SIZE), ctx.run.results,
        (lambda: judgeable_trend(ctx, cached_load_trend_event_size, EVENT_SIZE))
        if ctx.trends_enabled else None,
        all_runs=ctx.window.all_runs,
    )


def _machine_info(ctx: PageContext) -> SectionScope | None:
    return machine_info.render(
        load_machine_info(ctx.data_dir),
        run_meta=ctx.run_meta,
        results=ctx.run.results,
        load_trend_machine=lambda: ctx.trend_machine,
        load_trend_results=lambda: ctx.trend_results,
        trends_enabled=ctx.trends_enabled,
    )


def _logs(ctx: PageContext) -> None:
    """Per-config status and the log explorer."""
    logs.render(ctx.run.results, ctx.run_dir, ctx.run_meta)


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
