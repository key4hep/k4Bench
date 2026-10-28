"""The section bar, and how a section is drawn from what the sidebar loaded.

A page draws each section with a :data:`Renderer`: a function of one
:class:`PageContext`. The tabs keep their own signatures; each renderer only
says which part of the context its tab takes. Only the active section is
drawn, and the context loads nothing up front beyond the selected run's
results: each renderer asks for the files its tab reads
(:meth:`PageContext.run_dir`, the :class:`data.RunData` attributes), and hands
a history view a loader called only once that view is open, so in remote mode
a section downloads what it draws and no more.

The renderers here are those of the sections every page that has them draws
alike (:data:`SHARED_RENDERERS`), and the helpers a page's own renderers are
built from; each page defines the rest (:mod:`sim_page`, :mod:`reco.page`).
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
    EVENTS,
    RUN_BASE,
    Artifacts,
    RunData,
    cached_load_trend_event_size,
    cached_load_trend_event_timing,
    cached_load_trend_machine_info,
    cached_load_trend_results,
    load_machine_info,
)
from k4bench.analysis.loader import judgeable_config_rows
from k4bench.results.reliability_evidence import run_reliability_map
from sections import SectionScope
from sidebar import RunSelection, TrendWindow
from tabs import event_memory as event_memory_tab
from tabs import event_size, machine_info
from tabs.reports import overview, regressions, stack_changes


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
    #: The prefix of the page's widget keys (:attr:`suite_page.SuitePage.key_prefix`),
    #: which a section every page draws puts on its own, so that each page
    #: keeps its choices.
    key_prefix: str = ""

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


def _overview(ctx: PageContext) -> None:
    """Cross-detector comparison built from the nightly reports, scoped by the
    sidebar's platform, sample and trend window. Its Nightly Report view
    re-renders the e-group mail, whose deep links must be absolute — hence
    the dashboard's own URL."""
    sel = ctx.remote
    overview.render(
        ctx.config.data_url, sel.ref.suite, ctx.config.dashboard_url,
        sel.platform, sel.sample, ctx.window.dates, key_prefix=ctx.key_prefix,
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


def event_memory(ctx: PageContext) -> SectionScope | None:
    """The process's memory event by event: the Simulation page's Event Memory
    section, and the Reconstruction page's Memory looked at per event."""
    return event_memory_tab.render(
        ctx.run.event_data,
        history(ctx, cached_load_trend_event_timing, EVENTS),
        ctx.trends_enabled, all_runs=ctx.window.all_runs,
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


#: How each section every page that has it draws alike is drawn, by its name
#: there (their scopes are :data:`sections.SHARED_SCOPES`). A section that
#: offers a control over its own scope returns the scope it rendered, for the
#: note under the section bar.
SHARED_RENDERERS: dict[str, Renderer] = {
    "Overview": _overview,
    "Regressions": _regressions,
    "Stack Changes": _stack_changes,
    "Event Size": _event_size,
    "Machine Info": _machine_info,
}
