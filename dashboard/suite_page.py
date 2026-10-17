"""One page of the dashboard per suite: resolve the run in the sidebar, then
draw the active section.

Every suite's page is the same skeleton. In remote mode the sidebar walks the
suite's run tree on WebEOS (:mod:`sidebar`); in local mode it reads one run
directory. Either way the selected run is loaded once (:func:`data.load_run`)
and handed to the active section through a
:class:`section_dispatch.PageContext`. What differs between the suites — their
sections, the scope each honours, how each is drawn — is a :class:`SuitePage`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from config import Config
from data import (
    RunData,
    cached_load_results,
    cached_load_trend_machine_info,
    load_machine_info,
    load_run,
    run_metadata,
)
from k4bench.analysis.loader import run_suite
from k4bench.layout import RECO, SIM
from k4bench.results.reliability_evidence import run_reliability_map
from section_dispatch import Renderer, PageContext, current_section, pick_section
from sections import SCOPED, SectionScope
from sidebar import (
    RunSelection,
    TrendWindow,
    pick_run,
    pick_trend_window,
    render_refresh_button,
)
from tabs._reliability import render_sidebar_run_quality
from ui_chrome import (
    render_footer,
    render_run_status,
    render_scope_note,
    render_sidebar_footer,
)


#: Each suite's page title, as the top bar names it.
SUITE_TITLES = {SIM: "Simulation", RECO: "Reconstruction"}


@dataclass(frozen=True)
class SuitePage:
    """What one suite's page offers, and how the store is read for it."""

    #: The suite whose run tree the sidebar walks (``k4bench.layout.SUITES``).
    suite: str
    #: Every section, in display order.
    sections: tuple[str, ...]
    #: Sections that need the run store's history and are hidden in local mode.
    remote_only: frozenset[str]
    #: The slice of the sidebar each section honours (see :data:`sections.SECTION_SCOPE`).
    scopes: dict[str, SectionScope]
    #: How each section is drawn (see :data:`section_dispatch.RENDERERS`).
    renderers: dict[str, Renderer]
    #: Prefix of the page's own widget keys, so two pages never share a selection.
    key_prefix: str = ""
    #: Whether the suite has nightly reports, which bound the trend window.
    reports: bool = True
    #: Whether its sections read the run's per-component costs.
    components: bool = False

    @property
    def title(self) -> str:
        """The page's name in the top bar."""
        return SUITE_TITLES[self.suite]

    def visible_sections(self, trends_enabled: bool) -> list[str]:
        """The sections to offer, in display order.

        Gated on *remote mode* rather than on whether the current trend window
        happens to have data: that keeps the section set stable as the window
        changes, so the active section survives a window tweak, and an empty
        window shows an in-view "widen the window" message instead of a
        vanishing tab.
        """
        return [s for s in self.sections if trends_enabled or s not in self.remote_only]


def render_page(page: SuitePage) -> None:
    """Draw *page*: its sidebar, the selected run's status, then the active section."""
    config = Config.from_env()

    # Local mode has no store hierarchy above its one run directory, and no
    # trend history.
    selection: RunSelection | None = None
    run_meta: dict | None = None
    window = TrendWindow()
    run: RunData | None = None
    # The suite of a local directory that holds another suite's run.
    foreign_suite: str | None = None
    # Trends-capable sections are gated on *remote mode*, not on whether the
    # current trend window happens to have data. This keeps the section set and
    # each one's view selector stable across trend-window changes, so the active
    # section / sub-view is preserved when the user only adjusts the window; an
    # empty window shows an in-view "widen the window" message instead of
    # removing the option.
    sections = page.visible_sections(bool(config.data_url))
    section_key = f"{page.key_prefix}active_section"

    with st.sidebar:
        st.header("Data Source")
        render_refresh_button()

        if config.data_url:
            selection = pick_run(
                config.data_url, config.cache_dir,
                suite=page.suite, key_prefix=page.key_prefix,
            )
            if selection is None:
                return
            data_dir = selection.data_dir
            run_meta = run_metadata(data_dir)
            # Run-quality card for the selected release's latest run, right under
            # the stack selector (loads are cached, so the reuse below is free).
            render_sidebar_run_quality(
                load_machine_info(data_dir), cached_load_results(data_dir)
            )
            # A section that spans every detector draws the window over every
            # release on the platform; the others read the selected detector's.
            across_detectors = (
                page.scopes[current_section(sections, key=section_key)].detector
                is not SCOPED
            )
            window = pick_trend_window(
                config.data_url, selection, config.cache_dir, across_detectors,
                suite=page.suite, reports=page.reports,
            )
        else:
            # Kept for the session, so each page returns to its own directory.
            data_dir = st.text_input(
                "Data directory", value=config.data_dir,
                key=f"{page.key_prefix}data_dir", persist_state="session",
            )

        if not (data_dir and Path(data_dir).is_dir()):
            st.error(
                f"Data directory not found: '{data_dir}'. "
                "Check for typos or a missing directory."
            )
        # A remote page only ever walks its own suite's tree; a local directory
        # can hold any suite's run, which this page must not present as its own.
        elif not config.data_url and (suite := run_suite(Path(data_dir))) != page.suite:
            foreign_suite = suite
        else:
            run = load_run(data_dir, window.run_dirs, components=page.components)
            # Remote mode shows the run-quality card under the Stack selector.
            if not config.data_url:
                render_sidebar_run_quality(load_machine_info(data_dir), run.results)
            if not run.labels:
                st.warning("No benchmark data found in the specified directory.")

        render_sidebar_footer()

    if foreign_suite is not None:
        st.warning(
            f"**{data_dir}** holds a {SUITE_TITLES[foreign_suite].lower()} run. "
            f"Open it on the {SUITE_TITLES[foreign_suite]} page; this page shows "
            f"{page.title.lower()} runs only."
        )
        return

    if run is None or not run.labels:
        path_hint = f" in **{data_dir}**" if data_dir else ""
        st.info(
            f"No benchmark results found{path_hint}. "
            "Set the `K4BENCH_DATA_DIR` environment variable or update the path in the sidebar."
        )
        return

    # Run status banners (per-config detail + logs live in the Logs tab; the
    # per-run reliability status lives in the sidebar run-quality card).
    render_run_status(run.results, run_meta)

    # Of the window's trend frames, only results (loaded with the run) and
    # machine info are loaded on every rerun: together they make the shared
    # reliability map. The heavier trends are built (or, once cached,
    # deep-copied by st.cache_data) only when their own tab is active, so
    # switching between the other tabs never pays for the heaviest frame
    # (per-event timing across the whole window).
    trend_machine = (
        cached_load_trend_machine_info(window.run_dirs) if window.run_dirs else None
    )
    # Per-run reliability verdict, computed once from the full trend so every
    # historical tab shares one consistent warning / exclude filter that matches
    # the Machine Info tab's per-run verdict. Empty in local mode or when no
    # machine info is available, in which case the filter is a no-op.
    reliability = run_reliability_map(run.trend_results, trend_machine)

    active_section = pick_section(sections, key=section_key)

    # One muted line naming what the active section is actually showing. Driven
    # by the page's scope registry, so it states only the dimensions that
    # section honours — Overview spans every detector, Stack Changes' diff is
    # platform-wide — and never the time reference, which each tab prints itself.
    # Only its position is reserved here; it is written after the section body,
    # which may itself change the scope (see the fill below).
    scope_note = st.empty()

    scope_override = page.renderers[active_section](PageContext(
        config=config,
        selection=selection,
        data_dir=data_dir,
        run_meta=run_meta,
        run=run,
        window=window,
        trend_machine=trend_machine,
        reliability=reliability,
    ))

    # Fill the slot reserved above the section. A section that offers a control
    # over its own scope — a historical sub-view, Stack Changes' "Whole platform"
    # toggle — returns the scope it actually rendered, which is knowable only
    # once its body has run; hence the note is composed last and written back
    # into its earlier position.
    render_scope_note(
        active_section,
        detector=selection.detector if selection else None,
        platform=selection.platform if selection else None,
        sample=selection.sample if selection else None,
        release=selection.stack if selection else None,
        data_dir=data_dir, override=scope_override, slot=scope_note,
        scopes=page.scopes,
    )

    render_footer()
