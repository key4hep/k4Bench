"""The dashboard's entry point: resolve the run in the sidebar, then draw the
active section.

In remote mode the sidebar walks the WebEOS run store (:mod:`sidebar`); in local
mode it reads one run directory. Either way the selected run is loaded once
(:func:`data.load_run`) and handed to the active section through a
:class:`section_dispatch.PageContext`.
"""
from __future__ import annotations

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
from k4bench.results.reliability_evidence import run_reliability_map
from section_dispatch import RENDERERS, PageContext, pick_section
from sections import visible_sections
from sidebar import RunSelection, TrendWindow, pick_run, pick_trend_window
from tabs._reliability import render_sidebar_run_quality
from ui_chrome import (
    _render_footer,
    _render_sidebar_footer,
    configure_page,
    render_run_status,
    render_scope_note,
)


def main() -> None:
    configure_page()
    config = Config.from_env()

    # Local mode has no store hierarchy above its one run directory, and no
    # trend history.
    selection: RunSelection | None = None
    run_meta: dict | None = None
    window = TrendWindow()
    run: RunData | None = None

    with st.sidebar:
        st.header("Data Source")
        if st.button("Refresh Data", width="content"):
            # Deliberately NO st.rerun() here. This button renders *before* the
            # Detector/Platform/Sample/Stack selectboxes below, so rerunning now
            # would abort the run before those widgets are registered — Streamlit
            # then garbage-collects their session_state as "stale" and the
            # dropdowns snap back to their defaults (the bug where Refresh showed
            # the wrong sample). Instead we just clear the caches and let the run
            # continue: the selectboxes re-render (keeping the current selection)
            # and the now-empty caches are repopulated with fresh data below.
            st.cache_data.clear()

        if config.data_url:
            selection = pick_run(config.data_url, config.cache_dir)
            if selection is None:
                return
            data_dir = selection.data_dir
            run_meta = run_metadata(data_dir)
            # Run-quality card for the selected release's latest run, right under
            # the stack selector (loads are cached, so the reuse below is free).
            render_sidebar_run_quality(
                load_machine_info(data_dir), cached_load_results(data_dir)
            )
            window = pick_trend_window(config.data_url, selection, config.cache_dir)
        else:
            data_dir = st.text_input("Data directory", value=config.data_dir)

        if not (data_dir and Path(data_dir).is_dir()):
            st.error(
                f"Data directory not found: '{data_dir}'. "
                "Check for typos or a missing directory."
            )
        else:
            run = load_run(data_dir, window.run_dirs)
            # Remote mode shows the run-quality card under the Stack selector.
            if not config.data_url:
                render_sidebar_run_quality(load_machine_info(data_dir), run.results)
            if not run.labels:
                st.warning("No benchmark data found in the specified directory.")

        _render_sidebar_footer()

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
    # reliability map. The region- and event-timing trends are built (or, once
    # cached, deep-copied by st.cache_data) only when their own tab is active,
    # so switching between the other tabs never pays for the heaviest frame
    # (per-event timing across the whole window).
    trend_machine = (
        cached_load_trend_machine_info(window.run_dirs) if window.run_dirs else None
    )
    # Per-run reliability verdict, computed once from the full trend so every
    # historical tab shares one consistent warning / exclude filter that matches
    # the Machine Info tab's per-run verdict. Empty in local mode or when no
    # machine info is available, in which case the filter is a no-op.
    reliability = run_reliability_map(run.trend_results, trend_machine)

    # Trends-capable sections are gated on *remote mode*, not on whether the
    # current trend window happens to have data. This keeps the section set and
    # each one's view selector stable across trend-window changes, so the active
    # section / sub-view is preserved when the user only adjusts the window; an
    # empty window shows an in-view "widen the window" message instead of
    # removing the option.
    active_section = pick_section(visible_sections(bool(config.data_url)))

    # One muted line naming what the active section is actually showing. Driven
    # by sections.SECTION_SCOPE, so it states only the dimensions that section
    # honours — Overview spans every detector, Stack Changes' diff is
    # platform-wide — and never the time reference, which each tab prints itself.
    # Only its position is reserved here; it is written after the section body,
    # which may itself change the scope (see the fill below).
    scope_note = st.empty()

    scope_override = RENDERERS[active_section](PageContext(
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
    )

    _render_footer()


main()
