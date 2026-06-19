from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import Config
from data import (
    cached_load_event_timing,
    cached_load_region_timing,
    cached_load_results,
    cached_load_trend_results,
    cached_load_trend_region_timing,
    cached_load_trend_event_timing,
    cached_load_trend_machine_info,
    collect_labels,
    load_machine_info,
    run_metadata,
)
from remote_cache import (
    _cached_fetch_latest_run,
    _cached_fetch_runs_windowed,
    _cached_list_detectors,
    _cached_list_platforms,
    _cached_list_run_dates,
    _cached_scan_stack_samples,
)
from tabs import event_memory, event_timing, impact, machine_info, region_timing, trends
from trend_window import WINDOW_PRESETS, resolve_window
from ui_chrome import (
    _drop_stale_selection,
    _render_footer,
    _render_sidebar_footer,
    render_logs_tab,
    render_run_status,
)


def _force_plotly_relayout_on_tab_switch() -> None:
    """Make Plotly charts relayout when their ``st.tabs`` panel becomes visible.

    Streamlit renders every tab panel into the DOM at once and hides the inactive
    ones with ``display:none``. A chart that first lays out inside a hidden,
    zero-width panel computes a degenerate layout (e.g. a horizontal legend wrapped
    onto many rows that then overlaps the plot). Plotly only re-lays-out on a
    window ``resize`` event, but tab switches are purely client-side and never fire
    one — so the bad layout persists until the user manually resizes the window.

    This injects a one-off script that listens for tab-button clicks and dispatches
    a synthetic ``resize`` shortly after the panel is revealed, forcing every chart
    to relayout at its real width. The script reaches the parent document from a
    same-origin component iframe; it is idempotent (each button is bound once).
    """
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          function bind() {
            doc.querySelectorAll('button[data-baseweb="tab"]').forEach(function (btn) {
              if (btn.dataset.k4ResizeBound) return;
              btn.dataset.k4ResizeBound = "1";
              btn.addEventListener("click", function () {
                // Let the panel switch to display:block, then nudge Plotly.
                setTimeout(function () {
                  window.parent.dispatchEvent(new Event("resize"));
                }, 150);
              });
            });
          }
          bind();
          setTimeout(bind, 800);  // rebind in case tab buttons mount after this runs
        })();
        </script>
        """,
        height=0,
    )


def main() -> None:
    st.set_page_config(
        page_title="k4Bench Dashboard",
        page_icon="⚡",
        layout="wide",
    )

    # ── Global CSS tweaks ─────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Make the top toolbar transparent so it doesn't obscure tabs */
        [data-testid="stToolbar"],
        header[data-testid="stHeader"] {
            background: transparent !important;
            backdrop-filter: none !important;
        }
        /* The whitespace above the tabs is the (mostly empty) header band, not the
           block padding — so shrink the header itself while keeping the toolbar
           (menu / Deploy / Running) usable, and pull the content up under it. */
        header[data-testid="stHeader"] {
            height: 2.5rem !important;
            min-height: 2.5rem !important;
        }
        .block-container, .stMainBlockContainer {
            padding-top: 0.75rem !important;
            padding-bottom: 1rem !important;
        }
        /* Collapse the blank gap Streamlit inserts below plotly iframes */
        [data-testid="stPlotlyChart"] > div,
        .stPlotlyChart > div {
            line-height: 0;
        }
        /* Remove extra bottom margin on plotly chart wrappers */
        [data-testid="stPlotlyChart"],
        .stPlotlyChart {
            margin-bottom: 0 !important;
        }
        /* Tighten the gap between the last element and the footer */
        footer { margin-top: 0 !important; padding-top: 0 !important; }

        /* Let st.metric values wrap instead of truncating long strings with an
           ellipsis (e.g. a full OS name "AlmaLinux 9.7 (Seafoam Ocelot)" or a
           kernel version). Keeps every metric the same size while showing the
           value in full. */
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
        }

        /* Sticky footer: keep the copyright pinned to the bottom of the viewport
           when the page is short, but let it flow normally (and scroll) when the
           content is taller than the window. Achieved with a full-height flex
           column whose footer element gets margin-top:auto. */
        .stMainBlockContainer {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        /* Let the root vertical block (one or two wrapper levels deep) grow to
           fill the container so the auto margin below has space to consume. */
        .stMainBlockContainer > div:first-child,
        .stMainBlockContainer > div:first-child > [data-testid="stVerticalBlock"] {
            flex: 1 0 auto;
            display: flex;
            flex-direction: column;
        }
        /* Push the footer's element container to the bottom of that free space. */
        [data-testid="stElementContainer"]:has(.k4-footer) {
            margin-top: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    config = Config.from_env()

    # ``run_dirs``  — windowed set of cached run dirs across ALL stacks for this
    #                 (detector, platform, sample). Used by the Trends tabs.
    # ``data_dir``  — path to the latest run dir for the selected stack (single-run tabs).
    run_dirs:   tuple[str, ...]    = ()
    data_dir:   str | None         = None
    selected_run_meta: dict | None = None

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
            # ── Remote mode ────────────────────────────────────────────────────
            st.markdown(
                f"""
                <a href="{config.data_url}" target="_blank" style="text-decoration:none;">
                  <div style="
                    background: rgba(91,155,213,0.08);
                    border: 1px solid rgba(91,155,213,0.28);
                    border-radius: 8px;
                    padding: 0.45rem 0.75rem;
                    margin-bottom: 0.25rem;
                    display: flex;
                    align-items: center;
                    gap: 0.55rem;
                    transition: background 0.2s;
                  ">
                    <span style="font-size:1.1rem;line-height:1;">🗄️</span>
                    <div style="overflow:hidden;">
                      <div style="
                        font-size:0.63rem;
                        text-transform:uppercase;
                        letter-spacing:0.07em;
                        color:#7a9fbf;
                        font-weight:600;
                        margin-bottom:0.1rem;
                      ">WebEOS data</div>
                      <div style="
                        font-size:0.70rem;
                        color:#5b9bd5;
                        font-weight:500;
                        white-space:nowrap;
                        overflow:hidden;
                        text-overflow:ellipsis;
                        max-width:180px;
                      ">{config.data_url.rstrip('/')} ↗</div>
                    </div>
                  </div>
                </a>
                """,
                unsafe_allow_html=True,
            )

            # Detector
            try:
                detectors = _cached_list_detectors(config.data_url)
            except Exception as err:
                st.error(f"Failed to list detectors: {err}")
                return
            if not detectors:
                st.error("No detectors found at the configured WebEOS URL.")
                return
            detector = st.selectbox("Detector", detectors, key="sb_detector")
            if not detector:
                return

            # Platform
            try:
                platforms = _cached_list_platforms(config.data_url, detector)
            except Exception as err:
                st.error(f"Failed to list platforms: {err}")
                return
            if not platforms:
                st.warning(f"No platforms found for detector '{detector}'.")
                return
            _drop_stale_selection("sb_platform", platforms)
            platform = st.selectbox("Platform", platforms, key="sb_platform")
            if not platform:
                return

            # Scan the release tree once; derive both the sample union and the
            # per-sample stack list from the same {stack: [samples]} map.
            try:
                stack_samples = _cached_scan_stack_samples(
                    config.data_url, detector, platform
                )
            except Exception as err:
                st.error(f"Failed to scan releases: {err}")
                return

            # Sample — union across all stacks, so a sample stays selectable
            # (and its trend history visible) even when the newest release dropped it.
            samples = sorted({s for samps in stack_samples.values() for s in samps})
            if not samples:
                st.warning(f"No samples found for '{detector} / {platform}'.")
                return
            _drop_stale_selection("sb_sample", samples)
            sample = st.selectbox("Sample", samples, key="sb_sample")
            if not sample:
                return

            # Stack — only releases that actually contain the chosen sample, newest
            # first, so the default jumps to the latest release that has the sample.
            # Sort explicitly here so the "defaults to the newest" caption holds
            # regardless of the order scan_stack_samples happens to return.
            stacks = sorted(
                (stk for stk, samps in stack_samples.items() if sample in samps),
                reverse=True,
            )
            if not stacks:
                st.warning(f"No releases contain sample '{sample}' for '{detector} / {platform}'.")
                return
            _drop_stale_selection("sb_stack", stacks)
            stack = st.selectbox("Stack", stacks, key="sb_stack")
            st.caption(
                f"Available in {len(stacks)} release(s); defaults to the newest. "
                "Single-run tabs use it; Trends shows all releases."
            )
            if not stack:
                return

            # Single-run tabs only ever show the latest run for the selected
            # stack — fetch just that one (cached), not the whole date history.
            try:
                data_dir = _cached_fetch_latest_run(
                    config.data_url, detector, platform, stack, sample, config.cache_dir
                )
            except Exception as err:
                st.error(f"Failed to download latest run: {err}")
                return
            if not data_dir:
                st.warning("No runs found for the selected combination.")
                return
            selected_run_meta = run_metadata(data_dir)

            # ── Trend window ───────────────────────────────────────────────────
            # Discover available run dates across ALL stacks (directory listings
            # only, no file downloads), then download only the windowed subset.
            try:
                stacks_dates = _cached_list_run_dates(
                    config.data_url, detector, platform, sample
                )
            except Exception as err:
                st.warning(f"Could not scan trend run dates: {err}")
                stacks_dates = {}

            all_dates = sorted({
                d
                for dates in stacks_dates.values()
                for d in (pd.to_datetime(dt, errors="coerce") for dt in dates)
                if pd.notna(d)
            })
            if all_dates:
                lo_date = all_dates[0].date()
                hi_date = all_dates[-1].date()
                st.header("Trend window")
                preset = st.selectbox(
                    "Range", list(WINDOW_PRESETS), index=0,  # default: Last 7 days
                    key="sb_trend_preset",
                    help="Limits the date range plotted in the Trends tabs. "
                         "Smaller windows load faster.",
                )
                custom_range: tuple[date, date] | None = None
                custom_incomplete = False
                if preset == "Custom…":
                    picked = st.date_input(
                        "From → to",
                        value=(max(lo_date, hi_date - timedelta(days=90)), hi_date),
                        min_value=lo_date, max_value=hi_date,
                    )
                    if isinstance(picked, (tuple, list)) and len(picked) == 2:
                        custom_range = (picked[0], picked[1])
                    else:
                        # Mid-selection: don't fall back to the full range (which
                        # would trigger a download of every run) — wait for both dates.
                        custom_incomplete = True
                        st.info("Pick both a start and end date.")

                if not custom_incomplete:
                    start, end = resolve_window(
                        preset, [d.date() for d in all_dates], custom_range
                    )
                    windowed = {
                        stk: [
                            dt for dt in dates
                            if pd.notna(_d := pd.to_datetime(dt, errors="coerce"))
                            and start <= _d.date() <= end
                        ]
                        for stk, dates in stacks_dates.items()
                    }
                    windowed_items = tuple(
                        (stk, tuple(sorted(ds))) for stk, ds in sorted(windowed.items()) if ds
                    )
                    n_runs = sum(len(ds) for _, ds in windowed_items)
                    st.caption(
                        f"{start:%Y-%m-%d} → {end:%Y-%m-%d} · "
                        f"{n_runs} run(s) across {len(windowed_items)} release(s)"
                    )
                    try:
                        run_dirs = _cached_fetch_runs_windowed(
                            config.data_url, detector, platform, sample,
                            config.cache_dir, windowed_items,
                        )
                    except Exception as err:
                        st.warning(f"Could not load trend data: {err}")
                        run_dirs = ()

        else:
            # ── Local mode ─────────────────────────────────────────────────────
            data_dir = st.text_input("Data directory", value=config.data_dir)

        # ── Validate data_dir & load single-run data ───────────────────────────
        _data_path  = Path(data_dir) if data_dir else None
        _path_valid = bool(_data_path and _data_path.exists() and _data_path.is_dir())

        if not _path_valid:
            st.error(
                f"Data directory not found: '{data_dir}'. "
                "Check for typos or a missing directory."
            )
            results = event_data = region_data = None
            available_labels: list[str] = []
        else:
            results      = cached_load_results(data_dir)
            event_data   = cached_load_event_timing(data_dir)
            region_data  = cached_load_region_timing(data_dir)
            available_labels = collect_labels(results, event_data, region_data)

        if _path_valid and not available_labels:
            st.warning("No benchmark data found in the specified directory.")

        # ── Filters ────────────────────────────────────────────────────────────
        if not available_labels:
            selected_labels: list[str] = []
        else:
            st.header("Filters")
            selected_labels = st.multiselect(
                "Configurations", available_labels, default=available_labels
            )

        _render_sidebar_footer()

    if not available_labels:
        path_hint = f" in **{data_dir}**" if data_dir else ""
        st.info(
            f"No benchmark results found{path_hint}. "
            "Set the `K4BENCH_DATA_DIR` environment variable or update the path in the sidebar."
        )
        return

    # ── Run status banners (per-config detail + logs live in the Logs tab) ─────
    render_run_status(results, selected_run_meta)

    # ── Load trend data (remote only) ─────────────────────────────────────────
    trend_results_df = None
    trend_region_df  = None
    trend_event_df   = None
    trend_machine_df = None
    if run_dirs:
        trend_results_df = cached_load_trend_results(run_dirs)
        trend_region_df  = cached_load_trend_region_timing(run_dirs)
        trend_event_df   = cached_load_trend_event_timing(run_dirs)
        trend_machine_df = cached_load_trend_machine_info(run_dirs)

    # ── Build tab list ─────────────────────────────────────────────────────────
    # Trends-capable tabs are gated on *remote mode*, not on whether the current
    # trend window happens to have data. This keeps the tab set and each tab's
    # view selector stable across trend-window changes, so the active tab / sub-view
    # is preserved when the user only adjusts the window; an empty window shows an
    # in-view "widen the window" message instead of removing the option.
    trends_enabled = bool(config.data_url)
    tab_names = ["Run Trends", "Config Impact", "Region Timing", "Event Timing", "Event Memory", "Machine Info", "Logs"]
    if not trends_enabled:
        # Trends / Impact only make sense with multi-run (remote) data
        tab_names = tab_names[2:]

    tabs = st.tabs(tab_names)
    _force_plotly_relayout_on_tab_switch()
    tab_idx = 0

    # Trends (remote only) — uses all stacks so history is complete
    if trends_enabled:
        with tabs[tab_idx]:
            trends.render(trend_results_df, selected_labels)
        tab_idx += 1

        with tabs[tab_idx]:
            impact.render(trend_results_df, selected_labels)
        tab_idx += 1

    # Region Timing
    with tabs[tab_idx]:
        region_timing.render(region_data, trend_region_df, selected_labels, trends_enabled)
    tab_idx += 1

    # Event Timing
    with tabs[tab_idx]:
        event_timing.render(event_data, trend_event_df, selected_labels, trends_enabled)
    tab_idx += 1

    # Event Memory
    with tabs[tab_idx]:
        event_memory.render(event_data, trend_event_df, selected_labels, trends_enabled)
    tab_idx += 1

    # Machine Info
    with tabs[tab_idx]:
        minfo = load_machine_info(data_dir) if _path_valid else None
        machine_info.render(
            minfo,
            run_meta=selected_run_meta,
            results=results,
            trend_machine_df=trend_machine_df,
            trend_results_df=trend_results_df,
            trends_enabled=trends_enabled,
        )
    tab_idx += 1

    # Logs (per-config status + log viewer)
    with tabs[tab_idx]:
        render_logs_tab(results, data_dir if _path_valid else None)

    _render_footer()


main()
