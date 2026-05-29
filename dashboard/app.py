from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import Config
from data import (
    cached_load_event_timing,
    cached_load_region_timing,
    cached_load_results,
    cached_load_trend_results,
    cached_load_trend_region_timing,
    cached_load_trend_event_timing,
    collect_labels,
    list_run_metadata,
    load_machine_info,
)
from tabs import event_memory, event_timing, impact, machine_info, region_timing, trends


# ── Cached remote helpers ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching detectors...", ttl=3600)
def _cached_list_detectors(base_url: str) -> list[str]:
    from remote import list_detectors
    return list_detectors(base_url)


@st.cache_data(show_spinner="Fetching platforms...", ttl=3600)
def _cached_list_platforms(base_url: str, detector: str) -> list[str]:
    from remote import list_platforms
    return list_platforms(base_url, detector)


@st.cache_data(show_spinner="Scanning releases...", ttl=3600)
def _cached_scan_stack_samples(
    base_url: str, detector: str, platform: str
) -> dict[str, list[str]]:
    from remote import scan_stack_samples
    return scan_stack_samples(base_url, detector, platform)


@st.cache_data(show_spinner="Downloading runs...", ttl=3600)
def _cached_download_all_runs(
    base_url: str, detector: str, platform: str, stack: str, sample: str
) -> str:
    from remote import download_all_runs
    return str(download_all_runs(base_url, detector, platform, stack, sample))


@st.cache_data(show_spinner="Downloading trend data (all stacks)...", ttl=3600)
def _cached_download_all_stacks(
    base_url: str, detector: str, platform: str, sample: str
) -> str:
    from remote import download_all_stacks_for_sample
    return str(download_all_stacks_for_sample(base_url, detector, platform, sample))


def _render_footer() -> None:
    """Render a CERN / FCC copyright footer at the bottom of the page."""
    st.markdown(
        """
        <hr style="border:none;border-top:1px solid rgba(128,128,128,0.25);margin:2.5rem 0 0.8rem 0;">
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            gap:1.2rem;
            padding:0.2rem 0 1.2rem 0;
            font-size:0.80rem;
            color:#9a9a9a;
            line-height:1.7;
            text-align:center;
        ">
            <span style="font-size:1.8rem;opacity:0.75;">⚛️</span>
            <div>
                <strong style="color:#c0c0c0;letter-spacing:0.02em;">© 2026 CERN</strong>
                &nbsp;·&nbsp;
                For the benefit of the&nbsp;<a
                    href="https://fcc.web.cern.ch/"
                    target="_blank"
                    style="color:#5b9bd5;text-decoration:none;font-weight:600;"
                >FCC project</a>
                <br>
                Created by <strong style="color:#c0c0c0;">Joshua Falco Beirer</strong>
                &nbsp;<span style="opacity:0.6;">(CERN)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_footer() -> None:
    """Render a compact attribution note at the bottom of the sidebar."""
    st.markdown(
        """
        <hr style="border:none;border-top:1px solid rgba(128,128,128,0.2);margin:1.5rem 0 0.6rem 0;">
        <div style="font-size:0.72rem;color:#888;text-align:center;line-height:1.6;padding-bottom:0.4rem;">
            <strong style="color:#a0a0a0;">© 2026 CERN</strong><br>
            For the benefit of the<br>
            <a href="https://fcc.web.cern.ch/" target="_blank"
               style="color:#5b9bd5;text-decoration:none;">FCC project</a><br>
            <span style="opacity:0.7;">J. F. Beirer (CERN)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="DD4bench Dashboard",
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
        /* Push content down just enough so tabs aren't hidden under the toolbar */
        .block-container, .stMainBlockContainer {
            padding-top: 3.5rem !important;
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    config = Config.from_env()

    # ``trends_dir``  — flat temp dir with ALL stacks' runs for this (detector, platform, sample).
    #                   Used by the Trends tab and the historical sub-views.
    # ``sample_dir``  — temp dir with runs for the selected (detector, platform, stack, sample).
    #                   Used to pick the most recent run for single-run tabs.
    # ``data_dir``    — path to the selected date-level run dir (single-run tabs).
    trends_dir: str | None         = None
    sample_dir: str | None         = None
    data_dir:   str | None         = None
    selected_run_meta: dict | None = None

    with st.sidebar:
        st.header("Data Source")
        if st.button("Refresh Data", width="content"):
            st.cache_data.clear()
            st.rerun()

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
            detector = st.selectbox("Detector", detectors)
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
            platform = st.selectbox("Platform", platforms)
            if not platform:
                return

            # Scan the release tree once; derive both the sample union and the
            # per-sample stack list from the same {stack: [samples]} map.
            try:
                stack_samples = _cached_scan_stack_samples(
                    config.data_url, detector, platform
                )
            except Exception as err:
                st.error(f"Failed to list samples: {err}")
                return

            # Sample — union across all stacks, so a sample stays selectable
            # (and its trend history visible) even when the newest release dropped it.
            samples = sorted({s for samps in stack_samples.values() for s in samps})
            if not samples:
                st.warning(f"No samples found for '{detector} / {platform}'.")
                return
            sample = st.selectbox("Sample", samples)
            if not sample:
                return

            # Stack — only releases that actually contain the chosen sample, newest
            # first, so the default jumps to the latest release that has the sample.
            stacks = [stk for stk, samps in stack_samples.items() if sample in samps]
            if not stacks:
                st.warning(f"No releases contain sample '{sample}' for '{detector} / {platform}'.")
                return
            stack = st.selectbox("Stack", stacks)
            st.caption(
                f"Available in {len(stacks)} release(s); defaults to the newest. "
                "Single-run tabs use it; Trends shows all releases."
            )
            if not stack:
                return

            # Download runs for the selected stack (single-run tabs)
            try:
                sample_dir = _cached_download_all_runs(
                    config.data_url, detector, platform, stack, sample
                )
            except Exception as err:
                st.error(f"Failed to download runs: {err}")
                return

            # Download runs across ALL stacks for this sample (Trends + historical views)
            try:
                trends_dir = _cached_download_all_stacks(
                    config.data_url, detector, platform, sample
                )
            except Exception as err:
                st.warning(f"Could not load cross-stack trend data: {err}")
                trends_dir = None

            run_meta = list_run_metadata(sample_dir)
            if not run_meta:
                st.warning("No runs found for the selected combination.")
                return

            # Pick the most recent run date
            run_meta_sorted = sorted(
                run_meta,
                key=lambda m: m["run_date"] if pd.notna(m["run_date"]) else pd.Timestamp.min,
            )
            selected_run_meta = run_meta_sorted[-1]
            data_dir = selected_run_meta["run_dir"]

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
            "Set the `DD4BENCH_DATA_DIR` environment variable or update the path in the sidebar."
        )
        return

    # ── Load trend data (remote only) ─────────────────────────────────────────
    trend_results_df = None
    trend_region_df  = None
    trend_event_df   = None
    if trends_dir is not None:
        trend_results_df = cached_load_trend_results(trends_dir)
        trend_region_df  = cached_load_trend_region_timing(trends_dir)
        trend_event_df   = cached_load_trend_event_timing(trends_dir)

    # ── Build tab list ─────────────────────────────────────────────────────────
    tab_names = ["Run Trends", "Config Impact", "Region Timing", "Event Timing", "Event Memory", "Machine Info"]
    if trends_dir is None:
        # Trends / Impact only make sense with multi-run (remote) data
        tab_names = tab_names[2:]

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # Trends (remote only) — uses all stacks so history is complete
    if trends_dir is not None:
        with tabs[tab_idx]:
            trends.render(trend_results_df, selected_labels)
        tab_idx += 1

        with tabs[tab_idx]:
            impact.render(trend_results_df, selected_labels)
        tab_idx += 1

    # Region Timing
    with tabs[tab_idx]:
        region_timing.render(region_data, trend_region_df, selected_labels)
    tab_idx += 1

    # Event Timing
    with tabs[tab_idx]:
        event_timing.render(event_data, trend_event_df, selected_labels)
    tab_idx += 1

    # Event Memory
    with tabs[tab_idx]:
        event_memory.render(event_data, trend_event_df, selected_labels)
    tab_idx += 1

    # Machine Info
    with tabs[tab_idx]:
        minfo = load_machine_info(data_dir) if _path_valid else None
        machine_info.render(minfo, run_meta=selected_run_meta)

    _render_footer()


main()
