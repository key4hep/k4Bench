"""The sidebar's two pickers in remote mode: which benchmark run the single-run
tabs show, and which window of history the trend views read.

Both walk the WebEOS run store, so both exist in remote mode only; local mode
has one run directory and nothing above it. Every selectbox here is mirrored in
the page URL, which is what the nightly mail's and pull-request comments' deep
links set (see ``k4bench.regression.render.dashboard_link``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import streamlit as st

from k4bench.labels import pretty_release
from k4bench.layout import SIM
from remote_cache import (
    cached_fetch_latest_run,
    cached_fetch_runs_windowed,
    cached_list_detectors,
    cached_list_platforms,
    cached_list_report_dates,
    cached_list_run_dates,
    cached_scan_stack_samples,
    list_platform_stacks,
)
from trend_window import (
    DEFAULT_RELEASES,
    dated_releases,
    release_day,
    releases_between,
    report_span,
    reported_releases,
    runs_in_window,
    to_dates,
)
from ui_chrome import (
    drop_stale_selection,
    render_scope_reset,
    order_detectors,
    order_platforms,
    render_example_detector_badge,
    resource_link_card,
    seed_query_param,
)

@dataclass(frozen=True)
class RunSelection:
    """The sidebar's place in the run store, and the run the single-run tabs show."""

    detector: str
    platform: str
    sample: str
    stack: str
    #: The selected stack's latest run, downloaded into the local cache.
    data_dir: str


@dataclass(frozen=True)
class TrendWindow:
    """The history the trend views read: every run of the releases in the window."""

    #: Inclusive ``(start, end)`` of the report nights that judged the
    #: window's releases; ``None`` when nothing has run yet.
    dates: tuple[date, date] | None = None
    #: The windowed runs' cached directories, across all releases.
    run_dirs: tuple[str, ...] = ()


def _url_synced_selectbox(
    label: str, options: list[str], key: str, param: str
) -> tuple[str | None, str | None]:
    """A selectbox mirrored in ``?param=``, returning ``(value, dropped)``.

    A remembered value no longer among *options* is dropped first (see
    :func:`ui_chrome.drop_stale_selection`), a first visit opens on the URL's
    value, and the selection is written back so the address bar stays a deep
    link to what is on screen.
    """
    dropped = drop_stale_selection(key, options)
    seed_query_param(key, param, options)
    value = st.selectbox(label, options, key=key)
    if value:
        st.query_params[param] = value
    return value, dropped


def render_refresh_button() -> None:
    """The sidebar's "Refresh Data" button: drop every cached listing and run."""
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
        st.cache_resource.clear()


def pick_run(
    base_url: str, cache_dir: str, *, suite: str = SIM, key_prefix: str = "",
) -> RunSelection | None:
    """Detector → Platform → Sample → Stack in *suite*'s run tree, then the
    stack's latest run.

    Each list is read from the store below the level above it. ``None`` when a
    level cannot be listed or is empty, or the run cannot be downloaded — the
    reason is already on screen, and nothing below it can be drawn.
    *key_prefix* prefixes the selectboxes' keys, so each page keeps its own
    selection; the URL parameters they mirror are the same on every page.
    """
    st.markdown(
        resource_link_card(base_url, "🗄️", "WebEOS data", base_url.rstrip("/")),
        unsafe_allow_html=True,
    )

    try:
        detectors = cached_list_detectors(base_url, suite)
    except Exception as err:
        st.error(f"Failed to list detectors: {err}")
        return None
    if not detectors:
        st.error("No detectors found at the configured WebEOS URL.")
        return None
    detector, _ = _url_synced_selectbox(
        "Detector", order_detectors(detectors), f"{key_prefix}sb_detector", "detector"
    )
    if not detector:
        return None
    render_example_detector_badge(detector)

    try:
        platforms = cached_list_platforms(base_url, suite, detector)
    except Exception as err:
        st.error(f"Failed to list platforms: {err}")
        return None
    if not platforms:
        st.warning(f"No platforms found for detector '{detector}'.")
        return None
    platform, dropped = _url_synced_selectbox(
        "Platform", order_platforms(platforms), f"{key_prefix}sb_platform", "platform"
    )
    if not platform:
        return None
    render_scope_reset(dropped, platform, f"isn't available for {detector}")

    # Scan the release tree once; derive both the sample union and the
    # per-sample stack list from the same {stack: [samples]} map.
    try:
        stack_samples = cached_scan_stack_samples(base_url, suite, detector, platform)
    except Exception as err:
        st.error(f"Failed to scan releases: {err}")
        return None

    # Sample — union across all stacks, so a sample stays selectable
    # (and its trend history visible) even when the newest release dropped it.
    samples = sorted({s for samps in stack_samples.values() for s in samps})
    if not samples:
        st.warning(f"No samples found for '{detector} / {platform}'.")
        return None
    sample, dropped = _url_synced_selectbox(
        "Sample", samples, f"{key_prefix}sb_sample", "sample"
    )
    if not sample:
        return None
    # The Overview tab is scoped by sample across all detectors, so this
    # re-default silently changes which detectors it shows.
    render_scope_reset(dropped, sample, f"isn't benchmarked for {detector}")

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
        return None
    stack, _ = _url_synced_selectbox("Stack", stacks, f"{key_prefix}sb_stack", "stack")
    st.caption(
        f"Available in {len(stacks)} release(s); defaults to the newest. "
        "Single-run tabs use it; Trends shows all releases."
    )
    if not stack:
        return None

    # Single-run tabs only ever show the latest run for the selected
    # stack — fetch just that one (cached), not the whole date history.
    try:
        data_dir = cached_fetch_latest_run(
            base_url, suite, detector, platform, stack, sample, cache_dir
        )
    except Exception as err:
        st.error(f"Failed to download latest run: {err}")
        return None
    if not data_dir:
        st.warning("No runs found for the selected combination.")
        return None
    return RunSelection(detector, platform, sample, stack, data_dir)


def pick_trend_window(
    base_url: str, run: RunSelection, cache_dir: str, across_detectors: bool,
    *, suite: str = SIM, reports: bool = True,
) -> TrendWindow:
    """The Trend window section, and the runs it covers for *run*'s
    detector, platform and sample in *suite*'s run tree.

    The slider ranges over the releases *run*'s detector was benchmarked with,
    or, *across_detectors*, over every release benchmarked on the platform by
    any detector — what a section that compares the detectors draws, which
    must not shrink to the history of whichever one the sidebar has selected.
    The releases and run dates come from directory listings only; just the
    windowed runs are downloaded. A suite without nightly *reports* bounds the
    window by its runs alone.
    """
    try:
        stacks_dates = cached_list_run_dates(
            base_url, suite, run.detector, run.platform, run.sample
        )
    except Exception as err:
        st.warning(f"Could not scan trend run dates: {err}")
        stacks_dates = {}

    platform_stacks: list[str] = []
    if across_detectors:
        try:
            platform_stacks = list_platform_stacks(base_url, suite, run.platform)
        except Exception as err:
            st.warning(f"Could not list the platform's releases: {err}")
    # The selected detector's own releases are folded into the platform's so
    # a listing that missed one cannot hide its runs.
    releases = dated_releases([*platform_stacks, *stacks_dates])

    # The nightly report nights — one cached directory listing, the same one
    # the Overview tab reads. They bound the nights the Overview draws.
    report_nights: list[str] = []
    if reports:
        try:
            report_nights = cached_list_report_dates(base_url, suite)
        except Exception as err:
            st.warning(f"Could not list nightly reports: {err}")
    report_days = to_dates(report_nights)
    if across_detectors and report_days:
        # The Overview draws the reports, so a release tonight's batch is still
        # uploading has nothing to show there until its report lands.
        releases = reported_releases(releases, report_days[-1])
    if not releases:
        return TrendWindow()

    st.header("Trend window")
    opening = releases[-DEFAULT_RELEASES:]
    # No key: a widget without one is identified by its options and default,
    # so a new release — on the Overview, a newly reported one — re-opens the
    # slider on the newest releases rather than leaving it on a range that
    # stopped at yesterday's. So does a change of scope — another section, or
    # a detector with other releases — while the same releases keep the range.
    lo, hi = st.select_slider(
        "Releases", releases,
        value=(opening[0], opening[-1]),
        format_func=pretty_release,
        help=(
            "The Key4hep releases this tab draws, out of every release "
            "benchmarked on the platform by any detector and judged by a "
            "nightly report."
            if across_detectors else
            "The Key4hep releases the trend views read, out of those the "
            "selected detector was benchmarked with."
        ) + " Smaller windows load faster.",
    )
    selected = releases_between(releases, lo, hi)
    windowed = runs_in_window(stacks_dates, selected)
    first, last = release_day(selected[0]), release_day(selected[-1])
    caption = f"{first:%Y-%m-%d} → {last:%Y-%m-%d} · {len(selected)} release(s)"
    if not across_detectors:
        caption += f" · {sum(len(dates) for _, dates in windowed)} run(s)"
    st.caption(caption)
    try:
        run_dirs = cached_fetch_runs_windowed(
            base_url, suite, run.detector, run.platform, run.sample, cache_dir, windowed,
        )
    except Exception as err:
        st.warning(f"Could not load trend data: {err}")
        run_dirs = ()
    last_night = max([release_day(releases[-1]), *report_days])
    return TrendWindow(report_span(selected, releases, last_night), run_dirs)
