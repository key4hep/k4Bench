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
    WINDOW_PRESETS,
    dated_releases,
    release_day,
    report_span,
    runs_in_window,
    select_releases,
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

#: The preset a first visit opens on, when the URL names none.
DEFAULT_PRESET = "Last 10 releases"


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


def pick_run(base_url: str, cache_dir: str) -> RunSelection | None:
    """Detector → Platform → Sample → Stack, then the stack's latest run.

    Each list is read from the store below the level above it. ``None`` when a
    level cannot be listed or is empty, or the run cannot be downloaded — the
    reason is already on screen, and nothing below it can be drawn.
    """
    st.markdown(
        resource_link_card(base_url, "🗄️", "WebEOS data", base_url.rstrip("/")),
        unsafe_allow_html=True,
    )

    try:
        detectors = cached_list_detectors(base_url)
    except Exception as err:
        st.error(f"Failed to list detectors: {err}")
        return None
    if not detectors:
        st.error("No detectors found at the configured WebEOS URL.")
        return None
    detector, _ = _url_synced_selectbox(
        "Detector", order_detectors(detectors), "sb_detector", "detector"
    )
    if not detector:
        return None
    render_example_detector_badge(detector)

    try:
        platforms = cached_list_platforms(base_url, detector)
    except Exception as err:
        st.error(f"Failed to list platforms: {err}")
        return None
    if not platforms:
        st.warning(f"No platforms found for detector '{detector}'.")
        return None
    platform, dropped = _url_synced_selectbox(
        "Platform", order_platforms(platforms), "sb_platform", "platform"
    )
    if not platform:
        return None
    render_scope_reset(dropped, platform, f"isn't available for {detector}")

    # Scan the release tree once; derive both the sample union and the
    # per-sample stack list from the same {stack: [samples]} map.
    try:
        stack_samples = cached_scan_stack_samples(base_url, detector, platform)
    except Exception as err:
        st.error(f"Failed to scan releases: {err}")
        return None

    # Sample — union across all stacks, so a sample stays selectable
    # (and its trend history visible) even when the newest release dropped it.
    samples = sorted({s for samps in stack_samples.values() for s in samps})
    if not samples:
        st.warning(f"No samples found for '{detector} / {platform}'.")
        return None
    sample, dropped = _url_synced_selectbox("Sample", samples, "sb_sample", "sample")
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
    stack, _ = _url_synced_selectbox("Stack", stacks, "sb_stack", "stack")
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
            base_url, detector, platform, stack, sample, cache_dir
        )
    except Exception as err:
        st.error(f"Failed to download latest run: {err}")
        return None
    if not data_dir:
        st.warning("No runs found for the selected combination.")
        return None
    return RunSelection(detector, platform, sample, stack, data_dir)


def _pick_custom_range(releases: list[str], opening: list[str]) -> tuple[str, str]:
    """The custom range's oldest and newest release, opening on *opening*."""
    lo, hi = st.select_slider(
        "Releases", releases,
        value=(opening[0], opening[-1]),
        format_func=pretty_release,
    )
    return lo, hi


def pick_trend_window(base_url: str, run: RunSelection, cache_dir: str) -> TrendWindow:
    """The Trend window section, and the runs it covers for *run*'s
    detector, platform and sample.

    The releases and run dates come from directory listings only; just the
    windowed runs are downloaded.
    """
    try:
        stacks_dates = cached_list_run_dates(
            base_url, run.detector, run.platform, run.sample
        )
    except Exception as err:
        st.warning(f"Could not scan trend run dates: {err}")
        stacks_dates = {}
    run_days = to_dates(dt for dates in stacks_dates.values() for dt in dates)

    # Every release benchmarked on the platform, by any detector: the presets
    # count over these so that they cover the same releases whichever detector
    # is selected (see `trend_window.select_releases`). The selected
    # detector's own releases are folded in so a listing that missed one
    # cannot hide its runs.
    try:
        platform_stacks = list_platform_stacks(base_url, run.platform)
    except Exception as err:
        st.warning(f"Could not list the platform's releases: {err}")
        platform_stacks = []
    releases = dated_releases([*platform_stacks, *stacks_dates])
    if not releases:
        return TrendWindow()

    # The nightly report nights — one cached directory listing, the same one
    # the Overview tab reads. The newest is where a preset starts counting,
    # and they bound the nights the Overview draws.
    try:
        report_nights = cached_list_report_dates(base_url)
    except Exception as err:
        st.warning(f"Could not list nightly reports: {err}")
        report_nights = []
    report_days = to_dates(report_nights)

    st.header("Trend window")
    presets = list(WINDOW_PRESETS)
    seed_query_param("sb_trend_preset", "range", presets)
    st.session_state.setdefault("sb_trend_preset", DEFAULT_PRESET)
    preset = st.selectbox(
        "Range", presets,
        key="sb_trend_preset",
        help="Limits the Key4hep releases plotted in the Trends and "
             "Overview tabs. Counted over every release benchmarked on the "
             "platform, so it covers the same releases whichever detector "
             "is selected. Smaller windows load faster.",
    )
    st.query_params["range"] = preset
    anchor = report_days[-1] if report_days else None
    custom_range = None
    if preset == "Custom…":
        custom_range = _pick_custom_range(
            releases, select_releases(DEFAULT_PRESET, releases, None, anchor),
        )

    selected = select_releases(preset, releases, custom_range, anchor)
    windowed = runs_in_window(stacks_dates, selected)
    n_runs = sum(len(dates) for _, dates in windowed)
    first, last = release_day(selected[0]), release_day(selected[-1])
    caption = (
        f"{first:%Y-%m-%d} → {last:%Y-%m-%d} · {len(selected)} release(s) · "
        f"{n_runs} run(s) in {len(windowed)} of them"
    )
    # The window counts the platform's releases, not the detector's, so a
    # detector benchmarked for fewer of them fills only part of it. Naming
    # when it ran tells a short Run Trends tab apart from a broken window,
    # and turns an empty one into a range to widen to.
    own_releases = dated_releases(stacks_dates)
    if not n_runs and run_days:
        caption += f" · {run.detector} last ran {run_days[-1]:%Y-%m-%d}"
    elif run_days and own_releases and release_day(own_releases[0]) > first:
        caption += f" · {run.detector} first ran {run_days[0]:%Y-%m-%d}"
    st.caption(caption)
    try:
        run_dirs = cached_fetch_runs_windowed(
            base_url, run.detector, run.platform, run.sample, cache_dir, windowed,
        )
    except Exception as err:
        st.warning(f"Could not load trend data: {err}")
        run_dirs = ()
    last_night = max([release_day(releases[-1]), *report_days])
    return TrendWindow(report_span(selected, releases, last_night), run_dirs)
