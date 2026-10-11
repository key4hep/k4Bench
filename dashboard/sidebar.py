"""The sidebar's two pickers in remote mode: which benchmark run the single-run
tabs show, and which window of history the trend views read.

Both walk the WebEOS run store, so both exist in remote mode only; local mode
has one run directory and nothing above it. Every selectbox here is mirrored in
the page URL, which is what the nightly mail's and pull-request comments' deep
links set (see ``k4bench.regression.render.dashboard_link``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import streamlit as st

from remote_cache import (
    cached_fetch_latest_run,
    cached_fetch_runs_windowed,
    cached_list_detectors,
    cached_list_platforms,
    cached_list_report_dates,
    cached_list_run_dates,
    cached_scan_stack_samples,
)
from trend_window import (
    WINDOW_PRESETS,
    resolve_window,
    runs_in_window,
    to_dates,
    window_domain,
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
DEFAULT_PRESET = "Last 14 days"


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
    """The history the trend views read: every release's runs inside the window."""

    #: Inclusive ``(start, end)``; ``None`` when no window resolved — nothing
    #: has run yet, or a custom range is still being picked.
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


def _pick_custom_range(lo: date, hi: date) -> tuple[date, date] | None:
    """The custom range's two dates, or ``None`` while only one is picked.

    Mid-selection does not fall back to the full range, which would download
    every run, but waits for both dates.
    """
    picked = st.date_input(
        "From → to",
        value=(max(lo, hi - timedelta(days=90)), hi),
        min_value=lo, max_value=hi,
    )
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        return picked[0], picked[1]
    st.info("Pick both a start and end date.")
    return None


def pick_trend_window(base_url: str, run: RunSelection, cache_dir: str) -> TrendWindow:
    """The Trend window section, and the runs it covers for *run*'s
    detector, platform and sample across every release.

    The run dates come from directory listings only; just the windowed subset
    is downloaded.
    """
    try:
        stacks_dates = cached_list_run_dates(
            base_url, run.detector, run.platform, run.sample
        )
    except Exception as err:
        st.warning(f"Could not scan trend run dates: {err}")
        stacks_dates = {}
    run_days = to_dates(dt for dates in stacks_dates.values() for dt in dates)

    # The nightly report nights — one cached directory listing, the same
    # one the Overview tab reads. They are folded into the window's date
    # domain so the preset anchors on a date every detector shares; see
    # `trend_window.window_domain` for why the run dates alone are the
    # wrong anchor.
    try:
        report_nights = cached_list_report_dates(base_url)
    except Exception as err:
        st.warning(f"Could not list nightly reports: {err}")
        report_nights = []
    report_days = to_dates(report_nights)

    all_dates = window_domain(run_days, report_days)
    if not all_dates:
        return TrendWindow()

    st.header("Trend window")
    presets = list(WINDOW_PRESETS)
    seed_query_param("sb_trend_preset", "range", presets)
    st.session_state.setdefault("sb_trend_preset", DEFAULT_PRESET)
    preset = st.selectbox(
        "Range", presets,
        key="sb_trend_preset",
        help="Limits the date range plotted in the Trends and "
             "Overview tabs. Anchored on the newest nightly, so it "
             "covers the same dates whichever detector is "
             "selected. Smaller windows load faster.",
    )
    st.query_params["range"] = preset
    custom_range = None
    if preset == "Custom…":
        custom_range = _pick_custom_range(all_dates[0], all_dates[-1])
        if custom_range is None:
            return TrendWindow()

    # Presets count back from the newest report night, which every
    # detector shares — a run already uploaded ahead of tonight's report
    # must not pull the window's start forward for the detector that
    # uploaded it.
    start, end = resolve_window(
        preset, all_dates, custom_range,
        anchor=report_days[-1] if report_days else None,
    )
    windowed = runs_in_window(stacks_dates, start, end)
    n_runs = sum(len(dates) for _, dates in windowed)
    caption = (
        f"{start:%Y-%m-%d} → {end:%Y-%m-%d} · "
        f"{n_runs} run(s) across {len(windowed)} release(s)"
    )
    # The window spans every detector, so one that stopped being
    # benchmarked can have nothing in it. Naming its last run turns an
    # empty Run Trends tab into a range to widen to.
    if not n_runs and run_days:
        caption += f" · {run.detector} last ran {run_days[-1]:%Y-%m-%d}"
    st.caption(caption)
    try:
        run_dirs = cached_fetch_runs_windowed(
            base_url, run.detector, run.platform, run.sample, cache_dir, windowed,
        )
    except Exception as err:
        st.warning(f"Could not load trend data: {err}")
        run_dirs = ()
    return TrendWindow((start, end), run_dirs)
