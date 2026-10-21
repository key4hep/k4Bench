"""The sidebar's two pickers in remote mode: which benchmark run the single-run
tabs show, and which window of history the trend views read.

Both walk the WebEOS run store, so both exist in remote mode only; local mode
has one run directory and nothing above it. Every control here is mirrored in
the page URL under the names :mod:`url_params` gives, which is what the nightly
mail's and pull-request comments' deep links set (see
``k4bench.regression.render.dashboard_link``). Once resolved, the run is
written there exactly, so a copied link keeps showing the same measurement
after the release is run again.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import streamlit as st

from k4bench.labels import plural, pretty_release
from k4bench.layout import SIM, RunRef
from remote_cache import (
    cached_fetch_run,
    cached_fetch_runs_windowed,
    cached_list_detectors,
    cached_list_platforms,
    cached_list_report_dates,
    cached_list_run_dates,
    cached_list_runs,
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
from url_params import (
    DETECTOR,
    HISTORY,
    HISTORY_ALL,
    HISTORY_MODES,
    PLATFORM,
    RELEASES,
    RUN,
    SAMPLE,
    STACK,
    is_run_id,
    parse_release_range,
    release_range,
)

@dataclass(frozen=True)
class RunSelection:
    """The sidebar's place in the run store, and the run the single-run tabs show."""

    #: The selected run, exactly.
    ref: RunRef
    #: Its directory in the local download cache.
    data_dir: str

    @property
    def detector(self) -> str:
        return self.ref.detector

    @property
    def platform(self) -> str:
        return self.ref.platform

    @property
    def sample(self) -> str:
        return self.ref.sample

    @property
    def stack(self) -> str:
        """The Key4hep release the run measured."""
        return self.ref.stack

    @property
    def run(self) -> str:
        """The night the run was measured: its id."""
        return self.ref.date


@dataclass(frozen=True)
class TrendWindow:
    """The history the trend views read: every run of the releases in the window."""

    #: Inclusive ``(start, end)`` of the report nights that judged the
    #: window's releases (for a suite without reports, of the nights its runs
    #: ran); ``None`` when nothing has run yet.
    dates: tuple[date, date] | None = None
    #: The windowed runs' cached directories, across all releases.
    run_dirs: tuple[str, ...] = ()
    #: Whether the histories that draw one run per release draw every run
    #: instead, a release's repeats side by side.
    all_runs: bool = False


def _url_synced_selectbox(
    label: str, options: list[str], key: str, param: str, reason: str,
    format_func=str,
) -> str | None:
    """A selectbox mirrored in ``?param=``, returning its value.

    A remembered value no longer among *options* is dropped first (see
    :func:`ui_chrome.drop_stale_selection`), a first visit opens on the URL's
    value, and the selection is written back so the address bar stays a deep
    link to what is on screen. A dropped value, or a link's value that is not
    among *options*, is named under the selectbox with the *reason* it is not
    (see :func:`ui_chrome.render_scope_reset`). The value is kept for the whole
    session, so switching to another page and back returns to it: a page
    switch clears the URL, which therefore cannot carry it.
    """
    dropped = drop_stale_selection(key, options)
    rejected = seed_query_param(key, param, options)
    value = st.selectbox(
        label, options, key=key, persist_state="session", format_func=format_func,
    )
    if value:
        st.query_params[param] = value
        render_scope_reset(dropped or rejected, value, reason)
    return value


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
    """Detector → Platform → Sample → Stack → Run in *suite*'s run tree.

    Each list is read from the store below the level above it; the Run
    opens on the release's latest. ``None`` when a level cannot be listed or
    is empty, a link's run is not in the store, or the run cannot be
    downloaded — the reason is already on screen, and nothing below it can be
    drawn. *key_prefix* prefixes the selectboxes' keys, so each page keeps its
    own selection; the URL parameters they mirror are the same on every page.
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
    detector = _url_synced_selectbox(
        "Detector", order_detectors(detectors), f"{key_prefix}sb_detector", DETECTOR,
        "isn't benchmarked here",
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
    platform = _url_synced_selectbox(
        "Platform", order_platforms(platforms), f"{key_prefix}sb_platform", PLATFORM,
        f"isn't available for {detector}",
    )
    if not platform:
        return None

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
    # The Overview tab is scoped by sample across all detectors, so a
    # re-default here changes which detectors it shows: hence the note.
    sample = _url_synced_selectbox(
        "Sample", samples, f"{key_prefix}sb_sample", SAMPLE,
        f"isn't benchmarked for {detector}",
    )
    if not sample:
        return None

    # Stack — only releases that actually contain the chosen sample, newest
    # first, so the default jumps to the latest release that has the sample.
    # Sort explicitly here so the default is the newest
    # regardless of the order scan_stack_samples happens to return.
    stacks = sorted(
        (stk for stk, samps in stack_samples.items() if sample in samps),
        reverse=True,
    )
    if not stacks:
        st.warning(f"No releases contain sample '{sample}' for '{detector} / {platform}'.")
        return None
    stack = _url_synced_selectbox(
        "Stack", stacks, f"{key_prefix}sb_stack", STACK,
        f"doesn't hold {sample} for {detector}",
    )
    st.caption(f"Available in {plural(len(stacks), 'release')}")
    if not stack:
        return None

    run = _pick_run_date(
        base_url, suite, detector, platform, stack, sample, f"{key_prefix}sb_run",
    )
    if run is None:
        return None
    ref = RunRef(suite, detector, platform, stack, sample, run)
    try:
        data_dir = cached_fetch_run(
            base_url, suite, detector, platform, stack, sample, run, cache_dir,
        )
    except Exception as err:
        st.error(f"Failed to download the run measured {run} with {stack}: {err}")
        return None
    return RunSelection(ref, data_dir)


def _pick_run_date(
    base_url: str, suite: str, detector: str, platform: str, stack: str, sample: str,
    key: str,
) -> str | None:
    """The Run selectbox: which of *stack*'s runs of *sample* the single-run
    views show, by the night it was measured, newest first.

    A first visit opens on the link's run, or on the latest; the choice is
    then written to the URL, which pins it. Changing a level above keeps the
    run where the new release has one of the same night, and otherwise
    re-defaults to its latest, saying so. ``None``, with the reason on
    screen, when the release's runs cannot be listed and nothing pinned one,
    or when the link's run is not in the store: then the latest is offered
    with a button, never opened in its place.
    """
    linked = None if key in st.session_state else st.query_params.get(RUN)
    try:
        listed = cached_list_runs(base_url, suite, detector, platform, stack, sample)
    except Exception as err:
        pinned = st.session_state.get(key) or linked
        if not is_run_id(pinned):
            st.error(f"Failed to list the runs of {stack}: {err}")
            return None
        # A listing that did not answer says nothing about the pinned run,
        # which is therefore still shown — not replaced by a guess at the latest.
        st.warning(
            f"Could not list the runs of {stack}: {err}. Showing the pinned run "
            f"measured {pinned}; the others are offered once the listing answers."
        )
        runs, latest = [pinned], None
    else:
        runs = sorted((run for run in listed if is_run_id(run)), reverse=True)
        if not runs:
            st.warning("No runs found for the selected combination.")
            return None
        if linked and linked not in runs:
            _offer_latest_run(key, linked, runs[0], detector, platform, sample, stack)
            return None
        latest = runs[0]

    run = _url_synced_selectbox(
        "Run", runs, key, RUN, f"isn't a run of {stack}",
        format_func=lambda day: f"{day} · latest" if day == latest else day,
    )
    if run:
        # Runs are counted oldest first; a listing that did not answer has no count.
        place = f" · run {len(runs) - runs.index(run)} of {len(runs)}" if latest else ""
        st.caption(
            f"Measured **{run}**{place}",
            help=(
                "A release is re-run every night until the next one is tagged. "
                "The Run selector opens on the latest run. Single-run views show "
                "the selected run; historical views use the Trend window."
            ),
        )
    return run


def _offer_latest_run(
    key: str, linked: str, latest: str,
    detector: str, platform: str, sample: str, stack: str,
) -> None:
    """Say that the link's run *linked* is not in the store, and offer the
    release's *latest* in its place, with a button."""
    what = "is not in the store" if is_run_id(linked) else "does not name a run"
    st.warning(
        f"The linked run `{linked}` {what}: no {sample} run of {detector} on "
        f"{platform} with {stack} was measured that night."
    )

    def _open_latest() -> None:
        st.session_state[key] = latest
        st.query_params[RUN] = latest

    st.button(f"Open the latest run, {latest}", on_click=_open_latest, width="stretch")


def pick_trend_window(
    base_url: str, run: RunSelection, cache_dir: str, across_detectors: bool,
    *, suite: str = SIM, reports: bool = True, key_prefix: str = "",
    offer_all_runs: bool = False,
) -> TrendWindow:
    """The Trend window section, and the runs it covers for *run*'s
    detector, platform and sample in *suite*'s run tree.

    The slider ranges over the releases *run*'s detector was benchmarked with,
    or, *across_detectors*, over every release benchmarked on the platform by
    any detector — what a section that compares the detectors draws, which
    must not shrink to the history of whichever one the sidebar has selected.
    The releases and run dates come from directory listings only; just the
    windowed runs are downloaded. A suite without nightly *reports* bounds the
    window by its runs alone. *offer_all_runs* adds the choice of drawing
    every run of a release rather than its latest, for a section whose
    histories honour it. Both are mirrored in the URL (see :func:`_window_range`);
    *key_prefix* keeps each page's own.
    """
    # Whether every listing answered: one that did not leaves the slider's
    # releases short, against which a link's window must not be judged.
    complete = True
    try:
        stacks_dates = cached_list_run_dates(
            base_url, suite, run.detector, run.platform, run.sample
        )
    except Exception as err:
        st.warning(f"Could not scan trend run dates: {err}")
        stacks_dates, complete = {}, False

    platform_stacks: list[str] = []
    if across_detectors:
        try:
            platform_stacks = list_platform_stacks(base_url, suite, run.platform)
        except Exception as err:
            st.warning(f"Could not list the platform's releases: {err}")
            complete = False
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
            complete = False
    report_days = to_dates(report_nights)
    if across_detectors and report_days:
        # The Overview draws the reports, so a release tonight's batch is still
        # uploading has nothing to show there until its report lands.
        releases = reported_releases(releases, report_days[-1])
    if not releases:
        return TrendWindow()

    st.header("Trend window")
    opening = releases[-DEFAULT_RELEASES:]
    key = f"{key_prefix}trend_releases"
    lo, hi = st.select_slider(
        "Releases", releases,
        value=_window_range(key, releases, across_detectors, complete)
        or (opening[0], opening[-1]),
        key=key, persist_state="session",
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
    if complete:
        st.query_params[RELEASES] = release_range(lo, hi)
    selected = releases_between(releases, lo, hi)
    windowed = runs_in_window(stacks_dates, selected)
    first, last = release_day(selected[0]), release_day(selected[-1])
    caption = f"{first:%Y-%m-%d} → {last:%Y-%m-%d} · {plural(len(selected), 'release')}"
    if not across_detectors:
        caption += f" · {plural(sum(len(dates) for _, dates in windowed), 'run')}"
    st.caption(caption)
    try:
        run_dirs = cached_fetch_runs_windowed(
            base_url, suite, run.detector, run.platform, run.sample, cache_dir, windowed,
        )
    except Exception as err:
        st.warning(f"Could not load trend data: {err}")
        run_dirs = ()
    # The newest night the store knows of: its newest report, or, for a suite
    # no report judges, its newest run — a release keeps being re-run nightly
    # until the next one is tagged.
    nights = report_days if reports else to_dates(
        day for days in stacks_dates.values() for day in days
    )
    last_night = max([release_day(releases[-1]), *nights])
    all_runs = offer_all_runs and _pick_history(f"{key_prefix}trend_history") == HISTORY_ALL
    return TrendWindow(report_span(selected, releases, last_night), run_dirs, all_runs)


def _window_range(
    key: str, releases: list[str], across_detectors: bool, complete: bool,
) -> tuple[str, str] | None:
    """The range the Releases slider under *key* opens on when it has none:
    a link's ``?releases=`` on a first visit, else ``None`` for the newest.

    A new release — on the Overview, a newly reported one — re-opens the
    slider on the newest releases rather than leaving it on a range that
    stopped at yesterday's. So does a change of scope — another section, or a
    detector with other releases — while the same releases keep the range.
    A link's range is honoured on the first visit only, before anything has
    changed. It is judged only against *complete* listings, which are also
    the only ones recorded as the scope: a stalled one would have it dropped
    as unknown and then, once the listing answers, count as a new release.
    """
    scope_key = f"_{key}_scope"
    previous = st.session_state.get(scope_key)
    if complete:
        st.session_state[scope_key] = (tuple(releases), across_detectors)
    if previous is not None:
        if complete and previous != (tuple(releases), across_detectors):
            st.session_state.pop(key, None)
        return None
    linked = st.query_params.get(RELEASES)
    if not linked or not complete:
        return None
    bounds = parse_release_range(linked)
    if bounds is None or not set(bounds) <= set(releases):
        st.caption(
            f"⚠️ The linked window `{linked}` is not a range of the releases on "
            "offer — showing the newest."
        )
        return None
    # A link opens on its range even where this session had already drawn one.
    st.session_state.pop(key, None)
    return tuple(sorted(bounds, key=releases.index))


def _pick_history(key: str) -> str:
    """Whether the histories draw each release's latest run or all of them,
    mirrored in ``?history=``; the latest on a first visit without one."""
    rejected = seed_query_param(key, HISTORY, list(HISTORY_MODES))
    mode = st.radio(
        "Runs per release", list(HISTORY_MODES), format_func=HISTORY_MODES.get,
        key=key, persist_state="session",
        help=(
            "A release is re-run every night until the next one is tagged. "
            "**Latest run per release** draws one point per release, its newest "
            "run, on the release's date. **Every run** draws each run on the "
            "night it was measured, a release's repeats side by side, to show "
            "how much one release's measurements spread."
        ),
    )
    st.query_params[HISTORY] = mode
    if rejected:
        st.caption(f"⚠️ `{rejected}` is not a choice of runs — showing {HISTORY_MODES[mode]}.")
    return mode
