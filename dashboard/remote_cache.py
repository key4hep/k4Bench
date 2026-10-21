"""Streamlit-cached wrappers around :mod:`k4bench.remote`'s network calls.

These are thin ``@st.cache_data`` shims kept out of ``app.py`` so the entry
point stays focused on layout and data-source resolution. Each wrapper imports
its underlying ``k4bench.remote`` function lazily so importing this module is
cheap. Like those functions, each takes the suite it reads (see
:mod:`k4bench.layout`) right after the base URL.

``st.cache_data`` and ``st.cache_resource`` keep whatever their function
returns, for everyone, until the TTL runs out — and never an exception. So every
call that can come back short because WebEOS stalled runs its fetch with
``strict=True`` inside the cached function, and an incomplete result raises
:class:`~k4bench.remote.IncompleteFetch` instead of being kept. The next rerun
reads it again, rather than every viewer seeing a stall's gaps for an hour. What
happens to the incomplete result depends on what it is:

* **Discovery** — the release and run-date listings — raises to the caller.
  Callers decide from these which release is newest and which selections are
  valid, and a listing missing the newest release answers both wrongly; the
  exception is a ``RequestException``, so their existing fallbacks apply.
* **Content** — reports, run downloads, blame, stack provenance — is handed
  back as far as it was read, uncached, with a notice (:func:`_keep_complete`).
  A missing night or run leaves a gap on the page but decides nothing else.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import requests
import streamlit as st

if TYPE_CHECKING:
    from k4bench.regression.models import NightlyReport

_log = logging.getLogger(__name__)


def _keep_complete(cached, *args):
    """Call the cached *cached*; return what an incomplete read got, uncached.

    For content only (see the module docstring). Tells the viewer with a
    toast, since the page is then drawn from less than is on EOS: a missing
    run or night looks exactly like one that never happened.
    """
    from k4bench.remote import IncompleteFetch

    try:
        return cached(*args)
    except IncompleteFetch as exc:
        _log.warning("%s: %s — not cached", cached.__name__, exc)
        st.toast(
            "Some benchmark data could not be read from EOS just now, so parts "
            "of this page may be incomplete. It is read again on your next "
            "interaction.",
            icon="⚠️",
        )
        return exc.partial


def read_again(cached, *args):
    """The cached function *cached*'s answer for *args*, read from the store
    again rather than from the cache: for a link naming something newer than
    the cached listing holds, which may have been published since it was read."""
    cached.clear(*args)
    return cached(*args)


@st.cache_data(show_spinner="Fetching detectors...", ttl=3600)
def cached_list_detectors(base_url: str, suite: str) -> list[str]:
    from k4bench.remote import list_detectors
    return list_detectors(base_url, suite)


@st.cache_data(show_spinner="Fetching platforms...", ttl=3600)
def cached_list_platforms(base_url: str, suite: str, detector: str) -> list[str]:
    from k4bench.remote import list_platforms
    return list_platforms(base_url, suite, detector)


@st.cache_data(show_spinner="Scanning releases...", ttl=3600)
def cached_scan_stack_samples(
    base_url: str, suite: str, detector: str, platform: str
) -> dict[str, list[str]]:
    """Raises :class:`~k4bench.remote.IncompleteFetch` rather than return a
    scan missing a release: the sidebar would drop a selection that is valid."""
    from k4bench.remote import scan_stack_samples
    return scan_stack_samples(base_url, suite, detector, platform, strict=True)


@st.cache_data(show_spinner="Scanning run dates...", ttl=600)
def cached_list_run_dates(
    base_url: str, suite: str, detector: str, platform: str, sample: str
) -> dict[str, list[str]]:
    """Raises :class:`~k4bench.remote.IncompleteFetch` rather than return a
    listing missing a release: callers read the newest release off it."""
    from k4bench.remote import list_run_dates_all_stacks
    return list_run_dates_all_stacks(
        base_url, suite, detector, platform, sample, strict=True,
    )


@st.cache_data(show_spinner="Scanning Key4hep releases...", ttl=600)
def cached_list_stacks(base_url: str, suite: str, detector: str, platform: str) -> list[str]:
    from k4bench.remote import list_stacks
    return list_stacks(base_url, suite, detector, platform)


def list_platform_stacks(base_url: str, suite: str, platform: str) -> list[str]:
    """Every release *suite* benchmarked on *platform*, newest first.

    Unioned across detectors rather than read from one: detectors join and
    leave the matrix, so no single detector's history is the full set of
    releases. A detector whose listing did not answer raises: the union
    without its releases would pass for the whole set.
    """
    stacks: set[str] = set()
    for detector in cached_list_detectors(base_url, suite):
        try:
            stacks.update(cached_list_stacks(base_url, suite, detector, platform))
        except requests.HTTPError as exc:
            # A detector never benchmarked on this platform has no such
            # directory to list: a 404 is that answer. Only that is expected
            # here — a timeout or a 5xx says nothing about its releases, and a
            # broader catch would bury a bug in the listing as "no releases".
            if getattr(exc.response, "status_code", None) != 404:
                raise
    return sorted(stacks, reverse=True)


@st.cache_data(show_spinner="Fetching stack provenance...", ttl=3600)
def _fetch_stack_packages(
    base_url: str, suite: str, detector: str, platform: str, stack: str
) -> dict | None:
    from k4bench.remote import fetch_stack_packages
    return fetch_stack_packages(base_url, suite, detector, platform, stack, strict=True)


def cached_fetch_stack_packages(
    base_url: str, suite: str, detector: str, platform: str, stack: str
) -> dict | None:
    """Cached :func:`k4bench.remote.fetch_stack_packages`.

    A release is immutable once published, so this is cached for the full hour
    — the same stack is re-read every time it is an endpoint of a comparison.
    """
    return _keep_complete(_fetch_stack_packages, base_url, suite, detector, platform, stack)


@st.cache_data(show_spinner="Fetching measured stack...", ttl=3600)
def _fetch_stack_snapshot(
    base_url: str, suite: str, detector: str, platform: str, stack: str, sample: str | None,
):
    from k4bench.remote import fetch_stack_snapshot
    return fetch_stack_snapshot(base_url, suite, detector, platform, stack, sample, strict=True)


def cached_fetch_stack_snapshot(
    base_url: str, suite: str, detector: str, platform: str, stack: str, sample: str | None,
):
    return _keep_complete(
        _fetch_stack_snapshot, base_url, suite, detector, platform, stack, sample,
    )


@st.cache_data(show_spinner="Listing regression reports...", ttl=600)
def cached_list_report_dates(base_url: str, suite: str) -> list[str]:
    """Cached :func:`k4bench.remote.list_report_dates`, which raises — and so
    is not cached — when the listing did not answer; callers say so rather
    than "no reports yet"."""
    from k4bench.remote import list_report_dates
    return list_report_dates(base_url, suite)


@st.cache_data(show_spinner="Fetching blame...", ttl=600)
def _fetch_blame(base_url: str, suite: str, date: str) -> dict | None:
    from k4bench.remote import fetch_blame
    return fetch_blame(base_url, suite, date, strict=True)


def cached_fetch_blame(base_url: str, suite: str, date: str) -> dict | None:
    """Cached :func:`k4bench.remote.fetch_blame`.

    Most nights have no ``blame.json`` (only a confirmed, attributable
    regression produces one), so this returns ``None`` far more often than not.
    The TTL is deliberately shorter than the report's hour: the sidecar is
    uploaded up to ~15 minutes *after* its report (the blame stage runs last,
    behind GitHub and the model), and a ``None`` cached in that gap must not
    hide the freshly landed ranking for the rest of an hour. One small GET per
    viewed night per TTL is the whole cost.
    """
    return _keep_complete(_fetch_blame, base_url, suite, date)


@st.cache_resource(show_spinner=False, ttl=3600)
def _fetch_report(base_url: str, suite: str, date: str) -> NightlyReport | None:
    """One night's report, parsed; ``None`` when it is absent or malformed.

    A report is several MB of JSON and takes far longer to parse than to read,
    so the parsed object itself is cached, once per night, and shared by every
    session and every window that includes the night. Shared, not copied:
    callers must not modify it. A night the server did not answer raises
    :class:`~k4bench.remote.IncompleteFetch`, which is not cached.
    """
    from k4bench.regression.render import from_json
    from k4bench.remote import fetch_report

    raw = fetch_report(base_url, suite, date, strict=True)
    if not raw:
        return None
    try:
        return from_json(raw)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        _log.warning("skipping malformed report for %s — %s", date, exc)
        return None


def _fetch_reports(
    base_url: str, suite: str, dates: tuple[str, ...]
) -> dict[str, NightlyReport]:
    """Every night of *dates* with a report, parsed and keyed by date.

    A night missing from the cache costs a request, so the nights are read in
    parallel. Nights without a parseable report are absent from the result;
    when one is absent only because the server did not answer, the result
    raises instead (see :func:`cached_fetch_reports`).

    Thread count is Python's own default (no ``max_workers``). The concurrency
    ceiling lives one level down, in :mod:`k4bench.remote`'s shared,
    ``pool_block=True`` connection pool (see ``k4bench.remote._get_session``) —
    threads beyond it just queue for a slot instead of each opening its own
    connection, which is what fixed this fetch's intermittent, native
    interpreter crashes.
    """
    from concurrent.futures import ThreadPoolExecutor

    from k4bench.remote import IncompleteFetch

    failures: list[str] = []

    def _one(date: str) -> tuple[str, NightlyReport | None]:
        # Unexpected errors are guarded so one bad night can't abort the whole
        # window — logged (dashboard convention) but not surfaced as a UI
        # warning, since the night is simply absent from the result.
        try:
            return date, _fetch_report(base_url, suite, date)
        except IncompleteFetch as exc:
            failures.extend(exc.failures)
            return date, None
        except Exception:
            _log.exception("fetch_report: unexpected error for %s", date)
            return date, None

    with st.spinner("Fetching nightly reports..."), ThreadPoolExecutor() as pool:
        reports = {
            date: report for date, report in pool.map(_one, dates) if report is not None
        }
    if failures:
        raise IncompleteFetch(reports, failures)
    return reports


def cached_fetch_reports(
    base_url: str, suite: str, dates: tuple[str, ...]
) -> dict[str, NightlyReport]:
    """:func:`_fetch_reports`; when a night could not be read, the others with
    a toast instead of an exception."""
    return _keep_complete(_fetch_reports, base_url, suite, dates)


@st.cache_data(show_spinner="Listing runs...", ttl=600)
def cached_list_runs(
    base_url: str, suite: str, detector: str, platform: str, stack: str, sample: str,
) -> list[str]:
    """The run dates of one release's *sample*, newest first: discovery only,
    one directory listing. A listing that did not answer raises, and so is not
    cached: the sidebar must not take a stall for a release without the run a
    link pinned."""
    from k4bench.remote import list_runs
    return list_runs(base_url, suite, detector, platform, stack, sample)


@st.cache_data(show_spinner="Downloading run...", ttl=3600)
def cached_fetch_run(
    base_url: str, suite: str, detector: str, platform: str, stack: str, sample: str,
    date: str, cache_dir: str,
) -> str:
    """One exact run, downloaded once into the on-disk cache
    (:func:`k4bench.remote.ensure_run_cached`); its cached directory."""
    from k4bench.remote import ensure_run_cached
    return str(ensure_run_cached(
        base_url, suite, detector, platform, stack, sample, date, cache_root=cache_dir,
    ))


@st.cache_data(show_spinner="Downloading trend data...", ttl=3600)
def _fetch_runs_windowed(
    base_url: str,
    suite: str,
    detector: str,
    platform: str,
    sample: str,
    cache_dir: str,
    stacks_dates_items: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[str, ...]:
    """Download the windowed ``(stack, date)`` set and return cached run dirs.

    *stacks_dates_items* is a hashable ``((stack, (date, ...)), ...)`` so the cache
    key varies with the selected window — widening reuses already-cached runs and
    only the new runs are fetched.
    """
    from k4bench.remote import IncompleteFetch, fetch_runs_windowed
    stacks_dates = {stack: list(dates) for stack, dates in stacks_dates_items}
    try:
        runs = fetch_runs_windowed(
            base_url, suite, detector, platform, sample, stacks_dates,
            cache_root=cache_dir, strict=True,
        )
    except IncompleteFetch as exc:
        raise IncompleteFetch(
            tuple(sorted(r["run_dir"] for r in exc.partial)), exc.failures
        ) from exc
    return tuple(sorted(r["run_dir"] for r in runs))


def cached_fetch_runs_windowed(
    base_url: str,
    suite: str,
    detector: str,
    platform: str,
    sample: str,
    cache_dir: str,
    stacks_dates_items: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[str, ...]:
    """:func:`_fetch_runs_windowed`, cached only when every run landed."""
    return _keep_complete(
        _fetch_runs_windowed,
        base_url, suite, detector, platform, sample, cache_dir, stacks_dates_items,
    )
