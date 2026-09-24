"""Streamlit-cached wrappers around :mod:`k4bench.remote`'s network calls.

These are thin ``@st.cache_data`` shims kept out of ``app.py`` so the entry
point stays focused on layout and data-source resolution. Each wrapper imports
its underlying ``k4bench.remote`` function lazily so importing this module is
cheap.

``st.cache_data`` keeps whatever its function returns, for everyone, until the
TTL runs out — and never an exception. So every call that can come back short
because WebEOS stalled runs its fetch with ``strict=True`` inside the cached
function, and an incomplete result raises :class:`~k4bench.remote.IncompleteFetch`
instead of being kept. The next rerun reads it again, rather than every viewer
seeing a stall's gaps for an hour. What happens to the incomplete result depends
on what it is:

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

import streamlit as st

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


@st.cache_data(show_spinner="Fetching detectors...", ttl=3600)
def _cached_list_detectors(base_url: str) -> list[str]:
    from k4bench.remote import list_detectors
    return list_detectors(base_url)


@st.cache_data(show_spinner="Fetching platforms...", ttl=3600)
def _cached_list_platforms(base_url: str, detector: str) -> list[str]:
    from k4bench.remote import list_platforms
    return list_platforms(base_url, detector)


@st.cache_data(show_spinner="Scanning releases...", ttl=3600)
def _cached_scan_stack_samples(
    base_url: str, detector: str, platform: str
) -> dict[str, list[str]]:
    """Raises :class:`~k4bench.remote.IncompleteFetch` rather than return a
    scan missing a release: the sidebar would drop a selection that is valid."""
    from k4bench.remote import scan_stack_samples
    return scan_stack_samples(base_url, detector, platform, strict=True)


@st.cache_data(show_spinner="Scanning run dates...", ttl=600)
def _cached_list_run_dates(
    base_url: str, detector: str, platform: str, sample: str
) -> dict[str, list[str]]:
    """Raises :class:`~k4bench.remote.IncompleteFetch` rather than return a
    listing missing a release: callers read the newest release off it."""
    from k4bench.remote import list_run_dates_all_stacks
    return list_run_dates_all_stacks(base_url, detector, platform, sample, strict=True)


@st.cache_data(show_spinner="Scanning Key4hep releases...", ttl=600)
def _cached_list_stacks(base_url: str, detector: str, platform: str) -> list[str]:
    from k4bench.remote import list_stacks
    return list_stacks(base_url, detector, platform)


@st.cache_data(show_spinner="Fetching stack provenance...", ttl=3600)
def _fetch_stack_packages(
    base_url: str, detector: str, platform: str, stack: str
) -> dict | None:
    from k4bench.remote import fetch_stack_packages
    return fetch_stack_packages(base_url, detector, platform, stack, strict=True)


def _cached_fetch_stack_packages(
    base_url: str, detector: str, platform: str, stack: str
) -> dict | None:
    """Cached :func:`k4bench.remote.fetch_stack_packages`.

    A release is immutable once published, so this is cached for the full hour
    — the same stack is re-read every time it is an endpoint of a comparison.
    """
    return _keep_complete(_fetch_stack_packages, base_url, detector, platform, stack)


@st.cache_data(show_spinner="Listing regression reports...", ttl=600)
def _cached_list_report_dates(base_url: str) -> list[str]:
    """Cached :func:`k4bench.remote.list_report_dates`, which raises — and so
    is not cached — when the listing did not answer; callers say so rather
    than "no reports yet"."""
    from k4bench.remote import list_report_dates
    return list_report_dates(base_url)


@st.cache_data(show_spinner="Fetching blame...", ttl=600)
def _fetch_blame(base_url: str, date: str) -> dict | None:
    from k4bench.remote import fetch_blame
    return fetch_blame(base_url, date, strict=True)


def _cached_fetch_blame(base_url: str, date: str) -> dict | None:
    """Cached :func:`k4bench.remote.fetch_blame`.

    Most nights have no ``blame.json`` (only a confirmed, attributable
    regression produces one), so this returns ``None`` far more often than not.
    The TTL is deliberately shorter than the report's hour: the sidecar is
    uploaded up to ~15 minutes *after* its report (the blame stage runs last,
    behind GitHub and the model), and a ``None`` cached in that gap must not
    hide the freshly landed ranking for the rest of an hour. One small GET per
    viewed night per TTL is the whole cost.
    """
    return _keep_complete(_fetch_blame, base_url, date)


@st.cache_data(show_spinner="Fetching nightly reports...", ttl=3600)
def _fetch_reports(base_url: str, dates: tuple[str, ...]) -> dict[str, dict]:
    """Fetch a whole window of nightly reports in parallel, keyed by date.

    One report per night and each is a small JSON, so a cold window is
    dominated by request latency — fanning out cuts a 30-night first load
    from ~30 sequential round-trips to a few. Nights that fail to fetch are
    simply absent from the result; when one failed only because the server
    did not answer, the result raises instead of being cached (see
    :func:`_cached_fetch_reports`). Cached on the *dates* tuple: growing the
    window refetches it in one parallel burst rather than serially.

    Thread count is Python's own default (no ``max_workers``; this only ever
    runs on a cache miss — a Streamlit rerun that hits the cache spawns no
    threads at all). The concurrency ceiling lives one level down, in
    :mod:`k4bench.remote`'s shared, ``pool_block=True`` connection pool (see
    ``k4bench.remote._get_session``) — threads beyond it just queue for a
    slot instead of each opening its own connection, which is what fixed this
    fetch's intermittent, native interpreter crashes.
    """
    from concurrent.futures import ThreadPoolExecutor

    from k4bench.remote import IncompleteFetch, fetch_report

    failures: list[str] = []

    def _one(date: str) -> tuple[str, dict | None]:
        # fetch_report already handles absent and malformed reports (logs and
        # returns None), and raises only for a transient failure; this also
        # guards unexpected errors so one bad night can't abort the whole
        # window — logged (dashboard convention) but not surfaced as a UI
        # warning, since the night is simply absent from the result.
        try:
            return date, fetch_report(base_url, date, strict=True)
        except IncompleteFetch as exc:
            failures.extend(exc.failures)
            return date, None
        except Exception:
            _log.exception("fetch_report: unexpected error for %s", date)
            return date, None

    with ThreadPoolExecutor() as pool:
        reports = {date: raw for date, raw in pool.map(_one, dates) if raw}
    if failures:
        raise IncompleteFetch(reports, failures)
    return reports


def _cached_fetch_reports(base_url: str, dates: tuple[str, ...]) -> dict[str, dict]:
    """:func:`_fetch_reports`, cached only when every night could be read."""
    return _keep_complete(_fetch_reports, base_url, dates)


@st.cache_data(show_spinner="Downloading latest run...", ttl=3600)
def _cached_fetch_latest_run(
    base_url: str, detector: str, platform: str, stack: str, sample: str, cache_dir: str
) -> str | None:
    from k4bench.remote import ensure_latest_run_cached
    p = ensure_latest_run_cached(
        base_url, detector, platform, stack, sample, cache_root=cache_dir
    )
    return str(p) if p else None


@st.cache_data(show_spinner="Downloading trend data...", ttl=3600)
def _fetch_runs_windowed(
    base_url: str,
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
            base_url, detector, platform, sample, stacks_dates,
            cache_root=cache_dir, strict=True,
        )
    except IncompleteFetch as exc:
        raise IncompleteFetch(
            tuple(sorted(r["run_dir"] for r in exc.partial)), exc.failures
        ) from exc
    return tuple(sorted(r["run_dir"] for r in runs))


def _cached_fetch_runs_windowed(
    base_url: str,
    detector: str,
    platform: str,
    sample: str,
    cache_dir: str,
    stacks_dates_items: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[str, ...]:
    """:func:`_fetch_runs_windowed`, cached only when every run landed."""
    return _keep_complete(
        _fetch_runs_windowed,
        base_url, detector, platform, sample, cache_dir, stacks_dates_items,
    )
