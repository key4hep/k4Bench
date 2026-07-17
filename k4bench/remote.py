"""Discover and download benchmark run data from a WebEOS HTTP endpoint.

The WebEOS site serves an Apache-style directory listing at K4BENCH_DATA_URL.
Expected layout::

    {base_url}/
      {detector}/                                   e.g. ALLEGRO_o1_v03/
        {platform}/                                 e.g. x86_64-almalinux9-gcc14.2.0-opt/
          {stack}/                                  e.g. key4hep-2026-05-19/
            {sample}/                               e.g. single_e-_10GeV/
              {YYYY-MM-DD}/                         e.g. 2026-05-23/
                run_info.json
                machine_info.json
                {config}_results.csv
                {config}_events.json
                {config}_regions.json
                {config}.log
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote

import requests

_log = logging.getLogger(__name__)

_DIR_LINK_RE = re.compile(r'href="([^"/][^"]*/?)"', re.IGNORECASE)
_TIMEOUT = 15

#: One ``requests.Session`` for the whole process's lifetime, lazily created
#: on first use and never closed — every remote call goes through it (see
#: :func:`_session`). ``requests.Session``/urllib3's connection pools are
#: documented thread-safe for concurrent use *once built*; building one is
#: the unsafe part (a fresh ``requests.get()`` builds and tears one down on
#: every call), and that is exactly what caused this dashboard to segfault
#: intermittently: several threads each doing a cold TLS handshake at once,
#: or a Streamlit rerun abandoning a still-fetching thread's ad-hoc session
#: mid-handshake while a *new* rerun's own ad-hoc session starts up alongside
#: it. A single long-lived, already-warmed session removes both: there is
#: only ever one session's connection pools in play, for any thread, across
#: any number of overlapping or orphaned script reruns.
_session_lock = threading.Lock()
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """The process-wide shared session (see the module-level note above),
    created on first use. Double-checked locking: the lock is only ever taken
    on a cold start (or a race for the very first call), never on the
    steady-state path once ``_session`` is set."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
    return _session


def _default_max_workers(cap: int) -> int:
    """A worker-pool size that scales with the environment instead of a fixed
    literal, capped at *cap* (the widest fan-out a given fetch ever needs).

    Uses :func:`os.process_cpu_count` (Python 3.13+; respects a container's
    CPU affinity/quota, unlike :func:`os.cpu_count`, which reports the whole
    host's core count even inside a CPU-limited pod) plus a small constant
    headroom for the I/O-wait time these threads spend blocked on network
    calls rather than burning CPU — the same reasoning the standard library's
    own ``ThreadPoolExecutor`` default uses. Falls back to 1 CPU if the count
    is unavailable (some platforms/sandboxes report ``None``).
    """
    cpus = os.process_cpu_count() or 1
    return max(1, min(cap, cpus + 4))


def _list_subdirs(url: str) -> list[str]:
    """Return directory names (without trailing slash) from an Apache listing."""
    url = url.rstrip("/") + "/"
    resp = _get_session().get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    return [
        m.group(1).rstrip("/")
        for m in _DIR_LINK_RE.finditer(resp.text)
        if m.group(1).endswith("/") and not m.group(1).startswith("?")
    ]


def _list_files(url: str) -> list[str]:
    """Return file names (no slash) from an Apache directory listing."""
    url = url.rstrip("/") + "/"
    resp = _get_session().get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    return [
        m.group(1)
        for m in _DIR_LINK_RE.finditer(resp.text)
        if not m.group(1).endswith("/") and not m.group(1).startswith("?")
    ]


# ── Discovery helpers (one per hierarchy level) ───────────────────────────────

def list_detectors(base_url: str) -> list[str]:
    """Return available detector names.

    Underscore-prefixed directories at the top level (e.g. ``_reports/``, where
    the nightly regression reports live) are reserved for non-detector data and
    skipped.
    """
    return [d for d in _list_subdirs(base_url) if not d.startswith("_")]


def list_platforms(base_url: str, detector: str) -> list[str]:
    """Return available platforms for *detector*."""
    return _list_subdirs(f"{base_url.rstrip('/')}/{detector}")


def list_samples(base_url: str, detector: str, platform: str, stack: str) -> list[str]:
    """Return available physics samples for *(detector, platform, stack)*."""
    return sorted(_list_subdirs(
        f"{base_url.rstrip('/')}/{detector}/{platform}/{stack}"
    ))


def scan_stack_samples(
    base_url: str, detector: str, platform: str
) -> dict[str, list[str]]:
    """Return ``{stack: [samples]}`` for *(detector, platform)*, newest stack first.

    Single source of truth for both the cross-release sample union and the
    per-sample stack list, so the sidebar scans the release tree only once
    (one listing per stack) instead of twice. A sample may be added or dropped
    between Key4hep releases; callers derive the union and the per-sample stacks
    from this map. Stacks whose listing fails are skipped (logged at debug).
    """
    stacks = _list_subdirs(f"{base_url.rstrip('/')}/{detector}/{platform}")
    out: dict[str, list[str]] = {}
    for stack in sorted(stacks, reverse=True):
        try:
            out[stack] = list_samples(base_url, detector, platform, stack)
        except requests.RequestException as exc:
            _log.debug("scan_stack_samples: skipping stack %s — %s", stack, exc)
    return out


def list_runs(
    base_url: str,
    detector: str,
    platform: str,
    stack: str,
    sample: str,
) -> list[str]:
    """Return available run dates for a *(detector, platform, stack, sample)* combination, newest first."""
    runs = _list_subdirs(
        f"{base_url.rstrip('/')}/{detector}/{platform}/{stack}/{sample}"
    )
    return sorted(runs, reverse=True)


def list_run_dates_all_stacks(
    base_url: str,
    detector: str,
    platform: str,
    sample: str,
) -> dict[str, list[str]]:
    """Return ``{stack: [run_dates]}`` for *(detector, platform, sample)*.

    Discovery only — issues directory listings and **no** file downloads. The
    run-date directory names are ``YYYY-MM-DD``, so the full set of available
    dates per stack is obtained cheaply; this populates the trend-window control
    and lets the caller download only the runs inside the selected window.
    Stacks that do not contain *sample* (or whose listing fails) are skipped.
    """
    stacks = _list_subdirs(f"{base_url.rstrip('/')}/{detector}/{platform}")
    out: dict[str, list[str]] = {}
    for stack in stacks:
        url = f"{base_url.rstrip('/')}/{detector}/{platform}/{stack}/{sample}"
        try:
            runs = _list_subdirs(url)
        except requests.RequestException as exc:
            _log.debug("list_run_dates_all_stacks: skipping %s — %s", url, exc)
            continue  # sample may not exist for every stack
        if runs:
            out[stack] = sorted(runs)
    return out


# ── Download (persistent immutable cache) ──────────────────────────────────────

def _default_cache_root() -> Path:
    return Path(
        os.environ.get(
            "K4BENCH_CACHE_DIR", str(Path(tempfile.gettempdir()) / "k4bench_cache")
        )
    )


def ensure_run_cached(
    base_url: str,
    detector: str,
    platform: str,
    stack: str,
    sample: str,
    date: str,
    cache_root: str | None = None,
) -> Path:
    """Download one run into a stable cache path and return it.

    Cache layout: ``{cache_root}/{detector}/{platform}/{stack}/{sample}/{date}/``.
    Historical runs are immutable, so a run whose ``.complete`` sentinel exists is
    returned without any HTTP. To stay correct across concurrent reruns and
    processes, files are downloaded into a private temp dir and the finished run
    is published with a single atomic ``rename``: a reader therefore never sees a
    half-written ``run_dir``, and an interrupted download leaves no partial run
    behind (the temp dir is discarded) rather than a dir that looks cached.
    """
    root = Path(cache_root) if cache_root else _default_cache_root()
    run_dir = root / detector / platform / stack / sample / date
    sentinel = run_dir / ".complete"
    if sentinel.exists():
        return run_dir

    run_url = f"{base_url.rstrip('/')}/{detector}/{platform}/{stack}/{sample}/{date}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage in a sibling temp dir (same filesystem, so the publish rename is atomic).
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{date}.tmp-", dir=run_dir.parent))
    session = _get_session()
    try:
        for fname in _list_files(run_url):
            # Validate the *decoded* name: a percent-encoded separator (e.g.
            # "%2e%2e%2fevil.csv" → "../evil.csv") slips past a raw-name check but
            # escapes the run dir once written, so decode first, then reject any
            # name that is not a single, plain path component.
            decoded = unquote(fname)
            if (
                not decoded
                or decoded in (".", "..")
                or "/" in decoded
                or "\\" in decoded
                or Path(decoded).name != decoded
            ):
                raise ValueError(f"Unsafe filename in listing: {fname!r}")
            resp = session.get(f"{run_url}/{fname}", timeout=_TIMEOUT)
            resp.raise_for_status()
            (tmp_dir / decoded).write_bytes(resp.content)
        (tmp_dir / ".complete").write_text("")
        _publish_run_dir(tmp_dir, run_dir, sentinel)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return run_dir


def _publish_run_dir(tmp_dir: Path, run_dir: Path, sentinel: Path) -> None:
    """Atomically move a fully-staged *tmp_dir* into its final *run_dir*."""
    try:
        os.replace(tmp_dir, run_dir)
    except OSError:
        # ``run_dir`` already exists and is non-empty. Either another worker
        # published the same immutable run concurrently (sentinel present → trust
        # it, drop our copy) or a stale partial dir from an interrupted attempt is
        # in the way (clear it and retry, since rename can't replace a non-empty dir).
        if sentinel.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return
        shutil.rmtree(run_dir, ignore_errors=True)
        os.replace(tmp_dir, run_dir)


#: Widest fan-out any single fetch in this module uses — the cap passed to
#: :func:`_default_max_workers` by :func:`fetch_runs_windowed` (whose own
#: default was a flat, unconditional 16 before).
_MAX_RUN_FETCH_WORKERS = 16


def fetch_runs_windowed(
    base_url: str,
    detector: str,
    platform: str,
    sample: str,
    stacks_dates: dict[str, list[str]],
    cache_root: str | None = None,
    max_workers: int | None = None,
) -> list[dict]:
    """Fetch every ``(stack, date)`` in *stacks_dates* in parallel, returning a
    list of ``{"stack", "date", "run_dir"}`` for the runs successfully cached.

    Each run is fetched at most once (see :func:`ensure_run_cached`); callers pass
    an already date-windowed *stacks_dates* so only in-window runs are downloaded.
    Runs that fail to download are logged and skipped rather than aborting the load.

    *max_workers* defaults to :func:`_default_max_workers` (environment-scaled
    rather than a flat literal — a CPU-limited pod gets fewer threads than a
    developer's workstation) capped at :data:`_MAX_RUN_FETCH_WORKERS`, the
    widest fan-out this module needs. All workers share the module's one
    process-wide session (:func:`_get_session`), so the actual concurrency
    limit here is threads-doing-I/O-at-once, not TLS-session churn.
    """
    tasks = [(stack, date) for stack, dates in stacks_dates.items() for date in dates]
    if not tasks:
        return []
    if max_workers is None:
        max_workers = _default_max_workers(_MAX_RUN_FETCH_WORKERS)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = {
            pool.submit(
                ensure_run_cached,
                base_url, detector, platform, stack, sample, date, cache_root,
            ): (stack, date)
            for stack, date in tasks
        }
        for fut in as_completed(futures):
            stack, date = futures[fut]
            try:
                run_dir = fut.result()
            except (requests.RequestException, ValueError, OSError) as exc:
                _log.warning("fetch_runs_windowed: failed %s/%s — %s", stack, date, exc)
                continue
            results.append({"stack": stack, "date": date, "run_dir": str(run_dir)})
    return results


def list_stacks(base_url: str, detector: str, platform: str) -> list[str]:
    """Return the Key4hep releases benchmarked for *(detector, platform)*, newest first.

    Directory names as stored (``key4hep-{YYYY-MM-DD}``). Discovery only — one
    listing, no downloads.
    """
    return sorted(
        _list_subdirs(f"{base_url.rstrip('/')}/{detector}/{platform}"), reverse=True
    )


def fetch_stack_packages(
    base_url: str, detector: str, platform: str, stack: str
) -> dict | None:
    """Return the ``k4h_packages`` map of a release, or ``None``.

    Every detector benchmarked against a given release sourced the *same*
    stack, so any one run under it answers the question — this walks to the
    first run it finds and reads only that ``run_info.json`` (two listings and
    one small GET), rather than downloading a run directory.

    ``None`` covers both "no run found" and "that run predates provenance
    capture": in either case the release's packages are unknown, which a caller
    must not confuse with an empty stack.
    """
    root = f"{base_url.rstrip('/')}/{detector}/{platform}/{stack}"
    try:
        samples = sorted(_list_subdirs(root))
    except requests.RequestException as exc:
        _log.debug("fetch_stack_packages: no samples under %s — %s", root, exc)
        return None

    for sample in samples:
        try:
            dates = sorted(_list_subdirs(f"{root}/{sample}"), reverse=True)
        except requests.RequestException:
            continue
        for date in dates:
            url = f"{root}/{sample}/{date}/run_info.json"
            try:
                resp = _get_session().get(url, timeout=_TIMEOUT)
                resp.raise_for_status()
                packages = resp.json().get("k4h_packages")
            except (requests.RequestException, ValueError) as exc:
                _log.debug("fetch_stack_packages: %s — %s", url, exc)
                continue
            if packages:
                return packages
    return None


def list_report_dates(base_url: str) -> list[str]:
    """Return available nightly regression-report dates (newest first).

    Reports live at ``{base_url}/_reports/{YYYY-MM-DD}/report.json``, written
    by the nightly ``regression-report`` CI job. An absent ``_reports/`` tree
    (no report generated yet) is not an error — it returns an empty list.
    """
    try:
        return sorted(_list_subdirs(f"{base_url.rstrip('/')}/_reports"), reverse=True)
    except requests.RequestException as exc:
        _log.debug("list_report_dates: no _reports tree — %s", exc)
        return []


def fetch_report(base_url: str, date: str) -> dict | None:
    """Fetch and parse one nightly regression report, or ``None`` on failure."""
    url = f"{base_url.rstrip('/')}/_reports/{date}/report.json"
    try:
        resp = _get_session().get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        _log.warning("fetch_report: could not fetch %s — %s", url, exc)
        return None


def ensure_latest_run_cached(
    base_url: str,
    detector: str,
    platform: str,
    stack: str,
    sample: str,
    cache_root: str | None = None,
) -> Path | None:
    """Cache and return only the newest run for *(detector, platform, stack, sample)*.

    Single-run tabs only ever display the latest run, so there is no need to
    download the full date history for the selected stack.
    """
    runs = list_runs(base_url, detector, platform, stack, sample)  # newest first
    if not runs:
        return None
    return ensure_run_cached(
        base_url, detector, platform, stack, sample, runs[0], cache_root
    )
