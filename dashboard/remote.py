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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote

import requests

_log = logging.getLogger(__name__)

_DIR_LINK_RE = re.compile(r'href="([^"/][^"]*/?)"', re.IGNORECASE)
_TIMEOUT = 15


def _list_subdirs(url: str) -> list[str]:
    """Return directory names (without trailing slash) from an Apache listing."""
    url = url.rstrip("/") + "/"
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    return [
        m.group(1).rstrip("/")
        for m in _DIR_LINK_RE.finditer(resp.text)
        if m.group(1).endswith("/") and not m.group(1).startswith("?")
    ]


def _list_files(url: str) -> list[str]:
    """Return file names (no slash) from an Apache directory listing."""
    url = url.rstrip("/") + "/"
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    return [
        m.group(1)
        for m in _DIR_LINK_RE.finditer(resp.text)
        if not m.group(1).endswith("/") and not m.group(1).startswith("?")
    ]


# ── Discovery helpers (one per hierarchy level) ───────────────────────────────

def list_detectors(base_url: str) -> list[str]:
    """Return available detector names."""
    return _list_subdirs(base_url)


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
            resp = requests.get(f"{run_url}/{fname}", timeout=_TIMEOUT)
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


def fetch_runs_windowed(
    base_url: str,
    detector: str,
    platform: str,
    sample: str,
    stacks_dates: dict[str, list[str]],
    cache_root: str | None = None,
    max_workers: int = 16,
) -> list[dict]:
    """Fetch every ``(stack, date)`` in *stacks_dates* in parallel, returning a
    list of ``{"stack", "date", "run_dir"}`` for the runs successfully cached.

    Each run is fetched at most once (see :func:`ensure_run_cached`); callers pass
    an already date-windowed *stacks_dates* so only in-window runs are downloaded.
    Runs that fail to download are logged and skipped rather than aborting the load.
    """
    tasks = [(stack, date) for stack, dates in stacks_dates.items() for date in dates]
    if not tasks:
        return []

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
