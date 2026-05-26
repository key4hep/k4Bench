"""Discover and download benchmark run data from a WebEOS HTTP endpoint.

The WebEOS site serves an Apache-style directory listing at DD4BENCH_DATA_URL.
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

import re
import tempfile
from pathlib import Path

import requests

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


def list_stacks(base_url: str, detector: str, platform: str) -> list[str]:
    """Return available Key4hep stack releases for *(detector, platform)*, newest first."""
    stacks = _list_subdirs(f"{base_url.rstrip('/')}/{detector}/{platform}")
    return sorted(stacks, reverse=True)


def list_samples(base_url: str, detector: str, platform: str, stack: str) -> list[str]:
    """Return available physics samples for *(detector, platform, stack)*."""
    return sorted(_list_subdirs(
        f"{base_url.rstrip('/')}/{detector}/{platform}/{stack}"
    ))


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


# ── Download ──────────────────────────────────────────────────────────────────

def download_all_stacks_for_sample(
    base_url: str,
    detector: str,
    platform: str,
    sample: str,
) -> Path:
    """Download every run across **all** stacks for *(detector, platform, sample)*.

    Used by the Trends tab so it can plot performance across the full history of
    Key4hep nightly releases, not just the currently-selected stack.

    Returns a flat temp directory where each subdirectory is one run date
    (named ``{stack}__{date}`` to avoid collisions when two stacks share a date).
    Since ``_parse_run_dir`` prefers ``run_info.json`` for metadata, the directory
    name itself is irrelevant — all fields come from the JSON.
    """
    stacks = _list_subdirs(f"{base_url.rstrip('/')}/{detector}/{platform}")
    dest = Path(tempfile.mkdtemp(prefix="dd4bench_trends_"))
    for stack in stacks:
        stack_sample_url = (
            f"{base_url.rstrip('/')}/{detector}/{platform}/{stack}/{sample}"
        )
        try:
            runs = _list_subdirs(stack_sample_url)
        except Exception:
            continue  # sample may not exist for every stack — skip silently
        for run in runs:
            run_dir = dest / f"{stack}__{run}"
            run_dir.mkdir(exist_ok=True)
            run_url = f"{stack_sample_url}/{run}"
            for fname in _list_files(run_url):
                safe_name = Path(fname).name
                if not safe_name or safe_name != fname:
                    raise ValueError(f"Unsafe filename in listing: {fname!r}")
                resp = requests.get(f"{run_url}/{fname}", timeout=_TIMEOUT)
                resp.raise_for_status()
                (run_dir / safe_name).write_bytes(resp.content)
    return dest


def download_all_runs(
    base_url: str,
    detector: str,
    platform: str,
    stack: str,
    sample: str,
) -> Path:
    """Download every run date for *(detector, platform, stack, sample)* into
    ``{tmpdir}/{date}/`` subdirectories.

    Returns the sample-level temp directory so callers can either pick the
    latest run (for single-run tabs) or walk all subdirs (for the Trends tab).
    Streamlit's ``@st.cache_data`` prevents redundant downloads across reruns.
    """
    run_url_base = (
        f"{base_url.rstrip('/')}/{detector}/{platform}/{stack}/{sample}"
    )
    runs = _list_subdirs(run_url_base)
    dest = Path(tempfile.mkdtemp(prefix="dd4bench_"))
    for run in runs:
        run_dir = dest / run
        run_dir.mkdir(exist_ok=True)
        run_url = f"{run_url_base}/{run}"
        for fname in _list_files(run_url):
            safe_name = Path(fname).name
            if not safe_name or safe_name != fname:
                raise ValueError(f"Unsafe filename in listing: {fname!r}")
            resp = requests.get(f"{run_url}/{fname}", timeout=_TIMEOUT)
            resp.raise_for_status()
            (run_dir / safe_name).write_bytes(resp.content)
    return dest
