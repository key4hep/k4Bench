"""Discover and download benchmark run data from a WebEOS HTTP endpoint.

The WebEOS site serves an Apache-style directory listing at DD4BENCH_DATA_URL.
Expected layout::

    {base_url}/
      {detector}/                         e.g. IDEA_o1_v03/
        {YYYY-MM-DD}_{k4h_release}/       e.g. 2026-05-22_key4hep-2026-04-08/
          run_info.json
          {config}_results.csv
          {config}_events.json
          {config}_regions.json
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


def list_detectors(base_url: str) -> list[str]:
    """Return available detector names from the WebEOS root."""
    return _list_subdirs(base_url)


def list_runs(base_url: str, detector: str) -> list[str]:
    """Return available run identifiers for *detector*, newest-first."""
    runs = _list_subdirs(f"{base_url.rstrip('/')}/{detector}")
    return sorted(runs, reverse=True)


def download_all_runs(base_url: str, detector: str) -> Path:
    """Download every run for *detector* into ``{tmpdir}/{run_id}/`` subdirectories.

    Returns the detector-level temp directory so callers can either pick the
    latest run (for single-run tabs) or walk all subdirs (for the Trends tab).
    Streamlit's ``@st.cache_data`` prevents redundant downloads across reruns.
    """
    runs = _list_subdirs(f"{base_url.rstrip('/')}/{detector}")
    dest = Path(tempfile.mkdtemp(prefix="dd4bench_"))
    for run in runs:
        run_dir = dest / run
        run_dir.mkdir()
        run_url = f"{base_url.rstrip('/')}/{detector}/{run}"
        for fname in _list_files(run_url):
            content = requests.get(f"{run_url}/{fname}", timeout=_TIMEOUT).content
            (run_dir / fname).write_bytes(content)
    return dest
