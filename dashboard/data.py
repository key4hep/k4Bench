from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from dd4bench.analysis.loader import load_event_timing, load_region_timing, load_results

_log = logging.getLogger(__name__)

_EXPECTED_ERRORS = (FileNotFoundError, ValueError, pd.errors.ParserError)


@st.cache_data(show_spinner="Loading benchmark results...")
def cached_load_results(data_dir: str) -> pd.DataFrame | None:
    try:
        df = load_results(data_dir)
        return df if not df.empty else None
    except _EXPECTED_ERRORS:
        return None
    except Exception:
        _log.exception("cached_load_results: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading benchmark results from '{data_dir}'. Check logs for details.")
        return None


@st.cache_data(show_spinner="Loading event timing...")
def cached_load_event_timing(data_dir: str) -> dict | None:
    try:
        d = load_event_timing(data_dir)
        return d if d else None
    except _EXPECTED_ERRORS:
        return None
    except Exception:
        _log.exception("cached_load_event_timing: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading event timing from '{data_dir}'. Check logs for details.")
        return None


@st.cache_data(show_spinner="Loading region timing...")
def cached_load_region_timing(data_dir: str) -> dict | None:
    try:
        d = load_region_timing(data_dir)
        return d if d else None
    except _EXPECTED_ERRORS:
        return None
    except Exception:
        _log.exception("cached_load_region_timing: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading region timing from '{data_dir}'. Check logs for details.")
        return None


def _parse_run_dir(run_dir: Path) -> dict:
    """Extract run metadata from a date-level run directory.

    Expected path structure::

        {detector}/{platform}/{stack}/{sample}/{YYYY-MM-DD}/

    Prefers ``run_info.json`` when present; falls back to inferring fields
    from the directory path.
    """
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        try:
            with open(info_path) as fh:
                info = json.load(fh)
            k4h_release = info.get("k4h_release") or ""
            k4h_release_date_raw = info.get("k4h_release_date")
            if not k4h_release_date_raw and k4h_release:
                m = re.search(r"(\d{4}-\d{2}-\d{2})", k4h_release)
                k4h_release_date_raw = m.group(1) if m else None
            return {
                "run_dir":          str(run_dir),
                "run_date":         pd.to_datetime(info.get("date"), errors="coerce"),
                "platform":         info.get("platform", "unknown") or "unknown",
                "k4h_release":      k4h_release,
                "k4h_release_date": pd.to_datetime(k4h_release_date_raw, errors="coerce"),
                "sample":           info.get("sample", "unknown") or "unknown",
                "github_run_url":   info.get("github_run_url"),
                "commit_sha":       info.get("commit_sha"),
                "n_events":         info.get("n_events"),
            }
        except Exception:
            _log.warning("_parse_run_dir: could not read run_info.json in '%s'", run_dir)

    # ── Fallback: infer from path structure ───────────────────────────────────
    # Expected: .../detector/platform/stack/sample/YYYY-MM-DD
    parts = run_dir.parts
    run_date = pd.to_datetime(run_dir.name, errors="coerce", format="%Y-%m-%d")

    # Walk up: [-1]=date, [-2]=sample, [-3]=stack, [-4]=platform, [-5]=detector
    try:
        k4h_release = parts[-3] if len(parts) >= 3 else "unknown"
        k4h_release_date_raw = re.search(r"\d{4}-\d{2}-\d{2}", k4h_release)
        k4h_release_date = (
            pd.to_datetime(k4h_release_date_raw.group(), errors="coerce")
            if k4h_release_date_raw else pd.NaT
        )
        platform = parts[-4] if len(parts) >= 4 else "unknown"
        sample   = parts[-2] if len(parts) >= 2 else "unknown"
    except Exception:
        k4h_release = k4h_release_date = platform = sample = "unknown"  # type: ignore[assignment]

    return {
        "run_dir":          str(run_dir),
        "run_date":         run_date,
        "platform":         platform,
        "k4h_release":      k4h_release,
        "k4h_release_date": k4h_release_date,
        "sample":           sample,
        "github_run_url":   None,
        "commit_sha":       None,
        "n_events":         None,
    }


@st.cache_data(show_spinner=False, ttl=60)
def list_run_metadata(sample_dir: str) -> list[dict]:
    """Return metadata dicts for every run (date) subdirectory of *sample_dir*."""
    meta = []
    for run_dir in sorted(Path(sample_dir).iterdir()):
        if run_dir.is_dir():
            meta.append(_parse_run_dir(run_dir))
    return meta


@st.cache_data(show_spinner="Loading trend data...", ttl=3600)
def cached_load_trend_results(sample_dir: str) -> pd.DataFrame | None:
    """Load results from all run-date subdirectories of *sample_dir* and return
    a combined DataFrame for the Trends tab.

    Each row gets ``run_id``, ``run_date``, ``platform``, ``k4h_release``, and
    ``k4h_release_date`` columns.
    """
    frames = []
    for run_dir in sorted(Path(sample_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        meta = _parse_run_dir(run_dir)
        try:
            df = load_results(run_dir)
        except _EXPECTED_ERRORS:
            continue
        except Exception:
            _log.exception("cached_load_trend_results: error loading '%s'", run_dir)
            continue
        df["run_id"]           = run_dir.name
        df["run_date"]         = meta["run_date"]
        df["platform"]         = meta["platform"]
        df["k4h_release"]      = meta["k4h_release"]
        df["k4h_release_date"] = meta["k4h_release_date"]
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False, ttl=60)
def load_machine_info(run_dir: str) -> dict | None:
    """Load ``machine_info.json`` from a run directory, or return ``None`` if absent."""
    path = Path(run_dir) / "machine_info.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        _log.warning("load_machine_info: could not read '%s'", path)
        return None


def collect_labels(
    results: pd.DataFrame | None,
    event_data: dict | None,
    region_data: dict | None,
) -> list[str]:
    labels: set[str] = set()
    if results is not None and "label" in results.columns:
        labels.update(results["label"].unique())
    if event_data is not None:
        labels.update(event_data.keys())
    if region_data is not None:
        labels.update(region_data.keys())
    return sorted(labels)
