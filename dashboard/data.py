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
    """Extract (run_date, platform, k4h_release, k4h_release_date) from a run directory.

    Prefers ``run_info.json`` when present; falls back to parsing the directory
    name for both the new format ``{date}_{platform}_key4hep-{release_date}``
    and the old format ``{date}_key4hep-{release_date}``.
    """
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        try:
            with open(info_path) as fh:
                info = json.load(fh)
            # Support both the new key "k4h_release" and the old key "key4hep_release".
            k4h_release = (
                info.get("k4h_release")
                or info.get("key4hep_release")
                or ""
            )
            # Derive k4h_release_date from the release tag when the explicit
            # field is absent (old run_info.json format).
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
            }
        except Exception:
            _log.warning("_parse_run_dir: could not read run_info.json in '%s'", run_dir)

    # ── Fallback: parse directory name ────────────────────────────────────
    name = run_dir.name
    # Both formats start with YYYY-MM-DD_
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})_(.*)", name)
    if date_match:
        run_date = pd.to_datetime(date_match.group(1), errors="coerce")
        rest = date_match.group(2)
    else:
        run_date = pd.NaT
        rest = name

    k4h_match = re.search(r"key4hep-(\d{4}-\d{2}-\d{2})", rest)
    if k4h_match:
        k4h_release = f"key4hep-{k4h_match.group(1)}"
        k4h_release_date = pd.to_datetime(k4h_match.group(1), errors="coerce")
        platform_part = rest[: rest.index("key4hep-")].rstrip("_")
        platform = platform_part if platform_part else "unknown"
        if platform == "unknown":
            _log.warning(
                "_parse_run_dir: could not determine platform from dir name '%s'; defaulting to 'unknown'",
                run_dir.name,
            )
    else:
        k4h_release = rest
        k4h_release_date = pd.NaT
        platform = "unknown"
        _log.warning(
            "_parse_run_dir: could not parse key4hep release from dir name '%s'; "
            "platform='unknown', k4h_release=%r",
            run_dir.name,
            k4h_release,
        )

    return {
        "run_dir":          str(run_dir),
        "run_date":         run_date,
        "platform":         platform,
        "k4h_release":      k4h_release,
        "k4h_release_date": k4h_release_date,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def list_run_metadata(detector_dir: str) -> list[dict]:
    """Return metadata dicts for every run subdirectory, cheaply (no CSV loading)."""
    meta = []
    for run_dir in sorted(Path(detector_dir).iterdir()):
        if run_dir.is_dir():
            meta.append(_parse_run_dir(run_dir))
    return meta


@st.cache_data(show_spinner="Loading trend data...", ttl=3600)
def cached_load_trend_results(detector_dir: str) -> pd.DataFrame | None:
    """Load results from all run subdirectories and return a combined DataFrame.

    Each row gets ``run_id``, ``run_date``, ``platform``, ``k4h_release``, and
    ``k4h_release_date`` columns so the Trends tab can plot metrics over time.
    """
    frames = []
    for run_dir in sorted(Path(detector_dir).iterdir()):
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
