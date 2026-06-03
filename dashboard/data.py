from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from k4bench.analysis.loader import load_event_timing, load_region_timing, load_results  # noqa: F401 (re-exported)

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
    def _unknown_meta(run_dir: Path) -> dict:
        """Return a minimal metadata dict when no structured info is available."""
        return {
            "run_dir":          str(run_dir),
            "run_date":         pd.to_datetime(run_dir.name[:10], errors="coerce"),
            "platform":         "unknown",
            "k4h_release":      "unknown",
            "k4h_release_date": pd.NaT,
            "sample":           "unknown",
            "github_run_url":   None,
            "commit_sha":       None,
            "n_events":         None,
            "status":           None,
            "failed_configs":   [],
            "machine_consistent": None,
        }

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
                "status":           info.get("status"),
                "failed_configs":   info.get("failed_configs") or [],
                "machine_consistent": info.get("machine_consistent"),
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            _log.warning("_parse_run_dir: could not read run_info.json in '%s'", run_dir)
            # Do NOT fall through to path-based parsing: in the temp download
            # directories created by remote.py the parent path components are
            # meaningless tmp names, not the EOS hierarchy.
            return _unknown_meta(run_dir)

    # run_info.json absent — infer only the run date from the directory name.
    # Path-component parsing is intentionally omitted: run dirs live inside
    # temp directories whose parent paths carry no semantic meaning.
    _log.warning("_parse_run_dir: run_info.json missing in '%s'", run_dir)
    return _unknown_meta(run_dir)


def run_metadata(run_dir: str) -> dict:
    """Return metadata for a single run (date) directory.

    Thin public wrapper over :func:`_parse_run_dir` for callers that already hold
    one concrete run directory (e.g. the cached latest run for single-run tabs).
    """
    return _parse_run_dir(Path(run_dir))


@st.cache_data(show_spinner="Loading trend data...", ttl=3600)
def cached_load_trend_results(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Load results from the given *run_dirs* and return a combined DataFrame for
    the Trends tab.

    *run_dirs* is the already date-windowed set of cached run directories
    (see ``remote.fetch_runs_windowed``). Each row gets ``run_id``, ``run_date``,
    ``platform``, ``k4h_release``, and ``k4h_release_date`` columns.
    """
    if not run_dirs:
        return None
    frames = []
    for d in run_dirs:
        run_dir = Path(d)
        if not run_dir.is_dir():
            continue
        meta = _parse_run_dir(run_dir)
        try:
            df = load_results(run_dir)
        except _EXPECTED_ERRORS as exc:
            _log.debug("cached_load_trend_results: skipping '%s': %s", run_dir, exc)
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
    combined = pd.concat(frames, ignore_index=True)
    combined["run_date"]         = pd.to_datetime(combined["run_date"]).dt.normalize()
    combined["k4h_release_date"] = pd.to_datetime(combined["k4h_release_date"]).dt.normalize()
    combined["x_date"]           = combined["k4h_release_date"].fillna(combined["run_date"])
    return combined


@st.cache_data(show_spinner="Loading region timing trends...", ttl=3600)
def cached_load_trend_region_timing(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Load per-region timing summary across the given *run_dirs*.

    For each run directory, region timing is loaded for all available configs.
    Per detector per config per run, the median and mean event-level time are
    computed (event 0 excluded as warmup).

    *run_dirs* is the already date-windowed set of cached run directories
    (see ``remote.fetch_runs_windowed``).

    Returns a long-form DataFrame with columns:
        run_date, k4h_release_date, label, attribution, detector,
        median_time_s, mean_time_s
    or ``None`` if no data could be loaded.
    """
    if not run_dirs:
        return None

    rows: list[dict] = []
    for d in run_dirs:
        run_dir = Path(d)
        if not run_dir.is_dir():
            continue
        meta = _parse_run_dir(run_dir)
        try:
            region_data = load_region_timing(run_dir)
        except _EXPECTED_ERRORS as exc:
            _log.debug("cached_load_trend_region_timing: skipping '%s': %s", run_dir, exc)
            continue
        except Exception:
            _log.exception("cached_load_trend_region_timing: error loading '%s'", run_dir)
            continue

        for label, rdata in region_data.items():
            for attribution in ("at_location", "by_birth"):
                df_attr: pd.DataFrame = rdata.get(attribution)
                if df_attr is None or df_attr.empty:
                    continue
                # Exclude event 0 (warmup) from the index
                df_attr = df_attr[df_attr.index != 0]
                if df_attr.empty:
                    continue
                for detector in df_attr.columns:
                    vals = df_attr[detector].dropna().to_numpy()
                    if len(vals) == 0:
                        continue
                    s = pd.Series(vals)
                    n = len(vals)
                    rows.append({
                        "run_date":         meta["run_date"],
                        "k4h_release_date": meta["k4h_release_date"],
                        "k4h_release":      meta["k4h_release"],
                        "label":            label,
                        "attribution":      attribution,
                        "detector":         detector,
                        "n_events":         n,
                        "median_time_s":    float(s.median()),
                        "mean_time_s":      float(s.mean()),
                        "std_time_s":       float(s.std()) if n > 1 else 0.0,
                    })

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["run_date"]         = pd.to_datetime(df["run_date"]).dt.normalize()
    df["k4h_release_date"] = pd.to_datetime(df["k4h_release_date"]).dt.normalize()
    df["x_date"] = df["k4h_release_date"].fillna(df["run_date"])
    return df


@st.cache_data(show_spinner="Loading event timing trends...", ttl=3600)
def cached_load_trend_event_timing(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Load per-event timing and memory summary across the given *run_dirs*.

    For each run directory, event timing is loaded for all available configs.
    Per config per run, summary statistics are computed (event 0 excluded):
        mean_time_s, median_time_s, p95_time_s,
        mean_rss_mb, median_rss_mb, p95_rss_mb, max_rss_mb

    *run_dirs* is the already date-windowed set of cached run directories
    (see ``remote.fetch_runs_windowed``).

    Returns a long-form DataFrame with those columns plus
        run_date, k4h_release_date, k4h_release, label
    or ``None`` if no data could be loaded.
    """
    if not run_dirs:
        return None

    rows: list[dict] = []
    for d in run_dirs:
        run_dir = Path(d)
        if not run_dir.is_dir():
            continue
        meta = _parse_run_dir(run_dir)
        try:
            event_data = load_event_timing(run_dir)
        except _EXPECTED_ERRORS as exc:
            _log.debug("cached_load_trend_event_timing: skipping '%s': %s", run_dir, exc)
            continue
        except Exception:
            _log.exception("cached_load_trend_event_timing: error loading '%s'", run_dir)
            continue

        for label, df_ev in event_data.items():
            # Exclude event 0 (warmup)
            df_ev = df_ev[df_ev["event_number"] != 0]
            if df_ev.empty:
                continue
            row: dict = {
                "run_date":         meta["run_date"],
                "k4h_release_date": meta["k4h_release_date"],
                "k4h_release":      meta["k4h_release"],
                "label":            label,
            }
            if "event_time_s" in df_ev.columns:
                t = df_ev["event_time_s"].dropna()
                nt = len(t)
                if nt:
                    row["n_events"]      = nt
                    row["mean_time_s"]   = float(t.mean())
                    row["median_time_s"] = float(t.median())
                    row["p95_time_s"]    = float(t.quantile(0.95))
                    row["std_time_s"]    = float(t.std()) if nt > 1 else 0.0
            if "rss_end_mb" in df_ev.columns:
                r = df_ev["rss_end_mb"].dropna()
                nr = len(r)
                if nr:
                    row["n_events_rss"]  = nr
                    row["mean_rss_mb"]   = float(r.mean())
                    row["median_rss_mb"] = float(r.median())
                    row["p95_rss_mb"]    = float(r.quantile(0.95))
                    row["max_rss_mb"]    = float(r.max())
                    row["std_rss_mb"]    = float(r.std()) if nr > 1 else 0.0
            rows.append(row)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["run_date"]         = pd.to_datetime(df["run_date"]).dt.normalize()
    df["k4h_release_date"] = pd.to_datetime(df["k4h_release_date"]).dt.normalize()
    df["x_date"] = df["k4h_release_date"].fillna(df["run_date"])
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_machine_info(run_dir: str) -> dict | None:
    """Load ``machine_info.json`` from a run directory, or return ``None`` if absent."""
    path = Path(run_dir) / "machine_info.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
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
