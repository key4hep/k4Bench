"""Load benchmark results and per-event timing data for analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd



def load_results(log_dir: str | Path, labels: list[str] | None = None) -> pd.DataFrame:
    """Load benchmark results from a log directory into a DataFrame.

    Each ``{label}_results.csv`` file written by ``dd4bench`` is loaded and
    concatenated into a single DataFrame.

    Parameters
    ----------
    log_dir : str or Path
        Directory containing ``*_results.csv`` files.
    labels : list[str] or None
        Load only these run labels.  Loads all ``*_results.csv`` files when ``None``.

    Returns
    -------
    pd.DataFrame
        One row per run. Float columns are cast to ``float64``;
        integer columns that may contain NaN use nullable ``Int64``.
    """
    log_dir = Path(log_dir)
    _suffix = "_results.csv"

    if labels is not None:
        candidates = [(log_dir / f"{lbl}{_suffix}", lbl) for lbl in labels]
        missing = [lbl for path, lbl in candidates if not path.exists()]
        if missing:
            raise ValueError(f"Missing result files for labels: {missing}")
        paths = [path for path, _ in candidates if path.exists()]
    else:
        paths = sorted(log_dir.glob(f"*{_suffix}"))

    if not paths:
        raise ValueError(f"No *_results.csv files found in '{log_dir}'.")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    float_cols = [
        "wall_time_s", "user_cpu_s", "sys_cpu_s",
        "peak_rss_mb", "output_size_mb", "events_per_sec",
    ]
    int_cols = [
        "returncode", "n_events",
        "major_page_faults", "voluntary_ctx_switches", "involuntary_ctx_switches",
    ]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def load_event_timing(
    log_dir: str | Path,
    labels: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load per-event timing JSON files from a log directory.

    Each ``{label}_events.json`` file written by the DD4benchTimingAction
    plugin is parsed into a DataFrame.

    Parameters
    ----------
    log_dir : str or Path
        Directory containing ``*_events.json`` files.
    labels : list[str] or None
        Load only these run labels. If None, all ``*_events.json`` files
        in *log_dir* are loaded.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps label → DataFrame with columns
        ``event_number``, ``event_time_s``, ``rss_begin_mb``,
        ``rss_end_mb``, ``rss_delta_mb``.
    """
    log_dir = Path(log_dir)
    _suffix = "_events.json"

    if labels is not None:
        candidates = [(log_dir / f"{lbl}{_suffix}", lbl) for lbl in labels]
    else:
        candidates = [
            (p, p.name[: -len(_suffix)])
            for p in sorted(log_dir.glob(f"*{_suffix}"))
        ]

    if labels is not None:
        missing_files = [lbl for path, lbl in candidates if not path.exists()]
        if missing_files:
            raise ValueError(f"Missing event files for labels: {missing_files}")

    out: dict[str, pd.DataFrame] = {}
    for path, label in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            raw = json.load(f)
        _required = ["event_numbers", "event_times_s", "event_rss_begin_mb", "event_rss_end_mb"]
        _missing = [k for k in _required if k not in raw]
        if _missing:
            raise ValueError(f"{path} missing keys: {_missing}")
        lengths = {k: len(raw[k]) for k in _required}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"{path} has mismatched array lengths: {lengths}")
        df = pd.DataFrame(
            {
                "event_number": raw["event_numbers"],
                "event_time_s": raw["event_times_s"],
                "rss_begin_mb": raw["event_rss_begin_mb"],
                "rss_end_mb":   raw["event_rss_end_mb"],
            }
        )
        df["rss_delta_mb"] = df["rss_end_mb"] - df["rss_begin_mb"]
        out[label] = df

    return out


def load_region_timing(
    log_dir: str | Path,
    labels: list[str] | None = None,
) -> dict[str, dict]:
    """Load per-region timing JSON files from a log directory.

    Each ``{label}_regions.json`` file written by the DD4benchRegionTimingAction
    plugin is parsed into structured data.

    Parameters
    ----------
    log_dir : str or Path
        Directory containing ``*_regions.json`` files.
    labels : list[str] or None
        Load only these run labels.  If ``None``, all ``*_regions.json`` files
        in *log_dir* are loaded.

    Returns
    -------
    dict[str, dict]
        Maps label → dict with keys:

        - ``"meta"``: dict with schema_version, timer, overhead_ns, detectors,
          lv_counts.
        - ``"events"``: DataFrame with columns ``event_number``,
          ``event_wall_s``, ``event_region_sum_s``, ``event_unaccounted_s``.
        - ``"at_location"``: DataFrame indexed by ``event_number``, one column
          per top-level detector (seconds), time charged to where the Geant4
          step physically occurred.
        - ``"by_birth"``: same shape as ``at_location``, time charged to the
          detector where the primary track was created.
    """
    log_dir = Path(log_dir)
    _suffix = "_regions.json"

    if labels is not None:
        candidates = [(log_dir / f"{lbl}{_suffix}", lbl) for lbl in labels]
    else:
        candidates = [
            (p, p.name[: -len(_suffix)])
            for p in sorted(log_dir.glob(f"*{_suffix}"))
        ]

    if labels is not None:
        missing = [lbl for path, lbl in candidates if not path.exists()]
        if missing:
            raise ValueError(f"Missing region files for labels: {missing}")

    out: dict[str, dict] = {}
    for path, label in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            raw = json.load(f)

        _required = [
            "event_numbers", "event_wall_seconds",
            "event_region_sum_seconds", "event_unaccounted_seconds",
            "at_location_seconds", "by_birth_seconds",
        ]
        _missing = [k for k in _required if k not in raw]
        if _missing:
            raise ValueError(f"{path} missing keys: {_missing}")

        n_ev = len(raw["event_numbers"])
        for k in ["event_wall_seconds", "event_region_sum_seconds",
                  "event_unaccounted_seconds", "at_location_seconds", "by_birth_seconds"]:
            if len(raw[k]) != n_ev:
                raise ValueError(f"{path}: array length mismatch for '{k}'")

        events_df = pd.DataFrame({
            "event_number":       raw["event_numbers"],
            "event_wall_s":       raw["event_wall_seconds"],
            "event_region_sum_s": raw["event_region_sum_seconds"],
            "event_unaccounted_s": raw["event_unaccounted_seconds"],
        })

        ev_index = pd.Index(raw["event_numbers"], name="event_number")
        at_loc_df   = pd.DataFrame(raw["at_location_seconds"], index=ev_index).fillna(0.0)
        by_birth_df = pd.DataFrame(raw["by_birth_seconds"],    index=ev_index).fillna(0.0)

        out[label] = {
            "meta": {
                "schema_version":    raw.get("schema_version", 1),
                "attribution_method": raw.get("attribution", "dd4hep_top_level_detelement"),
                "timer":             raw.get("timer", "unknown"),
                "overhead_ns":       raw.get("per_step_timer_overhead_ns"),
                "detectors":         raw.get("indexed_top_level_detectors", []),
                "lv_counts":         raw.get("indexed_top_level_detector_lv_counts", {}),
            },
            "events":      events_df,
            "at_location": at_loc_df,
            "by_birth":    by_birth_df,
        }

    if not out:
        if labels is not None:
            raise ValueError(
                f"No region files found for labels={labels} in '{log_dir}'."
            )
        raise ValueError(f"No *_regions.json files found in '{log_dir}'.")

    return out
