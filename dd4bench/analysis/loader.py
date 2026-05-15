"""Load benchmark results and per-event timing data for analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Load benchmark results from a CSV file into a DataFrame.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV written by ``dd4bench`` (or
        :func:`~dd4bench.results.reporter.save_csv`).

    Returns
    -------
    pd.DataFrame
        One row per run. Float columns are cast to ``float64``;
        integer columns that may contain NaN use nullable ``Int64``.
    """
    df = pd.read_csv(csv_path)

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
