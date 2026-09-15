from __future__ import annotations

import math

import numpy as np
import pandas as pd

from k4bench.labels import METRIC_LABELS
from k4bench.regression.engine import recent_movement, same_release_spread

#: Stability table column holding the same-release repeat spread.
SAME_RELEASE_SPREAD_COL = "Same-release spread (stack changes excluded)"

#: Stability table column holding the all-runs movement.
MOVEMENT_COL = "Recent movement (all runs, incl. stack changes)"

#: Metrics whose typical value sits near zero, where a spread relative to the
#: median would be meaningless.
_ABSOLUTE_ONLY = {"rss_anon_slope_mb_per_event"}


def build_event_stats_table(
    event_data: dict,
    selected_labels: list[str], 
    column: str,
    unit: str,
    baseline_label: str | None,
    exclude_warmup: bool = True,
) -> pd.DataFrame:
    # Preserve event_data key order — this matches the palette index assignment in the plot.
    ordered_labels = [lbl for lbl in event_data if lbl in set(selected_labels)]

    rows = []
    for lbl in ordered_labels:
        df = event_data[lbl]
        if exclude_warmup and "event_number" in df.columns:
            df = df[df["event_number"] != 0]
        if column not in df.columns:
            continue
        arr = df[column].dropna().to_numpy()
        if len(arr) == 0:
            continue
        n = len(arr)
        mean = arr.mean()
        std = arr.std(ddof=1) if n > 1 else float("nan")
        sem = std / np.sqrt(n) if n > 1 else float("nan")
        se_std = std / np.sqrt(2 * (n - 1)) if n > 1 else float("nan")
        rows.append({
            "Run": lbl,
            "_mean": mean,
            f"Mean ({unit})": float(mean),
            f"SEM ({unit})": float(sem),
            f"Std ({unit})": float(std),
            f"SE(σ) ({unit})": float(se_std),
            f"Median ({unit})": float(np.median(arr)),
            f"Min ({unit})": float(arr.min()),
            f"Max ({unit})": float(arr.max()),
            "N events": n,
        })

    if not rows:
        return pd.DataFrame()

    stats_df = pd.DataFrame(rows).set_index("Run")

    if baseline_label and baseline_label in stats_df.index:
        baseline_mean = stats_df.loc[baseline_label, "_mean"]
        if baseline_mean != 0:
            stats_df["Ratio to baseline"] = stats_df["_mean"] / baseline_mean
            stats_df = stats_df.sort_values("Ratio to baseline")

    return stats_df


def select_top_n_by_ratio(
    event_data: dict,
    selected_labels: list[str],
    column: str,
    unit: str,
    baseline_label: str | None,
    exclude_warmup: bool,
    n: int,
) -> list[str]:
    """Return n labels: baseline + top (n-1) non-baseline runs by |ratio − 1|."""
    if len(selected_labels) <= n:
        return selected_labels
    if baseline_label is None:
        return selected_labels[:n]
    stats = build_event_stats_table(
        event_data, selected_labels, column, unit, baseline_label, exclude_warmup
    )
    if "Ratio to baseline" not in stats.columns:
        if baseline_label in selected_labels:
            return [baseline_label] + [lbl for lbl in selected_labels if lbl != baseline_label][: n - 1]
        return selected_labels[:n]
    non_bl = stats.index[stats.index != baseline_label]
    sorted_non_bl = (
        stats.loc[non_bl, "Ratio to baseline"]
        .sub(1).abs()
        .sort_values(ascending=False)
        .index
    )
    return [baseline_label] + list(sorted_non_bl[: n - 1])


def style_stats_table(stats_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    visible_cols = [c for c in stats_df.columns if not c.startswith("_")]
    float_cols = [
        c for c in visible_cols
        if stats_df[c].dtype == "float64" and c != "Ratio to baseline"
    ]
    fmt = {c: "{:.4g}" for c in float_cols}
    fmt["N events"] = "{:d}"
    if "Ratio to baseline" in visible_cols:
        fmt["Ratio to baseline"] = "{:.3f}"

    return stats_df[visible_cols].style.format(fmt)


def _fmt_spread(metric: str, spread: float, median: float, unit: str) -> str:
    text = f"{spread:.3g} {unit}".rstrip()
    if metric not in _ABSOLUTE_ONLY and math.isfinite(median) and median != 0:
        text += f" ({spread / abs(median) * 100:.2g}%)"
    return text


def build_stability_table(
    df: pd.DataFrame,
    metrics: dict[str, str],
    reliability: dict[str, bool | None] | None,
) -> pd.DataFrame:
    """How much each ``(config, metric)`` moves between runs, for display only.

    *metrics* maps each metric column to its display unit; columns absent from
    *df* are skipped. Per config, rows are put in the regression engine's order
    (release date, then run id — so a later rerun of an old release sits with
    that release, not at the end), runs the reliability map marks unreliable are
    dropped whatever the page's exclusion toggle says, and missing values are
    dropped so an absent night is a gap rather than a zero. Two readings follow:

    - the same-release spread
      (:func:`~k4bench.regression.engine.same_release_spread`), from
      consecutive runs of the *same* release only — stack changes excluded,
      but harness changes and noise included;
    - the recent movement (:func:`~k4bench.regression.engine.recent_movement`),
      over the last runs in that order regardless of release — also including
      stack changes. Descriptive only: it includes the newest release, which
      the engine's own noise floor for that release does not.

    Either reads ``N/A`` when the history is too short; a zero is never shown in
    place of a missing estimate. Returns an empty frame when no metric is
    present.
    """
    present = [m for m in metrics if m in df.columns]
    if not present or df.empty:
        return pd.DataFrame()
    reliability = reliability or {}
    ordered = df.sort_values(["x_date", "run_id"], kind="stable")
    ordered = ordered[[reliability.get(str(r)) is not False for r in ordered["run_id"]]]

    rows = []
    for label in sorted(ordered["label"].dropna().unique()):
        cfg = ordered[ordered["label"] == label]
        for metric in present:
            series = cfg[["x_date", metric]].dropna()
            if series.empty:
                continue
            values = series[metric].to_numpy(dtype=float)
            unit = metrics[metric]
            repeat = same_release_spread(values, series["x_date"].tolist())
            movement = recent_movement(values)
            rows.append({
                "Config": str(label),
                "Metric": METRIC_LABELS.get(metric, metric),
                SAME_RELEASE_SPREAD_COL: (
                    f"{_fmt_spread(metric, repeat[0], repeat[1], unit)} · {repeat[2]} pairs"
                    if repeat is not None else "N/A — too few same-release repeats"
                ),
                MOVEMENT_COL: (
                    _fmt_spread(metric, *movement, unit)
                    if movement is not None else "N/A"
                ),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Config")
