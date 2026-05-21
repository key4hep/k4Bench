from __future__ import annotations

import numpy as np
import pandas as pd

from dd4bench.analysis.plots import PALETTE as _PALETTE


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
    for palette_idx, lbl in enumerate(ordered_labels):
        df = event_data[lbl]
        if exclude_warmup:
            df = df[df["event_number"] != 0]
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
            "_palette_idx": palette_idx,
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
            return [baseline_label] + [l for l in selected_labels if l != baseline_label][: n - 1]
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

    def _row_bg(row: pd.Series) -> list[str]:
        idx = int(stats_df.loc[row.name, "_palette_idx"])
        hex_c = _PALETTE[idx % len(_PALETTE)]
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        return [f"background-color: rgba({r},{g},{b},0.12)"] * len(row)

    return (
        stats_df[visible_cols]
        .style
        .format(fmt)
        .apply(_row_bg, axis=1)
    )
