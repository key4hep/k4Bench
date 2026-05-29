"""Impact Analysis tab — styled table for comparing subdetector configurations.

Answers the question: "If I remove one subdetector, which gives the most gain?"
"""
from __future__ import annotations

import matplotlib.cm as _mcm
import matplotlib.colors as _mc
import numpy as np
import pandas as pd
import streamlit as st

from dd4bench.analysis.plots._theme import _TEMPLATE  # noqa: F401 (kept for consistency)

# ── Metrics ───────────────────────────────────────────────────────────────────
_METRICS = [
    ("wall_time_s",    "Wall Time"),
    ("peak_rss_mb",    "Peak RSS"),
    ("user_cpu_s",     "User CPU"),
    ("events_per_sec", "Throughput"),
]
_LOWER_IS_BETTER = {"wall_time_s", "peak_rss_mb", "user_cpu_s"}

# ── Colour palettes — official matplotlib diverging colormaps ─────────────────
# We sample at 0.15 / 0.5 / 0.85 (not the extreme ends) so colours stay pastel
# and dark text is always legible — no contrast flip needed.
_CMAP_NAMES = ["PiYG", "PRGn", "BrBG", "RdBu", "RdYlGn", "Spectral"]


def _palette(cmap_name: str) -> tuple[str, str, str]:
    """Return (bad_hex, mid_hex, good_hex) sampled from a matplotlib diverging cmap."""
    cmap = _mcm.get_cmap(cmap_name)
    return (
        _mc.to_hex(cmap(0.15)),   # bad  — pastel "left" end
        _mc.to_hex(cmap(0.50)),   # mid  — neutral centre (≈ white for most diverging cmaps)
        _mc.to_hex(cmap(0.85)),   # good — pastel "right" end
    )


# ── Colour helpers ────────────────────────────────────────────────────────────

def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp(c1: tuple, c2: tuple, t: float) -> tuple[int, int, int]:
    return (
        int(c1[0] + t * (c2[0] - c1[0])),
        int(c1[1] + t * (c2[1] - c1[1])),
        int(c1[2] + t * (c2[2] - c1[2])),
    )


def _score_to_css(score: float, bad: str, mid: str, good: str) -> str:
    """Map a quality score [0 = worst, 0.5 = neutral/baseline, 1 = best]
    to a CSS background string.  Palettes are pastel so dark text always works."""
    s = max(0.0, min(1.0, score))
    bc, mc, gc = _hex_rgb(bad), _hex_rgb(mid), _hex_rgb(good)
    if s <= 0.5:
        r, g, b = _lerp(bc, mc, s * 2)
    else:
        r, g, b = _lerp(mc, gc, (s - 0.5) * 2)
    return f"background-color: rgb({r},{g},{b}); color: #111827;"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _prep_data(trend_df: pd.DataFrame, selected_labels: list[str]) -> pd.DataFrame:
    # Dates and x_date are already normalised by cached_load_trend_results.
    df = trend_df[trend_df["label"].isin(selected_labels)].copy()
    df["x_date"]   = pd.to_datetime(df["x_date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.dropna(subset=["x_date"])
    df = df.loc[df.groupby(["label", "x_date"])["run_date"].idxmax()]
    return df.loc[df.groupby("label")["x_date"].idxmax()].copy()


# ── Main render ───────────────────────────────────────────────────────────────

def render(trend_df: pd.DataFrame | None, selected_labels: list[str]) -> None:
    if trend_df is None:
        st.info("No trend data available. Run the nightly benchmark at least once.")
        return
    if not selected_labels:
        st.info("Select at least one configuration in the sidebar.")
        return

    snapshot = _prep_data(trend_df, selected_labels)
    snap_labels = [lbl for lbl in selected_labels if lbl in snapshot["label"].values]
    if not snap_labels:
        st.warning("No snapshot data for the selected configurations.")
        return
    missing_snap = sorted(set(selected_labels) - set(snap_labels))
    if missing_snap:
        st.warning(f"No recent snapshot data for: {', '.join(missing_snap)}")

    present = [(col, lbl) for col, lbl in _METRICS if col in snapshot.columns]
    if not present:
        st.warning("No supported metrics found.")
        return

    snapshot = snapshot.set_index("label")
    metric_cols   = [c for c, _ in present]
    metric_labels = [lbl for _, lbl in present]
    raw = snapshot[metric_cols].loc[snap_labels]

    # ── Controls ──────────────────────────────────────────────────────────────
    st.subheader("Which configuration gives the most gain?")
    st.caption(
        "Each cell shows the config's metric as **% of the baseline**. "
        "100 % is always the neutral midpoint."
    )

    ctrl_bl, ctrl_sort, ctrl_pal, _ = st.columns([2, 2, 2, 3])
    with ctrl_bl:
        baseline_label = st.selectbox(
            "Baseline config",
            options=snap_labels,
            index=0,
            key="impact_baseline",
            help=(
                "The configuration that appears as **100%** in every column. "
                "All other configs are expressed as a percentage of this reference — "
                "below 100% is better for time/memory, above 100% is better for throughput."
            ),
        )
    with ctrl_sort:
        wall_default = next(
            (i for i, (col, _) in enumerate(present) if col == "wall_time_s"), 0
        )
        sort_by = st.selectbox(
            "Sort rows by", options=metric_labels, index=wall_default, key="impact_sort"
        )
    with ctrl_pal:
        palette_name = st.selectbox(
            "Colour palette", options=_CMAP_NAMES, index=_CMAP_NAMES.index("PRGn"), key="impact_palette"
        )

    bad_hex, mid_hex, good_hex = _palette(palette_name)

    # ── Build percentage DataFrame ────────────────────────────────────────────
    pct_df = pd.DataFrame(index=snap_labels, columns=metric_labels, dtype=float)
    for col, lbl in present:
        bl_val = raw.loc[baseline_label, col]
        if pd.isna(bl_val) or bl_val == 0:
            # Missing / zero baseline → leave the whole column as NaN rather
            # than propagating nonsense percentages silently.
            pct_df[lbl] = np.nan
        else:
            pct_df[lbl] = raw[col] / float(bl_val) * 100.0

    # ── Sort rows ─────────────────────────────────────────────────────────────
    sort_col = next((c for c, lbl in present if lbl == sort_by), None)
    if sort_col is not None:
        pct_df = pct_df.sort_values(sort_by, ascending=sort_col in _LOWER_IS_BETTER)

    snap_labels_sorted = pct_df.index.tolist()

    # ── Compute quality scores [0 = worst, 0.5 = baseline, 1 = best] ─────────
    # Centred at 100 % — the baseline row always scores exactly 0.5.
    score_df = pd.DataFrame(0.5, index=snap_labels_sorted, columns=metric_labels)
    winners: list[dict] = []

    for col, lbl in present:
        vals     = pct_df[lbl].values.astype(float)
        diffs    = vals - 100.0
        max_abs  = np.nanmax(np.abs(diffs))
        if max_abs > 0:
            if col in _LOWER_IS_BETTER:
                scores = 0.5 + (-diffs) / (2.0 * max_abs)
            else:
                scores = 0.5 + diffs / (2.0 * max_abs)
            score_df[lbl] = np.clip(scores, 0.0, 1.0)

        # Best alternative per metric (excluding baseline)
        others_mask = pct_df.index != baseline_label
        if others_mask.any():
            other_vals = pct_df.loc[others_mask, lbl]
            if col in _LOWER_IS_BETTER:
                winner_lbl = other_vals.idxmin()
                delta_d    = float(other_vals.min()) - 100.0
                arrow, clr = ("▼", "normal") if delta_d <= 0 else ("▲", "inverse")
            else:
                winner_lbl = other_vals.idxmax()
                delta_d    = float(other_vals.max()) - 100.0
                arrow, clr = ("▲", "normal") if delta_d >= 0 else ("▼", "inverse")
            winners.append(dict(
                label=lbl, winner=winner_lbl,
                delta=delta_d, arrow=arrow, delta_color=clr,
            ))

    # ── Styled table ──────────────────────────────────────────────────────────
    def _apply_colors(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        for lbl in df.columns:
            for row_lbl in df.index:
                score = float(score_df.loc[row_lbl, lbl])
                out.loc[row_lbl, lbl] = _score_to_css(score, bad_hex, mid_hex, good_hex)
        return out

    baseline_row_style = "font-style: italic; border-left: 3px solid #9ca3af;"

    styled = (
        pct_df.style
        .apply(_apply_colors, axis=None)
        .format("{:.1f}%")
        .apply_index(
            lambda idx: [
                baseline_row_style if v == baseline_label else ""
                for v in idx
            ],
            axis=0,
        )
    )

    if winners:
        st.markdown("**Best alternative per metric**")
        cols = st.columns(len(winners))
        for k, w in enumerate(winners):
            with cols[k]:
                st.metric(
                    label=w["label"],
                    value=w["winner"],
                    delta=f"{w['arrow']} {abs(w['delta']):.1f}% vs baseline",
                    delta_color=w["delta_color"],
                )

    st.dataframe(styled)
