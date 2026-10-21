from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from k4bench.analysis.plots import plot_event_memory
from k4bench.analysis.plots._utils import _drop_failed_reads
from sections import TREND_WINDOW_SCOPE, SectionScope
from stats import build_event_stats_table, style_stats_table
from tabs._historical_trends import render_historical_trends, render_stability_expander
from tabs._reliability import render_reliability_filter
from ui_utils import (
    baseline_selector_control,
    cached_event_bin_options,
    config_selector_control,
    histogram_display_controls,
    is_valid_df,
    PALETTES,
    view_control_row,
)


#: Per-event RSS kinds either view can show: label → event column. A kind is
#: offered only when the data on screen records it.
_MEMORY_KINDS = {
    "Total RSS": "rss_end_mb",
    "Anonymous RSS": "rss_anon_end_mb",
    "File-backed RSS": "rss_file_end_mb",
}

#: Shown first when recorded: the only event memory metric the regression
#: engine judges. Total and file-backed RSS include pages CVMFS publishes can
#: evict mid-run. Runs without the split fall back to the first kind offered.
_DEFAULT_KIND = "Anonymous RSS"

#: Warmup events the current-run view leaves out of plot and bins.
_EXCLUDE_EVENTS = (0,)

#: Historical statistics of each kind: (trend column, panel title).
_HIST_STATS = {
    "Total RSS": [
        ("median_rss_mb", "Median total RSS (MB)"),
        ("mean_rss_mb", "Mean total RSS (MB)"),
        ("std_rss_mb", "Total RSS std dev (MB)"),
    ],
    "Anonymous RSS": [
        ("median_rss_anon_mb", "Median anonymous RSS (MB)"),
        ("mean_rss_anon_mb", "Mean anonymous RSS (MB)"),
        ("std_rss_anon_mb", "Anonymous RSS std dev (MB)"),
        ("rss_anon_slope_mb_per_event", "Anonymous RSS growth (MB/event)"),
    ],
    "File-backed RSS": [
        ("mean_rss_file_mb", "Mean file-backed RSS (MB)"),
        ("std_rss_file_mb", "File-backed RSS std dev (MB)"),
    ],
}

#: Metrics of each kind whose run-to-run movement the historical view reports.
_STABILITY_METRICS = {
    "Total RSS": ["mean_rss_mb"],
    "Anonymous RSS": ["mean_rss_anon_mb", "rss_anon_slope_mb_per_event"],
    "File-backed RSS": ["mean_rss_file_mb"],
}

# Each statistic uses its own component's spread and valid event count.
_HIST_ERROR_SOURCES = {
    **{f"{stat}_rss_mb": ("std_rss_mb", "n_events_rss") for stat in ("median", "mean", "std")},
    **{
        f"{stat}_rss_anon_mb": ("std_rss_anon_mb", "n_events_rss_anon")
        for stat in ("median", "mean", "std")
    },
    **{
        f"{stat}_rss_file_mb": ("std_rss_file_mb", "n_events_rss_file")
        for stat in ("mean", "std")
    },
}

#: Sub-views, in dispatch order; the first is the fallback when the tab has no
#: history to offer.
_VIEWS = ["Current Run", "Historical Trends"]


def _memory_kind_control(available: list[str], key: str) -> str:
    """Render the "Memory" selectbox over *available* kinds and return the choice."""
    if key in st.session_state and st.session_state[key] not in available:
        st.session_state.pop(key, None)
    return st.selectbox(
        "Memory",
        options=available,
        index=available.index(_DEFAULT_KIND) if _DEFAULT_KIND in available else 0,
        key=key,
        bind="query-params",
        help=(
            "Which RSS, sampled at the end of each event, to show: anonymous "
            "(heap) RSS, the one regressions are judged on; total RSS; or its "
            "file-backed (mapped files) part, which CVMFS publishes can evict "
            "mid-run."
        ),
    )


def _column_data(event_data: dict, column: str) -> dict:
    """Return the runs with a valid post-warmup *column* sample, without failed reads."""
    out = {}
    for label, df in event_data.items():
        if column not in df.columns:
            continue
        df = _drop_failed_reads(df, column)
        if (~df["event_number"].isin(_EXCLUDE_EVENTS)).any():
            out[label] = df
    return out


def _render_current_run(
    event_data: dict,
    display_options_slot=None,
) -> None:
    """Render the current-run per-event memory view."""
    if not event_data:
        st.info("No event memory data available in this run.")
        return

    data_by_kind = {
        kind: _column_data(event_data, column) for kind, column in _MEMORY_KINDS.items()
    }
    available = [kind for kind, data in data_by_kind.items() if data]
    if not available:
        st.info("No event memory data available in this run.")
        return

    col_baseline, col_configs, col_kind = st.columns(
        [1, 3, 1], gap="medium", vertical_alignment="bottom",
    )
    with col_kind:
        kind = _memory_kind_control(available, "evt_memory_kind")
    column = _MEMORY_KINDS[kind]
    memory_data = data_by_kind[kind]
    current_labels = sorted(memory_data)

    with col_baseline:
        baseline_label = baseline_selector_control("evt_memory", current_labels)
    with col_configs:
        display_labels = config_selector_control(
            "evt_memory", memory_data, current_labels, baseline_label, column, "MB",
        )

    bin_options = cached_event_bin_options(
        memory_data, column, tuple(display_labels), _EXCLUDE_EVENTS
    )
    bins, palette_name, alpha, show_errors, show_mean_lines = (
        histogram_display_controls(
            "evt_memory", bin_options, len(display_labels), display_options_slot,
        )
    )

    fig = plot_event_memory(
        memory_data,
        column=column,
        labels=display_labels,
        baseline_label=baseline_label,
        show="both",
        exclude_events=list(_EXCLUDE_EVENTS),
        palette=PALETTES[palette_name],
        bins=bins,
        alpha=alpha,
        show_errors=show_errors,
        show_mean_lines=show_mean_lines,
    )
    st.plotly_chart(fig, width="stretch", key="evt_memory_current_chart")

    st.subheader("Statistics")
    stats = build_event_stats_table(
        memory_data, display_labels, column, "MB", baseline_label, True
    )
    if not stats.empty:
        st.dataframe(style_stats_table(stats), width="stretch")
    else:
        st.info("No valid statistics available (missing or empty data).")

    if set(display_labels) != set(current_labels):
        with st.expander(f"All configurations ({len(current_labels)})"):
            all_stats = build_event_stats_table(
                memory_data, current_labels, column, "MB", baseline_label, True
            )
            if not all_stats.empty:
                st.dataframe(style_stats_table(all_stats), width="stretch")


def _render_historical(
    trend_event_df: pd.DataFrame,
    reliability: dict[str, bool | None] | None = None,
    reliability_slot=None,
    display_options_slot=None,
    all_runs: bool = False,
) -> None:
    """Render historical trends of the chosen RSS kind, every run of a release
    when *all_runs*."""
    if not is_valid_df(trend_event_df):
        st.info(
            "No event memory trend data in the selected window. "
            "Widen the trend window in the sidebar."
        )
        return
    avail_labels = sorted(trend_event_df["label"].unique())
    if not avail_labels:
        st.info("No historical event memory data in the selected window.")
        return

    all_runs_df = trend_event_df
    trend_event_df = render_reliability_filter(
        trend_event_df, reliability, key="evt_memory_hist_exclude_unreliable",
        slot=reliability_slot,
    )
    if trend_event_df.empty:
        return

    # Runs without the RSS split leave its columns all-NaN once newer runs are
    # filtered out, so a kind needs recorded values, not just its columns.
    present_by_kind = {
        kind: [
            (col, lbl) for col, lbl in stats
            if col in trend_event_df.columns and trend_event_df[col].notna().any()
        ]
        for kind, stats in _HIST_STATS.items()
    }
    available = [kind for kind, stats in present_by_kind.items() if stats]
    if not available:
        st.info("No historical event memory statistics available.")
        return
    col_kind, _ = st.columns([1, 4], vertical_alignment="bottom")
    with col_kind:
        kind = _memory_kind_control(available, "evt_memory_hist_kind")
    present_stats = present_by_kind[kind]

    render_historical_trends(
        trend_event_df,
        avail_labels,
        present_stats,
        std_col="std_rss_mb",
        n_col_candidates=["n_events_rss", "n_events"],
        error_sources=_HIST_ERROR_SOURCES,
        unit="MB",
        units={"rss_anon_slope_mb_per_event": "MB/event"},
        key_prefix="evt_memory_hist",
        no_data_msg="No event memory trend data in the selected window.",
        display_options_slot=display_options_slot,
        all_runs=all_runs,
    )
    # Unreliable runs are dropped inside regardless of the toggle above.
    render_stability_expander(
        all_runs_df, _STABILITY_METRICS[kind], reliability, key="evt_memory_hist_stability",
    )


def render(
    event_data: dict | None,
    load_history: Callable[[], tuple[pd.DataFrame | None, dict[str, bool | None]]] | None = None,
    trends_enabled: bool = False,
    *,
    all_runs: bool = False,
) -> SectionScope | None:
    """The per-event memory views. *load_history* loads the window's event
    trend and its runs' reliability, only for the history, which draws every
    run of a release when *all_runs*."""
    if event_data is None and not trends_enabled:
        st.info("No event memory data available in the selected directory.")
        return None

    # The "Historical Trends" option is gated on remote mode (not on the current
    # window's data) so the view selector stays put when the trend window changes.
    view, reliability_slot, display_options_slot = view_control_row(
        _VIEWS if trends_enabled else [_VIEWS[0]], key="evt_memory_view_mode",
    )

    if view == "Current Run":
        if event_data is None:
            st.info("No event memory data available in the selected directory.")
        else:
            _render_current_run(event_data, display_options_slot)
        return None
    trend_event_df, reliability = load_history() if load_history else (None, None)
    _render_historical(
        trend_event_df, reliability,
        reliability_slot=reliability_slot,
        display_options_slot=display_options_slot,
        all_runs=all_runs,
    )
    # The trends span the window's releases, not the sidebar's one — reported
    # so the scope note above stops naming it.
    return TREND_WINDOW_SCOPE
