from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd
import streamlit as st

from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.plots import plot_event_timing
from k4bench.analysis.plots._utils import _default_baseline
from k4bench.analysis.plots.phases import latest_run_per_tag
from sections import TREND_WINDOW_SCOPE, SectionScope
from stats import build_event_stats_table, style_stats_table
from tabs._historical_trends import render_historical_trends
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


_HIST_STATS = [
    ("median_time_s", "Median time (s)"),
    ("mean_time_s",   "Mean time (s)"),
    ("std_time_s",    "Std dev (s)"),
]

#: Sub-views, in dispatch order; the first is the fallback when the tab has no
#: history to offer.
_VIEWS = ["Current Run", "Historical Trends"]

#: The prefix of the current-run view's widget keys.
_KEY_PREFIX = "evt_timing"

#: The "Timing of" choice that is the event's own time, before any algorithm.
_WHOLE_EVENT = "Whole event"
_TIMINGS_KEY = "evt_timing_timings"

#: The event that warms the job up, left out of every per-event view.
_WARM_UP_EVENT = 0


def _algorithm_times(components: Mapping[str, ComponentTiming]) -> dict[str, pd.Series]:
    """Each algorithm's own time in each event it ran in, every
    configuration's events together, the warm-up left out; slowest per event
    first."""
    times: dict[str, list[pd.Series]] = {}
    for timing in components.values():
        wall = timing.execute["wall_s"]
        wall = wall[wall.index != _WARM_UP_EVENT]
        for name in timing.algorithms():
            ran = wall[name].dropna()
            if len(ran):
                times.setdefault(name, []).append(ran)
    pooled = {name: pd.concat(parts, ignore_index=True) for name, parts in times.items()}
    return dict(sorted(pooled.items(), key=lambda item: -item[1].mean()))


def _per_event(seconds: float) -> str:
    return f"{seconds:.2f} s" if seconds >= 1 else f"{seconds * 1e3:.3g} ms"


def _slowest_algorithm(timing: ComponentTiming) -> str | None:
    """The algorithm of *timing* with the longest mean time per event, the
    warm-up left out; ``None`` when none ran."""
    wall = timing.execute["wall_s"]
    means = wall[wall.index != _WARM_UP_EVENT][timing.algorithms()].mean().dropna()
    return str(means.idxmax()) if len(means) else None


def _baseline_config(labels: list[str]) -> str:
    """The baseline the Baseline selector holds or will default to."""
    stored = st.session_state.get(f"{_KEY_PREFIX}_baseline")
    return stored if stored in labels else _default_baseline(labels)


def _timing_selector(components: Mapping[str, ComponentTiming], baseline: str) -> list[str]:
    """The "Timing of" multiselect: the whole event and any algorithms, each
    algorithm named with its mean time per event; the whole event and the
    *baseline* configuration's slowest algorithm at first. Never empty:
    cleared, it is the whole event."""
    times = _algorithm_times(components)
    slowest = _slowest_algorithm(components[baseline]) if baseline in components else None
    chosen = st.multiselect(
        "Timing of", [_WHOLE_EVENT, *times],
        default=[_WHOLE_EVENT, *([slowest] if slowest else [])], key=_TIMINGS_KEY,
        format_func=lambda name: (
            name if name == _WHOLE_EVENT else f"{name}  ·  {_per_event(times[name].mean())}"
        ),
        placeholder=_WHOLE_EVENT,
        help=(
            "The whole event, and any algorithms to set beside it: each one's own "
            "time in each event, what it spent outside the algorithms it ran "
            "itself. Algorithms are listed slowest first, with their mean time "
            "per event."
        ),
    )
    return chosen or [_WHOLE_EVENT]


def _algorithm_event_data(
    components: Mapping[str, ComponentTiming], algorithm: str,
) -> dict[str, pd.DataFrame]:
    """*algorithm*'s own time in each event it ran in, per configuration, as
    the per-event views read an event's time; configurations it did not run
    in are left out."""
    data = {}
    for label, timing in components.items():
        wall = timing.execute["wall_s"]
        if algorithm not in wall.columns:
            continue
        ran = wall[algorithm].dropna()
        if len(ran):
            data[label] = pd.DataFrame({
                "event_number": ran.index.to_numpy(), "event_time_s": ran.to_numpy(),
            })
    return data


def _series_name(timing: str, config: str, timings: list[str], configs: list[str]) -> str:
    """What one timing of one configuration is called on the chart: the
    configuration while there is one timing, the timing while there is one
    configuration, both otherwise."""
    if len(timings) == 1:
        return config
    if len(configs) == 1:
        return timing
    return f"{timing} · {config}"


def _series(
    timings: Mapping[str, Mapping[str, pd.DataFrame]], configs: list[str],
) -> dict[str, pd.DataFrame]:
    """Every timing of every one of *configs* that has it, by
    :func:`_series_name`, timing by timing."""
    names = list(timings)
    return {
        _series_name(timing, config, names, configs): data[config]
        for timing, data in timings.items() for config in configs if config in data
    }


def _references(
    timings: Mapping[str, Mapping[str, pd.DataFrame]], configs: list[str], baseline: str,
) -> dict[str, str]:
    """Each series' reference: the same timing of the *baseline*
    configuration, so a ratio never sets one timing against another."""
    names = list(timings)
    return {
        _series_name(timing, config, names, configs):
            _series_name(timing, baseline, names, configs)
        for timing, data in timings.items() for config in configs
        if config in data and baseline in data
    }


def _stats_table(
    timings: Mapping[str, Mapping[str, pd.DataFrame]], configs: list[str], baseline: str,
) -> pd.DataFrame:
    """The Statistics table of *configs*, each timing's rows set against
    the same timing of the *baseline* configuration."""
    names = list(timings)
    tables = [
        build_event_stats_table(
            series, list(series), "event_time_s", "s",
            _series_name(timing, baseline, names, configs), True,
        )
        for timing, data in timings.items()
        if (series := {
            _series_name(timing, config, names, configs): data[config]
            for config in configs if config in data
        })
    ]
    tables = [table for table in tables if not table.empty]
    return pd.concat(tables) if tables else pd.DataFrame()


def _quantity(timings: list[str]) -> str:
    """What the chart's time axes call the times."""
    if timings == [_WHOLE_EVENT]:
        return "Event time"
    return f"{timings[0]} time" if len(timings) == 1 else "Time per event"


def _render_current_run(
    timings: Mapping[str, Mapping[str, pd.DataFrame]],
    display_options_slot=None,
    controls: tuple | None = None,
) -> None:
    """Render the current-run per-event timing view of each of *timings*
    (timing → configuration → per-event times), the first choosing the
    configurations on offer; each timing is set against the same timing of
    the baseline, and timings of one configuration have no ratio. *controls*
    are the Baseline and Configurations selectors' places on a row the
    caller laid out, with the baseline alone compared at first; without
    them the selectors take a row of their own and compare the most
    different configurations."""
    names = list(timings)
    first = timings[names[0]]
    current_labels = sorted(first)
    if not current_labels:
        st.info(
            "No event timing data available in this run."
            if names[0] == _WHOLE_EVENT
            else f"**{names[0]}** did not run in any configuration of this run."
        )
        return

    col_baseline, col_configs = controls or st.columns(
        [1, 3], gap="medium", vertical_alignment="bottom",
    )
    with col_baseline:
        baseline_label = baseline_selector_control(_KEY_PREFIX, current_labels)
    with col_configs:
        display_labels = config_selector_control(
            _KEY_PREFIX, first, current_labels, baseline_label, "event_time_s", "s",
            compare_by_default=controls is None,
        )

    series = _series(timings, display_labels)
    labels = list(series)
    bin_options = cached_event_bin_options(series, "event_time_s", tuple(labels))
    bins, palette_name, alpha, show_errors, show_mean_lines = (
        histogram_display_controls(
            "evt_timing", bin_options, len(labels), display_options_slot,
        )
    )

    fig = plot_event_timing(
        series,
        labels=labels,
        baseline_label=_series_name(names[0], baseline_label, names, display_labels),
        show="both",
        exclude_events=[0],
        palette=PALETTES[palette_name],
        bins=bins,
        alpha=alpha,
        show_errors=show_errors,
        show_mean_lines=show_mean_lines,
        quantity=_quantity(names),
        references=_references(timings, display_labels, baseline_label),
    )
    st.plotly_chart(fig, width="stretch", key="evt_timing_current_chart")

    st.subheader("Statistics")
    stats = _stats_table(timings, display_labels, baseline_label)
    if not stats.empty:
        st.dataframe(style_stats_table(stats), width="stretch")
    else:
        st.info("No valid statistics available (missing or empty data).")

    if set(display_labels) != set(current_labels):
        with st.expander(f"All configurations ({len(current_labels)})"):
            all_stats = _stats_table(timings, current_labels, baseline_label)
            if not all_stats.empty:
                st.dataframe(style_stats_table(all_stats), width="stretch")


#: How many of the slowest algorithms the history draws beside the whole event,
#: at first.
_SLOWEST_DEFAULT = 5
_SLOWEST_KEY = "evt_timing_hist_slowest"


def _slowest_history(
    event_trend: pd.DataFrame | None,
    algorithm_trend: pd.DataFrame,
    reliability: dict[str, bool | None] | None = None,
    reliability_slot=None,
) -> tuple[pd.DataFrame | None, bool]:
    """One configuration's whole-event history beside its slowest
    algorithms', chosen with a Configuration selector and a Slowest
    algorithms slider on one row, the whole event first and the algorithms
    slowest first, each a series named after it; and whether the row's Log
    y axis toggle is on. The runs the reliability choice keeps, and of
    those each nightly tag's latest, rank the algorithms, as they are the
    ones drawn. ``None`` when the window has no history at all, an empty
    frame when the reliability choice left none."""
    # One reliability choice for both frames: the event rows have no component.
    both = render_reliability_filter(
        pd.concat(
            [df for df in (event_trend, algorithm_trend) if is_valid_df(df)],
            ignore_index=True,
        ),
        reliability, key="evt_timing_hist_exclude_unreliable", slot=reliability_slot,
    )
    if both.empty:
        return both, False
    event_trend = both[both["component"].isna()].drop(columns="component")
    algorithm_trend = both[both["component"].notna()]
    frames = [df for df in (event_trend, algorithm_trend) if is_valid_df(df)]
    configs = sorted(set().union(*(df["label"].unique() for df in frames)))
    config_col, slowest_col, log_col = st.columns(
        [2, 3, 1], gap="medium", vertical_alignment="bottom",
    )
    with config_col:
        config = st.selectbox(
            "Configuration", configs, index=configs.index(_baseline_config(configs)),
            key=f"{_KEY_PREFIX}_hist_config",
        )
    algorithms = algorithm_trend[algorithm_trend["label"] == config]
    if not algorithms.empty:
        algorithms = latest_run_per_tag(algorithms)
    ranked = (
        algorithms.groupby("component")["mean_time_s"].median().sort_values(ascending=False)
    )
    with slowest_col:
        n = _slowest_slider(len(ranked))
    with log_col:
        log_y = st.toggle(
            "Log y axis", value=True, key=f"{_KEY_PREFIX}_hist_log_y",
            help="The whole event and the algorithms' times differ by orders of "
                 "magnitude; on a log axis each one's changes stay visible.",
        )
    slowest = ranked.index[:n].tolist()
    parts = []
    if is_valid_df(event_trend):
        parts.append(event_trend[event_trend["label"] == config].assign(label=_WHOLE_EVENT))
    for name in slowest:
        parts.append(algorithms[algorithms["component"] == name].assign(label=name))
    parts = [part for part in parts if not part.empty]
    return (pd.concat(parts, ignore_index=True) if parts else None), log_y


def _slowest_slider(n_algorithms: int) -> int:
    """How many of the slowest of *n_algorithms* the history draws."""
    if n_algorithms == 0:
        return 0
    # A configuration with fewer algorithms than the stored choice draws all.
    if st.session_state.get(_SLOWEST_KEY, 0) > n_algorithms:
        st.session_state[_SLOWEST_KEY] = n_algorithms
    return st.slider(
        "Slowest algorithms", min_value=0, max_value=n_algorithms,
        value=min(_SLOWEST_DEFAULT, n_algorithms), key=_SLOWEST_KEY,
        help="How many of the configuration's slowest algorithms, by their mean "
             "time per event over the window, to draw beside the whole event: "
             "each one's own time in each event.",
    )


def _render_historical(
    trend_event_df: pd.DataFrame,
    reliability: dict[str, bool | None] | None = None,
    reliability_slot=None,
    display_options_slot=None,
    ordered: bool = False,
    log_y: bool = False,
) -> None:
    """Render the historical event timing trends view (3-panel: Median | Mean | Std):
    its series sorted, or in the frame's order when *ordered*, on log y axes
    when *log_y*."""
    if not is_valid_df(trend_event_df):
        st.info(
            "No event timing trend data in the selected window. "
            "Widen the trend window in the sidebar."
        )
        return
    avail_labels = (
        list(dict.fromkeys(trend_event_df["label"])) if ordered
        else sorted(trend_event_df["label"].unique())
    )
    if not avail_labels:
        st.info("No historical event timing data in the selected window.")
        return

    trend_event_df = render_reliability_filter(
        trend_event_df, reliability, key="evt_timing_hist_exclude_unreliable",
        slot=reliability_slot,
    )
    if trend_event_df.empty:
        return

    present_stats = [(col, lbl) for col, lbl in _HIST_STATS if col in trend_event_df.columns]
    if not present_stats:
        st.info("No historical event timing statistics available.")
        return

    render_historical_trends(
        trend_event_df, avail_labels, present_stats,
        std_col="std_time_s",
        n_col_candidates=["n_events"],
        unit="s",
        key_prefix="evt_timing_hist",
        no_data_msg="No event timing trend data in the selected window.",
        display_options_slot=display_options_slot,
        log_y=log_y,
    )


def render(
    event_data: dict | None,
    trend_event_df: pd.DataFrame | None,
    trends_enabled: bool = False,
    reliability: dict[str, bool | None] | None = None,
    *,
    components: Mapping[str, ComponentTiming] | None = None,
    load_algorithm_trend: Callable[[], pd.DataFrame | None] | None = None,
) -> SectionScope | None:
    """The per-event timing views. Given a k4run job's *components*, the
    current run's "Timing of" selector sets any algorithms' own time in each
    event beside the whole event's. Whenever the window holds any, the
    history draws one configuration's slowest algorithms beside its whole
    event, from *load_algorithm_trend*
    (:func:`~k4bench.analysis.trend.build_algorithm_timing_trend`), loaded
    only for the history."""
    if event_data is None and not trends_enabled:
        st.info("No event timing data available in the selected directory.")
        return None

    # The "Historical Trends" option is gated on remote mode (not on the current
    # window's data) so the view selector stays put when the trend window changes.
    view, reliability_slot, display_options_slot = view_control_row(
        _VIEWS if trends_enabled else [_VIEWS[0]], key="evt_timing_view_mode",
    )
    if view == "Current Run":
        timings, controls = [_WHOLE_EVENT], None
        if components:
            # One row: what is timed, then which configurations.
            timing_col, *controls = st.columns(
                [4, 1, 2], gap="medium", vertical_alignment="bottom",
            )
            with timing_col:
                timings = _timing_selector(
                    components, _baseline_config(sorted(event_data or components)),
                )
        if event_data is None and timings == [_WHOLE_EVENT]:
            st.info("No event timing data available in the selected directory.")
        else:
            _render_current_run(
                {
                    timing: (event_data or {}) if timing == _WHOLE_EVENT
                    else _algorithm_event_data(components, timing)
                    for timing in timings
                },
                display_options_slot,
                controls=tuple(controls) if controls else None,
            )
        return None
    # Whatever the selected run holds, the window's algorithm history.
    algorithm_trend = load_algorithm_trend() if load_algorithm_trend else None
    log_y = ordered = False
    if is_valid_df(algorithm_trend):
        trend_event_df, log_y = _slowest_history(
            trend_event_df, algorithm_trend, reliability, reliability_slot,
        )
        if trend_event_df is not None and trend_event_df.empty:
            return TREND_WINDOW_SCOPE
        # The reliability choice is made; the series are in drawing order.
        reliability, ordered = None, True
    _render_historical(
        trend_event_df, reliability,
        reliability_slot=reliability_slot,
        display_options_slot=display_options_slot,
        ordered=ordered,
        log_y=log_y,
    )
    # The trends span the window's releases, not the sidebar's one — reported
    # so the scope note above stops naming it.
    return TREND_WINDOW_SCOPE
