"""Regressions tab — the nightly cross-detector regression report.

Unlike every other tab this view is not scoped by the sidebar's
detector/platform/sample selection: it renders the precomputed
``_reports/{date}/report.json`` written to EOS by the nightly
``regression-report`` CI job (see ``k4bench/regression/``), covering **all**
detectors for one night. Only the per-metric drill-down chart downloads run
data, and only for the series being inspected.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from k4bench.regression.engine import Z_THRESHOLD
from k4bench.regression.models import (
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.regression.render import (
    _badge,
    _detector_badge,
    _fmt,
    _fmt_pct,
    _group_title,
    _metric_name,
    _pretty_sample,
    _quiet_summary,
    from_json,
)
from k4bench.regression.report_builder import (
    EVENT_METRICS,
    FETCH_WINDOW_RUNS,
    RUN_METRICS,
    _with_cpu_efficiency,
)
from k4bench.results.reliability_evidence import run_reliability_map
from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE
from data import (
    cached_load_trend_event_timing,
    cached_load_trend_machine_info,
    cached_load_trend_results,
)
from remote_cache import (
    _cached_fetch_report,
    _cached_fetch_runs_windowed,
    _cached_list_report_dates,
    _cached_list_run_dates,
)
from tabs._reliability import render_reliability_filter
from ui_utils import _is_valid_df, _to_rgba

#: Fill for the accepted-baseline band on the drill-down chart — same visual
#: device as machine_info's threshold shading, in the palette's first hue.
_BASELINE_FILL = "rgba(31,119,180,0.08)"

#: Docs section explaining how the step-change detector judges a metric, linked
#: from the tab's intro instead of restating the logic inline.
_ASSESSMENT_DOCS_URL = (
    "https://key4hep.github.io/k4Bench/user-guide/features/dashboard/#regressions-tab"
)


def render(data_url: str, cache_dir: str) -> None:
    st.caption(
        "Nightly step-change detection across every detector's benchmark history. "
        f"[Learn how regressions are assessed →]({_ASSESSMENT_DOCS_URL})"
    )

    dates = _cached_list_report_dates(data_url)
    if not dates:
        st.info(
            "No regression reports available yet. The nightly benchmark workflow "
            "uploads the first report after its next run."
        )
        return

    night = st.selectbox(
        "Report night",
        dates,
        index=0,
        help="Nightly reports are kept on EOS; pick an earlier night to see how "
             "it was judged at the time.",
    )
    raw = _cached_fetch_report(data_url, night)
    if not raw:
        st.warning(f"Could not load the report for {night} from EOS.")
        return
    report = from_json(raw)

    _render_banner(report)

    # `&detector=...` deep-links here (see k4bench.regression.render._dashboard_link,
    # used by the nightly regression email's "view in dashboard" links): the named
    # detector is moved to the front of the list and pre-expanded, so the reader
    # lands on the detector that triggered the alert.
    wanted_detector = st.query_params.get("detector")
    by_detector = list(report.by_detector().items())
    if wanted_detector:
        by_detector.sort(key=lambda item: item[0] != wanted_detector)

    for detector, groups in by_detector:
        _render_detector(
            detector, groups, data_url, cache_dir,
            expanded=(detector == wanted_detector),
        )


def _render_banner(report: NightlyReport) -> None:
    """The at-a-glance banner, mirroring machine_info's run-quality card."""
    n_fail = len(report.failures) + len(report.job_failures)
    with st.container(border=True):
        st.markdown(f"##### Nightly verdict at a glance — {report.report_night or 'no data'}")
        cols = st.columns(4)
        cols[0].metric(
            "Detectors checked", len(report.by_detector()),
            help="Detectors with benchmark history on EOS for this night. A detector "
                 "can span several (platform, sample) run groups, each judged "
                 "against its own baseline.",
        )
        cols[1].metric(
            "🔴 Regressed", len(report.regressions),
            help="Metrics that crossed both detection gates on two consecutive "
                 "reliable nights (confirmed), either direction — not judged good "
                 "or bad, only that it moved beyond the baseline twice in a row.",
        )
        cols[2].metric(
            "⚠️ Watch", len(report.watches),
            help="Metrics flagged for the first time this night. Not alerted on: "
                 "they either confirm on the next reliable night or clear.",
        )
        cols[3].metric(
            "❌ Failures", n_fail,
            help="Hard job failures: a config exiting non-zero, producing no "
                 "results, or a whole run missing for the night. These alert "
                 "immediately, no confirmation needed.",
        )


def _render_detector(
    detector: str,
    groups: list[RunGroupReport],
    data_url: str,
    cache_dir: str,
    *,
    expanded: bool = False,
) -> None:
    # Detectors always start collapsed — even when alerting — so a noisy sweep
    # night doesn't blow the page open into a wall of expanded charts. The badge
    # already telegraphs which detectors need attention; the reader opens those.
    # A `?detector=...` deep link (see `render`) overrides this to land pre-opened.
    with st.expander(f"{_detector_badge(groups)} {detector}", expanded=expanded):
        for i, group in enumerate(groups):
            # Sub-heading only when a detector has several (platform, sample)
            # groups — most have exactly one and a heading would be noise.
            if len(groups) > 1:
                st.markdown(f"**{_group_title(group)}**")
            _render_group(group, data_url, cache_dir, key=f"{detector}_{i}")


#: Human-readable metric names for the change-ledger row labels; the raw column
#: name (e.g. ``wall_time_s``) is preserved in the hover tooltip.
_METRIC_LABELS = {
    "wall_time_s":    "wall time",
    "user_cpu_s":     "user CPU",
    "peak_rss_mb":    "peak RSS",
    "cpu_efficiency": "CPU efficiency",
    "mean_time_s":    "mean event time",
    "median_time_s":  "median event time",
    "mean_rss_mb":    "mean event RSS",
}

#: Unit suffix per metric for axis titles (empty for dimensionless ratios).
_METRIC_UNITS = {
    "wall_time_s":    "s",
    "user_cpu_s":     "s",
    "peak_rss_mb":    "MB",
    "cpu_efficiency": "",
    "mean_time_s":    "s",
    "median_time_s":  "s",
    "mean_rss_mb":    "MB",
}


def _yaxis_label(item: MetricVerdict) -> str:
    """Human-readable y-axis title with units, e.g. ``Wall time (s)`` — the raw
    column name is kept only in the hover tooltip."""
    name = _METRIC_LABELS.get(item.metric, item.metric)
    name = name[:1].upper() + name[1:]
    unit = _METRIC_UNITS.get(item.metric, "")
    return f"{name} ({unit})" if unit else name

#: Status fills keyed on severity — an *attention level*, not a good/bad verdict.
#: Confirmed = critical red, watch = warning amber. Only the drill-down trend
#: markers still use these fills; the ledger table below carries severity as an
#: explicit 🔴/⚠️ badge column, so the state never reads by color alone.
_SEVERITY_FILL = {
    Severity.CONFIRMED: "#d03b3b",
    Severity.WATCH:     "#fab219",
}

#: Cap on ledger rows: beyond this, keep the worst by |Δ| so one sweep night
#: can't produce an unbounded table.
_MAX_ROWS = 40


def _pretty_metric(v: MetricVerdict) -> str:
    """Row-label metric name — the human label plus the sub-detector for
    region-level rows (``wall time · VertexBarrel``)."""
    name = _METRIC_LABELS.get(v.metric, v.metric)
    return f"{name} · {v.sub_detector}" if v.sub_detector else name


def _flag_table(flagged: list[MetricVerdict]) -> None:
    """Tonight's flagged metrics as a compact, sortable ledger — one row per
    (config, metric), worst first.

    A table is the one layout that stays readable from a single flag to a whole
    sweep night: extra rows scroll instead of crowding, every column re-sorts on
    a header click, and each row still reads at a glance — severity from the
    🔴/⚠️ badge, size and direction from the Δ bar. The per-series trend below
    remains the place to actually inspect a metric's history.
    """
    rows = [v for v in flagged if v.pct_change is not None]
    if not rows:
        return
    rows.sort(key=lambda v: (v.severity is not Severity.CONFIRMED, -abs(v.pct_change)))
    rows = rows[:_MAX_ROWS]
    # The bar encodes *magnitude* (|Δ%|, in whole percents), 0 → empty and the
    # night's worst → full, so a small flag never looks large. Direction rides
    # in its own column so the sign is never lost.
    span = (max(abs(v.pct_change) for v in rows) * 100) or 5.0

    df = pd.DataFrame(
        [
            {
                "": "🔴" if v.severity is Severity.CONFIRMED else "⚠️",
                "Config": v.label,
                "Metric": _pretty_metric(v),
                "Dir": "↑" if v.pct_change > 0 else "↓",
                "Δ vs baseline": abs(v.pct_change) * 100,
                "Current / baseline": f"{_fmt(v.value)} / {_fmt(v.baseline_median)}",
            }
            for v in rows
        ]
    )
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "": st.column_config.TextColumn(
                "", width="small",
                help="🔴 confirmed regression · ⚠️ watch (first flag, unconfirmed)",
            ),
            "Config": st.column_config.TextColumn("Config", width="medium"),
            "Dir": st.column_config.TextColumn(
                "Dir", width="small",
                help="↑ increase · ↓ decrease vs baseline — a plain direction, "
                     "not judged good or bad.",
            ),
            "Δ vs baseline": st.column_config.ProgressColumn(
                "Δ vs baseline",
                help="Size of the step from the baseline median (|Δ%|), scaled to "
                     "tonight's largest flag. Direction is the ↑/↓ column.",
                format="%.0f%%",
                min_value=0,
                max_value=span,
            ),
        },
    )


def _render_group(group: RunGroupReport, data_url: str, cache_dir: str, *, key: str) -> None:
    for msg in group.job_failures:
        st.error(f"**{msg}**", icon="❌")
    if group.failures:
        st.error(
            f"**{len(group.failures)} config failure(s):** "
            + "; ".join(f"{v.label} — {v.reason}" for v in group.failures),
            icon="❌",
        )
    for msg in group.notes:
        st.caption(f"❔ {msg}")

    flagged = [
        v for v in group.verdicts
        if v.severity in (Severity.WATCH, Severity.CONFIRMED)
    ]
    if flagged:
        _flag_table(flagged)
    if group.verdicts:
        st.caption(_quiet_summary(group))

    drillable: list[MetricVerdict] = [
        v for v in flagged if v.baseline_median is not None
    ]
    if not drillable:
        return
    options = ["—"] + [
        f"{_badge(v)} · {_metric_name(v)} · {v.label}" for v in drillable
    ]
    choice = st.selectbox(
        "Show trend",
        options,
        key=f"regr_drill_{key}",
        help="Render this metric's recent history with the baseline band the "
             "verdict was judged against. Downloads the run window for this "
             "group on first use.",
    )
    if choice != "—":
        _render_drilldown(drillable[options.index(choice) - 1], data_url, cache_dir)


#: Marker for the flagged night on the drill-down trend, keyed on the verdict's
#: severity — a warning triangle (amber) while it is only a WATCH, a filled red
#: circle once it CONFIRMS as a regression. Shape *and* color both carry the
#: state so it never reads by color alone, matching the ledger's encoding.
def _flag_marker(item: MetricVerdict) -> dict:
    if item.severity is Severity.WATCH:
        return dict(symbol="triangle-up", color=_SEVERITY_FILL[Severity.WATCH],
                    hover="⚠️ Watch")
    return dict(symbol="circle", color=_SEVERITY_FILL[Severity.CONFIRMED],
                hover="🔴 Regression")


def _prev_point(df: pd.DataFrame, item: MetricVerdict) -> tuple | None:
    """The plotted point immediately before the flagged night — the run that
    was ``⚠️ Watch``\\ ed before this one confirmed (confirmation is a
    two-strike rule). Returns ``(x_date, value)`` or ``None`` if there is no
    prior point on the trend."""
    prior = df[pd.to_datetime(df["x_date"]) < pd.to_datetime(item.run_date)]
    if prior.empty:
        return None
    row = prior.iloc[-1]  # df is sorted by x_date
    val = row[item.metric]
    return None if pd.isna(val) else (row["x_date"], val)


def _drilldown_caption(item: MetricVerdict) -> str:
    return f"**{item.reason}** — {item.label}, {_pretty_sample(item.sample)}"


def _series_key(verdict: MetricVerdict) -> str:
    """Stable per-series suffix identifying one drill-down chart's ``(detector,
    sample, label, metric)`` — shared by the reliability-filter and chart widget
    keys so the two can never key off different subsets of the same series."""
    return "_".join(filter(None, (
        verdict.detector, verdict.sample, verdict.label, verdict.metric,
    )))


def _metric_history(verdict: MetricVerdict, data_url: str, cache_dir: str):
    """Fetch this group's run window and return ``(df, reliability)`` for the
    verdict's metric series: *df* carries ``run_id``, ``x_date`` and the metric
    column, *reliability* is the ``{run_id: reliable}`` map so the caller can
    exclude unreliable runs the same way the Run Trends tab does."""
    stacks_dates = _cached_list_run_dates(
        data_url, verdict.detector, verdict.platform, verdict.sample
    )
    pairs = sorted(
        (date, stack) for stack, dates in stacks_dates.items() for date in dates
    )[-FETCH_WINDOW_RUNS:]
    window: dict[str, list[str]] = {}
    for date, stack in pairs:
        window.setdefault(stack, []).append(date)
    windowed_items = tuple(sorted(
        (stack, tuple(dates)) for stack, dates in window.items()
    ))
    run_dirs = _cached_fetch_runs_windowed(
        data_url, verdict.detector, verdict.platform, verdict.sample,
        cache_dir, windowed_items,
    )
    if not run_dirs:
        return None

    # Reuse the same ``@st.cache_data`` trend shims as the rest of the dashboard
    # (see ``dashboard/data.py``) rather than rebuilding from CSV: the results
    # trend feeds both the reliability map and the run-metric series, and every
    # Streamlit rerun (e.g. toggling the unreliable-runs filter below) would
    # otherwise re-parse the whole window. The cache keys on ``run_dirs``, so a
    # second drill-down on the same group hits the cache too.
    results_df = cached_load_trend_results(run_dirs)

    # Per-run reliability for this window — a per-run property (keyed by run_id)
    # independent of which metric is plotted, so it is computed once from the
    # machine-info + results trends and applied to whichever series df we build.
    reliability = run_reliability_map(
        results_df,
        cached_load_trend_machine_info(run_dirs),
    )

    if verdict.metric in EVENT_METRICS:
        df = cached_load_trend_event_timing(run_dirs)
        if not _is_valid_df(df):
            return None
        df = df[df["label"] == verdict.label]
    else:
        df = results_df
        if not _is_valid_df(df):
            return None
        if verdict.metric in RUN_METRICS:
            df = _with_cpu_efficiency(df)
        df = df[df["label"] == verdict.label]

    if df.empty or verdict.metric not in df.columns:
        return None
    df = df.sort_values("x_date")
    return df, reliability


def _render_drilldown(verdict: MetricVerdict, data_url: str, cache_dir: str) -> None:
    history = _metric_history(verdict, data_url, cache_dir)
    if history is None:
        st.warning("No history could be loaded for this metric.")
        return
    df, reliability = history

    # Exclude unreliable runs by default (with a toggle) — the same filter and
    # default as the Run Trends tab. The engine already judged the verdict on
    # reliable runs only, so this just keeps the plotted line consistent with
    # the baseline band; the flagged night is reliable and always survives.
    excl_key = "regr_drill_excl_" + _series_key(verdict)
    df = render_reliability_filter(df, reliability, key=excl_key, date_col="x_date")
    if df.empty:
        return
    x, y = df["x_date"], df[verdict.metric]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=_metric_name(verdict),
        line=dict(color=PALETTE[0], width=2),
        marker=dict(size=6, color=_to_rgba(PALETTE[0], 0.75)),
    ))
    # The accepted-baseline band this verdict was judged against: median ±
    # z-threshold × scaled MAD (the detection gate), same shading device as
    # machine_info's reliability threshold.
    med, mad = verdict.baseline_median, verdict.baseline_mad or 0.0
    fig.add_hline(y=med, line_dash="dash", line_color=PALETTE[0], line_width=1,
                  annotation_text="baseline median", annotation_font_size=11)
    if mad > 0:
        fig.add_hrect(y0=med - Z_THRESHOLD * mad, y1=med + Z_THRESHOLD * mad,
                      fillcolor=_BASELINE_FILL, line_width=0)
    # For a confirmed regression, also mark the night it was first *watched*
    # (the preceding reliable run — confirmation is a two-strike rule) with the
    # warning triangle, so the trend shows the ⚠️→🔴 progression, not just the
    # endpoint. Markers are drawn straight onto the plot with no legend row of
    # their own (the emoji hover names the state).
    if verdict.severity is Severity.CONFIRMED:
        prev = _prev_point(df, verdict)
        if prev is not None:
            fig.add_trace(go.Scatter(
                x=[prev[0]], y=[prev[1]], mode="markers", showlegend=False,
                marker=dict(size=13, symbol="triangle-up",
                            color=_SEVERITY_FILL[Severity.WATCH],
                            line=dict(width=1, color="#ffffff")),
                hovertemplate="⚠️ Watch (first flagged)<br>%{x|%Y-%m-%d}"
                              "<br>%{y:.4g}<extra></extra>",
            ))
    flag = _flag_marker(verdict)
    fig.add_trace(go.Scatter(
        x=[verdict.run_date], y=[verdict.value], mode="markers", showlegend=False,
        marker=dict(size=13, color=flag["color"], symbol=flag["symbol"],
                    line=dict(width=1, color="#ffffff")),
        hovertemplate=f"{flag['hover']}<br>%{{x|%Y-%m-%d}}<br>%{{y:.4g}}"
                      "<extra></extra>",
    ))
    # Label the x-axis by nightly tag (one angled tick per Key4hep release), the
    # same way Run Trends does — the underlying x is the release date (x_date).
    unique_dates = sorted(pd.to_datetime(pd.Series(x)).dropna().unique())
    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=unique_dates,
        ticktext=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates],
        tickangle=-30,
        title_text="Key4hep Nightly Tag",
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=360,
        margin=dict(l=10, r=10, t=30, b=90),
        yaxis_title=_yaxis_label(verdict),
        showlegend=False,
    )
    chart_key = "regr_chart_" + _series_key(verdict)
    st.plotly_chart(fig, width="stretch", key=chart_key)
    st.caption(_drilldown_caption(verdict))
