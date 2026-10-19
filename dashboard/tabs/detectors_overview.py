"""Overview tab — cross-detector comparison of the nightly benchmarks.

Compares every detector's baseline benchmark for the sidebar-selected platform
and sample, over the sidebar's shared Trend window, in four views:
**Performance Trends** (the two selected metrics' history), **Performance
Landscape** (time against memory, one point per detector at its own most recent
run — independent of which report night is open), **Regression Status** (one
report night's verdict banner, the per-detector roster, and the worst flag's
trend — the Regressions tab itself is scoped to one detector, so this is where
the cross-detector regression picture lives; the night is selectable and
defaults to the newest), and **Nightly Report** (that night's e-group mail
itself, see :mod:`tabs._nightly_email`).
The data comes from the precomputed ``_reports/{date}/report.json`` files on
EOS, whose verdicts carry the raw nightly value of every run/event metric for
**all** detectors — one small cached JSON fetch per night, no per-detector run
downloads. It is shaped into the views' inputs by :mod:`tabs._overview_data`;
this module holds the two figure views and the render flow, and the Regression
Status view is :mod:`tabs._regression_status`.

Only the first three views are scoped to the sidebar's platform/sample; the mail
is the whole night, across every scope, so it stays readable even when the
selected scope has nothing (see :func:`render`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import _TEMPLATE
from k4bench.labels import BASELINE_LABEL
from k4bench.layout import SIM
from k4bench.metrics import is_megabytes, metric_label, metric_title
from k4bench.regression.models import Severity
from remote_cache import cached_fetch_reports, cached_list_report_dates
from tabs import _nightly_email, _regression_status
from tabs._overview_data import (
    MEMORY_METRICS,
    METRIC_ORDER,
    TIME_METRICS,
    OverviewInputs,
    collapse_history,
    detector_family,
    drop_unreliable_runs,
    latest_snapshot,
    nights_in_window,
    overview_inputs,
    relative_history,
    scatter_points,
    stale_run_nights,
)
from tabs._regression_flags import add_severity_markers, render_flag_pills
from tabs._reliability import render_run_reliability_filter
from ui_chrome import EXAMPLE_DETECTORS, seed_query_param
from ui_utils import to_rgba

#: The tab's views, dispatched by the same View switcher as Region Timing and
#: Machine Info: the two figure views, then a night's verdicts, then that
#: night's report in the form the e-group received it. Only the first three
#: read the sidebar's platform/sample scope, so the mail is dispatched ahead of
#: the empty-scope notice.
_VIEWS = [
    "Performance Trends", "Performance Landscape", "Regression Status",
    "Nightly Report",
]

#: Session key of the View switcher, seeded from and written back to ``?view=`` so
#: a copied URL reopens the view it was copied from — along with the parameters
#: only that view reads (``?report=``, ``?tmetric=``/``?mmetric=``).
_VIEW_KEY = "det_ov_view_mode"


def _metric_title(metric: str) -> str:
    """Human-readable panel/axis title with units, e.g. ``Wall time (s)``.
    Sizes stored in MB are shown in GB (see :func:`_wide_in_display_units`)."""
    return metric_title(metric, "GB" if is_megabytes(metric) else None)


def _trend_y_title(metric: str, relative: bool) -> str:
    """Trend-panel y-axis title; in relative view the unit becomes percent of
    the detector's first plotted night."""
    if not relative:
        return _metric_title(metric)
    return f"{metric_label(metric)} (% of first night)"


def _detector_legend_columns(
    detectors: list[str],
    *,
    plot_h: int,
    t_margin: int,
    tick_clearance: int,
) -> tuple[dict[str, tuple[str, str]], dict[str, dict], int]:
    """One vertical Plotly legend per detector family, arranged below the plot.

    Returns ``(trace_specs, layout_legends, bottom_margin)``. ``trace_specs``
    maps each detector to its named legend and its compact entry label. A family
    with versioned detectors gets a heading (``ALLEGRO``) and entries below it
    (``o1_v03``, ``o2_v01``). Toolkit examples use their toolkit as the heading,
    so the unversioned ``SiD`` entry sits below ``DD4hep`` like the experiment
    variants sit below their family names.
    Separate legends are what make the grouping structural rather than an
    ordering trick, while ``legendgroup=detector`` still lets each variant be
    toggled independently across both panels and its regression markers.
    """
    by_family: dict[str, list[str]] = {}
    for detector in sorted(detectors):
        by_family.setdefault(detector_family(detector)[0], []).append(detector)
    if not by_family:
        return {}, {}, 160

    font_size = 12
    row_h = font_size + 8
    families = list(by_family.items())
    headings: dict[str, str] = {}
    display_labels = []
    for family, members in families:
        versioned = any(detector_family(d)[1] for d in members)
        example_toolkits = {
            EXAMPLE_DETECTORS[d][0] for d in members if d in EXAMPLE_DETECTORS
        }
        heading = family if versioned else (
            next(iter(example_toolkits)) if len(example_toolkits) == 1 else ""
        )
        headings[family] = heading
        if heading:
            display_labels.append(heading)
        display_labels.extend(
            detector_family(detector)[1] if versioned else detector
            for detector in members
        )
    longest_label = max(map(len, display_labels), default=0)
    # The nightly roster currently has five compact families. Let all five
    # share the first row; long names still reduce the column count before they
    # can crowd one another.
    max_columns = 2 if longest_label > 28 else 3 if longest_label > 18 else 5
    family_rows = [
        families[start:start + max_columns]
        for start in range(0, len(families), max_columns)
    ]
    row_gap = 12
    legend_row_heights = [
        max(
            len(members) + bool(headings[family])
            for _, members in family_row
        ) * row_h + 12
        for family_row in family_rows
    ]
    offset = tick_clearance + 75
    legend_h = sum(legend_row_heights) + row_gap * (len(family_rows) - 1)
    b_margin = max(160, offset + legend_h)
    total_h = plot_h + t_margin + b_margin

    trace_specs: dict[str, tuple[str, str]] = {}
    layout_legends: dict[str, dict] = {}
    for idx, (family, members) in enumerate(families):
        row_idx, col_idx = divmod(idx, max_columns)
        columns_in_row = len(family_rows[row_idx])
        preceding_height = (
            sum(legend_row_heights[:row_idx]) + row_gap * row_idx
        )
        legend_ref = "legend" if idx == 0 else f"legend{idx + 1}"
        versioned = any(detector_family(d)[1] for d in members)
        for detector in members:
            variant = detector_family(detector)[1]
            trace_specs[detector] = (
                legend_ref,
                variant if variant and versioned else detector,
            )
        layout_legends[legend_ref] = dict(
            orientation="v",
            title=dict(text=headings[family]),
            yref="container",
            yanchor="top",
            y=(b_margin - offset - preceding_height) / total_h,
            # Paper-referenced x positions keep the plotting area full-width;
            # container-referenced vertical legends make Plotly reserve side
            # margins for every family and squeeze the chart between them.
            xref="paper",
            xanchor="center",
            x=(col_idx + 0.5) / columns_in_row,
            tracegroupgap=0,
            font=dict(size=font_size),
            groupclick="togglegroup",
        )
    return trace_specs, layout_legends, b_margin


def _wide_in_display_units(wide: pd.DataFrame) -> pd.DataFrame:
    """A copy of the landscape's wide frame with MB sizes converted to GB for
    display; the stored columns stay MB (the reports' native unit) so the data
    helpers and the regression engine's numbers remain directly comparable."""
    wide = wide.copy()
    for metric in wide.columns:
        if is_megabytes(metric):
            wide[metric] = wide[metric] / 1024.0
    return wide


def _history_in_display_units(hist: pd.DataFrame) -> pd.DataFrame:
    """A copy of history rows with MB sizes converted to GB for display (see
    :func:`_wide_in_display_units`)."""
    if hist.empty:
        return hist
    hist = hist.copy()
    mem_rows = hist["metric"].map(is_megabytes).astype(bool)
    hist.loc[mem_rows, "value"] = hist.loc[mem_rows, "value"] / 1024.0
    return hist


def _value_axis(log: bool) -> dict:
    """Shared styling for a value (time/memory) axis: trimmed tick numbers
    with digit grouping, and on log scale a 1-2-5 tick pattern instead of
    Plotly's default every-digit labels (which crowd a narrow decade span)."""
    axis = dict(type="log" if log else "linear", tickformat=",~g")
    if log:
        axis["dtick"] = "D2"
    return axis


def _log_range(values: pd.Series, lo_frac: float, hi_frac: float) -> list[float] | None:
    """Range for a log axis, padded around the data in log space (Plotly log
    ranges are given in log10 units, asymmetric fractions of the decade span).
    A degenerate span (single detector or identical values) pads a fixed
    fraction of a decade; non-positive values (impossible for time/memory,
    guarded anyway) fall back to auto-ranging."""
    vals = values[values > 0]
    if vals.empty:
        return None
    d0, d1 = math.log10(float(vals.min())), math.log10(float(vals.max()))
    span = max(d1 - d0, 0.15)
    return [d0 - span * lo_frac, d1 + span * hi_frac]


def _history_figure(
    hist: pd.DataFrame,
    time_metric: str,
    mem_metric: str,
    styles: dict[str, tuple[str, str, str]],
    detectors: list[str],
    alpha: float = 0.75,
    log: bool = True,
    relative: bool = False,
    show_confirmed: bool = True,
    show_watch: bool = False,
    failures: pd.DataFrame | None = None,
) -> go.Figure | None:
    """The two metrics' history side by side (CPU, Memory), one legend below
    — the house pattern every other trend view in the dashboard uses (see
    e.g. ``tabs.trends``). On log scale (*log*) the detectors' >1-decade
    spread stays readable; linear is a toggle away. *relative* rescales each
    line to its first plotted night = 100 %. *show_confirmed* flags confirmed
    regressions (a halo + white-bordered badge, see
    :data:`_regression_flags.FLAG_MARKS`); *show_watch* additionally flags
    unconfirmed watch points.
    """
    if failures is not None and not failures.empty:
        hist = pd.concat([hist, failures], ignore_index=True)
    metrics = [time_metric, mem_metric]
    present = [] if hist.empty else [m for m in metrics if (hist["metric"] == m).any()]
    if not present:
        return None

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                         subplot_titles=["CPU", "Memory"])

    hover_y = "%{y:.1f} %" if relative else "%{y:.4g}"
    hist = hist.copy()
    hist["night_dt"] = pd.to_datetime(hist["night"])
    unique_dates = sorted(hist["night_dt"].dropna().unique())
    tick_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]
    marker_alpha = max(0.1, alpha - 0.2)
    shown: set[str] = set()
    plotted = [
        detector for detector in detectors
        if ((hist["detector"] == detector) & hist["metric"].isin(metrics)).any()
    ]
    t_margin = 50
    plot_h = 380
    legend_specs, legends, b_margin = _detector_legend_columns(
        plotted, plot_h=plot_h, t_margin=t_margin, tick_clearance=75,
    )
    for metric, col in ((time_metric, 1), (mem_metric, 2)):
        sub_m = hist[hist["metric"] == metric]
        if sub_m.empty:
            continue
        line_m = sub_m[sub_m["severity"] != Severity.FAILURE.value]
        for detector, (legend_ref, legend_name) in legend_specs.items():
            sub = line_m[line_m["detector"] == detector].sort_values("night_dt")
            if sub.empty:
                continue
            color, dash, symbol = styles[detector]
            # Deduped across the CPU and Memory panels so a detector with
            # both metrics only gets one legend entry.
            first = detector not in shown
            shown.add(detector)
            fig.add_trace(
                go.Scatter(
                    x=sub["night_dt"],
                    y=sub["value"],
                    mode="lines+markers",
                    name=legend_name,
                    legendgroup=detector,
                    legend=legend_ref,
                    showlegend=first,
                    line=dict(color=to_rgba(color, alpha), width=2, dash=dash),
                    marker=dict(size=7, symbol=symbol,
                                color=to_rgba(color, marker_alpha),
                                line=dict(color=color, width=1.5)),
                    customdata=sub["k4h_release"].fillna("unknown"),
                    hovertemplate=(
                        f"<b>{detector}</b><br>"
                        "Tag: %{customdata} (%{x|%Y-%m-%d})<br>"
                        f"{_metric_title(metric)}: {hover_y}<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )
        # Verdict flags on top of the lines, each behind its own toggle —
        # rendered by the shared helper so Overview and Run Trends ring points
        # identically.
        flag_severities = (
            "FAILURE",
            *(("CONFIRMED",) if show_confirmed else ()),
            *(("WATCH",) if show_watch else ()),
        )
        for severity in flag_severities:
            flagged = sub_m[sub_m["severity"] == severity]
            if flagged.empty:
                continue
            add_severity_markers(
                fig, flagged, x_col="night_dt", y_col="value",
                name_col="detector", severity=severity, hover_y=hover_y,
                row=1, col=col,
                legend_by_name={d: spec[0] for d, spec in legend_specs.items()},
            )
        fig.update_xaxes(
            type="date",
            tickmode="array",
            tickvals=unique_dates,
            ticktext=tick_labels,
            tickangle=-30,
            title_text="Key4hep Nightly Tag",
            row=1, col=col,
        )
        fig.update_yaxes(title_text=_trend_y_title(metric, relative),
                         row=1, col=col, **_value_axis(log and not relative))

    fig.update_layout(
        template=_TEMPLATE,
        height=plot_h + t_margin + b_margin,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        **legends,
    )
    return fig


def _landscape_figure(
    wide: pd.DataFrame,
    time_metric: str,
    mem_metric: str,
    styles: dict[str, tuple[str, str, str]],
    detectors: list[str],
    alpha: float = 0.75,
    log: bool = True,
    as_of: dict[str, str] | None = None,
) -> go.Figure | None:
    """The performance landscape: the selected time metric against the
    selected memory metric, one point per detector — closer to the origin is
    faster *and* leaner. One legend below, the same house pattern as every
    other trend view.

    *as_of* names the nightly tag behind each point. Points need not share one
    (each detector is shown at its own most recent run, see
    :func:`latest_snapshot`), so the tag rides the hover where the reader is
    already looking at the point it belongs to."""
    pts = scatter_points(wide, time_metric, mem_metric)
    if pts.empty:
        return None

    fig = go.Figure()
    plotted = [d for d in detectors if d in pts.index]
    t_margin = 50
    plot_h = 430
    legend_specs, legends, b_margin = _detector_legend_columns(
        plotted, plot_h=plot_h, t_margin=t_margin, tick_clearance=60,
    )
    for detector, (legend_ref, legend_name) in legend_specs.items():
        color, _, symbol = styles[detector]
        tag = (as_of or {}).get(detector)
        fig.add_trace(
            go.Scatter(
                x=[pts.loc[detector, time_metric]],
                y=[pts.loc[detector, mem_metric]],
                mode="markers",
                name=legend_name,
                legendgroup=detector,
                legend=legend_ref,
                showlegend=True,
                marker=dict(size=13, symbol=symbol, color=to_rgba(color, alpha),
                            line=dict(width=1.5, color=color)),
                hovertemplate=(
                    f"<b>{detector}</b><br>"
                    + (f"Tag: {tag}<br>" if tag else "")
                    + f"{_metric_title(time_metric)}: %{{x:.4g}}"
                    f"<br>{_metric_title(mem_metric)}: %{{y:.4g}}<extra></extra>"
                ),
            ),
        )

    x_axis = dict(_value_axis(log), title_text=_metric_title(time_metric))
    y_axis = dict(_value_axis(log), title_text=_metric_title(mem_metric))
    if log:
        x_axis["range"] = _log_range(pts[time_metric], 0.15, 0.15)
        y_axis["range"] = _log_range(pts[mem_metric], 0.15, 0.15)
    fig.update_xaxes(**x_axis)
    fig.update_yaxes(**y_axis)

    fig.update_layout(
        template=_TEMPLATE,
        height=plot_h + t_margin + b_margin,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        **legends,
    )
    return fig


def _landscape_notes(
    wide: pd.DataFrame,
    as_of: dict[str, str],
    time_metric: str,
    mem_metric: str,
    scoped_detectors: list[str],
    excluded: list[str],
) -> list[str]:
    """The caption under the landscape: which run each point comes from, then
    every scoped detector the chart cannot show and why.

    Points can date from different nights — each detector is shown at its own
    most recent run — so a tag they all share is stated once, and otherwise the
    stragglers are named with theirs rather than the chart implying one common
    night.
    """
    plotted = list(scatter_points(wide, time_metric, mem_metric).index)
    tags = {as_of[d] for d in plotted if d in as_of}
    notes: list[str] = []
    if len(tags) == 1:
        notes.append(f"Nightly tag: **{tags.pop()}**.")
    elif tags:
        newest = max(tags)
        stale = sorted(
            (d for d in plotted if as_of.get(d, newest) != newest),
            key=lambda d: as_of[d],
        )
        notes.append(
            f"Newest nightly tag: **{newest}** · each detector at its own "
            "most recent run — "
            + ", ".join(f"{d} ({as_of[d]})" for d in stale) + "."
        )
    dropped = sorted(set(wide.index) - set(plotted))
    if dropped:
        notes.append(f"Missing a landscape coordinate: {', '.join(dropped)}.")
    # *scoped_detectors* is read off the report groups, so it holds detectors
    # that ran and produced no metric at all (a hard-failed config is judged on
    # its return code) alongside those whose every run the toggle dropped.
    # Neither can be placed on the landscape, and neither is un-benchmarked.
    unplaceable = sorted(set(scoped_detectors) - set(wide.index))
    if unplaceable:
        notes.append(
            "No usable run in this window — failed, or excluded as unreliable: "
            f"{', '.join(unplaceable)}."
        )
    if excluded:
        notes.append(
            f"Not benchmarked with this sample/platform: {', '.join(excluded)}."
        )
    return notes


def _plotted_detectors(hist: pd.DataFrame, metrics: tuple[str, ...]) -> set[str]:
    """Detectors the trend chart can draw at least one line for — those with a
    value for one of the *metrics* on screen (see :func:`_history_figure`,
    which skips a detector with no rows for the panel's metric)."""
    if hist.empty:
        return set()
    return set(hist.loc[hist["metric"].isin(metrics), "detector"].unique())


def _trend_notes(
    hist: pd.DataFrame,
    window_rows: pd.DataFrame,
    time_metric: str,
    mem_metric: str,
    detectors: list[str],
    window_roster: list[str],
    excluded: list[str],
    last_run: dict[str, str],
) -> list[str]:
    """The caption under the trend chart: every detector in the scope the chart
    has no line for, and why.

    Absence is the one thing a cross-detector comparison must not report
    silently: a reader tracking a detector cannot tell "no data" from "no
    regression" if the line is simply gone. Both figure views therefore account
    for the detectors they cannot draw — this is the trend chart's half of the
    pair, :func:`_landscape_notes` the landscape's.

    The four causes are answered from four independent sources, because a
    metric frame cannot distinguish them on its own. *window_roster* is who
    actually **ran** in the window, read off the report groups, so a config that
    hard-failed — judged on its return code, carrying no metric verdict — is
    named as failed rather than dropped or called un-benchmarked. *window_rows*
    is the window's rows **before** the reliability filter, separating a
    detector the toggle dropped from one that produced nothing. *hist* is what
    is plotted, and *last_run* names the night each detector really last ran, so
    one that has fallen out of the window is placed rather than merely missed —
    it must come from the unfiltered group history, not from the landscape's
    ``as_of``, which follows the newest *plottable* run and would date a
    detector to before an excluded run it demonstrably made.

    A detector retired past the engine's missing-run grace period is dropped
    from the reports altogether, so it reaches none of these sources and cannot
    be named; widening the window to a range it still ran in brings it back.
    """
    metrics = (time_metric, mem_metric)
    plotted = _plotted_detectors(hist, metrics)
    missing = [d for d in dict.fromkeys([*detectors, *window_roster])
               if d not in plotted]

    in_window = _plotted_detectors(window_rows, metrics)
    any_metric = _plotted_detectors(window_rows, tuple(METRIC_ORDER))
    ran = set(window_roster)

    dropped   = [d for d in missing if d in in_window]
    rest      = [d for d in missing if d not in in_window]
    no_metric = [d for d in rest if d in any_metric]
    failed    = [d for d in rest if d not in any_metric and d in ran]
    absent    = [d for d in rest if d not in any_metric and d not in ran]

    notes: list[str] = []
    if dropped:
        notes.append(
            "Every run in the window excluded as unreliable: "
            f"{', '.join(dropped)}."
        )
    if no_metric:
        notes.append(
            f"No value for the selected metrics: {', '.join(no_metric)}."
        )
    if failed:
        notes.append(
            "Ran but produced no comparable metrics (see Regression Status): "
            f"{', '.join(failed)}."
        )
    if absent:
        notes.append(
            "No run in the trend window: "
            + ", ".join(
                f"{d} (last ran {last_run[d]})" if d in last_run else d
                for d in absent
            )
            + "."
        )
    if excluded:
        notes.append(
            f"Not benchmarked with this sample/platform: {', '.join(excluded)}."
        )
    return notes


def render(
    data_url: str, dashboard_url: str, platform: str, sample: str,
    window: tuple[date, date] | None,
) -> None:
    """The tab's four views (:data:`_VIEWS`), dispatched by a View switcher like
    the other multi-view tabs. *platform* and *sample* are the sidebar's
    selections, the same scoping as Run Trends. *window* is the sidebar's
    shared Trend window, as the report nights that judged its releases
    (``None`` only when the store has no release yet) — in that case
    :func:`nights_in_window` falls back to the latest
    :data:`_FALLBACK_NIGHTS`. It carries no detector scope: on this tab it
    ranges over every reported release benchmarked on the platform (see
    ``sidebar.pick_trend_window``), so the range this tab draws stays put
    when the sidebar detector changes rather than shrinking to that detector's
    releases and dropping the more recently benchmarked detectors off the
    charts. The window sets which report nights are fetched,
    and so how far back the Regression Status and Nightly Report pickers reach;
    the landscape reads whichever of them measured each detector last, plus the
    nights of any detector whose last run predates the window's end (see
    :func:`stale_run_nights`). *dashboard_url* is this deployment's own public
    URL, which only the Nightly Report view needs (see
    :mod:`tabs._nightly_email`).

    The selected view is itself deep-linkable through ``?view=``, so a copied
    URL reopens the view it was copied from rather than the default one — the
    only way the parameters a single view owns (``?report=`` above all) can
    survive being shared.

    A scope with no benchmarks is reported *inside* the view switcher rather
    than in place of it: three of the four views have nothing to draw without
    it, but the mail covers every scope the night measured and must stay
    reachable — a scope with no data is often exactly when it is worth
    reading."""
    try:
        dates = cached_list_report_dates(data_url, SIM)
    except Exception as err:
        st.warning(
            f"Could not list the nightly reports on EOS: {err}. "
            "Reload the page to try again."
        )
        return
    if not dates:
        st.info(
            "No regression reports available yet. The nightly benchmark "
            "workflow uploads the first report after its next run."
        )
        return

    # One parallel fetch for the whole window plus the latest night (which the
    # Regression Status picker opens on, and which carries the newest
    # measurement the landscape can show, even outside the window).
    latest_night = max(dates)
    hist_nights = nights_in_window(dates, window)
    fetch_nights = tuple(dict.fromkeys([latest_night, *hist_nights]))
    pinned_report = st.query_params.get("report")
    pinned_mail = st.query_params.get("view") == "Nightly Report" and pinned_report
    if pinned_mail:
        if pinned_report not in dates:
            st.error(f"The requested report ({pinned_report}) is not available.")
            return
        # A PR comment's archived report can lie outside the sidebar trend range.
        fetch_nights = tuple(dict.fromkeys([*fetch_nights, pinned_report]))
    reports = cached_fetch_reports(data_url, SIM, fetch_nights)
    if pinned_mail and pinned_report not in reports:
        st.error(f"Could not load the requested report ({pinned_report}) from EOS.")
        return
    if latest_night not in reports:
        if pinned_mail:
            # A deep-linked archived report stands on its own: the mail covers
            # every scope that night measured and needs nothing from the latest
            # report or the sidebar's window. Everything below this point does,
            # so the mail is rendered here rather than through the view switcher
            # — a PR comment's link must not break because an unrelated night
            # failed to fetch.
            st.warning(
                f"Could not load the latest report ({latest_night}) from EOS — "
                f"showing the requested {pinned_report} report on its own."
            )
            _nightly_email.render(data_url, dashboard_url, reports, latest_night)
            return
        st.warning(f"Could not load the latest report ({latest_night}) from EOS.")
        return

    # A detector that missed the newest night is carried in that report as a
    # stale group with its verdicts stripped, so its last real measurement sits
    # in an older report the window need not reach. One follow-up fetch for
    # those nights (bounded by the engine's missing-run grace period) keeps the
    # landscape on each detector's own most recent run instead of silently
    # showing it at the window's end — or not at all.
    stragglers = tuple(
        n for n in stale_run_nights(reports[latest_night], latest_night, platform, sample)
        if n in dates and n not in reports
    )
    if stragglers:
        reports.update(cached_fetch_reports(data_url, SIM, stragglers))

    _views(data_url, dashboard_url, overview_inputs(
        reports,
        platform=platform,
        sample=sample,
        latest_night=latest_night,
        hist_nights=hist_nights,
        fetch_nights=fetch_nights,
        snap_nights=tuple(n for n in (*fetch_nights, *stragglers) if n in reports),
    ))


# The views live in a fragment so switching one, toggling a metric, the scale,
# the exclude switch or a Confirmed/Watch pill reruns only this block — not the
# whole app (sidebar trend downloads, report reparse, every other tab). The
# inputs are built once per full rerun and passed in; a fragment rerun replays
# them. Keeping these clicks cheap matters on the CPU-capped single-replica
# deployment, where a burst of full reruns can starve the /_stcore/health probe
# and bounce the pod (surfacing as a 503).
@st.fragment
def _views(data_url: str, dashboard_url: str, inputs: OverviewInputs) -> None:
    # The chosen view is part of the URL, so a copied link reopens the one
    # being read. Without it every Overview link lands on the default view,
    # and the parameters the *other* views own — ``?report=`` above all,
    # which both Regression Status and Nightly Report speak — would be
    # carried in the URL with no picker on screen to seed from them.
    seed_query_param(_VIEW_KEY, "view", _VIEWS)
    # Seeding through session_state rules out an explicit default, so the
    # opening view is written the same way. Same pattern as the Regressions
    # night picker.
    if _VIEW_KEY not in st.session_state:
        st.session_state[_VIEW_KEY] = _VIEWS[0]
    view = st.radio(
        "**View**", _VIEWS, horizontal=True, key=_VIEW_KEY,
    ) or _VIEWS[0]
    st.query_params["view"] = view
    if view == "Nightly Report":
        _nightly_email.render(
            data_url, dashboard_url, inputs.mail_reports, inputs.latest_night,
        )
        return
    # Below the mail, so an empty scope can still read the night's report.
    if inputs.scope_empty:
        st.info(
            f"No detector has {BASELINE_LABEL} results for "
            f"**{inputs.sample}** on **{inputs.platform}** — pick another sample or "
            "platform in the sidebar, or read the whole night's report in "
            "**Nightly Report**."
        )
        return
    if view == "Regression Status":
        _regression_status.render(
            inputs.scoped_groups, inputs.status_frames, inputs.window_nights,
            inputs.latest_night, inputs.platform, inputs.sample, inputs.rel_hist,
        )
        return
    choice = _figure_toolbar(view, inputs)
    if view == "Performance Trends":
        _render_trends(inputs, choice)
    else:
        _render_landscape(inputs, choice)


#: The views of a suite without nightly reports: the two figure views alone.
FIGURE_VIEWS = _VIEWS[:2]


@st.fragment
def render_figures(inputs: OverviewInputs, *, key_prefix: str) -> None:
    """The Performance Trends and Landscape views for a suite no nightly
    report judges: *inputs* are built from its runs, whose values carry no
    verdict, so the Regressions flags are left out. *key_prefix* keeps the
    page's own selection, as for the sidebar."""
    view_key = f"{key_prefix}{_VIEW_KEY}"
    seed_query_param(view_key, "view", FIGURE_VIEWS)
    if view_key not in st.session_state:
        st.session_state[view_key] = FIGURE_VIEWS[0]
    view = st.radio("**View**", FIGURE_VIEWS, horizontal=True, key=view_key) or FIGURE_VIEWS[0]
    st.query_params["view"] = view
    if inputs.scope_empty:
        st.info(
            f"No detector has {BASELINE_LABEL} results for "
            f"**{inputs.sample}** on **{inputs.platform}** in the trend window — "
            "pick another sample or platform, or widen the window in the sidebar."
        )
        return
    choice = _figure_toolbar(view, inputs, key_prefix=key_prefix, flags=False)
    if view == "Performance Trends":
        _render_trends(inputs, choice, key_prefix)
    else:
        _render_landscape(inputs, choice, key_prefix)


@dataclass(frozen=True)
class _FigureChoice:
    """What the toolbar the two figure views share has selected."""

    time_metric: str
    mem_metric: str
    log: bool
    relative: bool
    show_confirmed: bool
    show_watch: bool
    #: The ``(run_night, detector)`` runs that failed the reliability check, and
    #: whether they are excluded.
    unreliable: set[tuple[str, str]]
    exclude_unreliable: bool


def _figure_toolbar(
    view: str, inputs: OverviewInputs, *, key_prefix: str = "", flags: bool = True,
) -> _FigureChoice:
    """The shaping controls shared by the two figure views. Same widget keys in
    both, so the chosen metrics survive a view switch; only Scale differs
    (Relative % only makes sense for a time series). *key_prefix* prefixes
    every key, so each page keeps its own choice; without *flags* the history
    carries no regression verdicts to mark."""
    trends = view == "Performance Trends"
    show_confirmed = show_watch = False
    toolbar = st.container(
        border=True, horizontal=True, vertical_alignment="bottom",
        width="stretch", gap="small",
    )
    with toolbar:
        comparison = st.container(
            horizontal=True, vertical_alignment="bottom", width="content",
            gap="small",
        )
        with comparison:
            seed_query_param(f"{key_prefix}det_ov_time_metric", "tmetric", TIME_METRICS)
            time_metric = st.selectbox(
                "Time", TIME_METRICS, key=f"{key_prefix}det_ov_time_metric",
                format_func=_metric_title, width=210,
                help="Per-event means/medians exclude the warmup event; "
                     "wall time and user CPU cover the whole run including "
                     "initialization.",
            )
            seed_query_param(f"{key_prefix}det_ov_mem_metric", "mmetric", MEMORY_METRICS)
            if f"{key_prefix}det_ov_mem_metric" not in st.session_state:
                memory_rows = inputs.hist_rows if trends else inputs.snap_rows
                available = set(memory_rows["metric"])
                st.session_state[f"{key_prefix}det_ov_mem_metric"] = next(
                    (metric for metric in MEMORY_METRICS if metric in available),
                    MEMORY_METRICS[0],
                )
            mem_metric = st.selectbox(
                "Memory",
                MEMORY_METRICS,
                key=f"{key_prefix}det_ov_mem_metric",
                format_func=_metric_title,
                width=210,
                help="Anonymous RSS is averaged over events. Virtual peak includes "
                "initialisation. Total RSS metrics remain residency diagnostics.",
            )
            if trends:
                scale = st.segmented_control(
                    "Scale", ["Log", "Linear", "Relative %"],
                    default="Log", key=f"{key_prefix}det_ov_scale", bind="query-params",
                    help="Log keeps the detectors' >1-decade spread readable; "
                         "Linear shows absolute values; Relative % rescales "
                         "each line to its first plotted night = 100%, so "
                         "drift is comparable across detectors of very "
                         "different absolute cost.",
                ) or "Log"
            else:
                scale = st.segmented_control(
                    "Scale", ["Log", "Linear"],
                    default="Log", key=f"{key_prefix}det_ov_scale_land",
                    bind="query-params",
                    help="Log keeps the detectors' >1-decade spread "
                         "readable; Linear shows absolute values.",
                ) or "Log"

        overlays = st.container(
            horizontal=True, vertical_alignment="bottom", width="stretch",
            horizontal_alignment="right", gap="small",
        )
        with overlays:
            if trends and flags:
                show_confirmed, show_watch = render_flag_pills(
                    f"{key_prefix}det_ov_flags", label="Regressions",
                )
            unreliable, exclude_unreliable = render_run_reliability_filter(
                inputs.rel_hist, key=f"{key_prefix}det_ov_exclude_unreliable", inline=True,
            )
    # Make the selected comparison shareable: ?tmetric=...&mmetric=...
    st.query_params["tmetric"] = time_metric
    st.query_params["mmetric"] = mem_metric
    return _FigureChoice(
        time_metric=time_metric,
        mem_metric=mem_metric,
        log=scale == "Log",
        relative=scale == "Relative %",
        show_confirmed=show_confirmed,
        show_watch=show_watch,
        unreliable=unreliable,
        exclude_unreliable=exclude_unreliable,
    )


def _render_trends(
    inputs: OverviewInputs, choice: _FigureChoice, key_prefix: str = "",
) -> None:
    """The Performance Trends view: both metrics' history, captioned with every
    scoped detector the chart has no line for.

    Unreliable runs are dropped before the tag reduction, never after: a flag
    belongs to the run that earned it, so excluding that run has to take its
    marker with it — and must not take the marker of a reliable rerun of the
    same tag. Failed measurements are kept apart throughout and plotted at their
    own values. The unfiltered rows stay bound for the caption, which tells a
    detector the toggle dropped from one that never ran in the window."""
    rows = inputs.hist_rows
    failed_rows = rows[rows["severity"] == Severity.FAILURE.value]
    kept = rows
    if choice.exclude_unreliable:
        kept = pd.concat([
            drop_unreliable_runs(
                rows[rows["severity"] != Severity.FAILURE.value], choice.unreliable,
            ),
            failed_rows,
        ], ignore_index=True)
    hist = collapse_history(kept)
    plotted = pd.concat([hist, failed_rows], ignore_index=True)

    shown = _history_in_display_units(plotted)
    if choice.relative:
        shown = relative_history(shown)
    hist_shown = shown[shown["severity"] != Severity.FAILURE.value]
    failed_shown = shown[shown["severity"] == Severity.FAILURE.value]
    undefined_failures = failed_shown[
        failed_shown["metric"].isin((choice.time_metric, choice.mem_metric))
        & failed_shown["value"].isna()
    ]
    failed_shown = failed_shown.dropna(subset=["value"])

    # Worked out before the figure: when there is nothing to draw at all the
    # reader needs this *most*, and that is exactly the branch that returns
    # without a chart to caption.
    notes = _trend_notes(
        plotted, rows, choice.time_metric, choice.mem_metric,
        inputs.detectors, inputs.window_roster, inputs.excluded, inputs.last_run,
    )
    if choice.relative and not undefined_failures.empty:
        names = ", ".join(sorted(undefined_failures["detector"].unique()))
        notes.append(
            "Failed measurements cannot be shown in Relative % without "
            f"a successful baseline: {names}."
        )
    fig = _history_figure(
        hist_shown, choice.time_metric, choice.mem_metric, inputs.styles,
        inputs.detectors, 0.75, choice.log, choice.relative,
        choice.show_confirmed, choice.show_watch, failed_shown,
    )
    if fig is None:
        st.info(" ".join([
            "No values for the selected metrics in this history window.",
            *notes,
        ]))
        return
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}det_ov_hist_chart")
    # The span of the x-axis, in nightly tags like Run Trends' "Data range" —
    # not of the report nights read, which run past the newest tag whenever a
    # night re-ran it.
    drawn = pd.concat([hist_shown, failed_shown], ignore_index=True)
    tags = sorted(drawn.loc[
        drawn["metric"].isin((choice.time_metric, choice.mem_metric)), "night"
    ].unique())
    span = (
        f"Latest night: **{inputs.latest_night}** · Data range: "
        f"**{tags[0]}** → **{tags[-1]}** "
        f"({len(tags)} nightly tag{'s' if len(tags) != 1 else ''})."
        if tags else f"Latest night: **{inputs.latest_night}**."
    )
    st.caption(" ".join([span, *notes]))


def _render_landscape(
    inputs: OverviewInputs, choice: _FigureChoice, key_prefix: str = "",
) -> None:
    """The Performance Landscape view: one point per detector at its own most
    recent run.

    An ordinary cross-detector comparison with no failure-marker vocabulary, so
    failed partial values are left to the explicitly marked historical views.
    The snapshot is filtered *before* it is taken, so an unreliable newest run
    falls back to the detector's last reliable one rather than dropping it off
    the landscape entirely."""
    rows = inputs.snap_rows[inputs.snap_rows["severity"] != Severity.FAILURE.value]
    if choice.exclude_unreliable:
        rows = drop_unreliable_runs(rows, choice.unreliable)
    wide, as_of = latest_snapshot(rows)
    fig = _landscape_figure(
        _wide_in_display_units(wide), choice.time_metric, choice.mem_metric,
        inputs.styles, inputs.detectors, 0.75, choice.log, as_of,
    )
    if fig is None:
        st.info(
            "No values for the selected metrics in this window."
            + (" Every run was excluded as unreliable."
               if choice.exclude_unreliable and choice.unreliable else "")
        )
        return
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}det_ov_land_chart")
    st.caption(" ".join(_landscape_notes(
        wide, as_of, choice.time_metric, choice.mem_metric,
        inputs.scoped_detectors, inputs.excluded,
    )))
