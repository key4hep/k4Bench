"""The Overview tab's Regression Status view — one report night across detectors.

The Regressions tab is scoped to one detector, so this is where the
cross-detector regression picture lives: a report-night picker, that night's
verdict banner and per-detector roster, and its worst flag's trend. Everything is
built from the cached nightly reports; nothing here downloads a run.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE
from k4bench.metrics import metric_title
from k4bench.regression.engine import Z_THRESHOLD
from k4bench.regression.models import RunGroupReport, Severity
from k4bench.regression.render import detector_badge
from tabs.reports import _blame
from tabs.reports._night_picker import render_night_picker
from tabs.reports.overview._data import (
    METRIC_ORDER,
    collapse_history,
    detector_status_rows,
    drop_unreliable_runs,
    history_rows,
)
from tabs._regression_flags import (
    add_severity_markers,
    attention_key,
    failed_config_labels,
    failed_metric_options,
)
from tabs.reports._metric_trend import render_metric_picker
from tabs._reliability import render_run_reliability_filter
from ui_utils import reset_widget_on_scope, to_rgba

#: Fill for the accepted-baseline band on the flag-trend chart — the same
#: visual device as the Regressions tab's drill-down.
_BASELINE_FILL = "rgba(31,119,180,0.08)"


def _render_regression_banner(groups: list[RunGroupReport], night: str) -> None:
    """The selected night's cross-detector verdict at a glance, over the same
    platform/sample scope as the rest of the tab — the summary the (now
    detector-scoped) Regressions tab no longer carries."""
    n_regr = sum(len(g.regressions) for g in groups)
    n_watch = sum(len(g.watches) for g in groups)
    n_fail = sum(len(g.failures) + len(g.job_failures) for g in groups)
    with st.container(border=True):
        st.markdown(f"##### Nightly verdict at a glance — {night}")
        cols = st.columns(4)
        cols[0].metric(
            "Detectors checked", len(groups),
            help="Detectors with a run group for the selected platform and "
                 "sample in this night's report, each judged against its own "
                 "baseline.",
        )
        cols[1].metric(
            "🔴 Regressed", n_regr,
            help="Metrics that crossed both detection gates on two consecutive "
                 "reliable nights (confirmed), either direction — not judged good "
                 "or bad, only that it moved beyond the baseline twice in a row.",
        )
        cols[2].metric(
            "⚠️ Watch", n_watch,
            help="Metrics flagged for the first time this night. Not alerted on: "
                 "they either confirm on the next reliable night or clear.",
        )
        cols[3].metric(
            "❌ Failures", n_fail,
            help="Hard job failures: a config exiting non-zero, producing no "
                 "results, or a whole run missing for the night. These alert "
                 "immediately, no confirmation needed.",
        )


def _render_detector_status(
    groups: list[RunGroupReport], night: str, platform: str, sample: str
) -> None:
    """Per-detector roster for the selected night — each row deep-links into
    the Regressions tab scoped to that detector and pinned to *night*."""
    rows = detector_status_rows(groups, platform, sample, night)
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "": st.column_config.TextColumn(
                    "", width="small",
                    help="Worst state tonight: ❌ failure · 🔴 confirmed "
                         "regression · ⚠️ watch · ❔ not judged · ✅ quiet",
                ),
                "🔴": st.column_config.NumberColumn(
                    "🔴", width="small", help="Confirmed regressions tonight.",
                ),
                "⚠️": st.column_config.NumberColumn(
                    "⚠️", width="small", help="First-time flags (unconfirmed).",
                ),
                "❌": st.column_config.NumberColumn(
                    "❌", width="small", help="Hard job/config failures.",
                ),
                "Worst flag": st.column_config.TextColumn(
                    "Worst flag",
                    help="The most severe flagged metric (confirmed before "
                         "watch, then largest |Δ|) and its config.",
                ),
                "Δ": st.column_config.NumberColumn(
                    "Δ", format="%+.1f%%",
                    help="Size and direction of the worst flag vs its baseline "
                         "median. Blank when the metric has no meaningful "
                         "relative change.",
                ),
                "Inspect": st.column_config.LinkColumn(
                    "Inspect", display_text="↗ Regressions", width="small",
                    help="Open the Regressions tab scoped to this detector "
                         "(and the selected platform and sample).",
                ),
            },
        )


def _flag_choices(groups: list[RunGroupReport]) -> list:
    """The selected night's flagged verdicts that the report history can plot
    (top-level rows of the compared metric set), worst first — the options of
    the Regression Status view's trend preview."""
    choices = []
    for group in groups:
        failures = failed_config_labels(group.verdicts)
        choices.extend(
            v for v in group.verdicts
            if v.severity in (Severity.WATCH, Severity.CONFIRMED)
            and v.label not in failures
            and v.sub_detector is None
            and v.metric in METRIC_ORDER
        )
        choices.extend(failed_metric_options(
            group.verdicts, metrics=set(METRIC_ORDER),
        ))
    return sorted(choices, key=attention_key)


def _flag_axis_title(verdict) -> str:
    """Axis title in the report's *stored* units (MB for memory): the flag
    trend draws the verdict's own baseline band, so the axis must match those
    raw numbers rather than the GB display the figure panels use."""
    return metric_title(verdict.metric)


def _flag_trend_figure(
    series: pd.DataFrame,
    verdict,
    failures: pd.DataFrame | None = None,
) -> go.Figure:
    """One flagged metric's history across the trend window, with the baseline
    band its verdict was judged against (median ± the detection gate), every
    flagged night ringed with the standard halo (the ⚠️→🔴 progression), and —
    for a confirmed step — the blame window shaded. Mirrors the Regressions
    tab's drill-down, but built entirely from the cached nightly reports."""
    df = series.copy()
    if failures is not None and not failures.empty:
        df = pd.concat([df, failures], ignore_index=True)
    df["night_dt"] = pd.to_datetime(df["night"])
    df = df.sort_values("night_dt")
    line = df[df["severity"] != Severity.FAILURE.value]

    fig = go.Figure()
    if not line.empty:
        fig.add_trace(go.Scatter(
            x=line["night_dt"],
            y=line["value"],
            mode="lines+markers",
            name=verdict.detector,
            line=dict(color=PALETTE[0], width=2),
            marker=dict(size=7, color=to_rgba(PALETTE[0], 0.55),
                        line=dict(color=PALETTE[0], width=1.5)),
            customdata=line["k4h_release"].fillna("unknown"),
            hovertemplate=(
                f"<b>{verdict.detector}</b><br>"
                "Tag: %{customdata} (%{x|%Y-%m-%d})<br>"
                f"{_flag_axis_title(verdict)}: %{{y:.4g}}<extra></extra>"
            ),
        ))
    med, mad = verdict.baseline_median, verdict.baseline_mad or 0.0
    if med is not None:
        fig.add_hline(y=med, line_dash="dash", line_color=PALETTE[0], line_width=1,
                      annotation_text="baseline median", annotation_font_size=11)
        if mad > 0:
            fig.add_hrect(y0=med - Z_THRESHOLD * mad, y1=med + Z_THRESHOLD * mad,
                          fillcolor=_BASELINE_FILL, line_width=0)
    for sev in ("FAILURE", "WATCH", "CONFIRMED"):
        flagged = df[df["severity"] == sev]
        if not flagged.empty:
            add_severity_markers(
                fig, flagged, x_col="night_dt", y_col="value",
                name_col="detector", severity=sev, hover_y="%{y:.4g}",
            )
    if _blame.has_window(verdict):
        # add_window_band reads the x span from an ``x_date`` column; the flag
        # trend's x is the same release-date axis under another name.
        _blame.add_window_band(fig, df.rename(columns={"night_dt": "x_date"}), verdict)

    unique_dates = sorted(df["night_dt"].dropna().unique())
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
        yaxis_title=_flag_axis_title(verdict),
        showlegend=False,
    )
    return fig


def _flag_trend_frames(
    night_frames: list[tuple[str, pd.DataFrame]],
    window_nights: list[str],
    night: str,
) -> list[tuple[str, pd.DataFrame]]:
    """The report nights the flagged-metric trend plots: the sidebar's trend
    window, plus the selected night when it falls outside it.

    The window is what the chart promises, and the verdict's baseline band
    belongs beside the nights it was judged over. Handing the trend every
    fetched night instead would drop the newest report into a historical view
    — a point weeks past the window on screen, sitting next to an older
    verdict's band. The selected night is kept regardless so its own point is
    never missing, which also covers the default case of the newest report
    lying beyond a window that ends earlier.
    """
    keep = {*window_nights, night}
    return [(n, frame) for n, frame in night_frames if n in keep]


def _render_flag_trend(
    groups: list[RunGroupReport],
    status_frames: list[tuple[str, pd.DataFrame]],
    platform: str,
    sample: str,
    status_rel_hist: pd.DataFrame,
    key_prefix: str = "",
) -> None:
    """The Regression Status view's trend preview — the shared worst-first
    picker, leading with the detector because this view spans them, above a
    chart that costs no run downloads: the series is the verdicts' raw nightly
    values across the already-fetched reports.

    Runs the same filter-before-collapse pipeline as the other historical views,
    on the same widget key: a chart of raw nightly measurements has to honour the
    unreliable-run exclusion, and honouring it in one Overview sub-view but not
    another would make the tab's answer depend on which view is open."""
    choices = _flag_choices(groups)
    if not choices:
        return
    st.markdown("###### Flagged-metric trend")
    # The picker re-defaults by itself whenever the stored verdict leaves the
    # option model, but “—” survives every model: without this, a hidden chart
    # would stay hidden through a scope change that surfaced a worse flag.
    reset_widget_on_scope(
        f"{key_prefix}det_ov_flag_trend", (platform, sample, tuple(choices)),
    )
    controls = st.container(
        horizontal=True, vertical_alignment="bottom", width="stretch", gap="medium",
    )
    with controls:
        picker = st.container(width="stretch")
        with picker:
            v = render_metric_picker(
                choices, key=f"{key_prefix}det_ov_flag_trend", include_detector=True,
                help="The flagged metric's history over the trend window, with the "
                     "baseline band its verdict was judged against — opens on the "
                     "worst flag; pick another or “—” to hide it. Built from the "
                     "nightly reports, no run downloads.",
            )
        actions = st.container(
            horizontal=True, horizontal_alignment="right",
            vertical_alignment="bottom", width="content",
        )
        reliability_slot = actions.empty()
    if v is None:
        return
    unreliable_pairs, exclude_unreliable = render_run_reliability_filter(
        status_rel_hist, key=f"{key_prefix}det_ov_exclude_unreliable",
        slot=reliability_slot,
    )
    rows = history_rows(status_frames, platform, sample, v.label)
    failed_rows = rows[rows["severity"] == Severity.FAILURE.value]
    if exclude_unreliable:
        rows = drop_unreliable_runs(
            rows[rows["severity"] != Severity.FAILURE.value],
            unreliable_pairs,
        )
    hist = collapse_history(rows)
    series = hist[(hist["detector"] == v.detector) & (hist["metric"] == v.metric)]
    failures = failed_rows[
        (failed_rows["detector"] == v.detector)
        & (failed_rows["metric"] == v.metric)
    ]
    if series.empty and failures.empty:
        st.info("No history for this metric in the current trend window.")
        return
    st.plotly_chart(
        _flag_trend_figure(series, v, failures),
        width="stretch", key=f"{key_prefix}det_ov_flag_chart",
    )
    st.caption(f"**{v.reason}** — {v.detector} · {v.label}")


#: Session key of the report-night picker. Shares ``?report=`` with the
#: Regressions tab's own picker — one parameter meaning "the report night being
#: read", so the roster's Inspect links, the nightly email's deep links and this
#: view all speak the same URL.
_NIGHT_KEY = "det_ov_report_night"


def _select_report_night(
    scoped_groups: dict[str, list[RunGroupReport]],
    latest_night: str,
    platform: str,
    sample: str,
    key_prefix: str = "",
) -> str:
    """The report night the status view describes — newest by default, with
    every already-fetched night that covers the scope on offer.

    The options are the nights this tab has loaded anyway (the sidebar's trend
    window plus the newest report), so picking one costs no download. The
    picker re-defaults when the sidebar scope changes (see
    :func:`tabs.reports._night_picker.render_night_picker` for why a carried-over
    night must not survive one).
    """
    nights = sorted(scoped_groups, reverse=True)
    return render_night_picker(
        nights,
        key=f"{key_prefix}{_NIGHT_KEY}",
        badge=lambda n: detector_badge(scoped_groups[n]),
        default=nights[0],
        latest=latest_night,
        reset_scope=(platform, sample),
        help="Which night's report the verdicts below come from — the badge is "
             "that night's worst state across the scoped detectors. Defaults "
             "to the newest; the trend window sets how far back the picker "
             "reaches. The figures' history and the landscape are unaffected.",
    )


def render(
    scoped_groups: dict[str, list[RunGroupReport]],
    night_frames: list[tuple[str, pd.DataFrame]],
    window_nights: list[str],
    latest_night: str,
    platform: str,
    sample: str,
    rel_hist: pd.DataFrame,
    key_prefix: str = "",
) -> None:
    """The Regression Status view: the report-night picker, that night's
    verdict banner and per-detector roster, and its worst flag's trend — the
    cross-detector regression picture the (detector-scoped) Regressions tab no
    longer carries. *key_prefix* prefixes every widget key, as the Overview's
    figure toolbar does, so each page keeps its own choices.

    *night_frames* spans the nights the picker offers (the trend window plus
    the newest report); *window_nights* names the window within them, which is
    what the flag trend is drawn over (see :func:`_flag_trend_frames`).

    Only the trend is filtered by *rel_hist*: the banner and roster report what
    the selected night's report *says*, including that a detector was too
    contended to judge, and a filter that hid those rows would hide the very
    runs it excluded."""
    if not scoped_groups:
        st.info(
            f"No detector has a run group for **{sample}** on **{platform}** "
            f"in the reports in view (newest: {latest_night})."
        )
        return
    night = _select_report_night(
        scoped_groups, latest_night, platform, sample, key_prefix,
    )
    groups = scoped_groups[night]
    _render_regression_banner(groups, night)
    _render_detector_status(groups, night, platform, sample)
    _render_flag_trend(
        groups, _flag_trend_frames(night_frames, window_nights, night),
        platform, sample, rel_hist, key_prefix,
    )
