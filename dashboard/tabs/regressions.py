"""Regressions tab — one night's regression report for the sidebar's scope.

Scoped, like the trend views, by the sidebar's detector/platform/sample
selection: the tab renders the matching run group from the precomputed
``_reports/{date}/report.json`` written to EOS by the nightly
``regression-report`` CI job (see ``k4bench/regression/``). The sidebar's
*release* selects which report nights are on offer (see
:func:`_candidate_nights`): several nights routinely re-benchmark one fixed
nightly, and the engine judges them all against the same frozen baseline, so
a confirmed regression repeats on every night of the release that trips —
though nights can still differ (WATCH → CONFIRMED progression, marginal OK
nights, pre-backfill reports). The default night is therefore the most
attention-worthy one, a picker exposes the release's other nights, and
``?report=`` pins one directly (the deep link emitted in alert emails). The
cross-detector at-a-glance picture lives in the Overview tab. Only the trend
preview downloads run data, and only for the series being inspected.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import requests
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
    _cached_fetch_blame,
    _cached_fetch_reports,
    _cached_fetch_runs_windowed,
    _cached_list_report_dates,
    _cached_list_run_dates,
)
from k4bench.blame.models import BlameReport, BlameSchemaError
from k4bench.provenance.diff import diff_packages
from tabs import _blame
from tabs._regression_flags import (
    add_severity_markers,
    attention_key,
    candidate_table,
    has_ranking as candidate_has_ranking,
)
from tabs._reliability import render_reliability_filter
from tabs.stack_changes import _release, deep_link, packages_for_release
from ui_chrome import _drop_stale_selection, seed_query_param
from ui_utils import (
    _is_valid_df,
    _METRIC_LABELS,
    _METRIC_UNITS,
    _reset_widget_on_scope,
    _to_rgba,
)

_log = logging.getLogger(__name__)

#: Fill for the accepted-baseline band on the drill-down chart — same visual
#: device as machine_info's threshold shading, in the palette's first hue.
_BASELINE_FILL = "rgba(31,119,180,0.08)"

#: Docs section explaining how the step-change detector judges a metric, linked
#: from the tab's intro instead of restating the logic inline.
_ASSESSMENT_DOCS_URL = (
    "https://key4hep.github.io/k4Bench/user-guide/features/dashboard/#regressions-tab"
)


def render(
    data_url: str, cache_dir: str, detector: str, platform: str, sample: str,
    stack: str,
) -> None:
    st.caption(
        "Nightly step-change detection for the sidebar's detector, platform, "
        "sample and release — the cross-detector picture lives in the Overview "
        f"tab. [Learn how regressions are assessed →]({_ASSESSMENT_DOCS_URL})"
    )

    dates = _cached_list_report_dates(data_url)
    if not dates:
        st.info(
            "No regression reports available yet. The nightly benchmark workflow "
            "uploads the first report after its next run."
        )
        return

    nights = _candidate_nights(data_url, detector, platform, sample, stack, dates)
    if nights is None:
        st.info(
            f"No nightly report covers release **{_release(stack)}**'s runs — "
            f"reports begin on {min(dates)}. Pick a newer release in the sidebar."
        )
        return
    reports, unavailable = _load_reports(data_url, nights)
    if not reports:
        st.warning(
            f"Could not load release **{_release(stack)}**'s report(s) from EOS."
        )
        return
    # A ?report= deep link pointing at a night that could not be loaded must say
    # so, rather than silently defaulting to another night and rewriting the URL
    # under the reader (who followed a link to a specific report).
    pinned = st.query_params.get("report")
    if pinned in unavailable:
        st.warning(
            f"The pinned **{pinned}** report could not be loaded — showing "
            "another night for this release instead.",
            icon="⚠️",
        )
    elif unavailable:
        st.caption(
            f"❔ {len(unavailable)} of {len(nights)} report night(s) for this "
            "release could not be loaded (a transient EOS error or a malformed "
            "report)."
        )
    default_night = _pick_night(reports, detector, platform, sample)
    night = _select_night(reports, default_night, detector, platform, sample, stack)
    st.query_params["report"] = night
    if night != max(dates):
        st.caption(
            f"Historical view — release **{_release(stack)}**'s **{night}** "
            "report. The sidebar default (newest release) shows the latest one."
        )
    report = reports[night]

    # The blame sidecar is best-effort and absent on most nights (only a
    # confirmed, attributable regression produces one) — a missing blame.json is
    # normal, not an error, so this is None far more often than not. A sidecar
    # that parses as JSON but not as a blame report is treated the same way:
    # blame hides, the report renders.
    raw_blame = _cached_fetch_blame(data_url, night)
    blame: BlameReport | None = None
    if raw_blame:
        try:
            blame = BlameReport.from_json(raw_blame)
        except BlameSchemaError:
            blame = None

    # A (detector, platform, sample) triple is one run group — the report's
    # unit of judgement — so the sidebar scope selects at most one.
    group = _night_group(report, detector, platform, sample)
    if group is None:
        _render_no_group_notice(report, detector, sample, night)
        return

    _render_banner(group)
    _render_group(
        group, data_url, cache_dir, blame=blame,
        key=f"{detector}_{platform}_{sample}",
        scope=(stack, night),
    )


def _load_reports(
    data_url: str, nights: list[str]
) -> tuple[dict[str, NightlyReport], list[str]]:
    """Fetch and parse each candidate night **independently**, returning the
    parseable reports and the nights that could not be loaded.

    The tab now loads the release's whole report history, not just the one night
    it shows, so a single half-uploaded or schema-drifted historical report must
    not blank every other night's — parsing per night (rather than in one
    comprehension) contains that blast radius to the offending night."""
    raws = _cached_fetch_reports(data_url, tuple(nights))
    reports: dict[str, NightlyReport] = {}
    unavailable: list[str] = []
    for n in nights:
        raw = raws.get(n)
        if not raw:
            unavailable.append(n)
            continue
        try:
            reports[n] = from_json(raw)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            _log.warning("regressions: skipping malformed report for %s — %s", n, exc)
            unavailable.append(n)
    return reports, unavailable


def _candidate_nights(
    data_url: str, detector: str, platform: str, sample: str, stack: str,
    dates: list[str],
) -> list[str] | None:
    """Report nights on offer for the sidebar's release, newest first.

    A report is written per *run*, under ``_reports/{run_date}`` — so several
    nights routinely re-benchmark one fixed release, and while the engine
    judges them all against the same frozen baseline, their reports can still
    differ (a WATCH night preceding the confirming one, a marginal OK night,
    or a report predating the release-grouped engine). Every night that
    benchmarked this release and has a report is therefore offered, so the
    most alarming night stays reachable whichever night is open. While a
    release is the one still being benchmarked (it owns the
    triple's newest run), the latest report is included too — it may be newer
    than the release's last run, the night whose "no run uploaded" failure must
    stay visible.

    ``None`` when the release's runs all predate the first report;
    ``[max(dates)]`` (the latest report) on a run-history listing failure or an
    empty listing — the only sensible fallback left.
    """
    try:
        stacks_dates = _cached_list_run_dates(data_url, detector, platform, sample)
    except requests.RequestException:
        # Only a listing failure falls back silently-to-the-reader-but-not-to-us:
        # the sidebar may have an old release selected, so showing the latest
        # report without saying so would pass off today's data as that
        # release's. A narrower except than a scan turning up genuinely empty
        # (below) — that case has no such mismatch risk.
        st.warning(
            "Could not check this detector's run history on EOS — showing the "
            "latest report; it may not match the sidebar's selected release.",
            icon="⚠️",
        )
        return [max(dates)]
    run_dates = stacks_dates.get(stack) or ()
    if not run_dates:
        # No run listing for this release — the latest report is the only
        # sensible answer left.
        return [max(dates)]
    dateset = set(dates)
    nights = {d for d in run_dates if d in dateset}
    newest_any = max(d for ds in stacks_dates.values() for d in ds)
    if max(run_dates) == newest_any:
        # Active release: also offer the latest report, so a night that
        # benchmarked nothing for this release (a missing-run failure) stays
        # visible even though it isn't one of the release's own run dates.
        nights.add(max(dates))
    if not nights:
        return None
    return sorted(nights, reverse=True)


def _night_group(
    report: NightlyReport, detector: str, platform: str, sample: str
) -> RunGroupReport | None:
    """The one run group in *report* matching the sidebar triple, or ``None``
    when the night has no group for it (a scope miss on that night)."""
    return next(
        (
            g for g in report.groups
            if (g.detector, g.platform, g.sample) == (detector, platform, sample)
        ),
        None,
    )


def _night_priority(
    report: NightlyReport, detector: str, platform: str, sample: str
) -> int:
    """How much attention a night's report warrants for the sidebar triple,
    used to default the picker to the most alarming night: ``2`` for a
    confirmed regression or any failure (matches :attr:`has_alertable`), ``1``
    for a watch, ``0`` for a quiet night or one with no group for the triple."""
    g = _night_group(report, detector, platform, sample)
    if g is None:
        return 0
    if g.regressions or g.failures or g.job_failures:
        return 2
    if g.watches:
        return 1
    return 0


def _pick_night(
    reports: dict[str, NightlyReport], detector: str, platform: str, sample: str,
) -> str:
    """The default report night for a re-benchmarked release: the most
    attention-worthy night, newest breaking ties. A confirmed night therefore
    wins over a quiet rerun, and a later confirmation wins over an earlier
    watch — the opposite failure mode from always taking the last run."""
    return max(
        reports,
        key=lambda n: (_night_priority(reports[n], detector, platform, sample), n),
    )


def _night_badge(
    report: NightlyReport, detector: str, platform: str, sample: str
) -> str:
    """The glance emoji (❌/🔴/⚠️/❔/✅) for a night's report, reusing the
    report's own detector badge so the picker speaks the same vocabulary as the
    Overview roster. ``❔`` when the night has no group for the triple."""
    g = _night_group(report, detector, platform, sample)
    return _detector_badge([g]) if g is not None else "❔"


_NIGHT_KEY = "regr_night"


def _forget_stale_scope(scope: tuple[str, ...]) -> None:
    """Reset the night picker when the sidebar scope changes.

    The picker uses one session key across every detector/platform/sample/
    release, but two scopes can share the same night *dates* while flagging
    their regression on *different* ones — so a night carried over from the
    previous scope could open a new one on a quiet report and hide exactly the
    regression this view exists to surface. On a scope change we drop both the
    stored night and the ``?report=`` we wrote for the old scope, so the new
    scope re-defaults. The incoming ``?report=`` on the first load (no prior
    scope recorded) is preserved, keeping deep links intact."""
    scope_key = "regr_night_scope"
    prev = st.session_state.get(scope_key)
    st.session_state[scope_key] = scope
    if prev is not None and prev != scope:
        st.session_state.pop(_NIGHT_KEY, None)
        st.query_params.pop("report", None)


def _select_night(
    reports: dict[str, NightlyReport], default_night: str,
    detector: str, platform: str, sample: str, stack: str,
) -> str:
    """Return the report night to render, always as a pill so the exact night
    on screen is never implicit — one pill and no caption for a release
    benchmarked on a single night, several pills otherwise. The picker
    defaults to *default_night* (the most attention-worthy), is authoritative
    via ``?report=`` for deep links, and re-defaults cleanly when the sidebar
    scope changes."""
    _forget_stale_scope((detector, platform, sample, stack))
    nights = sorted(reports, reverse=True)
    key = _NIGHT_KEY
    _drop_stale_selection(key, nights)          # stale night → re-default
    seed_query_param(key, "report", nights)     # ?report= wins when it's valid
    if key not in st.session_state:
        st.session_state[key] = default_night
    night = st.segmented_control(
        "Report night",
        nights,
        format_func=lambda n: f"{_night_badge(reports[n], detector, platform, sample)} {n}",
        key=key,
        help="Every night of a release is judged against the same baseline, "
             "so a confirmed regression repeats on each night that trips — "
             "but nights can still differ (the first strike is only a WATCH, "
             "and a marginal night can come out OK). Defaults to the most "
             "attention-worthy night; pick another to see that night's report "
             "and blame.",
    )
    if night is None:  # segmented_control lets the active pill be deselected
        night = default_night
    if len(nights) > 1:
        st.caption(f"Benchmarked on {len(nights)} nights — showing **{night}**.")
    return night


def _render_no_group_notice(
    report: NightlyReport, detector: str, sample: str, night: str
) -> None:
    """A scope miss names the scopes the report *does* cover, so the reader's
    next click is a sidebar switch rather than a dead end."""
    others = [g for g in report.groups if g.detector == detector]
    if others:
        scopes = "; ".join(_group_title(g) for g in others)
        st.info(
            f"The {night} report has no **{_pretty_sample(sample)}** run group "
            f"for **{detector}** on the selected platform. Judged that night: "
            f"{scopes} — switch the sidebar to one of those."
        )
    else:
        covered = ", ".join(sorted(report.by_detector())) or "none"
        st.info(
            f"**{detector}** is not in the {night} report. "
            f"Detectors covered: {covered}."
        )


def _render_banner(group: RunGroupReport) -> None:
    """The run group's verdict at a glance, mirroring machine_info's
    run-quality card. ``run_date`` is shown from the group, not the report
    night: a stale group (no run uploaded that night) carries its last run's
    date, and the mismatch with the picker is exactly the signal."""
    n_fail = len(group.failures) + len(group.job_failures)
    n_ok = sum(1 for v in group.verdicts if v.severity is Severity.OK)
    with st.container(border=True):
        st.markdown(
            f"##### Nightly verdict at a glance — {group.detector} · "
            f"{_pretty_sample(group.sample)} · {group.run_date or 'no data'}"
        )
        cols = st.columns(4)
        cols[0].metric(
            "🔴 Regressed", len(group.regressions),
            help="Metrics that crossed both detection gates on two consecutive "
                 "reliable nights (confirmed), either direction — not judged good "
                 "or bad, only that it moved beyond the baseline twice in a row.",
        )
        cols[1].metric(
            "⚠️ Watch", len(group.watches),
            help="Metrics flagged for the first time this night. Not alerted on: "
                 "they either confirm on the next reliable night or clear.",
        )
        cols[2].metric(
            "❌ Failures", n_fail,
            help="Hard job failures: a config exiting non-zero, producing no "
                 "results, or a whole run missing for the night. These alert "
                 "immediately, no confirmation needed.",
        )
        cols[3].metric(
            "✅ Within baseline", n_ok,
            help="Metrics inside the baseline's normal variation this night. "
                 "Metrics with too little reliable history to judge are counted "
                 "in neither column (see the summary line below).",
        )


def _yaxis_label(item: MetricVerdict) -> str:
    """Human-readable y-axis title with units, e.g. ``Wall time (s)`` — the raw
    column name is kept only in the hover tooltip."""
    name = _METRIC_LABELS.get(item.metric, item.metric)
    name = name[:1].upper() + name[1:]
    unit = _METRIC_UNITS.get(item.metric, "")
    return f"{name} ({unit})" if unit else name

def _window_changes(data_url: str, verdict: MetricVerdict) -> list | None:
    """Changed packages across a confirmed regression's bounded blame window,
    or ``None`` when either release's provenance is missing (aged off CVMFS, or
    a release benchmarked before capture)."""
    base = packages_for_release(data_url, verdict.platform, verdict.last_accepted_run_date)
    head = packages_for_release(data_url, verdict.platform, verdict.onset_run_date)
    if not base or not head:
        return None
    return diff_packages(base, head)


def _render_candidates(v: MetricVerdict, blame: BlameReport | None) -> None:
    """The ranked candidate pull requests for *v*, from the blame sidecar, or
    nothing when the night has no blame — or no ranking — for this verdict.

    Deliberately framed as *suggested, not evidence*: the score ranks how likely
    each PR is the cause, never asserts it — this repo leaves the call to a
    human. The forward package diff above already reaches the commit ranges; this
    narrows those to the individual PRs, orders them, and explains why."""
    if blame is None:
        return
    entry = blame.entry_for(v)
    if entry is None or not candidate_has_ranking(entry.candidates):
        return
    st.caption(
        "**Suggested cause — ranked candidate pull requests.** Ordered by how "
        "likely each is the cause, with a one-line reason. A lead to verify, not "
        "a verdict."
    )
    candidate_table(entry.candidates)


def _render_blame_card(data_url: str, v: MetricVerdict, blame: BlameReport | None) -> None:
    """One window's forward attribution: the upstream packages that moved in the
    blame window, each linking to its commit range, plus the ranked candidate
    PRs from the blame sidecar when present — so the reader reaches the likely
    pull request without leaving the row or loading the trend."""
    kind = _blame.classify(v)
    with st.container(border=True):
        if kind is _blame.WindowKind.SAME_STACK:
            st.caption(
                f"Appeared on **{v.onset_run_date}** — the same release as the baseline, "
                "so no tracked Key4hep package changed. Look at the benchmark "
                "code/config, inputs, runner environment, or noise."
            )
            return
        onset = v.onset_run_date
        baseline = v.last_accepted_run_date if kind is _blame.WindowKind.BOUNDED else None
        span = f"**{baseline} → {onset}**" if baseline else f"up to **{onset}**"
        st.caption(
            f"Appeared on **{onset}**, one reliable night before it confirmed — "
            f"the cause is in {span}."
        )
        changes = _window_changes(data_url, v) if baseline else None
        if changes is None:
            st.caption(
                "Stack provenance unavailable for these releases."
                if baseline else
                "No settled baseline before this step to bound the window on."
            )
        elif not changes:
            st.success("No tracked Key4hep package moved across this window.", icon="✅")
        else:
            st.markdown(
                f"**{len(changes)} package(s) moved:** " + _blame.changes_summary(changes)
            )
        _render_candidates(v, blame)
        st.link_button(
            "🔍 Open in Stack Changes →",
            deep_link(detector=v.detector, platform=v.platform, sample=v.sample,
                      head_release=onset, base_release=baseline),
        )


def _render_blame_cards(
    group: RunGroupReport, data_url: str, blame: BlameReport | None
) -> None:
    """Forward attribution for a group's confirmed regressions, without the
    drill-down. One card per distinct blame window — a run group's confirmed
    metrics usually share one, so the card only needs a representative
    verdict."""
    by_window: dict[tuple, MetricVerdict] = {}
    for v in group.verdicts:
        if _blame.has_window(v):
            by_window.setdefault((v.last_accepted_run_date, v.onset_run_date), v)
    if not by_window:
        return
    st.markdown("###### Upstream changes in the blame window")
    for verdict in by_window.values():
        _render_blame_card(data_url, verdict, blame)


def _render_group(
    group: RunGroupReport, data_url: str, cache_dir: str, *,
    blame: BlameReport | None, key: str, scope: tuple[str, str],
) -> None:
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
    if group.verdicts:
        st.caption(_quiet_summary(group))

    # The trend comes first — it's the first question a flagged night raises
    # ("what does this look like?") — the upstream-changes cards follow with
    # the "why", once there's a window to explain.
    drillable: list[MetricVerdict] = sorted(
        (v for v in flagged if v.baseline_median is not None), key=attention_key
    )
    if drillable:
        st.markdown("###### Flagged metric trend")
        with st.container(border=True):
            # Opens on the most severe flag rather than on "—", so the trend
            # answers that first question without a click. "—" stays
            # available to collapse the chart. Each option carries its own Δ
            # vs baseline, worst flag first, so scanning the list alone shows
            # the size of every flag — no separate ledger table needed.
            options = ["—"] + [
                f"{_badge(v)} · {_metric_name(v)} · {v.label} — Δ {_fmt_pct(v.pct_change)}"
                for v in drillable
            ]
            drill_key = f"regr_drill_{key}"
            _reset_widget_on_scope(drill_key, (*scope, tuple(options)))
            choice = st.selectbox(
                "Trend preview",
                options,
                index=1,
                key=drill_key,
                help="Recent history with the baseline band this verdict was "
                     "judged against. Opens on the most severe flag — pick "
                     "another, or “—” to hide the chart. Downloads data on "
                     "first use.",
            )
            if choice != "—":
                _render_drilldown(drillable[options.index(choice) - 1], data_url, cache_dir)

    _render_blame_cards(group, data_url, blame)


def _prev_point(df: pd.DataFrame, item: MetricVerdict) -> tuple | None:
    """The plotted point immediately before the flagged night — the fallback
    onset approximation for reports written before verdicts carried a recorded
    ``onset_*`` (see :func:`tabs._blame.onset_point`, which is exact when the
    field is present). Returns ``(x_date, value)`` or ``None`` if there is no
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


#: Extra Key4hep release tags plotted past the flagged night's own tag, when
#: available, so the chart shows whether a confirmed step held or the metric
#: moved again — not just the history it was judged against. Counted in
#: distinct tags (``stack``), not run dates: a rerun of the same tag must not
#: eat into the budget, or a tag re-benchmarked on several consecutive nights
#: would cap the window at zero *new* tags.
_FUTURE_TAGS = 3


def _metric_history(verdict: MetricVerdict, data_url: str, cache_dir: str):
    """Fetch this group's run window and return ``(df, reliability)`` for the
    verdict's metric series: *df* carries ``run_id``, ``x_date`` and the metric
    column, *reliability* is the ``{run_id: reliable}`` map so the caller can
    exclude unreliable runs the same way the Run Trends tab does."""
    stacks_dates = _cached_list_run_dates(
        data_url, verdict.detector, verdict.platform, verdict.sample
    )
    # Anchor the window on the verdict's own run, not today's newest, so an
    # older report drills down into the history it was judged against — the
    # flagged night, its onset and the blame band all land on the plotted data
    # instead of off the right edge — then extend a few Key4hep tags past it
    # (see _FUTURE_TAGS) when the release has since moved on. For the latest
    # report there is nothing past the anchor, so the common case is unchanged.
    all_pairs = sorted(
        (date, stack) for stack, dates in stacks_dates.items() for date in dates
    )
    anchor = next(
        (i for i in range(len(all_pairs) - 1, -1, -1) if all_pairs[i][0] <= verdict.run_id),
        None,
    )
    if anchor is None:
        pairs = []
    else:
        start = max(0, anchor - FETCH_WINDOW_RUNS + 1)
        seen_tags = {all_pairs[anchor][1]}
        new_tags = 0
        end = anchor + 1
        while end < len(all_pairs):
            if all_pairs[end][1] not in seen_tags:
                if new_tags >= _FUTURE_TAGS:
                    break
                seen_tags.add(all_pairs[end][1])
                new_tags += 1
            end += 1
        pairs = all_pairs[start:end]
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
        marker=dict(size=7, color=_to_rgba(PALETTE[0], 0.55),
                    line=dict(color=PALETTE[0], width=1.5)),
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
    # The flag markers reuse the Run Trends / Overview overlay
    # (:func:`add_severity_markers`) so all three tabs ring a flagged night with
    # the identical halo+badge, shape and colour — a single-point frame here
    # instead of the whole panel there. For a confirmed regression also mark the
    # night it was first *watched* (its recorded onset, or the preceding
    # reliable run for reports predating onset tracking) with the amber
    # triangle, and shade the release window the change entered in, so the trend
    # shows the ⚠️→🔴 progression and where upstream to look — not just the
    # endpoint.
    if verdict.severity is Severity.CONFIRMED:
        onset = _blame.onset_point(df, verdict) or _prev_point(df, verdict)
        if onset is not None:
            add_severity_markers(
                fig,
                pd.DataFrame({"x": [onset[0]], "y": [onset[1]], "name": [verdict.label]}),
                x_col="x", y_col="y", name_col="name",
                severity=Severity.WATCH.value, hover_y="%{y:.4g}",
            )
        if _blame.has_window(verdict):
            _blame.add_window_band(fig, df, verdict)
    add_severity_markers(
        fig,
        pd.DataFrame({"x": [verdict.run_date], "y": [verdict.value], "name": [verdict.label]}),
        x_col="x", y_col="y", name_col="name",
        severity=verdict.severity.value, hover_y="%{y:.4g}",
    )
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
