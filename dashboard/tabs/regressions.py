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

import requests
import streamlit as st

from k4bench.regression.models import (
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.regression.render import (
    _detector_badge,
    _group_title,
    _pretty_sample,
    from_json,
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
    attention_key,
    render_candidate_ranking,
)
from tabs._regression_trend import (
    render_metric_picker,
    render_metric_trend,
)
from tabs.stack_changes import _release, deep_link, packages_for_release
from ui_chrome import _drop_stale_selection, seed_query_param

_log = logging.getLogger(__name__)

def render(
    data_url: str, cache_dir: str, detector: str, platform: str, sample: str,
    stack: str,
) -> None:
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
            f"Historical view · release **{_release(stack)}** · report night "
            f"**{night}**"
        )
    report = reports[night]

    # Package/PR attribution belongs to the release, not to whichever repeat
    # measurement is open in the night picker. Freeze it at the release's
    # first report that produced a confirmed change window: later reruns may
    # advance an individual metric from WATCH to CONFIRMED, but no packages
    # changed between measurements of the same software. The chosen night's
    # banner and verdict details remain an honest historical snapshot.
    attribution = _release_attribution(
        reports, detector, platform, sample, stack,
    )
    attribution_night, attribution_group = (
        attribution if attribution is not None else (None, None)
    )

    # The blame sidecar is best-effort and absent on most nights (only a
    # confirmed, attributable regression produces one) — a missing blame.json
    # is normal, not an error. Read the canonical attribution night's sidecar,
    # not the selected rerun's, so ranked candidates stay stable too.
    raw_blame = (
        _cached_fetch_blame(data_url, attribution_night)
        if attribution_night is not None else None
    )
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
        attribution_group=attribution_group,
        attribution_night=attribution_night,
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


def _release_attribution(
    reports: dict[str, NightlyReport], detector: str, platform: str,
    sample: str, stack: str,
) -> tuple[str, RunGroupReport] | None:
    """The release's canonical ``(report night, group)`` for attribution.

    Reports remain chronological evidence snapshots, so a repeat measurement
    can turn a metric from WATCH into CONFIRMED without any software change.
    Package and PR attribution must not acquire a new comparison window merely
    because that evidence arrived later. Use the earliest report of *stack*
    containing a recorded confirmed window and reuse it for every rerun shown
    in this tab. If the first measurement was only WATCH, the first later
    confirmation becomes canonical and is also used when the earlier snapshot
    is selected.
    """
    for night in sorted(reports):
        group = _night_group(reports[night], detector, platform, sample)
        if group is None or group.k4h_release != stack:
            continue
        if any(_blame.has_window(v) for v in group.verdicts):
            return night, group
    return None


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
             "attention-worthy night; pick another to see that night's verdicts. "
             "Package and PR attribution stays fixed across the release's "
             "reruns.",
    )
    if night is None:  # segmented_control lets the active pill be deselected
        night = default_night
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
    """The run group's compact verdict counts."""
    n_fail = len(group.failures) + len(group.job_failures)
    n_ok = sum(1 for v in group.verdicts if v.severity is Severity.OK)
    with st.container(border=True):
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
                 "in neither column.",
        )


def _window_changes(data_url: str, verdict: MetricVerdict) -> list | None:
    """Changed packages across a confirmed regression's bounded blame window,
    or ``None`` when either release's provenance is missing (aged off CVMFS, or
    a release benchmarked before capture)."""
    base = packages_for_release(data_url, verdict.platform, verdict.last_accepted_run_date)
    head = packages_for_release(data_url, verdict.platform, verdict.onset_run_date)
    if not base or not head:
        return None
    return diff_packages(base, head)


def _render_blame_card(data_url: str, v: MetricVerdict, blame: BlameReport | None) -> None:
    """One window's forward attribution: the upstream packages that moved in the
    blame window, each linking to its commit range, plus the ranked candidate
    PRs from the blame sidecar when present — so the reader reaches the likely
    pull request without leaving the row or loading the trend."""
    kind = _blame.classify(v)
    with st.container(border=True):
        if kind is _blame.WindowKind.SAME_STACK:
            st.caption(
                f"Change detected within release **{v.onset_run_date}** · no tracked "
                "Key4hep package changed. Check benchmark code/config, inputs, "
                "runner environment, or noise."
            )
            return
        onset = v.onset_run_date
        baseline = v.last_accepted_run_date if kind is _blame.WindowKind.BOUNDED else None
        span = f"**{baseline} → {onset}**" if baseline else f"up to **{onset}**"
        st.caption(f"Change window: {span} · first detected **{onset}**")
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
        render_candidate_ranking(v, blame, show_empty=True)
        st.link_button(
            "🔍 Open in Stack Changes →",
            deep_link(detector=v.detector, platform=v.platform, sample=v.sample,
                      head_release=onset, base_release=baseline),
        )


def _render_blame_cards(
    group: RunGroupReport, data_url: str, blame: BlameReport | None,
    attribution_night: str,
) -> None:
    """Release-level forward attribution, frozen at *attribution_night*.

    One card is rendered per distinct window already present when the release
    first produced a confirmed blame window. Later repeat measurements use the
    same representatives and sidecar, so unchanged software cannot appear to
    acquire another package-change story merely because an additional metric
    reached its second strike.
    """
    by_window: dict[tuple, MetricVerdict] = {}
    for v in group.verdicts:
        if _blame.has_window(v):
            by_window.setdefault((v.last_accepted_run_date, v.onset_run_date), v)
    if not by_window:
        return
    st.markdown("###### Upstream changes in the blame window")
    st.caption(
        f"Shared across release reruns · first confirmed report night: "
        f"**{attribution_night}**"
    )
    for verdict in by_window.values():
        _render_blame_card(data_url, verdict, blame)


def _render_group(
    group: RunGroupReport, data_url: str, cache_dir: str, *,
    blame: BlameReport | None,
    attribution_group: RunGroupReport | None,
    attribution_night: str | None,
    key: str, scope: tuple[str, str],
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
    n_unknown = sum(
        1 for v in group.verdicts if v.severity is Severity.UNKNOWN
    )
    if n_unknown:
        st.caption(f"❔ {n_unknown} metric(s) not judged — insufficient history.")

    # The trend comes first — it's the first question a flagged night raises
    # ("what does this look like?") — the upstream-changes cards follow with
    # the "why", once there's a window to explain.
    drillable: list[MetricVerdict] = sorted(
        (v for v in flagged if v.baseline_median is not None), key=attention_key
    )
    if drillable:
        st.markdown("###### Flagged metric trend")
        # A report night (and a stack) supplies a different option model.
        # Reusing one Streamlit widget key and only clearing session state
        # updates the selected verdict/plot but can leave the browser's
        # displayed option text from the previous model. Give each scope a
        # distinct widget identity so label and value always move together.
        drill_key = f"regr_drill_{key}_{scope[0]}_{scope[1]}"
        selected = render_metric_picker(
            drillable,
            key=drill_key,
            help="Recent history with the baseline band this verdict was "
                 "judged against. Opens on the most severe flag — pick "
                 "another, or “—” to hide the chart. Downloads data on "
                 "first use.",
        )
        if selected is not None:
            render_metric_trend(
                selected, data_url, cache_dir,
                list_run_dates=_cached_list_run_dates,
                fetch_runs_windowed=_cached_fetch_runs_windowed,
                widget_namespace="regr",
            )
    if attribution_group is not None and attribution_night is not None:
        _render_blame_cards(
            attribution_group, data_url, blame, attribution_night,
        )
