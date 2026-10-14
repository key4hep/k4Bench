"""The Overview tab's data, shaped from the nightly reports — no rendering.

Everything here turns the precomputed ``reports/sim/{date}/report.json`` files
into the frames, rosters and styles the tab's views draw (:func:`overview_inputs`),
and is the tab's unit-test surface. The figures and the render flow live in
:mod:`tabs.detectors_overview`, the Regression Status view in
:mod:`tabs._regression_status`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlencode

import pandas as pd

from k4bench.labels import BASELINE_LABEL
from k4bench.regression.models import NightlyReport, RunGroupReport, Severity
from k4bench.regression.render import detector_badge
from tabs._regression_flags import (
    SEVERITY_RANK,
    attention_key,
    failed_config_labels,
    pretty_metric,
)
from ui_utils import DASHES, PALETTE_NAMES, PALETTES, SYMBOLS, auto_palette_index

#: The one config compared across detectors — the unpatched full-detector run
#: every sweep starts with (``baseline``, see ``k4bench.labels``).
#: Variant configs measure *within*-detector impact and live in the Config
#: Impact tab.
_BASELINE_LABEL = BASELINE_LABEL

#: The two panel families, each with its selectable equivalents (first entry
#: is the default). All are lower-is-better.
TIME_METRICS = ["mean_time_s", "median_time_s", "trimmed_mean_time_s",
                "wall_time_s", "user_cpu_s"]
MEMORY_METRICS = ["mean_rss_anon_mb", "peak_vmem_mb", "mean_rss_mb", "peak_rss_mb",
                  "mean_rss_file_mb"]
METRIC_ORDER: list[str] = [*TIME_METRICS, *MEMORY_METRICS]

#: Cap on report fetches when the sidebar provides no trend window (the store
#: has no release yet) — keeps the fallback from downloading years of nights.
_FALLBACK_NIGHTS = 30

_FRAME_COLUMNS = [
    "detector", "platform", "sample", "label", "metric", "value", "severity",
    "k4h_release", "run_date", "reliable",
]

#: Trailing version tokens of a detector directory name (``_o1_v03``, ``_v02``)
#: — everything before them is the detector *family* (see
#: :func:`detector_family`).
_VERSION_RE = re.compile(r"^(?P<family>.+?)(?P<variant>(?:_o\d+)?(?:_v\d+)?)$")

#: Matches the date embedded in a ``key4hep-YYYY-MM-DD`` nightly-release string.
_RELEASE_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _tag_date(k4h_release: str | None, fallback: str) -> str:
    """Date of the Key4hep nightly tag (``key4hep-2026-07-01`` → ``2026-07-01``).

    Falls back to *fallback* (the report/run night) when the release string
    carries no date, mirroring Run Trends' ``x_date`` (``k4h_release_date``
    with a run-date fallback)."""
    m = _RELEASE_DATE_RE.search(k4h_release or "")
    return m.group(0) if m else fallback


def _collapse_same_tag(frame: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Collapse same-nightly-tag reruns: within each *subset* group keep the
    newest CI run (largest ``run_night``), so a nightly benchmarked twice
    shows once — the same per-tag dedup Run Trends does. Requires a
    ``run_night`` column; it is dropped from the result."""
    return (
        frame.sort_values("run_night")
        .drop_duplicates(subset=subset, keep="last")
        .drop(columns="run_night")
    )


def report_metrics_frame(report: NightlyReport) -> pd.DataFrame:
    """One tidy row per usable metric verdict in *report*.

    Keeps every severity (OK/UNKNOWN included — their ``value`` is tonight's
    raw measurement; the severity column feeds the trend-flag markers), and
    drops what cannot be compared across detectors: region-level rows
    (``sub_detector`` set), verdicts outside :data:`METRIC_ORDER`
    (``returncode`` failures, ``cpu_efficiency``), and missing/non-finite
    values. ``reliable`` is the group's per-night host-reliability tri-state
    (``None`` on reports predating the field).

    ``run_date`` is the night the group's job actually ran, which is not always
    the night of the report carrying it: a batch that starts near midnight is
    dated a day earlier and still reported as tonight's. Carrying it is what
    lets the reliability filter address these rows at all — keyed on the report
    instead, they would never match the group's own ``reliable`` verdict.
    """
    failure_labels = {
        (g.detector, g.platform, g.sample, g.run_id):
        failed_config_labels(g.verdicts)
        for g in report.groups
    }
    rows = [
        {
            "detector":    g.detector,
            "platform":    g.platform,
            "sample":      g.sample,
            "label":       v.label,
            "metric":      v.metric,
            "value":       float(v.value),
            "severity":    (
                Severity.FAILURE.value
                if v.label in failure_labels[
                    (g.detector, g.platform, g.sample, g.run_id)
                ] else v.severity.value
            ),
            "k4h_release": g.k4h_release,
            "run_date":    g.run_date,
            "reliable":    g.reliable,
        }
        for g in report.groups
        for v in g.verdicts
        if v.sub_detector is None
        and v.metric in METRIC_ORDER
        and v.value is not None
        and math.isfinite(v.value)
    ]
    return pd.DataFrame(rows, columns=_FRAME_COLUMNS)


def report_reliability_frame(report: NightlyReport) -> pd.DataFrame:
    """One row per run group carrying its per-night host-reliability flag.

    Reliability lives on the *group* and on no metric verdict, so
    :func:`report_metrics_frame` cannot carry it however many rows a night has.
    An unreliable night is deliberately not *judged*, but its raw values are
    still recorded as ``UNKNOWN`` verdicts so the trend views can plot them (see
    ``report_builder._unjudged_verdicts``); a night that produced nothing at
    all — a hard-failed config, a job that never uploaded — has no verdict row
    to hang anything on. Extracting per-group is what lets the unreliable-run
    filter, and the tab's detector roster, see either case (columns:
    ``detector, platform, sample, run_date, k4h_release, missing_run,
    reliable``).

    ``run_date`` is the night the job actually ran — a group dated before the
    report night is normal for a batch that crossed midnight, and its
    reliability describes that run. ``missing_run`` is what separates those from
    the carried-forward placeholders for a run that never arrived (see
    :attr:`~k4bench.regression.models.RunGroupReport.missing_run`), which
    callers drop so the warning counts real runs.
    """
    rows = [
        {
            "detector":    g.detector,
            "platform":    g.platform,
            "sample":      g.sample,
            "run_date":    g.run_date,
            "k4h_release": g.k4h_release,
            "missing_run": g.missing_run,
            "reliable":    g.reliable,
        }
        for g in report.groups
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "detector", "platform", "sample", "run_date", "k4h_release",
            "missing_run", "reliable",
        ],
    )


def snapshot_runs(rows: pd.DataFrame) -> pd.DataFrame:
    """The :func:`history_rows` rows of the one run each detector's landscape
    point comes from: its newest nightly tag, and within that tag the newest
    run night.

    Picking the run *before* the metrics are pivoted into a point is what makes
    the landscape's one-run invariant hold. The newest *tag* alone is not
    enough: a tag benchmarked twice can have a rerun that re-measured only some
    metrics, and taking each metric's newest value independently — which is
    what :func:`collapse_history` does, correctly, for the trend lines — would
    then read a point's two coordinates off two different runs.
    """
    if rows.empty:
        return rows
    chosen = (
        rows.sort_values(["night", "run_night"])
        .drop_duplicates("detector", keep="last")
        .set_index("detector")[["night", "run_night"]]
    )
    return rows[[
        (night, run_night) == (chosen.at[d, "night"], chosen.at[d, "run_night"])
        for d, night, run_night in zip(
            rows["detector"], rows["night"], rows["run_night"]
        )
    ]]


def latest_snapshot(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Each detector's most recent measurement as one wide row (columns =
    metrics), plus ``{detector: nightly tag}`` naming the run behind each row.

    Built from :func:`history_rows` via :func:`snapshot_runs` — the *uncollapsed*
    rows, so the snapshot follows the *runs*, not whichever report night is open,
    and a detector that missed the newest night keeps its last measured point
    (labelled with its own tag) instead of dropping off the chart.
    """
    if rows.empty:
        return pd.DataFrame(), {}
    newest = snapshot_runs(rows)
    wide = newest.pivot_table(
        index="detector", columns="metric", values="value", aggfunc="first"
    )
    return wide, dict(zip(newest["detector"], newest["night"]))


def scatter_points(wide: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    """Detectors with both landscape coordinates, as a two-column frame."""
    pts = wide.reindex(columns=[x_metric, y_metric])
    return pts.dropna()


def nights_in_window(dates: list[str], window: tuple[date, date] | None) -> list[str]:
    """Filter report nights (``YYYY-MM-DD`` strings, any order) to the sidebar
    trend window, newest first. With no window, fall back to the latest
    :data:`_FALLBACK_NIGHTS` so an unset range never downloads years of reports."""
    if window is None:
        return sorted(dates, reverse=True)[:_FALLBACK_NIGHTS]
    start, end = window
    kept = [
        d for d in dates
        if pd.notna(ts := pd.to_datetime(d, errors="coerce"))
        and start <= ts.date() <= end
    ]
    return sorted(kept, reverse=True)


#: Columns every history frame carries, in order, after ``night``.
_HIST_COLS = ["detector", "metric", "value", "k4h_release", "severity", "reliable"]


def history_rows(
    night_frames: list[tuple[str, pd.DataFrame]],
    platform: str,
    sample: str,
    label: str,
) -> pd.DataFrame:
    """The scope's rows across nights, one row per **run**: columns
    ``night, run_night, detector, metric, value, k4h_release, severity,
    reliable``.

    ``night`` is the **Key4hep nightly tag** date (from ``k4h_release``), not
    the report/run date — the same x-axis as Run Trends. ``run_night`` is the
    report night the row was measured on; the two differ whenever a nightly is
    benchmarked more than once, and keeping both is what lets a caller drop one
    rerun of a tag without dropping the tag.

    Not plottable on its own — pass it through :func:`collapse_history` to get
    one point per tag, dropping unwanted runs in between, or through
    :func:`snapshot_runs` for the landscape's one run per detector. There is
    deliberately no helper that composes the two directly: every historical view
    here filters, so a one-call shortcut would only ever be the wrong one to
    reach for.
    """
    parts = []
    for report_night, frame in night_frames:
        sub = frame[
            (frame["platform"] == platform)
            & (frame["sample"] == sample)
            & (frame["label"] == label)
        ]
        if sub.empty:
            continue
        part = sub[_HIST_COLS].copy()
        # The run's own date, not the report's: a batch that starts near
        # midnight is dated a day earlier and still reported as tonight's, and
        # keying these rows on the report would put them out of reach of the
        # reliability filter, which knows that run by the date it ran. Falls
        # back to the report night only for a frame built before the column
        # existed.
        part["run_night"] = (
            sub["run_date"].fillna(report_night)
            if "run_date" in sub.columns else report_night
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["night", "run_night", *_HIST_COLS])
    rows = pd.concat(parts, ignore_index=True)
    rows["night"] = [
        _tag_date(rel, rn) for rel, rn in zip(rows["k4h_release"], rows["run_night"])
    ]
    return rows[["night", "run_night", *_HIST_COLS]].reset_index(drop=True)


def collapse_history(rows: pd.DataFrame) -> pd.DataFrame:
    """Reduce :func:`history_rows` to one point per (detector, metric, tag).

    A point here is a nightly **tag** — a release — so two CI runs that
    benchmarked the same nightly (a rerun) collapse to one: the newest
    successful run wins for the plotted value, but ``severity`` keeps the
    *worst non-failure* verdict across the tag's runs. Failed measurements stay
    outside this reduction and are plotted separately at their own values, so
    they cannot replace a healthy rerun or lend their severity to its value.
    Nights of one tag share a baseline yet can still differ (WATCH
    before the confirmation, a marginal OK night, or a report predating the
    release-grouped engine), and the flag must not be masked by the quieter
    night. Pairing a value from one reduction with a severity from another is
    the same thing the engine does when it summarises a release
    (:class:`k4bench.regression.models.ReleasePoint`); ``tabs.trends``'
    ``_tag_severity`` is this rule for Run Trends.

    The reduction only ever sees the runs it is handed, which is what keeps the
    reliability filter honest: filter *rows* first and an excluded run's verdict
    cannot reach the tag, so a release whose only flagged run was dropped stops
    flagging.
    """
    if rows.empty:
        return pd.DataFrame(columns=["night", *_HIST_COLS])
    rows = rows[rows["severity"] != Severity.FAILURE.value]
    if rows.empty:
        return pd.DataFrame(columns=["night", *_HIST_COLS])
    worst = (
        rows.assign(_rank=rows["severity"].map(lambda s: SEVERITY_RANK.get(s, 0)))
        .sort_values("_rank")
        .drop_duplicates(["detector", "metric", "night"], keep="last")
        .set_index(["detector", "metric", "night"])["severity"]
    )
    hist = _collapse_same_tag(rows, ["detector", "metric", "night"])
    hist["severity"] = [
        worst.get((d, m, n), s)
        for d, m, n, s in zip(hist["detector"], hist["metric"], hist["night"], hist["severity"])
    ]
    return hist[["night", *_HIST_COLS]].reset_index(drop=True)


def reliability_history(
    night_frames: list[tuple[str, pd.DataFrame]],
    platform: str,
    sample: str,
) -> pd.DataFrame:
    """Per-**run** host-reliability for the scope across nights: columns
    ``night, run_night, detector, reliable``.

    Takes :func:`report_reliability_frame` outputs (one per report night) and
    keeps every group that describes a real run, dropping only the
    carried-forward placeholders for a run that never arrived
    (``missing_run``) so the warning counts runs rather than absences. ``night``
    is the **Key4hep nightly tag** date, as in :func:`history_rows`, and
    ``run_night`` the date the run itself carries — one row per run, *not*
    collapsed to the tag.

    Keying on the run's own date rather than the report's is what keeps this
    frame addressing the same runs :func:`history_rows` does. A batch that
    starts near midnight is dated a day earlier and still reported as tonight's,
    so a report-night key would silently exclude those runs here while their
    values stayed in the history — leaving a contended run plotted with the
    exclusion on and nothing said about it. A run that appears in two reports
    (its own night's and the next one's) lands on one key and dedupes.

    Reliability is a property of the run, so same-tag reruns must stay apart
    here: collapsing them would make one contended rerun condemn its tag's
    reliable sibling, and would leave the exclusion unable to say *which*
    measurement it is dropping.
    """
    parts = []
    for _report_night, frame in night_frames:
        sub = frame[
            (frame["platform"] == platform)
            & (frame["sample"] == sample)
            & ~frame["missing_run"].fillna(False).astype(bool)
        ]
        if sub.empty:
            continue
        part = sub[["detector", "k4h_release", "reliable"]].copy()
        part["run_night"] = sub["run_date"]
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["night", "run_night", "detector", "reliable"])
    rel = pd.concat(parts, ignore_index=True)
    rel["night"] = [
        _tag_date(k, rn) for k, rn in zip(rel["k4h_release"], rel["run_night"])
    ]
    return (
        rel[["night", "run_night", "detector", "reliable"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def drop_unreliable_runs(
    rows: pd.DataFrame, pairs: set[tuple[str, str]]
) -> pd.DataFrame:
    """*rows* (from :func:`history_rows`) without the ``(run_night, detector)``
    runs in *pairs*.

    Addresses runs rather than tags so a rerun of a tag survives its contended
    sibling: the point stays on the chart, measured by the run that passed.
    """
    if not pairs or rows.empty:
        return rows
    return rows[[
        (rn, d) not in pairs
        for rn, d in zip(rows["run_night"], rows["detector"])
    ]]


def relative_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Rescale each (detector, metric) series to its first plotted night
    = 100 %, so drift is comparable across detectors whose absolute values
    differ by more than a decade. A zero first value yields NaN (blank line)
    rather than infinities."""
    if hist.empty:
        return hist
    out = hist.copy()
    out["_dt"] = pd.to_datetime(out["night"])
    out = out.sort_values("_dt")
    healthy = (
        out[out["severity"] != Severity.FAILURE.value]
        if "severity" in out.columns else out
    )
    base = (
        healthy.groupby(["detector", "metric"])["value"].first()
        if not healthy.empty else pd.Series(dtype=float)
    )
    keys = pd.MultiIndex.from_frame(out[["detector", "metric"]])
    baseline = pd.Series(base.reindex(keys).to_numpy(), index=out.index, dtype=float)
    out["value"] = out["value"] / baseline.where(baseline != 0) * 100.0
    return out.drop(columns="_dt")


def detector_status_rows(
    groups: list[RunGroupReport], platform: str, sample: str, night: str
) -> list[dict]:
    """One latest-night status row per detector in the sidebar scope, worst
    first: the detector's badge, its flag counts, its worst flagged metric
    (severity then |Δ| — the Regressions ledger's ordering) and a Regressions
    deep link scoped to the group's triple.

    The deep link pins the release (``stack``) and the exact report *night* it
    describes, so it lands on this row's report even after the release is
    re-benchmarked — the same pinning the nightly email links use. ``stack`` is
    omitted for a stale group with no release.
    """
    rows = []
    for g in groups:
        flagged = sorted(
            (v for v in g.verdicts if v.severity in (Severity.WATCH, Severity.CONFIRMED)),
            key=attention_key,
        )
        worst = flagged[0] if flagged else None
        rows.append({
            "": detector_badge([g]),
            "Detector": g.detector,
            "🔴": len(g.regressions),
            "⚠️": len(g.watches),
            "❌": len(g.failures) + len(g.job_failures),
            "Worst flag": f"{pretty_metric(worst)} · {worst.label}" if worst else "—",
            "Δ": (
                None if worst is None or worst.pct_change is None
                else worst.pct_change * 100
            ),
            "Inspect": "?" + urlencode({
                "tab": "Regressions", "detector": g.detector,
                "platform": platform, "sample": sample,
                **({"stack": g.k4h_release} if g.k4h_release else {}),
                "report": night,
            }),
        })
    rows.sort(key=lambda r: (-r["❌"], -r["🔴"], -r["⚠️"], r["Detector"]))
    return rows


def detector_family(detector: str) -> tuple[str, str]:
    """Split a detector directory name into (family, version variant):
    ``ALLEGRO_o1_v03`` → (``ALLEGRO``, ``o1_v03``), ``ILD_FCCee_v01`` →
    (``ILD_FCCee``, ``v01``), ``SiD`` → (``SiD``, ``""``)."""
    m = _VERSION_RE.match(detector)
    if not m:
        return detector, ""
    return m.group("family"), m.group("variant").lstrip("_")


def detector_styles(
    detectors: list[str], palette: list[str]
) -> dict[str, tuple[str, str, str]]:
    """``{detector: (colour, dash, symbol)}`` — colour follows the detector
    *family* (assigned alphabetically over the palette, stable regardless of
    which detectors have data tonight), while versions within a family cycle
    the dash pattern and marker symbol. Versions of one experiment therefore
    read as variations of the same series instead of unrelated colours."""
    families = sorted({detector_family(d)[0] for d in detectors})
    family_color = {f: palette[i % len(palette)] for i, f in enumerate(families)}
    by_family: dict[str, list[str]] = {}
    for detector in sorted(detectors):
        by_family.setdefault(detector_family(detector)[0], []).append(detector)
    styles: dict[str, tuple[str, str, str]] = {}
    for family, members in by_family.items():
        for idx, detector in enumerate(members):
            styles[detector] = (
                family_color[family],
                DASHES[idx % len(DASHES)],
                SYMBOLS[idx % len(SYMBOLS)],
            )
    return styles


def stale_run_nights(
    report: NightlyReport, report_night: str, platform: str, sample: str
) -> list[str]:
    """The report nights holding the last real run of each scoped detector that
    missed *report_night*, oldest first.

    A detector that misses a night is not dropped from that night's report: it
    is carried as a *stale* group, keeping its own older ``run_date`` but
    stripped of every verdict, leaving only the missing-run failure (see
    ``k4bench.regression.report_builder._finalize_report``). Its last actual
    measurement therefore lives in the report of *that* night — which the
    sidebar's trend window need not cover. Fetching those nights as well is
    what lets the landscape show such a detector at its own most recent run
    rather than at whatever older night the window happens to end on.

    Selected on the missing-run marker, not on the dates: a group dated before
    the report night is just as likely to be a job that started before midnight
    and ran for this very report, and that one's measurements are already here
    — chasing its date would only refetch a report we have the data from.
    """
    return sorted({
        g.run_date for g in report.groups
        if g.platform == platform and g.sample == sample
        and g.run_date and g.run_date != report_night and g.missing_run
    })


def last_run_nights(rel_hist: pd.DataFrame) -> dict[str, str]:
    """``{detector: nightly tag of its most recent run}`` from the per-run
    reliability history.

    Deliberately blind to the reliability flag and to the exclusion toggle: this
    answers "when did it last run", which the trend caption states as fact. The
    landscape's ``as_of`` answers the different question "which run is this
    point", and follows the newest *plottable* run — so an unreliable newest run
    sends it back to an earlier night, and reusing it here would date a detector
    to before a run it demonstrably made.
    """
    if rel_hist.empty:
        return {}
    return (
        rel_hist.sort_values(["night", "run_night"])
        .drop_duplicates("detector", keep="last")
        .set_index("detector")["night"]
        .to_dict()
    )


@dataclass(frozen=True)
class OverviewInputs:
    """Everything the Overview tab's views draw from, built once per full rerun
    from the fetched reports (:func:`overview_inputs`) and replayed unchanged on
    every fragment rerun.

    The row frames are kept **uncollapsed**, so the reliability filter can drop
    *runs* before the reduction — which differs between the two figure views:
    per metric for a trend line (:func:`collapse_history`), per run for a
    landscape point (:func:`snapshot_runs`).
    """

    platform: str
    sample: str
    latest_night: str
    #: The sidebar window's rows (:func:`history_rows`) — the trend chart's input.
    hist_rows: pd.DataFrame
    #: The rows of every night fetched for the landscape, so each detector's
    #: newest run is found whether or not it falls inside the window.
    snap_rows: pd.DataFrame
    #: Per-run reliability across every night the snapshot can read
    #: (:func:`reliability_history`).
    rel_hist: pd.DataFrame
    #: The window's report nights, newest first.
    window_nights: list[str]
    #: ``(night, metric frame)`` for the window plus the newest report — the
    #: Regression Status view's input.
    status_frames: list[tuple[str, pd.DataFrame]]
    #: ``night -> run groups`` for the scope, for the Regression Status picker.
    scoped_groups: dict[str, list[RunGroupReport]]
    #: The nights the Nightly Report view offers, unfiltered by the scope.
    mail_reports: dict[str, NightlyReport]
    #: Every detector the scope knows about, and its colour/dash/symbol.
    detectors: list[str]
    styles: dict[str, tuple[str, str, str]]
    #: Who ran in any fetched night (the landscape's reach), who ran inside the
    #: window (the trend chart's), and who this scope never benchmarked.
    scoped_detectors: list[str]
    window_roster: list[str]
    excluded: list[str]
    #: ``{detector: nightly tag of its most recent run}`` (:func:`last_run_nights`).
    last_run: dict[str, str]
    #: True when the three scoped views have nothing to draw at all.
    scope_empty: bool


def overview_inputs(
    reports: dict[str, NightlyReport],
    *,
    platform: str,
    sample: str,
    latest_night: str,
    hist_nights: list[str],
    fetch_nights: tuple[str, ...],
    snap_nights: tuple[str, ...],
) -> OverviewInputs:
    """The tab's view inputs from the fetched *reports*.

    *hist_nights* are the sidebar window's nights, *fetch_nights* those plus the
    newest report, and *snap_nights* every fetched night the landscape can read
    — *fetch_nights* plus the stragglers fetched for detectors whose last run
    predates the newest report (:func:`stale_run_nights`).
    """
    frames = {n: report_metrics_frame(rep) for n, rep in reports.items()}
    # Per-group reliability, kept separately: the flag lives on the group and
    # on no verdict, so ``frames`` cannot express it — and a run that produced
    # nothing to record has no verdict row to hang it on either (see
    # report_reliability_frame).
    rel_frames = {n: report_reliability_frame(rep) for n, rep in reports.items()}

    window_frames = [(n, frames[n]) for n in hist_nights if n in frames]
    hist_rows = history_rows(window_frames, platform, sample, _BASELINE_LABEL)
    hist = collapse_history(hist_rows)
    snap_rows = history_rows(
        [(n, frames[n]) for n in snap_nights], platform, sample, _BASELINE_LABEL,
    )

    # Reliability is read from the report *groups*, not the metric frame, which
    # carries the flag on no row of its own — and none at all for a night that
    # produced nothing to record. It spans every night the snapshot can read,
    # not just the window: the landscape plots the newest run it can find, which
    # is outside the window whenever the sidebar range ends before the latest
    # report or a straggler. Scoping this to the window instead would leave such
    # a run plotted with no warning beside it and no toggle to drop it — the one
    # state the filter exists to prevent. Spanning them all is also what lets
    # the Regression Status view share this one frame: it plots the selected
    # night, which is just as free to sit outside the window.
    rel_hist = reliability_history(
        [(n, rel_frames[n]) for n in snap_nights], platform, sample,
    )

    # Detector rosters are read off the report *groups* (via rel_hist), never
    # the metric frame. A config that hard fails is judged on its return code
    # and produces no metric verdict at all, so a metric-derived roster loses
    # exactly the detector most worth naming — and, when that detector has
    # metrics under some other sample, files it under "not benchmarked with
    # this sample/platform", which is the one thing it demonstrably is not.
    # reliability_history already reads the groups and drops the
    # carried-forward placeholders for a run that never arrived, so its
    # detector column *is* the roster of runs that really happened.
    #
    # ``window_roster`` is the narrower question the trend chart asks — who ran
    # inside the plotted window — while ``scoped_detectors`` spans every fetched
    # night, which is the range the landscape's newest-run fallback reaches.
    #
    # ``excluded`` asks a third question — who this sample/platform was never
    # benchmarked with — and so counts *membership*, not runs: a detector that
    # missed the night is carried as a placeholder group, which is proof the
    # scope covers it. Measuring membership by runs instead would file it under
    # "not benchmarked" whenever its last real run is out of reach, which is
    # exactly the case a straggler fetch is there to rescue and can fail to.
    #
    # All three are named before any reliability filter runs, so a detector
    # dropped as unreliable is never reported as "not benchmarked" instead.
    scoped_wide, _ = latest_snapshot(snap_rows)
    scoped_detectors = sorted(set(rel_hist["detector"]))
    window_roster = sorted(set(
        reliability_history(
            [(n, rel_frames[n]) for n in hist_nights if n in rel_frames],
            platform, sample,
        )["detector"]
    ))
    scope_members = {
        d
        for n in snap_nights
        for d in rel_frames[n].loc[
            (rel_frames[n]["platform"] == platform)
            & (rel_frames[n]["sample"] == sample),
            "detector",
        ].unique()
    }
    excluded = sorted(
        {d for n in snap_nights for d in rel_frames[n]["detector"].unique()}
        - scope_members
    )

    # The window's nights plus the newest report — the Regression Status view's
    # input and its picker's options. Nights fetched only to complete the
    # landscape are deliberately left out: they are one detector's stragglers,
    # not part of the range the reader chose. A night with no group for the
    # scope is dropped rather than offered as a dead option; one with groups but
    # no plottable values is kept, since a night whose configs all failed has
    # report groups and an empty metric frame, and hiding the failures would be
    # the worst miss.
    scoped_groups = {}
    for n in fetch_nights:
        if n not in reports:
            continue
        groups = [
            g for g in reports[n].groups
            if g.platform == platform and g.sample == sample
        ]
        if groups:
            scoped_groups[n] = groups

    # Every detector the scope knows about, including one carried only as a
    # missing-run placeholder: now that ``excluded`` counts it as a member, this
    # is the only roster left that can name it, and a detector nothing accounts
    # for is the failure the captions exist to prevent. One colour per detector
    # family, dash/symbol per version — stable across every panel, the palette
    # sized to the number of families so colours stay distinct without cycling.
    detectors = sorted(
        set(scoped_detectors) | set(window_roster) | scope_members
        | set(hist["detector"].unique())
    )
    n_families = len({detector_family(d)[0] for d in detectors})
    palette = PALETTES[PALETTE_NAMES[auto_palette_index(n_families)]]

    return OverviewInputs(
        platform=platform,
        sample=sample,
        latest_night=latest_night,
        hist_rows=hist_rows,
        snap_rows=snap_rows,
        rel_hist=rel_hist,
        window_nights=[n for n, _ in window_frames],
        status_frames=[(n, frames[n]) for n in fetch_nights if n in frames],
        scoped_groups=scoped_groups,
        # The Nightly Report view's picker: the same nights, *unfiltered* by the
        # sidebar scope, since the mail is the whole night. Stragglers are left
        # out for the same reason as above.
        mail_reports={n: reports[n] for n in fetch_nights if n in reports},
        detectors=detectors,
        styles=detector_styles(detectors, palette),
        scoped_detectors=scoped_detectors,
        window_roster=window_roster,
        excluded=excluded,
        # Read off the unfiltered group history, never the landscape's
        # ``as_of`` (see last_run_nights).
        last_run=last_run_nights(rel_hist),
        # Reported inside the views: the three scoped ones have nothing to
        # draw, but the Nightly Report does.
        scope_empty=scoped_wide.empty and hist.empty and not scoped_groups,
    )
