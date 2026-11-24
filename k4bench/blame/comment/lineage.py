"""What a comment says about its own lineage: the association with each
scope across the published reports, the observation history, and the retained
snapshots the next version starts from.
"""

from __future__ import annotations

import math
from collections import Counter

from k4bench.blame.attribute import Attribution
from k4bench.blame.comment.digest import (
    Reproducers,
)
from k4bench.blame.comment.markdown import (
    REGRESSION,
    _capitalized,
    _cell,
    _count,
    _pct,
)
from k4bench.blame.comment.plan import (
    CommentPlan,
    RegressionRow,
    _row_identity,
)
from k4bench.blame.comment.state import (
    _MARKER_SUFFIX,
    _MAX_RETAINED_ROWS,
    _OMITTED_OBSERVATIONS_PREFIX,
    CommentObservation,
    RetainedRow,
    _iso_date,
    _observation_marker,
    _retained_sort_key,
)
from k4bench.blame.comment.table import (
    _SHOW_PLATFORM_COLUMN,
    _score_source,
)
from k4bench.labels import compact_sample, pretty_platform
from k4bench.regression.render import (
    nightly_report_href,
)

#: How many detector/platform/sample scopes the association table names before
#: it starts counting them instead. A night that regresses across the whole
#: suite would otherwise paste a table nobody reads into someone's pull request
#: — the same reason the regression table itself is capped.
_MAX_ASSOCIATION_SCOPES = 8


#: Metrics named in one scope's cell before the rest are counted.
_MAX_ASSOCIATION_METRICS = 4


def _association_summary(
    state: dict[tuple, tuple[float | None, str, str]],
    min_score: float,
    dashboard_url: str | None,
    *,
    noun: str = REGRESSION,
) -> str:
    """The PR's association with each scope it is implicated in, current and
    historical alike.

    Only scopes carrying at least one regression at or above *min_score* get a
    row. A scope the review scored *down* is not evidence about this pull
    request, and a table listing it beside the attributed ones invites the
    reader to weigh it as though it were — but it is still counted, in one line
    below, because the alert above states a union that has to reconcile.
    """
    grouped: dict[tuple, list[tuple]] = {}
    for identity, (score, source, night) in state.items():
        grouped.setdefault(identity[:3], []).append((identity, score, source, night))

    def attributed_in(entries: list[tuple]) -> int:
        return sum(
            score is not None and score >= min_score for _, score, _, _ in entries
        )

    ordered = sorted(
        grouped.items(), key=lambda item: (-attributed_in(item[1]), item[0])
    )
    shown = [item for item in ordered if attributed_in(item[1])]
    # Every scope scored below the threshold is the one case where the table
    # would otherwise be a header and nothing else. Fall back to naming them
    # rather than rendering an empty table.
    unattributed = [item for item in ordered if not attributed_in(item[1])]
    if not shown:
        shown, unattributed = ordered, []
    listed, over_cap = shown[:_MAX_ASSOCIATION_SCOPES], shown[_MAX_ASSOCIATION_SCOPES:]

    lines = [
        "",
        "**Association with this PR, by detector and sample**",
        "",
        "Counts cover distinct metric/configuration/region combinations across the "
        f"published reports. Attribution uses each {noun}'s latest published AI score.",
        "",
        f"| Detector | Sample | Metrics / configurations | Attribution ≥ {_pct(min_score)} "
        f"| Likelihood | Reports |",
        "|:---|:---|:---|---:|---:|:---|",
    ]
    for (detector, platform, sample), entries in listed:
        metrics = sorted({identity[4] for identity, _, _, _ in entries})
        metric_text = ", ".join(
            _cell(metric) for metric in metrics[:_MAX_ASSOCIATION_METRICS]
        )
        if len(metrics) > _MAX_ASSOCIATION_METRICS:
            metric_text += f", +{len(metrics) - _MAX_ASSOCIATION_METRICS} more"
        configs = len({identity[3] for identity, _, _, _ in entries})
        metric_text += f"<br>{_count(configs, 'configuration')}"
        scores = [score for _, score, _, _ in entries if score is not None]
        likelihood = (
            _pct(scores[0]) if scores and min(scores) == max(scores)
            else f"{_pct(min(scores))}–{_pct(max(scores))}" if scores else "not scored"
        )
        nights = sorted({night for _, _, _, night in entries if _iso_date(night)})
        report_labels = []
        for night in ([nights[0], nights[-1]] if len(nights) > 1 else nights):
            href = nightly_report_href(dashboard_url, night)
            report_labels.append(f"[{night}]({href})" if href else night)
        report_text = " to ".join(report_labels) or "unknown"
        scope = _cell(detector)
        if _SHOW_PLATFORM_COLUMN or len({key[1] for key in grouped}) > 1:
            scope += f"<br>{_cell(pretty_platform(platform))}"
        lines.append(
            f"| {scope} | {_cell(compact_sample(sample))} | {metric_text} | "
            f"**{attributed_in(entries)} / {len(entries)}** | {likelihood} "
            f"| {report_text} |"
        )
    trailing = [
        note for note in (
            _capped_scopes_note(over_cap, noun=noun),
            _unattributed_note(unattributed, min_score, noun=noun),
        ) if note
    ]
    for note in trailing:
        lines += ["", note]
    lines += [
        "",
        "Report dates open the **full nightly report**, including other change windows.",
    ]
    return "\n".join(lines)


def _capped_scopes_note(over_cap: list[tuple], *, noun: str = REGRESSION) -> str | None:
    """The scopes the table had no room for, counted with their rows.

    Deliberately says nothing about attribution: the rows above it are the
    attributed ones in the ordinary case, but on the fallback path — where no
    scope reached the threshold at all — they are not, and one line cannot
    claim both."""
    if not over_cap:
        return None
    rows = sum(len(entries) for _key, entries in over_cap)
    return (
        f"{_count(len(over_cap), 'further scope')} "
        f"({_count(rows, noun)}) are not listed above."
    )


def _unattributed_note(
    unattributed: list[tuple], min_score: float, *, noun: str = REGRESSION,
) -> str | None:
    """The scopes the review scored down, counted rather than tabulated.

    Kept because the alert above states the union across every scope, and a
    table that silently dropped these would leave that number unexplained."""
    if not unattributed:
        return None
    rows = sum(len(entries) for _key, entries in unattributed)
    scores = [
        score for _key, entries in unattributed
        for _identity, score, _source, _night in entries if score is not None
    ]
    standing = (
        f"highest {_pct(max(scores))}" if scores else "none of them scored"
    )
    return (
        f"A further {_count(rows, noun)} in "
        f"{_count(len(unattributed), 'scope')} stayed below {_pct(min_score)} "
        f"and are not attributed to this PR ({standing})."
    )


def _observation_history(
    observations: list[CommentObservation], *, omitted: int, noun: str = REGRESSION,
) -> str:
    count = _count(len(observations), "material update")
    summary = count + (
        f" · {_count(omitted, 'earlier update')} omitted" if omitted else ""
    )
    omitted_marker = (
        f"{_OMITTED_OBSERVATIONS_PREFIX}{omitted}{_MARKER_SUFFIX}"
        if omitted else None
    )
    lines = [
        "<details>",
        f"<summary><b>🕘 Observation history</b> — {summary}</summary>",
        "",
        *([omitted_marker] if omitted_marker else []),
        *(_observation_marker(item) for item in observations),
        "",
        f"| Report | Change window | {_capitalized(noun)}s | Scopes | Directions |",
        "|:---|:---|---:|---:|:---|",
    ]
    for item in observations:
        report = f"[{item.report_night}]({item.url})" if item.url else item.report_night
        base = item.base_release or "…"
        lines.append(
            f"| {report} | `{base}` → `{item.onset_release}` | "
            f"{item.regressions} | {item.scopes} | "
            f"{_direction_text(item.up, item.down, item.none)} |"
        )
    if omitted:
        lines += [
            "",
            f"_{_count(omitted, 'earlier material update')} omitted to keep this "
            "comment within GitHub's size limit._",
        ]
    return "\n".join([*lines, "", "</details>"])


def _retained_state(
    plan: CommentPlan,
    attribution: Attribution | None,
    historical: list[RetainedRow],
    reproduce: "Reproducers | None" = None,
) -> list[RetainedRow]:
    """The bounded snapshot list the next material version starts from: every
    current row at tonight's evidence, then the still-unconfirmed history,
    strongest first and cut at the cap. A report with no night to date the
    snapshots by contributes none — an undatable confirmation could never be
    honestly labelled "last reported"."""
    state = list(historical)
    if plan.report_night:
        urls = reproduce.urls if reproduce is not None else {}
        state += [
            _snapshot(row, attribution, plan, urls.get(_row_identity(row), ""))
            for row in plan.rows
        ]
    state.sort(key=_retained_sort_key)
    return state[:_MAX_RETAINED_ROWS]


def _snapshot(
    row: RegressionRow,
    attribution: Attribution | None,
    plan: CommentPlan,
    recipe: str = "",
) -> RetainedRow:
    """One current row frozen at tonight's evidence — the form it would be
    resurfaced in if it stopped being confirmed tomorrow."""
    v = row.verdict
    likelihood, source = _score_source(row, attribution)
    pct = v.pct_change
    return RetainedRow(
        detector=v.detector,
        platform=v.platform,
        sample=v.sample,
        label=v.label,
        metric=v.metric,
        sub_detector=v.sub_detector or "",
        direction=str(getattr(v.direction, "value", v.direction)),
        pct=pct if pct is not None and math.isfinite(pct) else None,
        onset=v.onset_run_date or "",
        onset_run=v.onset_run_id or "",
        base=v.last_accepted_run_date or "",
        recipe=recipe,
        base_release=plan.base_release,
        base_run=v.last_accepted_run_id or "",
        onset_release=plan.onset_release,
        stack=row.stack,
        last_reported=plan.report_night,
        likelihood=likelihood,
        source=source,
        state=row.scope_state,
    )


def _direction_counts(rows: list[RegressionRow]) -> Counter[str]:
    return Counter(
        str(getattr(row.verdict.direction, "value", row.verdict.direction))
        for row in rows
    )


def _direction_text(up: int, down: int, none: int = 0) -> str:
    parts = [f"{up} UP", f"{down} DOWN"]
    if none:
        parts.append(f"{none} NONE")
    return " · ".join(parts)


def _observation_for(
    plan: CommentPlan,
    rows: list[RegressionRow],
    dashboard_url: str | None,
) -> CommentObservation | None:
    if not plan.report_night:
        return None
    directions = _direction_counts(rows)
    return CommentObservation(
        report_night=plan.report_night,
        base_release=plan.base_release,
        onset_release=plan.onset_release,
        regressions=len(rows),
        scopes=len(plan.scopes),
        up=directions["UP"],
        down=directions["DOWN"],
        none=directions["NONE"],
        url=_report_href(plan, rows, dashboard_url),
    )


def _report_href(
    plan: CommentPlan,
    rows: list[RegressionRow],
    dashboard_url: str | None,
) -> str | None:
    """An archived dashboard view pinned to this observation's report night."""
    if not rows or not plan.report_night:
        return None
    return nightly_report_href(dashboard_url, plan.report_night)
