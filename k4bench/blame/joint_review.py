"""Review every pull request selected in one change window together.

:mod:`k4bench.blame.attribute` reviews one pull request against its window, and
the comment builder used to run it once per selected pull request. Those
reviews are independent model calls over the same evidence, and they can
disagree with each other: on 2026-11-21 the review of ``k4geo#683`` explained
the ILD slowdowns with ``DD4hep#1715``, while the review of ``DD4hep#1715`` —
the same night, the same rows — scored it at 4-6% on a misreading of its diff
and withdrew its comment. One reading of a window cannot both blame and clear
the same change.

So when several pull requests are selected in one window, they are reviewed in
**one** call: the window's evidence once, each pull request under review with
its own first-pass prior, geometry reach, description and diff, and the rest of
the field as competitors. The model scores every scope for every pull request
under review and says, for a scope where more than one is likely, whether they
are **alternative** explanations of the same movement (only one can be the
cause) or **independent** contributions (each caused part of it). Several pull
requests can cause one regression, so neither answer is assumed; the likelihoods
are not forced to add up.

The reply is split back into one :class:`~k4bench.blame.attribute.Attribution`
per pull request, through the same only-echo parsing the single review uses, and
each carries the alternatives the model named for its rows
(:attr:`~k4bench.blame.attribute.Attribution.alternatives`). A window with one
selected pull request, a window without sweeps, and every failure fall back to
the single review — the joint call only ever replaces several independent
calls, never a review that would otherwise not have happened.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence

from k4bench.blame.attribute import (
    _MAX_DIFF_CHARS,
    _MAX_FILES_LISTED,
    _MAX_SUBJECT_BODY_CHARS,
    _MAX_SUMMARY_CHARS,
    _OUTPUT_TOKENS_BASE,
    _OUTPUT_TOKENS_PER_ROW,
    _SUBJECT_DIFF_FLOOR,
    MAX_COMPETITORS,
    Attribution,
    AttributionRequest,
    CompetingPR,
    _attributed_facts,
    _competitor_lines,
    _described,
    _fact_scope,
    _history_lines,
    _id_order,
    _package_lines,
    _parse_assessment,
    _parse_likelihoods,
    _parse_scopes,
    _prior_summary,
    _scope_rows,
    _summarised_facts,
    _sweep_detail_lines,
    _system_prompt,
    _without_row_ids,
    competitor_order,
    scope_ids,
)
from k4bench.blame.evidence import ExternalChange
from k4bench.blame.llm import MAX_OUTPUT_TOKENS, extract_json, one_line
from k4bench.blame.prompt import (
    allocate_diff_budget,
    body_block,
    detector_switch_lines,
    diff_block,
    format_files,
    historical_lines,
    log_prompt_size,
    platform_switch_lines,
    window_phrase,
)
from k4bench.blame.summary import (
    EVIDENCE_HEADER,
    external_change_lines,
    other_scope_lines,
    scope_evidence_lines,
    touch_lines,
)

_log = logging.getLogger(__name__)

#: The relations a scope's verdicts may stand in.
ONE_CAUSE = "one_cause"
SEVERAL_CAUSES = "several_causes"


def joinable(requests: Sequence[AttributionRequest]) -> bool:
    """Whether *requests* can be reviewed in one call: two or more, one suite
    and window, one set of offered rows, and the sweeps the joint prompt reads
    each scope from (without them the rows carry one pull request's prior
    inline, which a joint prompt cannot)."""
    if len(requests) < 2:
        return False
    first = requests[0]
    rows = {fact.id for fact in _attributed_facts(first)}
    return bool(first.sweeps) and all(
        (r.suite, r.base_release, r.onset_release) ==
        (first.suite, first.base_release, first.onset_release)
        and {fact.id for fact in _attributed_facts(r)} == rows
        for r in requests
    )


def labels(requests: Sequence[AttributionRequest]) -> dict[str, str]:
    """``slug -> "P1"…`` in the order *requests* are given."""
    return {request.slug: f"P{index}" for index, request in enumerate(requests, start=1)}


# ── Prompt ────────────────────────────────────────────────────────────────────

def _joint_system_prompt(requests: Sequence[AttributionRequest]) -> str:
    """The single review's system prompt, re-aimed at several pull requests:
    the same evidence rules, the same score bands, the same refusal to invent
    rows — and the question asked of each pull request under review."""
    base = _system_prompt(_with_union(requests))
    return (
        base.replace(
            "You review whether one merged pull request caused",
            "You review whether each of several merged pull requests caused",
        )
        + " Several pull requests are under review in this window, and you judge "
        "them together: one reading of the evidence for all of them, so the "
        "same movement is never explained by one pull request in one answer and "
        "cleared for it in another. Score every scope for every pull request "
        "under review, each 0-100 on its own; the scores are not shares and need "
        "not add up, because several changes can each cause part of a "
        "regression. Where more than one is likely for a scope, say whether "
        "they are alternative explanations of the same movement — only one of "
        "them can be its cause — or independent contributions that each caused "
        "part of it."
    )


def _with_union(requests: Sequence[AttributionRequest]) -> AttributionRequest:
    """The first request carrying every request's historical analogues and
    external changes, so the rules about them are included when any has them."""
    first = requests[0]
    historical = {}
    for request in requests:
        for pr in request.historical:
            historical.setdefault((pr.repo.lower(), pr.number, pr.boundary_id), pr)
    external: set[ExternalChange] = set()
    for request in requests:
        external.update(request.external_changes)
    return dataclasses.replace(
        first, historical=tuple(historical.values()),
        external_changes=tuple(sorted(external)),
    )


def _scope_blocks(requests: Sequence[AttributionRequest], names: dict[str, str]) -> list[str]:
    """One block per scope under judgement, with every pull request's prior."""
    first = requests[0]
    facts = _attributed_facts(first)
    ids = scope_ids(facts)
    sweeps = {sweep.scope: sweep for sweep in first.sweeps}
    summarised = _summarised_facts(first)
    lines = [
        "Scopes with confirmed regressions in this window — answer each as a "
        "whole by its id, for every pull request under review:",
    ]
    for scope, sid in ids.items():
        rows = [f for f in facts if _fact_scope(f) == scope]
        row_ids = sorted((f.id for f in rows), key=_id_order)
        lines.append("")
        lines.append(
            f"[{sid}] {scope[0]} · {scope[2]} · {scope[1]} — {len(rows)} "
            f"confirmed regression(s), ids {', '.join(row_ids[:8])}"
            + (f", … (+{len(rows) - 8} more; each is marked in the sweep table below)"
               if len(rows) > 8 else "")
        )
        for request in requests:
            own = [f for f in _attributed_facts(request) if _fact_scope(f) == scope]
            for line in _prior_summary(own):
                lines.append(line.replace("  - prior", f"  - {names[request.slug]} prior", 1))
        sweep = sweeps.get(scope)
        if sweep is None:
            continue
        fact = summarised.get(scope)
        lines += scope_evidence_lines(
            sweep,
            history=fact.history if fact else None,
            history_metric=fact.metric if fact else "",
            history_label=fact.label if fact else "baseline",
        )
    return lines


def _summary_lines(requests: Sequence[AttributionRequest], names: dict[str, str]) -> list[str]:
    first = requests[0]
    by_detector: dict[str, list] = {}
    for sweep in first.sweeps:
        by_detector.setdefault(sweep.detector, []).append(sweep)
    lines = [EVIDENCE_HEADER, ""]
    for request in requests:
        if request.touches:
            lines.append(
                f"What {names[request.slug]} ({request.slug}) changes in the "
                "benchmarked detectors' geometry:"
            )
            lines += touch_lines(request.touches, by_detector, described=_described(first))
            lines.append("")
    lines += _scope_blocks(requests, names)
    other = other_scope_lines(first.sweeps, exclude=set(scope_ids(_attributed_facts(first))))
    if other:
        lines += ["", *other]
    return lines


def _competitors(requests: Sequence[AttributionRequest]) -> list[CompetingPR]:
    """The rest of the field: every request's competitors, minus the pull
    requests under review, each at its strongest sighting."""
    under_review = {(r.repo.lower(), r.number) for r in requests}
    field: dict[tuple[str, int], CompetingPR] = {}
    for request in requests:
        for competitor in request.competitors:
            key = (competitor.repo.lower(), competitor.number)
            if key in under_review:
                continue
            kept = field.get(key)
            if kept is None or competitor_order(competitor) < competitor_order(kept):
                field[key] = competitor
    return sorted(field.values(), key=competitor_order)[:MAX_COMPETITORS]


def build_joint_prompt(requests: Sequence[AttributionRequest]) -> str:
    """The user message for one joint review of *requests* (see
    :func:`joinable`): the window once, every pull request under review with its
    prior and diff, the rest of the field, and the joint answer format."""
    first = requests[0]
    names = labels(requests)
    competitors = _competitors(requests)
    competitor_need = sum(len(c.patch) for c in competitors)
    subject_need = [len(r.patch) for r in requests]
    subjects_total = min(
        sum(subject_need),
        max(_SUBJECT_DIFF_FLOOR * len(requests), _MAX_DIFF_CHARS - competitor_need),
    )
    subject_budgets = allocate_diff_budget(subject_need, subjects_total)
    competitor_budgets = allocate_diff_budget(
        [len(c.patch) for c in competitors], max(0, _MAX_DIFF_CHARS - subjects_total),
    )
    by_detector: dict[str, list] = {}
    for sweep in first.sweeps:
        by_detector.setdefault(sweep.detector, []).append(sweep)
    union = _with_union(requests)

    subject_lines: list[str] = []
    for request, budget in zip(requests, subject_budgets):
        size = f"+{request.additions}/-{request.deletions}"
        subject_lines += [
            "",
            f"Pull request under review {names[request.slug]} — {request.slug}: "
            f"{request.title} ({size})",
        ]
        if request.files:
            subject_lines.append(f"  files: {format_files(request.files, _MAX_FILES_LISTED)}")
        subject_lines += body_block(request.body, _MAX_SUBJECT_BODY_CHARS)
        subject_lines += diff_block(request.patch, budget)

    listing = "; ".join(
        f"{names[r.slug]} = {r.slug} — {r.title} (+{r.additions}/-{r.deletions})"
        for r in requests
    )
    parts = [
        f"Change window: {window_phrase(first.base_release, first.onset_release)} "
        "(Key4hep release dates).",
        *(line for base, onset in first.platform_switches
          for line in platform_switch_lines(base, onset)),
        *(line for base, onset in first.detector_switches
          for line in detector_switch_lines(base, onset)),
        f"The pull requests under review: {listing}.",
        "",
        *_summary_lines(requests, names),
        *external_change_lines(union.external_changes),
        "",
        "Details:",
        *_sweep_detail_lines(first),
        *_history_lines(first),
        *_package_lines(first),
        *subject_lines,
        *_competitor_lines(competitors, competitor_budgets, by_detector, _described(first)),
        *historical_lines(union.historical),
        "",
        "Work in this order, scope by scope. From the evidence summary: what "
        "moved in this scope and in what shape — in the typical event, in a few "
        "long events, outside the event loop, in memory — and is it noise? Then, "
        "for each pull request under review: what its diff changes that this "
        "scope loads, whether that mechanism predicts what moved and what did "
        "not, here and in the other detectors it reaches, what contradicts it, "
        "and the likelihood that it caused this scope's regressions, with an "
        "override for any row whose evidence differs from its scope's. Last, if "
        "more than one pull request is likely for the scope, whether they are "
        "alternative explanations of the same movement or independent "
        "contributions.",
        _JOINT_RESPONSE.format(ids=", ".join(names.values())),
    ]
    return log_prompt_size(
        "attribute (joint)", "\n".join(parts),
        detail=(
            f"{', '.join(r.slug for r in requests)}, {len(first.regressions)} row(s), "
            f"{len(competitors)} competitor(s)"
        ),
    )


_JOINT_RESPONSE = (
    'Respond with JSON only, no prose: {{"step_assessment": {{"verdict": '
    '"real_change" | "likely_noise" | "insufficient_evidence", "reason": "<one '
    'sentence citing the evidence>"}}, "scopes": [{{"scope": "<a scope id given '
    'above, e.g. S1>", "reading": "<what moved in this scope and in what shape>", '
    '"verdicts": [{{"pr": "<one of {ids}>", "mechanism": "<what this pull '
    "request's diff changes that this scope loads — or that it changes nothing "
    'it loads>", "supports": "<the evidence for it here>", "contradicts": "<the '
    'evidence against it here, or the empty string>", "likelihood": <0-100>, '
    '"overrides": [{{"id": "<a row id of this scope>", "likelihood": <0-100>}}]}}], '
    '"relation": "one_cause" | "several_causes"}}], "summaries": {{"<each of '
    '{ids}>": "<2-4 sentences: which of these regressions that pull request is '
    'responsible for and why, naming the cross-configuration evidence that '
    'decided it>"}}}}. Give every scope a verdict for every one of {ids}; set '
    '"relation" to "one_cause" when the likely pull requests are alternative '
    'explanations of the same movement and "several_causes" when each caused '
    "part of it — it is read only where more than one is likely. Give a row an "
    "override only where its evidence differs from the rest of its scope. In "
    "the summaries, refer to regressions by their detector, sample and metric "
    "— never by id — and to the other pull requests by owner/repo#number, "
    "never by P-label: they are read by people who never see these ids. Answer "
    "every scope listed above and invent none."
)


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_joint(
    content: str, requests: Sequence[AttributionRequest],
) -> dict[str, Attribution]:
    """The joint reply as one :class:`Attribution` per pull request it answered
    usably — likelihoods, its own summary, the shared step assessment — keyed
    by slug. A pull request the reply did not answer usably is absent, and the
    caller reviews it on its own."""
    data = extract_json(content)
    if not isinstance(data, dict):
        return {}
    names = labels(requests)
    slugs = {label: slug for slug, label in names.items()}
    assessment = _parse_assessment(data, slug="joint")
    summaries = data.get("summaries") if isinstance(data.get("summaries"), dict) else {}
    raw_scopes = data.get("scopes") if isinstance(data.get("scopes"), list) else []

    per_subject: dict[str, list[dict]] = {label: [] for label in slugs}
    relations: dict[str, str] = {}
    for scope in raw_scopes:
        if not isinstance(scope, dict):
            continue
        sid = str(scope.get("scope") or "").strip()
        relation = str(scope.get("relation") or "").strip()
        if relation in (ONE_CAUSE, SEVERAL_CAUSES):
            relations[sid] = relation
        verdicts = scope.get("verdicts")
        if not isinstance(verdicts, list):
            continue
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                continue
            label = str(verdict.get("pr") or "").strip().upper()
            if label in per_subject:
                per_subject[label].append({
                    "scope": sid, "reading": scope.get("reading"), **verdict,
                })

    out: dict[str, Attribution] = {}
    for request in requests:
        label = names[request.slug]
        own = {"scopes": per_subject[label]}
        facts = _attributed_facts(request)
        likelihoods = _parse_likelihoods(
            own, {f.id for f in facts}, slug=request.slug,
            scope_rows=_scope_rows(request),
        )
        summary = _without_row_ids(
            one_line(summaries.get(label), _MAX_SUMMARY_CHARS), facts, slug=request.slug,
        )
        if not likelihoods or not summary or assessment is None:
            _log.warning(
                "attribute (joint): %s — no usable answer in the joint reply "
                "(%d row(s), summary %s, assessment %s); reviewing it on its own",
                request.slug, len(likelihoods), "yes" if summary else "no",
                "yes" if assessment else "no",
            )
            continue
        scopes = _parse_scopes(own, request)
        for judgement in scopes:
            _log.info(
                "attribute (joint): %s — %s %s: %.0f%% — reading: %s | mechanism: %s "
                "| supports: %s | contradicts: %s | relation: %s",
                request.slug, judgement.scope_id, " · ".join(judgement.scope),
                judgement.likelihood, judgement.reading or "-",
                judgement.mechanism or "-", judgement.supports or "-",
                judgement.contradicts or "-", relations.get(judgement.scope_id, "-"),
            )
        out[request.slug] = Attribution(
            summary=summary, likelihoods=likelihoods, assessment=assessment,
            scopes=scopes,
        )

    # Alternatives: for a scope the reply called one cause, each pull request's
    # rows carry the others' likelihoods for the same rows.
    for request in requests:
        attribution = out.get(request.slug)
        if attribution is None:
            continue
        ids = scope_ids(_attributed_facts(request))
        scope_of = {f.id: ids[_fact_scope(f)] for f in _attributed_facts(request)}
        alternatives: dict[str, tuple[tuple[str, float], ...]] = {}
        for fact_id in attribution.likelihoods:
            if relations.get(scope_of.get(fact_id, "")) != ONE_CAUSE:
                continue
            others = tuple(
                (other.slug, out[other.slug].likelihoods[fact_id])
                for other in requests
                if other.slug != request.slug and other.slug in out
                and fact_id in out[other.slug].likelihoods
            )
            if others:
                alternatives[fact_id] = others
        out[request.slug] = dataclasses.replace(attribution, alternatives=alternatives)
    return out


def output_tokens(requests: Sequence[AttributionRequest]) -> int:
    """The output allowance for a joint reply: the single review's, per pull
    request under review."""
    rows = len(_attributed_facts(requests[0]))
    return min(
        MAX_OUTPUT_TOKENS,
        _OUTPUT_TOKENS_BASE * len(requests) + _OUTPUT_TOKENS_PER_ROW * rows * len(requests),
    )
