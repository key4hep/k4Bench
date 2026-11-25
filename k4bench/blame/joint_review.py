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
under review and names, per scope, the groups of them that are **competing**
explanations of the same movement (only one of a group can be its cause); a
likely pull request in no group is an **independent** contribution. Several
pull requests can cause one regression, so neither answer is assumed; the
likelihoods are not forced to add up.

The joint review is one unit from first call to last. Answers the reply left
out — a row's verdict, a pull request's summary — are re-asked **jointly**, with
the window shown again and the reply's earlier answers as context, never through
one pull request's own review: an independent reviewer would know nothing of
the joint reading and could clear what it blamed. The reply is then split back
into one :class:`~k4bench.blame.attribute.Attribution` per pull request, through
the same only-echo parsing the single review uses, and each carries the
competing explanations of its rows
(:attr:`~k4bench.blame.attribute.Attribution.alternatives`), taken from the
*final* likelihoods once every round is in. A joint review that cannot answer
every pull request under review answers none of them, and the window gets no
comment that night — exactly as a single review that does not answer
(:func:`k4bench.blame.comment.build_comments`).

Which windows are reviewed jointly is deterministic (:func:`joinable`): one
with a single selected pull request, or whose requests do not share their rows
and sweeps, keeps the single review.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence

from k4bench.blame.attribute import (
    _MAX_COMPLETION_ROUNDS,
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
    ScopeJudgement,
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
from k4bench.blame.llm import MAX_OUTPUT_TOKENS, ChatClient, extract_json, one_line
from k4bench.blame.prompt import (
    StepAssessment,
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

def _offered_rows(request: AttributionRequest) -> dict[str, tuple]:
    """``row id -> regression identity`` for the rows offered to the model.

    Row ids are positional (``r1``… in identity order), so two requests can
    share an id set while meaning different regressions by it; the joint reply
    is split back per pull request by id, which is only sound when every id
    names the same regression in every request."""
    return {
        f.id: (f.detector, f.platform, f.sample, f.label, f.metric, f.sub_detector or "")
        for f in _attributed_facts(request)
    }


def joinable(requests: Sequence[AttributionRequest]) -> bool:
    """Whether *requests* can be reviewed in one call: two or more, one suite
    and window, the same offered rows under the same ids, and the same sweeps,
    which the joint prompt reads every scope from (without them the rows carry
    one pull request's prior inline, which a joint prompt cannot)."""
    if len(requests) < 2:
        return False
    first = requests[0]
    rows = _offered_rows(first)
    sweeps = [sweep.scope for sweep in first.sweeps]
    return bool(sweeps) and all(
        (r.suite, r.base_release, r.onset_release) ==
        (first.suite, first.base_release, first.onset_release)
        and _offered_rows(r) == rows
        and [sweep.scope for sweep in r.sweeps] == sweeps
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


def build_joint_prompt(
    requests: Sequence[AttributionRequest],
    *,
    earlier: _JointReply | None = None,
    missing: dict[str, _Missing] | None = None,
) -> str:
    """The user message for one joint review of *requests* (see
    :func:`joinable`): the window once, every pull request under review with its
    prior and diff, the rest of the field, and the joint answer format.

    A follow-up round passes the reply so far (*earlier*) and what it left out
    (*missing*): the whole window is shown again, followed by the earlier
    answers as standing context, and only the answer is narrowed — the same
    shape as the single review's completion
    (:meth:`~k4bench.blame.attribute.OpenAICompatAttributor._complete`)."""
    first = requests[0]
    names = labels(requests)
    competitors = _competitors(requests)
    competitor_need = sum(len(c.patch) for c in competitors)
    subject_need = [len(r.patch) for r in requests]
    # Each subject is promised the single review's floor, but never more in
    # total than the whole diff budget: past five large pull requests the
    # floors alone would overrun it, and they then share it evenly instead.
    subjects_total = min(
        sum(subject_need),
        max(
            min(_SUBJECT_DIFF_FLOOR * len(requests), _MAX_DIFF_CHARS),
            _MAX_DIFF_CHARS - competitor_need,
        ),
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
        "more than one pull request is likely for the scope, which of them are "
        "competing explanations of the same movement and which independent "
        "contributions.",
        *(_earlier_lines(requests, names, earlier, missing) if missing else ()),
        _JOINT_FORMAT.format(ids=", ".join(names.values()))
        + (_JOINT_ONLY if missing else _JOINT_ALL.format(ids=", ".join(names.values()))),
    ]
    return log_prompt_size(
        "attribute (joint)" + (" follow-up" if missing else ""), "\n".join(parts),
        detail=(
            f"{', '.join(r.slug for r in requests)}, {len(first.regressions)} row(s), "
            f"{len(competitors)} competitor(s)"
        ),
    )


def _earlier_lines(
    requests: Sequence[AttributionRequest],
    names: dict[str, str],
    earlier: _JointReply | None,
    missing: dict[str, _Missing],
) -> list[str]:
    """The reply so far, as context a follow-up round must stand by, and what
    it is asked to add."""
    lines = ["", "Your earlier reply already answered part of this window, and those answers stand:"]
    for request in requests:
        label = names[request.slug]
        answer = earlier.answers.get(request.slug) if earlier is not None else None
        judged = ", ".join(
            f"{j.scope_id} {j.likelihood:.0f}" for j in (answer.scopes if answer else ())
        )
        lines.append(f"  - {label} ({request.slug}): {judged or 'no scope verdicts yet'}")
        if answer is not None and answer.summary:
            lines.append(f"    summary: {answer.summary}")
    if earlier is not None:
        for sid, groups in sorted(earlier.competing.items(), key=lambda kv: int(kv[0][1:])):
            named = "; ".join(
                " / ".join(sorted(names[slug] for slug in group)) for group in sorted(groups, key=sorted)
            )
            lines.append(f"  - competing in {sid}: {named}")
    wanted = []
    for request in requests:
        gap = missing.get(request.slug)
        if gap is None:
            continue
        label = names[request.slug]
        if gap.rows:
            ids = scope_ids(_attributed_facts(request))
            scope_of = {f.id: ids[_fact_scope(f)] for f in _attributed_facts(request)}
            scopes = sorted({scope_of[i] for i in gap.rows}, key=lambda sid: int(sid[1:]))
            wanted.append(
                f"{label}'s verdicts in scope(s) {', '.join(scopes)} for row(s) "
                f"{', '.join(sorted(gap.rows, key=_id_order))}"
            )
        if gap.summary:
            wanted.append(f"{label}'s summary")
    lines.append(
        "Re-read all of the above as evidence — the cross-configuration pattern "
        "still decides — and keep to the earlier answers, but answer only what is "
        f"still missing: {'; '.join(wanted)}."
    )
    return lines


_JOINT_FORMAT = (
    'Respond with JSON only, no prose: {{"step_assessment": {{"verdict": '
    '"real_change" | "likely_noise" | "insufficient_evidence", "reason": "<one '
    'sentence citing the evidence>"}}, "scopes": [{{"scope": "<a scope id given '
    'above, e.g. S1>", "reading": "<what moved in this scope and in what shape>", '
    '"verdicts": [{{"pr": "<one of {ids}>", "mechanism": "<what this pull '
    "request's diff changes that this scope loads — or that it changes nothing "
    'it loads>", "supports": "<the evidence for it here>", "contradicts": "<the '
    'evidence against it here, or the empty string>", "likelihood": <0-100>, '
    '"overrides": [{{"id": "<a row id of this scope>", "likelihood": <0-100>}}]}}], '
    '"competing": [["<one of {ids}>", "<another>"]]}}], "summaries": {{"<one of '
    '{ids}>": "<2-4 sentences: which of these regressions that pull request is '
    'responsible for and why, naming the cross-configuration evidence that '
    'decided it>"}}}}. In "competing", list each group of the likely pull '
    "requests that are competing explanations of the same movement in this "
    "scope — only one of a group can be its cause. Name a group only when that "
    "holds for every row of the scope its members are likely for; leave out a "
    "pull request that contributed independently, and give [] when no two are "
    "competing. Give a row an override only where its evidence differs from the "
    "rest of its scope. In the summaries, refer to regressions by their "
    "detector, sample and metric — never by id — and to the other pull requests "
    "by owner/repo#number, never by P-label: they are read by people who never "
    "see these ids. "
)

#: The first round's completeness rule.
_JOINT_ALL = (
    "Give every scope a verdict for every one of {ids}, a summary for every one "
    "of them, and answer every scope listed above and invent none."
)

#: A follow-up round's: only what the earlier reply left out.
_JOINT_ONLY = (
    "Answer only what is listed above as still missing: verdicts only from "
    "those pull requests and in those scopes, summaries only for those whose "
    'summary is missing, and "competing" for the scopes you answer. Invent '
    "nothing."
)


# ── Parse ─────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _Answer:
    """What the reply so far says about one pull request under review."""

    likelihoods: dict[str, float] = dataclasses.field(default_factory=dict)
    summary: str = ""
    scopes: list[ScopeJudgement] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _JointReply:
    """The joint reply so far, merged across rounds: the shared step
    assessment, each pull request's answer by slug, and per scope id the groups
    of slugs named as competing explanations of one movement."""

    assessment: StepAssessment | None
    answers: dict[str, _Answer]
    competing: dict[str, set[frozenset[str]]]


@dataclasses.dataclass(frozen=True)
class _Missing:
    """What one pull request's answer still lacks: offered rows without a
    verdict, and whether its summary is missing."""

    rows: tuple[str, ...]
    summary: bool


def _parse_round(
    content: str,
    requests: Sequence[AttributionRequest],
    *,
    wanted: dict[str, set[str]] | None = None,
) -> _JointReply | None:
    """One round's reply, or ``None`` when it is not a JSON object.

    *wanted* restricts each pull request's verdicts to the rows a follow-up
    asked for (``slug -> row ids``); the first round accepts every offered row.
    Only-echo holds per pull request exactly as in the single review."""
    data = extract_json(content)
    if not isinstance(data, dict):
        return None
    names = labels(requests)
    slugs = {label: slug for slug, label in names.items()}
    summaries = data.get("summaries") if isinstance(data.get("summaries"), dict) else {}
    raw_scopes = data.get("scopes") if isinstance(data.get("scopes"), list) else []
    scope_rows = _scope_rows(requests[0])

    per_subject: dict[str, list[dict]] = {label: [] for label in slugs}
    competing: dict[str, set[frozenset[str]]] = {}
    for scope in raw_scopes:
        if not isinstance(scope, dict):
            continue
        sid = str(scope.get("scope") or "").strip()
        if sid in scope_rows:
            groups = _competing_groups(scope.get("competing"), slugs)
            if groups:
                competing[sid] = groups
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

    answers: dict[str, _Answer] = {}
    for request in requests:
        label = names[request.slug]
        own = {"scopes": per_subject[label]}
        facts = _attributed_facts(request)
        known = {f.id for f in facts}
        if wanted is not None:
            known &= wanted.get(request.slug, set())
        answers[request.slug] = _Answer(
            likelihoods=_parse_likelihoods(
                own, known, slug=request.slug, scope_rows=_scope_rows(request),
            ) if known else {},
            summary=_without_row_ids(
                one_line(summaries.get(label), _MAX_SUMMARY_CHARS), facts,
                slug=request.slug,
            ),
            scopes=list(_parse_scopes(own, request)),
        )
    return _JointReply(
        assessment=_parse_assessment(data, slug="joint"),
        answers=answers, competing=competing,
    )


def _competing_groups(raw: object, slugs: dict[str, str]) -> set[frozenset[str]]:
    """A scope's ``competing`` entry as groups of slugs, keeping only groups
    that still name two or more pull requests under review."""
    if not isinstance(raw, list):
        return set()
    groups = set()
    for group in raw:
        if not isinstance(group, list):
            continue
        members = frozenset(
            slugs[label] for item in group
            if (label := str(item or "").strip().upper()) in slugs
        )
        if len(members) >= 2:
            groups.add(members)
    return groups


def _merge(reply: _JointReply, more: _JointReply) -> int:
    """Fold a follow-up round into *reply*; how many answers it added (rows
    and summaries). Earlier answers are never replaced: they were read against
    the same window, and the follow-up was asked to stand by them."""
    added = 0
    for slug, answer in more.answers.items():
        kept = reply.answers[slug]
        for row_id, score in answer.likelihoods.items():
            if row_id not in kept.likelihoods:
                kept.likelihoods[row_id] = score
                added += 1
        if not kept.summary and answer.summary:
            kept.summary = answer.summary
            added += 1
        judged = {j.scope_id for j in kept.scopes}
        kept.scopes += [j for j in answer.scopes if j.scope_id not in judged]
    for sid, groups in more.competing.items():
        reply.competing.setdefault(sid, set()).update(groups)
    return added


def _missing(
    reply: _JointReply, requests: Sequence[AttributionRequest],
) -> dict[str, _Missing]:
    """``slug -> what its answer lacks``, for every pull request with a gap."""
    out = {}
    for request in requests:
        answer = reply.answers[request.slug]
        rows = tuple(
            f.id for f in _attributed_facts(request) if f.id not in answer.likelihoods
        )
        if rows or not answer.summary:
            out[request.slug] = _Missing(rows=rows, summary=not answer.summary)
    return out


def _alternatives(
    request: AttributionRequest, reply: _JointReply,
) -> dict[str, tuple[tuple[str, float], ...]]:
    """``row id -> ((slug, likelihood), …)``: for each row *request* was scored
    on, the pull requests named with it in a competing group of the row's scope
    and their likelihoods for that row — read from the final, merged answers."""
    ids = scope_ids(_attributed_facts(request))
    scope_of = {f.id: ids[_fact_scope(f)] for f in _attributed_facts(request)}
    out: dict[str, tuple[tuple[str, float], ...]] = {}
    for row_id in reply.answers[request.slug].likelihoods:
        rivals = sorted({
            slug
            for group in reply.competing.get(scope_of.get(row_id, ""), ())
            if request.slug in group
            for slug in group
            if slug != request.slug and row_id in reply.answers[slug].likelihoods
        })
        if rivals:
            out[row_id] = tuple(
                (slug, reply.answers[slug].likelihoods[row_id]) for slug in rivals
            )
    return out


def _attributions(
    reply: _JointReply, requests: Sequence[AttributionRequest],
) -> dict[str, Attribution]:
    """Every pull request's :class:`Attribution`, or ``{}`` when any of them
    has no likelihood or no summary: the joint reading is one unit, and a part
    of it is not a review of the window."""
    unanswered = [
        r.slug for r in requests
        if not reply.answers[r.slug].likelihoods or not reply.answers[r.slug].summary
    ]
    if unanswered:
        _log.warning(
            "attribute (joint): %s — no usable answer for %s after the follow-up "
            "rounds; no review for this window tonight",
            ", ".join(r.slug for r in requests), ", ".join(unanswered),
        )
        return {}
    names = labels(requests)
    for sid, groups in sorted(reply.competing.items()):
        _log.info(
            "attribute (joint): %s competing: %s", sid,
            "; ".join(" / ".join(sorted(names[s] for s in g)) for g in sorted(groups, key=sorted)),
        )
    out = {}
    for request in requests:
        answer = reply.answers[request.slug]
        for judgement in answer.scopes:
            _log.info(
                "attribute (joint): %s — %s %s: %.0f%% — reading: %s | mechanism: %s "
                "| supports: %s | contradicts: %s",
                request.slug, judgement.scope_id, " · ".join(judgement.scope),
                judgement.likelihood, judgement.reading or "-",
                judgement.mechanism or "-", judgement.supports or "-",
                judgement.contradicts or "-",
            )
        out[request.slug] = Attribution(
            summary=answer.summary, likelihoods=dict(answer.likelihoods),
            assessment=reply.assessment, scopes=tuple(answer.scopes),
            alternatives=_alternatives(request, reply),
        )
    return out


def parse_joint(
    content: str, requests: Sequence[AttributionRequest],
) -> dict[str, Attribution]:
    """A single joint reply as one :class:`Attribution` per pull request, keyed
    by slug — or ``{}`` unless it answered every one of them usably."""
    reply = _parse_round(content, requests)
    if reply is None or reply.assessment is None:
        return {}
    return _attributions(reply, requests)


# ── Review ────────────────────────────────────────────────────────────────────

def review_jointly(
    client: ChatClient, requests: Sequence[AttributionRequest],
) -> dict[str, Attribution]:
    """One joint review of *requests*: the first call, then up to
    :data:`~k4bench.blame.attribute._MAX_COMPLETION_ROUNDS` joint follow-ups
    for whatever it left out, each required to make progress.

    Every pull request in *requests* answered, or ``{}``: a call that fails, a
    reply without a step assessment, and a pull request still without a verdict
    or a summary after the follow-ups all decline the whole window. Rows still
    unanswered for a pull request that is otherwise answered keep their
    first-pass score, exactly as in the single review."""
    if not joinable(requests):
        return {}
    group = ", ".join(r.slug for r in requests)
    system = _joint_system_prompt(requests)
    try:
        content, finish_reason = client.complete(
            system, build_joint_prompt(requests),
            max_output_tokens=output_tokens(requests),
        )
    except Exception as exc:
        _log.warning(
            "attribute (joint): %s — LLM call failed (%s); no review for this "
            "window tonight", group, exc,
        )
        return {}
    reply = _parse_round(content, requests)
    if reply is None or reply.assessment is None:
        _log.warning(
            "attribute (joint): %s — unusable reply (finish_reason=%s, %s); no "
            "review for this window tonight; response prefix=%r",
            group, finish_reason,
            "not JSON" if reply is None else "no step_assessment", content[:500],
        )
        return {}

    for _ in range(_MAX_COMPLETION_ROUNDS):
        missing = _missing(reply, requests)
        if not missing:
            break
        _log.info(
            "attribute (joint): %s — re-asking jointly for %s", group,
            "; ".join(
                f"{slug}: {len(gap.rows)} row(s)" + (" + summary" if gap.summary else "")
                for slug, gap in missing.items()
            ),
        )
        try:
            content, _finish = client.complete(
                system,
                build_joint_prompt(requests, earlier=reply, missing=missing),
                max_output_tokens=output_tokens(requests, missing),
            )
        except Exception as exc:
            _log.warning(
                "attribute (joint): %s — follow-up failed (%s); keeping what is "
                "answered", group, exc,
            )
            break
        more = _parse_round(
            content, requests,
            wanted={slug: set(gap.rows) for slug, gap in missing.items()},
        )
        if more is None or not _merge(reply, more):
            _log.warning(
                "attribute (joint): %s — follow-up answered nothing missing; "
                "keeping what is answered", group,
            )
            break
    return _attributions(reply, requests)


def output_tokens(
    requests: Sequence[AttributionRequest],
    missing: dict[str, _Missing] | None = None,
) -> int:
    """The output allowance for a joint reply: the single review's, per pull
    request asked — for a follow-up, per pull request and row still missing."""
    if missing is None:
        rows = len(_attributed_facts(requests[0]))
        need = _OUTPUT_TOKENS_BASE * len(requests) + _OUTPUT_TOKENS_PER_ROW * rows * len(requests)
    else:
        need = sum(
            _OUTPUT_TOKENS_BASE + _OUTPUT_TOKENS_PER_ROW * len(gap.rows)
            for gap in missing.values()
        )
    return min(MAX_OUTPUT_TOKENS, need)
