"""A drafted comment as the GitHub-flavoured Markdown that is posted.

:class:`CommentDraft` is what :func:`~k4bench.blame.comment.build.build_comments`
hands the publisher, and :meth:`CommentDraft.render` turns it into a
:class:`PRComment` once the comments already posted in its lineage are known —
including the facts digest that decides whether a standing comment is edited.
Externally authored prose (titles, model summaries) is defanged on its way in.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from k4bench.blame.attribute import Attribution
from k4bench.blame.comment.digest import (
    Reproducers,
    _digest_from,
    _retained_fact,
)
from k4bench.blame.comment.markdown import (
    _MAX_DESCRIPTION_CHARS,
    _MAX_SUMMARY_CHARS,
    _cell,
    _count,
    _one_line,
    _pct,
    _sentences,
)
from k4bench.blame.comment.selection import (
    _DEFAULT_MIN_SCORE,
    CommentPlan,
    RegressionRow,
    _row_identity,
    _sorted_others,
)
from k4bench.blame.comment.state import (
    _FACTS_MARKER_PREFIX,
    _MARKER_SUFFIX,
    _MAX_OBSERVATIONS,
    _MAX_RETAINED_ROWS,
    _OMITTED_OBSERVATIONS_PREFIX,
    CommentObservation,
    LineageState,
    RetainedRow,
    _cumulative_marker,
    _iso_date,
    _observation_key,
    _observation_marker,
    _report_counts,
    _retained_identity,
    _retained_marker,
    _retained_sort_key,
    _retained_window,
    _verdict_identity,
)
from k4bench.blame.models import (
    RANKING_DISCLOSURE,
    CandidatePR,
)
from k4bench.blame.prompt import movement
from k4bench.labels import compact_sample, pretty_platform
from k4bench.regression.render import (
    nightly_report_href,
    regression_href,
    stack_changes_href,
    window_href,
)

#: Regression rows shown in the table, and candidate rows shown for the rest of
#: the window. Every other row is still scored and linked through the dashboard.
#: A sweep can confirm three hundred near-identical rows and a comment over
#: GitHub's 65,536-character limit is rejected outright, so these hard bounds
#: are part of the write contract rather than only a presentation preference.
_TARGET_TABLE_ROWS = 5
_MAX_OTHER_CANDIDATES = 5
_UNKNOWN_ONSET = "unknown"

#: Likelihood points between this PR and the closest other candidate at or
#: under which the ranking is called a weak preference in words. Wide enough to
#: catch a genuinely crowded field, narrow enough that an ordinary night — where
#: the ranker picked one PR out of the pack — says nothing extra.
_CROWDED_SPREAD = 10.0

#: Whether the regression table carries a **Platform** column.
#:
#: Off: the suite builds on one platform today, so the column would repeat one
#: slug down every row. Purely a rendering choice — platform remains a first-class
#: scope dimension everywhere else (row identity, grouping, outcomes, dashboard
#: links, the facts digest, package provenance, both prompts), and the renderer
#: below is written to show the column the moment this is flipped. It is a
#: constant rather than a count of the platforms actually present, so the table's
#: shape is a decision someone made, not an accident of one night's data.
_SHOW_PLATFORM_COLUMN = False

#: Where the comment's own footer points: the page describing how a regression
#: is attributed to a pull request, on the published docs site (``site_url`` in
#: mkdocs.yml + the page's nav path).
_METHOD_URL = "https://key4hep.github.io/k4Bench/user-guide/features/pr-comments/"

#: Who to write to about a comment. Rendered as a ``mailto:`` link, so a reader
#: who thinks the bot got it wrong can reach a person in one click rather than
#: replying into a thread nobody may be watching.
_CONTACT_EMAIL = "jbeirer@cern.ch"


@dataclass(frozen=True)
class PRComment:
    """One rendered comment and where it goes.

    ``marker`` is the hidden key the upsert recognises and ``facts_digest``
    fingerprints the benchmark facts behind the body; both are hidden lines at
    the top of ``body``, so a comment always carries the keys that identify it
    and the state it was rendered from. An empty ``facts_digest`` makes the
    upsert compare whole bodies instead (:attr:`CommentDraft.keyed_by_facts`).
    """

    repo: str
    number: int
    marker: str
    body: str
    score: float
    facts_digest: str = ""

    @property
    def target(self) -> str:
        """``owner/repo#123`` — how this comment is named in logs."""
        return f"{self.repo}#{self.number}"


@dataclass(frozen=True)
class CommentDraft:
    """One comment decided and reviewed, rendered only when it is written.

    The body draws on state the comments already posted in its lineage carry —
    the exact regression identities behind the headline, the rows retained from
    earlier versions, and the observation history — and only the publisher has
    read those. So :func:`build_comments` stops here, and :meth:`render` builds
    the body once the prior bodies are known. A dry run renders with none, which
    is exactly what a first comment says.

    ``facts_payload`` is the canonical set of benchmark facts the digest hashes
    (:func:`_facts_payload`), settled at build time from the plan, the review
    request and the published recipes. ``keyed_by_facts`` is off only for a
    preview, which iterates on wording against deliberately unchanged facts and
    must therefore be compared on its whole body
    (:func:`k4bench.blame.publish._upsert`).
    """

    repo: str
    number: int
    marker: str
    plan: CommentPlan
    attribution: Attribution | None
    facts_payload: dict[str, Any]
    #: The recipes published for this comment's rows (:class:`Reproducers`).
    reproduce: Reproducers | None = None
    dashboard_url: str | None = None
    min_score: float = _DEFAULT_MIN_SCORE
    keyed_by_facts: bool = True

    @property
    def target(self) -> str:
        """``owner/repo#123`` — how this comment is named in logs."""
        return f"{self.repo}#{self.number}"

    @property
    def score(self) -> float:
        return self.plan.top_score

    @property
    def observation(self) -> CommentObservation | None:
        """This version as the observation history records it, or ``None`` for
        a report with no night to date it by."""
        return _observation_for(self.plan, self.plan.rows, self.dashboard_url)

    def render(self, previous_bodies: Sequence[str] = ()) -> PRComment:
        """The comment as it will be written, given the comparable bodies
        already posted — the survivor of this lineage first, then any other
        comparable comment a converging window is absorbing."""
        return _render(self, LineageState.from_bodies(previous_bodies))


def _render(draft: CommentDraft, lineage: LineageState) -> PRComment:
    """One draft as a GitHub-flavoured Markdown comment, given its lineage.

    A single comment for the ``(pull request, window)``: the claim, the window,
    the model's reasoning and any qualifier on it, the table of the most likely
    rows, the association with each scope, the observation history and the
    competing pull requests.

    Everything that draws on earlier versions comes from structured hidden state
    (:class:`LineageState`), never from their Markdown:

    * the **headline counts and association table** describe the exact identity
      union across the lineage and tonight's report, using the newest published
      attribution for a repeated identity;
    * **retained rows** rejoin the table's selection pool beside tonight's
      confirmed rows, and a fresh bounded state marker is written for the next
      version;
    * the **facts digest** covers the table as it renders: which historical rows
      render, their frozen facts and their standing in tonight's report are
      deterministic inputs a reader can see — while a retained row's mere
      heartbeat (tonight's date) is not;
    * the **observation history** carries every material version forward.

    Archive links pin the report being rendered. Report dates stay outside the
    digest, so they alone never make an unchanged regression re-notify a pull
    request.
    """
    plan, attribution = draft.plan, draft.attribution
    by_likelihood = sorted(
        plan.rows, key=lambda row: _row_sort_key(row, attribution)
    )

    # Tonight's rows overlay the lineage's: a reconfirmed identity keeps only
    # its newest published attribution.
    cumulative = dict(lineage.cumulative)
    for row in plan.rows:
        cumulative[_row_identity(row)] = (
            *_score_source(row, attribution), plan.report_night,
        )
    cumulative_marker = _cumulative_marker(cumulative)
    payload = dict(draft.facts_payload)
    if cumulative_marker:
        identities = sorted(cumulative)
        scopes = len({identity[:3] for identity in identities})
        reviewed = [score for score, source, _ in cumulative.values()
                    if score is not None and source == "reviewer"]
        carried = [score for score, source, _ in cumulative.values()
                   if score is not None and source == "ranker"]
        alert = _alert(
            plan, by_likelihood, attribution, min_score=draft.min_score,
            n_regressions=len(identities), n_scopes=scopes,
            scores=(reviewed, carried),
        )
        association = _association_summary(
            cumulative, draft.min_score, draft.dashboard_url,
        )
        canonical = json.dumps(identities, separators=(",", ":"))
        payload["cumulative"] = {
            "digest": hashlib.sha256(canonical.encode()).hexdigest()[:16],
            "regressions": len(identities),
            "scopes": scopes,
        }
    else:
        # A union too large to carry costs this night's merge — the state is
        # still in the bodies it would have merged — so the headline falls back
        # to counting tonight's report alone.
        alert = _alert(plan, by_likelihood, attribution, min_score=draft.min_score)
        association = _association_summary(
            {
                _row_identity(row): (*_score_source(row, attribution), plan.report_night)
                for row in plan.rows
            },
            draft.min_score, draft.dashboard_url,
        )

    # Read before the table, so the line under it can count each report the
    # rows came from, and again by the history below — one merge, so the two
    # can never disagree about what a report contained.
    observations, omitted = _merged_observations(draft, lineage)

    # Any identity confirmed tonight is superseded by its current evidence —
    # including a current ``not_candidate`` or unscored standing, which must
    # not be papered over by an old score.
    current = {_row_identity(row) for row in plan.rows}
    historical = sorted(
        (row for row in lineage.retained if _retained_identity(row) not in current),
        key=_retained_sort_key,
    )
    region, shown_past = _details_region(
        plan, attribution, dashboard_url=draft.dashboard_url,
        historical=historical,
        retained_marker=_retained_marker(_retained_state(
            plan, attribution, historical, draft.reproduce,
        )),
        reproduce=draft.reproduce, observations=observations,
    )
    digest = _digest_from(payload, [_retained_fact(row) for row in shown_past])

    history = None
    if draft.observation is not None:
        # A history of one repeats the window and counts the comment already
        # states above it, and has nothing to compare them against; it earns its
        # place from the second material update on. The hidden markers are still
        # emitted — they are the lineage's observation state, and dropping them
        # would restart the history at one entry every night, permanently.
        history = (
            "\n".join(_observation_marker(item) for item in observations)
            if len(observations) < 2 and not omitted
            else _observation_history(observations, omitted=omitted)
        )

    body = "\n".join(
        part for part in (
            draft.marker,
            f"{_FACTS_MARKER_PREFIX}{digest} -->",
            cumulative_marker or None,
            "### 📊 Possible performance change traced to this pull request",
            "",
            alert,
            "",
            _window_line(plan),
            region,
            # Below the table, not above it: the assessment is the reasoning
            # behind the alert and belongs next to it, and this summary is the
            # drill-down on the alert's counts — which reads after the rows it
            # generalizes, beside the per-report line asking the same question.
            association,
            history,
            _others_section(plan, draft.dashboard_url),
            "",
            "---",
            "",
            # The contact line is this renderer's own, not part of the shared
            # disclosure. A machine-written accusation needs a human at the end
            # of it, and a named address reaches one whether or not anyone is
            # watching the thread. k4Bench's name carries the page describing
            # how this attribution is made rather than the repository root:
            # someone who doubts the claim wants the method, and a README makes
            # them go looking for it.
            f"<sub>🤖 {RANKING_DISCLOSURE} Posted automatically by "
            f"[k4Bench]({_METHOD_URL}) — questions or feedback: "
            f"[{_CONTACT_EMAIL}](mailto:{_CONTACT_EMAIL})</sub>",
        ) if part is not None
    )
    return PRComment(
        repo=draft.repo,
        number=draft.number,
        marker=draft.marker,
        body=body,
        score=draft.score,
        facts_digest=digest if draft.keyed_by_facts else "",
    )


def _details_region(
    plan: CommentPlan,
    attribution: Attribution | None,
    *,
    dashboard_url: str | None,
    historical: list[RetainedRow],
    retained_marker: str | None,
    reproduce: Reproducers | None = None,
    observations: Sequence[CommentObservation] = (),
) -> tuple[str, list[RetainedRow]]:
    """The middle of a comment — assessment, regression table and its
    reference links — rendered from structured inputs alone. Returns the region
    and the historical rows it actually rendered, which are what the digest
    must cover."""
    by_likelihood = sorted(
        plan.rows, key=lambda row: _row_sort_key(row, attribution)
    )
    shown = _selected_rows(by_likelihood, historical, attribution)
    shown_current = [entry.current for entry in shown if entry.current is not None]
    shown_past = [entry.past for entry in shown if entry.past is not None]
    links = _row_links(plan, shown_current, dashboard_url)
    past_links = _past_links(shown_past, dashboard_url)
    definitions = dict(links)
    definitions.update(dict(past_links.values()))
    region = "\n".join(
        part for part in (
            retained_marker,
            _assessment(plan, by_likelihood, attribution, rendered=shown_current),
            _crowded_note(plan),
            _table(
                plan, by_likelihood, attribution,
                shown=shown, links=links, past_links=past_links,
                dashboard_url=dashboard_url, reproduce=reproduce,
                observations=observations,
            ),
            _link_definitions(definitions),
        ) if part is not None
    )
    return region, shown_past


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
        "published reports. Attribution uses each regression's latest published AI score.",
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
            _capped_scopes_note(over_cap),
            _unattributed_note(unattributed, min_score),
        ) if note
    ]
    for note in trailing:
        lines += ["", note]
    lines += [
        "",
        "Report dates open the **full nightly report**, including other change windows.",
    ]
    return "\n".join(lines)


def _capped_scopes_note(over_cap: list[tuple]) -> str | None:
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
        f"({_count(rows, 'regression')}) are not listed above."
    )


def _unattributed_note(unattributed: list[tuple], min_score: float) -> str | None:
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
        f"A further {_count(rows, 'regression')} in "
        f"{_count(len(unattributed), 'scope')} stayed below {_pct(min_score)} "
        f"and are not attributed to this PR ({standing})."
    )


def _merged_observations(
    draft: CommentDraft, lineage: LineageState
) -> tuple[list[CommentObservation], int]:
    """This comment's material versions, newest first, and how many aged out.

    Every comparable body's observations carried forward and tonight's laid over
    them, keyed by report night and window: re-running a changed report for the
    same night replaces that night's entry, and a later material change appends
    one. Archive links are re-pointed on the way through, so observations
    recorded against an older link shape migrate on the next material edit.

    Absorbed lineages are read too, because convergence hands their *rows* to
    this comment (:func:`_merged_retained`) and a table drawing a row from a
    report the history never recorded leaves both the history and the line under
    the table describing a lineage this comment no longer has.

    Entries are keyed by **report night and window**, not by night alone. The
    publisher only converges windows that are containment-comparable with the
    current one, but two absorbed windows can still be siblings — ``(09-01,
    09-02]`` and ``(09-02, 09-03]`` are each contained by ``(09-01, 09-03]``
    and neither contains the other — and those describe *disjoint* sets of
    regressions observed on the same night. Keying by night alone would drop one
    of them, taking its rows' provenance with it.
    """
    observations = {
        key: replace(
            item, url=nightly_report_href(
                draft.dashboard_url or item.url, item.report_night,
            ),
        )
        for key, item in lineage.observations.items()
    }
    omitted = lineage.omitted
    observation = draft.observation
    if observation is not None:
        observations[_observation_key(observation)] = observation
    ordered = sorted(
        observations.values(),
        key=lambda item: (
            item.report_night, item.base_release or "", item.onset_release,
        ),
        reverse=True,
    )
    if len(ordered) > _MAX_OBSERVATIONS:
        omitted += len(ordered) - _MAX_OBSERVATIONS
        ordered = ordered[:_MAX_OBSERVATIONS]
    return ordered, omitted


def _observation_history(
    observations: list[CommentObservation], *, omitted: int
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
        "| Report | Change window | Regressions | Scopes | Directions |",
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


def _likelihood(row: RegressionRow, attribution: Attribution | None) -> float | None:
    """What this row is shown as, and ordered by — or ``None`` when nothing has
    been claimed about it.

    The review's score when it gave one; otherwise the per-configuration
    ranker's. A row the review omitted is not a zero — an unanswered row keeps
    the judgement that was already made about it — and a row neither pass judged
    has no likelihood at all. Rendering that as 0% would read as "the models
    cleared this one", which nobody said.

    A row the pull request is *not a candidate* for never carries a likelihood,
    not even one the review offered. That state is a deterministic fact — the
    candidate search in that scope was complete and this change is not in the
    commit range behind the regression — and it outranks a model's opinion about
    it. Letting a stray high score on such a row into the table would put a
    percentage next to a regression the pipeline knows this change cannot have
    shipped in, and (through the withdrawal gate below) would let it hold up a
    comment on its own."""
    if row.scope_state == "not_candidate":
        return None
    if attribution is None:
        return row.scope_score
    return attribution.likelihoods.get(row.fact_id, row.scope_score)


def _score_source(
    row: RegressionRow, attribution: Attribution | None
) -> tuple[float | None, str]:
    likelihood = _likelihood(row, attribution)
    if likelihood is None or not (
        math.isfinite(likelihood) and 0 <= likelihood <= 100
    ):
        return None, ""
    reviewed = attribution is not None and row.fact_id in attribution.likelihoods
    return likelihood, "reviewer" if reviewed else "ranker"


def _row_sort_key(row: RegressionRow, attribution: Attribution | None) -> tuple:
    """Most likely first, then the full detector before its removal
    configurations, then the largest movement, then identity — so the table is
    stable across nights and a re-render triggers no edit. Rows nobody scored
    sort last: they are evidence about the window, not claims about this pull
    request, and they must not head a table that reads top-down.

    The baseline leads its ties because a review scores a scope as a whole, so
    its rows share one likelihood, and the full detector is the configuration a
    reader measures the claim against — a removal configuration's larger
    percentage is often only a smaller denominator."""
    likelihood = _likelihood(row, attribution)
    return (
        likelihood is None,
        -(likelihood if likelihood is not None else 0.0),
        row.verdict.label != "baseline",
        -movement(row.verdict.pct_change),
        *_row_identity(row),
    )


@dataclass(frozen=True)
class _PoolRow:
    """One candidate table row — tonight's evidence or a retained snapshot —
    with the sort key both kinds share, so a single ranked pool orders them."""

    key: tuple
    current: RegressionRow | None = None
    past: RetainedRow | None = None


def _selected_rows(
    current: list[RegressionRow],
    historical: list[RetainedRow],
    attribution: Attribution | None,
) -> list[_PoolRow]:
    """The rows the table shows, in ranked order.

    The pool is every currently confirmed row plus every retained historical
    row, ranked by the shared likelihood/movement/identity key. Selection then
    reserves, in order: the pool's **globally strongest row** — the alert quotes
    the strongest evidence, so the table must never hide it, however old its
    onset; the strongest *current* row, for the same reason when a historical
    row leads; one representative per current onset — an undated group before
    the dated ones, newest dated first — so a newer step cannot vanish behind
    five rows of an older one; and finally the strongest remaining rows, all
    inside the hard cap."""
    pool = [
        _PoolRow(key=_row_sort_key(row, attribution), current=row)
        for row in current
    ] + [
        _PoolRow(key=_retained_sort_key(row), past=row) for row in historical
    ]
    pool.sort(key=lambda entry: entry.key)
    current_pool = [entry for entry in pool if entry.current is not None]

    chosen: dict[tuple, _PoolRow] = {}

    def _reserve(entry: _PoolRow) -> None:
        if len(chosen) < _TARGET_TABLE_ROWS:
            chosen.setdefault(entry.key, entry)

    if pool:
        _reserve(pool[0])
    if current_pool:
        _reserve(current_pool[0])
    strongest_per_onset: dict[str, _PoolRow] = {}
    for entry in current_pool:
        strongest_per_onset.setdefault(_onset_label(entry.current), entry)
    covered = {
        _onset_label(entry.current)
        for entry in chosen.values()
        if entry.current is not None
    }
    dated = sorted(set(strongest_per_onset) - {_UNKNOWN_ONSET}, reverse=True)
    undated = [_UNKNOWN_ONSET] if _UNKNOWN_ONSET in strongest_per_onset else []
    for onset in undated + dated:
        if onset not in covered:
            _reserve(strongest_per_onset[onset])
    for entry in pool:
        _reserve(entry)
    shown = sorted(chosen.values(), key=lambda entry: entry.key)
    return shown


def _onset_label(row: RegressionRow) -> str:
    """The one grouping key used for a row's step onset — what the table
    reserves a place for, so every step inside a containing window is
    represented."""
    return row.verdict.onset_run_date or _UNKNOWN_ONSET


def _row_window(row: RegressionRow) -> tuple[str, str]:
    """A row's *own* tightest change window, which can be narrower than the
    comment's: a metric that settled later carries a later last-accepted
    release, and the step is only bounded by the pair the metric itself
    measured."""
    v = row.verdict
    return (v.last_accepted_run_date or "", v.onset_run_date or "")


def _window_cell(base: str, onset: str) -> str:
    """One row's change window as the table shows it. An unbounded base is
    written as an upper bound rather than invented, matching
    :func:`_window_line`."""
    onset_text = f"`{_cell(onset or _UNKNOWN_ONSET)}`"
    return f"`{_cell(base)}` → {onset_text}" if base else f"≤ {onset_text}"


def _alert(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    min_score: float,
    n_regressions: int | None = None,
    n_scopes: int | None = None,
    scores: tuple[list[float], list[float]] | None = None,
) -> str:
    """The headline claim as a GitHub warning alert: what the benchmarks
    measured, and how strongly a model ties it to this pull request.

    It opens with the measurement and only then estimates, because they are two
    different kinds of statement and the comment is careful to keep them apart
    everywhere else. The estimate is named as one — so a reader who stops at the
    alert stops at percentages rather than at an unqualified accusation — and the
    model behind it is said out loud, matching what the assessment below calls
    itself.

    With the lineage's identity union passed in, the measured totals and score
    clauses describe that union. A reconfirmed identity keeps only its newest
    published attribution.

    A peak alone is misleading, so each clause pairs it with *reach*: one row at
    95% out of forty reads very differently from thirty-eight of them, and the
    count of rows at or above the threshold is what tells them apart. (A window
    of one regression has no reach to report, and says the one score.) The
    threshold is named rather than summarised as "certain": it is a configured
    number (``min_score``), the same one that decided this comment exists at all,
    and a reader who thinks it is set wrong can go and see what it is set to.

    Every number is attributed to whoever produced it (:func:`_scored`), in one
    clause per model. A partial review is the case that makes this matter: the
    reviewer answers one row at 20%, the comment survives on another row the
    ranker left at 91%, and a single blended sentence would credit the reviewer
    with a claim it did not make while contradicting the count it did. Hence
    "likely contributor" is claimed only by a model whose own scores actually
    reach the threshold.

    A window nothing was scored against says nothing in the second sentence
    rather than reaching for a number — the rows are still real, and the table
    is where their states are spelled out."""
    # Rows count regressions; `scopes` counts (detector, platform, sample) run
    # groups. Keep both axes in the sentence: several metrics can regress in one
    # scope, while the scope count states how broadly those regressions reached.
    # Not called "configuration": everywhere else that word means the sweep
    # label, which is what the table's Config column holds.
    # A count is only readable next to the population it counts, and this
    # comment states two: the union over every report the lineage has covered,
    # which only :func:`_render` passes in, and tonight's report,
    # which is what the fallback below counts and what the line under the
    # table counts per report (:func:`_reports_line`). Each names its own, so the two
    # can be read side by side without guessing which is which.
    cumulative = n_regressions is not None
    n_regressions = len(rows) if n_regressions is None else n_regressions
    n_scopes = len(plan.scopes) if n_scopes is None else n_scopes
    where = (
        "in the reports covering this PR's change window" if cumulative
        else "in this PR's change window"
    )
    if n_regressions == 1:
        what = f"a regression {where}"
    elif n_scopes == 1:
        what = (
            f"{_count(n_regressions, 'regression')} within one "
            f"detector/platform/sample scope {where}"
        )
    else:
        what = (
            f"{_count(n_regressions, 'regression')} across "
            f"{_count(n_scopes, 'detector/platform/sample scope')} {where}"
        )
    measured = f"k4Bench's nightly benchmarks confirmed {what}."
    reviewed, carried = scores or _scored(rows, attribution)
    if not reviewed and not carried:
        return f"> [!WARNING]\n> {measured}"
    clauses = []
    if reviewed:
        clauses.append(_reviewer_clause(reviewed, min_score=min_score))
    if carried:
        clauses.append(_ranker_clause(
            carried, min_score=min_score,
            population=n_regressions,
            # Every regression the review did not answer, including the ones
            # nobody scored: the clause counts *within* that set, so it has to
            # know how big the set is. Counted against ``n_regressions`` and
            # never against ``rows``: with the union passed in, ``reviewed``
            # describes the whole cumulative lineage while ``rows`` is only tonight's
            # plan, and differencing the two populations can go negative — a
            # clause about "-1 regressions it did not score".
            unreviewed=max(0, n_regressions - len(reviewed)) if reviewed else 0,
        ))
    return f"> [!WARNING]\n> {measured} {' '.join(clauses)}"


def _reviewer_clause(reviewed: list[float], *, min_score: float) -> str:
    """What the cross-configuration review found, in its own voice.

    "A likely contributor" is a claim about *these* scores, so a review that put
    nothing at or above the threshold does not make it — it reports what it
    found, and any surviving claim is left to the clause that earns it."""
    over = sum(1 for likelihood in reviewed if likelihood >= min_score)
    scored = _count(len(reviewed), "regression")
    highest = _pct(max(reviewed))
    if not over:
        return (
            f"The AI reviewer scored {scored} and put none at "
            f"{_pct(min_score)} or above (highest {highest})."
        )
    lead = "The AI reviewer estimates this PR is a likely contributor:"
    if len(reviewed) == 1:
        return (
            f"{lead} it scored the one regression at {highest}, at or above "
            f"the {_pct(min_score)} threshold."
        )
    verb = "is" if over == 1 else "are"
    return (
        f"{lead} {over} of the {scored} it scored {verb} attributed to it at "
        f"{_pct(min_score)} or above, the highest at {highest}."
    )


def _ranker_clause(
    carried: list[float], *, min_score: float, unreviewed: int,
    population: int,
) -> str:
    """What the per-configuration pass scored, in *its* voice.

    Two shapes, because the rows mean different things in each. Alone
    (``unreviewed`` zero — no reviewer configured), this is the whole estimate.
    After a review, these are rows the review did not answer, and *unreviewed*
    is how many such rows there are in total: the ones with no score at all
    belong to neither model, so they are named as a difference rather than
    counted into a clause that would then claim scores for them."""
    over = sum(1 for likelihood in carried if likelihood >= min_score)
    scored = _count(len(carried), "regression")
    population_text = _count(population, "regression")
    highest = _pct(max(carried))
    if unreviewed:
        which = (
            "The one regression it did not score" if unreviewed == 1
            else f"Of the {_count(unreviewed, 'regression')} it did not score, "
                 f"{len(carried)}"
        )
        if len(carried) == 1:
            # One row's count, reach and maximum are three ways of saying one
            # number, and the threshold is already named in the clause before.
            return f"{which} keeps a first-pass ranker score of {highest}."
        reach = (
            f"{over} of them at {_pct(min_score)} or above"
            if over else f"none at {_pct(min_score)} or above"
        )
        return (
            f"{which} keep a first-pass ranker score, {reach} "
            f"(highest {highest})."
        )
    if not over:
        subject = scored if len(carried) == population else f"{scored} of {population_text}"
        return (
            f"The AI ranker scored {subject} and put none at {_pct(min_score)} "
            f"or above (highest {highest})."
        )
    lead = "The AI ranker estimates this PR is a likely contributor:"
    if len(carried) == population == 1:
        return (
            f"{lead} it scored the one regression at {highest}, at or above "
            f"the {_pct(min_score)} threshold."
        )
    verb = "is" if over == 1 else "are"
    return (
        f"{lead} {over} of {population_text} {verb} attributed to it at "
        f"{_pct(min_score)} or above, the highest at {highest}."
    )


def _scored(
    rows: list[RegressionRow], attribution: Attribution | None
) -> tuple[list[float], list[float]]:
    """``(what the review scored, what only the first pass scored)``.

    The table and the withdrawal gate read a row's *effective* likelihood
    (:func:`_likelihood`); the alert cannot, because a count is worthless without
    knowing whose judgement it holds. On a wide window the split is routine
    rather than rare — everything past
    :data:`~k4bench.blame.attribute._MAX_ATTRIBUTED_ROWS` is first-pass-only by
    construction.

    Rows with no likelihood at all (``not_candidate``, or unscored by both
    passes) appear in neither list — the alert claims nothing about them."""
    reviewed: list[float] = []
    carried: list[float] = []
    for row in rows:
        likelihood = _likelihood(row, attribution)
        if likelihood is None:
            continue
        if attribution is not None and row.fact_id in attribution.likelihoods:
            reviewed.append(likelihood)
        else:
            carried.append(likelihood)
    return reviewed, carried


def _window_line(plan: CommentPlan) -> str:
    """The change window as a single caption line — the Key4hep release dates
    that bound the step, shared by every row below. An open-ended window says so
    here, where the dates it is missing one of are."""
    if plan.base_release:
        window = f"`{plan.base_release}` → `{plan.onset_release}`"
    else:
        window = (
            f"≤ `{plan.onset_release}` — open-ended: no earlier settled "
            "measurement bounds it"
        )
    # The dates are Key4hep release dates, not calendar dates the benchmark ran:
    # said in the label so a reader does not read them as run days.
    return f"**Change window** (Key4hep releases): {window}"


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


def _assessment(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    rendered: list[RegressionRow] | None = None,
) -> str | None:
    """The model's reasoning as a labelled blockquote — the label is where the
    comment openly says an AI made this call.

    With a cross-configuration review, that is its summary: it saw every
    configuration that moved and every one that did not, so it is the account
    that can actually explain the pattern. It does not necessarily account for
    every row in the table, though — a very wide window offers the review only
    its largest movements, and a reply may answer only some of what it was
    offered — so a partial review says which rows it covered. A narrative reading
    "this PR does not fit the affected set" printed above rows still carrying an
    unrelated 91% must not look like it was talking about them.

    With no reviewer configured at all — the only way a comment is rendered
    without a review (:func:`build_comments`) — the comment quotes the
    per-configuration ranker's one-liner for its strongest row, and then it
    claims "the most likely cause" only when this PR outranks every other
    candidate — a comment can fire on any score above ``min_score``, and a PR the
    ranker placed second must not be told it came first. Nothing is rendered when
    neither model explained itself: an unexplained score is not comment-worthy
    prose, and it already stands in the table."""
    if attribution is not None:
        text = _sentences(attribution.summary, _MAX_SUMMARY_CHARS)
        if text:
            return (
                f"\n> 🤖 **The AI reviewer's assessment:** {text}"
                + _evidence_note(attribution)
                + _coverage_note(rendered if rendered is not None else rows, attribution)
            )
        return None
    lead = rows[0] if rows else None
    text = _one_line(lead.scope_reason, _MAX_DESCRIPTION_CHARS) if lead else ""
    if not text or lead.scope_score is None:
        return None
    # Only judged candidates can be outranked: an unscored one is not behind
    # this pull request, it is simply unknown, so it cannot support a claim to
    # first place either.
    outranks_all = all(
        other.ranked and other.score < lead.scope_score
        for other, _scope in plan.others.values()
    )
    claim = "the most likely" if outranks_all else "a likely"
    return (
        f"\n> 🤖 **The AI ranker judged this PR {claim} cause of the "
        f"regression:** {text}"
    )


def _evidence_note(attribution: Attribution) -> str:
    """One line when the review could not judge whether the movements are real.

    ``likely_noise`` never reaches here — it withdraws the comment outright
    (:func:`build_comments`) — so the only case left is
    ``insufficient_evidence``: the benchmark history behind these steps is too
    short for the review to corroborate them. That does *not* overturn the
    detector's own confirmation, which is a two-strike statistical judgement over
    the measurements themselves and is why the comment exists at all. It does
    change how much weight a reader should give the paragraph above it, so it is
    said out loud rather than left for nobody to know.

    ``real_change`` prints nothing. A caveat that appears on every comment is one
    every reader learns to skip, which is how the one that matters gets skipped
    too."""
    assessment = attribution.assessment
    if assessment is None or assessment.verdict != "insufficient_evidence":
        return ""
    return (
        "\n>\n> <sub>The benchmark history behind these steps was too short for "
        "the review to judge whether they are a real change; the regressions "
        "themselves are confirmed by the nightly detector.</sub>"
    )


def _coverage_note(
    rows: list[RegressionRow], attribution: Attribution
) -> str:
    """What the assessment above does *not* cover, when it covers less than all.

    Measured over the *current* rows this comment renders, not over the whole
    window — and the wording says "current", because a table carrying rows
    retained from earlier versions shows more rows than this count covers and a
    bare "of the four shown" would not add up for the reader counting them. A
    retained row is outside the count for the same reason it is outside the
    review: it is not part of tonight's evidence. A row goes unreviewed either
    because the window carried more
    regressions than the prompt offers or because the reply simply skipped it,
    and it then keeps whatever the first pass left it with — a score, or one of
    the states that has none (:func:`_likelihood`). That only misleads a reader
    who can see the row: a caveat about rows nobody renders warns of a
    discrepancy nothing on the page shows, and on a wide window a single dropped
    row would print one every night. The wording avoids promising a score for
    those rows, since plenty of them have none, and stays agreement-free so one
    unreviewed row reads as well as twenty."""
    unreviewed = sum(1 for row in rows if row.fact_id not in attribution.likelihoods)
    if not unreviewed:
        return ""
    return (
        f"\n>\n> <sub>This assessment covers "
        f"{_count(len(rows) - unreviewed, 'regression')} of the {len(rows)} "
        "current regressions shown; anything it did not answer keeps its "
        "first-pass state, and its score where there is one.</sub>"
    )


def _row_links(
    plan: CommentPlan, rows: list[RegressionRow], dashboard_url: str | None
) -> dict[str, str]:
    """``{fact id: href}`` for every row the table will render.

    Each row goes to its *own* regression pinned in the dashboard's Stack
    Changes view (:func:`~k4bench.regression.render.regression_href`), where the
    metric's trend, its onset and the window's package diff sit in one place —
    the reader's question is "did my change do this?", and that view is the one
    that answers it without a second click. A row whose verdict cannot be pinned
    (no onset identity) falls back to its configuration's Regressions view, which
    at least lands on the right window.

    The hrefs are ~400 characters each and a night can carry hundreds of rows;
    writing one inline per row is what pushes a wide night past GitHub's
    65,536-character comment limit, where a comment is rejected outright rather
    than truncated. Markdown *reference* links move each URL into a definition at
    the bottom of the body, and only rendered rows get one. The labels are the
    rows' own fact ids — already assigned in identity order — so a body is stable
    across nights and a re-render triggers no edit."""
    if not dashboard_url:
        return {}
    links = {}
    for row in rows:
        href = regression_href(
            dashboard_url,
            verdict=row.verdict,
            base_release=plan.base_release, onset_release=plan.onset_release,
        ) or window_href(
            dashboard_url,
            detector=row.verdict.detector, platform=row.verdict.platform,
            sample=row.verdict.sample,
            base_release=plan.base_release, onset_release=plan.onset_release,
            stack=row.stack,
        )
        if href:
            links[row.fact_id] = href
    return links


def _link_definitions(links: dict[str, str]) -> str | None:
    """The reference-link definitions :func:`_row_links` promised.

    Markdown renders these as nothing at all, so they sit at the end of the body
    where they interrupt no one."""
    if not links:
        return None
    return "\n" + "\n".join(
        f"[{label}]: {href}" for label, href in sorted(links.items())
    )


def _table_head(*, show_window: bool, show_reproduce: bool) -> list[str]:
    """The table header, including the optional platform, change-window and
    reproduce columns."""
    header = ["Metric", "Detector"]
    align = [":---", ":---"]
    if _SHOW_PLATFORM_COLUMN:
        header.append("Platform")
        align.append(":---")
    header += ["Sample", "Config"]
    align += [":---", ":---"]
    if show_window:
        header.append("Change window")
        align.append(":---")
    header.append("Change")
    align.append("---:")
    header.append("Attribution")
    align.append("---:")
    if show_reproduce:
        header.append("Reproduce")
        align.append(":---")
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
    ]


def _row_line(
    row: RegressionRow,
    attribution: Attribution | None,
    links: dict[str, str],
    *,
    show_window: bool,
    recipe: str,
) -> str:
    """One regression as a table row — the same cells whichever ordering placed
    it, so both tables read identically row for row.

    The dashboard link hangs off the **metric** cell because that is what it
    opens, and only when :func:`_row_links` emitted a definition for this row.
    Metric and configuration keep their raw names: those are what the dashboard
    labels the series with."""
    v = row.verdict
    metric = (
        f"`{_cell(v.metric)}`"
        + (f" · {_cell(v.sub_detector)}" if v.sub_detector else "")
    )
    cells = [
        f"[{metric}][{row.fact_id}]" if row.fact_id in links else metric,
        _cell(v.detector),
    ]
    if _SHOW_PLATFORM_COLUMN:
        cells.append(_cell(pretty_platform(v.platform)))
    cells += [
        _cell(compact_sample(v.sample)),
        f"`{_cell(v.label)}`",
    ]
    if show_window:
        cells.append(_window_cell(*_row_window(row)))
    cells.append(_change_cell(v.pct_change))
    cells.append(_likelihood_cell(row, attribution))
    if recipe is not None:
        cells.append(_recipe_cell(recipe))
    return "| " + " | ".join(cells) + " |"


def _past_row_line(
    row: RetainedRow,
    past_links: dict[tuple, tuple[str, str]],
    *,
    show_window: bool,
    recipe: str | None,
) -> str:
    """One retained row with its last published movement and likelihood."""
    metric = (
        f"`{_cell(row.metric)}`"
        + (f" · {_cell(row.sub_detector)}" if row.sub_detector else "")
    )
    reference = past_links.get(_retained_identity(row))
    cells = [
        f"[{metric}][{reference[0]}]" if reference else metric,
        _cell(row.detector),
    ]
    if _SHOW_PLATFORM_COLUMN:
        cells.append(_cell(pretty_platform(row.platform)))
    cells += [
        _cell(compact_sample(row.sample)),
        f"`{_cell(row.label)}`",
    ]
    if show_window:
        cells.append(_window_cell(*_retained_window(row)))
    cells += [
        _change_cell(row.pct),
        _past_likelihood_cell(row),
    ]
    if recipe is not None:
        # The recipe published on the night this row was last confirmed. It is
        # carried in the snapshot rather than rebuilt, because only the night
        # that published it can vouch for the file being there.
        cells.append(_recipe_cell(recipe))
    return "| " + " | ".join(cells) + " |"


def _recipe_cell(url: str) -> str:
    """The link to this row's own published recipe.

    Empty when nothing was published for the row, rather than a link to
    commands for a different configuration than the one whose line the reader
    followed."""
    return f"[🔁 recipe ↗]({_cell(url)})" if url else ""


def _past_likelihood_cell(row: RetainedRow) -> str:
    """The likelihood recorded on the row's last published version — the same
    three shapes a current cell has, so an unscored or not-a-candidate snapshot
    can never resurface wearing a number. A score that drifted on a night the
    digest held unchanged was never written, so this is the last *published*
    number rather than the last one the ranker produced."""
    if row.state == "not_candidate":
        return _NOT_A_CANDIDATE
    if row.likelihood is None:
        return _UNSCORED
    return _pct(row.likelihood)


def _past_links(
    rows: list[RetainedRow], dashboard_url: str | None
) -> dict[tuple, tuple[str, str]]:
    """``{identity: (label, href)}`` for the retained rows the table renders.

    Reconstructed from the snapshot's validated structured fields through the
    shared link helper — never from a stored URL. The link pins the archived
    Regressions view for the row's own window, stack and last-reported night,
    which is where a regression that tonight's report no longer confirms can
    still be read. Both run ids are passed: they are what qualifies a
    same-release window, whose releases alone name several different windows
    and would land the reader on whichever the view ordered first
    (:func:`~k4bench.regression.render.window_token`). Labels are ``h1``,
    ``h2``, … in identity order, disjoint from the current rows' fact ids."""
    if not dashboard_url:
        return {}
    links: dict[tuple, tuple[str, str]] = {}
    ordered = sorted(rows, key=_retained_identity)
    for index, row in enumerate(ordered, start=1):
        href = window_href(
            dashboard_url,
            detector=row.detector,
            platform=row.platform,
            sample=row.sample,
            base_release=row.base_release,
            onset_release=row.onset_release,
            base_run=row.base_run or None,
            onset_run=row.onset_run or None,
            stack=row.stack or None,
            report_night=row.last_reported,
        )
        if href:
            links[_retained_identity(row)] = (f"h{index}", href)
    return links


def _table(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    shown: list[_PoolRow],
    links: dict[str, str],
    past_links: dict[tuple, tuple[str, str]],
    dashboard_url: str | None,
    reproduce: Reproducers | None = None,
    observations: Sequence[CommentObservation] = (),
) -> str:
    """The selected rows of the window, most likely first — tonight's confirmed
    regressions and any retained rows that outrank them.

    Each row carries its **own tightest change window** rather than the
    comment's: a metric that settled later entered on a narrower range, and
    pinning the step to the pair that metric actually measured is what makes a
    row checkable.

    One table rather than one section per configuration: which configurations
    moved — and, read against the review's summary, which did not — is the
    substance of the claim, and a reader weighing it needs to see the pattern at
    once. Selection reserves the strongest evidence and one row per represented
    onset (:func:`_selected_rows`); a night wider than the cap says
    how many more there are in one line rather than pasting or folding them: a detector-removal sweep
    can confirm three hundred near-identical rows, which no one reads and GitHub
    will not accept — and a collapsed ``<details>`` block would not help, since
    collapsed Markdown still counts against the body limit. Every shown row
    links into the dashboard, which is where the complete set — and any
    re-sorting — lives.

    Rows are ordered by attribution likelihood, not claimed as attributed: the
    table deliberately keeps rows the review scored *down*, and a 20% row under a
    heading claiming attribution would read as an accusation the numbers next to
    it deny. That ordering can push the window's largest movement past the cap;
    the line below counts what each report carried and links all of it
    (:func:`_reports_line`).

    A retained historical row keeps the movement, likelihood and recipe last
    published for it; the cumulative alert uses the same newest-published
    rule."""
    urls = reproduce.urls if reproduce is not None else {}
    recipes = {
        id(entry): (
            urls.get(_row_identity(entry.current), "")
            if entry.current is not None
            else entry.past.recipe
        )
        for entry in shown
    }
    # The column exists only when at least one shown row has somewhere to send
    # the reader; an all-empty column is a header and nothing else.
    show_recipe = any(recipes.values())
    # Shown whenever a *rendered* row's own window is not simply the comment's
    # — a narrower base, a newer onset, or a retained row carrying a window of
    # its own. Decided over the rows that render rather than the whole plan: a
    # narrower window carried only by a row the cap cut is not on the page, and
    # a column repeating the header's window once per line is a column of
    # nothing.
    show_window = {
        _row_window(entry.current) if entry.current is not None
        else _retained_window(entry.past)
        for entry in shown
    } != {(plan.base_release or "", plan.onset_release)}
    lines = [
        "",
        # A bold caption, not a Markdown heading: it reads at the same size as the
        # window line above it (:func:`_window_line`), so the two captions the
        # reader scans first sit at one level rather than the table's shouting over
        # the window that scopes it.
        "📊 **Regressions in this window, ranked by AI-based attribution likelihood**",
        "",
        *_table_head(show_window=show_window, show_reproduce=show_recipe),
    ]
    for entry in shown:
        recipe = recipes[id(entry)] if show_recipe else None
        if entry.current is not None:
            lines.append(_row_line(
                entry.current, attribution, links,
                show_window=show_window, recipe=recipe,
            ))
        else:
            lines.append(_past_row_line(
                entry.past, past_links, show_window=show_window, recipe=recipe,
            ))
    shown_current = sum(1 for entry in shown if entry.current is not None)
    reports = _reports_line(
        len(rows), len(shown), shown_current, observations,
        dashboard_url=dashboard_url,
    )
    if reports:
        lines += ["", reports]
    return "\n".join(lines)


#: How many reports the line under the table names before it starts counting
#: them instead. Three sets of counts and links is what stays readable as one
#: sentence; a lineage longer than that is what the observation history is for.
_MAX_NAMED_REPORTS = 3


def _reports_line(
    n_rows: int,
    shown: int,
    shown_current: int,
    observations: Sequence[CommentObservation],
    *,
    dashboard_url: str | None,
) -> str | None:
    """What each report in this lineage carried, as one line under the table.

    A wide night is not folded into a second copy of the table: a
    detector-removal sweep confirms three hundred near-identical rows, which no
    one reads whether or not they are behind a disclosure, and which GitHub will
    not accept in one comment anyway. So the table is a selection, and this line
    is what says so — per report, because the comment's evidence is assembled
    from several overlapping nightly snapshots and a single total would hide
    which one each count came from.

    Every date is its own link into that night's full report, where the complete
    set and every re-sorting of it live. Only the dates are linked — with the
    arrow that conventionally means "this opens somewhere else" — so the counts
    read as prose and the click target is the thing being opened.

    The rows counted as shown are every row the table drew, current and retained
    alike: the reader counts lines on the page, not the bookkeeping behind them.
    """
    counts = _report_counts(observations)
    # Nothing to add when one report's regressions all reached the table: the
    # line would restate the rows immediately above it.
    if not counts or (len(counts) < 2 and shown_current >= n_rows):
        return None
    named = []
    for index, (night, regressions) in enumerate(counts[:_MAX_NAMED_REPORTS]):
        href = nightly_report_href(dashboard_url, night)
        where = f"[{night} report ↗]({href})" if href else f"{night} report"
        # The noun is carried once, by the leading clause; repeating it on every
        # report turns one sentence into a list of identical clauses.
        noun = " of regressions" if index == 0 else ""
        if regressions is None:
            # Converged sibling windows whose overlap this state cannot resolve
            # (:func:`_report_counts`). The report is still named — its rows are
            # in the table above — but no number is invented for it.
            count = f"an unspecified number{noun}"
        elif index == 0:
            count = f"**{_count(regressions, 'regression')}**"
        else:
            count = f"**{regressions}**"
        named.append(f"{count} in the {where}")
    line = ", ".join(named)
    if len(counts) > _MAX_NAMED_REPORTS:
        line += (
            f", and {_count(len(counts) - _MAX_NAMED_REPORTS, 'earlier report')}"
        )
    drawn = (
        "the most likely one is shown above" if shown == 1
        else f"the {shown} most likely are shown above"
    )
    return f"{line} — {drawn}."


def _others_section(plan: CommentPlan, dashboard_url: str | None = None) -> str:
    """The rest of the candidates scored across this window, with their
    likelihoods — the reader needs to see what else was in the frame to weigh
    the claim against this PR, including the case where nothing else was.

    Collapsed by default, but the summary carries the strongest competing score
    without being opened: how far ahead this PR sits — the difference between a
    ranking that picked it and one that barely preferred it — belongs in front of
    a reader who expands nothing. The table is capped at
    :data:`_MAX_OTHER_CANDIDATES`, with any surplus counted rather than pasted.

    The candidates are named, never linked (:func:`_pr_ref`), so the section
    closes with the window's package diff — where the complete field, and every
    package behind it, can be read without notifying anyone. It is offered, not
    claimed as their provenance: see :func:`_package_diff_line`."""
    others = _sorted_others(plan)
    diff = _package_diff_line(plan, dashboard_url)
    if not others:
        return "\n".join([
            "",
            "> [!NOTE]",
            "> This was the only pull request found across every tracked "
            "package that changed in this window."
            + (f" {diff}" if diff else ""),
        ])

    shown = others[:_MAX_OTHER_CANDIDATES]
    strongest = _closest_candidate(plan)
    headline = (
        f"highest {_pct(strongest.score)}" if strongest is not None
        else "none of them scored by the ranker"
    )
    lines = [
        "",
        "<details>",
        "<summary><b>📋 Other pull requests in this window</b> — "
        f"{_count(len(others), 'candidate')}, {headline}"
        "</summary>",
        "",
        "| Pull request | Likelihood |",
        "|:---|---:|",
        *(
            f"| {_pr_ref(c)} — {_cell(_one_line(c.title, 80))} | "
            f"{_candidate_score_cell(c)} |"
            for c, _scope in shown
        ),
    ]
    if len(others) > len(shown):
        lines += ["", f"_…and {_count(len(others) - len(shown), 'more candidate')}._"]
    if diff:
        lines += ["", diff]
    lines += ["", "</details>"]
    return "\n".join(lines)


def _package_diff_line(plan: CommentPlan, dashboard_url: str | None) -> str | None:
    """The window's package diff, one link per build platform that contributed
    a row — offered as somewhere to look, never as the candidates' provenance.

    The distinction is load-bearing. :func:`_record_others` folds in every
    entry's candidates with no window filter at all, while only an entry
    measuring *exactly* this comment's window describes this window's release
    diff (:func:`_confirmed_rows`): a row can enter on a narrower range of its
    own, and :attr:`CommentPlan.packages_unavailable_on` exists to name the
    platforms whose diff for this window was never read. So a candidate here can
    have been found in a range whose package set is not the one behind this
    link, and "every candidate came from" would assert a provenance the pipeline
    does not establish.

    Per platform because provenance is recorded that way: a plan is keyed by
    pull request and window and never by platform, so its rows can span several,
    while the release diff behind them is kept separately for each
    (:class:`CommentPlan`). Two platforms' diffs are two measurements.

    Within a platform the scope parameters only choose which regression ledger
    the Stack Changes view draws underneath the diff, so the lowest row identity
    is as good a landing place as any — picked by identity rather than by list
    order so the sentence is byte-stable across runs."""
    platforms = sorted({row.verdict.platform for row in plan.rows})
    links = []
    for platform in platforms:
        if any(
            row.verdict.base_platform != row.verdict.onset_run_platform
            for row in plan.rows if row.verdict.platform == platform
        ):
            # Stack Changes compares releases of one platform; migration
            # rows already link to their window in the Regressions view.
            continue
        lead = min(
            (row.verdict for row in plan.rows if row.verdict.platform == platform),
            key=_verdict_identity,
        )
        href = stack_changes_href(
            dashboard_url,
            detector=lead.detector,
            platform=platform,
            sample=lead.sample,
            base_release=plan.base_release,
            onset_release=plan.onset_release,
        )
        if not href:
            continue
        label = (
            "package diff for this window" if len(platforms) == 1
            else _cell(pretty_platform(platform))
        )
        links.append(f"[{label} ↗]({href})")
    if not links:
        return None
    if len(links) == 1 and len(platforms) == 1:
        return f"Inspect the {links[0]}."
    return (
        "Inspect this window's package diffs, one per platform: "
        f"{', '.join(links)}."
    )


def _pr_ref(candidate: CandidatePR) -> str:
    """A competing candidate named as ``owner/repo#123``, inert on purpose.

    These pull requests are *not* the ones being commented on — they are the
    field the ranking was made against — and GitHub turns any reference to them,
    a bare ``owner/repo#123`` or a link carrying their URL, into a cross-
    reference on their own timeline, notifying everyone subscribed there. A PR
    that was merely a candidate should not collect a notification every time
    another window implicates someone else, so the number is broken with a
    zero-width space: unchanged to a reader, unparsed by GitHub, and
    unclickable. Whoever wants the full field has the package-diff link this
    section closes with (:func:`_package_diff_line`)."""
    zwsp = "​"  # U+200B zero-width space
    return _cell(f"{candidate.repo}#{zwsp}{candidate.number}")


def _closest_candidate(plan: CommentPlan) -> CandidatePR | None:
    """The strongest competing candidate the first pass actually scored.

    :func:`_sorted_others` is judged-first, so the leading entry is that
    candidate when there is one. A field nobody scored has no closest
    candidate: an unscored rival is an unknown, not a near miss, and neither the
    competing-field headline nor :func:`_crowded_note` may quote a percentage no
    model produced."""
    others = _sorted_others(plan)
    top = others[0][0] if others else None
    return top if top is not None and top.ranked else None


def _crowded_note(plan: CommentPlan) -> str | None:
    """Said out loud when the ranking does not clearly favour this PR — in
    words, rather than leaving the reader to subtract two numbers. Nothing is
    said when no competing candidate was scored (:func:`_closest_candidate`): an
    unscored rival is not a near miss, it is an unknown, and no gap can be
    computed from it.

    Which way the preference runs is the whole point, so the note is
    direction-aware. A PR the ranker placed *behind* another candidate is told so
    however wide the gap is: that is the single most important qualifier on a
    comment accusing it. A PR that is ahead hears about it only when the lead is
    thin (``_CROWDED_SPREAD``) — a caveat printed on every comfortable night is
    wallpaper, and the score and the summary line already say what a comfortable
    lead looks like.

    It renders directly under the assessment, where the claim it qualifies is
    made — not down beside the competing field, where a reader has already
    finished reading the accusation."""
    closest = _closest_candidate(plan)
    if closest is None:
        return None
    delta = plan.top_score - closest.score
    if not math.isfinite(delta) or delta > _CROWDED_SPREAD:
        return None
    # Prose has to agree with the percentages sitting right above it, not with
    # the raw scores behind them: :func:`_pct` rounds each score independently,
    # so a sub-point raw gap can render as a one-point *displayed* gap and vice
    # versa. Rounding both scores the same way :func:`_pct` does, then
    # differencing, keeps the two in lockstep.
    mine = int(round(plan.top_score))
    theirs = int(round(closest.score))
    display_delta = mine - theirs
    points = abs(display_delta)
    if display_delta < 0:
        return (
            f"\n_The closest other candidate scored {_count(points, 'point')} "
            "**higher** than this PR — the ranker's preference in this window "
            "runs against it, not for it._"
        )
    if points == 0:
        separation = "Nothing separates this PR from the closest other candidate"
    else:
        verb = "separates" if points == 1 else "separate"
        separation = (
            f"Only {_count(points, 'point')} {verb} this PR from the closest "
            "other candidate"
        )
    return (
        f"\n_{separation} — the ranker is expressing a weak preference here, "
        "not a clear pick._"
    )


#: What a likelihood cell says when no model ever scored that pair. Short enough
#: for a table cell, and a phrase rather than a number: "0%" would claim a
#: judgement, and an empty cell would look like a rendering bug.
_UNSCORED = "_not scored_"

#: And what it says when the pipeline knows why there is no score — this pull
#: request is not in the commit range behind that regression at all. Stated
#: plainly because it is the one cell in the table that argues for the reader.
_NOT_A_CANDIDATE = "_not a candidate_"


def _likelihood_cell(row: RegressionRow, attribution: Attribution | None) -> str:
    """A row's attribution cell.

    A row nobody scored says so in words. Those rows are in the table because
    the window is what the comment is about — a regression this pull request was
    not even a candidate for is evidence a reader should see, and it must not
    arrive wearing a percentage."""
    if row.scope_state == "not_candidate":
        return _NOT_A_CANDIDATE
    likelihood = _likelihood(row, attribution)
    return _UNSCORED if likelihood is None else _pct(likelihood)


def _candidate_score_cell(candidate: CandidatePR) -> str:
    """A competing candidate's likelihood, or that nobody gave it one."""
    return _pct(candidate.score) if candidate.ranked else _UNSCORED


def _change_cell(pct_change: float | None) -> str:
    """A metric's step as a signed percentage with a direction marker.
    ``pct_change`` is a fraction on :class:`MetricVerdict`, matching the
    report's own formatting. The markers are monochrome glyphs, not coloured
    ones: they state the sign of the step and nothing more, because k4Bench
    cannot tell whether a step in either direction is welcome — a drop can be
    an optimization or work that is no longer being done
    (:class:`~k4bench.regression.models.Direction`). The gap between marker and
    number is non-breaking so a narrow column wraps the cell as a whole rather
    than stranding the marker on its own line.

    Emphasised with ``**``, so every caller must be somewhere GitHub renders
    Markdown — inside a raw ``<summary>`` the asterisks would reach the reader
    as asterisks."""
    if pct_change is None or not math.isfinite(pct_change):
        return "—"
    arrow = "▲" if pct_change >= 0 else "▼"
    return f"{arrow}&nbsp;**{pct_change:+.1%}**"
