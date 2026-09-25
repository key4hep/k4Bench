"""Review one pull request against a whole change window, with a language model.

:mod:`k4bench.blame.rank` answers *"which of these pull requests caused this
configuration's regressions?"* — once per ``(detector, platform, sample)`` run
group. That is the right question for the dashboard and the sidecar, where every
regression row wants a likelihood scoped to the run it was measured on.

It is the wrong question for a pull-request comment. The comment makes a claim in
someone else's repository about one specific change, and the strongest evidence
for or against that claim is *cross-configuration*: the same step hitting ALLEGRO
and not IDEA, under the same sample and the same platform, says something no
per-configuration call can see, because no per-configuration call is ever shown
the other configurations. So this module asks the transposed question — *"which of
this window's regressions did **this** pull request cause?"* — once per
``(pull request, change window)``, and it is shown the whole window rather than
one slice of it: the confirmed regressions across every detector, sample,
platform and benchmark configuration; the configurations that ran the same
window and did *not* confirm; the release's package diff; and the other pull
requests that landed in the window, with their diffs and the first pass's
judgement of them.

The prompt is bounded, and honestly so. Every regression of the window is
*collected* — that is what makes the exculpatory rows visible — but only the
:data:`_MAX_ATTRIBUTED_ROWS` largest movements and the
:data:`MAX_COMPETITORS` strongest competitors are put in front of the model. A
row past that cap is never scored by this pass: it keeps the first pass's
likelihood, it cannot be answered even if the model guesses its id, and the
comment states how much of its table the review actually covered.

The guarantees mirror :mod:`k4bench.blame.rank`'s, because the failure modes are
the same and the consequences here are larger:

* **Only-echo.** :func:`_parse_attribution` drops any row id the prompt did not
  offer (:func:`_attributed_facts`), so a regression the model invented — or one
  it guessed the id of past the row cap — is structurally impossible to surface.
  A row the model simply omitted keeps the first pass's score: an unanswered row
  is not a zero, and the comment says how many rows the review covered.

* **Honest failure.** Every failure path — HTTP error, timeout, malformed JSON,
  a reply with no usable rows — returns ``None``, and this module never decides
  what that costs. :func:`k4bench.blame.comment.build_comments` does: with a
  reviewer configured, ``None`` means *no comment that night*, because a
  first-pass-only body posted now would share its facts digest with the reviewed
  body rendered later and could never be replaced by it. A blocked comment is
  recoverable tomorrow; a frozen degraded one is not, and both beat an invented
  one.

* **Narrowing at the target level.** This pass never causes a comment on a pull
  request selection did not already implicate: selection happens entirely on the
  first pass's scores (:mod:`k4bench.blame.comment`), and the only *outcome* this
  pass can add is withdrawal — a review that leaves every row under the threshold
  drops the comment. Within an already-selected pull request it is a full second
  opinion: an individual row's likelihood may go up as well as down, because a
  row the first pass judged blind to the other configurations is exactly what
  cross-configuration evidence exists to correct. What that cannot do is widen
  the bot's reach, which is the property being protected.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Literal, Protocol

from k4bench.blame.evidence import MetricHistory, ScopeOutcome
from k4bench.blame.geometry import DetectorTouch
from k4bench.blame.history import HistoricalPR
from k4bench.blame.llm import (
    MAX_OUTPUT_TOKENS,
    ChatClient,
    chat_client_from_env,
    extract_json,
    one_line,
    parse_score,
)
from k4bench.blame.prompt import (
    ASSESSMENT_RULE,
    ASSESSMENT_VALUES,
    HISTORICAL_ANALOGUE_RULE,
    NOISE_RULE,
    SCORE_BAND_RULE,
    UNTRUSTED_EVIDENCE_RULE,
    WEIGHING_RULE,
    allocate_diff_budget,
    body_block,
    diff_block,
    direction_phrase,
    format_files,
    historical_lines,
    history_block,
    history_clause,
    log_prompt_size,
    measurement_phrase,
    outcome_lines,
    platform_line,
    platform_switch_lines,
    region_lines,
    sample_line,
    window_phrase,
)
from k4bench.blame.summary import (
    EVIDENCE_HEADER,
    SWEEP_METRICS,
    other_scope_lines,
    representative,
    scope_evidence_lines,
    scope_name,
    sweep_table_lines,
    touch_lines,
)
from k4bench.blame.sweep import ScopeSweep
from k4bench.regression.models import RegionDelta

_log = logging.getLogger(__name__)


# ── The request/response contract ─────────────────────────────────────────────

#: What the first pass had to say about the reviewed pull request *in one
#: regression's own scope*. Four states, because they are four different pieces
#: of evidence and only one of them is a number:
#:
#: * ``"ranked"`` — the pull request was a candidate there and the first pass
#:   scored it. ``scope_score`` carries that score, 0/100 included.
#: * ``"not_candidate"`` — candidate discovery for that scope was complete and
#:   the pull request was not in it. Strong *exculpatory* evidence: the change
#:   is not in the commit range that produced this regression.
#: * ``"unranked"`` — it was a candidate, but the first pass returned no
#:   judgement about it (a partial ranking response). Unknown, not zero.
#: * ``"discovery_incomplete"`` — the candidate population or changed-file
#:   evidence for that scope is not known to be complete (a truncated or
#:   unavailable range, or no sidecar entry at all), so absence proves nothing
#:   and presence is not fully evidenced.
#:
#: Kept as explicit states rather than folded into a likelihood prior: three of
#: the four have no honest numeric value, and inventing one — 0 for "we never
#: asked" above all — is how unknown evidence turns into negative evidence.
ScopeCandidateState = Literal[
    "ranked", "not_candidate", "unranked", "discovery_incomplete",
]


@dataclass(frozen=True)
class RegressionFact:
    """One confirmed regression offered for attribution — one row of the comment's
    table, and one row the model must score.

    ``id`` is the opaque handle the model echoes back (``"r1"``, ``"r2"``, …)
    rather than a re-typed six-field identity: a model that mis-spells a detector
    name loses the row, while a model that mis-types ``"r7"`` is caught by
    only-echo.

    ``scope_state`` is what the first pass knew about the reviewed pull request
    *here* (see :data:`ScopeCandidateState`), and ``scope_score``/
    ``scope_reason`` carry its judgement when — and only when — that state is
    ``"ranked"``. A prior the review may revise in either direction; the reason
    is diff-grounded, so it also tells the model what an earlier reading of the
    same diff concluded. ``scope_score`` is ``None`` in every other state: a row
    the first pass never judged has no prior, and the prompt says so in words
    rather than printing a 0/100 nobody wrote.

    ``history`` is this metric's own recent releases
    (:mod:`k4bench.blame.evidence`). Every row that has one is given its
    one-clause summary and the largest movers get the full table, because the
    question this pass exists to answer — did *this* pull request cause *these*
    rows — has a third answer the cross-configuration evidence alone cannot
    reach: that the rows are this series' ordinary noise and nothing caused
    them. ``None`` for a report written before histories were recorded.
    """

    id: str
    detector: str
    platform: str
    sample: str
    label: str
    metric: str
    metric_family: str
    sub_detector: str | None
    direction: str
    pct_change: float | None
    value: float | None = None
    baseline_median: float | None = None
    z_score: float | None = None
    scope_score: float | None = None
    scope_reason: str = ""
    scope_state: ScopeCandidateState = "discovery_incomplete"
    history: MetricHistory | None = None
    #: Where inside the detector this step landed, largest movement first — the
    #: decomposition that turns "this configuration got slower" into a claim a
    #: diff can be checked against. Empty for a memory row, or a run with no
    #: region timing recorded.
    regions: tuple[RegionDelta, ...] = ()


@dataclass(frozen=True)
class CompetingPR:
    """Another pull request that landed in the same window.

    "Did this PR cause it?" is a comparative question, and the first pass never
    asked it that way — it scored every candidate independently. Handing the
    review the rest of the field, with diffs, is what lets it answer *"no, and
    ``owner/repo#123`` fits the affected set better"*. ``patch`` is best-effort:
    a competitor whose diff could not be refetched still appears with its paths,
    its size and the first pass's reason, which is diff-grounded already.

    ``scope_score`` is ``None`` for a competitor the first pass never judged.
    That is not a low score and must never be shown as one: a partially ranked
    field must not make the pull requests nobody looked at read as the ones
    everybody cleared.

    ``scope`` names the run scope that score came from. The first pass judged
    this pull request once per scope and those readings can disagree — 95 where
    it touched the affected detector, 10 elsewhere — and only the strongest is
    carried here, because the prompt cannot hold every scope's reading of thirty
    competitors. Naming it keeps a one-scope judgement from being read as a
    window-wide one, which is the flattening this whole pass exists to undo.
    Empty when there is no score to attribute to a scope.
    """

    repo: str
    number: int
    url: str
    title: str
    files: tuple[str, ...] = ()
    additions: int = 0
    deletions: int = 0
    scope_score: float | None = None
    scope_reason: str = ""
    scope: str = ""
    patch: str = ""
    #: The author's own account of the change — best-effort like ``patch``, and
    #: fenced as the untrusted prose it is.
    body: str = ""
    #: The benchmarked detectors this competitor's files reach
    #: (:func:`~k4bench.blame.geometry.detector_touches`).
    touches: tuple[DetectorTouch, ...] = ()


@dataclass(frozen=True)
class PackageChangeFact:
    """One package that moved across the window — the shape of the release diff.

    ``status`` distinguishes a package that merely advanced (``CHANGED``) from one
    that appeared or disappeared (``ADDED``/``REMOVED``), which are different
    kinds of event: a package entering the stack can change a run without any
    pull request in anyone's commit range.

    A fact belongs to exactly one build platform — the one whose provenance it
    was read from (see
    :attr:`~k4bench.blame.comment.CommentPlan.packages_by_platform`). The same
    package can appear on two platforms with two different statuses, and that
    difference is evidence about reach, so it is never merged away.
    """

    package: str
    status: str
    compare_url: str | None = None


@dataclass(frozen=True)
class AttributionRequest:
    """Everything the review sees for one ``(pull request, change window)``."""

    repo: str
    number: int
    title: str
    base_release: str | None
    onset_release: str
    files: tuple[str, ...] = ()
    patch: str = ""
    #: The reviewed pull request's own description. The one place its author
    #: states, in their own words, what the change is for — and untrusted for
    #: exactly the same reason.
    body: str = ""
    additions: int = 0
    deletions: int = 0
    regressions: tuple[RegressionFact, ...] = ()
    outcomes: tuple[ScopeOutcome, ...] = ()
    competitors: tuple[CompetingPR, ...] = ()
    #: The release diff, kept **per build platform** rather than unioned.
    #: Provenance is recorded per platform, so two platforms can carry different
    #: package sets, different unchanged counts, and different statuses for the
    #: same package. A union combined with one unchanged count would state a
    #: denominator ("2 of 20 tracked") that no platform ever measured.
    packages_by_platform: dict[str, tuple[PackageChangeFact, ...]] = field(
        default_factory=dict
    )
    #: ``platform -> tracked packages that stood still`` on that platform.
    unchanged_by_platform: dict[str, int] = field(default_factory=dict)
    #: Platforms that regressed in this window but whose release diff for
    #: exactly this window was not read. Stated in the prompt, because a silent
    #: omission would read as "nothing changed there".
    packages_unavailable_on: tuple[str, ...] = ()
    #: ``(base platform, onset platform)`` for each platform migration some row's
    #: window was measured across. The migration is a cause in its own right, and
    #: its release diff is labelled ``"<base> → <onset>"`` in
    #: :attr:`packages_by_platform`.
    platform_switches: tuple[tuple[str, str], ...] = ()
    #: The older-boundary pull requests the *first* pass asked to read before it
    #: produced the score that selected this comment
    #: (:mod:`k4bench.blame.history`), re-fetched from the sidecar's persisted
    #: references. Empty on every review whose first pass used none.
    #:
    #: They are here so that both passes weigh the same material. This pass
    #: exists to revise the first's judgement, and a revision made without the
    #: evidence that judgement rested on is not a second opinion — it is a
    #: different question, answered against a smaller world, whose disagreement
    #: with the first would mean nothing. They remain **analogues**: they shipped
    #: before the window opened, they are never scored, never accused, and never
    #: rendered as candidates.
    historical: tuple[HistoricalPR, ...] = ()
    #: Every scope in the night's report that measured this window
    #: (:mod:`k4bench.blame.sweep`): the scopes with regressions, shown as whole
    #: removal sweeps, and every other scope, shown as the controls they are.
    #: Empty renders the regression rows and :attr:`outcomes` instead.
    sweeps: tuple[ScopeSweep, ...] = ()
    #: The benchmarked detectors the reviewed pull request's files reach, and
    #: what it changes in each (:func:`~k4bench.blame.geometry.detector_touches`).
    touches: tuple[DetectorTouch, ...] = ()

    @property
    def slug(self) -> str:
        """``owner/repo#123`` — how this review is named in logs."""
        return f"{self.repo}#{self.number}"


@dataclass(frozen=True)
class StepAssessment:
    """What the review made of the movements themselves, across the window.

    The same three readings the first pass gives (see
    :class:`k4bench.blame.rank.StepAssessment`), asked again here because this
    pass sees strictly more: every configuration's history, not one
    configuration's. A review that concludes ``likely_noise`` withdraws the
    comment entirely — the one outcome this pass can add on its own — so the
    field is not decoration: it is the bot declining to accuse anybody of a
    wobble."""

    verdict: str
    reason: str = ""

    @property
    def likely_noise(self) -> bool:
        return self.verdict == "likely_noise"


@dataclass(frozen=True)
class ScopeJudgement:
    """The review's reasoning about one scope, in the order it was asked for:
    what moved and how, what the pull request changes that the scope loads,
    what supports and what contradicts it, and the likelihood that follows.

    Transient: the likelihood reaches the comment through
    :attr:`Attribution.likelihoods`, and the rest is logged for whoever reviews
    the comment — it is the reasoning, not the claim."""

    scope_id: str
    scope: tuple[str, str, str]
    likelihood: float
    reading: str = ""
    mechanism: str = ""
    supports: str = ""
    contradicts: str = ""


@dataclass(frozen=True)
class Attribution:
    """The review's verdict: a likelihood per regression row, the narrative that
    explains the pattern behind them, and what it made of the movements
    themselves.

    The likelihoods are answered per scope and spread over its rows, with a
    row's own answer where the review gave one (:attr:`scopes`)."""

    summary: str
    likelihoods: dict[str, float]  # RegressionFact.id -> 0-100
    assessment: StepAssessment | None = None
    scopes: tuple[ScopeJudgement, ...] = ()

    @property
    def top_score(self) -> float:
        """The strongest row the review actually answered.

        Not what the withdrawal gate reads: a reply may answer only some rows,
        and the rest keep their per-configuration score, so the gate is measured
        on the *effective* likelihood of every row in the plan (see
        :func:`k4bench.blame.comment.build_comments`). This is the review's own
        high-water mark — useful for logging and for judging a reply, not for
        deciding a comment."""
        return max(self.likelihoods.values(), default=0.0)


class Attributor(Protocol):
    """The narrow seam the comment builder reviews through — model-agnostic."""

    def attribute(self, request: AttributionRequest) -> Attribution | None:
        """Score every regression in *request*, or return ``None`` to decline.

        Only ids present in ``request.regressions`` may appear in the result;
        anything else the caller drops. ``None`` — not an empty
        :class:`Attribution` — is the decline, so "the model said nothing" stays
        distinguishable from "the model said zero".
        """
        ...


# ── The OpenAI-compatible adapter ─────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You review whether one merged pull request caused a set of software "
    "performance regressions measured by a nightly benchmark suite. You are given "
    "an evidence summary computed from the measurements — for every benchmark "
    "scope (detector, physics sample, build platform) with confirmed regressions "
    "in the release window the pull request shipped in: its whole "
    "detector-removal sweep, what moved and in what shape, the metrics' own "
    "history and the machines that measured them; what the pull request's diff "
    "changes in each benchmarked detector it reaches, and what those detectors "
    "measured; and what every other benchmark did in the same window. Then the "
    "details, the packages that changed in the release, the pull request's diff, "
    "and every other pull request that landed in the same window, with its diff. "
    "Judge each scope as a whole, 0-100 for how likely THIS pull request caused "
    "its regressions, and give a row its own likelihood only where the evidence "
    "for that row differs from the rest of its scope. Then write a short summary "
    "explaining the pattern behind your scores. "
    "Reason across scopes — that is the point of this review. A change to one "
    "detector's geometry or reconstruction should move that detector and not the "
    "others; a change to shared infrastructure (framework, allocation, I/O, "
    "logging, build flags) should move many of them at once. A step present in "
    "one detector but absent in another that ran the same sample on the same "
    "platform argues against a shared-infrastructure cause, and if the diff "
    "touches nothing specific to the affected detector it argues against this "
    "pull request entirely. A configuration that moved without confirming is "
    "weak agreement, not disagreement. "
    "Prefer a coherent story — the affected set matching what the diff can "
    "actually reach — over scoring each row in isolation, and judge each scope on "
    "its own evidence: what holds for one detector is not evidence about "
    "another. If another pull request in the window fits the evidence better, "
    "say so in the summary and name it as owner/repo#number. Never write a URL. "
    + NOISE_RULE
    + WEIGHING_RULE
    + SCORE_BAND_RULE
    + ASSESSMENT_RULE
    + UNTRUSTED_EVIDENCE_RULE
    + "Do not invent scopes or regressions: answer only the scope and row ids "
    "you were given. Output JSON only."
)

#: Total *diff* budget (chars) across the reviewed PR and every competitor. Wider
#: than the ranker's, because this prompt carries the whole window's diffs rather
#: than one configuration's, and the comment it feeds is the outward-facing
#: artifact. Oversized diffs shrink evenly (:func:`~k4bench.blame.prompt.allocate_diff_budget`).
_MAX_DIFF_CHARS = 60000

#: Chars of that budget reserved for the reviewed pull request itself before the
#: competitors share the rest. The whole review is about *this* diff; a window
#: with thirty competing pull requests must not be able to squeeze it out.
_SUBJECT_DIFF_FLOOR = 12000

#: Display/prompt bounds. Rows beyond the cap keep their per-configuration score
#: rather than going unscored, and competitors are cut by strength first.
_MAX_ATTRIBUTED_ROWS = 500
#: Public, because the caller must cut the field to this *before* fetching a diff
#: for each competitor (:func:`k4bench.blame.comment._attribution_request`) —
#: fetching a hundred patches to prompt with thirty is a hundred GitHub round
#: trips inside one shared timeout.
MAX_COMPETITORS = 30
_MAX_FILES_LISTED = 12
_MAX_OUTCOMES_LISTED = 40

#: Description budgets. The reviewed pull request earns more room than the
#: alternatives it is weighed against — the whole review is about that one
#: change, and thirty competitors' descriptions would otherwise cost more than
#: their diffs.
_MAX_SUBJECT_BODY_CHARS = 2000
_MAX_COMPETITOR_BODY_CHARS = 400

#: Rows given a *full* history table. Every scope's summary already states its
#: representative metric's history, so the tables are detail: enough for the
#: model to see how these series behave, for the window's largest movements,
#: since those are the rows a comment is written about.
_MAX_HISTORY_BLOCKS = 4

#: Follow-up rounds re-asking for offered rows a reply left unanswered. Each
#: round costs a full request, so this is small; a model that has skipped the
#: same row twice is refusing it, not forgetting it.
_MAX_COMPLETION_ROUNDS = 2

#: The summary is a short paragraph, not a sentence, so the output allowance
#: starts higher than the ranker's per-row figure and still scales with rows.
#: It also covers the step assessment, which is written once per reply.
_OUTPUT_TOKENS_BASE = 1280
_OUTPUT_TOKENS_PER_ROW = 96

#: Longest narrative kept. The contract asks for 2-4 sentences; the renderer caps
#: again for display, but a runaway reply should not reach it in the first place.
_MAX_SUMMARY_CHARS = 1200


@dataclass
class OpenAICompatAttributor:
    """Attributor backed by any OpenAI *chat-completions* endpoint.

    Transport — retries, backoff, output-budget growth, JSON-mode compatibility —
    belongs to the injected :class:`~k4bench.blame.llm.ChatClient`; what is here
    is the prompt and the parse. Any failure returns ``None`` rather than
    raising; what the caller does with that is its decision, not this
    adapter's.
    """

    client: ChatClient

    def attribute(self, request: AttributionRequest) -> Attribution | None:
        if not request.regressions:
            return None
        try:
            content, finish_reason = self.client.complete(
                _system_prompt(request),
                build_user_prompt(request),
                max_output_tokens=min(
                    MAX_OUTPUT_TOKENS,
                    _OUTPUT_TOKENS_BASE
                    + _OUTPUT_TOKENS_PER_ROW * len(request.regressions),
                ),
            )
        except Exception as exc:
            # Timeout, connection error, HTTP status, bad shape — one outcome.
            _log.warning(
                "attribute: %s — LLM call failed (%s); no cross-configuration "
                "review for this pull request", request.slug, exc,
            )
            return None

        attribution = _parse_attribution(content, request)
        if attribution is None:
            _log.warning(
                "attribute: %s — unusable reply (finish_reason=%s); no "
                "cross-configuration review; response prefix=%r",
                request.slug, finish_reason, content[:500],
            )
            return None
        return self._complete(request, attribution)

    def _complete(
        self, request: AttributionRequest, attribution: Attribution
    ) -> Attribution:
        """Re-ask for the offered rows the reply left unanswered.

        A model handed a wide window reliably scores nearly all of it and then
        drops a handful — not because those rows are hard, but because it stopped
        enumerating. Those rows would otherwise fall back to their first-pass
        score and be disclaimed in the comment, which reads as a weaker review
        than actually happened. So the gap is asked again, in follow-up rounds
        carrying the *whole* window — every regression, every clean control,
        every competitor — and narrowing only the *answer* to the missing ids
        (``only_ids`` in :func:`build_user_prompt`). Showing the model just the
        unanswered rows would ask a different question than the one this pass
        exists for: an ALLEGRO row is judged by whether IDEA moved with it, and a
        follow-up that has dropped IDEA cannot see that. The replies are merged
        into the first round's likelihoods; the summary always stays the first
        round's, since that is the one written against the whole window.

        Best-effort throughout: a round that fails, declines, or answers nothing
        new stops the loop and keeps what is already scored. Bounded by
        :data:`_MAX_COMPLETION_ROUNDS` and by requiring strict progress, so a
        model that simply refuses a row costs a fixed number of calls, never a
        spin."""
        offered = {fact.id: fact for fact in _attributed_facts(request)}
        likelihoods = dict(attribution.likelihoods)
        rounds = 0
        for _ in range(_MAX_COMPLETION_ROUNDS):
            missing = [offered[i] for i in offered if i not in likelihoods]
            if not missing:
                break
            rounds += 1
            # The ids are named, not just counted: the same row going unanswered
            # every night points at that row, while a different one each time is
            # the model losing count over a long enumeration. The two call for
            # opposite investigations and the count alone cannot tell them apart.
            _log.info(
                "attribute: %s — re-asking for %d unanswered row(s) of %d: %s",
                request.slug, len(missing), len(offered),
                ", ".join(
                    f"{f.id} ({f.detector} {f.label} {f.metric})"
                    for f in missing[:5]
                ) + (f", … (+{len(missing) - 5} more)" if len(missing) > 5 else ""),
            )
            try:
                content, _finish = self.client.complete(
                    _system_prompt(request),
                    build_user_prompt(
                        request, only_ids=[f.id for f in missing]
                    ),
                    max_output_tokens=min(
                        MAX_OUTPUT_TOKENS,
                        _OUTPUT_TOKENS_BASE + _OUTPUT_TOKENS_PER_ROW * len(missing),
                    ),
                )
            except Exception as exc:
                _log.warning(
                    "attribute: %s — follow-up for %d unanswered row(s) failed "
                    "(%s); keeping what is scored",
                    request.slug, len(missing), exc,
                )
                break
            data = extract_json(content)
            more = (
                _parse_likelihoods(
                    data, {f.id for f in missing}, slug=request.slug,
                    scope_rows=_scope_rows(request),
                )
                if isinstance(data, dict) else {}
            )
            if not more:
                _log.warning(
                    "attribute: %s — follow-up answered none of the %d "
                    "unanswered row(s); keeping what is scored",
                    request.slug, len(missing),
                )
                break
            likelihoods.update(more)
        still_missing = len(offered) - len(likelihoods)
        if still_missing:
            _log.warning(
                # The rounds actually *attempted*, not the cap: a loop that
                # stopped after one failed round and one that spent its whole
                # budget are different faults, and reading the cap back would
                # report them identically.
                "attribute: %s — %d row(s) left unscored after %d follow-up "
                "round(s); they keep their first-pass state",
                request.slug, still_missing, rounds,
            )
        # Summary, assessment and the scopes' reasoning all stay the first
        # round's: they are readings of the whole window, and a follow-up round
        # asks only for the rows that went unanswered.
        return Attribution(
            summary=attribution.summary,
            likelihoods=likelihoods,
            assessment=attribution.assessment,
            scopes=attribution.scopes,
        )


def _system_prompt(request: AttributionRequest) -> str:
    """The system prompt for *request* — with the analogue rule only when there
    are analogues.

    Composed rather than constant so a review that carries no historical evidence
    is asked in exactly the words it was asked in before this feature existed. A
    standing rule about material the prompt does not contain is at best noise and
    at worst an invitation to hunt for it."""
    if not request.historical:
        return _SYSTEM_PROMPT
    return _SYSTEM_PROMPT + HISTORICAL_ANALOGUE_RULE


def attributor_from_env() -> Attributor | None:
    """An :class:`OpenAICompatAttributor` from ``K4BENCH_LLM_*``, or ``None``.

    Reviewing is *off by default*, exactly like ranking: without an endpoint and
    a model the comment builder renders from the per-configuration scores alone.

    ``K4BENCH_LLM_SUMMARY_MODEL`` optionally overrides ``K4BENCH_LLM_MODEL`` for
    this pass only. There is one call per commented pull request and at most ten
    a night, against one call per regression window for the ranker — so spending
    a stronger model on the outward-facing artifact is a variable, not a second
    endpoint, key or code path.
    """
    client = chat_client_from_env(model_env="K4BENCH_LLM_SUMMARY_MODEL")
    return None if client is None else OpenAICompatAttributor(client=client)


# ── Prompt assembly ───────────────────────────────────────────────────────────

_RESPONSE_INSTRUCTION = (
    'Respond with JSON only, no prose: {"step_assessment": {"verdict": '
    '"real_change" | "likely_noise" | "insufficient_evidence", "reason": "<one '
    'sentence citing the evidence>"}, "scopes": [{"scope": "<a scope id given '
    'above, e.g. S1>", "reading": "<what moved in this scope and in what '
    'shape>", "mechanism": "<what this pull request\'s diff changes that this '
    'scope loads — or that it changes nothing it loads>", "supports": "<the '
    'evidence for this pull request here>", "contradicts": "<the evidence '
    'against it here — a detector it changed the same way that did not move, a '
    'pattern its mechanism does not predict — or the empty string>", '
    '"likelihood": <0-100>, "overrides": [{"id": "<a row id of this scope>", '
    '"likelihood": <0-100>}]}], "summary": "<2-4 sentences: which of these '
    'regressions this pull request is responsible for and why, naming the '
    'cross-configuration evidence that decided it>"}. '
    # The summary is quoted into a pull-request comment, where nothing defines
    # the ids.
    "Give a row an override only where its evidence differs from the rest of "
    "its scope. In the summary, refer to regressions by their detector, sample "
    "and metric — never by id: it is read by people who never see these ids. "
)

#: The scope sentence closing :data:`_RESPONSE_INSTRUCTION`. Which rows to
#: answer for is the one thing a follow-up round changes about the ask, so it is
#: swapped here — one sentence saying it, never two disagreeing.
_SCORE_ALL = "Answer every scope listed above and invent none."
_SCORE_ONLY = (
    "Answer ONLY for the rows listed as unanswered above — their scopes, or "
    "overrides for those rows — leave every other scope out, and invent none."
)


def _by_movement(fact: RegressionFact) -> tuple:
    """Biggest step first, identity breaking ties. A missing or non-finite change
    counts as no movement rather than comparing false against everything, which
    would leave the order dependent on input order."""
    pct = fact.pct_change
    magnitude = abs(pct) if pct is not None and math.isfinite(pct) else 0.0
    return (
        -magnitude, fact.detector, fact.platform, fact.sample,
        fact.label, fact.metric, fact.sub_detector or "",
    )


def _attributed_facts(request: AttributionRequest) -> list[RegressionFact]:
    """The regressions actually placed in the prompt — biggest step first.

    The single definition of what was *offered*, because the prompt and the parse
    must agree on it exactly. A window wider than :data:`_MAX_ATTRIBUTED_ROWS`
    shows the model the largest movements and leaves the rest at their
    per-configuration score; only-echo then has to be enforced against this set
    and not against every regression in the request, or ``"r61"`` — an id the
    model was never shown and can only have guessed — would be accepted as a
    judgement of a row nobody reviewed.
    """
    return sorted(request.regressions, key=_by_movement)[:_MAX_ATTRIBUTED_ROWS]


def _regression_lines(request: AttributionRequest) -> list[str]:
    """One block per regression to be scored, grouped by run configuration.

    Grouped rather than listed flat because the cross-configuration comparison is
    the whole task: the model should be able to read "ALLEGRO_o1_v03 moved, IDEA
    did not" off the shape of the prompt, not reconstruct it from sixty
    identically-shaped lines. Each row carries its id, so scoring never depends on
    re-typing an identity.

    The first pass's prior rides on **each row**, not once per run group. Rows in
    one group can genuinely disagree about it: a row's candidate population is
    its own metric's change range, and those ranges differ inside one comment
    window — the same detector, platform and sample can hold a row this pull
    request was ranked 92 on and a row it is not a candidate for at all.
    Printing one prior above both would state the 92 and lose the absence, which
    is the exculpatory half."""
    # A row that does not fit the cap is not scored at all — only-echo leaves it
    # out of the answer, and the caller keeps its per-configuration score rather
    # than inventing one.
    facts = _attributed_facts(request)
    by_scope: dict[tuple[str, str, str], list[RegressionFact]] = {}
    for fact in facts:
        by_scope.setdefault((fact.detector, fact.platform, fact.sample), []).append(fact)

    lines = [
        "Confirmed regressions in this window — score each by its id. "
        "'prior' is what the earlier per-configuration pass knew about this "
        "pull request for that row:",
    ]
    for (detector, platform, sample), facts in by_scope.items():
        lines.append("")
        lines.append(f"### {detector}")
        lines.append(sample_line(sample, prefix="  "))
        lines.append(platform_line(platform, prefix="  "))
        for fact in facts:
            subject = f"{fact.metric} ({fact.label})"
            if fact.sub_detector:
                subject += f" [{fact.sub_detector}]"
            detail = measurement_phrase(
                fact.value, fact.baseline_median, fact.z_score,
            )
            lines.append(
                f"  - [{fact.id}] {subject} {fact.metric_family} "
                f"{direction_phrase(fact.direction, fact.pct_change)}"
                + (f" ({detail})" if detail else "")
            )
            lines.append(f"      prior: {_prior_phrase(fact)}")
            # One clause per row, a full table for a few (below): a row whose
            # series wobbles weekly and a row whose series has never moved must
            # not arrive looking identical, and at this width only the clause
            # fits on every row.
            clause = history_clause(fact.history)
            if clause:
                lines.append(f"      history: {clause}")
    return lines


def _history_lines(request: AttributionRequest) -> list[str]:
    """Full history tables for the window's largest movements.

    The clause on each row says *what* the series does; the table says how, and
    a model asked to weigh a step against its own noise needs to see the noise
    at least once. Capped hard — a wide window holds hundreds of rows and their
    histories repeat — and the cap is stated, because silently showing three of
    two hundred would read as a window where only three rows have a past."""
    facts = [f for f in _attributed_facts(request) if f.history or f.regions]
    if not facts:
        return []
    shown = facts[:_MAX_HISTORY_BLOCKS]
    lines = [
        "",
        "Recent history of the metrics that moved most — this is how each series "
        "behaves when nothing is done to it:",
    ]
    for fact in shown:
        subject = f"{fact.detector} · {fact.metric} ({fact.label})"
        if fact.sub_detector:
            subject += f" [{fact.sub_detector}]"
        lines.append("")
        lines += history_block(fact.history, title=f"[{fact.id}] {subject} — ")
        lines += region_lines(fact.regions)
    remaining = len(facts) - len(shown)
    if remaining > 0:
        lines.append(
            f"  ({remaining} further scored row(s) have a history summarised in "
            f"one line above rather than shown in full.)"
        )
    return lines


def scope_ids(facts: Sequence[RegressionFact]) -> dict[tuple[str, str, str], str]:
    """``scope -> "S1"…`` for the scopes of *facts*, in identity order — the
    handles the model answers per scope with, reproducible from the facts
    alone."""
    scopes = sorted({(f.detector, f.platform, f.sample) for f in facts})
    return {scope: f"S{index}" for index, scope in enumerate(scopes, start=1)}


def _fact_scope(fact: RegressionFact) -> tuple[str, str, str]:
    return (fact.detector, fact.platform, fact.sample)


def _id_order(fact_id: str) -> tuple:
    """``r2`` before ``r10``: ids read in the order they were assigned."""
    digits = fact_id.lstrip("rR")
    return (0, int(digits)) if digits.isdigit() else (1, fact_id)


def _prior_summary(facts: Sequence[RegressionFact]) -> list[str]:
    """What the first pass knew about the reviewed pull request across one
    scope's rows: one line when they agree — rows of one rank group share one
    ranking — and one line per distinct prior, counted, when they do not."""
    groups: dict[tuple, list[RegressionFact]] = {}
    for fact in facts:
        score = None if fact.scope_score is None else round(fact.scope_score)
        groups.setdefault((fact.scope_state, score, fact.scope_reason), []).append(fact)
    if len(groups) == 1:
        return [f"  - prior on every row: {_prior_phrase(facts[0])}"]
    return [
        f"  - prior on {len(rows)} row(s) ({', '.join(f.id for f in rows[:6])}"
        f"{', …' if len(rows) > 6 else ''}): {_prior_phrase(rows[0])}"
        for rows in groups.values()
    ]


def _with_ids(sweep: ScopeSweep, facts: Sequence[RegressionFact]) -> ScopeSweep:
    """*sweep* with each confirmed regression's id on its table cell."""
    ids = {(f.label, f.metric): f.id for f in facts if not f.sub_detector}
    return dataclasses.replace(sweep, rows=tuple(
        dataclasses.replace(row, cells=tuple(
            dataclasses.replace(cell, fact_id=ids.get((row.label, cell.metric), ""))
            for cell in row.cells
        ))
        for row in sweep.rows
    ))


def _unplaced(sweep: ScopeSweep | None, facts: Sequence[RegressionFact]) -> list[RegressionFact]:
    """The rows a sweep table cannot show, which keep a line of their own."""
    shown = {m for metrics in SWEEP_METRICS.values() for m in metrics}
    if sweep is None:
        return list(facts)
    labels = {row.label for row in sweep.rows}
    return [
        f for f in facts
        if f.sub_detector or f.metric not in shown or f.label not in labels
    ]


def _fact_line(fact: RegressionFact) -> list[str]:
    subject = f"{fact.metric} ({fact.label})"
    if fact.sub_detector:
        subject += f" [{fact.sub_detector}]"
    detail = measurement_phrase(fact.value, fact.baseline_median, fact.z_score)
    lines = [
        f"  - [{fact.id}] {subject} {fact.metric_family} "
        f"{direction_phrase(fact.direction, fact.pct_change)}"
        + (f" ({detail})" if detail else "")
    ]
    clause = history_clause(fact.history)
    if clause:
        lines.append(f"      history: {clause}")
    return lines


def _scope_blocks(request: AttributionRequest) -> list[str]:
    """The evidence summary's part about the scopes under judgement: one block
    per scope, with its id, its rows, the first pass's prior, and the readings
    of its sweep, event records, history and machines."""
    facts = _attributed_facts(request)
    ids = scope_ids(facts)
    sweeps = {sweep.scope: sweep for sweep in request.sweeps}
    lines = [
        "Scopes with confirmed regressions in this window — answer each as a "
        "whole by its id:",
    ]
    for scope, sid in ids.items():
        rows = [f for f in facts if _fact_scope(f) == scope]
        sweep = sweeps.get(scope)
        lines.append("")
        row_ids = sorted((f.id for f in rows), key=_id_order)
        lines.append(
            f"[{sid}] {scope[0]} · {scope[2]} · {scope[1]} — {len(rows)} "
            f"confirmed regression(s), ids {', '.join(row_ids[:8])}"
            + (f", … (+{len(rows) - 8} more; each is marked in the sweep table below)"
               if len(rows) > 8 else "")
        )
        lines += _prior_summary(rows)
        if sweep is None:
            continue
        ranked = sorted(rows, key=_by_movement)
        rep = representative(
            sweep, [(f.label, f.metric, f) for f in ranked if f.history],
        )
        lines += scope_evidence_lines(
            sweep,
            history=rep[2].history if rep else None,
            history_metric=rep[1] if rep else "",
            history_label=rep[0] if rep else "baseline",
        )
    return lines


def _summary_lines(request: AttributionRequest) -> list[str]:
    """The evidence summary: what the pull request changes in benchmarked
    geometry, every scope under judgement, and every other scope that measured
    the window."""
    by_detector: dict[str, list[ScopeSweep]] = {}
    for sweep in request.sweeps:
        by_detector.setdefault(sweep.detector, []).append(sweep)
    lines = [EVIDENCE_HEADER, ""]
    if request.touches:
        lines.append(
            f"What {request.slug} changes in the benchmarked detectors' geometry:"
        )
        lines += touch_lines(request.touches, by_detector)
        lines.append("")
    lines += _scope_blocks(request)
    scored = set(scope_ids(_attributed_facts(request)))
    other = other_scope_lines(request.sweeps, exclude=scored)
    if other:
        lines += ["", *other]
    return lines


def _sweep_detail_lines(request: AttributionRequest) -> list[str]:
    """Every configuration of every scope under judgement, each confirmed
    regression marked with its id — or, without sweeps, the regression rows
    themselves."""
    facts = _attributed_facts(request)
    if not request.sweeps:
        return _regression_lines(request)
    sweeps = {sweep.scope: sweep for sweep in request.sweeps}
    lines: list[str] = []
    for scope, sid in scope_ids(facts).items():
        rows = [f for f in facts if _fact_scope(f) == scope]
        sweep = sweeps.get(scope)
        lines.append("")
        lines.append(f"[{sid}] {scope_name(sweep) if sweep else ' · '.join(scope)}:")
        if sweep is not None:
            lines += sweep_table_lines(_with_ids(sweep, rows))
        unplaced = _unplaced(sweep, rows)
        if unplaced:
            lines.append("  Rows the table does not show:")
            for fact in unplaced:
                lines += _fact_line(fact)
    return lines


def _prior_phrase(fact: RegressionFact) -> str:
    """What the first pass concluded about the reviewed pull request for *this*
    row — a score only where one was actually given.

    The three unscored states are spelled out instead, because each is a
    different piece of evidence and none of them is 0/100. "Not among the
    candidates, and the candidate list was complete" in particular argues
    *against* this pull request having caused the row — printing it as a zero
    would say the same thing far less clearly, and a zero the model reads as a
    judgement would say something false about who made it."""
    if fact.scope_state == "ranked" and fact.scope_score is not None:
        return (
            f"ranked {fact.scope_score:.0f}/100 by the per-configuration pass"
            + (f" — {fact.scope_reason}" if fact.scope_reason else "")
        )
    if fact.scope_state == "unranked":
        return (
            "this pull request was a candidate for this regression but the "
            "first pass returned no score for it — no prior, not a low one"
        )
    if fact.scope_state == "not_candidate":
        return (
            "this pull request is NOT among the candidates for this regression: "
            "the candidate search for its change range was complete and this "
            "change is not in it. Weigh that as evidence"
        )
    return (
        "candidate discovery or changed-file evidence for this regression was "
        "incomplete, so nothing follows from whether this pull request appears "
        "in it or how relevant its visible files look"
    )


def _package_lines(request: AttributionRequest) -> list[str]:
    """What moved in the release, and how much did not — one block per platform.

    The unchanged count is the half of the diff that bounds the search: "three of
    twenty tracked packages moved" tells the model the regression has to come out
    of those three, or out of something k4Bench does not track at all.

    That claim is only true *per platform*. Provenance is read per platform, so
    unioning two platforms' changed packages and pairing the union with one
    unchanged count would state a ratio neither platform measured. Each platform
    therefore reports its own diff against its own denominator; where a package
    appears on one and not the other, or with a different status, that difference
    survives — it bounds the reach of the change the way a detector that stayed
    flat does."""
    platforms = sorted(
        set(request.packages_by_platform) | set(request.unchanged_by_platform)
    )
    lines: list[str] = []
    for platform in platforms:
        packages = request.packages_by_platform.get(platform, ())
        unchanged = request.unchanged_by_platform.get(platform, 0)
        if not packages and not unchanged:
            continue
        crossing = any(
            platform == f"{base} → {onset}" for base, onset in request.platform_switches
        )
        where = f" on {platform}" if len(platforms) > 1 or crossing else ""
        lines += [
            "",
            f"Packages that changed across the release window{where} "
            f"({len(packages)} of {len(packages) + unchanged} tracked):",
        ]
        for package in packages:
            note = "" if package.status == "CHANGED" else f" [{package.status}]"
            lines.append(f"- {package.package}{note}")
    if request.packages_unavailable_on:
        lines += [
            "",
            "No release diff was read for this exact window on: "
            + ", ".join(request.packages_unavailable_on)
            + " — do not read that as nothing having changed there.",
        ]
    return lines


def _subject_lines(request: AttributionRequest, budget: int) -> list[str]:
    """The pull request under review: identity, size, paths, diff."""
    size = f"+{request.additions}/-{request.deletions}"
    lines = [
        "",
        f"The pull request under review — {request.slug}: {request.title} ({size})",
    ]
    if request.files:
        lines.append(f"  files: {format_files(request.files, _MAX_FILES_LISTED)}")
    lines += body_block(request.body, _MAX_SUBJECT_BODY_CHARS)
    lines += diff_block(request.patch, budget)
    return lines


def competitor_order(competitor: CompetingPR) -> tuple:
    """Strongest first, then the unscored, then identity.

    Public because the caller must cut the field to :data:`MAX_COMPETITORS` in
    *this* order before fetching any diffs, and a prompt showing a different
    thirty than the caller fetched would be its own bug.

    An unscored candidate sorts as a block after the scored ones rather than at
    the 0/100 end. It has no likelihood, so it cannot be interleaved with the
    ones that do without implying it — but it is never dropped for lacking one
    either: the cap keeps whole candidates, and "nobody judged this PR" stays
    visible as an alternative."""
    return (
        competitor.scope_score is None,
        -(competitor.scope_score if competitor.scope_score is not None else 0.0),
        competitor.repo,
        competitor.number,
    )


def _competitor_lines(
    competitors: list[CompetingPR],
    budgets: list[int],
    by_detector: dict[str, list[ScopeSweep]] | None = None,
) -> list[str]:
    """The rest of the window, each with the first pass's reading of it.

    This block is what turns "is this PR guilty?" into a question with an
    alternative answer. The earlier score and reason ride along because they are
    a diff-grounded summary the review gets for free — and because a competitor
    the first pass rated higher than the PR under review is exactly the case the
    comment must not overclaim in."""
    if not competitors:
        return [
            "",
            "No other pull request was found in any package that changed across "
            "this window — this is the only candidate.",
        ]
    lines = [
        "",
        "Other pull requests that landed in the same window — the alternatives "
        "this one is being weighed against:",
    ]
    for competitor, budget in zip(competitors, budgets):
        size = f"+{competitor.additions}/-{competitor.deletions}"
        lines.append("")
        lines.append(
            f"- {competitor.repo}#{competitor.number} — {competitor.title} ({size})"
        )
        lines.append(f"  url: {competitor.url}")
        if competitor.files:
            lines.append(
                f"  files: {format_files(competitor.files, _MAX_FILES_LISTED)}"
            )
        if competitor.scope_score is None:
            # Never "0/100": the first pass returning nothing about a competitor
            # must not make it look like the alternative everyone ruled out.
            lines.append(
                "  earlier per-configuration review: none — this candidate was "
                "not scored by the first pass"
            )
        else:
            # Named with its scope: this is the strongest of that candidate's
            # per-scope judgements, not a verdict on the whole window, and the
            # difference is exactly what the reviewer is here to weigh.
            where = f" in {competitor.scope}" if competitor.scope else ""
            lines.append(
                f"  strongest earlier per-configuration review{where}: "
                f"{competitor.scope_score:.0f}/100"
                + (f" — {competitor.scope_reason}" if competitor.scope_reason else "")
            )
        if competitor.touches:
            lines += touch_lines(competitor.touches, by_detector or {})
        # The author's own material — description then diff — stays together and
        # last, after k4Bench's own account of the candidate.
        lines += body_block(competitor.body, _MAX_COMPETITOR_BODY_CHARS)
        lines += diff_block(competitor.patch, budget)
    return lines


def build_user_prompt(
    request: AttributionRequest, *, only_ids: Sequence[str] = ()
) -> str:
    """The user message: the window, what regressed, what did not, what changed in
    the release, the pull request under review, and the rest of the field.

    Public so a caller can log or snapshot exactly what was asked — the prompt is
    the whole substance of this stage, and a comment nobody can reconstruct the
    input of is not reviewable.

    *only_ids* narrows what is *asked for* without narrowing what is *shown* —
    the whole window stays in the prompt. See
    :meth:`OpenAICompatAttributor._complete`, its only caller, for why."""
    window = window_phrase(request.base_release, request.onset_release)

    competitors = sorted(request.competitors, key=competitor_order)[:MAX_COMPETITORS]
    # The reviewed diff is reserved first, then the competitors waterfill what is
    # left: the review is about *this* pull request, and a window with thirty
    # candidates must not be able to price its diff out of its own prompt. Above
    # the floor the subject may take whatever no competitor needs.
    competitor_need = sum(len(c.patch) for c in competitors)
    subject_budget = min(
        len(request.patch),
        max(_SUBJECT_DIFF_FLOOR, _MAX_DIFF_CHARS - competitor_need),
    )
    competitor_budgets = allocate_diff_budget(
        [len(c.patch) for c in competitors],
        max(0, _MAX_DIFF_CHARS - subject_budget),
    )

    by_detector: dict[str, list[ScopeSweep]] = {}
    for sweep in request.sweeps:
        by_detector.setdefault(sweep.detector, []).append(sweep)
    size = f"+{request.additions}/-{request.deletions}"
    parts = [
        f"Change window: {window} (Key4hep release dates).",
        *(
            line
            for base, onset in request.platform_switches
            for line in platform_switch_lines(base, onset)
        ),
        f"The pull request under review: {request.slug} — {request.title} ({size}).",
        "",
        *_summary_lines(request),
        "",
        "Details:",
        *_sweep_detail_lines(request),
        *_history_lines(request),
        *(
            outcome_lines(request.outcomes, _MAX_OUTCOMES_LISTED)
            if not request.sweeps else ()
        ),
        *_package_lines(request),
        # Diffs last, so no amount of code can push the evidence out of the
        # model's reading.
        *_subject_lines(request, subject_budget),
        *_competitor_lines(competitors, competitor_budgets, by_detector),
        # Last, and on their own budget: the analogues are background to the
        # window above, and the reviewed pull request and its competitors must
        # never lose a character of diff to them.
        *historical_lines(request.historical),
        "",
        f"Work in this order, scope by scope. From the evidence summary: what "
        f"moved in this scope and in what shape — in the typical event, in a "
        f"few long events, outside the event loop, in memory — and is it noise? "
        f"What does {request.slug}'s diff change that this scope loads, does "
        f"that mechanism predict what moved and what did not, here and in the "
        f"other detectors it reaches, and what contradicts it? Then the "
        f"likelihood that {request.slug} caused this scope's regressions, and "
        f"an override for any row whose evidence differs from its scope's.",
        *_unanswered_instruction(only_ids, request),
        _RESPONSE_INSTRUCTION + (_SCORE_ONLY if only_ids else _SCORE_ALL),
    ]
    return log_prompt_size(
        "attribute", "\n".join(parts),
        detail=(
            f"{request.slug}, {len(request.regressions)} row(s), "
            f"{len(competitors)} competitor(s)"
            + (
                f", {len(request.historical)} historical analogue(s)"
                if request.historical else ""
            )
        ),
    )


def _unanswered_instruction(
    only_ids: Sequence[str], request: AttributionRequest | None = None,
) -> list[str]:
    """The line that turns the full prompt into a request for a few rows.

    Placed after the decision instruction and before the response format, so it
    reads as a narrowing of what to *return* rather than of what to weigh — the
    paragraphs above still say to judge the window as a whole, and they must keep
    meaning that. The rows' scopes are named too, since a scope is how they are
    answered."""
    if not only_ids:
        return []
    scopes = ""
    if request is not None:
        facts = {f.id: f for f in _attributed_facts(request)}
        ids = scope_ids(list(facts.values()))
        named = sorted(
            {ids[_fact_scope(facts[i])] for i in only_ids if i in facts},
            key=lambda sid: int(sid[1:]),
        )
        if named:
            scopes = f" (in scope(s) {', '.join(named)})"
    return [
        "",
        "Your earlier reply already scored the rest of this window. Re-read all "
        "of the above as evidence — the cross-configuration pattern is still what "
        "decides these rows — but answer only for the ids left unanswered"
        f"{scopes}: {', '.join(only_ids)}.",
    ]


# ── Defensive response parsing ────────────────────────────────────────────────

#: A bracket holding nothing but row ids — ``"(r316, r317 and r318)"``. The
#: space before it is part of the match, so the replacement owns its own
#: spacing.
_ID_GROUP = re.compile(
    r"\s*[(\[]\s*(?:r\d+)(?:\s*(?:,|;|/|&|and|·)\s*r\d+)*\s*[)\]]",
    re.IGNORECASE,
)
#: A bare id anywhere else. Bounded both sides so ``v1r2`` and ``r2d2`` survive.
_ID_TOKEN = re.compile(r"(?<![0-9A-Za-z_])r\d+(?![0-9A-Za-z_])", re.IGNORECASE)


def _without_row_ids(
    summary: str, facts: Sequence[RegressionFact], *, slug: str = ""
) -> str:
    """*summary* with the prompt's row ids rewritten into names a reader of the
    comment can recognise. The prompt asks for prose without them; this is what
    makes it true.

    Every id is *replaced*, never deleted. Telling an appositive — "the steps in
    IDEA_o2_v01 (r316, r317)" — from a group carrying its own sentence — "Only
    (r316, r317) regressed" — takes more grammar than a regex has, and guessing
    wrong turns the second into "Only regressed", which no longer says what the
    model said. Repetitive prose is a much smaller failure than altered meaning.

    What varies is only how much of the identity is worth repeating: a group
    whose detector the sentence has just named is expanded to bare metric names,
    which is what makes the common appositive read naturally instead of stuttering
    the detector three times.

    Ids from *this window* only: one that matches no offered row cannot be
    resolved, and a guessed expansion would be worse than the bare token.
    """
    known = {fact.id.lower(): fact for fact in facts}
    if not known or not _ID_TOKEN.search(summary):
        return summary

    def _rewrite_group(match: re.Match) -> str:
        ids = [i.lower() for i in _ID_TOKEN.findall(match.group())]
        if any(i not in known for i in ids):
            return match.group()  # not ours to resolve; leave it exactly as-is
        before = summary[:match.start()]
        names = [
            known[i].metric if _names_detector(before, known[i]) else
            _fact_phrase(known[i])
            for i in ids
        ]
        return " (" + _join(names) + ")"

    cleaned = _ID_GROUP.sub(_rewrite_group, summary)
    cleaned = _ID_TOKEN.sub(
        lambda m: _fact_phrase(known[m.group().lower()])
        if m.group().lower() in known else m.group(),
        cleaned,
    )
    cleaned = " ".join(cleaned.split())
    if cleaned != summary:
        _log.info(
            "attribute: %s — rewrote row ids out of the review's summary; they "
            "mean nothing to a reader of the comment", slug or "?",
        )
    return cleaned


#: How far back a bracket's own clause is taken to reach, for deciding whether
#: the detector has just been named. Long enough for "the steps in IDEA_o2_v01",
#: short enough that a detector named two sentences ago does not count.
_NEARBY_CHARS = 60


def _names_detector(before: str, fact: RegressionFact) -> bool:
    """Whether the text just before a bracket already names its detector."""
    return fact.detector in before[-_NEARBY_CHARS:]


def _join(names: list[str]) -> str:
    """``"a"`` / ``"a and b"`` / ``"a, b and c"`` — an id list read out loud."""
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _fact_phrase(fact: RegressionFact) -> str:
    """A regression named the way the comment's reader meets it elsewhere — its
    detector and metric, with the benchmark configuration in between when it is
    not the default one, since that is then part of what identifies the row."""
    where = fact.detector
    if fact.label and fact.label != "baseline":
        where += f" {fact.label}"
    return f"{where} {fact.metric}"

def _scope_rows(request: AttributionRequest) -> dict[str, set[str]]:
    """``scope id -> offered row ids`` — what a scope's answer applies to."""
    facts = _attributed_facts(request)
    ids = scope_ids(facts)
    rows: dict[str, set[str]] = {sid: set() for sid in ids.values()}
    for fact in facts:
        rows[ids[_fact_scope(fact)]].add(fact.id)
    return rows


def _parse_likelihoods(
    data: dict,
    known: set[str],
    *,
    slug: str = "",
    scope_rows: dict[str, set[str]] | None = None,
) -> dict[str, float]:
    """The row likelihoods of a reply, keyed by id and filtered to *known*.

    A reply answers per scope (``scopes``), and a scope's likelihood applies to
    every offered row of it; a row's own answer — an ``overrides`` entry of its
    scope, or an ``attributions`` row — takes precedence over its scope's. An
    override naming a row of another scope is dropped, not moved: the model
    placed it against the wrong evidence.

    Only-echo lives here: an id the prompt did not offer is a guess, and a guess
    about an unreviewed row is exactly what this pipeline must not publish. Shape
    drift — not an object, a row missing ``id``, a ``likelihood`` that is not a
    number — skips that entry rather than raising.

    Every way an entry is dropped is *counted and logged*, because all of them
    look identical downstream: the row simply has no score, and "the model never
    mentioned it", "it answered with prose", and "it echoed the same id twice"
    are different problems with the same symptom."""
    by_scope: dict[str, float] = {}
    by_row: dict[str, float] = {}
    counts = {"malformed": 0, "unknown id": 0, "unreadable likelihood": 0,
              "duplicate id": 0, "unknown scope": 0, "override outside its scope": 0}
    entries = 0

    def row_score(row: object, allowed: set[str]) -> None:
        nonlocal entries
        entries += 1
        if not isinstance(row, dict):
            counts["malformed"] += 1
            return
        row_id = row.get("id")
        if row_id is None or str(row_id) not in known:
            counts["unknown id"] += 1
            return
        if str(row_id) not in allowed:
            counts["override outside its scope"] += 1
            return
        score = parse_score(row.get("likelihood"))
        if score is None:
            counts["unreadable likelihood"] += 1
            return
        if str(row_id) in by_row:
            counts["duplicate id"] += 1
        by_row[str(row_id)] = score

    scopes = data.get("scopes")
    if isinstance(scopes, list):
        for scope in scopes:
            entries += 1
            if not isinstance(scope, dict):
                counts["malformed"] += 1
                continue
            sid = str(scope.get("scope") or "").strip()
            rows = (scope_rows or {}).get(sid)
            if rows is None:
                counts["unknown scope"] += 1
                continue
            score = parse_score(scope.get("likelihood"))
            if score is None:
                counts["unreadable likelihood"] += 1
            else:
                for row_id in rows & known:
                    by_scope[row_id] = score
            overrides = scope.get("overrides")
            if isinstance(overrides, list):
                for override in overrides:
                    row_score(override, rows & known)
    rows = data.get("attributions")
    if isinstance(rows, list):
        for row in rows:
            row_score(row, known)
    likelihoods = {**by_scope, **by_row}
    if any(counts.values()):
        _log.warning(
            "attribute: %s — reply carried %d entr(ies), %d row(s) usable: %s",
            slug or "?", entries, len(likelihoods),
            ", ".join(f"{n} {what}" for what, n in counts.items() if n),
        )
    return likelihoods


#: The longest clause kept from a scope's reasoning. It is logged, not
#: published, and a clause longer than this is not the one sentence asked for.
_MAX_REASONING_CHARS = 400


def _parse_scopes(
    data: dict, request: AttributionRequest,
) -> tuple[ScopeJudgement, ...]:
    """The reply's per-scope reasoning, for the offered scopes only."""
    raw = data.get("scopes")
    if not isinstance(raw, list):
        return ()
    by_id = {sid: scope for scope, sid in scope_ids(_attributed_facts(request)).items()}
    judgements = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("scope") or "").strip()
        score = parse_score(item.get("likelihood"))
        if sid not in by_id or score is None:
            continue
        judgements.append(ScopeJudgement(
            scope_id=sid, scope=by_id[sid], likelihood=score,
            reading=one_line(item.get("reading"), _MAX_REASONING_CHARS),
            mechanism=one_line(item.get("mechanism"), _MAX_REASONING_CHARS),
            supports=one_line(item.get("supports"), _MAX_REASONING_CHARS),
            contradicts=one_line(item.get("contradicts"), _MAX_REASONING_CHARS),
        ))
    return tuple(judgements)


def _parse_attribution(
    content: str, request: AttributionRequest
) -> Attribution | None:
    """Turn the model's reply into an :class:`Attribution`, or ``None``.

    Enforces only-echo against the rows the prompt actually *offered*
    (:func:`_attributed_facts`), not against every row in the request: an id the
    model was never shown is a guess, and a guess about an unreviewed row is
    exactly what this pipeline must not publish. Shape drift — not an object, no
    ``attributions`` list, a row missing ``id``, a ``likelihood`` that is not a
    number — skips that row rather than raising. A reply with no usable row at all
    is a decline, not an empty verdict: rendering a table of zeros the model never
    committed to would be the confident wrong answer this pipeline exists to
    avoid.
    """
    data = extract_json(content)
    if not isinstance(data, dict):
        return None
    facts = _attributed_facts(request)
    likelihoods = _parse_likelihoods(
        data, {f.id for f in facts}, slug=request.slug,
        scope_rows=_scope_rows(request),
    )
    if not likelihoods:
        return None
    summary = _without_row_ids(
        one_line(data.get("summary"), _MAX_SUMMARY_CHARS), facts,
        slug=request.slug,
    )
    if not summary:
        # The scores without the narrative would be a table of numbers with no
        # stated reasoning, in a comment posted to someone else's repository.
        # Falling back to the per-configuration verdict — which does carry a
        # reason — is the more honest degradation.
        _log.warning(
            "attribute: %s — scored %d row(s) but gave no summary; declining",
            request.slug, len(likelihoods),
        )
        return None
    assessment = _parse_assessment(data, slug=request.slug)
    if assessment is None:
        # Required, exactly like the summary, and for a stronger reason: this is
        # the pass that decides whether a public accusation is posted, and the
        # gate downstream reads this field. A reply without it is
        # indistinguishable from one that never considered whether the movements
        # are real, and accepting it would let a model skip the question and
        # still produce a comment — the precise gap this field was added to
        # close. Declining costs a night and is recoverable; posting on an
        # unconsidered step is not.
        _log.warning(
            "attribute: %s — scored %d row(s) but gave no usable "
            "step_assessment; declining, so no comment tonight",
            request.slug, len(likelihoods),
        )
        return None
    scopes = _parse_scopes(data, request)
    for judgement in scopes:
        _log.info(
            "attribute: %s — %s %s: %.0f%% — reading: %s | mechanism: %s | "
            "supports: %s | contradicts: %s",
            request.slug, judgement.scope_id, " · ".join(judgement.scope),
            judgement.likelihood, judgement.reading or "-",
            judgement.mechanism or "-", judgement.supports or "-",
            judgement.contradicts or "-",
        )
    return Attribution(
        summary=summary, likelihoods=likelihoods, assessment=assessment,
        scopes=scopes,
    )


def _parse_assessment(data: dict, *, slug: str = "") -> StepAssessment | None:
    """The review's read of the movements, or ``None`` when it gave none.

    ``None`` is a real state — an older model, a skipped field, a verdict
    outside :data:`~k4bench.blame.prompt.ASSESSMENT_VALUES` — and it means *not
    assessed*, never ``real_change``. That direction matters here more than
    anywhere: the only thing this field does on its own is *withhold* a comment,
    so reading an absent assessment as a positive one would silently restore the
    comment the field exists to prevent.
    """
    raw = data.get("step_assessment")
    if isinstance(raw, str):
        verdict, reason = raw, ""  # a model that answered with the bare verdict
    elif isinstance(raw, dict):
        verdict = str(raw.get("verdict") or "")
        reason = one_line(raw.get("reason"), _MAX_SUMMARY_CHARS)
    else:
        return None
    verdict = verdict.strip().lower().replace(" ", "_").replace("-", "_")
    if verdict not in ASSESSMENT_VALUES:
        return None
    if verdict != "real_change":
        _log.info("attribute: %s — review reads the step as %s", slug or "?", verdict)
    return StepAssessment(verdict=verdict, reason=reason)
