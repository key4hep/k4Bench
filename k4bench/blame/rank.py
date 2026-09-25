"""Rank a regression's candidate pull requests with a language model.

The builder collects, for one release-boundary window, every pull request in
the commit range of every package that moved across it — but *which* of them
caused the step is a judgement over the real diffs, not a path match. This
module makes that judgement with a model: it is handed every metric that
stepped across the window and each candidate's code change, and returns a
0–100 likelihood and a one-line reason per PR.

Three properties are load-bearing, and shape the whole module:

* **Model-independence.** The one adapter, :class:`OpenAICompatRanker`, is a
  prompt and a parser over a :class:`~k4bench.blame.llm.ChatClient` — the shared
  OpenAI-compatible transport, with no vendor SDK and no pinned model. Provider,
  model and key are environment variables (:func:`ranker_from_env`), so switching
  from one free endpoint to another is a settings change, never a code change.
  :class:`Ranker` is a ``Protocol`` so a second adapter can drop in without the
  builder knowing.

* **Only-reorder.** The model may only score the candidates it was given.
  :func:`_parse_rankings` drops any ``(repo, pr)`` the request did not contain,
  so a hallucinated PR number is structurally impossible to surface — never
  merely unlikely. The builder drops unknown keys a second time (defence in
  depth).

* **Honest failure.** Every failure path — no config, HTTP error, timeout,
  malformed JSON — returns ``{}``. "No ranking" is a real state the rest of the
  pipeline already handles (the dashboard hides the ledger, the email omits the
  "most likely" line); a confident wrong culprit would be worse than none. The
  score is a *lead for a human*, in keeping with this repo's "no evidence ⇒ no
  verdict" culture — never a verdict.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from typing import Protocol

from k4bench.blame.evidence import MetricHistory, ScopeOutcome
from k4bench.blame.history import (
    REQUEST_KEY,
    HistoricalBoundary,
    HistoricalEvidence,
    HistoricalIndex,
    HistoricalPR,
    HistoricalRequestError,
    parse_request,
)
from k4bench.blame.geometry import DetectorTouch
from k4bench.blame.github import path_under
from k4bench.blame.llm import (
    MAX_OUTPUT_TOKENS,
    ChatClient,
    chat_client_from_env,
    coerce_int,
    extract_json,
    one_line,
    parse_score,
)
from k4bench.blame.prompt import (
    ASSESSMENT_RULE,
    ASSESSMENT_VALUES,
    HARNESS_PACKAGE_NOTE,
    HISTORICAL_ANALOGUE_RULE,
    NOISE_RULE,
    SCORE_BAND_RULE,
    UNTRUSTED_EVIDENCE_RULE,
    WEIGHING_RULE,
    allocate_favoured_diff_budget,
    body_block,
    compact_dir,
    diff_block,
    direction_phrase,
    format_files,
    geometry_reach,
    geometry_tree,
    historical_lines,
    historical_offer_lines,
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
    file_change_phrase,
    other_scope_lines,
    representative,
    scope_evidence_lines,
    scope_name,
    scope_state,
    sweep_table_lines,
    touch_lines,
)
from k4bench.blame.sweep import ScopeSweep
from k4bench.regression.models import RegionDelta

_log = logging.getLogger(__name__)


# ── The request/response contract ─────────────────────────────────────────────

@dataclass(frozen=True)
class RankCandidate:
    """One pull request offered to the ranker.

    ``repo``/``number`` are the key the builder matches the ranking back on;
    ``patch`` is the bounded diff sample (:mod:`k4bench.blame.github`) and is
    transient — it is model input, never persisted."""

    repo: str  # "owner/repo" slug
    number: int
    title: str
    files: tuple[str, ...] = ()
    patch: str = ""
    #: The author's own account of the change. Frequently states the mechanism —
    #: and sometimes the expected cost — more plainly than the diff shows it, and
    #: is as untrusted as the diff: it is fenced the same way.
    body: str = ""
    #: How large the change is. A three-line change and a three-thousand-line one
    #: deserve different priors, and the second pass was already shown this.
    additions: int = 0
    deletions: int = 0
    #: Every benchmarked detector this pull request's files reach, and what it
    #: changes in each (:func:`~k4bench.blame.geometry.detector_touches`):
    #: the other detectors a geometry change reaches are this run's controls.
    #: Empty when the run records no geometry, or the change reaches none.
    touches: tuple[DetectorTouch, ...] = ()


@dataclass(frozen=True)
class MetricStep:
    """One metric's step across the shared release window — several of these
    can ride in one :class:`RankRequest` when more than one metric stepped
    across the same release boundary, so the model judges the candidates
    against the window's full picture rather than a single arbitrary metric.

    ``label`` is the benchmark config the metric was measured under (e.g. a
    removal sweep's ``baseline`` vs. ``no_<detector>``) — deliberately
    *not* a grouping key: labels sharing a window still get one shared
    ranking, not one call each, but each step keeps its own label so the model
    can tell "baseline regressed" apart from "only no_HCAL regressed",
    which is itself a clue."""

    metric: str
    metric_family: str
    direction: str
    pct_change: float | None
    label: str
    sub_detector: str | None = None
    #: The measurement behind the percentage, and how far outside the baseline's
    #: own spread it sits. A percentage alone cannot distinguish a marginal step
    #: from an unmistakable one, and the model was previously shown only the
    #: percentage.
    value: float | None = None
    baseline_median: float | None = None
    z_score: float | None = None
    #: This metric's own recent releases (:mod:`k4bench.blame.evidence`), or
    #: ``None`` for a report that predates recorded histories. The evidence that
    #: lets a model conclude the step is noise and blame nobody.
    history: MetricHistory | None = None
    #: Where inside the detector this step landed, largest movement first. A
    #: mechanism the model can match a diff against before it reads any code —
    #: empty for a memory metric, or a run that recorded no region timing.
    regions: tuple[RegionDelta, ...] = ()


@dataclass(frozen=True)
class RankRequest:
    """Everything the ranker sees for one release-boundary window: every metric
    that stepped across it, every configuration that measured the same window
    without stepping, and every candidate PR across every package that changed
    in the window."""

    metrics: tuple[MetricStep, ...]
    detector: str
    platform: str
    sample: str
    base_release: str | None
    onset_release: str
    candidates: tuple[RankCandidate, ...] = ()
    #: The build platforms the window's base and onset runs were measured on,
    #: when either is not ``platform`` — a series continuing a replaced
    #: platform's history. ``None`` means ``platform``.
    base_platform: str | None = None
    onset_platform: str | None = None
    #: Tracked packages that did **not** move across this window. The other half
    #: of the diff, and the half that bounds the search: "three of twenty-one
    #: moved" tells the model the cause is in those three or in something
    #: k4Bench does not track at all. The second pass has always been shown this;
    #: the first had not.
    n_unchanged: int = 0
    #: The geometry tree this run actually loads, e.g. ``FCCee/ALLEGRO/`` — the
    #: seam that turns "does this diff reach this detector?" from an inference
    #: over path names into a fact. Empty when the run recorded no geometry path
    #: (every run benchmarked before it was captured), and the prompt then says
    #: nothing rather than guessing.
    geometry_tree: str = ""
    #: The controls: configurations that measured this window and stayed flat.
    #: Cross-configuration evidence is what tells a shared-infrastructure cause
    #: apart from a detector-specific one, and this pass used to be shown none
    #: of it — every call saw one configuration and could only conclude *within*
    #: it.
    outcomes: tuple[ScopeOutcome, ...] = ()
    #: The ``owner/repo`` slug of the benchmark harness when its own commits
    #: moved across this window, else ``""``. The harness is categorically
    #: unlike the stack packages — it decides how the run is invoked and
    #: measured rather than running inside it — and the prompt says so
    #: (:data:`~k4bench.blame.prompt.HARNESS_PACKAGE_NOTE`) wherever its
    #: candidates appear, so a model never judges them as simulation code.
    harness_repo: str = ""
    #: What older release boundaries the model may ask to see the code behind,
    #: and the seam it is retrieved through (:mod:`k4bench.blame.history`).
    #: ``None`` — the default, and every environment without
    #: ``K4BENCH_LLM_HISTORICAL_DIFFS`` set — offers nothing, makes no extra
    #: model call and reads no extra GitHub page. The offer travels on the
    #: request rather than through :class:`Ranker` so that a second adapter can
    #: ignore it entirely without the builder knowing.
    history: HistoricalIndex | None = None
    #: Every configuration of this run's scope across the window
    #: (:mod:`k4bench.blame.sweep`) — the removal sweep as one picture. ``None``
    #: renders the per-metric bullets and the non-confirming configurations
    #: (:attr:`outcomes`) instead.
    sweep: ScopeSweep | None = None
    #: Every scope in the night's report that measured this window, this one
    #: included: what the other detectors a candidate reaches did, and what the
    #: rest of the suite did in the same window.
    window_sweeps: tuple[ScopeSweep, ...] = ()


@dataclass(frozen=True)
class Ranking:
    """The ranker's verdict on one PR: a 0–100 likelihood it is the cause, a
    one-line reason grounded in its diff, and what argues against it.

    ``against`` is the counter-evidence the model was asked to state for its own
    judgement — the configuration that stayed flat, the part of the mechanism the
    diff does not actually show. It is *optional*: a reply without it still
    ranks, because rejecting the row would cost a real judgement to gain a
    stylistic one. Writing it is what keeps a plausible-sounding title from
    passing as a mechanism, and it gives a human the fastest way to dismiss a
    wrong lead."""

    score: float
    description: str
    against: str = ""


@dataclass(frozen=True)
class StepAssessment:
    """What the model made of the *movement*, before any question of who caused
    it.

    Separate from the rankings because it answers a different question, and
    because without somewhere to say "this step is noise" a model can only
    express that by scoring every candidate low — which is indistinguishable
    downstream from "I looked and found nothing", and loses the one conclusion a
    human most needs. ``verdict`` is one of
    :data:`~k4bench.blame.prompt.ASSESSMENT_VALUES`; anything else is dropped at
    the parse, so a surface rendering this never has to defend against a word
    nobody defined."""

    verdict: str
    reason: str = ""

    @property
    def likely_noise(self) -> bool:
        return self.verdict == "likely_noise"


@dataclass(frozen=True)
class RankResult:
    """One rank call's outcome: a judgement per candidate, and the model's read
    of the step itself.

    A dataclass rather than a bare mapping because the assessment belongs to the
    *window*, not to any candidate — attaching it to one of them would make it
    look like a property of that pull request. ``rankings`` may be partial (a
    reply that ran out of rows), and ``assessment`` may be ``None`` (a model that
    declined to give one); both are ordinary, and neither invalidates the other.
    """

    rankings: dict[tuple[str, int], Ranking] = dataclasses_field(
        default_factory=dict
    )
    assessment: StepAssessment | None = None
    #: The historical analogues this judgement was actually made with, empty on
    #: every ranking that used none. Carried out so the builder can persist a
    #: reproducible *reference* to each (:class:`~k4bench.blame.models.HistoricalRef`)
    #: and the outward-facing review can be shown the same material — a score
    #: reached with evidence the second pass never sees is a score the second
    #: pass cannot honestly revise. Patches and bodies ride here because this is
    #: transient model input; only the reference is persisted.
    historical: tuple[HistoricalPR, ...] = ()

    def __bool__(self) -> bool:
        """True when anything was actually judged — what the builder tests."""
        return bool(self.rankings)


class Ranker(Protocol):
    """The narrow seam the builder ranks through — model-agnostic by design."""

    def rank(self, request: RankRequest) -> RankResult:
        """Score every candidate in *request*, and read the step itself.

        Only PRs present in ``request.candidates`` may appear in
        :attr:`RankResult.rankings`; anything else the caller drops. Return an
        empty :class:`RankResult` to decline — the builder then leaves the
        candidates unranked, an honest state, not an error.
        """
        ...


# ── The OpenAI-compatible adapter ─────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You attribute a software performance regression to the pull request most "
    "likely responsible — or to none of them, when that is what the evidence "
    "says. You are given the run context the regression was measured in — one "
    "detector, one physics sample, one build platform — and an evidence summary "
    "computed from the measurements: every configuration of that run's "
    "detector-removal sweep across the same release window, what moved and in "
    "what shape, the metrics' own history and the machines that measured them, "
    "which candidates change this detector's geometry and what the other "
    "benchmarked detectors they change did, and what every other benchmark did "
    "in the same window. Then the details, and for each package that changed "
    "the pull requests in its commit range with their code diffs. "
    "Score each PR independently 0-100 for how likely it caused the regressions "
    "as a whole, give a one-sentence reason grounded in the diff, and state what "
    "argues against it. "
    "For every candidate, ask whether it makes sense that this change affected "
    "the metrics of the run in the context — its detector, its sample, its "
    "build, or the shared infrastructure that run goes through (framework, "
    "allocation, I/O, logging, build flags). A shared-infrastructure cause is a "
    "perfectly good answer: say so at that level rather than inventing a "
    "detector- or sample-specific mechanism the diff does not show. A "
    "configuration that ran the same window and stayed flat bounds what any "
    "cause can reach: a step in one detector and not another that ran the same "
    "sample on the same platform argues against a shared-infrastructure cause. "
    + NOISE_RULE
    + WEIGHING_RULE
    + SCORE_BAND_RULE
    + ASSESSMENT_RULE
    + UNTRUSTED_EVIDENCE_RULE
    + "Do not invent PRs. Output JSON only."
)

#: Total *diff* budget (chars) across all candidates. Per-PR patches are
#: already bounded in :mod:`k4bench.blame.github`; this is the backstop that
#: keeps a wide window (many PRs) inside a small-context model by waterfilling
#: the budget — every oversized diff shrinks evenly, after the candidates
#: touching the run's own compact directory are served first (see
#: :func:`~k4bench.blame.prompt.allocate_favoured_diff_budget`), and file paths
#: and titles always survive, so every PR is still scored, at worst from
#: metadata.
_MAX_PROMPT_CHARS = 45000
_MAX_FILES_LISTED = 12
_MAX_DESCRIPTION_CHARS = 200
#: Description kept per candidate. Long enough for the paragraph that says what
#: a change does and why, short enough that a template-heavy repository cannot
#: spend the prompt on checklists.
_MAX_BODY_CHARS = 1200
#: Counter-evidence is a clause, not an essay — and unlike the reason it is not
#: rendered on the outward-facing surfaces, so it earns less room.
_MAX_AGAINST_CHARS = 200

#: Metrics given a full history table. One window can confirm a dozen correlated
#: metrics (``wall_time_s``, ``user_cpu_s`` and ``mean_time_s`` step together),
#: and their histories say the same thing three times; the evidence summary
#: already states the representative one's reading, so the largest movers'
#: tables are detail, bounded whatever the window's width.
_MAX_HISTORY_BLOCKS = 3

#: Non-confirming configurations listed. The same cap the cross-configuration
#: pass uses, for the same reason: enough to establish the pattern, bounded
#: against a sweep that runs hundreds of configurations.
_MAX_OUTCOMES_LISTED = 40

#: Output allowance asked for per candidate, on top of the client's configured
#: floor — a wide window has more rows to write, and a reasoning model spends
#: hidden tokens before any of them.
_OUTPUT_TOKENS_PER_CANDIDATE = 512

#: Room for the evidence reading and the step assessment on top of the
#: per-candidate rows: they are written once per call, not once per candidate,
#: so they scale with nothing.
_OUTPUT_TOKENS_ASSESSMENT = 768

# Keep trying within a fixed bound when a model omits rows. Each retry
# deliberately receives the complete request again: the candidates are
# comparative context, not independent questions.
_MAX_RESPONSE_ATTEMPTS = 10


@dataclass
class OpenAICompatRanker:
    """Ranker backed by any OpenAI *chat-completions* endpoint.

    All transport — retries, backoff, output-budget growth, JSON-mode
    compatibility — belongs to the injected :class:`~k4bench.blame.llm.ChatClient`;
    what is left here is the prompt, the parse, and the follow-up call that
    completes a partial answer. On final failure :meth:`rank` returns an empty
    :class:`RankResult` rather than raising: a late report beats a blocked one,
    and blame is a best-effort sidecar."""

    client: ChatClient

    def rank(self, request: RankRequest) -> RankResult:
        """Score every candidate, retrieving historical evidence at most once.

        With no historical index on the request — the default, and the whole of
        production until ``K4BENCH_LLM_HISTORICAL_DIFFS`` is set — this is one
        prompt, one parse, and bounded retries for rows a reply omitted.

        With an index, the first call may come back asking for the code behind
        an older boundary instead of answering. That is the *only* thing it can
        ask for and the only extra round it gets: the follow-up prompt carries
        the evidence and no index, so there is nothing valid left to request.
        Every way that round can fail — an invented id, a package nobody offered,
        a range GitHub would not give up completely — returns an empty result.
        Falling back to the preliminary rankings in the asking reply is the one
        thing that must not happen: the model told us it wanted to see this code
        before judging, and publishing the judgement it made *without* it would
        be publishing a score under an expectation the application refused.
        """
        if not request.candidates:
            return RankResult()
        index = request.history
        if index is None or not index:
            return self._rank_rounds(request)
        outcome = self._retrieval_round(request, index)
        if outcome is None:
            return RankResult()
        prior, evidence = outcome
        return self._rank_rounds(request, prior=prior, evidence=evidence)

    def _retrieval_round(
        self, request: RankRequest, index: HistoricalIndex
    ) -> tuple[tuple[str, str] | None, HistoricalEvidence | None] | None:
        """The offer call: ``(reply to reuse, evidence)``, or ``None`` to decline.

        Three outcomes, and they are genuinely three. A reply with no request is
        the ordinary one, and it is *reused* rather than re-asked — a night where
        nobody wants history costs exactly one model call, which is what makes
        the feature affordable enough to leave on. A valid request is retrieved
        and the reply discarded. Anything else — a failed call, a request that
        cannot be honoured, evidence that could not be read in full — declines.
        """
        # Every boundary in the tail is *described*, including the ones whose
        # release diff could not be read — a gap in a list of dates reads as a
        # boundary where nothing moved, which is the opposite of what an unread
        # diff means. Only :attr:`HistoricalIndex.offered` may be asked for, and
        # :func:`parse_request` is what enforces that.
        offer = index.boundaries
        try:
            content, finish_reason = self._complete(
                request, offer=offer, offer_omitted=index.boundaries_omitted,
            )
        except Exception as exc:
            _log.warning(
                "rank: %s/%s %s — LLM call failed (%s); no ranking for this window",
                request.detector, request.sample, request.onset_release, exc,
            )
            return None
        try:
            ask = parse_request(extract_json(content), offer)
        except HistoricalRequestError as exc:
            _log.warning(
                "rank: %s/%s %s — the model asked for historical evidence it was "
                "never offered (%s); declining rather than ranking on the "
                "preliminary scores it wrote alongside the ask",
                request.detector, request.sample, request.onset_release, exc,
            )
            return None
        if ask is None:
            return (content, finish_reason), None

        _log.info(
            "rank: %s/%s %s — model requested historical evidence [%s] of the %d "
            "retrievable boundary(ies) among %d described; stated reason: %s",
            request.detector, request.sample, request.onset_release,
            ask.describe(), len(index.offered), len(offer), ask.reason,
        )
        try:
            evidence = index.provider.fetch(ask)
        except Exception as exc:  # noqa: BLE001 — a raising provider is a decline
            _log.warning(
                "rank: %s/%s %s — historical retrieval raised (%s); leaving this "
                "window unranked",
                request.detector, request.sample, request.onset_release, exc,
            )
            return None
        if not evidence.complete:
            _log.warning(
                "rank: %s/%s %s — historical evidence for [%s] could not be read "
                "completely (%s); leaving this window unranked rather than "
                "judging it on a partial view of evidence the model asked for",
                request.detector, request.sample, request.onset_release,
                ask.describe(), evidence.reason or "no reason recorded",
            )
            return None
        _log.info(
            "rank: %s/%s %s — retrieved %d historical pull request(s): %s",
            request.detector, request.sample, request.onset_release,
            len(evidence.prs), ", ".join(pr.slug for pr in evidence.prs) or "none",
        )
        return None, evidence

    def _rank_rounds(
        self,
        request: RankRequest,
        *,
        prior: tuple[str, str] | None = None,
        evidence: HistoricalEvidence | None = None,
    ) -> RankResult:
        """The ranking call and its bounded completion retries.

        *prior* is a reply already in hand from the offer call — reused rather
        than re-asked, so a window where the model wanted no history costs one
        call. *evidence* is the retrieved analogues; every completion retry
        carries the same ones and, like this round, no index — so "one retrieval
        round" holds structurally rather than by a counter nobody could forget
        to increment.
        """
        expected = {(c.repo, c.number) for c in request.candidates}
        combined: dict[tuple[str, int], Ranking] = {}
        assessment: StepAssessment | None = None
        historical = evidence.prs if evidence is not None else ()
        for response_attempt in range(_MAX_RESPONSE_ATTEMPTS):
            if prior is not None:
                content, finish_reason = prior
                prior = None
            else:
                try:
                    content, finish_reason = self._complete(
                        request, evidence=evidence
                    )
                except Exception as exc:
                    # Timeout, connection error, HTTP status, bad shape — all the
                    # same final outcome. Preserve any valid rows from a prior
                    # partial response; strict publishers will still reject it.
                    _log.warning(
                        "rank: LLM call failed (%s) — %d/%d candidates ranked",
                        exc, len(combined), len(expected),
                    )
                    return RankResult(
                        rankings=combined, assessment=assessment,
                        historical=historical,
                    )

            if evidence is not None and _asks_again(content):
                # The model has been given what it asked for and is asking
                # again. There is no second round to give it, so this is an
                # explicit statement that it is not ready to judge — exactly the
                # signal the first round honours by discarding preliminary
                # scores, and it does not change meaning because it arrived one
                # round later. Publishing the rankings beside it would take a
                # judgement the model itself called provisional.
                #
                # This is reachable by prompt injection: a historical body or
                # patch can try to induce the member. That costs a ranking,
                # never a wrong one — the failure stays on the side this
                # pipeline fails to.
                _log.warning(
                    "rank: %s/%s %s — the follow-up asked for further historical "
                    "evidence there is no round left to supply; declining rather "
                    "than publishing the scores it wrote alongside the ask",
                    request.detector, request.sample, request.onset_release,
                )
                return RankResult()
            combined.update(_parse_rankings(content, request))
            # First reading of the step wins. A retry exists to fill in rows the
            # reply ran out of room for, and the assessment is a judgement of the
            # window rather than of any row — re-asking it and taking the newer
            # answer would silently overwrite a considered reading with one made
            # under a prompt asking for something else.
            if assessment is None and (reading := _parse_reading(content)):
                _log.info(
                    "rank: %s/%s %s — evidence reading: %s",
                    request.detector, request.sample, request.onset_release, reading,
                )
            assessment = assessment or _parse_assessment(content)
            missing = expected - set(combined)
            if not missing:
                return RankResult(
                    rankings=combined, assessment=assessment, historical=historical,
                )

            retrying = response_attempt + 1 < _MAX_RESPONSE_ATTEMPTS
            retry_note = (
                f"; retrying (response attempt "
                f"{response_attempt + 2}/{_MAX_RESPONSE_ATTEMPTS})"
                if retrying else ""
            )
            _log.warning(
                "rank: LLM returned %s ranking (%d/%d candidates; "
                "finish_reason=%s); missing: %s; response prefix=%r%s",
                "no usable" if not combined else "a partial",
                len(combined), len(expected), finish_reason,
                ", ".join(f"{repo}#{number}" for repo, number in sorted(missing)),
                content[:500],
                retry_note,
            )
        return RankResult(
            rankings=combined, assessment=assessment, historical=historical,
        )

    def _complete(
        self,
        request: RankRequest,
        *,
        offer: tuple[HistoricalBoundary, ...] = (),
        offer_omitted: int = 0,
        evidence: HistoricalEvidence | None = None,
    ) -> tuple[str, str]:
        """POST the prompt and return ``(assistant text, finish reason)``.

        *offer* and *evidence* are mutually exclusive by construction: the offer
        belongs to the first call and the evidence to the one that answers it,
        and a prompt carrying both would let a model that has already been given
        what it asked for ask again."""
        return self.client.complete(
            _SYSTEM_PROMPT + (
                HISTORICAL_ANALOGUE_RULE if evidence is not None else ""
            ),
            _build_user_prompt(
                request, offer=offer, offer_omitted=offer_omitted, evidence=evidence,
            ),
            max_output_tokens=min(
                MAX_OUTPUT_TOKENS,
                _OUTPUT_TOKENS_ASSESSMENT
                + _OUTPUT_TOKENS_PER_CANDIDATE * len(request.candidates),
            ),
        )


def ranker_from_env() -> Ranker | None:
    """An :class:`OpenAICompatRanker` from ``K4BENCH_LLM_*``, or ``None``.

    Ranking is *off by default*: unset ``K4BENCH_LLM_URL`` or
    ``K4BENCH_LLM_MODEL`` returns ``None``, and the builder then collects
    candidates without scoring them. Only a configured environment (CI with the
    secrets, or a dev box for backfill) enables the model."""
    client = chat_client_from_env()
    return None if client is None else OpenAICompatRanker(client=client)


# ── Prompt assembly ───────────────────────────────────────────────────────────

_RESPONSE_INSTRUCTION = (
    'Respond with JSON only, no prose: {"evidence_reading": "<two or three '
    'sentences: what the evidence summary says about the movement — its shape, '
    'where in the sweep it is and is not, the history and machines — and about '
    'the other detectors the candidates reach>", "step_assessment": {"verdict": '
    '"real_change" | "likely_noise" | "insufficient_evidence", "reason": "<one '
    'sentence citing the evidence>"}, "rankings": [{"repo": "<owner/repo>", '
    '"pr": <number>, "likelihood": <0-100>, "reason": "<one sentence: the '
    'mechanism in the diff and how it fits what moved, at whatever level the '
    'diff supports — this detector and sample, or the shared code the run goes '
    'through>", "against": "<one sentence on what contradicts it — a detector '
    'it changed the same way that did not move, a pattern it does not '
    'predict — or the empty string if genuinely nothing does>"}]}. '
    'Score every candidate listed above and invent none.'
)


def _step_lines(
    request: RankRequest, metrics: tuple[MetricStep, ...] | None = None,
) -> list[str]:
    """One bullet per metric that stepped, with its measurement and a one-line
    reading of its own history — for *metrics*, every step by default.

    Both ride on the bullet rather than only in the table below it, because not
    every metric gets a table: the tables are capped at the largest movers, and a
    metric past that cap would otherwise arrive as a bare percentage with no way
    to judge its size and no hint of what its series normally does."""
    steps = request.metrics if metrics is None else metrics
    if not steps:
        return []
    lines = ["- Metrics that stepped across the window:"]
    for step in steps:
        subject = f"{step.metric} ({step.label})"
        if step.sub_detector:
            subject += f" [{step.sub_detector}]"
        detail = measurement_phrase(
            step.value, step.baseline_median, step.z_score,
        )
        lines.append(
            f"  - {subject} {direction_phrase(step.direction, step.pct_change)}"
            + (f" ({detail})" if detail else "")
        )
        clause = history_clause(step.history)
        if clause:
            lines.append(f"      history: {clause}")
    return lines


def _by_movement(step: MetricStep) -> tuple:
    """Biggest step first, identity breaking ties — which metrics earn a history
    table. A missing or non-finite change counts as no movement rather than
    comparing false against everything, which would leave the order dependent on
    input order."""
    pct = step.pct_change
    magnitude = abs(pct) if pct is not None and math.isfinite(pct) else 0.0
    return (-magnitude, step.metric, step.label, step.sub_detector or "")


def _history_lines(request: RankRequest) -> list[str]:
    """History tables for the window's largest movers.

    Capped rather than exhaustive: a detector-removal sweep can confirm dozens
    of correlated metrics in one window, and a dozen full tables would crowd out
    the diffs. The evidence summary already carries the representative
    metric's reading, and the cap is stated when it bites, because a prompt
    that silently showed three of twelve histories would read as a window where
    only three metrics have a past."""
    if not any(step.history for step in request.metrics):
        return []
    ranked = sorted(request.metrics, key=_by_movement)
    shown = [step for step in ranked if step.history][:_MAX_HISTORY_BLOCKS]
    lines: list[str] = []
    for step in shown:
        subject = f"{step.metric} ({step.label})"
        if step.sub_detector:
            subject += f" [{step.sub_detector}]"
        lines.append("")
        lines += history_block(step.history, title=f"{subject} — ")
        lines += region_lines(step.regions)
    remaining = sum(1 for step in ranked if step.history) - len(shown)
    if remaining > 0:
        # Never "with a similar history": nothing checks that, and the model
        # would be entitled to read it as a statement that they agree.
        lines.append(
            f"  ({remaining} further metric(s) stepped in this window; their "
            f"full history tables are omitted here"
            + (
                ", and each is in the sweep table above.)"
                if request.sweep is not None
                else ", and each is summarised on its own line above.)"
            )
        )
    return lines


def _context_lines(request: RankRequest) -> list[str]:
    """The labelled run context — detector, sample, platform, release window.

    Spelled out line by line because one shared library can regress several
    detectors in the same window: each detector is ranked in its own call, and
    a terse header is too easy to under-weight against a large diff — the
    answer must be about *this* run, not the most prominent detector in the
    diff."""
    base_platform = request.base_platform or request.platform
    onset_platform = request.onset_platform or request.platform
    return [
        f"- Detector: {request.detector}",
        sample_line(request.sample),
        platform_line(request.platform),
        f"- Release window: "
        f"{window_phrase(request.base_release, request.onset_release)}",
        *(
            platform_switch_lines(base_platform, onset_platform)
            if base_platform != onset_platform else ()
        ),
    ]


def _unshown_steps(request: RankRequest) -> tuple[MetricStep, ...]:
    """The steps the sweep table cannot show — a sub-detector metric, or one
    outside the table's columns — which keep their own bullet."""
    shown = {m for metrics in SWEEP_METRICS.values() for m in metrics}
    return tuple(
        step for step in request.metrics
        if step.sub_detector or step.metric not in shown
    )


def _reach_lines(request: RankRequest) -> list[str]:
    """The candidates whose files are in this detector's geometry, each with
    the other benchmarked detectors it changed the same way and what those
    measured. Positive only: a change can reach every detector through a
    shared driver or material without touching one detector's files, so no
    line is written about the candidates that do not appear here."""
    this = request.detector
    by_detector: dict[str, list[ScopeSweep]] = {}
    for sweep in request.window_sweeps:
        by_detector.setdefault(sweep.detector, []).append(sweep)
    lines = []
    for candidate in request.candidates:
        touch = next((t for t in candidate.touches if t.detector == this), None)
        if touch is None:
            continue
        if touch.own_files:
            what = "changes this run's compact directory: " + "; ".join(
                [file_change_phrase(c) for c in touch.changes]
                + ([f"{touch.unread} file(s) with no readable hunk"] if touch.unread else [])
            )
        else:
            what = (
                f"changes {len(touch.tree_files)} file(s) elsewhere in this "
                f"detector's geometry tree, none in its compact directory"
            )
        line = f"  - {candidate.repo}#{candidate.number} {what}."
        for detector in touch.same_as:
            states = by_detector.get(detector, [])
            measured = "; ".join(
                f"{sweep.sample}: {scope_state(sweep)}" for sweep in states
            ) or "none of its runs measured this window"
            line += (
                f" It makes the same change to {detector}, which measured in "
                f"this window — {measured}."
            )
        others = [
            t.detector for t in candidate.touches
            if t.detector != this and t.own_files and t.detector not in touch.same_as
        ]
        if others:
            line += (
                f" It also changes the compact directories of {', '.join(others)} "
                f"— see its entry below."
            )
        lines.append(line)
    if not lines:
        return []
    return ["- Candidates whose changed files are in this detector's geometry:", *lines]


def _summary_lines(request: RankRequest) -> list[str]:
    """The evidence summary: this run's scope, the candidates reaching its
    geometry, and every other scope that measured the window."""
    scope = (request.detector, request.platform, request.sample)
    lines = [EVIDENCE_HEADER, ""]
    if request.sweep is not None:
        lines.append(f"This run — {scope_name(request.sweep)}:")
        ranked = sorted(request.metrics, key=_by_movement)
        rep = representative(
            request.sweep,
            [(step.label, step.metric, step) for step in ranked if step.history],
        )
        lines += scope_evidence_lines(
            request.sweep,
            history=rep[2].history if rep else None,
            history_metric=rep[1] if rep else "",
            history_label=rep[0] if rep else "baseline",
        )
    else:
        lines += _step_lines(request)
    lines += _reach_lines(request)
    other = other_scope_lines(request.window_sweeps, exclude={scope})
    if other:
        lines += ["", *other]
    return lines


def _detail_lines(request: RankRequest) -> list[str]:
    """The details behind the summary: the sweep table, any step it cannot
    show, the largest movers' histories — or, without a sweep, the
    non-confirming configurations."""
    lines: list[str] = []
    if request.sweep is not None:
        lines += sweep_table_lines(request.sweep)
        lines += _step_lines(request, _unshown_steps(request))
    lines += _history_lines(request)
    if request.sweep is None:
        lines += outcome_lines(request.outcomes, _MAX_OUTCOMES_LISTED)
    return lines


def _render_candidate(
    candidate: RankCandidate,
    diff_budget: int,
    geometry: str = "",
    own_dir: str = "",
    *,
    request: RankRequest | None = None,
) -> str:
    """One PR's prompt block.

    The number, title and file paths are always included; the diff is clipped
    to this candidate's *diff_budget* share, so a wide window degrades every
    oversized diff evenly rather than overflowing a small-context model."""
    size = f"+{candidate.additions}/-{candidate.deletions}"
    lines = [f"- #{candidate.number} — {candidate.title} ({size})"]
    if candidate.files:
        lines.append(f"  files: {format_files(candidate.files, _MAX_FILES_LISTED)}")
        if candidate.touches and request is not None:
            by_detector: dict[str, list[ScopeSweep]] = {}
            for sweep in request.window_sweeps:
                by_detector.setdefault(sweep.detector, []).append(sweep)
            lines += touch_lines(
                candidate.touches, by_detector, this_detector=request.detector,
            )
        else:
            # A fact, where the run recorded enough to state one: whether this
            # change lands in the geometry tree this detector actually loads.
            reach = geometry_reach(candidate.files, geometry, own_dir)
            if reach:
                lines.append(reach)
    lines += body_block(candidate.body, _MAX_BODY_CHARS)
    lines += diff_block(candidate.patch, diff_budget)
    return "\n".join(lines)


def _build_user_prompt(
    request: RankRequest,
    *,
    offer: tuple[HistoricalBoundary, ...] = (),
    offer_omitted: int = 0,
    evidence: HistoricalEvidence | None = None,
) -> str:
    """The user message: the run context, the evidence summary, the details
    behind it, then every candidate grouped by package, each with its fair
    share of the total diff budget — diffs last, so no amount of code can push
    the evidence out of the model's reading.

    *offer* appends the lightweight index of older boundaries the model may ask
    for, *offer_omitted* how many of them the index cap cut; *evidence* appends
    the analogues it did ask for. Never both an offer and evidence — an index in
    the follow-up prompt would be a second retrieval round, which the protocol
    does not have."""
    # The harness is not one of the tracked stack packages, so it never rides
    # in the moved/stood-still count — that number is a statement about the
    # simulation stack, and folding the harness in would quietly corrupt it.
    packages = sorted(
        {c.repo for c in request.candidates} - {request.harness_repo}
    )
    parts = [
        f"Run context — the {request.detector} run these metrics were "
        f"measured on; judge every candidate against it:",
        *_context_lines(request),
        "",
        *_summary_lines(request),
        "",
        "Details:",
        *_detail_lines(request),
        "",
        # The denominator bounds the search: whatever caused this is in the
        # packages that moved, or in something k4Bench does not track.
        f"{len(packages)} of {len(packages) + request.n_unchanged} tracked "
        f"package(s) moved across this window; the rest stood still.",
    ]
    if request.harness_repo:
        # Said whether or not any harness pull request survived as a candidate:
        # "the harness changed" is part of what the window's evidence has to be
        # weighed against either way.
        harness_shown = any(
            c.repo == request.harness_repo for c in request.candidates
        )
        parts.append(
            f"The benchmark harness itself ({request.harness_repo}) also "
            f"changed across this window"
            + (
                "; its pull requests are listed below."
                if harness_shown
                else "; none of its pull requests are listed as candidates."
            )
        )
    parts += [
        "",
        "Candidate pull requests, grouped by package — score each on its own:",
    ]
    # A candidate touching the compact directory this run loads is served
    # first: its diff sample leads with that directory's hunks, and an even
    # share on a wide window cuts them off before the line that matters.
    own_dir = compact_dir(request.geometry_tree)
    budgets = allocate_favoured_diff_budget(
        [len(c.patch) for c in request.candidates],
        [any(path_under(f, own_dir) for f in c.files) for c in request.candidates],
        _MAX_PROMPT_CHARS,
    )
    budget_for = dict(zip(request.candidates, budgets))

    by_repo: dict[str, list[RankCandidate]] = {}
    for candidate in request.candidates:
        by_repo.setdefault(candidate.repo, []).append(candidate)

    for repo, candidates in by_repo.items():
        parts.append("")
        parts.append(f"## {repo}")
        if repo == request.harness_repo:
            parts.append(HARNESS_PACKAGE_NOTE)
        for candidate in candidates:
            parts.append(_render_candidate(
                candidate, budget_for[candidate],
                geometry_tree(request.geometry_tree), own_dir, request=request,
            ))

    # After the candidates, so the current window is read first and the older
    # boundaries are met as what they are — background to it, never a second set
    # of candidates competing with it for the model's attention.
    parts += historical_offer_lines(offer, omitted=offer_omitted)
    if evidence is not None:
        parts += historical_lines(evidence.prs, asked=True)

    parts.append("")
    parts.append(
        f"Work in this order. First, from the evidence summary: did these "
        f"metrics really change, and in what shape — in the typical event, in a "
        f"few long events, outside the event loop, in memory — and is that the "
        f"series' own noise? Give that as step_assessment. Then, for each pull "
        f"request, ask what its diff changes that the {request.detector} run "
        f"with {request.sample} goes through — this detector's geometry, or "
        f"shared code the run executes — whether that mechanism predicts which "
        f"configurations moved and which did not, here and in the other "
        f"detectors it reaches, and what contradicts it; let that decide the "
        f"score, the reason and what argues against it."
    )
    parts.append(_RESPONSE_INSTRUCTION)
    historical = "" if evidence is None else (
        f", {len(evidence.prs)} historical analogue(s)"
    )
    return log_prompt_size(
        "rank", "\n".join(parts),
        detail=(
            f"{request.detector}/{request.sample} {request.onset_release}, "
            f"{len(request.candidates)} candidate(s), "
            f"{len(request.metrics)} metric(s)"
            + (f", {len(offer)} boundary(ies) offered" if offer else "")
            + historical
        ),
    )


# ── Defensive response parsing ────────────────────────────────────────────────

def _parse_rankings(
    content: str, request: RankRequest
) -> dict[tuple[str, int], Ranking]:
    """Turn the model's reply into ``{(repo, number): Ranking}``.

    Enforces the only-reorder rule here as well as in the builder: a
    ``(repo, pr)`` the request did not contain is dropped, so no invented PR can
    reach the caller. Any shape drift — not an object, no ``rankings`` list, a
    row missing ``repo``/``pr``, a ``likelihood`` that is not a number — yields
    ``{}`` or skips that row rather than raising; a skipped row surfaces as a
    missing candidate for the retry/coverage machinery, never as a made-up
    score."""
    known = {(c.repo, c.number) for c in request.candidates}
    data = extract_json(content)
    if not isinstance(data, dict):
        return {}
    rows = data.get("rankings")
    if not isinstance(rows, list):
        return {}

    out: dict[tuple[str, int], Ranking] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        repo = row.get("repo")
        number = coerce_int(row.get("pr"))
        if repo is None or number is None:
            continue
        key = (str(repo), number)
        if key not in known:
            continue  # only-reorder: never surface a PR the input didn't hold
        score = parse_score(row.get("likelihood"))
        if score is None:
            continue  # unreadable likelihood: reject the row, don't publish 0%
        description = one_line(row.get("reason"), _MAX_DESCRIPTION_CHARS)
        if not description:
            # The contract demands a reason for every judgement — a bare score
            # is indistinguishable from an unranked default downstream.
            # Rejecting the row leaves the candidate "missing", which triggers
            # the follow-up attempt instead of dooming the sidecar at the
            # coverage gate.
            continue
        out[key] = Ranking(
            score=score,
            description=description,
            # Optional, deliberately: the counter-evidence sharpens a judgement
            # but is not the judgement. Rejecting a scored, reasoned row for
            # lacking it would trade a real ranking for a stylistic one, and
            # leave the candidate reading as "never judged".
            against=one_line(row.get("against"), _MAX_AGAINST_CHARS),
        )
    return out


#: The evidence reading is logged, not stored, and a clause longer than this is
#: an essay the contract did not ask for.
_MAX_READING_CHARS = 800


def _parse_reading(content: str) -> str:
    """The model's reading of the evidence summary, or ``""``."""
    data = extract_json(content)
    if not isinstance(data, dict):
        return ""
    return one_line(data.get("evidence_reading"), _MAX_READING_CHARS)


def _asks_again(content: str) -> bool:
    """Whether a reply carries a historical request the protocol cannot honour.

    Only a *present, non-null* member counts. A model echoing the field back as
    ``null`` — which JSON-mode models do with a schema they were shown once — is
    saying it wants nothing, and reading that as a refusal to judge would throw
    away a good ranking over a punctuation habit.
    """
    data = extract_json(content)
    return isinstance(data, dict) and data.get(REQUEST_KEY) is not None


def _parse_assessment(content: str) -> StepAssessment | None:
    """The model's read of the step itself, or ``None`` when it gave none.

    ``None`` is a first-class outcome — an older model, a reply that skipped the
    field, a verdict outside :data:`~k4bench.blame.prompt.ASSESSMENT_VALUES` —
    and every consumer treats it as "not assessed", never as "real change". The
    field exists to let a model say a step is noise; inventing a default would
    put a word in its mouth in exactly the direction the field was added to
    avoid.
    """
    data = extract_json(content)
    if not isinstance(data, dict):
        return None
    raw = data.get("step_assessment")
    if isinstance(raw, str):
        verdict, reason = raw, ""  # a model that answered with the bare verdict
    elif isinstance(raw, dict):
        verdict = str(raw.get("verdict") or "")
        reason = one_line(raw.get("reason"), _MAX_DESCRIPTION_CHARS)
    else:
        return None
    verdict = verdict.strip().lower().replace(" ", "_").replace("-", "_")
    if verdict not in ASSESSMENT_VALUES:
        return None
    return StepAssessment(verdict=verdict, reason=reason)
