"""Decide which pull requests hear about a regression, and what they are told.

The nightly ranker already answers "which PR most likely caused this step"
(:mod:`k4bench.blame.rank`), but that answer only reaches people who read the
e-group mail or open the dashboard — never the author of the change. This module
turns a night's ``report.json`` + ``blame.json`` into a set of pull-request
comments; :mod:`k4bench.blame.publish` posts them.

The work happens in two halves, because they have different failure domains:

* :func:`select` is **pure** — no network, no token, no clock — so the whole
  "who gets told what" decision is unit-testable, and the CLI can print exactly
  what would be posted (``--dry-run``) without touching GitHub.
* :func:`build_comments` renders those selections, and is where the optional
  second model pass (:mod:`k4bench.blame.attribute`) and the diff fetch it needs
  arrive — as *injected callables*, the same seam
  :mod:`k4bench.blame.builder` uses for its ranker.

Commenting in someone else's repository is an outward-facing act on the strength
of a model's judgement, so the gates are deliberately narrow and all of them
must pass:

* the candidate's repository is on the **allowlist** — an empty allowlist means
  the bot is inert;
* the ranker actually **judged** the candidate — an unranked one has no opinion
  attached to it, and no threshold is low enough to be cleared by a missing
  judgement;
* the ranker's likelihood is at or above ``min_score`` (default 80);
* the pull request is **merged** — an open PR cannot have shipped in a release;
* the blame entry's candidate discovery was **complete**
  (:attr:`~k4bench.blame.models.BlameEntry.discovery_incomplete`) — naming one PR
  out of a knowingly partial set is exactly the overclaim the ranker itself
  refuses to make;
* the night is under the ``max_comments`` cap — a storm is a bug, not a night;
* and, when a cross-configuration review ran, it did not acquit the pull request
  outright (:func:`build_comments`'s withdrawal gate).

One comment covers one ``(pull request, change window)`` pair — the reader's
question is "did my change do this?", asked once — and :func:`marker_for` gives
that pair a stable hidden key so a later night edits the existing comment
instead of posting a second one. Everything the window regressed goes into a
single table ordered by attribution likelihood, across every detector, sample,
platform and benchmark configuration: which configurations moved *and which did
not* is the substance of the claim, so it is one table a reader can scan rather
than one configuration in full and the rest in a footnote.

A comment is written once and thereafter only *edited*, never retracted: when
the regression resolves, or the candidate drops below ``min_score``, tonight's
selection simply stops producing it and the comment already on the pull request
is left exactly as it stands. That is deliberate. It records what the benchmarks
saw at the time — which remains true after the metric recovers — and silently
rewriting or deleting a comment people may have replied to is worse than leaving
a dated one in place. Follow-ups belong in the thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from k4bench.blame.attribute import (
    MAX_COMPETITORS,
    Attribution,
    Attributor,
    AttributionRequest,
    CompetingPR,
    PackageChangeFact,
    RegressionFact,
    ScopeCandidateState,
    ScopeOutcome,
    competitor_order,
)
from k4bench.blame.models import (
    RANKING_DISCLOSURE,
    BlameEntry,
    BlameReport,
    CandidatePR,
)
from k4bench.labels import pretty_platform, pretty_sample
from k4bench.regression.models import MetricVerdict, NightlyReport, Severity
from k4bench.regression.render import (
    regression_href,
    stack_changes_href,
    window_href,
    window_token,
)

_log = logging.getLogger(__name__)

#: Marker format version. Bumping it makes every existing comment invisible to
#: the upsert (a *new* comment is posted rather than the old one edited), so it
#: changes only when a body is no longer an in-place successor of the old one.
MARKER_VERSION = "v1"

#: Regression rows shown before the table folds into a disclosure, rows kept
#: inside that disclosure, and candidate rows shown for the rest of the window.
#: All three are display caps: the selection above them is complete, and every
#: row is still scored — only the rendering is bounded.
#:
#: The folded rows are capped rather than complete because a detector-removal
#: sweep confirms one row per removed sub-detector — a single night can carry
#: three hundred, nearly all repeating the same movement — and because a comment
#: over GitHub's 65,536-character limit is rejected outright. The dashboard link
#: on every row is where the full set lives.
_MAX_TABLE_ROWS = 5
_MAX_FOLDED_ROWS = 25
_MAX_OTHER_CANDIDATES = 5

#: Likelihood points between this PR and the closest other candidate at or
#: under which the ranking is called a weak preference in words. Wide enough to
#: catch a genuinely crowded field, narrow enough that an ordinary night — where
#: the ranker picked one PR out of the pack — says nothing extra.
_CROWDED_SPREAD = 10.0

#: Longest model explanation quoted verbatim — the per-configuration ranker's
#: one-liner, or the cross-configuration review's short paragraph. Both contracts
#: ask for less than this; a model that ignores its contract must not be able to
#: paste an essay into someone's pull request.
_MAX_DESCRIPTION_CHARS = 400
_MAX_SUMMARY_CHARS = 700

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

#: Metric names named in a "moved but did not confirm" outcome line. Enough to
#: show *what* is drifting there, not so many that the negative evidence turns
#: into a second report.
_MAX_WATCHED_METRICS = 6

#: Where the comment's own footer points: the page describing how a regression
#: is attributed to a pull request, on the published docs site (``site_url`` in
#: mkdocs.yml + the page's nav path).
_METHOD_URL = "https://key4hep.github.io/k4Bench/user-guide/features/pr-comments/"

_DEFAULT_MIN_SCORE = 80.0
_DEFAULT_MAX_COMMENTS = 10


class CommentConfigError(ValueError):
    """The comment config is not shaped like a :class:`CommentPolicy`.

    Raised rather than defaulted: every field here decides whether — and where —
    the bot writes to a repository it does not own, so a typo must stop the
    step, never silently widen or narrow its reach."""


class CommentStormError(RuntimeError):
    """More comments than ``max_comments`` — the attribution is suspect, so the
    whole night is suppressed rather than posting the loudest few accusations
    into repositories k4Bench does not own.

    Raised rather than returned as an empty list so a caller can tell a *tripped
    circuit breaker* (something is wrong with tonight's attribution) apart from
    an *ordinary quiet night* (nothing crossed the threshold) — the two look
    identical in the comment count but mean opposite things to whoever is
    watching the bot."""

    def __init__(self, count: int, cap: int, targets: list[str]):
        self.count = count
        self.cap = cap
        self.targets = tuple(targets)
        super().__init__(
            f"{count} comments exceed the max_comments cap of {cap}: "
            + ", ".join(targets)
        )


@dataclass(frozen=True)
class CommentPolicy:
    """Who may be commented on, and how confidently.

    ``repos`` holds lowercase ``owner/repo`` slugs; GitHub slugs are
    case-insensitive, so matching is done on the lowered form while the
    candidate's own spelling is what gets displayed. An empty ``repos`` disables
    the bot entirely.
    """

    min_score: float = _DEFAULT_MIN_SCORE
    repos: frozenset[str] = frozenset()
    max_comments: int = _DEFAULT_MAX_COMMENTS

    @property
    def enabled(self) -> bool:
        return bool(self.repos)

    def targets(self, candidate: CandidatePR) -> bool:
        """True when *candidate* is a pull request the bot may write to **at
        all** — the repo and merged gates.

        Both are properties of the pull request itself rather than of any one
        regression, so a candidate that fails either can never be commented on,
        however it scores — which is why :meth:`allows` composes this gate with
        the judgement-dependent ones rather than repeating them."""
        return candidate.repo.lower() in self.repos and bool(candidate.merged_at)

    def allows(self, candidate: CandidatePR) -> bool:
        """True when *candidate* clears the repo, merged, ranked and score gates
        — i.e. when this judgement is strong enough to *cause* a comment.

        :attr:`~k4bench.blame.models.CandidatePR.ranked` is checked before the
        score and is not redundant with it. An unranked candidate carries
        ``score`` 0.0 as a placeholder, and a ``min_score`` of 0 — which the
        config accepts — would otherwise let *every* unjudged pull request in an
        allowlisted repository be commented on, on the strength of a model
        opinion that was never given. No threshold can be low enough to be
        cleared by the absence of a judgement."""
        return (
            self.targets(candidate)
            and candidate.ranked
            and math.isfinite(candidate.score)
            and candidate.score >= self.min_score
        )

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> CommentPolicy:
        """Build a policy from the parsed ``.github/blame-comments.yml``.

        Unknown keys, wrong types and out-of-range values raise
        :class:`CommentConfigError` — see the class docstring for why this one
        config is strict where the report schemas are forgiving.
        """
        if data is None:
            data = {}
        if not isinstance(data, dict):
            # A falsey-but-present document (``false``, ``0``, ``[]``) is
            # malformed, not "no config": only an absent one defaults to inert.
            raise CommentConfigError("comment config must be a mapping")
        unknown = set(data) - {"min_score", "max_comments", "repos"}
        if unknown:
            raise CommentConfigError(f"unknown key(s): {', '.join(sorted(unknown))}")

        min_score = _positive_number(
            data.get("min_score", _DEFAULT_MIN_SCORE), "min_score"
        )
        if min_score > 100:
            raise CommentConfigError("min_score must be between 0 and 100")
        max_comments = _positive_int(
            data.get("max_comments", _DEFAULT_MAX_COMMENTS), "max_comments"
        )

        raw_repos = data.get("repos", [])
        if raw_repos is None:  # `repos:` with no value is an empty allowlist
            raw_repos = []
        if not isinstance(raw_repos, list):
            # ``repos: false`` / ``repos: k4geo`` — a scalar is not an allowlist.
            raise CommentConfigError("repos must be a list of owner/repo slugs")
        repos = set()
        for slug in raw_repos:
            # Validate the *stripped* slug: ``"owner/ "`` must not slip through
            # the slash check and then be stored as the truncated ``"owner/"``.
            cleaned = slug.strip() if isinstance(slug, str) else slug
            if not isinstance(cleaned, str) or cleaned.count("/") != 1 \
                    or cleaned.startswith("/") or cleaned.endswith("/"):
                raise CommentConfigError(f"not an owner/repo slug: {slug!r}")
            repos.add(cleaned.lower())
        return cls(min_score=min_score, repos=frozenset(repos), max_comments=max_comments)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CommentConfigError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise CommentConfigError(f"{name} must be a non-negative number")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    """A count that must be a whole number, at least one. A float like ``2.9``
    is a typo, not a rounding hint — silently truncating it would post one fewer
    comment than the config appears to ask for; a zero disables the bot in a way
    an empty ``repos`` already expresses more honestly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CommentConfigError(f"{name} must be a whole number")
    if value < 1:
        raise CommentConfigError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PRComment:
    """One rendered comment and where it goes.

    ``marker`` is the hidden key the upsert recognises and ``facts_digest``
    fingerprints the benchmark facts behind the body; both are hidden lines at
    the top of ``body``, so a comment always carries the keys that identify it
    and the state it was rendered from.
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


def marker_for(base_release: str | None, onset_release: str | None) -> str:
    """The hidden HTML key identifying a comment about one change window.

    The window is the identity because it is what the comment is *about*: the
    same PR implicated in a genuinely different window gets its own comment,
    while tonight's re-confirmation of the same window edits the one already
    there. Reuses :func:`~k4bench.regression.render.window_token` so the key and
    the dashboard link it carries name the window identically.
    """
    return (
        f"<!-- k4bench-blame-comment:{MARKER_VERSION} "
        f"window={window_token(base_release, onset_release)} -->"
    )


#: Prefix of the second hidden line — see :func:`_facts_digest` and
#: :func:`facts_digest_of`.
_FACTS_MARKER_PREFIX = "<!-- k4bench-blame-facts:"


def facts_digest_of(body: str) -> str:
    """The facts digest carried by an already-posted comment, or ``""``.

    The read half of :func:`_facts_digest`, used by
    :func:`k4bench.blame.publish._upsert` to decide whether a differing body
    represents a real change. A body carrying no readable digest line returns
    ``""``, and the caller falls back to comparing whole bodies."""
    for line in body.split("\n", 3)[:3]:
        line = line.strip()
        if line.startswith(_FACTS_MARKER_PREFIX) and line.endswith("-->"):
            return line[len(_FACTS_MARKER_PREFIX):-3].strip()
    return ""


# ── Selection ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegressionRow:
    """One confirmed regression this pull request is being asked about — one row
    of the comment's table.

    ``scope_state`` is what the first pass knew about this pull request *in this
    row's own scope* (:data:`~k4bench.blame.attribute.ScopeCandidateState`), and
    ``scope_score``/``scope_reason`` carry the ranker's judgement when that state
    is ``"ranked"`` — the prior the review starts from, and the likelihood shown
    when no review runs. Every other state leaves ``scope_score`` at ``None``:
    the row is real evidence about the window (it is why it is collected at all),
    but nobody scored this pull request against it, and a 0% in that cell would
    be an accusation's worth of difference from the truth.

    ``fact_id`` is the opaque handle the review echoes back (see
    :class:`~k4bench.blame.attribute.RegressionFact`); it is assigned in identity
    order, never score order, so the same night's rows always carry the same ids.
    ``stack`` is the group's Key4hep release *directory* — the dashboard's
    ``?stack=`` vocabulary, kept per row so each link names the stack that row
    actually ran against.
    """

    verdict: MetricVerdict
    stack: str
    fact_id: str = ""
    scope_score: float | None = None
    scope_reason: str = ""
    scope_state: ScopeCandidateState = "discovery_incomplete"

    @property
    def scope(self) -> tuple[str, str, str]:
        return (self.verdict.detector, self.verdict.platform, self.verdict.sample)


@dataclass
class CommentPlan:
    """Everything one comment is decided from: one pull request, one change
    window, every regression of that window, and the evidence around them.

    ``outcomes`` is the negative evidence — the configurations that measured the
    same window and did *not* confirm — which only :func:`select` can compute,
    because only the report knows which groups ran at all.

    ``rows`` holds **every** confirmed regression whose onset falls in this
    window, whatever this pull request's standing in each — scored badly,
    unscored, or not a candidate there at all; ``selected`` records whether one
    complete, ranked judgement scored high enough to warrant the comment. The two
    are deliberately separate — see :func:`select`.

    A plan is keyed by pull request and window, never by platform, so its rows
    can come from several build platforms at once. The release diff is therefore
    kept per platform in ``package_facts`` and ``unchanged``, and handed to the
    review that way: two platforms' diffs are two measurements, not one.
    """

    repo: str
    number: int
    subject: CandidatePR
    base_release: str | None
    onset_release: str
    rows: list[RegressionRow] = field(default_factory=list)
    others: dict[tuple[str, int], CandidatePR] = field(default_factory=dict)
    outcomes: tuple[ScopeOutcome, ...] = ()
    #: ``platform -> {(package, status): compare_url}`` — one release diff per
    #: platform, exactly as provenance recorded it.
    package_facts: dict[str, dict[tuple[str, str], str | None]] = field(
        default_factory=dict
    )
    #: ``platform -> tracked packages that stood still`` on that platform.
    unchanged: dict[str, int] = field(default_factory=dict)
    selected: bool = False

    @property
    def target(self) -> str:
        return f"{self.repo}#{self.number}"

    @property
    def packages_by_platform(self) -> dict[str, tuple[PackageChangeFact, ...]]:
        """The window's package diff, per platform, as the review is shown it.

        Never unioned across platforms. Provenance is read per platform, so a
        package can move on one and stand still on another, or move to a
        different status — and a union paired with a single unchanged count
        would quote a "N of M tracked" ratio that no platform ever measured."""
        return {
            platform: tuple(
                PackageChangeFact(package=package, status=status, compare_url=url)
                for (package, status), url in sorted(facts.items())
            )
            for platform, facts in sorted(self.package_facts.items())
        }

    @property
    def platforms(self) -> set[str]:
        return {row.verdict.platform for row in self.rows}

    @property
    def top_score(self) -> float:
        """The strongest per-configuration likelihood across this window — what
        selection was made on, and how comments are ordered. Only rows the first
        pass actually judged carry one."""
        return max(
            (row.scope_score for row in self.rows if row.scope_score is not None),
            default=0.0,
        )

    @property
    def scopes(self) -> set[tuple[str, str, str]]:
        return {row.scope for row in self.rows}


def select(
    report: NightlyReport,
    blame: BlameReport,
    policy: CommentPolicy,
) -> list[CommentPlan]:
    """The comments this night warrants, worst-first — decided, not yet rendered.

    Driven from the *report*'s confirmed regressions rather than from the
    sidecar's entries, so a comment can only ever describe a regression that is
    confirmed in tonight's report — a stale entry has nothing to attach to.

    Selection and evidence collection are **two passes**, and the order matters.

    *Phase one* finds the targets: a ``(repo, number, base, onset)`` for which
    some blame entry — one with complete candidate discovery — carries a ranked
    first-pass judgement of a merged, allowlisted pull request at or above
    ``min_score``. Nothing else can cause a comment.

    *Phase two* then rebuilds each target's evidence from the **whole window**,
    not from the entries the pull request happened to appear in. Every confirmed
    regression whose onset falls inside the window becomes a row, and each row
    records what the first pass knew about the subject *there*
    (:data:`~k4bench.blame.attribute.ScopeCandidateState`):

    * it was a candidate and was scored — the prior the review starts from;
    * it was a candidate and was not scored — unknown, and never a zero;
    * it was **not** a candidate and the search there was complete — the
      strongest exculpatory evidence this pipeline produces, and precisely the
      row a one-pass collection loses, because a pull request absent from a
      scope's candidate list never reaches the loop that would collect it;
    * discovery there was incomplete (or the sidecar has no entry for it at
      all) — nothing follows from absence, and the row says so.

    Collecting rows by candidacy — the obvious shape — is wrong in both
    directions. It drops the row where the subject scored 30 (the exculpatory
    half of an accusation that scored 92 elsewhere), and it drops the row where
    the subject is not a candidate at all: "IDEA regressed in the same window and
    this change is not even in the range that produced it" is the single most
    useful thing the review can be told, and it lives in an entry the subject
    does not appear in. Neither would resurface as negative evidence either —
    :func:`_outcomes_for` correctly refuses to call a configuration clean when it
    confirmed a step in this window, so a dropped row is invisible, not demoted.

    An incomplete scope is **represented, never suppressed**: it renders and
    prompts as "no conclusion available here". Suppressing the whole comment for
    it was the other candidate rule, and was rejected — the accusation itself
    already requires a *complete* scope to have cleared the threshold, so an
    unrelated truncated range on some other detector adds no risk of a false
    claim, while silencing on it would let one force-pushed branch anywhere in
    the stack mute a well-evidenced comment. What is not acceptable is dropping
    such a scope silently, which is what makes this a stated state rather than a
    filter.

    Overshooting ``max_comments`` raises :class:`CommentStormError` rather than
    returning a truncated list — a night that loud is a bug, not a night, and
    blind-posting ten accusations into repositories k4Bench does not own is the
    exact harm the gates exist to prevent. It is raised, not returned empty, so
    the caller can tell it apart from an ordinary night that simply implicated
    no one.
    """
    if not policy.enabled:
        return []

    # Resolved once and shared by both passes: the join is a linear scan of the
    # sidecar, and a wide night (a removal sweep confirms hundreds of rows) would
    # otherwise repeat it for every plan.
    confirmed = [
        (verdict, stack, blame.entry_for(verdict))
        for verdict, stack in _confirmed_rows(report)
    ]
    plans = _targets(confirmed, policy)
    for plan in plans:
        _collect_window(confirmed, plan)
        plan.outcomes = _outcomes_for(report, plan)

    ordered = sorted(plans, key=lambda p: (-p.top_score, p.repo, p.number))
    if len(ordered) > policy.max_comments:
        _log.warning(
            "select: %d comments exceed the max_comments cap of %d — a night this "
            "loud is a bug, not a night; posting none of them",
            len(ordered), policy.max_comments,
        )
        raise CommentStormError(
            len(ordered), policy.max_comments, [p.target for p in ordered]
        )
    return ordered


def _confirmed_rows(report: NightlyReport) -> Iterator[tuple[MetricVerdict, str]]:
    """Every confirmed regression in tonight's report, with its group's release.

    Walked group by group rather than through ``report.regressions`` so each
    verdict keeps its group's release *directory* — the dashboard links and the
    clean-control comparison both need it."""
    for group in report.groups:
        for verdict in group.regressions:
            yield verdict, group.k4h_release


#: One confirmed regression as both passes read it: the verdict, the release
#: directory its group ran, and the sidecar entry attributing it (``None`` when
#: the sidecar has none — an unattributable window, or missing provenance).
_Confirmed = tuple[MetricVerdict, str, "BlameEntry | None"]


def _targets(
    confirmed: list[_Confirmed], policy: CommentPolicy
) -> list[CommentPlan]:
    """Phase one: the ``(pull request, window)`` pairs a comment may be made
    about, and nothing about what those comments will say.

    A target needs one *complete* first-pass judgement clearing every gate —
    allowlisted repo, merged, ranked, at or above ``min_score``, from an entry
    whose candidate discovery was complete. Evidence is gathered afterwards
    (:func:`_collect_window`), so no row can widen or narrow the field here."""
    plans: dict[tuple[str, int, str | None, str], CommentPlan] = {}
    for _verdict, _stack, entry in confirmed:
        if entry is None or entry.discovery_incomplete:
            continue
        for candidate in entry.candidates:
            if not policy.allows(candidate):
                continue
            key = (
                candidate.repo.lower(), candidate.number,
                entry.base_release, entry.onset_release,
            )
            plan = plans.get(key)
            if plan is None:
                plans[key] = CommentPlan(
                    repo=candidate.repo, number=candidate.number,
                    subject=candidate,
                    base_release=entry.base_release,
                    onset_release=entry.onset_release,
                    selected=True,
                )
            elif candidate.score > plan.subject.score:
                # Every metric of a run group shares one ranking, so these are
                # equal in valid builder output; keep the strongest defensively
                # so the identity rendered never depends on which metric was
                # walked first.
                plan.subject = candidate
    return list(plans.values())


def _collect_window(confirmed: list[_Confirmed], plan: CommentPlan) -> None:
    """Phase two: fill *plan* with the whole window's evidence.

    Every confirmed regression whose onset falls inside the window is a row,
    whatever the subject's standing in it — that is what makes this a review of
    the window rather than of the accusation. The same predicate
    (:func:`_steps_in_window`) decides here and in :func:`_outcomes_for`, so the
    two partition the night exactly: a configuration that stepped in this window
    is a row, and one that did not is a candidate control. Nothing falls between
    them."""
    window = (plan.base_release, plan.onset_release)
    ident = (plan.repo.lower(), plan.number)
    for verdict, stack, entry in confirmed:
        if not _steps_in_window(verdict, window):
            continue
        state, candidate = _scope_state(entry, ident)
        if entry is not None:
            _record_packages(plan, entry, verdict.platform)
            _record_others(plan, entry, ident)
        plan.rows.append(RegressionRow(
            verdict=verdict, stack=stack, scope_state=state,
            scope_score=candidate.score if candidate is not None else None,
            scope_reason=candidate.description if candidate is not None else "",
        ))
    # Ids ride on identity order so they are reproducible from the plan alone: a
    # night re-run must ask the model about "r3" and mean the same regression it
    # meant last time.
    plan.rows.sort(key=_row_identity)
    plan.rows = [
        RegressionRow(
            verdict=row.verdict, stack=row.stack, fact_id=f"r{index}",
            scope_score=row.scope_score, scope_reason=row.scope_reason,
            scope_state=row.scope_state,
        )
        for index, row in enumerate(plan.rows, start=1)
    ]


def _scope_state(
    entry: BlameEntry | None, ident: tuple[str, int]
) -> tuple[ScopeCandidateState, CandidatePR | None]:
    """What the first pass knew about *ident* in one regression's own scope.

    No entry at all is read as ``"discovery_incomplete"``, not as absence: the
    sidecar carries no entry when provenance was missing or the window was not
    attributable, and in none of those cases was a candidate population ever
    established. Only a *complete* entry that does not list the pull request
    licenses the claim that it was not in the range."""
    if entry is None or entry.discovery_incomplete:
        return "discovery_incomplete", None
    candidate = next(
        (
            c for c in entry.candidates
            if (c.repo.lower(), c.number) == ident
        ),
        None,
    )
    if candidate is None:
        return "not_candidate", None
    if not candidate.ranked:
        return "unranked", None
    return "ranked", candidate


def _record_others(plan: CommentPlan, entry: BlameEntry, ident: tuple[str, int]) -> None:
    """Fold one entry's other candidates into the field this comment is weighed
    against — including from scopes the subject was never a candidate in, which
    is where the alternative that fits the evidence better often lives.

    A candidate seen twice keeps its judged reading: a ranked sighting beats an
    unranked one whatever the scores say, since an unranked one carries no score
    at all."""
    for other in entry.candidates:
        other_ident = (other.repo.lower(), other.number)
        if other_ident == ident:
            continue
        previous = plan.others.get(other_ident)
        if previous is None or _candidate_rank(other) > _candidate_rank(previous):
            plan.others[other_ident] = other


def _candidate_rank(candidate: CandidatePR) -> tuple[bool, float]:
    return (candidate.ranked, candidate.score)


def _record_packages(plan: CommentPlan, entry: BlameEntry, platform: str) -> None:
    """Fold one entry's release diff into *plan*, under the platform it was read
    on.

    Called for every row, not only the first: the entries behind one comment can
    come from different platforms, and their package sets are read from
    per-platform provenance. Repeats are free — an entry re-seen on a platform
    already recorded adds nothing."""
    facts = plan.package_facts.setdefault(platform, {})
    for repo in entry.repos:
        facts.setdefault((repo.package, repo.status), repo.compare_url)
    # Every entry of one platform in one window is read from the same diff, so
    # this is the same count each time; taking the smallest keeps a surprise in
    # the sidecar from inflating a claim about what stood still.
    plan.unchanged[platform] = min(
        plan.unchanged.get(platform, entry.n_unchanged), entry.n_unchanged
    )


def _outcomes_for(
    report: NightlyReport, plan: CommentPlan
) -> tuple[ScopeOutcome, ...]:
    """The benchmark configurations that measured this window and did **not**
    confirm.

    The negative evidence the cross-configuration review turns on: "ALLEGRO
    moved and IDEA did not" is only readable if the *did not* is stated, and so
    is the sharper within-detector version — "baseline moved and without_HCAL
    did not" — which is why a configuration, not a run group, is the unit here.
    Excluding a whole group because one of its configurations regressed would
    delete exactly the control the prompt asks the model to reason from: the
    baseline that stepped and the detector-removal run that did not live in the
    *same* group.

    A configuration counts only when it genuinely produced a clean measurement
    to compare against — it ran the same release the regressed rows were
    measured on, its host was judged reliable, its group had no job failure, it
    recorded no metric failure of its own, it confirmed no step inside this
    window, and at least one of its metrics could actually be judged. Everything
    else is silence from a run that did not happen or cannot be read, and silence
    must never be rendered as evidence of absence: ``reliable is None`` means *no
    evidence either way*, so it is treated like an unreliable run rather than
    like a clean one, and a metric with too little history to judge
    (``UNKNOWN``) is unread rather than flat — it never contributes to the clean
    verdict, and the ones that remain are counted onto the outcome so the prompt
    can state the gap."""
    window = (plan.base_release, plan.onset_release)
    regressed = plan.scopes
    stacks = {row.stack for row in plan.rows}
    outcomes = []
    for group in report.groups:
        if group.reliable is not True or group.job_failures:
            continue  # a run that cannot be trusted is not a clean result
        # The comparison that matters is against the *same measurement* the
        # regressed rows came from — the release this night ran, which is
        # generally long past the window's onset (a step that entered on
        # 2026-06-25 is still being re-measured on 2026-06-27). Comparing
        # against the onset release instead would find nothing, since no group
        # in tonight's report measured it. A group that ran a different release
        # than the regressed rows is not a like-for-like control.
        if group.k4h_release not in stacks:
            continue
        by_label: dict[str, list[MetricVerdict]] = {}
        for verdict in group.verdicts:
            by_label.setdefault(verdict.label, []).append(verdict)
        for label, verdicts in by_label.items():
            if any(_steps_in_window(v, window) for v in verdicts):
                continue  # stepped in this very window — it is not a control
            if any(v.severity is Severity.FAILURE for v in verdicts):
                continue  # a configuration that partly failed did not run clean
            unjudged = sum(1 for v in verdicts if v.severity is Severity.UNKNOWN)
            if unjudged == len(verdicts):
                # Nothing here was judged at all — the configuration ran, but
                # every metric is still warming up. "No evidence" rendered as
                # "did not move" is the false control this whole function is
                # written to avoid.
                continue
            watched = tuple(sorted(
                {v.metric for v in verdicts if v.severity is Severity.WATCH}
            ))[:_MAX_WATCHED_METRICS]
            outcomes.append(ScopeOutcome(
                detector=group.detector, platform=group.platform,
                sample=group.sample, label=label,
                status="watch" if watched else "clean", watched=watched,
                unjudged=unjudged,
            ))
    # Controls from a run group that *did* regress first: those are the
    # like-for-like comparisons — same detector, same sample, same platform,
    # same night — and the prompt lists only the first
    # :data:`~k4bench.blame.attribute._MAX_OUTCOMES_LISTED` of these.
    return tuple(sorted(
        outcomes,
        key=lambda o: (
            (o.detector, o.platform, o.sample) not in regressed,
            o.detector, o.sample, o.platform, o.label,
        ),
    ))


def _steps_in_window(
    verdict: MetricVerdict, window: tuple[str | None, str]
) -> bool:
    """Does *verdict* place a confirmed step inside this comment's window?

    Onset, not the whole ``(base, onset)`` pair: the base is *inferred* per
    metric series — the last release that metric was settled on — so the same
    step can be reported against different bases by different metrics, and
    requiring both to match would read a configuration that stepped on exactly
    this release as one that never moved. Anything that stepped strictly inside
    the window is ours too; a step onsetting after it left this window flat and
    is still a control for it. An unplaceable onset counts as inside, because a
    step nobody can date is not evidence of flatness."""
    if verdict.severity is not Severity.CONFIRMED:
        return False
    base, onset = window
    at = verdict.onset_run_date
    if at is None:
        return True
    return at <= onset and (base is None or at > base)


# ── Building ──────────────────────────────────────────────────────────────────

#: How a caller supplies one pull request's diff — ``(repo, number) -> patch``,
#: empty when it could not be fetched. Injected rather than imported so this
#: module stays free of the network, and so the CLI can memoize a night's
#: fetches (one window's subject is another's competitor).
PatchFor = Callable[[str, int], str]


def build_comments(
    plans: list[CommentPlan],
    *,
    attributor: Attributor | None = None,
    patch_for: PatchFor | None = None,
    dashboard_url: str | None = None,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> list[PRComment]:
    """Render *plans*, reviewing each against the whole window if it can.

    With an *attributor* configured, every plan gets one cross-configuration
    review (:mod:`k4bench.blame.attribute`) whose likelihoods order and fill the
    table. What the review cannot do is *widen*: it never introduces a pull
    request selection did not already implicate, and the only outcome it adds is
    withdrawal — a plan is dropped when no row is left standing at or above
    *min_score*. Within a plan it is a genuine second opinion, and an individual
    row's likelihood may come back higher than the first pass's as well as
    lower: that pass scored the row without ever seeing the other
    configurations, which is the deficiency this one exists to correct. A plan
    the review declines — no model, a failed call, an unusable reply — still
    renders from the per-configuration scores it already had.

    "Left standing" is the operative phrase: the gate reads each row's
    *effective* likelihood, the review's score where it gave one and the
    per-configuration score where it did not. A partial reply is an accepted
    outcome (:func:`~k4bench.blame.attribute._parse_attribution`), so measuring
    withdrawal on the review's scores alone would let one low answer about one
    row acquit a pull request the review never disputed on the others.

    The rendered body carries nothing that varies from night to night (no run
    URL, no report-night query param): a regression that stands for a week is
    one comment, and the upsert must see unchanged *facts* so it edits nothing
    and re-notifies no one.
    """
    comments = []
    for plan in plans:
        attribution = _review(plan, attributor=attributor, patch_for=patch_for)
        # Measured on what the table will actually show, not on the review's
        # own scores: a partial reply leaves the rows it omitted at their
        # per-configuration likelihood (:func:`_likelihood`), and a row the
        # review never spoke about must not be able to withdraw a comment the
        # ranker put at 91% and the review left standing.
        effective_top = max(
            (
                likelihood for row in plan.rows
                if (likelihood := _likelihood(row, attribution)) is not None
            ),
            default=0.0,
        )
        if attribution is not None and effective_top < min_score:
            _log.info(
                "build_comments: %s withdrawn — the cross-configuration review "
                "left every regression under %g%% (highest %.0f%%)",
                plan.target, min_score, effective_top,
            )
            continue
        comments.append(_render(plan, attribution, dashboard_url=dashboard_url))
    return comments


def _review(
    plan: CommentPlan,
    *,
    attributor: Attributor | None,
    patch_for: PatchFor | None,
) -> Attribution | None:
    """One plan's cross-configuration review, or ``None`` when it did not happen.

    Every failure — no attributor, no diff source, a raising fetch, a raising or
    declining model — is the same ``None``: the comment then renders from the
    per-configuration scores, which is exactly what it did before this stage
    existed. A degraded comment beats a blocked one."""
    if attributor is None:
        return None
    fetch = patch_for or (lambda _repo, _number: "")
    try:
        request = _attribution_request(plan, fetch)
    except Exception as exc:  # noqa: BLE001 — a diff fetch must not lose the comment
        _log.warning(
            "build_comments: %s — could not assemble the review request (%s); "
            "falling back to the per-configuration scores", plan.target, exc,
        )
        return None
    try:
        return attributor.attribute(request)
    except Exception as exc:  # noqa: BLE001 — an adapter that raises is a decline
        _log.warning(
            "build_comments: %s — the cross-configuration review failed (%s); "
            "falling back to the per-configuration scores", plan.target, exc,
        )
        return None


def _attribution_request(plan: CommentPlan, fetch: PatchFor) -> AttributionRequest:
    """The whole window, as the reviewing model is shown it."""
    return AttributionRequest(
        repo=plan.repo,
        number=plan.number,
        title=plan.subject.title,
        base_release=plan.base_release,
        onset_release=plan.onset_release,
        files=plan.subject.files,
        patch=fetch(plan.repo, plan.number),
        additions=plan.subject.additions,
        deletions=plan.subject.deletions,
        regressions=tuple(_fact(row) for row in plan.rows),
        outcomes=plan.outcomes,
        competitors=tuple(
            _competitor(other, fetch)
            # Cut the field to what the prompt can actually carry *before*
            # fetching anything: the prompt keeps the strongest
            # `MAX_COMPETITORS` in this same order, so a window with a hundred
            # candidates would otherwise spend a hundred GitHub round trips —
            # inside one shared timeout — to show thirty.
            for other in _sorted_others(plan)[:MAX_COMPETITORS]
        ),
        packages_by_platform=plan.packages_by_platform,
        unchanged_by_platform=dict(plan.unchanged),
    )


def _competitor(other: CandidatePR, fetch: PatchFor) -> CompetingPR:
    """One competing candidate as the review sees it — with a score only if the
    first pass gave it one."""
    return CompetingPR(
        repo=other.repo, number=other.number, url=other.url,
        title=other.title, files=other.files,
        additions=other.additions, deletions=other.deletions,
        scope_score=other.score if other.ranked else None,
        scope_reason=other.description,
        patch=fetch(other.repo, other.number),
    )


def _fact(row: RegressionRow) -> RegressionFact:
    v = row.verdict
    return RegressionFact(
        id=row.fact_id,
        detector=v.detector, platform=v.platform, sample=v.sample,
        label=v.label, metric=v.metric, metric_family=v.metric_family,
        sub_detector=v.sub_detector, direction=str(getattr(v.direction, "value", v.direction)),
        pct_change=v.pct_change, value=v.value,
        baseline_median=v.baseline_median, z_score=v.z_score,
        scope_score=row.scope_score, scope_reason=row.scope_reason,
        scope_state=row.scope_state,
    )


def _sorted_others(plan: CommentPlan) -> list[CandidatePR]:
    """The competing candidates, strongest first and the unjudged last — the
    order both the prompt and the rendered disclosure use, and the order the
    competitor cap cuts on (:func:`~k4bench.blame.attribute.competitor_order`,
    which this must agree with)."""
    return sorted(
        plan.others.values(),
        key=lambda c: competitor_order(_competitor(c, lambda _r, _n: "")),
    )


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render(
    plan: CommentPlan,
    attribution: Attribution | None,
    *,
    dashboard_url: str | None,
) -> PRComment:
    """One plan as a GitHub-flavoured Markdown comment.

    A single comment for the ``(pull request, window)``: the claim, the window,
    the model's reasoning, and one table of every regression the window carries,
    ordered by how likely this pull request is to be behind each."""
    marker = marker_for(plan.base_release, plan.onset_release)
    rows = sorted(
        plan.rows, key=lambda row: _row_sort_key(row, attribution)
    )
    digest = _facts_digest(plan)
    # Only the rows that survive the table's caps are linked: a definition no
    # row references is dead weight against the comment-size limit.
    links = _row_links(
        plan, rows[:_MAX_TABLE_ROWS + _MAX_FOLDED_ROWS], dashboard_url
    )

    body = "\n".join(
        part for part in (
            marker,
            f"{_FACTS_MARKER_PREFIX}{digest} -->",
            "### 📉 Possible performance regression traced to this pull request",
            "",
            _alert(plan),
            "",
            _window_line(plan),
            _assessment(plan, rows, attribution),
            _table(plan, rows, attribution, links=links),
            _others_section(plan),
            _where_to_look(plan, rows, dashboard_url=dashboard_url),
            _link_definitions(links),
            "",
            "---",
            "",
            # The reply invitation is this renderer's own, not part of the
            # shared disclosure: the e-group mail carries the same sentence to
            # readers with no thread to answer in. k4Bench's name carries the
            # page describing how this attribution is made rather than the
            # repository root: someone who doubts a machine-written accusation
            # wants the method, and a README makes them go looking for it.
            f"<sub>🤖 {RANKING_DISCLOSURE} Posted automatically by "
            f"[k4Bench]({_METHOD_URL}) — reply here if this attribution looks "
            "wrong.</sub>",
        ) if part is not None
    )
    return PRComment(
        repo=plan.repo,
        number=plan.number,
        marker=marker,
        body=body,
        score=plan.top_score,
        facts_digest=digest,
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


def _row_sort_key(row: RegressionRow, attribution: Attribution | None) -> tuple:
    """Most likely first, then the largest movement, then identity — so the
    table is stable across nights and a re-render triggers no edit. Rows nobody
    scored sort last: they are evidence about the window, not claims about this
    pull request, and they must not head a table that reads top-down."""
    likelihood = _likelihood(row, attribution)
    return (
        likelihood is None,
        -(likelihood if likelihood is not None else 0.0),
        -_movement(row),
        *_row_identity(row),
    )


def _row_identity(row: RegressionRow) -> tuple:
    v = row.verdict
    return (
        v.detector, v.platform, v.sample, v.label, v.metric, v.sub_detector or "",
    )


def _movement(row: RegressionRow) -> float:
    """A row's step size, with a non-finite change counting as no movement —
    matching what :func:`_change_cell` renders for it. A NaN in a sort key would
    compare false against everything and leave the order dependent on input
    order, which is the one thing these keys exist to rule out."""
    pct = row.verdict.pct_change
    return abs(pct) if pct is not None and math.isfinite(pct) else 0.0


def _alert(plan: CommentPlan) -> str:
    """The headline claim as a GitHub warning alert: one short sentence that
    reads on a single line, since everything below it — the window, the
    reasoning, what moved — is the specifics."""
    n_scopes = len(plan.scopes)
    what = (
        "a regression in"
        if n_scopes == 1
        else f"regressions in {_count(n_scopes, 'configuration')} of"
    )
    return (
        "> [!WARNING]\n"
        f"> k4Bench's nightly benchmarks confirmed {what} this PR's change "
        "window."
    )


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
    return f"📆 **Change window:** {window}"


def _assessment(
    plan: CommentPlan, rows: list[RegressionRow], attribution: Attribution | None
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

    Without a review, the comment falls back to
    the per-configuration ranker's one-liner for its strongest row, and then it
    claims "the most likely cause" only when this PR outranks every other
    candidate — a comment can fire on any score above ``min_score``, and a PR the
    ranker placed second must not be told it came first. Nothing is rendered when
    neither model explained itself: an unexplained score is not comment-worthy
    prose, and it already stands in the table."""
    if attribution is not None:
        text = _one_line(attribution.summary, _MAX_SUMMARY_CHARS)
        if text:
            return (
                f"\n> 🤖 **The AI reviewer's assessment:** {text}"
                + _coverage_note(rows, attribution)
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
        for other in plan.others.values()
    )
    claim = "the most likely" if outranks_all else "a likely"
    return (
        f"\n> 🤖 **The AI ranker judged this PR {claim} cause of the "
        f"regression:** {text}"
    )


def _coverage_note(
    rows: list[RegressionRow], attribution: Attribution
) -> str:
    """What the assessment above does *not* cover, when it covers less than all.

    Two ways a row goes unreviewed: the window carried more regressions than the
    prompt offers (only the largest movements are shown), or the reply simply
    skipped it. Either way the row keeps its per-configuration likelihood
    (:func:`_likelihood`) and the table shows it — so the summary has to say it
    was not part of what the reviewer weighed. Nothing is added when every row
    was answered, which is the ordinary night."""
    unreviewed = sum(1 for row in rows if row.fact_id not in attribution.likelihoods)
    if not unreviewed:
        return ""
    return (
        f"\n>\n> <sub>This assessment covers "
        f"{_count(len(rows) - unreviewed, 'regression')} of {len(rows)}; the "
        f"remaining {unreviewed} keep the per-configuration ranker's score and "
        "were not part of it.</sub>"
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


def _table(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    links: dict[str, str],
) -> str:
    """Every regression in the window, most likely first.

    One table rather than one section per configuration: which configurations
    moved — and, read against the review's summary, which did not — is the
    substance of the claim, and a reader weighing it needs to see the pattern at
    once. The first rows are visible, the next fold into a disclosure, and a
    night wider than that says how many more there are rather than pasting them:
    a detector-removal sweep can confirm three hundred near-identical rows, which
    no one reads and GitHub will not accept. Every row links to its own regression
    in the dashboard, which is where the complete set lives.

    That link hangs off the **metric** cell, because that is what it opens: the
    metric's own trend, its onset and the window's package diff. Metric and
    configuration keep their raw names — they are the identifiers the dashboard
    labels the series with, so a reader can find it.

    The **Platform** column is switched off (:data:`_SHOW_PLATFORM_COLUMN`)
    because the suite currently builds on exactly one platform, and a column
    repeating the same slug on every row is noise. That is a presentation policy
    and nothing more: platform stays part of every row's identity, of the
    grouping, of the links, of the digest and of both prompts. Flipping the
    switch is what a second platform needs — not a change to how rows are
    collected."""
    header = ["Metric", "Detector"]
    align = [":---", ":---"]
    if _SHOW_PLATFORM_COLUMN:
        header.append("Platform")
        align.append(":---")
    header += ["Sample", "Config", "Change", "Attribution"]
    align += [":---", ":---", "---:", "---:"]

    def _line(row: RegressionRow) -> str:
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
            _cell(pretty_sample(v.sample)),
            f"`{_cell(v.label)}`",
            _change_cell(v.pct_change),
            _likelihood_cell(row, attribution),
        ]
        return "| " + " | ".join(cells) + " |"

    shown = rows[:_MAX_TABLE_ROWS]
    folded = rows[_MAX_TABLE_ROWS:_MAX_TABLE_ROWS + _MAX_FOLDED_ROWS]
    omitted = len(rows) - len(shown) - len(folded)
    head = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
    ]
    lines = [
        "",
        # "reviewed against", not "attributed to": the table deliberately keeps
        # rows the review scored *down*, and a 20% row under a heading claiming
        # attribution reads as an accusation the numbers next to it deny.
        "##### 📊 Regressions reviewed against this pull request",
        "",
        *head,
        *(_line(row) for row in shown),
    ]
    if folded:
        summary = _count(len(folded) + omitted, "further regression")
        lines += [
            "",
            "<details>",
            f"<summary><b>{summary} in this window</b></summary>",
            "",
            *head,
            *(_line(row) for row in folded),
        ]
        if omitted:
            lines += [
                "",
                f"_…and {_count(omitted, 'more regression')}, in the dashboard._",
            ]
        lines += ["", "</details>"]
    return "\n".join(lines)


def _others_section(plan: CommentPlan) -> str:
    """The rest of the candidates scored across this window, with their
    likelihoods — the reader needs to see what else was in the frame to weigh
    the claim against this PR, including the case where nothing else was.
    Collapsed by default, but the summary line carries the strongest competing
    score without being opened: how far ahead this PR sits is the difference
    between a ranking that picked it and one that barely preferred it, and that
    belongs in front of a reader who expands nothing.

    The candidates are named, never linked — see :func:`_pr_ref`."""
    others = _sorted_others(plan)
    if not others:
        return "\n".join([
            "",
            "> [!NOTE]",
            "> This was the only pull request found across every tracked "
            "package that changed in this window.",
        ])

    shown = others[:_MAX_OTHER_CANDIDATES]
    # ``others`` is judged-first (:func:`_sorted_others`), so the strongest
    # *scored* candidate leads when there is one. A field nobody scored says
    # that instead of quoting a percentage no model produced.
    strongest = others[0] if others[0].ranked else None
    headline = (
        f"highest {_pct(strongest.score)}" if strongest is not None
        else "none of them scored by the ranker"
    )
    lines = [
        *(
            note for note in (
                _crowded_note(plan, strongest) if strongest is not None else None,
            ) if note
        ),
        "",
        "<details>",
        "<summary><b>Other pull requests in this window</b> — "
        f"{_count(len(others), 'candidate')}, {headline}"
        "</summary>",
        "",
        "| Pull request | Likelihood |",
        "|:---|---:|",
        *(
            f"| {_pr_ref(c)} — {_cell(_one_line(c.title, 80))} | "
            f"{_candidate_score_cell(c)} |"
            for c in shown
        ),
    ]
    if len(others) > len(shown):
        lines += ["", f"_…and {_count(len(others) - len(shown), 'more candidate')}._"]
    lines += ["", "</details>"]
    return "\n".join(lines)


def _pr_ref(candidate: CandidatePR) -> str:
    """A competing candidate named as ``owner/repo#123``, inert on purpose.

    These pull requests are *not* the ones being commented on — they are the
    field the ranking was made against — and GitHub turns any reference to them,
    a bare ``owner/repo#123`` or a link carrying their URL, into a cross-
    reference on their own timeline, notifying everyone subscribed there. A PR
    that was merely a candidate should not collect a notification every time
    another window implicates someone else, so the number is broken with a
    zero-width space: unchanged to a reader, unparsed by GitHub, and
    unclickable. Whoever wants the full field has the package-diff link in
    *Where to look*."""
    zwsp = "​"  # U+200B zero-width space
    return _cell(f"{candidate.repo}#{zwsp}{candidate.number}")


def _crowded_note(plan: CommentPlan, closest: CandidatePR) -> str | None:
    """Said out loud when the ranking does not clearly favour this PR — in
    words, rather than leaving the reader to subtract two numbers. *closest* is
    always a candidate the first pass actually scored; an unscored one is not a
    near miss, it is an unknown, and no gap can be computed from it.

    Which way the preference runs is the whole point, so the note is
    direction-aware. A PR the ranker placed *behind* another candidate is told so
    however wide the gap is: that is the single most important qualifier on a
    comment accusing it. A PR that is ahead hears about it only when the lead is
    thin (``_CROWDED_SPREAD``) — a caveat printed on every comfortable night is
    wallpaper, and the score and the summary line already say what a comfortable
    lead looks like."""
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


def _where_to_look(
    plan: CommentPlan, rows: list[RegressionRow], *, dashboard_url: str | None
) -> str | None:
    """The window-wide link that lets a reader check the claim rather than take
    it: every package that moved across these two releases.

    Per-regression dashboard views are already on each row's metric cell, so
    what is left here is the one thing that is not per-row: the unpinned diff,
    for a reader who wants the packages without a metric selected — including
    one whose rows all fell past the table's caps. The link names the *window*,
    which does not change from one night to the next: no ``report=`` night and
    no CI-run URL, both of which would vary nightly and edit a standing comment
    for no reason.

    A dashboard view is always one configuration, and the package diff behind it
    is read from *that platform's* provenance — so a window spanning platforms
    gets one link each, named after the platform it opens. One link labelled
    "every package that changed" that in fact shows one platform's packages
    would be a claim the view does not support."""
    if not rows:
        return None
    lead = rows[0].verdict
    by_platform = {
        row.verdict.platform: row.verdict for row in rows
    }
    # The leading row's scope names the link when there is only one platform —
    # the ordinary case, and the one that keeps the sentence simple.
    scopes = [lead] if len(by_platform) == 1 else [
        by_platform[platform] for platform in sorted(by_platform)
    ]
    links = []
    for scope in scopes:
        href = stack_changes_href(
            dashboard_url,
            detector=scope.detector, platform=scope.platform, sample=scope.sample,
            base_release=plan.base_release, onset_release=plan.onset_release,
        )
        if not href:
            continue
        where = (
            "" if len(scopes) == 1
            else f" on {pretty_platform(scope.platform)}"
        )
        links.append(
            f"- 📦 [Every package that changed across this window{where}]({href})"
        )
    if not links:
        return None
    return "\n".join(["", "##### 🔎 Where to look", "", *links])


def _facts_digest(plan: CommentPlan) -> str:
    """A fingerprint of the *benchmark facts* behind a comment.

    The publisher edits a standing comment only when this changes
    (:func:`k4bench.blame.publish._upsert`), so the rule is: everything
    deterministic that a reader would call a change goes in, and everything a
    model re-rolls each night stays out.

    **In**, because they are measurements and because they visibly change what
    the comment says: the window; every regression row's identity — platform
    included — and how far it moved; what the first pass knew about this pull
    request in each of those scopes; the clean and watch outcomes with their
    watched metrics and unjudged counts; the per-platform package diff and
    unchanged counts; and which pull requests were in the field, with whether
    each was judged at all.

    The outcomes especially. A comment posted while IDEA had no reliable result
    reads differently once IDEA delivers a clean measurement of the same window
    — that control weakens the attribution and the review is shown it — and a
    digest covering only the positive rows would leave the old reasoning
    standing on the pull request forever, because nothing it hashed had moved.

    **Out**: the narrative, and every model score — the review's likelihoods and
    the ranker's scoring alike. Those drift between nights without anything
    having happened, and a competitor sliding from 84.4 to 84.6 is not worth
    re-notifying everyone watching a pull request. (Whether a candidate was
    scored *at all* is a different thing, and is in: it changes the table cell
    and the prompt from a percentage to "not scored".)

    Serialized as canonical JSON rather than joined strings so a field's value
    can never migrate into its neighbour's — ``a|b`` and ``a`` + ``|b`` hash
    alike, and identities here are user-supplied names.
    """
    payload = {
        "window": [plan.base_release or "", plan.onset_release],
        "rows": [
            {
                "id": list(_row_identity(row)),
                "moved": _canonical_pct(row.verdict.pct_change),
                "state": row.scope_state,
            }
            for row in sorted(plan.rows, key=_row_identity)
        ],
        "outcomes": [
            {
                "scope": [o.detector, o.platform, o.sample, o.label],
                "status": o.status,
                "watched": list(o.watched),
                "unjudged": o.unjudged,
            }
            for o in plan.outcomes
        ],
        "packages": {
            platform: [[p.package, p.status] for p in packages]
            for platform, packages in plan.packages_by_platform.items()
        },
        "unchanged": dict(sorted(plan.unchanged.items())),
        "competitors": [
            {"pr": f"{other.repo}#{other.number}", "ranked": other.ranked}
            for other in _sorted_others(plan)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _canonical_pct(pct: float | None) -> str:
    """A step size as fixed text, so a float's repr can never move a digest."""
    return f"{pct:.4f}" if pct is not None and math.isfinite(pct) else "-"


def _pct(score: float) -> str:
    return f"{int(round(score))}%"


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
    report's own formatting. Both arrows are red on purpose: whichever way a
    confirmed regression moved, it moved the wrong way."""
    if pct_change is None or not math.isfinite(pct_change):
        return "—"
    arrow = "🔺" if pct_change >= 0 else "🔻"
    return f"{arrow} **{pct_change:+.1%}**"


def _count(n: int, noun: str) -> str:
    """``3 candidates`` / ``1 candidate`` — every noun this comment counts
    pluralises with a plain ``s``."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _one_line(text: str, limit: int) -> str:
    """Model- or GitHub-authored text flattened to one line, defanged, clipped.

    Newlines would break out of a table cell or a blockquote, so they are
    collapsed rather than escaped; :func:`_defang` then pulls the teeth from any
    Markdown/HTML the prose carries before it lands in an outward-facing
    comment."""
    flat = _defang(" ".join((text or "").split()))
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _defang(text: str) -> str:
    """Neutralise the active Markdown/HTML in quoted, externally-authored prose.

    A PR title or a model's summary is untrusted text pasted into a comment the
    bot posts in someone else's repository. Left as-is it could:

    * ``@login`` — ping a person on every nightly edit (the same ban the whole
      bot honours by never rendering an author with an ``@``);
    * ``#123`` — cross-reference an unrelated issue, notifying its subscribers;
      a title like "Revert #45" carries one for free (:func:`_pr_ref` applies the
      same rule to the references this module writes itself, and the review is
      asked to name alternatives in exactly that form);
    * ``<!-- … -->`` / ``<tag>`` — hide following content, or inject markup;
    * ``[text](url)`` / ``![alt](url)`` — put an arbitrary clickable destination,
      or a remote image, into a comment the bot signs its own name to;
    * ``https://...`` / ``www....`` — GitHub autolinks a bare URL, so the prose
      needs no Markdown at all to become a link. A pull-request URL autolinked
      this way also cross-references that PR's timeline, which is the very
      notification :func:`_pr_ref` goes out of its way not to send.

    A zero-width space at each sequence's join breaks what GitHub would act on
    while leaving the text visually unchanged: after the trigger character for
    the prefix forms, and between ``]`` and ``(`` for a link, whose two halves
    are what make it one. Emphasis and backticks are deliberately left alone —
    they restyle the quoted text but cannot carry a reader anywhere. Table pipes
    are left to :func:`_cell`, which the cell paths still apply on top of this."""
    zwsp = "​"  # U+200B zero-width space
    return (
        text.replace("@", "@" + zwsp)
        .replace("#", "#" + zwsp)
        .replace("<", "<" + zwsp)
        .replace("](", "]" + zwsp + "(")
        .replace("![", "!" + zwsp + "[")
        .replace("://", ":" + zwsp + "//")
        .replace("www.", "www" + zwsp + ".")
        .replace("WWW.", "WWW" + zwsp + ".")
    )


def _cell(text: str | None) -> str:
    """Text safe inside a Markdown table cell: a pipe would end the column."""
    return (text or "").replace("|", "\\|")
