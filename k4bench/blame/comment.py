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
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from k4bench.blame.attribute import (
    Attribution,
    Attributor,
    AttributionRequest,
    CompetingPR,
    PackageChangeFact,
    RegressionFact,
    ScopeOutcome,
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
        however it scores. That is what makes this the right gate for deciding
        which pull requests get a plan built at all: everything a plan collects
        beyond that point is evidence, and evidence is not filtered by score
        (see :func:`select`)."""
        return candidate.repo.lower() in self.repos and bool(candidate.merged_at)

    def allows(self, candidate: CandidatePR) -> bool:
        """True when *candidate* clears the repo, score and merged gates —
        i.e. when this scoring is strong enough to *cause* a comment."""
        return (
            self.targets(candidate)
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
    represents a real change. A comment posted before digests existed returns
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

    ``scope_score``/``scope_reason`` are the *per-configuration* ranker's
    judgement of this pull request in this row's run group: the prior the review
    starts from, and the likelihood shown when no review runs. ``fact_id`` is the
    opaque handle the review echoes back (see
    :class:`~k4bench.blame.attribute.RegressionFact`); it is assigned in identity
    order, never score order, so the same night's rows always carry the same ids.
    ``stack`` is the group's Key4hep release *directory* — the dashboard's
    ``?stack=`` vocabulary, kept per row so each link names the stack that row
    actually ran against.
    """

    verdict: MetricVerdict
    stack: str
    fact_id: str = ""
    scope_score: float = 0.0
    scope_reason: str = ""

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

    ``rows`` holds **every** regression of the window this pull request was a
    candidate for, including the ones it scored badly on; ``selected`` records
    whether any of them scored high enough to warrant the comment. The two are
    deliberately separate — see :func:`select`.
    """

    repo: str
    number: int
    subject: CandidatePR
    base_release: str | None
    onset_release: str
    rows: list[RegressionRow] = field(default_factory=list)
    others: dict[tuple[str, int], CandidatePR] = field(default_factory=dict)
    outcomes: tuple[ScopeOutcome, ...] = ()
    packages: tuple[PackageChangeFact, ...] = ()
    n_unchanged: int = 0
    selected: bool = False

    @property
    def target(self) -> str:
        return f"{self.repo}#{self.number}"

    @property
    def top_score(self) -> float:
        """The strongest per-configuration likelihood across this window — what
        selection was made on, and how comments are ordered."""
        return max((row.scope_score for row in self.rows), default=0.0)

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

    Collection and selection are separate passes over different gates, and the
    order matters. A plan collects every regression this pull request was a
    candidate for, at *whatever* the ranker scored it there; only then is the
    plan kept, and only if at least one of those scorings clears ``min_score``.
    Filtering rows by score while collecting — the obvious shape — would hide
    precisely the evidence the cross-configuration review exists to weigh: a PR
    that scores 92 on ALLEGRO and 30 on IDEA in the same window is a PR whose
    reach the IDEA row bounds, and dropping that row leaves the review looking
    at the accusation with the exculpatory half removed. (It would not resurface
    as negative evidence either: :func:`_outcomes_for` correctly refuses to call
    a configuration clean when it confirmed a step in this window.)

    Overshooting ``max_comments`` raises :class:`CommentStormError` rather than
    returning a truncated list — a night that loud is a bug, not a night, and
    blind-posting ten accusations into repositories k4Bench does not own is the
    exact harm the gates exist to prevent. It is raised, not returned empty, so
    the caller can tell it apart from an ordinary night that simply implicated
    no one.
    """
    if not policy.enabled:
        return []

    plans: dict[tuple[str, int, str | None, str], CommentPlan] = {}
    # Walked group by group rather than through ``report.regressions`` so each
    # verdict keeps its group's release directory — the dashboard links need it.
    for group in report.groups:
        for verdict in group.regressions:
            entry = blame.entry_for(verdict)
            if entry is None or entry.discovery_incomplete:
                continue
            candidates = entry.candidates
            for candidate in candidates:
                if not policy.targets(candidate):
                    continue  # unreachable repo or unmerged: no comment, ever
                ident = (candidate.repo.lower(), candidate.number)
                key = (*ident, entry.base_release, entry.onset_release)
                plan = plans.get(key)
                if plan is None:
                    plan = plans[key] = CommentPlan(
                        repo=candidate.repo, number=candidate.number,
                        subject=candidate,
                        base_release=entry.base_release,
                        onset_release=entry.onset_release,
                        packages=_packages_of(entry),
                        n_unchanged=entry.n_unchanged,
                    )
                elif candidate.score > plan.subject.score:
                    # Every metric of a run group shares one ranking, so these
                    # are equal in valid builder output; keep the strongest
                    # defensively so the identity rendered never depends on
                    # which metric was walked first.
                    plan.subject = candidate
                # One scoring above the threshold is what causes the comment;
                # every scoring, high or low, is what the comment is judged on.
                plan.selected = plan.selected or policy.allows(candidate)
                plan.rows.append(RegressionRow(
                    verdict=verdict, stack=group.k4h_release,
                    scope_score=candidate.score, scope_reason=candidate.description,
                ))
                for other in candidates:
                    other_ident = (other.repo.lower(), other.number)
                    if other_ident == ident:
                        continue
                    previous = plan.others.get(other_ident)
                    if previous is None or other.score > previous.score:
                        plan.others[other_ident] = other

    ordered = sorted(
        (plan for plan in plans.values() if plan.selected),
        key=lambda p: (-p.top_score, p.repo, p.number),
    )
    for plan in ordered:
        # Ids ride on identity order so they are reproducible from the plan
        # alone: a night re-run must ask the model about "r3" and mean the same
        # regression it meant last time.
        plan.rows.sort(key=_row_identity)
        plan.rows = [
            RegressionRow(
                verdict=row.verdict, stack=row.stack, fact_id=f"r{index}",
                scope_score=row.scope_score, scope_reason=row.scope_reason,
            )
            for index, row in enumerate(plan.rows, start=1)
        ]
        plan.outcomes = _outcomes_for(report, plan)

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


def _packages_of(entry: BlameEntry) -> tuple[PackageChangeFact, ...]:
    """The window's package diff, as the review is shown it."""
    return tuple(
        PackageChangeFact(
            package=repo.package, status=repo.status, compare_url=repo.compare_url
        )
        for repo in entry.repos
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
    recorded no metric failure of its own, and it holds no confirmed step in
    this window. Everything else is silence from a run that did not happen or
    cannot be read, and silence must never be rendered as evidence of absence:
    ``reliable is None`` means *no evidence either way*, so it is treated like
    an unreliable run rather than like a clean one."""
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
            if any(v.severity is Severity.CONFIRMED and _window_of(v) == window
                   for v in verdicts):
                continue  # stepped in this very window — it is not a control
            if any(v.severity is Severity.FAILURE for v in verdicts):
                continue  # a configuration that partly failed did not run clean
            watched = tuple(sorted(
                {v.metric for v in verdicts if v.severity is Severity.WATCH}
            ))[:_MAX_WATCHED_METRICS]
            outcomes.append(ScopeOutcome(
                detector=group.detector, platform=group.platform,
                sample=group.sample, label=label,
                status="watch" if watched else "clean", watched=watched,
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


def _window_of(verdict: MetricVerdict) -> tuple[str | None, str | None]:
    return (verdict.last_accepted_run_date, verdict.onset_run_date)


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
    table. The review may only ever **narrow**: a plan is withdrawn when no row
    is left standing at or above *min_score*, while a plan it declines to review
    — no model, a failed call, an unusable reply — still renders from the
    per-configuration scores it already had. Selection was made on those scores,
    so a second opinion is allowed to acquit, never to accuse.

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
            (_likelihood(row, attribution) for row in plan.rows), default=0.0
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
            CompetingPR(
                repo=other.repo, number=other.number, url=other.url,
                title=other.title, files=other.files,
                additions=other.additions, deletions=other.deletions,
                scope_score=other.score, scope_reason=other.description,
                patch=fetch(other.repo, other.number),
            )
            for other in _sorted_others(plan)
        ),
        packages=plan.packages,
        n_unchanged=plan.n_unchanged,
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
    )


def _sorted_others(plan: CommentPlan) -> list[CandidatePR]:
    """The competing candidates, strongest first — a stable order for both the
    prompt and the rendered disclosure."""
    return sorted(plan.others.values(), key=lambda c: (-c.score, c.repo, c.number))


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


def _likelihood(row: RegressionRow, attribution: Attribution | None) -> float:
    """What this row is shown as, and ordered by.

    The review's score when it gave one; otherwise the per-configuration
    ranker's. A row the review omitted is not a zero — an unanswered row keeps
    the judgement that was already made about it."""
    if attribution is None:
        return row.scope_score
    return attribution.likelihoods.get(row.fact_id, row.scope_score)


def _row_sort_key(row: RegressionRow, attribution: Attribution | None) -> tuple:
    """Most likely first, then the largest movement, then identity — so the
    table is stable across nights and a re-render triggers no edit."""
    return (-_likelihood(row, attribution), -_movement(row), *_row_identity(row))


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
    that can actually explain the pattern. Without one, the comment falls back to
    the per-configuration ranker's one-liner for its strongest row, and then it
    claims "the most likely cause" only when this PR outranks every other
    candidate — a comment can fire on any score above ``min_score``, and a PR the
    ranker placed second must not be told it came first. Nothing is rendered when
    neither model explained itself: an unexplained score is not comment-worthy
    prose, and it already stands in the table."""
    if attribution is not None:
        text = _one_line(attribution.summary, _MAX_SUMMARY_CHARS)
        if text:
            return f"\n> 🤖 **The AI reviewer's assessment:** {text}"
        return None
    lead = rows[0] if rows else None
    text = _one_line(lead.scope_reason, _MAX_DESCRIPTION_CHARS) if lead else ""
    if not text:
        return None
    outranks_all = all(
        other.score < lead.scope_score for other in plan.others.values()
    )
    claim = "the most likely" if outranks_all else "a likely"
    return (
        f"\n> 🤖 **The AI ranker judged this PR {claim} cause of the "
        f"regression:** {text}"
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
    labels the series with, so a reader can find it. The platform earns a column
    only when the window spans more than one — a column repeating the same slug
    on every row is noise, but a row that quietly ran somewhere else must say
    so."""
    multi_platform = len({row.verdict.platform for row in rows}) > 1
    header = ["Metric", "Detector"]
    align = [":---", ":---"]
    if multi_platform:
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
        if multi_platform:
            cells.append(_cell(pretty_platform(v.platform)))
        cells += [
            _cell(pretty_sample(v.sample)),
            f"`{_cell(v.label)}`",
            _change_cell(v.pct_change),
            _pct(_likelihood(row, attribution)),
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
        "##### 📊 Regressions attributed to this pull request",
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
    lines = [
        *(note for note in (_crowded_note(plan, others[0]),) if note),
        "",
        "<details>",
        "<summary><b>Other pull requests in this window</b> — "
        f"{_count(len(others), 'candidate')}, highest {_pct(others[0].score)}"
        "</summary>",
        "",
        "| Pull request | Likelihood |",
        "|:---|---:|",
        *(
            f"| {_pr_ref(c)} — {_cell(_one_line(c.title, 80))} | {_pct(c.score)} |"
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
    words, rather than leaving the reader to subtract two numbers.

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
    one whose rows all fell past the table's caps. A dashboard view is
    always one configuration, so the package diff is named from the leading row's
    scope. The link names the *window*, which does not change from one night to
    the next: no ``report=`` night and no CI-run URL, both of which would vary
    nightly and edit a standing comment for no reason."""
    if not rows:
        return None
    lead = rows[0].verdict
    packages = stack_changes_href(
        dashboard_url,
        detector=lead.detector, platform=lead.platform, sample=lead.sample,
        base_release=plan.base_release, onset_release=plan.onset_release,
    )
    if not packages:
        return None
    return "\n".join([
        "",
        "##### 🔎 Where to look",
        "",
        f"- 📦 [Every package that changed across this window]({packages})",
    ])


def _facts_digest(plan: CommentPlan) -> str:
    """A fingerprint of the *benchmark facts* behind a comment.

    The model's prose is regenerated every night and will not repeat itself
    word for word, so comparing whole bodies would edit — and re-notify —
    a standing regression nightly, which is exactly what
    :mod:`k4bench.blame.publish` refuses to do. This digest covers what a reader
    would call a change: the window, the regressions and how far they moved, and
    *which* pull requests were in the field. Deliberately not the narrative, and
    no score of any kind — neither the review's likelihoods nor the ranker's
    scoring of the competing candidates. All of it is model output that drifts
    between nights without anything having happened, and a competitor sliding
    from 84.4 to 84.6 is not a fact a reader would want re-notifying about."""
    parts = [plan.base_release or "", plan.onset_release]
    for row in sorted(plan.rows, key=_row_identity):
        pct = row.verdict.pct_change
        moved = f"{pct:.4f}" if pct is not None and math.isfinite(pct) else "-"
        parts.append("|".join((*_row_identity(row), moved)))
    for other in _sorted_others(plan):
        parts.append(f"{other.repo}#{other.number}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _pct(score: float) -> str:
    return f"{int(round(score))}%"


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
