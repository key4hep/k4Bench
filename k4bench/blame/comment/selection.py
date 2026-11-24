"""Which pull requests hear about tonight's regressions — decided, not rendered.

:func:`select` is pure: no network, no token, no clock. The whole "who gets told
what" decision is therefore unit-testable, and the CLI can print exactly what
would be posted without touching GitHub. See :mod:`k4bench.blame.comment` for the
gates every comment has to pass.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from k4bench.blame.attribute import (
    CompetingPR,
    ScopeCandidateState,
    competitor_order,
)
from k4bench.blame.comment.plan import (
    CommentPlan,
    RegressionRow,
    ScopeStanding,
    _row_identity,
)
from k4bench.blame.comment.state import (
    window_contains,
)
from k4bench.blame.evidence import (
    ScopeOutcome,
    outcomes_for_window,
    steps_in_window,
)
from k4bench.blame.geometry import benchmarked_geometry
from k4bench.blame.models import (
    BlameEntry,
    BlameReport,
    CandidatePR,
    rank_group_key,
)
from k4bench.blame.sweep import window_sweeps
from k4bench.layout import SUITES
from k4bench.regression.models import MetricVerdict, NightlyReport

_log = logging.getLogger(__name__)
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
    """Who may be commented on for one suite's regressions, and how
    confidently.

    ``repos`` holds lowercase ``owner/repo`` slugs; GitHub slugs are
    case-insensitive, so matching is done on the lowered form while the
    candidate's own spelling is what gets displayed. An empty ``repos`` disables
    the bot for the suite.
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
    def from_config(cls, data: dict[str, Any] | None, suite: str) -> CommentPolicy:
        """Build *suite*'s policy from the parsed ``.github/blame-comments.yml``.

        ``repos`` lists each suite's repositories under its name; a suite it
        does not name has none. The score threshold and the comment cap are
        every suite's. Unknown keys, unknown suites, wrong types and
        out-of-range values raise :class:`CommentConfigError` — whichever
        suite's policy is built, since a typo in another suite's list is as
        wrong as one in this — see the class docstring for why this one config
        is strict where the report schemas are forgiving.
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

        if suite not in SUITES:
            raise CommentConfigError(f"unknown suite {suite!r}")
        by_suite = data.get("repos", {})
        if by_suite is None:  # `repos:` with no value is no allowlist at all
            by_suite = {}
        if not isinstance(by_suite, dict):
            # A bare list names no suite, and a scalar is not an allowlist.
            raise CommentConfigError("repos must map each suite to a list of owner/repo slugs")
        unknown = set(by_suite) - set(SUITES)
        if unknown:
            raise CommentConfigError(f"unknown suite(s) in repos: {', '.join(sorted(map(str, unknown)))}")
        allowlists = {name: _allowlist(raw) for name, raw in by_suite.items()}
        return cls(
            min_score=min_score, repos=allowlists.get(suite, frozenset()),
            max_comments=max_comments,
        )


def _allowlist(raw_repos: object) -> frozenset[str]:
    """One suite's ``repos`` list as lowercase ``owner/repo`` slugs."""
    if raw_repos is None:  # `sim:` with no value is an empty allowlist
        raw_repos = []
    if not isinstance(raw_repos, list):
        # ``sim: false`` / ``sim: k4geo`` — a scalar is not an allowlist.
        raise CommentConfigError("each suite's repos must be a list of owner/repo slugs")
    repos = set()
    for slug in raw_repos:
        # Validate the *stripped* slug: ``"owner/ "`` must not slip through
        # the slash check and then be stored as the truncated ``"owner/"``.
        cleaned = slug.strip() if isinstance(slug, str) else slug
        if not isinstance(cleaned, str) or cleaned.count("/") != 1 \
                or cleaned.startswith("/") or cleaned.endswith("/"):
            raise CommentConfigError(f"not an owner/repo slug: {slug!r}")
        repos.add(cleaned.lower())
    return frozenset(repos)


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
    geometry = benchmarked_geometry(report)
    for plan in plans:
        plan.report_night = report.report_night
        plan.suite = report.suite
        _collect_window(confirmed, plan)
        plan.outcomes = _outcomes_for(report, plan)
        plan.geometry_paths = _geometry_paths_for(report, plan)
        plan.sweeps = window_sweeps(
            report,
            base_release=plan.base_release,
            onset_release=plan.onset_release,
            stacks={row.stack for row in plan.rows},
        )
        plan.geometry = dict(geometry)

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


def _geometry_paths_for(report: NightlyReport, plan: CommentPlan) -> tuple[str, ...]:
    """The compact files the run groups behind *plan*'s rows load, read off
    the report the way the builder reads them for the first pass. Groups that
    recorded none contribute nothing."""
    scopes = plan.scopes
    return tuple(sorted({
        g.geometry_path for g in report.groups
        if (g.detector, g.platform, g.sample) in scopes and g.geometry_path
    }))


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


def _collapse_nested_windows(plans: list[CommentPlan]) -> list[CommentPlan]:
    """Collapse one pull request's nested windows into one target.

    Equal-onset windows describe one step with per-series lower bounds, so the
    tightest base survives. When onsets differ, a strictly containing plan
    survives because its evidence includes the narrower plan plus any newer
    steps. Non-containing windows remain separate findings.
    """
    best: dict[tuple[str, int, str], CommentPlan] = {}
    for plan in plans:
        key = (plan.repo.lower(), plan.number, plan.onset_release)
        kept = best.get(key)
        if kept is None:
            best[key] = plan
        elif window_contains(kept.window, plan.window):
            if kept.subject.score > plan.subject.score:
                plan.subject = kept.subject
            best[key] = plan
        elif plan.subject.score > kept.subject.score:
            kept.subject = plan.subject

    by_onset = list(best.values())
    survivors = [
        plan
        for plan in by_onset
        if not any(
            other.repo.lower() == plan.repo.lower()
            and other.number == plan.number
            and other.onset_release != plan.onset_release
            and window_contains(other.window, plan.window)
            for other in by_onset
        )
    ]
    for survivor in survivors:
        subjects = (
            plan.subject
            for plan in by_onset
            if plan.repo.lower() == survivor.repo.lower()
            and plan.number == survivor.number
            and window_contains(survivor.window, plan.window)
        )
        survivor.subject = max(subjects, key=lambda subject: subject.score)
    return survivors


def _targets(
    confirmed: list[_Confirmed], policy: CommentPolicy
) -> list[CommentPlan]:
    """Phase one: the ``(pull request, window)`` pairs a comment may be made
    about, and nothing about what those comments will say.

    A target needs one *complete* first-pass judgement clearing every gate —
    allowlisted repo, merged, ranked, at or above ``min_score``, from an entry
    whose candidate discovery was complete, and whose step the ranker did not
    read as noise. Evidence is gathered afterwards (:func:`_collect_window`), so
    no row can widen or narrow the field here.

    Windows nested inside one another are one target, not several; independent
    non-containing windows remain separate — see :func:`_collapse_nested_windows`."""
    plans: dict[tuple[str, int, str | None, str], CommentPlan] = {}
    for _verdict, _stack, entry in confirmed:
        if entry is None or entry.discovery_incomplete:
            continue
        if entry.base_release == entry.onset_release:
            # A same-release window, which this layer cannot name. Every
            # identity here is the release pair alone: :func:`marker_for` keys
            # a comment on ``window_token(base, onset)`` without the run ids
            # that tell two windows inside one release apart, and
            # :func:`~k4bench.blame.evidence.steps_in_window` reads the pair as
            # half-open, so ``(D, D]`` cannot even hold the verdict that formed
            # it. Selecting one would post a comment with no rows under a key
            # a second window of that release would collide with.
            #
            # Nothing is lost today: a same-release window's upstream diff is
            # empty by construction, so its only candidates are k4Bench's own
            # commits, and this repository is not one a comment may be posted
            # to. Supporting these means run-qualifying the marker, the
            # containment algebra and the row predicate together — a change to
            # the comment identity, not a filter — so the gate is explicit
            # rather than left to that emptiness.
            _log.info(
                "select: %s/%s %s — same-release window %s; the comment "
                "identity cannot name one window inside a release",
                entry.detector, entry.sample, entry.metric, entry.onset_release,
            )
            continue
        if entry.assessment is not None and entry.assessment.likely_noise:
            # The ranker scored these candidates *and* concluded the movement
            # itself is most likely this series' own noise. Those two statements
            # together say "there is probably nothing here", and a comment in
            # someone else's repository is exactly the wrong thing to do with
            # them — a high score under a noise verdict is the overconfident
            # attribution this assessment exists to catch. The entry is still
            # written, ranked and rendered on the dashboard and in the email,
            # where a human reads it with the verdict beside it; only the
            # outward-facing accusation is withheld.
            _log.info(
                "select: %s/%s %s — ranker reads the step as likely noise; "
                "no pull-request comment for this window",
                entry.detector, entry.sample, entry.metric,
            )
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
    return _collapse_nested_windows(list(plans.values()))


def _collect_window(confirmed: list[_Confirmed], plan: CommentPlan) -> None:
    """Phase two: fill *plan* with the whole window's evidence.

    Every confirmed regression whose onset falls inside the window is a row,
    whatever the subject's standing in it — that is what makes this a review of
    the window rather than of the accusation. The same predicate
    (:func:`~k4bench.blame.evidence.steps_in_window`) decides here and in the
    control set, so the two partition the night exactly: a configuration that stepped in this window
    is a row, and one that did not is a candidate control. Nothing falls between
    them."""
    window = (plan.base_release, plan.onset_release)
    ident = (plan.repo.lower(), plan.number)
    for verdict, stack, entry in confirmed:
        if not steps_in_window(verdict, window):
            continue
        state, candidate = _scope_state(entry, ident)
        # Recorded whether or not the sidecar has an entry: a platform whose
        # regression has *no* entry at all (missing provenance, an
        # unattributable window) had no release diff read for it either, and is
        # exactly the kind of gap :attr:`CommentPlan.packages_unavailable_on`
        # exists to name.
        platform = _window_platform(verdict)
        plan.platforms_seen.add(platform)
        if verdict.base_platform != verdict.onset_run_platform:
            plan.platform_switches.add((verdict.base_platform, verdict.onset_run_platform))
        if verdict.spans_detector_switch:
            plan.detector_switches.add((verdict.base_detector, verdict.onset_run_detector))
        if entry is not None:
            # Only an entry measuring *this comment's* window describes this
            # comment's release diff. A row can enter the window on a narrower
            # range of its own (a metric settled later carries a later base),
            # and folding that range's packages in would build a changed-package
            # set — and a "N of M tracked" denominator — that no provenance
            # read ever produced.
            if (entry.base_release, entry.onset_release) == window:
                _record_packages(plan, entry, platform)
                plan.external_changes.update(entry.external_changes)
                # Same window rule as the packages, and for the same reason: only
                # an entry that examined *this* window read the historical
                # evidence this comment's first-pass score rests on. An entry
                # covering a narrower range asked its own question and may have
                # asked for different analogues.
                _record_historical(plan, entry)
            _record_others(plan, entry, ident)
            if state == "ranked":
                _record_standing(plan, verdict, entry, candidate, ident)
        plan.rows.append(RegressionRow(
            verdict=verdict, stack=stack, scope_state=state,
            scope_score=candidate.score if candidate is not None else None,
            scope_reason=candidate.description if candidate is not None else "",
            # Taken from *this row's own* entry, which is joined on the row's
            # identity and window: the boundary counts describe that metric's
            # history tail, so they are correct even for a row that entered on a
            # narrower range than this comment's window.
            boundary_changes=dict(entry.boundary_changes) if entry else {},
        ))
    # Ids ride on identity order so they are reproducible from the plan alone: a
    # night re-run must ask the model about "r3" and mean the same regression it
    # meant last time.
    plan.rows.sort(key=_row_identity)
    plan.rows = [
        RegressionRow(
            verdict=row.verdict, stack=row.stack, fact_id=f"r{index}",
            scope_score=row.scope_score, scope_reason=row.scope_reason,
            scope_state=row.scope_state, boundary_changes=row.boundary_changes,
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

    A competing pull request is judged once per scope, and those judgements can
    disagree sharply — 95 on one detector, 10 on another. Only the strongest is
    carried (the prompt cannot hold every scope's reading of thirty
    competitors), so the scope it came from is carried with it: "95/100 on
    IDEA · debug" is a usable alternative, while a bare "95/100" invites the
    reviewer to read a one-scope judgement as a window-wide one — the very
    flattening this second pass exists to undo.

    A ranked sighting beats an unranked one whatever the scores say, since an
    unranked one carries no score at all."""
    for other in entry.candidates:
        other_ident = (other.repo.lower(), other.number)
        if other_ident == ident:
            continue
        sighting = (other, _scope_label(entry))
        previous = plan.others.get(other_ident)
        if previous is None or _candidate_rank(other) > _candidate_rank(previous[0]):
            plan.others[other_ident] = sighting
            plan.others_scope[other_ident] = (
                entry.detector, entry.platform, entry.sample,
            )


def _record_standing(
    plan: CommentPlan,
    verdict: MetricVerdict,
    entry: BlameEntry,
    candidate: CandidatePR,
    ident: tuple[str, int],
) -> None:
    """Record the subject's first-pass score in *verdict*'s rank group and the
    strongest other candidate that same ranking call scored. Every entry of one
    group carries the same ranking, so a repeat changes nothing."""
    key = rank_group_key(verdict)
    if key in plan.standings:
        return
    others = [
        other for other in entry.candidates
        if (other.repo.lower(), other.number) != ident
    ]
    ranked = [other for other in others if other.ranked]
    plan.standings[key] = ScopeStanding(
        score=candidate.score,
        rival=max(ranked, key=lambda other: other.score, default=None),
        unranked_rivals=len(ranked) < len(others),
    )


def _record_historical(plan: CommentPlan, entry: BlameEntry) -> None:
    """Fold one entry's historical references into *plan*, de-duplicated.

    Every entry of a rank group carries the same selection (one call produced
    it), and a comment window can span several rank groups on several platforms,
    so the same analogue arrives repeatedly and two genuinely different ones can
    arrive together. Keyed on ``(repo, number)``: the same pull request retrieved
    for two boundaries is one piece of evidence to re-fetch and to render, and
    the first sighting's boundary is as good as the second's for naming it."""
    for ref in entry.historical_evidence:
        plan.historical.setdefault(ref.key, ref)


def _scope_label(entry: BlameEntry) -> str:
    """How a scope is named to the reviewing model — the same order the outcome
    lines use, so one vocabulary describes the whole prompt."""
    return f"{entry.detector} · {entry.sample} · {entry.platform}"


def _candidate_rank(candidate: CandidatePR) -> tuple[bool, float]:
    return (candidate.ranked, candidate.score)


def _window_platform(verdict: MetricVerdict) -> str:
    """The provenance a verdict's window diff was read from: its own platform,
    or ``"<base> → <onset>"`` for a window measured across a platform migration,
    whose diff belongs to neither platform alone."""
    if verdict.base_platform != verdict.onset_run_platform:
        return f"{verdict.base_platform} → {verdict.onset_run_platform}"
    return verdict.onset_run_platform


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
    """The configurations that measured this plan's window and did not confirm.

    A thin adapter over :func:`k4bench.blame.evidence.outcomes_for_window`,
    which both passes share: the controls are the same evidence whether the
    question is "which pull request caused this configuration's regressions" or
    "which regressions did this pull request cause", and two implementations of
    "did this configuration stay flat" would be two answers.

    The stacks come from the rows themselves — the releases the regressed rows
    were actually measured on, which is generally well past the window's onset —
    and the regressed scopes only order the result, keeping the like-for-like
    controls at the front of whatever the prompt's cap keeps.
    """
    return outcomes_for_window(
        report,
        base_release=plan.base_release,
        onset_release=plan.onset_release,
        stacks={row.stack for row in plan.rows},
        regressed_scopes=plan.scopes,
    )


def _competitor(
    other: CandidatePR,
    scope: str,
    fetch: Callable[[str, int], str],
    body_fetch: Callable[[str, int], str] = lambda _r, _n: "",
) -> CompetingPR:
    """One competing candidate as the review sees it — with a score only if the
    first pass gave it one, and the scope that gave it. *fetch* and *body_fetch*
    are the review's diff and description sources
    (:data:`~k4bench.blame.comment.build.PatchFor`, :data:`~k4bench.blame.comment.build.BodyFor`)."""
    return CompetingPR(
        repo=other.repo, number=other.number, url=other.url,
        title=other.title, files=other.files,
        additions=other.additions, deletions=other.deletions,
        scope_score=other.score if other.ranked else None,
        scope_reason=other.description,
        scope=scope if other.ranked else "",
        patch=fetch(other.repo, other.number),
        body=body_fetch(other.repo, other.number),
    )


def _sorted_others(plan: CommentPlan) -> list[tuple[CandidatePR, str]]:
    """The competing candidates, strongest first and the unjudged last — the
    order both the prompt and the rendered disclosure use, and the order the
    competitor cap cuts on (:func:`~k4bench.blame.attribute.competitor_order`,
    which this must agree with)."""
    return sorted(
        plan.others.values(),
        key=lambda pair: competitor_order(
            _competitor(pair[0], pair[1], lambda _r, _n: "")
        ),
    )
