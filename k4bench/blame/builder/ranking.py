"""The ranker's input for one rank group, and its answer folded back.

Assembles a :class:`~k4bench.blame.rank.RankRequest` from the group's verdicts,
its candidates and the context read off the report, asks the ranker once per
group with every failure degraded to an unranked window, and applies the scores
to the candidates.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

from k4bench.blame.evidence import ExternalChange, history_from_verdict, outcomes_for_window
from k4bench.blame.geometry import detector_touches
from k4bench.blame.github import (
    PRText,
    diff_sample,
    low_signal_path,
)
from k4bench.blame.history import HistoricalIndex
from k4bench.blame.models import (
    RankGroupKey,
    RepoBlame,
    StepAssessment,
)
from k4bench.blame.prompt import HARNESS_PACKAGE, compact_dir, geometry_tree
from k4bench.blame.rank import (
    MetricStep,
    RankCandidate,
    Ranker,
    Ranking,
    RankRequest,
    RankResult,
)
from k4bench.blame.sweep import ScopeSweep, window_sweeps
from k4bench.layout import SIM
from k4bench.regression.models import MetricVerdict, NightlyReport

_log = logging.getLogger(__name__)

def _outcomes(
    report: NightlyReport, verdict: MetricVerdict, cache: dict[tuple, tuple]
) -> tuple:
    """The configurations that measured *verdict*'s window and did not confirm.

    The cross-configuration evidence the first pass previously had none of: it
    ranks one run group at a time, so without this it cannot tell a step that hit
    every detector from one that hit only this one — and those two windows point
    at opposite kinds of cause. Cached per ``(scope, window)``, since every metric
    of a run group shares both and the walk covers the whole report.

    The controls are drawn from the *whole* report, not just this scope: a
    detector that stayed flat is only informative because it is a different
    detector. Ordering puts this scope's own configurations first — the
    ``baseline`` vs. ``no_<detector>`` comparison is the sharpest control the
    suite produces, and it must survive the prompt's cap.

    A same-release window is also read and cached by its two runs, as
    :func:`_sweeps` reads it: its releases alone are the empty interval
    ``(D, D]``, which would offer the step that formed it as a clean control.
    """
    scope = (verdict.detector, verdict.platform, verdict.sample)
    base_run = onset_run = None
    if verdict.last_accepted_run_date == verdict.onset_run_date:
        base_run, onset_run = verdict.last_accepted_run_id, verdict.onset_run_id
    key = (scope, verdict.last_accepted_run_date, verdict.onset_run_date, base_run, onset_run)
    if key not in cache:
        stacks = _stacks(report, verdict)
        cache[key] = outcomes_for_window(
            report,
            base_release=verdict.last_accepted_run_date,
            onset_release=verdict.onset_run_date or "",
            stacks=stacks,
            regressed_scopes={scope},
            base_run=base_run,
            onset_run=onset_run,
        )
    return cache[key]


def _stacks(report: NightlyReport, verdict: MetricVerdict) -> set[str]:
    """The releases *verdict*'s run group measured — what makes another group a
    like-for-like measurement of the same window."""
    scope = (verdict.detector, verdict.platform, verdict.sample)
    return {
        g.k4h_release for g in report.groups
        if (g.detector, g.platform, g.sample) == scope and g.k4h_release
    }


def _sweeps(
    report: NightlyReport, verdict: MetricVerdict, cache: dict[tuple, tuple],
) -> tuple[ScopeSweep, ...]:
    """Every scope that measured *verdict*'s window, cached per window.

    A same-release window is also read and cached by its two runs, as
    :func:`rank_group_key` groups it: its releases alone are the empty interval
    ``(D, D]``, which would place the step that formed it in another window."""
    stacks = _stacks(report, verdict)
    base_run = onset_run = None
    if verdict.last_accepted_run_date == verdict.onset_run_date:
        base_run, onset_run = verdict.last_accepted_run_id, verdict.onset_run_id
    key = (
        verdict.last_accepted_run_date, verdict.onset_run_date,
        base_run, onset_run, frozenset(stacks),
    )
    if key not in cache:
        cache[key] = window_sweeps(
            report,
            base_release=verdict.last_accepted_run_date,
            onset_release=verdict.onset_run_date or "",
            stacks=stacks,
            base_run=base_run,
            onset_run=onset_run,
        )
    return cache[key]


def _geometry_path(report: NightlyReport, verdict: MetricVerdict) -> str:
    """The compact geometry file the run group behind *verdict* loads.

    Read off the report rather than carried on the verdict: it is a property of
    the *run*, not of one metric, and every group in a night records it once.
    Empty for runs benchmarked before the path was captured — the prompt then
    states nothing about geometry reach rather than guessing at it."""
    scope = (verdict.detector, verdict.platform, verdict.sample)
    return next(
        (
            g.geometry_path for g in report.groups
            if (g.detector, g.platform, g.sample) == scope and g.geometry_path
        ),
        "",
    )


def _assessment(result: RankResult) -> StepAssessment | None:
    """The sidecar's view of the ranker's step assessment.

    Re-shaped rather than passed through: :mod:`k4bench.blame.rank` owns the
    model contract and :mod:`k4bench.blame.models` owns what ``blame.json``
    stores, and letting one class serve both would tie the on-disk schema every
    dashboard parses to the wording of a prompt."""
    if result.assessment is None:
        return None
    return StepAssessment(
        verdict=result.assessment.verdict, reason=result.assessment.reason
    )


def _rank_group(
    ranker: Ranker,
    verdicts: list[MetricVerdict],
    repos: list[RepoBlame],
    texts: dict[tuple[str, int], PRText],
    rank_group: RankGroupKey,
    rank_cache: dict[RankGroupKey, RankResult],
    *,
    outcomes: tuple,
    changed_count: Callable[..., int | None],
    n_unchanged: int = 0,
    geometry_path: str = "",
    history: HistoricalIndex | None = None,
    sweeps: tuple[ScopeSweep, ...] = (),
    geometry: dict[str, str] | None = None,
    externals: tuple[ExternalChange, ...] = (),
    suite: str = SIM,
) -> RankResult:
    """The ranker's judgement of one rank group: a score per candidate, and its
    read of the step itself.

    Memoized on *rank_group* (:func:`rank_group_key`) — detector, platform,
    sample and the window, plus the window's run ids when its two ends are one
    release; never just the release boundary: every confirmed metric of *one run group*
    that stepped across one release boundary shares one diff and one candidate
    set, so it needs a single inference rather than one per metric — *every*
    metric in *verdicts* rides in the one prompt (see
    :class:`~k4bench.blame.rank.MetricStep`), so the model judges the
    candidates against that group's full picture, and the dashboard/email show
    that one verdict for every metric sharing the group, not a table each. A
    *different* detector or sample can share the same platform and release
    dates (one library regressing several detectors at once) — grouping on the
    release boundary alone would silently merge their unrelated metrics into
    one prompt mislabelled with a single detector/sample. ``label`` (a removal
    sweep's ``baseline`` vs. ``no_<detector>`` runs) is deliberately
    *not* part of this key — those still collapse into one verdict, each
    metric just carries its own label into the prompt.

    Ranking is skipped entirely when any repo's candidate discovery or file
    evidence came back incomplete (unavailable or truncated): the model would
    judge a partial or partially evidenced set, and its "most likely" would
    overclaim. An empty result (the ranker declined, raised, or was skipped)
    leaves the candidates unranked.
    """
    if any(
        r.commits_unavailable or r.truncated or r.truncation_reasons
        for r in repos
    ):
        details = ", ".join(
            f"{r.repo or r.package}: "
            f"{', '.join(r.truncation_reasons) if r.truncation_reasons else 'unspecified'}"
            for r in repos
            if r.commits_unavailable or r.truncated or r.truncation_reasons
        )
        _log.warning(
            "blame: %s/%s %s: candidate discovery or file evidence incomplete "
            "— leaving unranked (%s)",
            verdicts[0].detector, verdicts[0].sample,
            ", ".join(f"{v.metric} ({v.label})" for v in verdicts),
            details,
        )
        return RankResult()
    if rank_group not in rank_cache:
        rank_cache[rank_group] = _run_ranker(
            ranker, verdicts, repos, texts,
            outcomes=outcomes, changed_count=changed_count,
            n_unchanged=n_unchanged, geometry_path=geometry_path,
            history=history, sweeps=sweeps, geometry=geometry, externals=externals,
            suite=suite,
        )
    return rank_cache[rank_group]


def _run_ranker(
    ranker: Ranker,
    verdicts: list[MetricVerdict],
    repos: list[RepoBlame],
    texts: dict[tuple[str, int], PRText],
    *,
    outcomes: tuple,
    changed_count: Callable[..., int | None],
    n_unchanged: int = 0,
    geometry_path: str = "",
    history: HistoricalIndex | None = None,
    sweeps: tuple[ScopeSweep, ...] = (),
    geometry: dict[str, str] | None = None,
    externals: tuple[ExternalChange, ...] = (),
    suite: str = SIM,
) -> RankResult:
    """One guarded rank call. Any exception degrades to an empty result and is
    cached as such, so a broken ranker is asked at most once per detector/
    platform/sample/window and never aborts the report — blame's best-effort
    isolation, extended to the model."""
    try:
        request = _rank_request(
            verdicts, repos, texts,
            outcomes=outcomes, changed_count=changed_count,
            n_unchanged=n_unchanged, geometry_path=geometry_path,
            history=history, sweeps=sweeps, geometry=geometry, externals=externals,
            suite=suite,
        )
        if not request.candidates and not request.external_changes:
            return RankResult()
        result = ranker.rank(request)
        _warn_if_miscalibrated(request, result)
        return result
    except Exception:
        _log.exception("blame: ranker raised — leaving this window's candidates unranked")
        return RankResult()


def _packages_changed(
    verdict: MetricVerdict,
    changed_count: Callable[..., int | None],
) -> dict[str, int | None]:
    """``release -> tracked packages that moved entering it`` across one
    verdict's history tail.

    The annotation that turns a list of numbers into a calibration: a release
    where the metric moved and *nothing* in the stack changed measures this
    series' own noise directly, with the software held fixed. The oldest point
    has no predecessor in the tail and therefore no boundary — ``None``, unread,
    which is what the prompt says about it. So is a boundary where the detector
    config changed: the geometry moved there whatever the stack did.
    """
    changed: dict[str, int | None] = {}
    previous: str | None = None
    previous_platform: str | None = None
    previous_detector: str | None = None
    for point in verdict.history:
        platform = point.platform or verdict.platform
        detector = point.detector or verdict.detector
        changed[point.run_date] = (
            changed_count(
                platform, previous, point.run_date,
                None if previous_platform == platform else previous_platform,
            )
            if previous and previous_detector == detector else None
        )
        previous, previous_platform, previous_detector = (
            point.run_date, platform, detector,
        )
    return changed


def _rank_request(
    verdicts: list[MetricVerdict],
    repos: list[RepoBlame],
    texts: dict[tuple[str, int], PRText],
    *,
    outcomes: tuple,
    changed_count: Callable[..., int | None],
    n_unchanged: int = 0,
    geometry_path: str = "",
    history: HistoricalIndex | None = None,
    sweeps: tuple[ScopeSweep, ...] = (),
    geometry: dict[str, str] | None = None,
    externals: tuple[ExternalChange, ...] = (),
    suite: str = SIM,
) -> RankRequest:
    """Assemble the ranker's input: every metric that stepped across the shared
    window with its own recent history, the configurations that measured the
    same window without stepping, and every candidate PR across the changed
    repos, each carried with its transient patch. *verdicts* all share the run
    group (detector, platform, sample) and window, so the first stands in for
    those shared facts; each metric keeps its own ``label`` (verdicts sharing a
    group and window can still come from different benchmark configs, e.g. a
    removal sweep's ``baseline`` and ``no_<detector>`` runs).

    A verdict from a report written before histories were recorded simply
    carries none, and the prompt renders without that block — the ranking path
    must not depend on a field a historical backfill cannot supply.

    Each candidate's diff sample reads this run's own compact directory first,
    then its geometry tree (:func:`~k4bench.blame.github.diff_sample`): a pull
    request touching several detector variants would otherwise spend its sample
    on whichever sort first, and the variant this run loads could be missing
    from it. Without per-file hunks, the generic sample stands.

    *sweeps* are every scope that measured the window (this one included) and
    *geometry* the compact file each benchmarked detector loads: together they
    are the sweep table, the other detectors each candidate reaches
    (:func:`~k4bench.blame.geometry.detector_touches`), and what those did."""
    v = verdicts[0]
    metrics = tuple(
        MetricStep(
            metric=m.metric, metric_family=m.metric_family,
            direction=m.direction.value, pct_change=m.pct_change,
            label=m.label, sub_detector=m.sub_detector,
            value=m.value, baseline_median=m.baseline_median, z_score=m.z_score,
            history=history_from_verdict(
                m, packages_changed=_packages_changed(m, changed_count)
            ),
            regions=m.region_deltas,
            components=m.component_deltas,
        )
        for m in verdicts
    )
    own_dirs = (compact_dir(geometry_path),)
    trees = (geometry_tree(geometry_path),)

    def patch(text: PRText) -> str:
        if not text.files:
            return text.patch
        return diff_sample(text.files, own_dirs=own_dirs, trees=trees)

    candidates = tuple(
        RankCandidate(
            repo=pr.repo, number=pr.number, title=pr.title,
            files=pr.files,
            patch=patch(texts.get((pr.repo, pr.number), PRText())),
            body=texts.get((pr.repo, pr.number), PRText()).body,
            additions=pr.additions, deletions=pr.deletions,
            touches=detector_touches(
                pr.files, texts.get((pr.repo, pr.number), PRText()).files,
                geometry or {},
            ),
        )
        for repo in repos
        for pr in repo.candidates
    )
    scope = (v.detector, v.platform, v.sample)
    return RankRequest(
        metrics=metrics,
        suite=suite,
        detector=v.detector, platform=v.platform, sample=v.sample,
        base_release=v.last_accepted_run_date,
        onset_release=v.onset_run_date,
        base_platform=v.last_accepted_platform,
        onset_platform=v.onset_platform,
        base_detector=v.last_accepted_detector,
        onset_detector=v.onset_detector,
        candidates=candidates,
        outcomes=outcomes,
        n_unchanged=n_unchanged,
        geometry_tree=geometry_path,
        harness_repo=next(
            (r.repo or "" for r in repos if r.package == HARNESS_PACKAGE), ""
        ),
        history=history,
        sweep=next((sweep for sweep in sweeps if sweep.scope == scope), None),
        window_sweeps=sweeps,
        external_changes=externals,
    )


#: A score above this on a candidate that changes nothing a benchmark executes
#: is a calibration failure worth a line in the log. Set above the comment
#: threshold's neighbourhood on purpose: a model idly putting a docs change at
#: 30 is ordinary hedging, while one putting it at 70 is not reading the diff.
_CANARY_SCORE = 60.0


def _warn_if_miscalibrated(request: RankRequest, result: RankResult) -> None:
    """Log when the ranker scores a change that cannot have caused a runtime
    regression.

    Every window carries a few candidates that are structurally incapable of
    moving a benchmark — documentation, licences, CI configuration — and how a
    model treats *those* is a free, ground-truth-free read on whether it is
    reading diffs or matching words. Diagnostic only: it never suppresses a
    ranking or moves a score, because a rule confident enough to act on would
    have to classify paths confidently enough to be wrong about a real one."""
    by_key = {(c.repo, c.number): c for c in request.candidates}
    suspect = [
        f"{repo}#{number} ({ranking.score:.0f})"
        for (repo, number), ranking in result.rankings.items()
        if ranking.score >= _CANARY_SCORE
        and (candidate := by_key.get((repo, number))) is not None
        and candidate.files
        and all(low_signal_path(f) for f in candidate.files)
    ]
    if suspect:
        _log.warning(
            "blame: %s/%s %s — the ranker scored documentation/CI-only "
            "change(s) at or above %.0f: %s. That window's ranking is not "
            "reading the diffs; treat its scores with suspicion.",
            request.detector, request.sample, request.onset_release,
            _CANARY_SCORE, ", ".join(sorted(suspect)),
        )


def _apply_rankings(
    repo: RepoBlame, rankings: dict[tuple[str, int], Ranking]
) -> RepoBlame:
    """Fold the ranker's verdict onto a repo's candidates, matched on
    ``(repo, number)``. A candidate the ranking omitted — a partial response
    leaves some out — keeps ``ranked=False``, the *no judgement* state, rather
    than a 0.0 that would read as one; a ranking keyed to a PR not in this repo
    is never looked up, so unknown keys drop out here as required."""
    candidates = tuple(
        dataclasses.replace(
            pr, score=ranking.score, description=ranking.description,
            against=ranking.against, ranked=True,
        )
        if (ranking := rankings.get((pr.repo, pr.number))) is not None
        else pr
        for pr in repo.candidates
    )
    return dataclasses.replace(repo, candidates=candidates)
