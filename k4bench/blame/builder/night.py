"""One night's blame build, keyed by release.

The stack diff across a release boundary, the pull requests resolved for each
changed repository and the historical retrieval are shared by every rank group
of the night, so :class:`_NightBlame` caches them once. What a single run
recorded is read in :mod:`.runs`; the ranker's input is assembled in
:mod:`.ranking`.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from k4bench.blame.builder.ranking import (
    _apply_rankings,
    _assessment,
    _geometry_path,
    _outcomes,
    _packages_changed,
    _rank_group,
    _sweeps,
)
from k4bench.blame.builder.runs import (
    K4BenchCommitForRun,
    _external_facts,
    _harness_candidate_signal,
    _harness_change,
    _loaded_packages,
)
from k4bench.blame.geometry import benchmarked_geometry
from k4bench.blame.github import (
    GitHubClient,
    PRText,
    RateLimitError,
    RepoResolution,
    resolve_repo_prs,
)
from k4bench.blame.history import (
    HistoricalEvidence,
    HistoricalIndex,
    HistoricalPR,
    HistoricalRequest,
    build_index,
    cap_evidence,
)
from k4bench.blame.models import (
    BlameEntry,
    BlameReport,
    HistoricalRef,
    RankGroupKey,
    RepoBlame,
    StepAssessment,
    rank_group_key,
)
from k4bench.blame.rank import (
    Ranker,
    RankResult,
)
from k4bench.blame.sweep import ScopeSweep
from k4bench.provenance.diff import (
    CHANGED,
    PackageChange,
    diff_packages,
    package_key,
    unchanged_packages,
)
from k4bench.regression.models import MetricVerdict, NightlyReport

_log = logging.getLogger(__name__)

PackagesForRelease = Callable[[str, str], "dict | None"]


def _attributable(v: MetricVerdict) -> bool:
    """True for a confirmed regression whose window can name a change.

    A ``(base, onset]`` window spanning two releases can be diffed upstream; a
    *same-release* window (``base == onset``) proves the upstream stack did not
    move at all — which makes it the highest-signal case for the harness's own
    commits, the one thing that varies between two runs of one release. Both
    are admitted; a same-release window still produces an entry only when the
    harness actually moved (see :meth:`_NightBlame.entry_for`). An open window
    (no settled baseline) has nothing to anchor either end on and is dropped."""
    base, onset = v.last_accepted_run_date, v.onset_run_date
    return bool(onset and base and base <= onset)


def build_blame_report(
    report: NightlyReport,
    *,
    packages_for_release: PackagesForRelease,
    k4bench_commit_for_run: K4BenchCommitForRun | None = None,
    provenance_for_run: Callable | None = None,
    github: GitHubClient | None = None,
    ranker: Ranker | None = None,
    generated_at: str | None = None,
    historical_diffs: bool = True,
) -> BlameReport:
    """Build the night's blame from *report* and injected provenance/GitHub access.

    ``ranker`` is the optional ranking stage (:mod:`k4bench.blame.rank`): given
    one, each entry's candidates are scored and described; given ``None`` (the
    default, and every environment without ``K4BENCH_LLM_*`` configured), they
    are collected unranked. Ranking is fully isolated — a ranker that fails or
    raises leaves that window's candidates unranked, never aborting the report.

    ``historical_diffs`` controls on-demand historical evidence
    (:mod:`k4bench.blame.history`): each rank group is additionally offered a
    lightweight index of the older release boundaries in its own metrics'
    history, and the model may ask to read the code behind one of them before
    judging. On by default, and inert without a GitHub client whatever it is set
    to — an index that cannot be redeemed is an offer the application cannot
    keep. Passing ``False`` turns it off entirely: no index, no extra model
    call, no extra GitHub read, and no historical material in the prompts.
    """
    night = _NightBlame(
        report,
        packages_for_release=_memoized(packages_for_release),
        provenance_for_run=_memoized(provenance_for_run) if provenance_for_run else None,
        k4bench_commit_for_run=(
            _memoized(k4bench_commit_for_run)
            if k4bench_commit_for_run is not None else None
        ),
        github=github,
        ranker=ranker,
        historical_diffs=historical_diffs,
    )
    entries = [
        entry for v in night.verdicts if (entry := night.entry_for(v)) is not None
    ]
    return BlameReport(
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        report_night=report.report_night,
        entries=tuple(entries),
    )


class _NightBlame:
    """One night's blame build: the injected sources, everything the night reads
    once and shares across its regressions, and the GitHub rate-limit latch.

    It is also the :class:`~k4bench.blame.history.HistoricalProvider` every rank
    group's historical index is redeemed through (:meth:`fetch`), so a
    retrieval reuses the night's release diffs, repository resolutions and latch
    rather than rebuilding them.
    """

    def __init__(
        self,
        report: NightlyReport,
        *,
        packages_for_release: PackagesForRelease,
        k4bench_commit_for_run: K4BenchCommitForRun | None,
        provenance_for_run: Callable | None,
        github: GitHubClient | None,
        ranker: Ranker | None,
        historical_diffs: bool,
    ) -> None:
        self.provenance_for_run = provenance_for_run
        self.external_cache = {}
        self.loaded_cache: dict[RankGroupKey, frozenset[str] | None] = {}
        self.report = report
        self.packages_for_release = packages_for_release
        self.k4bench_commit_for_run = k4bench_commit_for_run
        self.github = github
        self.ranker = ranker
        self.historical_diffs = historical_diffs
        self.verdicts = [v for v in report.regressions if _attributable(v)]

        #: Every confirmed metric that stepped across a given (detector,
        #: platform, sample) run group's release boundary, gathered upfront so
        #: the ranker sees that group's full picture — not just whichever verdict
        #: happens to reach it first. Keyed finer than the diff/resolution caches
        #: below: two *different* detectors or samples can share the same
        #: platform and release dates — a library regressing several detectors in
        #: one release — and must never be batched into one prompt under one
        #: detector/sample's identity. ``label`` (a removal sweep's ``baseline``
        #: vs. ``no_<detector>`` runs, say) is deliberately *not* part of this
        #: key: labels sharing a group and window still get one collapsed
        #: verdict, not one call each — only detector/sample are independent
        #: enough to require splitting. Each verdict keeps its own label in the
        #: prompt (see :class:`~k4bench.blame.rank.MetricStep`) so the model can
        #: still tell configs apart without the batch being fragmented over them.
        #: A same-release window additionally keys on its two *run* ids — see
        #: :func:`rank_group_key`, which owns the whole rule.
        self.verdicts_by_rank_group: dict[RankGroupKey, list[MetricVerdict]] = {}
        for v in self.verdicts:
            self.verdicts_by_rank_group.setdefault(rank_group_key(v), []).append(v)

        #: Keyed ``(base platform, base release, onset platform, onset
        #: release)``: a window opened on a replaced platform reads its base end
        #: there.
        self.diff_cache: dict[tuple[str, str, str, str], list[PackageChange] | None] = {}
        self.unchanged_cache: dict[tuple[str, str, str, str], int] = {}
        self.resolution_cache: dict[tuple[str, str, str], RepoResolution] = {}
        #: The harness's own movement, resolved once per rank group and keyed
        #: like :attr:`verdicts_by_rank_group` (:func:`rank_group_key`) — which is
        #: what keeps two different run windows inside one release from sharing
        #: a range they do not have. The range itself comes from the group's run
        #: ids (:func:`_harness_change`), so a group is resolved once,
        #: consistently, rather than by whichever verdict is reached first.
        #: Deliberately separate from :attr:`diff_cache`: that one is
        #: release-keyed and shared with the historical walk, whose boundaries
        #: have no run ids at all, and the harness must never leak into it.
        self.harness_cache: dict[RankGroupKey, PackageChange | None] = {}
        #: One rank inference per rank group, shared by every metric that
        #: stepped across it (they see the same diff and candidate set) — the
        #: dashboard and the email show one verdict per group, not one per
        #: metric. Keyed like :attr:`verdicts_by_rank_group`, not the coarser
        #: diff/resolution window: the diff/candidate PRs really are
        #: platform+release-scoped (every detector on a platform shares one
        #: package set), but the *prompt text* names one detector/sample and
        #: must not be shared across them.
        self.rank_cache: dict[RankGroupKey, RankResult] = {}
        #: Non-confirming configurations per ``(scope, window)`` — the controls
        #: the ranker weighs reach against. Cached because one window's controls
        #: are the same for every metric of a run group, and computing them walks
        #: the whole report.
        self.outcome_cache: dict[tuple, tuple] = {}
        #: Every scope that measured a window
        #: (:func:`~k4bench.blame.sweep.window_sweeps`), per ``(window,
        #: stacks)``: every rank group of one window reads the same suite-wide
        #: picture, and building it walks the whole report.
        self.sweep_cache: dict[tuple, tuple[ScopeSweep, ...]] = {}
        #: The compact file each benchmarked detector loads, for mapping a
        #: candidate's files onto the detectors they reach.
        self.geometry = benchmarked_geometry(report)
        #: Set once GitHub throttles: from then on repos keep their diffs but get
        #: no candidates, rather than each retry re-hitting the same wall.
        self.rate_limited = False
        #: What the historical retrieval cost, for the operator: reads that
        #: reached GitHub versus ones the night had already paid for. A shared
        #: history is the common case (every metric of a platform walks the same
        #: release tail), so a cache that stopped working would show up here long
        #: before it showed up in a bill.
        self.history_reads = {"api": 0, "cache": 0}

    def changed_packages(
        self, platform: str, base: str | None, onset: str, base_platform: str | None = None,
    ) -> list[PackageChange] | None:
        """The tracked packages that moved across one release boundary, or
        ``None`` when that boundary's provenance could not be read.

        Shares :attr:`diff_cache` with the attribution windows — the two ask
        exactly the same question of exactly the same data, and a boundary that
        is a history step for one metric is the change window of another.
        ``[]`` (the stack stood still) and ``None`` (nobody could look) are
        different answers and stay different all the way to the prompt.

        *platform* measured *onset*; *base_platform* measured *base* when that
        was a different platform."""
        if not base or base >= onset:
            return None
        key = (base_platform or platform, base, platform, onset)
        if key not in self.diff_cache:
            self.diff_cache[key], self.unchanged_cache[key] = _diff_window(
                self.packages_for_release, *key
            )
        return self.diff_cache[key]

    def changed_count(
        self, platform: str, base: str | None, onset: str, base_platform: str | None = None,
    ) -> int | None:
        """How many tracked packages moved across one boundary — the number the
        history table states, and the one a metric's own noise is read against."""
        changes = self.changed_packages(platform, base, onset, base_platform)
        return None if changes is None else len(changes)

    def fetch(self, request: HistoricalRequest) -> HistoricalEvidence:
        """Retrieve one validated historical selection, or refuse it.

        Reuses what this build already has and a standalone provider would have
        to rebuild: the memoized release diffs (so a boundary already read costs
        nothing) and :attr:`resolution_cache` keyed on ``(slug, base, head)`` (so
        a range some other window already resolved is not resolved again). The
        rate-limit latch is shared for the same reason — a throttled token is a
        fact about the night, not about one lookup.

        Every incomplete path returns an incomplete result rather than raising:
        the ranker turns that into an unranked window, which is the honest
        answer, and a raised exception would have to be caught somewhere to say
        the same thing less clearly.
        """
        if self.github is None:
            return HistoricalEvidence(
                complete=False, reason="no GitHub client is configured"
            )
        prs: list[HistoricalPR] = []
        for selection in request.selections:
            boundary = selection.boundary
            changes = self.changed_packages(
                boundary.platform, boundary.base_release, boundary.onset_release,
                boundary.base_platform,
            )
            if changes is None:
                return HistoricalEvidence(
                    complete=False,
                    reason=f"provenance for boundary {boundary.id} is unreadable",
                )
            by_name = {change.name: change for change in changes}
            for package in selection.packages:
                change = by_name.get(package.name)
                repo = change.repo if change is not None else None
                if (
                    change is None
                    or change.status != CHANGED
                    or not change.base_commit or not change.head_commit
                    or repo is None or repo.forge != "github"
                ):
                    # The index said this package was retrievable and the diff
                    # now says otherwise, which means the two were read from
                    # different provenance. Refusing is the only honest move.
                    return HistoricalEvidence(
                        complete=False,
                        reason=(
                            f"{package.name} has no readable commit range at "
                            f"boundary {boundary.id}"
                        ),
                    )
                if self.rate_limited:
                    return HistoricalEvidence(
                        complete=False,
                        reason="GitHub is rate-limited for the rest of the night",
                    )
                key = (repo.slug, change.base_commit, change.head_commit)
                if key in self.resolution_cache:
                    self.history_reads["cache"] += 1
                else:
                    self.history_reads["api"] += 1
                    try:
                        self.resolution_cache[key] = resolve_repo_prs(
                            self.github, repo.slug, change.base_commit, change.head_commit
                        )
                    except RateLimitError:
                        self.rate_limited = True
                        return HistoricalEvidence(
                            complete=False, reason="GitHub rate limit"
                        )
                    except Exception as exc:  # noqa: BLE001 — refuse, never raise
                        return HistoricalEvidence(
                            complete=False,
                            reason=f"resolving {repo.slug} failed: {exc}",
                        )
                resolution = self.resolution_cache[key]
                if (
                    resolution.commits_unavailable
                    or resolution.truncated
                    or resolution.truncation_reasons
                ):
                    return HistoricalEvidence(
                        complete=False,
                        reason=(
                            f"{repo.slug} at boundary {boundary.id} came back "
                            f"incomplete ("
                            + (
                                ", ".join(sorted(resolution.truncation_reasons))
                                or "commits unavailable"
                            )
                            + ")"
                        ),
                    )
                prs.extend(
                    HistoricalPR(
                        boundary_id=boundary.id,
                        base_release=boundary.base_release,
                        onset_release=boundary.onset_release,
                        package=package.name,
                        repo=pr.repo,
                        number=pr.number,
                        title=pr.title,
                        files=pr.files,
                        additions=pr.additions,
                        deletions=pr.deletions,
                        body=resolution.bodies.get(pr.number, ""),
                        patch=resolution.patches.get(pr.number, ""),
                    )
                    for pr in resolution.candidates
                )
        evidence = cap_evidence(prs)
        _log.info(
            "blame: historical retrieval [%s] -> %d pull request(s), complete=%s "
            "(%d GitHub read(s), %d served from cache)",
            request.describe(), len(evidence.prs), evidence.complete,
            self.history_reads["api"], self.history_reads["cache"],
        )
        return evidence

    def historical_index(
        self, rank_verdicts: list[MetricVerdict], window: tuple[str, str | None, str, str],
    ) -> HistoricalIndex | None:
        """The older boundaries this rank group may ask to read, or ``None``.

        Built from the rank group's own metrics' history tails — the releases
        the prompt already shows — so the offer never names a boundary the model
        has no reason to care about. The current window is excluded: its
        packages are already in the prompt as scored candidates, and offering
        them again as "history" would invite the same pull requests to be
        reasoned about twice under two different rules."""
        if not self.historical_diffs or self.github is None:
            return None
        base_platform, base, platform, onset = window
        index = build_index(
            [
                (
                    v.platform,
                    [point.run_date for point in v.history],
                    [point.platform for point in v.history],
                )
                for v in rank_verdicts
            ],
            changed_packages=self.changed_packages,
            exclude={(base_platform, base or "", platform, onset)}
            | _detector_switch_boundaries(rank_verdicts),
        )
        # The index is the offer; this build is how it is redeemed. Built apart
        # so the offer stays a pure function of provenance the report already
        # read, and testable without anything that can reach GitHub.
        return dataclasses.replace(index, provider=self)

    def resolve(self, change: PackageChange) -> RepoResolution:
        """Resolve one changed repo's PRs, memoized on ``(slug, base, head)``.

        Only a *changed* GitHub package with both endpoints is resolvable — an
        added/removed package has no range, and a non-GitHub host no resolvable
        PRs; those return a plain empty resolution. A resolvable repo that could
        *not* be asked — the night is rate-limited, or the resolution raised —
        comes back with ``commits_unavailable`` set instead: "no candidates" and
        "never looked" must stay distinguishable, or a partial candidate set
        would read as a complete one."""
        repo = change.repo
        if (
            self.github is None
            or change.status != CHANGED
            or repo is None or repo.forge != "github"
            or not change.base_commit or not change.head_commit
        ):
            return RepoResolution()
        if self.rate_limited:
            return RepoResolution(commits_unavailable=True)

        key = (repo.slug, change.base_commit, change.head_commit)
        if key not in self.resolution_cache:
            try:
                self.resolution_cache[key] = resolve_repo_prs(
                    self.github, repo.slug, change.base_commit, change.head_commit
                )
            except RateLimitError:
                _log.warning("blame: GitHub rate limit — remaining repos get no candidates")
                self.rate_limited = True
                return RepoResolution(commits_unavailable=True)
            except Exception:
                _log.exception("blame: resolving %s failed", repo.slug)
                self.resolution_cache[key] = RepoResolution(commits_unavailable=True)
        return self.resolution_cache[key]

    def entry_for(self, v: MetricVerdict) -> BlameEntry | None:
        """*v*'s blame entry, or ``None`` when its window cannot name a change."""
        window = (
            v.base_platform, v.last_accepted_run_date,
            v.onset_run_platform, v.onset_run_date,
        )
        if window not in self.diff_cache:
            if window[0] == window[2] and window[1] == window[3]:
                # A same-release window: both runs sourced the same immutable
                # stack, so the upstream diff is ``[]`` *by construction* — no
                # provenance read can change that answer, and an unreadable
                # package map costs only the unchanged count, never the entry.
                packages = self.packages_for_release(v.onset_run_platform, v.onset_run_date)
                self.diff_cache[window] = []
                self.unchanged_cache[window] = len(packages) if packages else 0
            else:
                self.diff_cache[window], self.unchanged_cache[window] = _diff_window(
                    self.packages_for_release, *window
                )
        changes = self.diff_cache[window]
        if changes is None:
            # Upstream provenance is unreadable. Skip even when the harness is
            # known to have moved: an entry naming only k4Bench while the stack
            # is unknown would read as "the stack was checked and stood still".
            return None
        rank_group = rank_group_key(v)
        if rank_group not in self.loaded_cache:
            self.loaded_cache[rank_group] = _loaded_packages(
                self.verdicts_by_rank_group[rank_group], self.provenance_for_run,
            )
        loaded = self.loaded_cache[rank_group]
        if loaded is not None:
            # A package neither end of this group's runs mapped, and that has
            # no unmapped code they could have run, cannot have moved its
            # measurements, so its pull requests are never candidates.
            changes = [c for c in changes if package_key(c.name) in loaded]
        if rank_group not in self.harness_cache:
            self.harness_cache[rank_group] = _harness_change(
                self.k4bench_commit_for_run, self.verdicts_by_rank_group[rank_group]
            )
        harness = self.harness_cache[rank_group]
        if harness is not None:
            # Appended per verdict, never written into the release-keyed diff
            # or its caches: the harness range belongs to this group's runs.
            # ``n_unchanged`` stays untouched too — it counts the *stack*, and
            # the harness is not one of its packages.
            changes = [*changes, harness]
        if rank_group not in self.external_cache:
            self.external_cache[rank_group] = _external_facts(
                self.verdicts_by_rank_group[rank_group], self.provenance_for_run,
            )
        externals = self.external_cache[rank_group]
        if not changes and not externals:
            # The releases differ but every tracked package is identical, and
            # the harness (if it could be read at all) did not move either.
            return None

        repos: list[RepoBlame] = []
        texts: dict[tuple[str, int], PRText] = {}
        for change in changes:
            resolution = self.resolve(change)
            if change is harness and resolution.candidates:
                resolution = dataclasses.replace(
                    resolution,
                    candidates=[
                        c for c in resolution.candidates
                        if _harness_candidate_signal(c.files)
                    ],
                )
            repos.append(_repo_blame(change, resolution))
            for pr in resolution.candidates:
                texts[(pr.repo, pr.number)] = PRText(
                    patch=resolution.patches.get(pr.number, ""),
                    body=resolution.bodies.get(pr.number, ""),
                    files=resolution.files.get(pr.number, ()),
                )

        assessment: StepAssessment | None = None
        historical: tuple[HistoricalRef, ...] = ()
        if self.ranker is not None:
            group_verdicts = self.verdicts_by_rank_group[rank_group]
            result = _rank_group(
                self.ranker, group_verdicts, repos, texts,
                rank_group, self.rank_cache,
                outcomes=_outcomes(self.report, v, self.outcome_cache),
                changed_count=self.changed_count,
                n_unchanged=self.unchanged_cache[window],
                geometry_path=_geometry_path(self.report, v),
                history=self.historical_index(group_verdicts, window),
                sweeps=_sweeps(self.report, v, self.sweep_cache),
                geometry=self.geometry,
                externals=externals,
            )
            if result.rankings:
                repos = [_apply_rankings(r, result.rankings) for r in repos]
            assessment = _assessment(result)
            # The selection is the rank group's, not the metric's: one call
            # produced it, and every entry sharing that call records the same
            # references so the comment pass reads one evidence set whichever
            # entry it holds.
            historical = tuple(_historical_ref(pr) for pr in result.historical)

        return BlameEntry(
            detector=v.detector, platform=v.platform, sample=v.sample,
            label=v.label, metric=v.metric, sub_detector=v.sub_detector,
            base_release=v.last_accepted_run_date, onset_release=v.onset_run_date,
            repos=tuple(repos), n_unchanged=self.unchanged_cache[window],
            assessment=assessment,
            # Persisted for the cross-configuration pass, which has no provenance
            # access of its own: without this it would render every boundary of
            # every history as unread and lose the sharpest noise measurement the
            # suite produces. Unknown boundaries are simply absent (see
            # :func:`_packages_changed`), so the map never claims a stack stood
            # still that nobody looked at.
            boundary_changes={
                release: count
                for release, count in _packages_changed(v, self.changed_count).items()
                if count is not None
            },
            historical_evidence=historical,
            external_changes=externals,
        )


def _memoized(lookup):
    """A string-keyed lookup answering each argument tuple once — serving both
    the ``(platform, release)`` package lookup and the run-keyed harness-commit
    lookup (:data:`K4BenchCommitForRun`).

    The history tails multiplied the question: a window's own diff asks for two
    releases, a twelve-release tail asks for twelve, and every metric of every
    detector on a platform asks for the same ones. The underlying lookup globs a
    run cache or fetches over the network, and neither should happen twice for
    one key. ``None`` (nothing readable) is cached too — it is an answer, and
    re-asking would re-pay for it."""
    cache: dict[tuple, object] = {}

    def memoized(*key: str):
        if key not in cache:
            cache[key] = lookup(*key)
        return cache[key]

    return memoized


def _diff_window(
    packages_for_release: PackagesForRelease,
    base_platform: str, base: str, onset_platform: str, onset: str,
) -> tuple[list[PackageChange] | None, int]:
    """The changed packages and unchanged count for one window, or ``(None, 0)``
    when either release's provenance is unavailable. Each end is read under the
    platform it was measured on."""
    base_pkgs = packages_for_release(base_platform, base)
    head_pkgs = packages_for_release(onset_platform, onset)
    if not base_pkgs or not head_pkgs:
        _log.info(
            "blame: no provenance for %s %s..%s %s",
            base_platform, base, onset_platform, onset,
        )
        return None, 0
    return diff_packages(base_pkgs, head_pkgs), len(unchanged_packages(base_pkgs, head_pkgs))


def _repo_blame(change: PackageChange, resolution: RepoResolution) -> RepoBlame:
    """Compose a :class:`RepoBlame`: the diff facts from *change* and the
    candidate PRs GitHub found in its range.

    Candidates are left **unranked** here — a ``score``/``description`` is the
    ranking stage's job, and it judges every candidate of a regression together
    (a PR in one repo can only be assessed against the others), so ranking
    belongs above the per-repo assembly, not inside it.
    """
    repo = change.repo
    return RepoBlame(
        package=change.name,
        repo=repo.slug if repo and repo.forge == "github" else None,
        base_commit=change.base_commit,
        head_commit=change.head_commit,
        compare_url=change.compare_url,
        status=change.status,
        candidates=tuple(resolution.candidates),
        commits_unavailable=resolution.commits_unavailable,
        truncated=resolution.truncated or bool(resolution.truncation_reasons),
        truncation_reasons=tuple(sorted(resolution.truncation_reasons)),
    )


def _historical_ref(pr: HistoricalPR) -> HistoricalRef:
    """One retrieved analogue as the sidecar stores it — a reference, never the
    text.

    The diff and the description that reached the model are dropped here on
    purpose: they are re-fetchable from GitHub forever (the comment pass fetches
    them again from exactly this reference), and a sidecar the dashboard parses
    on every page load has no business carrying a copy of somebody's patch."""
    return HistoricalRef(
        boundary_id=pr.boundary_id,
        base_release=pr.base_release,
        onset_release=pr.onset_release,
        package=pr.package,
        repo=pr.repo,
        pr=pr.number,
        title=pr.title,
        files=pr.files,
        additions=pr.additions,
        deletions=pr.deletions,
    )


def _detector_switch_boundaries(verdicts: list[MetricVerdict]) -> set[tuple]:
    """``(base platform, base, onset platform, onset)`` for every boundary in
    *verdicts*' history tails where the detector config changed — a geometry
    version switch, never an upstream change worth offering as history."""
    switches = set()
    for v in verdicts:
        for earlier, later in zip(v.history, v.history[1:]):
            if (earlier.detector or v.detector) != (later.detector or v.detector):
                switches.add((
                    earlier.platform or v.platform, earlier.run_date,
                    later.platform or v.platform, later.run_date,
                ))
    return switches
