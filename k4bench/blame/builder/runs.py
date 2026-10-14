"""What the two exact runs of one rank group's window recorded.

The harness's own commit range and the loaded stack vary per run, not per
release, so they are read from the runs a verdict names and never enter the
night's release-keyed caches.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from k4bench.blame.evidence import ExternalChange, external_changes
from k4bench.blame.github import low_signal_path
from k4bench.blame.prompt import HARNESS_PACKAGE
from k4bench.provenance.diff import (
    PackageChange,
    package_key,
)
from k4bench.regression.models import MetricVerdict

_log = logging.getLogger(__name__)

#: ``(detector, platform, release, sample, run_id) -> (commit_sha,
#: github_run_url)`` — what one exact *run* recorded about the k4Bench
#: checkout that produced it, or ``None`` when that run's ``run_info.json``
#: cannot be read. Keyed by run rather than by release because the harness's
#: commit varies per run: several consecutive runs routinely share one Key4hep
#: release, each benchmarked by a different k4Bench commit, so the
#: release-keyed :data:`PackagesForRelease` shape would answer with whichever
#: run it happened to read first. And keyed down to detector and sample —
#: naming one ``run_info.json`` — rather than "any run of that night": a
#: manual dispatch or a partially failed re-run can leave the same date
#: carrying different commits under different benchmark groups, and the group
#: being attributed must read its own run's provenance, never a sibling's.
#: ``github_run_url`` may be ``None`` on runs that predate the field; the
#: commit alone still places the run in the default repository.
K4BenchCommitForRun = Callable[
    [str, str, str, str, str], "tuple[str, str | None] | None"
]


#: Where the harness lives when a run's own provenance does not say. Every
#: current run records its workflow URL and the repository is parsed from it
#: (:func:`_repo_url_from_run_url`); this constant covers only runs written
#: before that field existed.
_K4BENCH_REPO_URL = "https://github.com/key4hep/k4Bench.git"


#: Harness paths that provably cannot move a measurement: documentation, the
#: dashboard and its assets/deployment, the test suite, and the blame pipeline
#: itself (it reads reports after the fact; nothing it computes feeds a
#: measured value). Everything else is kept — the geometry patcher, the runner,
#: the timing plugin, the nightly scripts and workflows, packaging — because
#: the filter is written by *exclusion*: an exhaustive list of what CAN move a
#: measurement would silently rot the first time a measurement-relevant file
#: was added, and a filter that hides the true culprit fails in exactly the
#: direction this feature exists to fix. A surviving irrelevant candidate only
#: costs the ranker a dismissal.
_HARNESS_LOW_SIGNAL_PREFIXES = (
    "docs/",
    "doc/",
    "dashboard/",
    "assets/",
    "openshift/",
    "tests/",
    "k4bench/blame/",
    "mkdocs.",
)


def _repo_url_from_run_url(run_url: str | None) -> str | None:
    """The harness repository behind a run's recorded workflow URL, or ``None``.

    ``github_run_url`` looks like
    ``https://github.com/key4hep/k4Bench/actions/runs/30603409866`` — the first
    two path segments are the repository. Derived rather than assumed so a fork
    running its own benchmarks blames its own pull requests, not upstream's."""
    if not run_url:
        return None
    prefix = "https://github.com/"
    if not run_url.startswith(prefix):
        return None
    segments = [s for s in run_url[len(prefix):].split("/") if s]
    if len(segments) < 2:
        return None
    return f"https://github.com/{segments[0]}/{segments[1]}.git"


def _harness_change(
    k4bench_commit_for_run, verdicts: list[MetricVerdict]
) -> PackageChange | None:
    """k4Bench's own movement across one rank group's window, or ``None``.

    The endpoints come from the *run ids* already on the verdicts — the base
    end is the run named by ``last_accepted_run_id`` (a run of the base
    release), the onset end the run named by ``onset_run_id`` (a run of the
    onset release) — never from a release-keyed lookup: a release is routinely
    measured by several runs at different harness commits, and "the first
    readable run under the release" would pick one essentially at random. Each
    endpoint is read from this group's own run — the lookup is keyed down to
    detector and sample — so a same-day re-run that left a sibling group at a
    different commit cannot leak into this group's range.

    A *same-release* group agrees on its run ids by construction — they are
    part of its key (:func:`rank_group_key`). A cross-release group is keyed on
    release dates only, so nothing enforces agreement there; when its verdicts
    disagree, the narrowest range consistent with all of them wins (newest base
    run, oldest onset run) — a false negative is cheaper than accusing a pull
    request that merged after the step was already measured — and the
    disagreement is logged.

    ``None`` — no lookup injected, an endpoint that cannot be read, a harness
    that did not move, or two endpoints belonging to different repositories
    (a compare range across two forks would be meaningless; an end with no
    recorded workflow URL counts as the default repository, not as a wildcard
    matching whatever the other end says) — must leave the report exactly as
    it was before this lookup existed."""
    if k4bench_commit_for_run is None:
        return None
    v = verdicts[0]
    base_ids = {m.last_accepted_run_id for m in verdicts if m.last_accepted_run_id}
    onset_ids = {m.onset_run_id for m in verdicts if m.onset_run_id}
    if not base_ids or not onset_ids:
        return None
    if len(base_ids) > 1 or len(onset_ids) > 1:
        _log.warning(
            "blame: %s/%s %s..%s — verdicts sharing this rank group carry "
            "disagreeing run ids (base %s, onset %s); using the narrowest "
            "harness range consistent with all of them",
            v.detector, v.sample, v.last_accepted_run_date, v.onset_run_date,
            sorted(base_ids), sorted(onset_ids),
        )
    base_id, onset_id = max(base_ids), min(onset_ids)
    if base_id >= onset_id:
        # The narrowing collapsed or inverted the range. Run ids are dates, so
        # a base run at or after the onset run describes no interval at all —
        # and a reversed compare URL is silently, confidently wrong, which is
        # the one output this whole feature exists to avoid.
        _log.warning(
            "blame: %s/%s %s..%s — the harness range narrowed to %s..%s, which "
            "is not an interval; not attributing the harness for this window",
            v.detector, v.sample, v.last_accepted_run_date, v.onset_run_date,
            base_id, onset_id,
        )
        return None
    # Each end is read under the platform it ran on: a window opened on a
    # replaced platform keeps its base run's provenance there.
    base_end = next(m for m in verdicts if m.last_accepted_run_id == base_id)
    onset_end = next(m for m in verdicts if m.onset_run_id == onset_id)
    base = k4bench_commit_for_run(
        base_end.base_detector, base_end.base_platform,
        base_end.last_accepted_run_date, v.sample, base_id,
    )
    onset = k4bench_commit_for_run(
        onset_end.onset_run_detector, onset_end.onset_run_platform,
        onset_end.onset_run_date, v.sample, onset_id,
    )
    if base is None or onset is None:
        return None
    base_sha, base_url = base
    onset_sha, onset_url = onset
    if not base_sha or not onset_sha or base_sha == onset_sha:
        return None
    # A run with no recorded workflow URL *means* the default repository (see
    # :data:`_K4BENCH_REPO_URL`), so each end is resolved to a concrete
    # repository before they are compared. Comparing the raw parses instead
    # would skip the check whenever one end lacked a URL and then resolve both
    # commits in the other end's repository — a fork's onset paired with an
    # upstream base would silently become a fork-only range.
    base_repo = _repo_url_from_run_url(base_url) or _K4BENCH_REPO_URL
    onset_repo = _repo_url_from_run_url(onset_url) or _K4BENCH_REPO_URL
    if base_repo != onset_repo:
        _log.warning(
            "blame: %s/%s %s..%s — the window's runs were produced by "
            "different harness repositories (%s vs %s); a compare range "
            "across them would be meaningless, so the harness is not "
            "attributed for this window",
            v.detector, v.sample, v.last_accepted_run_date, v.onset_run_date,
            base_repo, onset_repo,
        )
        return None
    return PackageChange(
        name=HARNESS_PACKAGE,
        base_commit=base_sha,
        head_commit=onset_sha,
        version="",  # the harness has no stack version string
        repo_url=onset_repo,  # == base_repo, checked above
    )


def _harness_candidate_signal(files: tuple[str, ...]) -> bool:
    """Whether a harness pull request could move a measurement at all.

    Most k4Bench pull requests in any window are dashboard or documentation
    work. A candidate is dropped only when **every** changed file is provably
    irrelevant (:data:`_HARNESS_LOW_SIGNAL_PREFIXES`, plus the shared
    documentation-basename judgement of :func:`low_signal_path`); one file
    outside those areas keeps it, and a candidate with **no** file list is
    kept too — an unreadable change cannot be proven harmless."""
    if not files:
        return True
    return not all(
        low_signal_path(f) or f.startswith(_HARNESS_LOW_SIGNAL_PREFIXES)
        for f in files
    )


def _run_snapshots(v: MetricVerdict, lookup) -> tuple:
    """The recorded stacks of *v*'s base run and onset run."""
    return (
        lookup(v.base_detector, v.base_platform, v.last_accepted_run_date,
               v.sample, v.last_accepted_run_id),
        lookup(v.onset_run_detector, v.onset_run_platform, v.onset_run_date,
               v.sample, v.onset_run_id),
    )


def _external_facts(verdicts, lookup) -> tuple[ExternalChange, ...]:
    if lookup is None:
        return ()
    facts = set()
    for v in verdicts:
        try:
            base, head = _run_snapshots(v, lookup)
            facts.update(external_changes(base, head,
                         scope=f"{v.detector} · {v.sample} · {v.platform}"))
        except Exception:
            _log.exception("blame: could not read loaded environment facts")
    return tuple(sorted(facts))


def _loaded_packages(verdicts, lookup) -> frozenset[str] | None:
    """Keys of every package the base or onset run of any of *verdicts*
    mapped or could have run unmapped, or ``None`` when any of those runs did
    not record both sets.

    An empty loaded set is a failed mapping, since every run maps stack
    libraries, so it rules nothing out either."""
    if lookup is None:
        return None
    loaded = set()
    for v in verdicts:
        try:
            snapshots = _run_snapshots(v, lookup)
        except Exception:
            _log.exception("blame: could not read loaded packages")
            return None
        for snapshot in snapshots:
            if snapshot is None or not snapshot.loaded or snapshot.unmappable is None:
                return None
            loaded.update(package_key(name) for name in snapshot.loaded | snapshot.unmappable)
    return frozenset(loaded)
