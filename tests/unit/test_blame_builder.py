"""Unit tests for :mod:`k4bench.blame.builder` — turning a nightly report plus
injected provenance/GitHub access into a :class:`BlameReport`, offline."""

from __future__ import annotations

from k4bench.blame import builder as builder_mod
from k4bench.blame.builder import build_blame_report
from k4bench.blame.github import GitHubClient, RateLimitError, RepoResolution
from k4bench.blame.models import CandidatePR
from k4bench.blame.rank import Ranking
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)

_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_GH = "https://github.com/key4hep/k4geo.git"
_GL = "https://gitlab.cern.ch/acts/OpenDataDetector.git"


def _verdict(*, onset="2026-07-04", base="2026-07-03", metric="wall_time_s",
             severity=Severity.CONFIRMED, sub=None) -> MetricVerdict:
    return MetricVerdict(
        detector="ALLEGRO_o1_v03", platform=_PLAT, sample="single_e",
        label="baseline", metric_family="time", metric=metric, sub_detector=sub,
        run_id="2026-07-05", run_date="2026-07-05", value=120.0,
        baseline_median=100.0, baseline_mad=1.0, pct_change=0.2, z_score=6.0,
        severity=severity, direction=Direction.UP, reason="step",
        onset_run_id=onset, onset_run_date=onset,
        last_accepted_run_id=base, last_accepted_run_date=base,
    )


def _report(verdicts) -> NightlyReport:
    group = RunGroupReport(
        detector="ALLEGRO_o1_v03", platform=_PLAT, sample="single_e",
        k4h_release="key4hep-2026-07-05", run_date="2026-07-05", run_id="2026-07-05",
        verdicts=list(verdicts),
    )
    return NightlyReport(generated_at="2026-07-05T00:00:00", groups=[group])


def _pkgs(commit: str, url: str = _GH) -> dict:
    return {"commit": commit, "version": "develop", "repo_url": url}


def _provenance(mapping):
    """A ``(platform, release) -> packages`` lookup from an explicit dict."""
    return lambda platform, release: mapping.get((platform, release))


def _stub_resolve(monkeypatch, fn):
    monkeypatch.setattr(builder_mod, "resolve_repo_prs", fn)


def test_bounded_window_collects_candidates(monkeypatch):
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40), "dd4hep": _pkgs("d" * 40, _GH)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40), "dd4hep": _pkgs("d" * 40, _GH)},
    })

    def fake_resolve(client, slug, base, head):
        return RepoResolution(candidates=[
            CandidatePR(repo=slug, number=10, title="t", author="a", url="u",
                        files=("FCCee/ALLEGRO/x.xml",), additions=5, deletions=1),
        ])
    _stub_resolve(monkeypatch, fake_resolve)

    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    assert len(blame.entries) == 1
    entry = blame.entries[0]
    assert entry.onset_release == "2026-07-04" and entry.base_release == "2026-07-03"
    assert entry.n_unchanged == 1  # dd4hep didn't move
    assert [r.package for r in entry.repos] == ["k4geo"]  # only k4geo changed
    cand = entry.candidates[0]
    assert cand.number == 10
    # The builder collects candidates but does not rank them: score/description
    # are left for the ranking stage to fill.
    assert cand.score == 0.0 and cand.description == ""


def test_same_stack_window_is_skipped(monkeypatch):
    _stub_resolve(monkeypatch, lambda *a, **k: RepoResolution())
    provenance = _provenance({(_PLAT, "2026-07-04"): {"k4geo": _pkgs("a" * 40)}})
    blame = build_blame_report(
        _report([_verdict(onset="2026-07-04", base="2026-07-04")]),
        packages_for_release=provenance, github=GitHubClient(),
    )
    assert blame.entries == ()


def test_open_window_is_skipped(monkeypatch):
    _stub_resolve(monkeypatch, lambda *a, **k: RepoResolution())
    v = _verdict()
    open_v = MetricVerdict(**{**v.__dict__, "last_accepted_run_date": None,
                             "last_accepted_run_id": None})
    blame = build_blame_report(
        _report([open_v]), packages_for_release=_provenance({}), github=GitHubClient(),
    )
    assert blame.entries == ()


def test_missing_provenance_is_skipped(monkeypatch):
    _stub_resolve(monkeypatch, lambda *a, **k: RepoResolution())
    # Only the head release is known; the baseline aged off CVMFS → no diff.
    provenance = _provenance({(_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)}})
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    assert blame.entries == ()


def test_no_github_writes_diffs_without_candidates():
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=None,
    )
    entry = blame.entries[0]
    assert entry.repos[0].package == "k4geo"
    assert entry.repos[0].compare_url  # the diff is still recorded
    assert entry.candidates == []      # but no PRs without a client


def test_rate_limit_degrades_to_diffs_only(monkeypatch):
    def boom(*a, **k):
        raise RateLimitError("throttled")
    _stub_resolve(monkeypatch, boom)
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    # The regression is still recorded with its diff; it just has no candidates,
    # and the repo says so — "never asked" must not read as "empty range".
    assert len(blame.entries) == 1
    assert blame.entries[0].candidates == []
    assert blame.entries[0].repos[0].package == "k4geo"
    assert blame.entries[0].repos[0].commits_unavailable is True


def test_rate_limit_midway_marks_remaining_repos_and_suppresses_ranking(monkeypatch):
    calls = []

    def resolve_then_throttle(client, slug, base, head):
        calls.append(slug)
        if len(calls) == 1:
            return RepoResolution(candidates=[
                CandidatePR(repo=slug, number=10, title="t", author="a", url="u"),
            ])
        raise RateLimitError("throttled")
    _stub_resolve(monkeypatch, resolve_then_throttle)
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40),
                                "dd4hep": _pkgs("d" * 40, _GH)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40),
                                "dd4hep": _pkgs("e" * 40, _GH)},
    })
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(90.0, "confident")})
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance,
        github=GitHubClient(), ranker=ranker,
    )
    entry = blame.entries[0]
    flags = {r.package: r.commits_unavailable for r in entry.repos}
    assert sum(flags.values()) == 1        # the throttled repo is flagged …
    assert entry.discovery_incomplete      # … so the entry says it saw a partial set
    # … and the partial candidate set is never ranked: a "most likely" over
    # candidates that were never examined would overclaim.
    assert ranker.requests == []
    assert all(c.score == 0.0 and c.description == "" for c in entry.candidates)


def test_resolution_error_marks_repo_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection reset")
    _stub_resolve(monkeypatch, boom)
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    # A transient network/JSON failure is not an empty range either.
    assert blame.entries[0].repos[0].commits_unavailable is True


def test_non_github_repo_gets_diff_but_no_resolution(monkeypatch):
    calls = []
    def spy_resolve(client, slug, base, head):
        calls.append(slug)
        return RepoResolution()
    _stub_resolve(monkeypatch, spy_resolve)
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"opendatadetector": _pkgs("a" * 40, _GL)},
        (_PLAT, "2026-07-04"): {"opendatadetector": _pkgs("c" * 40, _GL)},
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    repo = blame.entries[0].repos[0]
    assert repo.package == "opendatadetector"
    assert repo.repo is None          # not a GitHub slug
    assert repo.compare_url           # GitLab compare link still resolves
    assert calls == []                # GitHub was never asked


def test_watch_verdict_gets_no_blame():
    # Only CONFIRMED regressions are attributed; a WATCH has no confirmed onset.
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    blame = build_blame_report(
        _report([_verdict(severity=Severity.WATCH)]),
        packages_for_release=provenance, github=None,
    )
    assert blame.entries == ()


# ── Ranking stage ─────────────────────────────────────────────────────────────

class _FakeRanker:
    """Records the requests it sees and returns a scripted mapping (or raises)."""

    def __init__(self, mapping=None, exc=None):
        self.mapping = mapping or {}
        self.exc = exc
        self.requests = []

    def rank(self, request):
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        return self.mapping


_MOVED = _provenance({
    (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
    (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
})


def _two_candidates(monkeypatch):
    def fake_resolve(client, slug, base, head):
        return RepoResolution(
            candidates=[
                CandidatePR(repo=slug, number=10, title="t10", author="a", url="u10"),
                CandidatePR(repo=slug, number=11, title="t11", author="a", url="u11"),
            ],
            patches={10: "diff for 10", 11: "diff for 11"},
        )
    _stub_resolve(monkeypatch, fake_resolve)


def test_ranker_scores_land_on_the_right_candidate(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(72.0, "raises the step count")})
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    cands = {c.number: c for c in blame.entries[0].candidates}
    assert cands[10].score == 72.0 and cands[10].description == "raises the step count"
    # A candidate the ranker didn't score keeps its unranked defaults.
    assert cands[11].score == 0.0 and cands[11].description == ""
    # The request carried every candidate, each with its transient patch.
    req = ranker.requests[0]
    patch_by_number = {c.number: c.patch for c in req.candidates}
    assert patch_by_number == {10: "diff for 10", 11: "diff for 11"}


def test_ranker_unknown_keys_are_dropped(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({
        ("key4hep/k4geo", 10): Ranking(50.0, "real"),
        ("key4hep/ghost", 999): Ranking(99.0, "hallucinated"),  # not in the input
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    numbers = {c.number for c in blame.entries[0].candidates}
    assert numbers == {10, 11}  # the ghost PR never materializes
    assert {c.repo for c in blame.entries[0].candidates} == {"key4hep/k4geo"}


def test_ranker_exception_degrades_to_unranked_without_aborting(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker(exc=RuntimeError("model exploded"))
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    # The report is intact — the regression, its diff and its candidates survive.
    assert len(blame.entries) == 1
    assert blame.entries[0].repos[0].package == "k4geo"
    assert all(c.score == 0.0 for c in blame.entries[0].candidates)  # just unranked


def test_ranker_called_once_per_window(monkeypatch):
    # Two confirmed metrics that stepped across the same release boundary share
    # one diff and one candidate set → a single inference, applied to both.
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    report = _report([_verdict(metric="wall_time_s"), _verdict(metric="peak_rss_mb")])
    blame = build_blame_report(
        report, packages_for_release=_MOVED, github=GitHubClient(), ranker=ranker,
    )
    assert len(ranker.requests) == 1  # one call for the shared window
    assert len(blame.entries) == 2
    for entry in blame.entries:
        assert next(c for c in entry.candidates if c.number == 10).score == 60.0


def test_rank_request_carries_every_metric_sharing_the_window(monkeypatch):
    # The model's judgement must be informed by every metric that stepped, not
    # just whichever verdict happened to reach the ranker first.
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    report = _report([_verdict(metric="wall_time_s"), _verdict(metric="peak_rss_mb")])
    build_blame_report(
        report, packages_for_release=_MOVED, github=GitHubClient(), ranker=ranker,
    )
    assert [m.metric for m in ranker.requests[0].metrics] == ["wall_time_s", "peak_rss_mb"]
