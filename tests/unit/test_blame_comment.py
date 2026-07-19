"""Unit tests for :mod:`k4bench.blame.comment` — who gets commented on, and
what the comment says. Everything here is offline: the module is pure by design
so the "do we write into someone else's repository?" decision is testable
without a token."""

from __future__ import annotations

from dataclasses import replace

import pytest

from k4bench.blame.comment import (
    CommentConfigError,
    CommentPolicy,
    marker_for,
    select,
)
from k4bench.blame.models import BlameEntry, BlameReport, CandidatePR, RepoBlame
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)

_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_DASH = "https://k4bench-dashboard.app.cern.ch"


def _policy(**kw) -> CommentPolicy:
    return CommentPolicy.from_config({"repos": ["key4hep/k4geo"], **kw})


def _verdict(*, metric="wall_time_s", label="baseline", onset="2026-07-04",
             base="2026-07-03", pct=0.2, detector="ALLEGRO_o1_v03",
             sample="single_e-_10GeV", sub=None) -> MetricVerdict:
    return MetricVerdict(
        detector=detector, platform=_PLAT, sample=sample,
        label=label, metric_family="time", metric=metric, sub_detector=sub,
        run_id="2026-07-05", run_date="2026-07-04", value=120.0,
        baseline_median=100.0, baseline_mad=1.0, pct_change=pct, z_score=6.0,
        severity=Severity.CONFIRMED, direction=Direction.UP, reason="step",
        onset_run_id=onset, onset_run_date=onset,
        last_accepted_run_id=base, last_accepted_run_date=base,
        first_confirmed_run_id="2026-07-05",
    )


def _report(*verdicts: MetricVerdict) -> NightlyReport:
    groups: dict[tuple, RunGroupReport] = {}
    for v in verdicts:
        key = (v.detector, v.platform, v.sample)
        group = groups.get(key)
        if group is None:
            group = groups[key] = RunGroupReport(
                detector=v.detector, platform=v.platform, sample=v.sample,
                k4h_release="key4hep-2026-07-04", run_date="2026-07-05",
                run_id="2026-07-05", verdicts=[],
            )
        group.verdicts.append(v)
    return NightlyReport(generated_at="2026-07-05T00:00:00", groups=list(groups.values()))


def _candidate(number=1234, repo="key4hep/k4geo", score=91.0, merged="2026-07-04T09:00:00Z",
               title="Add a per-step material lookup") -> CandidatePR:
    return CandidatePR(
        repo=repo, number=number, title=title, author="alice",
        url=f"https://github.com/{repo}/pull/{number}", merged_at=merged,
        files=("src/a.cpp",), additions=40, deletions=2,
        score=score, description="Adds a lookup on the hot path of every step.",
    )


def _blame(verdicts, candidates, *, truncated=False, unavailable=False) -> BlameReport:
    """A sidecar attributing every verdict in *verdicts* to *candidates*."""
    entries = [
        BlameEntry(
            detector=v.detector, platform=v.platform, sample=v.sample,
            label=v.label, metric=v.metric, sub_detector=v.sub_detector,
            base_release=v.last_accepted_run_date, onset_release=v.onset_run_date,
            repos=(
                RepoBlame(
                    package="k4geo", repo="key4hep/k4geo",
                    base_commit="a" * 40, head_commit="c" * 40,
                    compare_url="https://github.com/key4hep/k4geo/compare/a...c",
                    status="CHANGED", candidates=tuple(candidates),
                    commits_unavailable=unavailable, truncated=truncated,
                ),
            ),
        )
        for v in verdicts
    ]
    return BlameReport(
        generated_at="2026-07-05T01:00:00", report_night="2026-07-05",
        entries=tuple(entries),
    )


def _select(report, blame, policy=None):
    return select(
        report, blame, policy or _policy(),
        dashboard_url=_DASH, actions_url="https://github.com/key4hep/k4Bench/actions/runs/1",
    )


# ── The policy ────────────────────────────────────────────────────────────────

def test_policy_defaults_to_inert():
    # The shipped config: no repository enabled, so nothing is ever written.
    policy = CommentPolicy.from_config({"min_score": 80, "max_comments": 10, "repos": []})
    assert policy.enabled is False
    assert policy.min_score == 80.0


@pytest.mark.parametrize("bad", [
    {"repos": ["k4geo"]},                     # not owner/repo
    {"repos": "key4hep/k4geo"},               # not a list
    {"min_score": "eighty"},                  # not a number
    {"min_score": 140},                       # out of range
    {"max_comments": -1},                     # negative
    {"treshold": 80},                         # typo'd key, silently narrowing
])
def test_policy_rejects_malformed_config(bad):
    # A config that decides where the bot writes must fail loudly, never default.
    with pytest.raises(CommentConfigError):
        CommentPolicy.from_config(bad)


def test_policy_matches_repo_case_insensitively():
    policy = CommentPolicy.from_config({"repos": ["Key4hep/K4geo"]})
    assert policy.allows(_candidate(repo="key4hep/k4geo")) is True


# ── Selection gates ───────────────────────────────────────────────────────────

def test_confident_candidate_in_an_enabled_repo_is_selected():
    v = _verdict()
    comments = _select(_report(v), _blame([v], [_candidate()]))
    assert [(c.repo, c.number) for c in comments] == [("key4hep/k4geo", 1234)]


def test_below_threshold_candidate_is_not_selected():
    v = _verdict()
    assert _select(_report(v), _blame([v], [_candidate(score=79.0)])) == []


def test_repo_outside_the_allowlist_is_not_selected():
    v = _verdict()
    other = _candidate(repo="key4hep/DD4hep")
    assert _select(_report(v), _blame([v], [other])) == []


def test_unmerged_candidate_is_not_selected():
    # An open PR cannot have shipped in the release the step entered with.
    v = _verdict()
    assert _select(_report(v), _blame([v], [_candidate(merged=None)])) == []


@pytest.mark.parametrize("flags", [{"truncated": True}, {"unavailable": True}])
def test_incomplete_discovery_is_never_commented_on(flags):
    # The ranker refuses to name a culprit out of a knowingly partial candidate
    # set; posting one into someone's PR would be the same overclaim, louder.
    v = _verdict()
    assert _select(_report(v), _blame([v], [_candidate()], **flags)) == []


def test_watch_verdicts_are_not_commented_on():
    # Only confirmed regressions reach report.regressions, so a sidecar entry
    # for anything else has nothing to attach to.
    v = _verdict()
    report = NightlyReport(
        generated_at="2026-07-05T00:00:00",
        groups=[RunGroupReport(
            detector=v.detector, platform=v.platform, sample=v.sample,
            k4h_release="key4hep-2026-07-04", run_date="2026-07-05", run_id="2026-07-05",
            verdicts=[replace(v, severity=Severity.WATCH)],
        )],
    )
    assert _select(report, _blame([v], [_candidate()])) == []


def test_metrics_sharing_a_window_collapse_into_one_comment():
    a, b = _verdict(metric="wall_time_s"), _verdict(metric="mean_time_s", pct=0.14)
    comments = _select(_report(a, b), _blame([a, b], [_candidate()]))
    assert len(comments) == 1
    body = comments[0].body
    assert "`wall_time_s`" in body and "`mean_time_s`" in body


def test_a_second_window_gets_its_own_comment():
    # Two genuinely different change windows are two claims about the same PR,
    # and must not overwrite each other.
    old = _verdict(metric="peak_rss_mb", onset="2026-06-20", base="2026-06-19")
    new = _verdict(metric="wall_time_s")
    comments = _select(_report(old, new), _blame([old, new], [_candidate()]))
    assert len({c.marker for c in comments}) == 2


def test_max_comments_caps_the_night():
    verdicts = [_verdict(metric=f"m{i}", sample=f"s{i}") for i in range(4)]
    candidates = [_candidate(number=100 + i) for i in range(4)]
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=tuple(
            _blame([v], [c]).entries[0] for v, c in zip(verdicts, candidates, strict=True)
        ),
    )
    comments = _select(_report(*verdicts), blame, _policy(max_comments=2))
    assert len(comments) == 2


# ── The rendered body ─────────────────────────────────────────────────────────

def test_body_carries_the_marker_for_its_window():
    v = _verdict()
    comment = _select(_report(v), _blame([v], [_candidate()]))[0]
    assert comment.marker == marker_for("2026-07-03", "2026-07-04")
    assert comment.body.startswith(comment.marker)


def test_body_states_likelihood_reason_and_disclosure():
    v = _verdict()
    body = _select(_report(v), _blame([v], [_candidate()]))[0].body
    assert "91%" in body
    assert "hot path of every step" in body
    assert "AI-generated PR ranking" in body  # never presented as proof


def test_body_lists_the_other_candidates_with_their_likelihoods():
    v = _verdict()
    others = [_candidate(), _candidate(number=1180, score=22.0, title="Unrelated cleanup")]
    body = _select(_report(v), _blame([v], others))[0].body
    assert "key4hep/k4geo#1180" in body and "22%" in body


def test_body_says_so_when_nothing_else_was_in_the_frame():
    v = _verdict()
    body = _select(_report(v), _blame([v], [_candidate()]))[0].body
    assert "only pull request found" in body


def test_body_links_the_window_in_the_dashboard():
    v = _verdict()
    body = _select(_report(v), _blame([v], [_candidate()]))[0].body
    assert "window=2026-07-03..2026-07-04" in body        # the scoped Regressions view
    assert "&to=2026-07-04" in body                       # the package diff
    assert "actions/runs/1" in body                       # the run that produced it
    # ?stack= is the dashboard's release *directory*, not the bare release date
    # a verdict carries — a bare date selects nothing and silently falls back.
    assert "stack=key4hep-2026-07-04" in body


def test_body_renders_without_a_dashboard_url():
    # Offline/local rendering must still produce a usable comment.
    v = _verdict()
    comments = select(_report(v), _blame([v], [_candidate()]), _policy())
    assert "Where to look" not in comments[0].body
    assert "91%" in comments[0].body


def test_open_ended_window_is_described_as_such():
    v = _verdict(base=None)
    body = _select(_report(v), _blame([v], [_candidate()]))[0].body
    assert "no earlier settled measurement" in body


def test_table_cells_survive_hostile_text():
    # A pipe in a model-written reason or a PR title would end the column.
    v = _verdict()
    hostile = CandidatePR(
        repo="key4hep/k4geo", number=1, title="a | b\nsecond line",
        author="alice", url="https://github.com/key4hep/k4geo/pull/1",
        merged_at="2026-07-04T00:00:00Z", score=90.0, description="ranked",
    )
    headline = replace(_candidate(number=2, score=95.0), description="Line one\nline two")
    body = _select(_report(v), _blame([v], [headline, hostile]))[0].body
    row = next(line for line in body.splitlines() if "key4hep/k4geo#1]" in line)
    # Two columns: the title's pipe is escaped, so it opens no third one.
    assert row.replace("\\|", "").count("|") == 3
    assert "a \\| b second line" in row  # pipe escaped, newline collapsed
    assert "Line one line two" in body  # the same for the quoted reason


def test_body_is_stable_across_identical_nights():
    # The upsert only edits when the body changes, so an unchanged night must
    # render byte-identically — no set ordering leaking into the output.
    a, b = _verdict(metric="wall_time_s"), _verdict(metric="mean_time_s", pct=0.14)
    first = _select(_report(a, b), _blame([a, b], [_candidate()]))[0].body
    second = _select(_report(b, a), _blame([b, a], [_candidate()]))[0].body
    assert first == second
