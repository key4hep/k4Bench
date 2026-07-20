"""Unit tests for :mod:`k4bench.blame.comment` — who gets commented on, and
what the comment says.

Everything here is offline. :func:`~k4bench.blame.comment.select` is pure by
design, so the "do we write into someone else's repository?" decision is
testable without a token; :func:`~k4bench.blame.comment.build_comments` takes
its model and its diff source as arguments, so the cross-configuration review is
driven here by a recording fake rather than an endpoint."""

from __future__ import annotations

from dataclasses import replace

import pytest

from k4bench.blame.attribute import (
    MAX_COMPETITORS,
    Attribution,
    build_user_prompt,
)
from k4bench.blame.comment import (
    CommentConfigError,
    CommentPolicy,
    CommentStormError,
    build_comments,
    facts_digest_of,
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
from k4bench.regression.render import regression_href

_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_DASH = "https://k4bench-dashboard.app.cern.ch"
#: What the renderer breaks a GitHub-active sequence with — invisible to a
#: reader, inert to GitHub's reference and mention parsers.
_ZWSP = "​"


def _policy(**kw) -> CommentPolicy:
    return CommentPolicy.from_config({"repos": ["key4hep/k4geo"], **kw})


def _verdict(*, metric="wall_time_s", label="baseline", onset="2026-07-04",
             base="2026-07-03", pct=0.2, detector="ALLEGRO_o1_v03",
             sample="single_e-_10GeV", sub=None, platform=_PLAT,
             severity=Severity.CONFIRMED) -> MetricVerdict:
    return MetricVerdict(
        detector=detector, platform=platform, sample=sample,
        label=label, metric_family="time", metric=metric, sub_detector=sub,
        run_id="2026-07-05", run_date=onset, value=120.0,
        baseline_median=100.0, baseline_mad=1.0, pct_change=pct, z_score=6.0,
        severity=severity, direction=Direction.UP, reason="step",
        onset_run_id=onset, onset_run_date=onset,
        last_accepted_run_id=base, last_accepted_run_date=base,
        first_confirmed_run_id="2026-07-05",
    )


def _report(*verdicts: MetricVerdict, night="2026-07-05", **group_kw) -> NightlyReport:
    # Reliability is a tri-state and only ``True`` is a trustworthy run, so the
    # groups here are reliable unless a test says otherwise — a fixture left at
    # the ``None`` default would silently be "no evidence", never a control.
    group_kw.setdefault("reliable", True)
    groups: dict[tuple, RunGroupReport] = {}
    for v in verdicts:
        key = (v.detector, v.platform, v.sample)
        group = groups.get(key)
        if group is None:
            group = groups[key] = RunGroupReport(
                detector=v.detector, platform=v.platform, sample=v.sample,
                k4h_release="key4hep-2026-07-04", run_date=night,
                run_id=night, verdicts=[], **group_kw,
            )
        group.verdicts.append(v)
    return NightlyReport(generated_at=f"{night}T00:00:00", groups=list(groups.values()))


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
            n_unchanged=18,
        )
        for v in verdicts
    ]
    return BlameReport(
        generated_at="2026-07-05T01:00:00", report_night="2026-07-05",
        entries=tuple(entries),
    )


def _blame_of(*pairs) -> BlameReport:
    """A sidecar built from explicit ``(verdict, candidates)`` pairs — the shape
    a window needs when its scopes carry *different* candidate scores."""
    return BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=tuple(_blame([v], cands).entries[0] for v, cands in pairs),
    )


class _FakeAttributor:
    """A scripted cross-configuration review that records what it was asked.

    ``scores`` maps a regression's ``fact_id`` to a likelihood; anything not
    named is simply not answered, which is a real case the renderer must handle
    (the row keeps its per-configuration score)."""

    def __init__(self, scores=None, *, summary="ALLEGRO moved and IDEA did not.",
                 declines=False, raises=None):
        self.scores = scores or {}
        self.summary = summary
        self.declines = declines
        self.raises = raises
        self.requests: list = []

    def attribute(self, request):
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if self.declines:
            return None
        return Attribution(summary=self.summary, likelihoods=dict(self.scores))


def _plans(report, blame, policy=None):
    return select(report, blame, policy or _policy())


def _comments(report, blame, policy=None, *, attributor=None, patch_for=None,
              dashboard_url=_DASH):
    policy = policy or _policy()
    return build_comments(
        _plans(report, blame, policy),
        attributor=attributor, patch_for=patch_for,
        dashboard_url=dashboard_url, min_score=policy.min_score,
    )


def _row(body: str, needle: str) -> str:
    return next(line for line in body.splitlines() if needle in line)


def _table_rows(body: str) -> list[str]:
    # A row opens with its metric cell, linked (``| [`wall_time_s`][r1] |``)
    # when the dashboard is configured and bare when it is not.
    return [
        line for line in body.splitlines()
        if line.startswith("| `") or line.startswith("| [`")
    ]


# ── The policy ────────────────────────────────────────────────────────────────

def test_policy_defaults_to_inert():
    # An empty allowlist: no repository enabled, so nothing is ever written.
    policy = CommentPolicy.from_config({"min_score": 80, "max_comments": 10, "repos": []})
    assert policy.enabled is False
    assert policy.min_score == 80.0


@pytest.mark.parametrize("bad", [
    {"repos": ["k4geo"]},                     # not owner/repo
    {"repos": "key4hep/k4geo"},               # not a list
    {"min_score": "eighty"},                  # not a number
    {"min_score": 140},                       # out of range
    {"max_comments": -1},                     # negative
    {"max_comments": 0},                      # zero is "disable", not a cap
    {"max_comments": 2.5},                    # a fractional cap is a typo
    {"max_comments": True},                   # a bool is not a count
    {"treshold": 80},                         # typo'd key, silently narrowing
    False,                                     # a falsey document is not "no config"
    {"repos": False},                         # a scalar is not an allowlist
    {"repos": ["owner/ "]},                   # slug is "owner/" once stripped
])
def test_policy_rejects_malformed_config(bad):
    # A config that decides where the bot writes must fail loudly, never default.
    with pytest.raises(CommentConfigError):
        CommentPolicy.from_config(bad)


def test_policy_matches_repo_case_insensitively():
    policy = CommentPolicy.from_config({"repos": ["Key4hep/K4geo"]})
    assert policy.allows(_candidate(repo="key4hep/k4geo")) is True


@pytest.mark.parametrize("absent", [None, {}, {"repos": None}])
def test_absent_or_empty_config_is_inert_not_an_error(absent):
    # Only a *present but malformed* document raises; a missing one, an empty
    # mapping, or an explicitly empty `repos:` all mean "the bot is off".
    assert CommentPolicy.from_config(absent).enabled is False


# ── Selection gates ───────────────────────────────────────────────────────────

def test_confident_candidate_in_an_enabled_repo_is_selected():
    v = _verdict()
    plans = _plans(_report(v), _blame([v], [_candidate()]))
    assert [(p.repo, p.number) for p in plans] == [("key4hep/k4geo", 1234)]


def test_below_threshold_candidate_is_not_selected():
    v = _verdict()
    assert _plans(_report(v), _blame([v], [_candidate(score=79.0)])) == []


def test_repo_outside_the_allowlist_is_not_selected():
    v = _verdict()
    other = _candidate(repo="key4hep/DD4hep")
    assert _plans(_report(v), _blame([v], [other])) == []


def test_unmerged_candidate_is_not_selected():
    # An open PR cannot have shipped in the release the step entered with.
    v = _verdict()
    assert _plans(_report(v), _blame([v], [_candidate(merged=None)])) == []


@pytest.mark.parametrize("flags", [{"truncated": True}, {"unavailable": True}])
def test_incomplete_discovery_is_never_commented_on(flags):
    # The ranker refuses to name a culprit out of a knowingly partial candidate
    # set; posting one into someone's PR would be the same overclaim, louder.
    v = _verdict()
    assert _plans(_report(v), _blame([v], [_candidate()], **flags)) == []


def test_watch_verdicts_are_not_commented_on():
    # Only confirmed regressions reach report.regressions, so a sidecar entry
    # for anything else has nothing to attach to.
    v = _verdict()
    report = _report(replace(v, severity=Severity.WATCH))
    assert _plans(report, _blame([v], [_candidate()])) == []


def test_metrics_sharing_a_window_collapse_into_one_comment():
    a, b = _verdict(metric="wall_time_s"), _verdict(metric="mean_time_s", pct=0.14)
    comments = _comments(_report(a, b), _blame([a, b], [_candidate()]))
    assert len(comments) == 1
    body = comments[0].body
    assert "`wall_time_s`" in body and "`mean_time_s`" in body


def test_a_low_scoring_configuration_of_a_selected_pr_is_still_collected():
    # The PR is selected because it scored 92 on ALLEGRO; the ranker gave it 30
    # on the IDEA regression of the same window. That IDEA row is exactly the
    # cross-configuration evidence the review exists to weigh — it bounds what
    # the diff reached — and it is not recoverable as negative evidence either,
    # since IDEA *did* confirm a step here. So it is collected, not filtered.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    blame = _blame_of((allegro, [_candidate(score=92.0)]),
                      (idea, [_candidate(score=30.0)]))
    attributor = _FakeAttributor({"r1": 92.0, "r2": 30.0})
    comments = _comments(_report(allegro, idea), blame, attributor=attributor)
    assert len(comments) == 1
    scored = attributor.requests[0].regressions
    assert {(f.detector, f.scope_score) for f in scored} == {
        ("ALLEGRO_o1_v03", 92.0), ("IDEA_o1_v03", 30.0),
    }
    assert "30%" in comments[0].body


def test_a_pr_below_the_threshold_everywhere_is_still_not_selected():
    # Collecting low-scoring rows must not become a way in: the plan is kept
    # only when some scoring of it crosses min_score.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    blame = _blame_of((allegro, [_candidate(score=42.0)]),
                      (idea, [_candidate(score=30.0)]))
    assert _plans(_report(allegro, idea), blame) == []


def test_a_second_window_gets_its_own_comment():
    # Two genuinely different change windows are two claims about the same PR,
    # and must not overwrite each other.
    old = _verdict(metric="peak_rss_mb", onset="2026-06-20", base="2026-06-19")
    new = _verdict(metric="wall_time_s")
    comments = _comments(_report(old, new), _blame([old, new], [_candidate()]))
    assert len({c.marker for c in comments}) == 2


def test_over_the_cap_raises_a_storm_error():
    # A night louder than max_comments is a bug, not a night: rather than post
    # the top N accusations into repos we don't own, the whole night is dropped —
    # and raising (not returning []) lets the CLI tell it apart from a quiet night.
    verdicts = [_verdict(metric=f"m{i}", sample=f"s{i}") for i in range(4)]
    candidates = [_candidate(number=100 + i) for i in range(4)]
    blame = _blame_of(*zip(verdicts, ([c] for c in candidates), strict=True))
    with pytest.raises(CommentStormError) as exc:
        _plans(_report(*verdicts), blame, _policy(max_comments=2))
    assert exc.value.count == 4 and exc.value.cap == 2


def test_at_the_cap_still_posts():
    # The cap is a ceiling, not a trigger: exactly max_comments is fine.
    verdicts = [_verdict(metric=f"m{i}", sample=f"s{i}") for i in range(2)]
    candidates = [_candidate(number=100 + i) for i in range(2)]
    blame = _blame_of(*zip(verdicts, ([c] for c in candidates), strict=True))
    assert len(_plans(_report(*verdicts), blame, _policy(max_comments=2))) == 2


# ── What the review is shown ──────────────────────────────────────────────────

def test_the_review_is_asked_about_every_configuration_at_once():
    # The whole point of the second pass: one request carrying every scope of
    # the window, not one request per scope.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb", pct=0.11)
    attributor = _FakeAttributor({"r1": 90.0, "r2": 20.0})
    _comments(_report(allegro, idea), _blame([allegro, idea], [_candidate()]),
              attributor=attributor)
    assert len(attributor.requests) == 1
    request = attributor.requests[0]
    assert {f.detector for f in request.regressions} == {
        "ALLEGRO_o1_v03", "IDEA_o1_v03",
    }
    assert request.repo == "key4hep/k4geo" and request.number == 1234


def test_the_review_is_shown_what_measured_the_window_and_stayed_clean():
    # "ALLEGRO moved and IDEA did not" is the evidence this stage exists for.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea_clean = _verdict(detector="IDEA_o1_v03", severity=Severity.OK)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_clean), _blame([allegro], [_candidate()]),
              attributor=attributor)
    outcomes = attributor.requests[0].outcomes
    assert [(o.detector, o.status) for o in outcomes] == [("IDEA_o1_v03", "clean")]


def test_a_configuration_that_moved_without_confirming_is_reported_as_watching():
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea_watch = _verdict(detector="IDEA_o1_v03", severity=Severity.WATCH,
                          metric="peak_rss_mb")
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_watch), _blame([allegro], [_candidate()]),
              attributor=attributor)
    outcome = attributor.requests[0].outcomes[0]
    assert (outcome.status, outcome.watched) == ("watch", ("peak_rss_mb",))


@pytest.mark.parametrize("group_kw", [
    {"reliable": False},                       # the host was not trustworthy
    {"reliable": None},                        # no reliability evidence at all
    {"job_failures": ["no run uploaded"]},     # the run did not really happen
])
def test_a_run_that_cannot_be_trusted_is_not_evidence_of_absence(group_kw):
    # Silence from a broken run must never be shown to the model as a clean
    # measurement — that is the difference between "IDEA did not move" and
    # "IDEA was not measured".
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03", severity=Severity.OK)
    report = _report(allegro)
    report.groups.extend(_report(idea, **group_kw).groups)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(report, _blame([allegro], [_candidate()]), attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_detector_removal_run_is_a_control_for_its_own_baseline():
    # The sharpest control the suite produces is *inside* a run group: baseline
    # stepped, without_HCAL did not, same detector, sample, platform and night —
    # which places the cost inside the HCAL. Judging the negative evidence per
    # run group would delete exactly this comparison, since the regression it is
    # a control for lives in the same group.
    baseline = _verdict(label="baseline")
    without_hcal = _verdict(label="without_HCAL", severity=Severity.OK)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(baseline, without_hcal), _blame([baseline], [_candidate()]),
              attributor=attributor)
    outcomes = attributor.requests[0].outcomes
    assert [(o.detector, o.label, o.status) for o in outcomes] == [
        ("ALLEGRO_o1_v03", "without_HCAL", "clean"),
    ]


def test_a_configuration_that_partly_failed_is_not_a_clean_control():
    # A metric that failed outright is a configuration that did not measure,
    # not one that measured and stayed flat.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea_ok = _verdict(detector="IDEA_o1_v03", metric="wall_time_s",
                       severity=Severity.OK)
    idea_failed = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb",
                           severity=Severity.FAILURE)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_ok, idea_failed),
              _blame([allegro], [_candidate()]), attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_configuration_that_measured_another_release_is_not_evidence():
    # The control has to be a like-for-like measurement: a group that ran a
    # different Key4hep release than the regressed rows says nothing about them.
    # Note this is the release the group *ran*, not the window's onset — a step
    # that entered on 2026-06-25 is still being re-measured weeks later, so
    # matching on the onset would find no control at all.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    other_stack = _report(_verdict(detector="IDEA_o1_v03", severity=Severity.OK))
    other_stack.groups[0].k4h_release = "key4hep-2026-06-01"
    report = _report(allegro)
    report.groups.extend(other_stack.groups)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(report, _blame([allegro], [_candidate()]), attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_control_is_found_even_though_the_onset_is_long_past():
    # The regression entered on 2026-06-25 and is still confirmed while the
    # nightlies measure 2026-06-27. The clean configurations measuring *that*
    # release are the evidence, and an earlier version of this rule found none.
    allegro = _verdict(detector="ALLEGRO_o1_v03", onset="2026-06-25", base="2026-06-24")
    idea = _verdict(detector="IDEA_o1_v03", severity=Severity.OK,
                    onset="2026-06-25", base="2026-06-24")
    report = _report(allegro, idea)
    for group in report.groups:
        group.k4h_release = "key4hep-2026-06-27"
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(report, _blame([allegro], [_candidate()]), attributor=attributor)
    assert [o.detector for o in attributor.requests[0].outcomes] == ["IDEA_o1_v03"]


def test_a_configuration_with_nothing_judged_is_not_a_clean_control():
    # UNKNOWN is "too little history to judge", not "flat". A configuration whose
    # every metric is still warming up measured nothing that can disagree with
    # the regressed rows, and showing it as one that did not move is the false
    # negative evidence that can talk the review out of a real attribution.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03", severity=Severity.UNKNOWN)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea), _blame([allegro], [_candidate()]),
              attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_partly_judged_configuration_is_offered_with_its_gap_stated():
    # Some coverage is still evidence — as long as the prompt says how much.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea_ok = _verdict(detector="IDEA_o1_v03", metric="wall_time_s",
                       severity=Severity.OK)
    idea_new = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb",
                        severity=Severity.UNKNOWN)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_ok, idea_new),
              _blame([allegro], [_candidate()]), attributor=attributor)
    outcome = attributor.requests[0].outcomes[0]
    assert (outcome.detector, outcome.status, outcome.unjudged) == (
        "IDEA_o1_v03", "clean", 1,
    )
    assert "too little history to judge" in build_user_prompt(
        attributor.requests[0]
    )


def test_a_step_at_the_same_onset_is_not_a_control_whatever_its_base():
    # The base is inferred per metric series — the last release *that* metric was
    # settled on — so two configurations that stepped on the same release can
    # report different bases. Requiring the whole window to match would read the
    # second one as a configuration that never moved.
    allegro = _verdict(detector="ALLEGRO_o1_v03", onset="2026-07-04", base="2026-07-03")
    idea_stepped = _verdict(detector="IDEA_o1_v03", onset="2026-07-04",
                            base="2026-06-28")
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_stepped), _blame([allegro], [_candidate()]),
              attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_step_from_a_different_window_is_still_a_control():
    # A configuration that stepped weeks earlier and has been settled since was
    # flat across *this* window, which is the only question being asked of it.
    allegro = _verdict(detector="ALLEGRO_o1_v03", onset="2026-07-04", base="2026-07-03")
    idea_old = _verdict(detector="IDEA_o1_v03", onset="2026-06-10", base="2026-06-09")
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_old), _blame([allegro], [_candidate()]),
              attributor=attributor)
    assert [o.detector for o in attributor.requests[0].outcomes] == ["IDEA_o1_v03"]


def test_only_the_competitors_the_prompt_can_carry_are_fetched():
    # A diff fetch is a GitHub round trip inside a shared timeout, and the prompt
    # keeps only the strongest MAX_COMPETITORS: fetching the rest buys nothing.
    v = _verdict()
    rivals = [
        _candidate(number=2000 + n, repo="key4hep/DD4hep", score=float(n))
        for n in range(MAX_COMPETITORS + 8)
    ]
    fetched: list[tuple[str, int]] = []

    def patch_for(repo, number):
        fetched.append((repo, number))
        return "diff"

    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), _blame([v], [_candidate(), *rivals]),
              attributor=attributor, patch_for=patch_for)
    # The subject plus the capped field, and the ones kept are the strongest.
    assert len(fetched) == MAX_COMPETITORS + 1
    assert (v.detector, 2000) not in fetched


def test_the_review_is_shown_the_diffs_the_release_and_the_competing_field():
    v = _verdict()
    rival = _candidate(number=1180, repo="key4hep/DD4hep", score=64.0, title="Field map")
    fetched: list[tuple[str, int]] = []

    def patch_for(repo, number):
        fetched.append((repo, number))
        return f"diff of {repo}#{number}"

    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), _blame([v], [_candidate(), rival]),
              attributor=attributor, patch_for=patch_for)
    request = attributor.requests[0]
    assert request.patch == "diff of key4hep/k4geo#1234"
    assert [(c.repo, c.number) for c in request.competitors] == [
        ("key4hep/DD4hep", 1180),
    ]
    assert request.competitors[0].patch == "diff of key4hep/DD4hep#1180"
    assert request.competitors[0].scope_score == 64.0
    assert [p.package for p in request.packages] == ["k4geo"]
    assert request.n_unchanged == 18
    assert fetched == [("key4hep/k4geo", 1234), ("key4hep/DD4hep", 1180)]


def test_the_first_passs_score_rides_along_as_the_priors():
    v = _verdict()
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), _blame([v], [_candidate(score=87.0)]), attributor=attributor)
    fact = attributor.requests[0].regressions[0]
    assert fact.scope_score == 87.0
    assert fact.scope_reason == "Adds a lookup on the hot path of every step."
    assert fact.direction == "UP"


def test_row_ids_are_assigned_by_identity_not_by_score():
    # A re-run must ask the model about "r2" and mean the same regression, so
    # the ids cannot depend on an ordering the model itself influences.
    a = _verdict(metric="a_metric", pct=0.05)
    b = _verdict(metric="b_metric", pct=0.40)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(a, b), _blame([a, b], [_candidate()]), attributor=attributor)
    facts = attributor.requests[0].regressions
    assert [(f.id, f.metric) for f in facts] == [("r1", "a_metric"), ("r2", "b_metric")]


# ── The single table ──────────────────────────────────────────────────────────

def test_the_table_is_ordered_by_the_reviews_likelihood():
    # The per-configuration ranker scored both scopes the same; the review has
    # seen the whole window and disagrees, and its order is what a reader sees.
    allegro = _verdict(detector="ALLEGRO_o1_v03", metric="wall_time_s")
    idea = _verdict(detector="IDEA_o1_v03", metric="wall_time_s")
    attributor = _FakeAttributor({"r1": 30.0, "r2": 95.0})
    body = _comments(_report(allegro, idea), _blame([allegro, idea], [_candidate()]),
                     attributor=attributor)[0].body
    rows = _table_rows(body)
    assert "IDEA_o1_v03" in rows[0] and "95%" in rows[0]
    assert "ALLEGRO_o1_v03" in rows[1] and "30%" in rows[1]


def test_a_row_the_review_skipped_keeps_its_per_configuration_score():
    # An unanswered row is not a zero.
    a = _verdict(metric="a_metric")
    b = _verdict(metric="b_metric")
    attributor = _FakeAttributor({"r1": 88.0})
    body = _comments(_report(a, b), _blame([a, b], [_candidate(score=91.0)]),
                     attributor=attributor)[0].body
    assert "88%" in _row(body, "a_metric")
    assert "91%" in _row(body, "b_metric")


def test_only_the_first_rows_are_visible_and_the_rest_fold_away():
    verdicts = [_verdict(metric=f"m{i}", pct=(20 - i) / 100) for i in range(8)]
    body = _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0].body
    head, _, tail = body.partition("<details>")
    assert len(_table_rows(head)) == 5
    assert "3 further regressions in this window" in tail
    assert len(_table_rows(tail)) == 3


def test_a_detector_sweeps_worth_of_rows_still_fits_in_a_github_comment():
    # A detector-removal sweep confirms one row per removed sub-detector: a real
    # night has carried 318. Pasting them all is both unreadable and, past
    # GitHub's 65,536-character limit, *rejected outright* — the comment would
    # simply fail to post. The folded rows are capped and the rest counted.
    verdicts = [
        _verdict(metric=f"m{i % 4}", label=f"without_Sub{i}", pct=(300 - i) / 1000)
        for i in range(318)
    ]
    comment = _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0]
    assert len(comment.body) < 65_536
    assert len(_table_rows(comment.body)) == 30      # 5 visible + 25 folded
    assert "313 further regressions in this window" in comment.body
    assert "…and 288 more regressions, in the dashboard._" in comment.body


def test_the_urls_live_in_reference_definitions_not_in_the_rows():
    # The dashboard URL is ~400 characters; inlining one per row is what blew
    # the size limit. A row carries a two-character label, and only the rows
    # that survive the table's caps get a definition at all.
    a = _verdict(metric="wall_time_s")
    b = _verdict(metric="peak_rss_mb", pct=0.1)
    body = _comments(_report(a, b), _blame([a, b], [_candidate()]))[0].body
    assert _row(body, "peak_rss_mb").count(_DASH) == 0
    assert body.count(f"[r1]: {_DASH}") == 1
    assert body.count(f"[r2]: {_DASH}") == 1
    assert body.count(_DASH) == 3        # two row links + the package diff

    many = [_verdict(metric=f"m{i}", pct=(300 - i) / 1000) for i in range(40)]
    body = _comments(_report(*many), _blame(many, [_candidate()]))[0].body
    assert body.count(f"]: {_DASH}") == 30      # 5 visible + 25 folded, no more


def test_the_platform_column_appears_only_when_platforms_differ():
    one = _verdict(detector="ALLEGRO_o1_v03")
    body = _comments(_report(one), _blame([one], [_candidate()]))[0].body
    assert "| Platform |" not in body

    other = _verdict(detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-dbg")
    body = _comments(_report(one, other), _blame([one, other], [_candidate()]))[0].body
    assert "| Platform |" in body
    assert "debug" in body and "optimized" in body


def _entry_with(verdict, packages, *, n_unchanged=18) -> BlameEntry:
    """One sidecar entry whose release diff is spelled out per package."""
    return replace(
        _blame([verdict], [_candidate()]).entries[0],
        repos=tuple(
            RepoBlame(
                package=package, repo=f"key4hep/{package}",
                base_commit="a" * 40, head_commit="c" * 40,
                compare_url=f"https://github.com/key4hep/{package}/compare/a...c",
                status="CHANGED", candidates=(_candidate(),),
            )
            for package in packages
        ),
        n_unchanged=n_unchanged,
    )


def test_a_comment_spanning_platforms_is_shown_every_platforms_package_diff():
    # A plan is keyed by pull request and window, never by platform, while the
    # release provenance a diff is read from *is* per platform. Taking whichever
    # entry was walked first would tell the review one platform's changed-package
    # set while showing it both platforms' regressions.
    dbg = "x86_64-almalinux9-gcc14.2.0-dbg"
    opt = _verdict(platform=_PLAT)
    debug = _verdict(platform=dbg)
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=(
            _entry_with(opt, ["k4geo"], n_unchanged=18),
            _entry_with(debug, ["k4geo", "DD4hep"], n_unchanged=17),
        ),
    )
    attributor = _FakeAttributor({"r1": 90.0, "r2": 90.0})
    _comments(_report(opt, debug), blame, attributor=attributor)
    request = attributor.requests[0]
    assert sorted(p.package for p in request.packages) == ["DD4hep", "k4geo"]
    # The one only the debug build recorded says so; the shared one does not.
    facts = {p.package: p.platforms for p in request.packages}
    assert facts["DD4hep"] == (dbg,) and facts["k4geo"] == ()
    # The unchanged claim is the one that holds on both platforms.
    assert request.n_unchanged == 17
    assert "recorded on x86_64-almalinux9-gcc14.2.0-dbg only" in build_user_prompt(request)


def test_each_row_links_to_its_own_regression_in_the_dashboard():
    # A reader's question is about *their* metric, so the row opens the exact
    # regression: the package diff for the window with that metric selected
    # under it, where its trend and onset are.
    allegro = _verdict(detector="ALLEGRO_o1_v03", metric="wall_time_s")
    idea = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb", sub="ECalBarrel")
    body = _comments(_report(allegro, idea), _blame([allegro, idea], [_candidate()]))[0].body
    for verdict in (allegro, idea):
        # The row carries the reference label; the definition carries the URL.
        label = _row(body, f"`{verdict.metric}`").split("][")[1].split("]")[0]
        definition = _row(body, f"[{label}]: ")
        assert "tab=Stack+Changes" in definition
        assert f"detector={verdict.detector}" in definition
        assert "from=2026-07-03" in definition and "to=2026-07-04" in definition
        # The reg_* params pin one verdict: the tab needs the onset to tell two
        # onsets of the same release apart, and the region for a region metric.
        assert f"reg_metric={verdict.metric}" in definition
        assert "reg_onset=2026-07-04" in definition
    assert "reg_region=ECalBarrel" in body


def test_a_regression_with_no_onset_identity_is_not_pinned():
    # Two onsets can measure the same release, so the dashboard needs the onset
    # to know which step is meant: without one there is no link that selects the
    # right regression, and the comment falls back to the window (see
    # :func:`~k4bench.blame.comment._row_links`) rather than pinning the wrong
    # one.
    v = replace(_verdict(), onset_run_id=None, onset_run_date=None)
    assert regression_href(
        _DASH, verdict=v, base_release="2026-07-03", onset_release="2026-07-04"
    ) is None


def test_the_window_wide_package_diff_is_the_only_link_left_over():
    v = _verdict()
    body = _comments(_report(v), _blame([v], [_candidate()]))[0].body
    assert body.count("Where to look") == 1
    assert "&to=2026-07-04" in body
    # Nothing that varies from night to night: no report-night query param and
    # no CI-run URL, either of which would edit a standing comment every night.
    assert "report=" not in body
    assert "actions/runs" not in body


def test_the_old_two_section_layout_is_gone():
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    body = _comments(_report(allegro, idea), _blame([allegro, idea], [_candidate()]))[0].body
    assert "Also affected in this window" not in body
    assert "What moved" not in body
    assert body.count("Regressions reviewed against this pull request") == 1


# ── The claim, and the withdrawal gate ────────────────────────────────────────

def test_the_review_supplies_the_narrative():
    v = _verdict()
    attributor = _FakeAttributor(
        {"r1": 92.0}, summary="Only ALLEGRO moved; IDEA ran the same sample clean.",
    )
    body = _comments(_report(v), _blame([v], [_candidate()]), attributor=attributor)[0].body
    assert "The AI reviewer's assessment" in body
    assert "IDEA ran the same sample clean" in body
    assert "92%" in body


def test_a_review_that_clears_every_row_withdraws_the_comment():
    # Selection happens on the first pass; the second may only narrow.
    v = _verdict()
    attributor = _FakeAttributor({"r1": 12.0})
    assert _comments(_report(v), _blame([v], [_candidate()]),
                     attributor=attributor) == []


def test_one_row_above_the_threshold_is_enough_to_keep_the_comment():
    a, b = _verdict(metric="a_metric"), _verdict(metric="b_metric")
    attributor = _FakeAttributor({"r1": 12.0, "r2": 81.0})
    comments = _comments(_report(a, b), _blame([a, b], [_candidate()]),
                         attributor=attributor)
    assert len(comments) == 1
    assert "12%" in _row(comments[0].body, "a_metric")


def test_a_row_the_review_left_alone_can_hold_the_comment_up():
    # A partial reply is an accepted outcome: the rows it omitted keep their
    # per-configuration score. The withdrawal gate therefore has to read what
    # the table will show — one low answer about one row must not acquit a PR
    # the review never disputed on the row that caused the comment.
    a, b = _verdict(metric="a_metric"), _verdict(metric="b_metric")
    blame = _blame_of((a, [_candidate(score=91.0)]), (b, [_candidate(score=88.0)]))
    attributor = _FakeAttributor({"r2": 20.0})   # r1 omitted, keeps its 91
    comments = _comments(_report(a, b), blame, attributor=attributor)
    assert len(comments) == 1
    assert "91%" in _row(comments[0].body, "a_metric")
    assert "20%" in _row(comments[0].body, "b_metric")


def test_a_partial_review_says_how_much_of_the_table_it_speaks_for():
    # Otherwise a narrative about the one row the reviewer answered reads as the
    # verdict on the rows above it that it never saw — a summary saying "this PR
    # does not fit" printed over an untouched 91%.
    a, b = _verdict(metric="a_metric"), _verdict(metric="b_metric")
    blame = _blame_of((a, [_candidate(score=91.0)]), (b, [_candidate(score=88.0)]))
    attributor = _FakeAttributor({"r2": 20.0}, summary="This PR does not fit.")
    body = _comments(_report(a, b), blame, attributor=attributor)[0].body
    assert "This assessment covers 1 regression of 2" in body
    assert "not part of it" in body


def test_a_review_that_answered_everything_adds_no_coverage_caveat():
    v = _verdict()
    attributor = _FakeAttributor({"r1": 92.0})
    body = _comments(_report(v), _blame([v], [_candidate()]),
                     attributor=attributor)[0].body
    assert "This assessment covers" not in body


@pytest.mark.parametrize("attributor", [
    None,
    _FakeAttributor(declines=True),
    _FakeAttributor(raises=RuntimeError("endpoint on fire")),
])
def test_without_a_usable_review_the_comment_still_renders(attributor):
    # No model, a decline, an adapter that raises: all the same degradation to
    # the per-configuration scores. A degraded comment beats a blocked one — and
    # nothing is withdrawn on the strength of a review that did not happen.
    v = _verdict()
    comments = _comments(_report(v), _blame([v], [_candidate()]), attributor=attributor)
    assert len(comments) == 1
    body = comments[0].body
    assert "91%" in body
    assert "hot path of every step" in body      # the ranker's own reason
    assert "AI-generated PR ranking" in body     # never presented as proof


def test_a_failing_diff_fetch_does_not_cost_the_comment():
    v = _verdict()

    def patch_for(repo, number):
        raise RuntimeError("GitHub is down")

    attributor = _FakeAttributor({"r1": 95.0})
    comments = _comments(_report(v), _blame([v], [_candidate()]),
                         attributor=attributor, patch_for=patch_for)
    assert len(comments) == 1
    assert attributor.requests == []  # the review never ran; the comment stands


# ── The claim's honesty, without a review ─────────────────────────────────────

def test_reason_label_does_not_overclaim_when_outranked():
    # The comment gate is min_score, not "ranked first": a PR at 85% fires even
    # when another candidate sits at 92% right below it in the others table, so
    # the label must not call it the *most* likely cause.
    v = _verdict()
    outranked = _candidate(score=85.0)
    top = _candidate(number=1180, repo="key4hep/DD4hep", score=92.0)
    body = _comments(_report(v), _blame([v], [outranked, top]))[0].body
    assert "judged this PR a likely cause" in body
    assert "most likely" not in body


def test_reason_label_claims_most_likely_only_when_top_ranked():
    v = _verdict()
    runner_up = _candidate(number=1180, repo="key4hep/DD4hep", score=22.0)
    body = _comments(_report(v), _blame([v], [_candidate(), runner_up]))[0].body
    assert "judged this PR the most likely cause" in body


def test_body_carries_the_marker_for_its_window():
    v = _verdict()
    comment = _comments(_report(v), _blame([v], [_candidate()]))[0]
    assert comment.marker == marker_for("2026-07-03", "2026-07-04")
    assert comment.body.startswith(comment.marker)


def test_body_lists_the_other_candidates_with_their_likelihoods():
    v = _verdict()
    others = [_candidate(), _candidate(number=1180, score=22.0, title="Unrelated cleanup")]
    body = _comments(_report(v), _blame([v], others))[0].body
    assert f"key4hep/k4geo#{_ZWSP}1180" in body and "22%" in body


def test_the_competing_field_is_gathered_across_the_whole_window():
    # A candidate that only competed in one configuration is still part of the
    # field the claim is made against, and keeps its strongest score.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    weak = _candidate(number=1180, repo="key4hep/DD4hep", score=20.0, title="Field map")
    strong = replace(weak, score=64.0)
    body = _comments(
        _report(allegro, idea),
        _blame_of((allegro, [_candidate(), weak]), (idea, [_candidate(), strong])),
    )[0].body
    summary = _row(body, "<summary>")
    assert "1 candidate" in summary and "highest 64%" in summary


def test_competing_candidates_are_named_but_never_referenced():
    # A PR that was only ever a candidate must not collect a cross-reference —
    # and with it a notification for everyone subscribed to it — every time some
    # other window implicates someone else. Neither its URL nor a live
    # `owner/repo#123` may appear; the broken number reads the same to a human.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=22.0)
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert "key4hep/DD4hep#1180" not in body
    assert other.url not in body
    assert f"key4hep/DD4hep#{_ZWSP}1180" in body


def test_a_hash_in_external_prose_references_nothing():
    # "Revert #45" in a candidate's title would cross-reference issue 45 in the
    # repository the comment is posted to — the same spam, smuggled in.
    v = _verdict()
    other = _candidate(number=1180, score=22.0, title="Revert #45 for now")
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert "#45" not in body and f"#{_ZWSP}45" in body


@pytest.mark.parametrize(
    ("runner_up", "expected"),
    [
        (86.0, "Only 5 points separate this PR"),
        (90.0, "Only 1 point separates this PR"),
        (91.0, "Nothing separates this PR"),
    ],
)
def test_a_close_ranking_admits_it_is_a_weak_preference(runner_up, expected):
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=runner_up)
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert expected in body
    assert "weak preference" in body


def test_a_ranking_that_ran_against_this_pr_says_which_way_it_ran():
    # 85% is not 85% "out of nowhere": another candidate scored 97% in the same
    # window, so the ranker preferred someone else and this PR merely also
    # cleared the bar. Saying "only 12 points separate them" would hide which way
    # the preference ran — the one thing the author needs to know.
    v = _verdict()
    mine = _candidate(score=85.0)
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=97.0)
    body = _comments(_report(v), _blame([v], [mine, other]))[0].body
    assert "scored 12 points **higher** than this PR" in body
    assert "runs against it" in body
    # …and the headline claim is downgraded to match.
    assert "a likely cause" in body and "the most likely cause" not in body


def test_crowded_prose_matches_displayed_percentages_at_a_rounding_boundary():
    # 90.49 displays as 90%, 90.51 displays as 91% (see _pct) — a one-point
    # *displayed* gap — even though the raw scores are 0.02 apart, which would
    # round to 0 "points" if the prose were computed from raw deltas instead of
    # from the same rounding _pct uses.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=90.51)
    body = _comments(_report(v), _blame([v], [_candidate(score=90.49), other]))[0].body
    assert "scored 1 point **higher** than this PR" in body
    assert "Nothing separates this PR" not in body


def test_a_clear_ranking_adds_no_caveat():
    # A caveat printed every night is wallpaper: it fires only when the field is
    # genuinely crowded, and the scores speak for a comfortable lead.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=22.0)
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert "weak preference" not in body


def test_body_invites_a_correction():
    # The bot writes into repositories k4Bench does not own, so the author is
    # told what to do when the call is wrong, not only that it might be.
    v = _verdict()
    body = _comments(_report(v), _blame([v], [_candidate()]))[0].body
    assert "reply here if this attribution looks wrong" in body


def test_body_says_so_when_nothing_else_was_in_the_frame():
    v = _verdict()
    body = _comments(_report(v), _blame([v], [_candidate()]))[0].body
    assert "only pull request found" in body


def test_body_renders_without_a_dashboard_url():
    # Offline/local rendering must still produce a usable comment.
    v = _verdict()
    body = _comments(_report(v), _blame([v], [_candidate()]), dashboard_url=None)[0].body
    assert "Where to look" not in body
    assert "91%" in body
    assert "ALLEGRO_o1_v03" in body  # the row still names its detector, unlinked


def test_open_ended_window_is_described_as_such():
    v = _verdict(base=None)
    body = _comments(_report(v), _blame([v], [_candidate()]))[0].body
    assert "no earlier settled measurement" in body


# ── Untrusted text ────────────────────────────────────────────────────────────

def test_table_cells_survive_hostile_text():
    # A pipe in a model-written reason or a PR title would end the column.
    v = _verdict()
    hostile = CandidatePR(
        repo="key4hep/k4geo", number=1, title="a | b\nsecond line",
        author="alice", url="https://github.com/key4hep/k4geo/pull/1",
        merged_at="2026-07-04T00:00:00Z", score=90.0, description="ranked",
    )
    headline = replace(_candidate(number=2, score=95.0), description="Line one\nline two")
    body = _comments(_report(v), _blame([v], [headline, hostile]))[0].body
    row = _row(body, f"key4hep/k4geo#{_ZWSP}1 ")
    # Two columns: the title's pipe is escaped, so it opens no third one.
    assert row.replace("\\|", "").count("|") == 3
    assert "a \\| b second line" in row  # pipe escaped, newline collapsed
    assert "Line one line two" in body  # the same for the quoted reason


def test_external_prose_is_defanged_of_mentions_and_markup():
    # A PR title and a model reason are untrusted text pasted into a comment the
    # bot posts in someone else's repo: an @mention must not ping, an HTML
    # comment must not hide content, an image must not load. A zero-width space
    # breaks each trigger while leaving the words readable.
    v = _verdict()
    headline = replace(
        _candidate(number=3, score=95.0),
        description="blame <!-- hidden --> @alice and <script>",
    )
    other = _candidate(
        number=1180, repo="key4hep/DD4hep", score=30.0,
        title="ping @team see ![x](http://e/i.png)",
    )
    body = _comments(_report(v), _blame([v], [headline, other]))[0].body
    assert "@team" not in body and f"@{_ZWSP}team" in body          # title mention
    assert "@alice" not in body and f"@{_ZWSP}alice" in body        # reason mention
    assert f"!{_ZWSP}[" in body                                     # image defused
    assert "<script>" not in body and f"<{_ZWSP}script>" in body


def test_the_reviews_narrative_is_defanged_like_any_other_quoted_prose():
    # The review is *asked* to name a better-fitting alternative as
    # owner/repo#number — which is exactly the cross-reference the bot refuses
    # to send. It renders as inert text.
    v = _verdict()
    attributor = _FakeAttributor(
        {"r1": 90.0},
        summary="AIDASoft/DD4hep#77 fits better; see https://evil.example and @alice",
    )
    body = _comments(_report(v), _blame([v], [_candidate()]),
                     attributor=attributor)[0].body
    assert "DD4hep#77" not in body and f"DD4hep#{_ZWSP}77" in body
    assert "https://evil.example" not in body
    assert "@alice" not in body


def test_external_prose_cannot_carry_an_active_link():
    # The sharpest version of the same problem: no mention, no markup, just a
    # link. A Markdown link puts an arbitrary destination into a comment the bot
    # signs its own name to, and a bare pull-request URL — which GitHub autolinks
    # with no syntax at all — cross-references that PR's timeline, the exact
    # notification _pr_ref refuses to send. Both must land as inert text.
    v = _verdict()
    headline = replace(
        _candidate(number=3, score=95.0),
        description="see https://evil.example/x and www.evil.example",
    )
    other = _candidate(
        number=1180, repo="key4hep/DD4hep", score=30.0,
        title="[click me](https://evil.example) "
              "https://github.com/key4hep/DD4hep/pull/1180",
    )
    body = _comments(_report(v), _blame([v], [headline, other]))[0].body
    row = _row(body, f"DD4hep#{_ZWSP}1180")
    assert "](" not in row and f"]{_ZWSP}(" in row            # no link in the title
    assert "https://evil.example" not in body                # no autolinked URL
    assert f"https:{_ZWSP}//evil.example" in body
    assert "www.evil.example" not in body
    assert f"www{_ZWSP}.evil.example" in body
    # The bot's *own* links — the dashboard views it renders itself — are
    # untouched: only quoted, externally-authored prose is defanged.
    assert f"]({_DASH}" in body
    # The only live HTML comments are the bot's own hidden lines; the one
    # smuggled into the reason is broken by the same zero-width space.
    assert body.count("<!--") == 2 and body.startswith("<!--")


# ── Stability, and the facts digest ───────────────────────────────────────────

def test_body_is_stable_across_identical_nights():
    # The upsert only edits when something changed, so an unchanged night must
    # render byte-identically — no set ordering leaking into the output.
    a, b = _verdict(metric="wall_time_s"), _verdict(metric="mean_time_s", pct=0.14)
    first = _comments(_report(a, b), _blame([a, b], [_candidate()]))[0].body
    second = _comments(_report(b, a), _blame([b, a], [_candidate()]))[0].body
    assert first == second


def test_a_non_finite_change_does_not_destabilise_the_order():
    # A NaN in the sort key compares false against everything, which would leave
    # the table in whatever order the verdicts happened to arrive in — the one
    # thing the key exists to rule out. It sorts as no movement instead,
    # matching the "—" the cell renders for it.
    a = _verdict(metric="wall_time_s", pct=float("nan"))
    b = _verdict(metric="mean_time_s", pct=0.14)
    c = _verdict(metric="peak_rss_mb", pct=None)
    first = _comments(_report(a, b, c), _blame([a, b, c], [_candidate()]))[0].body
    second = _comments(_report(c, a, b), _blame([c, a, b], [_candidate()]))[0].body
    assert first == second
    rows = _table_rows(first)
    # Biggest real movement first; the two immeasurable ones fall to identity.
    assert "mean_time_s" in rows[0]
    assert "—" in rows[1] and "—" in rows[2]


def test_body_is_stable_across_consecutive_nights():
    # A standing regression renders byte-identically on the next night too, so
    # the upsert edits nothing and re-notifies no one.
    v = _verdict()
    monday = _comments(_report(v, night="2026-07-05"), _blame([v], [_candidate()]))[0].body
    tuesday = _comments(_report(v, night="2026-07-06"), _blame([v], [_candidate()]))[0].body
    assert monday == tuesday


def test_scope_walk_order_does_not_change_the_body():
    # A competing PR can carry a different likelihood in each scope of the same
    # window. Whichever scope was walked first, one comment is produced and the
    # body is identical, so a reordering between nights does not re-edit it.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    hi = _candidate()
    lo = _candidate(number=1180, repo="key4hep/DD4hep", score=25.0, title="Other work")
    top = replace(lo, score=70.0)

    forward = _comments(_report(allegro, idea),
                        _blame_of((allegro, [hi, lo]), (idea, [hi, top])))
    reverse = _comments(_report(idea, allegro),
                        _blame_of((idea, [hi, top]), (allegro, [hi, lo])))
    assert len(forward) == 1  # one comment for the PR+window, whatever the order
    assert forward[0].body == reverse[0].body


def test_the_facts_digest_ignores_the_model_and_tracks_the_benchmarks():
    # The narrative is regenerated nightly and will not repeat itself word for
    # word; editing a standing comment for that would notify everyone watching
    # the PR for nothing. The digest covers what a reader would call a change.
    v = _verdict()
    blame = _blame([v], [_candidate()])
    first = _comments(_report(v), blame, attributor=_FakeAttributor(
        {"r1": 92.0}, summary="Only ALLEGRO moved."))[0]
    second = _comments(_report(v), blame, attributor=_FakeAttributor(
        {"r1": 88.0}, summary="ALLEGRO alone shows the step."))[0]
    assert first.body != second.body
    assert first.facts_digest == second.facts_digest

    moved_further = _comments(_report(_verdict(pct=0.55)),
                              _blame([_verdict(pct=0.55)], [_candidate()]))[0]
    assert moved_further.facts_digest != first.facts_digest


def test_the_digest_ignores_a_competitors_score_drifting():
    # The field is a fact; what the ranker scored it is model output. A rival
    # sliding from 84.4 to 84.6 crosses a rounding boundary and would otherwise
    # re-render, edit and re-notify a standing comment for nothing.
    v = _verdict()
    rival = _candidate(number=1180, repo="key4hep/DD4hep", score=84.4)
    first = _comments(_report(v), _blame([v], [_candidate(), rival]))[0]
    second = _comments(_report(v), _blame(
        [v], [_candidate(), replace(rival, score=84.6)]))[0]
    assert first.facts_digest == second.facts_digest


def test_the_digest_is_carried_in_the_body_and_readable_back():
    v = _verdict()
    comment = _comments(_report(v), _blame([v], [_candidate()]))[0]
    assert facts_digest_of(comment.body) == comment.facts_digest
    assert facts_digest_of("no markers here") == ""


def test_the_digest_notices_a_change_in_the_competing_field():
    # A new candidate appearing in the window changes what the claim was made
    # against, which is a real change worth an edit.
    v = _verdict()
    alone = _comments(_report(v), _blame([v], [_candidate()]))[0]
    crowded = _comments(_report(v), _blame(
        [v], [_candidate(), _candidate(number=1180, repo="key4hep/DD4hep", score=30.0)],
    ))[0]
    assert alone.facts_digest != crowded.facts_digest


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
