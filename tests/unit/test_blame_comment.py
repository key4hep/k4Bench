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
    CommentStormError,
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
#: What the renderer breaks a GitHub-active sequence with — invisible to a
#: reader, inert to GitHub's reference and mention parsers.
_ZWSP = "\u200b"


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


def _report(*verdicts: MetricVerdict, night="2026-07-05") -> NightlyReport:
    groups: dict[tuple, RunGroupReport] = {}
    for v in verdicts:
        key = (v.detector, v.platform, v.sample)
        group = groups.get(key)
        if group is None:
            group = groups[key] = RunGroupReport(
                detector=v.detector, platform=v.platform, sample=v.sample,
                k4h_release="key4hep-2026-07-04", run_date=night,
                run_id=night, verdicts=[],
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
        )
        for v in verdicts
    ]
    return BlameReport(
        generated_at="2026-07-05T01:00:00", report_night="2026-07-05",
        entries=tuple(entries),
    )


def _select(report, blame, policy=None):
    return select(report, blame, policy or _policy(), dashboard_url=_DASH)


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


def test_over_the_cap_raises_a_storm_error():
    # A night louder than max_comments is a bug, not a night: rather than post
    # the top N accusations into repos we don't own, the whole night is dropped —
    # and raising (not returning []) lets the CLI tell it apart from a quiet night.
    verdicts = [_verdict(metric=f"m{i}", sample=f"s{i}") for i in range(4)]
    candidates = [_candidate(number=100 + i) for i in range(4)]
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=tuple(
            _blame([v], [c]).entries[0] for v, c in zip(verdicts, candidates, strict=True)
        ),
    )
    with pytest.raises(CommentStormError) as exc:
        _select(_report(*verdicts), blame, _policy(max_comments=2))
    assert exc.value.count == 4 and exc.value.cap == 2


def test_at_the_cap_still_posts():
    # The cap is a ceiling, not a trigger: exactly max_comments is fine.
    verdicts = [_verdict(metric=f"m{i}", sample=f"s{i}") for i in range(2)]
    candidates = [_candidate(number=100 + i) for i in range(2)]
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=tuple(
            _blame([v], [c]).entries[0] for v, c in zip(verdicts, candidates, strict=True)
        ),
    )
    assert len(_select(_report(*verdicts), blame, _policy(max_comments=2))) == 2


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


def test_reason_label_does_not_overclaim_when_outranked():
    # The comment gate is min_score, not "ranked first": a PR at 85% fires even
    # when another candidate sits at 92% right below it in the others table, so
    # the label must not call it the *most* likely cause.
    v = _verdict()
    outranked = _candidate(score=85.0)
    top = _candidate(number=1180, repo="key4hep/DD4hep", score=92.0)
    body = _select(_report(v), _blame([v], [outranked, top]))[0].body
    assert "judged this PR a likely cause" in body
    assert "most likely" not in body


def test_reason_label_claims_most_likely_only_when_top_ranked():
    v = _verdict()
    runner_up = _candidate(number=1180, repo="key4hep/DD4hep", score=22.0)
    body = _select(_report(v), _blame([v], [_candidate(), runner_up]))[0].body
    assert "judged this PR the most likely cause" in body


def test_body_lists_the_other_candidates_with_their_likelihoods():
    v = _verdict()
    others = [_candidate(), _candidate(number=1180, score=22.0, title="Unrelated cleanup")]
    body = _select(_report(v), _blame([v], others))[0].body
    assert f"key4hep/k4geo#{_ZWSP}1180" in body and "22%" in body


def test_competing_candidates_are_named_but_never_referenced():
    # A PR that was only ever a candidate must not collect a cross-reference —
    # and with it a notification for everyone subscribed to it — every time some
    # other window implicates someone else. Neither its URL nor a live
    # `owner/repo#123` may appear; the broken number reads the same to a human.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=22.0)
    body = _select(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert "key4hep/DD4hep#1180" not in body
    assert other.url not in body
    assert f"key4hep/DD4hep#{_ZWSP}1180" in body


def test_a_hash_in_external_prose_references_nothing():
    # "Revert #45" in a candidate's title would cross-reference issue 45 in the
    # repository the comment is posted to — the same spam, smuggled in.
    v = _verdict()
    other = _candidate(number=1180, score=22.0, title="Revert #45 for now")
    body = _select(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert "#45" not in body and f"#{_ZWSP}45" in body


def test_the_strongest_competing_score_shows_without_expanding():
    # How far ahead this PR sits is the difference between a ranking that picked
    # it and one that barely preferred it, so the collapsed summary carries the
    # top competing likelihood for a reader who expands nothing.
    v = _verdict()
    others = [
        _candidate(),
        _candidate(number=1180, repo="key4hep/DD4hep", score=22.0, title="Cleanup"),
        _candidate(number=1181, repo="key4hep/DD4hep", score=64.0, title="Field map"),
    ]
    body = _select(_report(v), _blame([v], others))[0].body
    summary = next(line for line in body.splitlines() if "<summary>" in line)
    assert "2 candidates" in summary and "highest 64%" in summary


@pytest.mark.parametrize(
    ("runner_up", "expected"),
    [
        (86.0, "Only 5 points separate this PR"),
        (90.0, "Only 1 point separates this PR"),
        (91.0, "Nothing separates this PR"),
        (95.0, "Only 4 points separate this PR"),  # crowded from above, too
    ],
)
def test_a_close_ranking_admits_it_is_a_weak_preference(runner_up, expected):
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=runner_up)
    body = _select(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert expected in body
    assert "weak preference" in body


def test_a_clear_ranking_adds_no_caveat():
    # A caveat printed every night is wallpaper: it fires only when the field is
    # genuinely crowded, and the scores speak for a comfortable lead.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=22.0)
    body = _select(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert "weak preference" not in body


def test_body_invites_a_correction():
    # The bot writes into repositories k4Bench does not own, so the author is
    # told what to do when the call is wrong, not only that it might be.
    v = _verdict()
    body = _select(_report(v), _blame([v], [_candidate()]))[0].body
    assert "reply here if this attribution looks wrong" in body


def test_body_says_so_when_nothing_else_was_in_the_frame():
    v = _verdict()
    body = _select(_report(v), _blame([v], [_candidate()]))[0].body
    assert "only pull request found" in body


def test_body_links_the_window_in_the_dashboard():
    v = _verdict()
    body = _select(_report(v), _blame([v], [_candidate()]))[0].body
    assert "window=2026-07-03..2026-07-04" in body        # the scoped Regressions view
    assert "&to=2026-07-04" in body                       # the package diff
    # ?stack= is the dashboard's release *directory*, not the bare release date
    # a verdict carries — a bare date selects nothing and silently falls back.
    assert "stack=key4hep-2026-07-04" in body
    # Nothing that varies from night to night: no report-night query param and no
    # CI-run URL, either of which would edit a standing comment every night.
    assert "report=" not in body
    assert "actions/runs" not in body


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
    row = next(line for line in body.splitlines() if f"key4hep/k4geo#{_ZWSP}1 " in line)
    # Two columns: the title's pipe is escaped, so it opens no third one.
    assert row.replace("\\|", "").count("|") == 3
    assert "a \\| b second line" in row  # pipe escaped, newline collapsed
    assert "Line one line two" in body  # the same for the quoted reason


def test_external_prose_is_defanged_of_mentions_and_markup():
    # A PR title and a model reason are untrusted text pasted into a comment the
    # bot posts in someone else's repo: an @mention must not ping, an HTML
    # comment must not hide content, an image must not load. A zero-width space
    # breaks each trigger while leaving the words readable. The reason is quoted
    # for the headline PR; a candidate's title shows in the others table.
    v = _verdict()
    headline = replace(
        _candidate(number=3, score=95.0),
        description="blame <!-- hidden --> @alice and <script>",
    )
    other = _candidate(
        number=1180, repo="key4hep/DD4hep", score=30.0,
        title="ping @team see ![x](http://e/i.png)",
    )
    body = _select(_report(v), _blame([v], [headline, other]))[0].body
    zwsp = _ZWSP
    assert "@team" not in body and f"@{zwsp}team" in body          # title mention
    assert "@alice" not in body and f"@{zwsp}alice" in body        # reason mention
    assert f"!{zwsp}[" in body                                     # image defused
    assert "<script>" not in body and f"<{zwsp}script>" in body
    # The only live HTML comment is the bot's own marker on the first line; the
    # one smuggled into the reason is broken by the same zero-width space.
    assert body.count("<!--") == 1 and body.startswith("<!--")


def test_body_is_stable_across_identical_nights():
    # The upsert only edits when the body changes, so an unchanged night must
    # render byte-identically — no set ordering leaking into the output.
    a, b = _verdict(metric="wall_time_s"), _verdict(metric="mean_time_s", pct=0.14)
    first = _select(_report(a, b), _blame([a, b], [_candidate()]))[0].body
    second = _select(_report(b, a), _blame([b, a], [_candidate()]))[0].body
    assert first == second


def test_body_is_stable_across_consecutive_nights():
    # A standing regression renders byte-identically on the next night too, so
    # the upsert edits nothing and re-notifies no one. Nothing that changes from
    # night to night — the report night, a per-run CI URL — may leak into it.
    v = _verdict()
    monday = _select(_report(v, night="2026-07-05"), _blame([v], [_candidate()]))[0].body
    tuesday = _select(_report(v, night="2026-07-06"), _blame([v], [_candidate()]))[0].body
    assert monday == tuesday


def test_scope_walk_order_does_not_change_the_body():
    # The ranker scores a candidate once per (detector, platform, sample) scope,
    # so a competing PR can carry a different likelihood in each scope of the
    # same window. Whichever scope was walked first, the same one leads the
    # comment and the body is identical, so a reordering between nights does not
    # re-edit a standing comment.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    hi = _candidate()
    lo = _candidate(number=1180, repo="key4hep/DD4hep", score=25.0, title="Other work")
    top = _candidate(number=1180, repo="key4hep/DD4hep", score=70.0, title="Other work")

    def blame(*pairs):
        return BlameReport(
            generated_at="x", report_night="2026-07-05",
            entries=tuple(_blame([v], cands).entries[0] for v, cands in pairs),
        )

    forward = _select(_report(allegro, idea), blame((allegro, [hi, lo]), (idea, [hi, top])))
    reverse = _select(_report(idea, allegro), blame((idea, [hi, top]), (allegro, [hi, lo])))
    assert len(forward) == 1  # one comment for the PR+window, whatever the order
    assert forward[0].body == reverse[0].body
    # The leading scope is rendered in full, so its competitor is the one named.
    assert "25%" in forward[0].body


def test_only_the_leading_configuration_is_rendered_in_full():
    # A 95% ALLEGRO judgement and an 81% IDEA judgement are two rankings, not one
    # 95% ranking of both. The strongest leads the comment in full; the other
    # keeps its own likelihood in one summary row rather than a second section.
    allegro = _verdict(detector="ALLEGRO_o1_v03", pct=0.21)
    idea = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb", pct=0.11)
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=(
            _blame([allegro], [_candidate(score=95.0)]).entries[0],
            _blame([idea], [_candidate(score=81.0)]).entries[0],
        ),
    )
    body = _select(_report(allegro, idea), blame)[0].body
    assert "🎯 95% — ALLEGRO_o1_v03" in body      # the lead, with its own section
    assert "🎯 81% — IDEA_o1_v03" not in body     # not a second full section
    assert body.count("What moved") == 1
    # …but the reader still learns it moved, by how much, and how likely the
    # ranker held this PR to be *there* — the lead's score speaks only for itself.
    assert "Also affected in this window" in body
    assert "1 further benchmark configuration" in body
    row = next(line for line in body.splitlines() if "IDEA_o1_v03" in line)
    assert "81%" in row and "peak_rss_mb" in row and "+11.0%" in row
    assert "detector=IDEA_o1_v03" in row  # one click to the rest of the window


def test_the_largest_movement_leads_among_equally_ranked_configurations():
    # Same likelihood in both scopes: the one that moved furthest is the one
    # worth reading in full.
    small = _verdict(detector="ALLEGRO_o1_v03", pct=0.05)
    large = _verdict(detector="IDEA_o1_v03", pct=0.40)
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=(
            _blame([small], [_candidate()]).entries[0],
            _blame([large], [_candidate()]).entries[0],
        ),
    )
    body = _select(_report(small, large), blame)[0].body
    assert "🎯 91% — IDEA_o1_v03" in body


def test_summary_rows_name_only_what_differs_from_the_leading_configuration():
    # Same sample and platform as the lead: the detector alone identifies the
    # row. A row that ran a different sample says which one.
    lead = _verdict(detector="ALLEGRO_o1_v03", pct=0.30)
    same = _verdict(detector="IDEA_o1_v03", pct=0.20)
    other = _verdict(detector="CLD_o2_v07", pct=0.10, sample="p8_ee_Zbb_ecm91")
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=tuple(
            _blame([v], [_candidate()]).entries[0] for v in (lead, same, other)
        ),
    )
    body = _select(_report(lead, same, other), blame)[0].body
    rows = {
        d: next(line for line in body.splitlines() if f"[{d}" in line)
        for d in ("IDEA_o1_v03", "CLD_o2_v07")
    }
    assert "Single e⁻" not in rows["IDEA_o1_v03"]
    assert "Z → bb" in rows["CLD_o2_v07"]


def test_a_single_configuration_carries_no_summary_section():
    v = _verdict()
    body = _select(_report(v), _blame([v], [_candidate()]))[0].body
    assert "Also affected" not in body
