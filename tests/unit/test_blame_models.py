"""Unit tests for :mod:`k4bench.blame.models` — serialization and the
verdict↔entry join that keeps ``blame.json`` decoupled from ``report.json``."""

from __future__ import annotations

import pytest

from k4bench.blame.models import (
    BlameEntry,
    BlameReport,
    BlameSchemaError,
    CandidatePR,
    RepoBlame,
    ranking_coverage,
)
from k4bench.regression.models import Direction, MetricVerdict, Severity


def _pr(
    number: int, score: float = 0.0, repo: str = "key4hep/k4geo", ranked: bool = True
) -> CandidatePR:
    return CandidatePR(
        repo=repo, number=number, title=f"PR {number}", author="alice",
        url=f"https://github.com/{repo}/pull/{number}", merged_at="2026-07-04T00:00:00Z",
        files=("FCCee/ALLEGRO/compact/x.xml",), additions=10, deletions=2,
        score=score, description="lowers the tracker step limit", ranked=ranked,
    )


def _entry(**over) -> BlameEntry:
    base = dict(
        detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e", label="baseline", metric="wall_time_s", sub_detector=None,
        base_release="2026-07-03", onset_release="2026-07-04",
        repos=(RepoBlame(
            package="k4geo", repo="key4hep/k4geo",
            base_commit="a" * 40, head_commit="c" * 40,
            compare_url="https://github.com/key4hep/k4geo/compare/a...c",
            status="changed", candidates=(_pr(1, score=3.0), _pr(2, score=5.0)),
        ),),
        n_unchanged=60,
    )
    base.update(over)
    return BlameEntry(**base)


def test_round_trips_through_json():
    report = BlameReport(
        generated_at="2026-07-05T00:00:00", report_night="2026-07-05",
        entries=(_entry(),),
    )
    restored = BlameReport.from_json(report.to_json())
    assert restored == report


def test_from_json_drops_unknown_keys():
    # blame.json is read by whatever dashboard is deployed; a newer writer adding
    # a field must not break an older reader.
    data = BlameReport(
        generated_at="g", report_night="2026-07-05", entries=(_entry(),)
    ).to_json()
    data["future_top_level"] = 1
    data["entries"][0]["future_entry_field"] = 2
    data["entries"][0]["repos"][0]["future_repo_field"] = 3
    data["entries"][0]["repos"][0]["candidates"][0]["future_pr_field"] = 4

    restored = BlameReport.from_json(data)
    assert restored.report_night == "2026-07-05"
    cand = restored.entries[0].repos[0].candidates[0]
    assert cand.number in (1, 2)
    assert cand.description == "lowers the tracker step limit"


def test_candidates_are_flattened_worst_first():
    # The flat ledger sorts by score desc regardless of which repo a PR is in.
    entry = _entry(repos=(
        RepoBlame(package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
                  head_commit="c" * 40, compare_url=None, status="changed",
                  candidates=(_pr(1, score=1.0),)),
        RepoBlame(package="dd4hep", repo="AIDASoft/DD4hep", base_commit="d" * 40,
                  head_commit="e" * 40, compare_url=None, status="changed",
                  candidates=(_pr(9, score=7.0, repo="AIDASoft/DD4hep"),)),
    ))
    assert [c.number for c in entry.candidates] == [9, 1]


def test_entry_for_joins_on_verdict_identity_and_window():
    report = BlameReport("g", "2026-07-05", entries=(_entry(),))
    matching = MetricVerdict(
        detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e", label="baseline", metric_family="time",
        metric="wall_time_s", sub_detector=None, run_id="2026-07-05",
        run_date="2026-07-05", value=1.0, baseline_median=1.0, baseline_mad=0.1,
        pct_change=0.2, z_score=5.0, severity=Severity.CONFIRMED,
        direction=Direction.UP, reason="step",
        onset_run_id="2026-07-04", onset_run_date="2026-07-04",
        last_accepted_run_id="2026-07-03", last_accepted_run_date="2026-07-03",
    )
    assert report.entry_for(matching) is report.entries[0]

    # A different metric on the same series has no blame entry.
    other = MetricVerdict(**{**matching.__dict__, "metric": "peak_rss_mb"})
    assert report.entry_for(other) is None

    # Same identity, different window: a sidecar left over from an earlier
    # build must never attach its ranking to a regression whose window it did
    # not examine.
    moved = MetricVerdict(**{**matching.__dict__, "onset_run_date": "2026-07-06"})
    assert report.entry_for(moved) is None


def test_ranking_coverage_counts_each_entry_and_accepts_zero_with_reason():
    ranked_zero = _pr(1, score=0.0)
    # Never scored: the state, not the number, is what coverage counts — this
    # one carries a score only because a hostile sidecar could.
    missing = CandidatePR(
        repo="key4hep/k4geo", number=2, title="PR 2", author="alice", url="u",
        score=99.0, description="", ranked=False,
    )
    repos = (RepoBlame(
        package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
        head_commit="c" * 40, compare_url=None, status="changed",
        candidates=(ranked_zero, missing),
    ),)
    # Each metric is ranked on its own, so entries sharing a window still each
    # owe a judgement per candidate — the zero score with a reason counts, the
    # empty description does not, in both entries.
    report = BlameReport("g", "2026-07-05", entries=(
        _entry(repos=repos),
        _entry(metric="user_cpu_s", repos=repos),
    ))
    assert ranking_coverage(report) == (2, 4, ["key4hep/k4geo#2"])


def test_ranking_coverage_exempts_incomplete_discovery():
    # An entry whose candidate list is known to be partial is deliberately left
    # unranked by the builder — completeness checks must not fail it.
    unranked = CandidatePR(
        repo="key4hep/k4geo", number=7, title="PR 7", author="alice", url="u",
    )
    incomplete = _entry(repos=(RepoBlame(
        package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
        head_commit="c" * 40, compare_url=None, status="changed",
        candidates=(unranked,), truncated=True,
    ),))
    assert incomplete.discovery_incomplete is True
    report = BlameReport("g", "2026-07-05", entries=(incomplete,))
    assert ranking_coverage(report) == (0, 0, [])


def test_from_json_raises_schema_error_on_malformed_shapes():
    # Valid JSON, wrong structure — each must raise the one dedicated schema
    # error the dashboard/notifier boundaries catch, never a bare TypeError.
    for data in (
        [],                                       # top level is a list
        {"entries": [{}]},                        # entry missing required fields
        {"entries": ["not-an-object"]},
        {"entries": [_entry().to_dict() | {"repos": [{"candidates": ["x"]}]}]},
    ):
        with pytest.raises(BlameSchemaError):
            BlameReport.from_json(data)


def _with_candidate_field(**patch) -> dict:
    data = BlameReport("g", "2026-07-05", entries=(_entry(),)).to_json()
    data["entries"][0]["repos"][0]["candidates"][0] |= patch
    return data


def test_from_json_rejects_wrongly_typed_fields():
    # Valid JSON whose values can't be coerced to their declared types must
    # fail *inside* the schema boundary, not later in a sort or email format.
    for patch in (
        {"score": "very likely"},
        {"number": "not-a-number"},
        {"files": 7},  # not iterable of paths
    ):
        with pytest.raises(BlameSchemaError):
            BlameReport.from_json(_with_candidate_field(**patch))


def test_from_json_coerces_lenient_but_renderable_values():
    # A numeric string score is fine; a non-finite one degrades to the unranked
    # 0.0 rather than poisoning sorts and formats downstream.
    report = BlameReport.from_json(_with_candidate_field(score="72"))
    assert report.entries[0].repos[0].candidates[0].score == 72.0
    report = BlameReport.from_json(_with_candidate_field(score=float("nan")))
    assert report.entries[0].repos[0].candidates[0].score == 0.0


# ── Ranked is a state, not a score ────────────────────────────────────────────
# ``score == 0.0`` has to mean one thing only: "the model looked at this pull
# request and rated it zero". "Nobody ever asked" is a different fact with
# different consequences — it must never clear a threshold, and it must never be
# shown as a judgement — so it travels as its own field.

def test_the_ranked_state_survives_a_round_trip():
    judged_zero = _pr(1, score=0.0, ranked=True)
    never_asked = _pr(2, score=0.0, ranked=False)
    entry = _entry(repos=(RepoBlame(
        package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
        head_commit="c" * 40, compare_url=None, status="CHANGED",
        candidates=(judged_zero, never_asked),
    ),))
    restored = BlameReport.from_json(
        BlameReport("g", "2026-07-05", entries=(entry,)).to_json()
    )
    by_number = {c.number: c for c in restored.entries[0].candidates}
    # Identical scores, opposite states — and the states are what survived.
    assert by_number[1].score == by_number[2].score == 0.0
    assert by_number[1].ranked and not by_number[2].ranked


def test_a_sidecar_that_records_no_judgement_is_read_as_unranked():
    # A file with no ``ranked`` key never recorded a judgement, so it has not
    # made one — whatever score sits beside it. Reading that as "ranked" is the
    # only unsafe direction, since it is what clears a comment threshold.
    data = BlameReport("g", "2026-07-05", entries=(_entry(repos=(RepoBlame(
        package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
        head_commit="c" * 40, compare_url=None, status="CHANGED",
        candidates=(_pr(1, score=95.0),),
    ),)),)).to_json()
    del data["entries"][0]["repos"][0]["candidates"][0]["ranked"]

    restored = BlameReport.from_json(data)
    assert not restored.entries[0].candidates[0].ranked


def test_the_flat_ledger_puts_the_unjudged_after_the_judged():
    # An unranked candidate has no likelihood at all, so it cannot sit *among*
    # the scores — least of all at the 0% end, where it would read as the
    # ranker's weakest pick rather than as one it never rated.
    entry = _entry(repos=(RepoBlame(
        package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
        head_commit="c" * 40, compare_url=None, status="CHANGED",
        candidates=(
            _pr(1, score=0.0, ranked=False),
            _pr(2, score=0.0, ranked=True),
        ),
    ),))
    assert [c.number for c in entry.candidates] == [2, 1]
