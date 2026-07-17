"""Unit tests for :mod:`k4bench.blame.models` — serialization and the
verdict↔entry join that keeps ``blame.json`` decoupled from ``report.json``."""

from __future__ import annotations

from k4bench.blame.models import (
    BlameEntry,
    BlameReport,
    CandidatePR,
    RepoBlame,
)
from k4bench.regression.models import Direction, MetricVerdict, Severity


def _pr(number: int, score: float = 0.0, repo: str = "key4hep/k4geo") -> CandidatePR:
    return CandidatePR(
        repo=repo, number=number, title=f"PR {number}", author="alice",
        url=f"https://github.com/{repo}/pull/{number}", merged_at="2026-07-04T00:00:00Z",
        files=("FCCee/ALLEGRO/compact/x.xml",), additions=10, deletions=2,
        score=score, description="lowers the tracker step limit",
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


def test_entry_for_joins_on_verdict_identity():
    report = BlameReport("g", "2026-07-05", entries=(_entry(),))
    matching = MetricVerdict(
        detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e", label="baseline", metric_family="time",
        metric="wall_time_s", sub_detector=None, run_id="2026-07-05",
        run_date="2026-07-05", value=1.0, baseline_median=1.0, baseline_mad=0.1,
        pct_change=0.2, z_score=5.0, severity=Severity.CONFIRMED,
        direction=Direction.UP, reason="step",
    )
    assert report.entry_for(matching) is report.entries[0]

    # A different metric on the same series has no blame entry.
    other = MetricVerdict(**{**matching.__dict__, "metric": "peak_rss_mb"})
    assert report.entry_for(other) is None
