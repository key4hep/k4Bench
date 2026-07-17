"""Smoke test for ``.github/scripts/backfill_blame.py`` — the historical blame
backfill. Drives :func:`run_backfill` with fully injected IO (no WebEOS, no
xrootd, no network) and a fake ranker, asserting the three properties that make
a backfill safe to re-run: dry-run writes nothing, ``--apply`` uploads shaped
blame with scores, and an already-annotated night is skipped."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from k4bench.blame import builder as builder_mod
from k4bench.blame.github import GitHubClient, RepoResolution
from k4bench.blame.models import CandidatePR
from k4bench.blame.rank import Ranking
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.regression.render import to_json

_BACKFILL_SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "backfill_blame.py"
)
_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_K4GEO = "https://github.com/key4hep/k4geo.git"


def _backfill():
    """Import the CLI by path (registered in sys.modules so its dataclasses
    resolve)."""
    spec = importlib.util.spec_from_file_location("backfill_blame", _BACKFILL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report_json() -> dict:
    """A one-night report with a single confirmed, attributable regression."""
    verdict = MetricVerdict(
        detector="DET", platform=_PLAT, sample="single_e", label="baseline",
        metric_family="time", metric="wall_time_s", sub_detector=None,
        run_id="2026-07-04", run_date="2026-07-04", value=120.0,
        baseline_median=100.0, baseline_mad=1.0, pct_change=0.2, z_score=6.0,
        severity=Severity.CONFIRMED, direction=Direction.UP, reason="step",
        onset_run_id="2026-07-04", onset_run_date="2026-07-04",
        last_accepted_run_id="2026-07-03", last_accepted_run_date="2026-07-03",
    )
    group = RunGroupReport(
        detector="DET", platform=_PLAT, sample="single_e",
        k4h_release="key4hep-2026-07-04", run_date="2026-07-04", run_id="2026-07-04",
        verdicts=[verdict],
    )
    return to_json(NightlyReport(generated_at="2026-07-04T00:00:00", groups=[group]))


def _stack_packages(detector, platform, stack):
    """k4geo at commit ``a`` before the boundary, ``c`` after — so it diffs."""
    commit = "a" * 40 if stack.endswith("2026-07-03") else "c" * 40
    return {"k4geo": {"commit": commit, "version": "develop", "repo_url": _K4GEO}}


class _FakeRanker:
    def rank(self, request):
        return {(c.repo, c.number): Ranking(90.0, "raises the step count")
                for c in request.candidates}


@pytest.fixture
def io_and_store(monkeypatch):
    """A BackfillIO over an in-memory store, with GitHub resolution faked so the
    window yields one candidate PR for the ranker to score."""
    backfill = _backfill()
    report = _report_json()
    uploaded: dict[str, str] = {}

    def fake_resolve(client, slug, base, head):
        return RepoResolution(
            candidates=[CandidatePR(
                repo=slug, number=1234, title="Lower the step limit",
                author="alice", url=f"https://github.com/{slug}/pull/1234",
            )],
            patches={1234: "@@\n+ more steps"},
        )
    monkeypatch.setattr(builder_mod, "resolve_repo_prs", fake_resolve)

    io = backfill.BackfillIO(
        list_dates=lambda: ["2026-07-04"],
        fetch_report=lambda night: report if night == "2026-07-04" else None,
        blame_present=lambda night: night in uploaded,
        upload=lambda night, text: uploaded.__setitem__(night, text),
        stack_packages=_stack_packages,
    )
    return backfill, io, uploaded


def test_dry_run_writes_nothing(io_and_store):
    backfill, io, uploaded = io_and_store
    stats = backfill.run_backfill(io, github=GitHubClient(), ranker=_FakeRanker(), apply=False)
    assert uploaded == {}              # nothing uploaded on a dry run
    assert stats.would_upload == 1     # but it reports what it would do
    assert stats.uploaded == 0


def test_apply_uploads_shaped_blame_with_scores(io_and_store):
    backfill, io, uploaded = io_and_store
    stats = backfill.run_backfill(io, github=GitHubClient(), ranker=_FakeRanker(), apply=True)
    assert stats.uploaded == 1
    assert set(uploaded) == {"2026-07-04"}

    data = json.loads(uploaded["2026-07-04"])
    assert data["report_night"] == "2026-07-04"
    entry = next(e for e in data["entries"] if e["metric"] == "wall_time_s")
    cand = entry["repos"][0]["candidates"][0]
    assert cand["number"] == 1234
    assert cand["score"] == 90.0
    assert cand["description"] == "raises the step count"


def test_apply_is_idempotent(io_and_store):
    backfill, io, uploaded = io_and_store
    first = backfill.run_backfill(io, github=GitHubClient(), ranker=_FakeRanker(), apply=True)
    assert first.uploaded == 1
    snapshot = dict(uploaded)

    # A second run sees the night already annotated and skips it — a throttled
    # backfill resumes cleanly instead of re-inferring what is done.
    second = backfill.run_backfill(io, github=GitHubClient(), ranker=_FakeRanker(), apply=True)
    assert second.uploaded == 0
    assert second.skipped_existing == 1
    assert uploaded == snapshot


def test_overwrite_reprocesses_existing(io_and_store):
    backfill, io, uploaded = io_and_store
    backfill.run_backfill(io, github=GitHubClient(), ranker=_FakeRanker(), apply=True)
    again = backfill.run_backfill(
        io, github=GitHubClient(), ranker=_FakeRanker(), apply=True, overwrite=True
    )
    assert again.uploaded == 1          # forced past the idempotency gate
    assert again.skipped_existing == 0


def test_since_until_limit_filter_nights():
    backfill = _backfill()
    nights = ["2026-07-06", "2026-07-05", "2026-07-04", "2026-07-03"]
    assert backfill._select_nights(nights, since="2026-07-04", until=None, limit=None) == [
        "2026-07-06", "2026-07-05", "2026-07-04"
    ]
    assert backfill._select_nights(nights, since=None, until="2026-07-04", limit=None) == [
        "2026-07-04", "2026-07-03"
    ]
    assert backfill._select_nights(nights, since=None, until=None, limit=2) == [
        "2026-07-06", "2026-07-05"
    ]
