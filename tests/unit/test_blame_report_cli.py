"""Unit tests for the exit status of ``.github/scripts/blame_report.py``.

The nightly workflow reads the status to tell a night with nothing to attribute
(0) from one whose attribution failed outright (1) or came back partial (2), so
a model that could not be reached cannot pass as a quiet night. The builder is
stubbed: what is under test is the coverage gate, not the attribution.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "blame_report.py"


@pytest.fixture
def blame_report():
    spec = importlib.util.spec_from_file_location("blame_report_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(blame_report, monkeypatch, tmp_path, *, entries, coverage, ranker=object()):
    """Run ``main`` over a stub builder; return ``(exit status, blame.json path)``."""
    import k4bench.blame.builder as builder
    import k4bench.blame.models as models
    import k4bench.blame.rank as rank

    report = tmp_path / "report.json"
    report.write_text(json.dumps({"generated_at": "x", "groups": []}))
    blame = SimpleNamespace(
        entries=entries, report_night="2026-10-27", to_json=lambda: {"entries": []},
    )
    monkeypatch.setattr(builder, "build_blame_report", lambda *a, **k: blame)
    monkeypatch.setattr(models, "ranking_coverage", lambda _blame: coverage)
    monkeypatch.setattr(rank, "ranker_from_env", lambda: ranker)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    status = blame_report.main([
        "--report", str(report), "--output-dir", str(tmp_path / "out"),
    ])
    return status, tmp_path / "out" / "blame.json"


_ONE_ENTRY = [SimpleNamespace(candidates=[])]


def test_nothing_to_attribute_is_a_quiet_night(blame_report, monkeypatch, tmp_path):
    status, sidecar = _run(
        blame_report, monkeypatch, tmp_path, entries=[], coverage=(0, 0, []),
    )
    assert status == 0
    assert not sidecar.exists()


def test_a_fully_ranked_night_writes_the_sidecar(blame_report, monkeypatch, tmp_path):
    status, sidecar = _run(
        blame_report, monkeypatch, tmp_path, entries=_ONE_ENTRY, coverage=(4, 4, []),
    )
    assert status == 0
    assert sidecar.is_file()


def test_a_night_the_model_ranked_nothing_of_fails_without_a_sidecar(
    blame_report, monkeypatch, tmp_path, capsys,
):
    # The 2026-10-27 night: every call came back 402, so 0 of 1080 rows ranked.
    status, sidecar = _run(
        blame_report, monkeypatch, tmp_path, entries=_ONE_ENTRY,
        coverage=(0, 1080, ["AIDASoft/DD4hep#1706"]),
    )
    assert status == blame_report.EXIT_FAILED == 1
    assert not sidecar.exists()
    assert "ranking failed — 0/1080 rankable candidate rows ranked" in capsys.readouterr().err


def test_a_partly_ranked_night_still_publishes_what_was_ranked(
    blame_report, monkeypatch, tmp_path,
):
    status, sidecar = _run(
        blame_report, monkeypatch, tmp_path, entries=_ONE_ENTRY,
        coverage=(7, 9, ["AIDASoft/DD4hep#1706", "AIDASoft/podio#1048"]),
    )
    assert status == blame_report.EXIT_PARTIAL == 2
    assert sidecar.is_file()


def test_without_a_model_the_candidates_are_written_unranked(
    blame_report, monkeypatch, tmp_path,
):
    status, sidecar = _run(
        blame_report, monkeypatch, tmp_path, entries=_ONE_ENTRY,
        coverage=(0, 9, ["x#1"]), ranker=None,
    )
    assert status == 0
    assert sidecar.is_file()
