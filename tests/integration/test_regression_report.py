"""End-to-end smoke test for ``.github/scripts/regression_report.py`` and the
``blame_report.py`` sidecar it feeds.

Builds a tiny synthetic run-dir history with the EOS layout and runs the real
CLIs against it (local ``--data-dir`` mode, no network), asserting the shape of
the written ``report.json``/``report.md`` and ``blame.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"
_SCRIPT = _SCRIPTS / "regression_report.py"
_BLAME_SCRIPT = _SCRIPTS / "blame_report.py"
_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_STACK = "key4hep-2026-01-01"


def _write_run(run_dir: Path, night: str, wall_time_s: float) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "platform": _PLAT, "k4h_release": _STACK, "sample": "single_e",
    }))
    (run_dir / "baseline_results.csv").write_text(
        "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        f"baseline,0,10,{wall_time_s},1024.0,{wall_time_s * 0.98},{10.0 / wall_time_s}\n"
    )
    (run_dir / "machine_info.json").write_text(json.dumps({
        "hostname": "host-a", "cpu_physical_cores": 8, "cpu_logical_cores": 16,
        "load_avg_1m_start": 0.5, "load_avg_1m_end": 0.5,
        "ram_total_gb": 64.0, "ram_available_gb_start": 32.0,
        "ram_available_gb_end": 32.0, "swap_in_pages": 0, "swap_out_pages": 0,
        "thermal_throttle_events": 0,
    }))


def test_regression_report_cli_local_mode(tmp_path):
    # 10 steady nights, then a persisting +20% step on the last two → one
    # confirmed wall-time regression in tonight's report.
    walls = [100.0, 100.4, 99.6, 100.2, 99.8, 100.3, 99.7, 100.1, 99.9, 100.0,
             120.0, 120.5]
    sample_root = tmp_path / "data" / "DET" / _PLAT / _STACK / "single_e"
    d0 = date.fromisoformat("2026-01-01")
    for i, wall in enumerate(walls):
        night = (d0 + timedelta(days=i)).isoformat()
        _write_run(sample_root / night, night, wall)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT),
         "--data-dir", str(tmp_path / "data"), "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    data = json.loads((out_dir / "report.json").read_text())
    summary = data["summary"]
    assert summary["report_night"] == "2026-01-12"
    assert summary["n_detectors"] == 1
    assert summary["has_alertable"] is True
    assert summary["n_regressions"] >= 1
    confirmed = [
        v for g in data["groups"] for v in g["verdicts"]
        if v["severity"] == "CONFIRMED" and v["direction"] == "UP"
    ]
    assert any(v["metric"] == "wall_time_s" for v in confirmed)

    md = (out_dir / "report.md").read_text()
    assert "k4Bench nightly regression report — 2026-01-12" in md
    assert "🔴 Regression" in md


_K4GEO = "https://github.com/key4hep/k4geo.git"
_DD4HEP = "https://github.com/AIDASoft/DD4hep.git"


def _write_run_with_provenance(
    run_dir: Path, night: str, stack: str, wall_time_s: float, k4geo_commit: str
) -> None:
    """Like :func:`_write_run` but under an explicit release *stack* and carrying
    a ``k4h_packages`` map — so a step across a release boundary produces a real
    ``(baseline, onset]`` blame window over changed provenance."""
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "platform": _PLAT, "k4h_release": stack, "sample": "single_e",
        "k4h_packages": {
            "k4geo": {"commit": k4geo_commit, "version": "develop", "repo_url": _K4GEO},
            "dd4hep": {"commit": "d" * 40, "version": "develop", "repo_url": _DD4HEP},
        },
    }))
    (run_dir / "baseline_results.csv").write_text(
        "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        f"baseline,0,10,{wall_time_s},1024.0,{wall_time_s * 0.98},{10.0 / wall_time_s}\n"
    )
    (run_dir / "machine_info.json").write_text(json.dumps({
        "hostname": "host-a", "cpu_physical_cores": 8, "cpu_logical_cores": 16,
        "load_avg_1m_start": 0.5, "load_avg_1m_end": 0.5,
        "ram_total_gb": 64.0, "ram_available_gb_start": 32.0,
        "ram_available_gb_end": 32.0, "swap_in_pages": 0, "swap_out_pages": 0,
        "thermal_throttle_events": 0,
    }))


def test_blame_report_cli_local_mode(tmp_path):
    # 10 steady nights on release 2026-01-01 (k4geo commit A), then a persisting
    # +20% step on two nights that measured a *new* release, 2026-01-11 (k4geo
    # commit C). The confirmed wall-time regression's window is therefore
    # (2026-01-01, 2026-01-11], across which only k4geo moved.
    data_dir = tmp_path / "data"
    d0 = date.fromisoformat("2026-01-01")
    for i in range(10):
        night = (d0 + timedelta(days=i)).isoformat()
        wall = 100.0 + (0.4 if i % 2 else -0.4)
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / "key4hep-2026-01-01" / "single_e" / night,
            night, "key4hep-2026-01-01", wall, "a" * 40,
        )
    for i, night in enumerate(("2026-01-11", "2026-01-12")):
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / "key4hep-2026-01-11" / "single_e" / night,
            night, "key4hep-2026-01-11", 120.0 + i * 0.5, "c" * 40,
        )

    out_dir = tmp_path / "out"
    assert subprocess.run(
        [sys.executable, str(_SCRIPT), "--data-dir", str(data_dir),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    ).returncode == 0

    # No GITHUB_TOKEN in the environment → diffs only, no network. Explicitly
    # stripped so a token set in CI can't turn this offline test into a live one.
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    result = subprocess.run(
        [sys.executable, str(_BLAME_SCRIPT),
         "--report", str(out_dir / "report.json"),
         "--output-dir", str(out_dir),
         "--data-dir", str(data_dir)],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr

    blame = json.loads((out_dir / "blame.json").read_text())
    assert blame["report_night"] == "2026-01-12"
    entries = [e for e in blame["entries"] if e["metric"] == "wall_time_s"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["base_release"] == "2026-01-01"
    assert entry["onset_release"] == "2026-01-11"
    assert entry["n_unchanged"] == 1  # dd4hep held still
    repos = {r["package"]: r for r in entry["repos"]}
    assert set(repos) == {"k4geo"}    # only k4geo moved
    assert repos["k4geo"]["base_commit"] == "a" * 40
    assert repos["k4geo"]["head_commit"] == "c" * 40
    assert repos["k4geo"]["compare_url"]  # diff link recorded even without a token
    assert repos["k4geo"]["candidates"] == []  # no PRs resolved without GitHub


def test_blame_report_removes_stale_sidecar_when_nothing_is_attributable(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"generated_at": "x", "groups": []}))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "blame.json"
    stale.write_text('{"stale": true}')
    env = {
        k: v for k, v in os.environ.items()
        if k not in {
            "GITHUB_TOKEN", "K4BENCH_LLM_URL", "K4BENCH_LLM_MODEL",
            "K4BENCH_LLM_API_KEY",
        }
    }
    result = subprocess.run(
        [sys.executable, str(_BLAME_SCRIPT), "--report", str(report_path),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert not stale.exists()


def _load_script(path):
    """Import a ``.github/scripts`` CLI by file path. Registering it in
    ``sys.modules`` before exec is required so its module-level ``@dataclass``
    can resolve its own module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_blame_report_refuses_incomplete_configured_ranking(tmp_path, monkeypatch):
    from k4bench.blame import builder as builder_mod
    from k4bench.blame import rank as rank_mod
    from k4bench.blame.models import BlameEntry, BlameReport, CandidatePR, RepoBlame

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"generated_at": "x", "groups": []}))
    incomplete = BlameReport("g", "2026-01-12", entries=(BlameEntry(
        detector="DET", platform=_PLAT, sample="single_e", label="baseline",
        metric="wall_time_s", sub_detector=None, base_release="2026-01-01",
        onset_release="2026-01-11", repos=(RepoBlame(
            package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
            head_commit="c" * 40, compare_url=None, status="changed",
            candidates=(CandidatePR(
                repo="key4hep/k4geo", number=1, title="PR", author="alice",
                url="u", score=90.0, description="",
            ),),
        ),),
    ),))
    monkeypatch.setattr(builder_mod, "build_blame_report", lambda *a, **k: incomplete)
    monkeypatch.setattr(rank_mod, "ranker_from_env", lambda: object())
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    blame_cli = _load_script(_BLAME_SCRIPT)
    out_dir = tmp_path / "out"
    assert blame_cli.main([
        "--report", str(report_path), "--output-dir", str(out_dir),
    ]) == 1
    assert not (out_dir / "blame.json").exists()


def test_blame_report_with_ranker_over_local_tree(tmp_path, monkeypatch):
    # Same tree as above, but this time we run the *builder* in-process with a
    # fake GitHub resolution and a fake ranker — proving the whole chain (real
    # report build → real provenance read from the tree → diff → ranking →
    # serialized blame.json carries per-PR scores) with no network.
    data_dir = tmp_path / "data"
    d0 = date.fromisoformat("2026-01-01")
    for i in range(10):
        night = (d0 + timedelta(days=i)).isoformat()
        wall = 100.0 + (0.4 if i % 2 else -0.4)
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / "key4hep-2026-01-01" / "single_e" / night,
            night, "key4hep-2026-01-01", wall, "a" * 40,
        )
    for i, night in enumerate(("2026-01-11", "2026-01-12")):
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / "key4hep-2026-01-11" / "single_e" / night,
            night, "key4hep-2026-01-11", 120.0 + i * 0.5, "c" * 40,
        )

    out_dir = tmp_path / "out"
    assert subprocess.run(
        [sys.executable, str(_SCRIPT), "--data-dir", str(data_dir),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    ).returncode == 0

    from k4bench.blame import builder as builder_mod
    from k4bench.blame.builder import build_blame_report
    from k4bench.blame.github import GitHubClient, RepoResolution
    from k4bench.blame.models import CandidatePR
    from k4bench.blame.rank import Ranking
    from k4bench.regression.render import from_json as report_from_json

    # Provenance from the local tree, via the real CLI helper (no network).
    blame_cli = _load_script(_BLAME_SCRIPT)
    packages_for_release = blame_cli._make_packages_for_release(
        [str(data_dir)], None, {_PLAT: ["DET"]}
    )

    def fake_resolve(client, slug, base, head):
        return RepoResolution(
            candidates=[CandidatePR(
                repo=slug, number=1234, title="Lower the step limit", author="alice",
                url=f"https://github.com/{slug}/pull/1234",
            )],
            patches={1234: "@@ -1 +1 @@\n+ more steps"},
        )
    monkeypatch.setattr(builder_mod, "resolve_repo_prs", fake_resolve)

    class _FakeRanker:
        def rank(self, request):
            scored = {
                (c.repo, c.number): Ranking(88.0, "raises the tracker step count")
                for c in request.candidates
            }
            scored[("key4hep/ghost", 999)] = Ranking(100.0, "invented")  # must vanish
            return scored

    report = report_from_json(json.loads((out_dir / "report.json").read_text()))
    blame = build_blame_report(
        report, packages_for_release=packages_for_release,
        github=GitHubClient(), ranker=_FakeRanker(),
    )
    (out_dir / "blame.json").write_text(json.dumps(blame.to_json(), indent=2))

    data = json.loads((out_dir / "blame.json").read_text())
    entry = next(e for e in data["entries"] if e["metric"] == "wall_time_s")
    cand = entry["repos"][0]["candidates"][0]
    assert cand["number"] == 1234
    assert cand["score"] == 88.0
    assert cand["description"] == "raises the tracker step count"
    assert "patch" not in cand  # the transient diff is never persisted
    all_numbers = {
        c["number"] for e in data["entries"] for r in e["repos"] for c in r["candidates"]
    }
    assert 999 not in all_numbers  # the invented PR was dropped
