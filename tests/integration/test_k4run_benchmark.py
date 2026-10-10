"""Integration test for the k4run benchmark: a real k4run job, measured by the
k4BenchAuditor, run from a staged copy of its configuration directory.

Requires a Key4hep environment with k4run and the Gaudi test algorithms.
"""

from __future__ import annotations

import shutil

import pytest

from k4bench.analysis.loader import load_component_timing, load_event_timing, load_results
from k4bench.benchmark.k4run import K4runConfig, run_k4run_benchmark
from k4bench.results.reporter import save_csv

requires_k4run = pytest.mark.skipif(
    shutil.which("k4run") is None, reason="k4run is not on PATH"
)

# Sleeps --sleep seconds per event, and reads a file next to itself, so it
# runs only from a copy of its own directory.
JOB = """\

from Configurables import ApplicationMgr, GaudiTesting__SleepyAlg
from k4FWCore.parseArgs import parser

parser.add_argument("--sleep", type=int, default=0)
args, _ = parser.parse_known_args()
Path("echo.txt").write_text(Path("settings.txt").read_text())
nap = GaudiTesting__SleepyAlg("Nap", SleepTime=args.sleep)
ApplicationMgr(EvtSel="NONE", EvtMax=1, TopAlg=[nap])
"""


@pytest.mark.integration
@requires_k4run
def test_k4run_benchmark_measures_the_job_and_its_variant(tmp_path):
    stage = tmp_path / "Config"
    stage.mkdir()
    (stage / "job.py").write_text(JOB)
    (stage / "settings.txt").write_text("from the stage directory\n")
    log_dir = tmp_path / "logs"

    results = run_k4run_benchmark(K4runConfig(
        options=["job.py"],
        n_events=2,
        log_dir=log_dir,
        stage_dir=stage,
        variants={"nap": ["--sleep", "1"]},
    ))

    assert [r.label for r in results] == ["baseline", "variant_nap"], results
    assert all(r.succeeded for r in results), (log_dir / "baseline.log").read_text()[-3000:]
    assert all(r.wall_time_s and r.peak_rss_mb and r.peak_vmem_mb for r in results)
    # The job wrote echo.txt into its working directory, not into the stage
    # directory or the caller's.
    assert all(r.output_size_mb and r.output_size_mb > 0 for r in results)
    assert not (stage / "echo.txt").exists()

    events = load_event_timing(log_dir)
    assert events["baseline"]["event_number"].tolist() == [0, 1]
    assert (events["baseline"]["event_time_s"] < 0.5).all()
    assert (events["variant_nap"]["event_time_s"] > 0.99).all()

    timing = load_component_timing(log_dir)
    assert set(timing) == {"baseline", "variant_nap"}
    wall = timing["variant_nap"].execute["wall_s"]
    assert ((wall["Nap"] > 0.99) & (wall["Nap"] < 1.5)).all()
    assert "Nap.SleepTime = 1;" in (log_dir / "variant_nap_joboptions.opts").read_text()

    save_csv(results, log_dir)
    assert sorted(load_results(log_dir)["label"]) == ["baseline", "variant_nap"]


@pytest.mark.integration
@requires_k4run
def test_failed_k4run_job_is_a_failed_result(tmp_path):
    stage = tmp_path / "Config"
    stage.mkdir()
    (stage / "job.py").write_text("raise RuntimeError('broken options')\n")

    (result,) = run_k4run_benchmark(K4runConfig(
        options=["job.py"], n_events=1, log_dir=tmp_path / "logs", stage_dir=stage,
    ))

    assert not result.succeeded
    assert "broken options" in (tmp_path / "logs" / "baseline.log").read_text()
