"""Unit tests for one nightly benchmark job (``.github/scripts/benchmark_job.py``):
resolving its record, the k4bench command it runs, and what it records."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
GEOMETRY = ROOT / "tests" / "fixtures" / "minimal_geometry" / "minimal.xml"

CI = {
    "GITHUB_RUN_ID": "123",
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "key4hep/k4Bench",
    "GITHUB_SHA": "a" * 40,
    "K4H_STACK_SETUP": "/cvmfs/view/setup.sh",
    "K4H_PLATFORM": "x86_64-el9-gcc16-opt",
    "K4H_RELEASE": "2026-10-10",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def job_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return _load("benchmark_job")


@pytest.fixture
def k4geo(tmp_path, monkeypatch):
    root = tmp_path / "k4geo"
    (root / "FCCee").mkdir(parents=True)
    (root / "FCCee" / "minimal.xml").write_text(GEOMETRY.read_text())
    monkeypatch.setenv("K4GEO", str(root))
    return root


def _record(**overrides) -> dict[str, str]:
    return {
        "config": "minimal", "sample": "single_e-_10GeV", "xml": "FCCee/minimal.xml",
        "n_events": "10", "ddsim_args": "--enableGun", "steering_file": "", "timeout": "",
        "verbose": "false", "sweep": "false", "input_files": "", "sweep_detectors": "",
        "include_only": "", "exclude_only": "", **overrides,
    }


class Processes:
    """Stands in for subprocess.run: records each command and answers with
    *returncode*, running *on_k4bench* for a k4bench command."""

    def __init__(self, returncode: int = 0, on_k4bench=lambda argv: None):
        self.calls: list[tuple[list[str], dict | None]] = []
        self.returncode = returncode
        self.on_k4bench = on_k4bench

    def __call__(self, command, *, env=None, check=False):
        self.calls.append((command, env))
        if "k4bench" in command:
            self.on_k4bench(command)
            return SimpleNamespace(returncode=self.returncode)
        return SimpleNamespace(returncode=0)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


def test_relative_geometry_is_taken_from_k4geo(job_module, k4geo):
    job = job_module.resolve(_record(), dict(os.environ))
    assert job.detector_xml == k4geo / "FCCee" / "minimal.xml"
    assert job.xml_path == "FCCee/minimal.xml"
    assert job.detector == "minimal"


def test_geometry_variables_are_expanded(job_module, k4geo, monkeypatch):
    monkeypatch.setenv("DD4hepINSTALL", str(k4geo))
    job = job_module.resolve(_record(xml="$DD4hepINSTALL/FCCee/minimal.xml"), dict(os.environ))
    # Absolute once expanded, so it is not taken relative to $K4GEO again.
    assert job.detector_xml == k4geo / "FCCee" / "minimal.xml"
    assert job.xml_path == f"{k4geo}/FCCee/minimal.xml"


def test_missing_geometry_is_a_job_error(job_module, k4geo):
    with pytest.raises(job_module.JobError, match="XML not found"):
        job_module.resolve(_record(xml="FCCee/absent.xml"), dict(os.environ))


def test_steering_file_goes_in_front_and_its_directory_on_pythonpath(job_module, k4geo, tmp_path, monkeypatch):
    steering = tmp_path / "fcc" / "steer.py"
    steering.parent.mkdir()
    steering.write_text("")
    monkeypatch.setenv("FCCCONFIG", str(steering.parent))
    monkeypatch.setenv("PYTHONPATH", "/stack/python")

    job = job_module.resolve(_record(steering_file="$FCCCONFIG/steer.py"), dict(os.environ))

    assert job.steering_path == str(steering)
    assert job.ddsim_args == f"--steeringFile {steering} --enableGun"
    assert job.pythonpath == f"{steering.parent}:/stack/python"


def test_without_a_steering_file_pythonpath_is_untouched(job_module, k4geo):
    job = job_module.resolve(_record(), dict(os.environ))
    assert job.steering_path == ""
    assert job.pythonpath is None


def test_missing_steering_file_is_a_job_error(job_module, k4geo):
    with pytest.raises(job_module.JobError, match="steering file not found"):
        job_module.resolve(_record(steering_file="/no/steer.py"), dict(os.environ))


def test_input_is_fetched_locally_and_read_from_there(job_module, k4geo, tmp_path, monkeypatch):
    steering = tmp_path / "steer.py"
    steering.write_text("")
    processes = Processes()
    monkeypatch.setattr(job_module.subprocess, "run", processes)

    job = job_module.resolve(
        _record(
            input_files="root://eospublic.cern.ch//eos/zbb/events.hepmc",
            steering_file=str(steering),
            ddsim_args="--random.seed 42",
        ),
        dict(os.environ),
    )

    assert processes.calls == [(
        ["xrdcp", "--force", "root://eospublic.cern.ch//eos/zbb/events.hepmc", "/tmp/events.hepmc"],
        None,
    )]
    assert job.ddsim_args == (
        f"--inputFiles /tmp/events.hepmc --steeringFile {steering} --random.seed 42"
    )


# ---------------------------------------------------------------------------
# k4bench command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, flags",
    [
        ({}, []),
        ({"sweep": "true"}, ["--sweep"]),
        ({"sweep_detectors": "EcalBarrel HcalBarrel"}, ["--sweep-detectors", "EcalBarrel", "HcalBarrel"]),
        ({"include_only": "EcalBarrel"}, ["--include-only", "EcalBarrel"]),
        ({"exclude_only": "Muon Yoke"}, ["--exclude-only", "Muon", "Yoke"]),
        ({"verbose": "true"}, ["--verbose"]),
    ],
)
def test_k4bench_command_carries_the_sweep_and_output_options(job_module, k4geo, overrides, flags):
    job = job_module.resolve(_record(**overrides), dict(os.environ))
    assert job.k4bench_argv(Path("logs/benchmark")) == [
        "--xml", str(k4geo / "FCCee" / "minimal.xml"),
        "--events", "10",
        "--output-dir", "logs/benchmark",
        *flags,
        # The = form, so ddsim arguments starting with -- are not read as options.
        "--ddsim-args=--enableGun",
    ]


def test_k4bench_command_omits_empty_ddsim_args(job_module, k4geo):
    job = job_module.resolve(_record(ddsim_args=""), dict(os.environ))
    assert not any(a.startswith("--ddsim-args") for a in job.k4bench_argv(Path("out")))


def test_runner_cpu_set_pins_k4bench(job_module, k4geo, monkeypatch):
    processes = Processes(returncode=4)
    monkeypatch.setattr(job_module.subprocess, "run", processes)
    job = job_module.resolve(_record(), dict(os.environ))

    assert job_module.run(job, Path("out"), {**os.environ, "RUNNER_CPU_SET": "0-3"}) == 4
    command, _ = processes.calls[0]
    assert command[:4] == ["taskset", "-c", "0-3", "k4bench"]

    job_module.run(job, Path("out"), {k: v for k, v in os.environ.items() if k != "RUNNER_CPU_SET"})
    command, _ = processes.calls[1]
    assert command[0] == "k4bench"


def test_k4bench_runs_with_the_steering_pythonpath(job_module, k4geo, tmp_path, monkeypatch):
    steering = tmp_path / "steer.py"
    steering.write_text("")
    processes = Processes()
    monkeypatch.setattr(job_module.subprocess, "run", processes)
    job = job_module.resolve(_record(steering_file=str(steering)), dict(os.environ))
    job_module.run(job, Path("out"), dict(os.environ))
    _, env = processes.calls[0]
    assert env["PYTHONPATH"].startswith(f"{tmp_path}:")


# ---------------------------------------------------------------------------
# The whole job
# ---------------------------------------------------------------------------


def _main(job_module, record, tmp_path, monkeypatch, processes) -> int:
    for key, value in CI.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("BENCHMARK_JOB", json.dumps(record))
    monkeypatch.setattr(job_module.subprocess, "run", processes)
    monkeypatch.setattr(sys.modules["run_info"], "add_stack_provenance", lambda *_: None)
    return job_module.main([str(tmp_path / "out")])


def test_job_records_its_run_and_returns_the_benchmark_exit_code(job_module, k4geo, tmp_path, monkeypatch):
    def benchmark(command):
        out = Path(command[command.index("--output-dir") + 1])
        (out / "baseline_results.csv").write_text("label\nbaseline\n")

    rc = _main(job_module, _record(), tmp_path, monkeypatch, Processes(returncode=3, on_k4bench=benchmark))

    # A failed configuration still records the run, so its results are uploaded.
    assert rc == 3
    out = tmp_path / "out"
    info = json.loads((out / "run_info.json").read_text())
    assert info["detector"] == "minimal"
    assert info["platform"] == CI["K4H_PLATFORM"]
    assert info["configs"] == ["baseline"]
    assert (out / "machine_info.json").is_file()
    assert not (out / "_machine_info_start.json").exists()


def test_job_that_cannot_start_records_nothing(job_module, k4geo, tmp_path, monkeypatch, capsys):
    processes = Processes()
    rc = _main(job_module, _record(xml="FCCee/absent.xml"), tmp_path, monkeypatch, processes)
    assert rc == 1
    assert "XML not found" in capsys.readouterr().err
    assert processes.calls == []
    assert not (tmp_path / "out" / "run_info.json").exists()


def test_failed_input_fetch_records_nothing(job_module, k4geo, tmp_path, monkeypatch):
    def failing(command, *, env=None, check=False):
        raise subprocess.CalledProcessError(54, command)

    rc = _main(job_module, _record(input_files="root://eos/x.hepmc"), tmp_path, monkeypatch, failing)
    assert rc == 1
    assert not (tmp_path / "out" / "run_info.json").exists()


def test_failed_machine_info_records_no_run(job_module, k4geo, tmp_path, monkeypatch):
    # run_info.json is what lets nightly_benchmark.sh upload, so a run whose
    # metadata could not be finalised must not leave one behind.
    def broken(_out_dir):
        raise OSError("disk full")

    monkeypatch.setattr(sys.modules["machine_info"], "cmd_finalize", broken)
    with pytest.raises(OSError):
        _main(job_module, _record(), tmp_path, monkeypatch, Processes())
    assert not (tmp_path / "out" / "run_info.json").exists()


# ---------------------------------------------------------------------------
# Contracts with the workflow, the matrix and the nightly script
# ---------------------------------------------------------------------------


def test_every_repository_record_resolves_into_a_command(job_module, monkeypatch):
    # The record is the job's only input, so each key it reads must be in every
    # record list_benchmarks.py writes.
    list_benchmarks = _load("list_benchmarks")
    records = [
        r for path in sorted((ROOT / ".github" / "benchmarks").glob("*.yml"))
        for r in list_benchmarks.expand(path)
    ]
    assert records
    for rec in records:
        job = job_module.Job(rec, rec["xml"], Path(rec["xml"]), "", rec["ddsim_args"], None)
        assert job.k4bench_argv(Path("out"))[:2] == ["--xml", rec["xml"]]


def test_workflow_hands_the_whole_record_to_the_container():
    workflow = (ROOT / ".github" / "workflows" / "benchmark-detector.yml").read_text()
    assert "BENCHMARK_JOB:    ${{ toJSON(matrix) }}" in workflow
    assert "-e BENCHMARK_JOB" in workflow


def test_nightly_uploads_under_the_recorded_detector_and_date(tmp_path):
    script = (ROOT / ".github" / "scripts" / "nightly_benchmark.sh").read_text()
    assert 'python3 .github/scripts/benchmark_job.py "${OUTPUT_DIR}"' in script
    start = script.index("read -r DETECTOR DATE")
    snippet = script[start : script.index('/run_info.json")"', start) + len('/run_info.json")"')]
    (tmp_path / "run_info.json").write_text(json.dumps({"detector": "CLD_o2_v09", "date": "2026-10-10"}))
    done = subprocess.run(
        ["bash", "-c", f'set -euo pipefail\n{snippet}\necho "$DETECTOR|$DATE"'],
        env={**os.environ, "OUTPUT_DIR": str(tmp_path)},
        capture_output=True, text=True, check=True,
    )
    assert done.stdout.strip() == "CLD_o2_v09|2026-10-10"
