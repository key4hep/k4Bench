"""Unit tests for the nightly ``run_info.json`` record (``.github/scripts/run_info.py``)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
GEOMETRY = ROOT / "tests" / "fixtures" / "minimal_geometry" / "minimal.xml"

ENV = {
    "GITHUB_RUN_ID": "123",
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "key4hep/k4Bench",
    "GITHUB_SHA": "a" * 40,
    "K4H_STACK_SETUP": "/cvmfs/view/setup.sh",
    "K4H_PLATFORM": "x86_64-almalinux9-gcc14.2.0-opt",
    "K4H_RELEASE": "2026-10-04",
    "RUNNER_CPU_SET": "0-3",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_info(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return _load("run_info")


@pytest.fixture
def job_type(run_info):
    return _load("benchmark_job").Job


def _job(job_type, **record):
    rec = {
        "config": "minimal", "sample": "single_e-_10GeV", "xml": "minimal.xml",
        "n_events": "10", "ddsim_args": "", "steering_file": "", "timeout": "",
        "verbose": "false", "sweep": "false", "input_files": "", "sweep_detectors": "",
        "include_only": "", "exclude_only": "", **record,
    }
    return job_type(
        record=rec,
        xml_path=rec["xml"],
        detector_xml=GEOMETRY,
        steering_path="",
        ddsim_args=rec["ddsim_args"],
        pythonpath=None,
    )


def _write(run_info, job, tmp_path) -> dict:
    run_info.write_run_info(tmp_path, job, ENV, date="2026-10-05")
    return json.loads((tmp_path / "run_info.json").read_text())


@pytest.mark.parametrize(
    "args, seed",
    [
        ("", None),
        ("--enableGun", None),
        ("--random.seed 42 --enableGun", 42),
        ("--random.seed=7", 7),
        # A sample-level seed overrides the detector-level one ddsim saw first.
        ("--random.seed 42 --enableGun --random.seed 9", 9),
        ("--random.seed notanumber", None),
        ("--random.seed", None),
    ],
)
def test_random_seed_is_the_last_one_ddsim_reads(run_info, args, seed):
    assert run_info.random_seed(args) == seed


def test_record_carries_job_parameters_and_ci_identity(run_info, job_type, tmp_path, monkeypatch):
    monkeypatch.setattr(run_info, "add_stack_provenance", lambda *_: None)
    (tmp_path / "baseline_results.csv").write_text("label\nbaseline\n")
    (tmp_path / "no_EcalBarrel_results.csv").write_text("label\nno_EcalBarrel\n")
    job = _job(
        job_type,
        ddsim_args="--random.seed 42 --enableGun",
        input_files="root://eos/a.hepmc root://eos/b.hepmc",
        sweep="true",
        verbose="true",
    )

    info = _write(run_info, job, tmp_path)

    assert info["date"] == "2026-10-05"
    assert info["platform"] == ENV["K4H_PLATFORM"]
    assert info["k4h_release"] == "key4hep-2026-10-04"
    assert info["k4h_release_date"] == "2026-10-04"
    assert info["k4h_stack_setup"] == ENV["K4H_STACK_SETUP"]
    assert info["detector"] == "minimal"
    assert info["sample"] == "single_e-_10GeV"
    assert info["github_run_url"] == "https://github.com/key4hep/k4Bench/actions/runs/123"
    assert info["commit_sha"] == ENV["GITHUB_SHA"]
    assert info["n_events"] == 10
    assert info["sweep"] is True
    assert info["ddsim_args"] == "--random.seed 42 --enableGun"
    assert info["random_seed"] == 42
    assert info["verbose"] is True
    assert info["runner_cpu_set"] == "0-3"
    assert info["input_files"] == ["root://eos/a.hepmc", "root://eos/b.hepmc"]
    assert info["configs"] == ["baseline", "no_EcalBarrel"]
    assert info["configured_labels"] == [
        "baseline", "no_InnerTracker", "no_OuterTracker", "no_EcalBarrel", "no_HcalBarrel",
    ]


def test_record_keeps_configured_and_resolved_paths_apart(run_info, job_type, tmp_path, monkeypatch):
    monkeypatch.setattr(run_info, "add_stack_provenance", lambda *_: None)
    job = dataclasses.replace(
        _job(job_type, xml="$K4GEO/minimal.xml", steering_file="$FCCCONFIG/steer.py"),
        xml_path="/k4geo/minimal.xml",
        steering_path="/fcc/steer.py",
        ddsim_args="--inputFiles /tmp/a.hepmc --steeringFile /fcc/steer.py --enableGun",
    )

    info = _write(run_info, job, tmp_path)

    assert info["configured_xml_path"] == "$K4GEO/minimal.xml"
    assert info["xml_path"] == "/k4geo/minimal.xml"
    assert info["steering_file"] == "$FCCCONFIG/steer.py"
    assert info["resolved_steering_file"] == "/fcc/steer.py"
    # ddsim_args name the /tmp copy ddsim read.
    assert info["ddsim_args"] == "--inputFiles /tmp/a.hepmc --steeringFile /fcc/steer.py --enableGun"


@pytest.mark.parametrize(
    "record, expected",
    [
        ({}, ["baseline"]),
        ({"sweep_detectors": "EcalBarrel"}, ["baseline", "no_EcalBarrel"]),
        ({"include_only": "EcalBarrel HcalBarrel"}, ["only_EcalBarrel_HcalBarrel"]),
        ({"exclude_only": "EcalBarrel"}, ["no_EcalBarrel"]),
    ],
)
def test_configured_labels_follow_the_sweep_options(run_info, job_type, record, expected):
    assert run_info.configured_labels(_job(job_type, **record)) == expected


def test_unresolvable_roster_is_recorded_as_unknown(run_info, job_type, tmp_path, monkeypatch, capsys):
    from k4bench.benchmark import ddsim

    def broken(*_args):
        raise RuntimeError("geometry unreadable")

    monkeypatch.setattr(ddsim, "planned_config_labels", broken)
    monkeypatch.setattr(run_info, "add_stack_provenance", lambda *_: None)
    info = _write(run_info, _job(job_type, sweep="true"), tmp_path)
    assert info["configured_labels"] is None
    assert info["configs"] == []
    assert "could not resolve configured labels" in capsys.readouterr().err


def test_stack_provenance_is_read_from_the_resolved_view(run_info, job_type, tmp_path, monkeypatch):
    from k4bench.provenance import stack

    seen = []

    def read_stack(setup):
        seen.append(setup)
        return Path("/cvmfs/view/manifest.json"), {"DD4hep": "abc"}

    monkeypatch.setattr(stack, "read_stack", read_stack)
    monkeypatch.setenv("KEY4HEP_STACK", "/somewhere/else/setup.sh")
    info = _write(run_info, _job(job_type), tmp_path)
    assert seen == [ENV["K4H_STACK_SETUP"]]
    assert info["k4h_stack_manifest"] == "/cvmfs/view/manifest.json"
    assert info["k4h_packages"] == {"DD4hep": "abc"}


def test_unreadable_stack_provenance_does_not_fail_the_job(run_info, job_type, tmp_path, monkeypatch):
    from k4bench.provenance import stack

    def read_stack(_setup):
        raise OSError("view vanished")

    monkeypatch.setattr(stack, "read_stack", read_stack)
    info = _write(run_info, _job(job_type), tmp_path)
    assert "k4h_packages" not in info


def test_ddsim_record_names_its_tool(run_info, job_type, tmp_path, monkeypatch):
    monkeypatch.setattr(run_info, "add_stack_provenance", lambda *_: None)
    info = _write(run_info, _job(job_type, tool="ddsim"), tmp_path)
    assert next(iter(info)) == "tool" and info["tool"] == "ddsim"
    assert "k4run_args" not in info


def test_k4run_record_carries_the_job_and_its_roster(run_info, job_type, tmp_path, monkeypatch):
    monkeypatch.setattr(run_info, "add_stack_provenance", lambda *_: None)
    (tmp_path / "baseline_results.csv").write_text("label\nbaseline\n")
    job = dataclasses.replace(
        _job(
            job_type, tool="k4run", options="CLDReconstruction.py",
            stage_dir="$CLDCONFIG/share/CLDConfig",
            input_files="https://example.org/sim.root",
        ),
        stage_path="/cvmfs/cldconfig/share/CLDConfig",
        k4run_args="--inputFiles /tmp/k4bench-inputs/sim.root --cms 91",
        variants={"truth_tracking": "--truthTracking"},
    )

    info = _write(run_info, job, tmp_path)

    assert info["tool"] == "k4run"
    assert info["options"] == ["CLDReconstruction.py"]
    assert info["stage_dir"] == "$CLDCONFIG/share/CLDConfig"
    assert info["resolved_stage_dir"] == "/cvmfs/cldconfig/share/CLDConfig"
    assert info["k4run_args"] == "--inputFiles /tmp/k4bench-inputs/sim.root --cms 91"
    assert info["variants"] == {"truth_tracking": "--truthTracking"}
    assert info["input_files"] == ["https://example.org/sim.root"]
    assert info["random_seed"] is None
    assert info["configs"] == ["baseline"]
    # A killed job's missing variant is then a missing config, as for a sweep.
    assert info["configured_labels"] == ["baseline", "variant_truth_tracking"]
