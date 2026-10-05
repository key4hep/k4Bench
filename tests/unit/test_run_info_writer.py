"""Unit tests for the nightly ``run_info.json`` writer (``.github/scripts/run_info.py``)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / ".github" / "scripts" / "run_info.py"
GEOMETRY = ROOT / "tests" / "fixtures" / "minimal_geometry" / "minimal.xml"

ENV = {
    "GITHUB_RUN_ID": "123",
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "key4hep/k4Bench",
    "GITHUB_SHA": "a" * 40,
    "K4H_STACK_SETUP": "/cvmfs/view/setup.sh",
    "VERBOSE": "false",
    "RUNNER_CPU_SET": "0-3",
}


@pytest.fixture
def run_info():
    spec = importlib.util.spec_from_file_location("run_info_writer", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def nightly_env(monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)


def _argv(output_dir: Path, *extra: str) -> list[str]:
    return [
        str(output_dir),
        "--detector=minimal",
        "--sample=single_e-_10GeV",
        "--date=2026-10-05",
        "--platform=x86_64-almalinux9-gcc14.2.0-opt",
        "--release=2026-10-04",
        "--n-events=10",
        f"--detector-xml={GEOMETRY}",
        "--xml-path=minimal.xml",
        *extra,
    ]


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


def test_record_carries_job_parameters_and_ci_identity(run_info, nightly_env, tmp_path, monkeypatch):
    monkeypatch.setattr(run_info, "add_stack_provenance", lambda *_: None)
    (tmp_path / "baseline_results.csv").write_text("label\nbaseline\n")
    (tmp_path / "no_EcalBarrel_results.csv").write_text("label\nno_EcalBarrel\n")

    assert run_info.main(_argv(
        tmp_path,
        "--ddsim-args=--random.seed 42 --enableGun",
        "--input-files=root://eos/a.hepmc root://eos/b.hepmc",
        "--sweep=true",
    )) == 0

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["k4h_release"] == "key4hep-2026-10-04"
    assert info["k4h_release_date"] == "2026-10-04"
    assert info["k4h_stack_setup"] == ENV["K4H_STACK_SETUP"]
    assert info["github_run_url"] == "https://github.com/key4hep/k4Bench/actions/runs/123"
    assert info["commit_sha"] == ENV["GITHUB_SHA"]
    assert info["n_events"] == 10
    assert info["sweep"] is True
    assert info["ddsim_args"] == "--random.seed 42 --enableGun"
    assert info["random_seed"] == 42
    assert info["verbose"] is False
    assert info["runner_cpu_set"] == "0-3"
    assert info["input_files"] == ["root://eos/a.hepmc", "root://eos/b.hepmc"]
    assert info["configs"] == ["baseline", "no_EcalBarrel"]
    assert info["configured_labels"] == [
        "baseline", "no_InnerTracker", "no_OuterTracker", "no_EcalBarrel", "no_HcalBarrel",
    ]


@pytest.mark.parametrize(
    "extra, expected",
    [
        ((), ["baseline"]),
        (("--sweep-detectors=EcalBarrel",), ["baseline", "no_EcalBarrel"]),
        (("--include-only=EcalBarrel HcalBarrel",), ["only_EcalBarrel_HcalBarrel"]),
        (("--exclude-only=EcalBarrel",), ["no_EcalBarrel"]),
    ],
)
def test_configured_labels_follow_the_sweep_options(run_info, extra, expected, tmp_path):
    args = run_info._parser().parse_args(_argv(tmp_path, *extra))
    assert run_info.configured_labels(args) == expected


def test_unresolvable_roster_is_recorded_as_unknown(run_info, nightly_env, tmp_path, monkeypatch, capsys):
    from k4bench.benchmark import ddsim

    def broken(*_args):
        raise RuntimeError("geometry unreadable")

    monkeypatch.setattr(ddsim, "planned_config_labels", broken)
    monkeypatch.setattr(run_info, "add_stack_provenance", lambda *_: None)
    assert run_info.main(_argv(tmp_path, "--sweep=true")) == 0
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["configured_labels"] is None
    assert info["configs"] == []
    assert "could not resolve configured labels" in capsys.readouterr().err


def test_stack_provenance_is_read_from_the_resolved_view(run_info, nightly_env, tmp_path, monkeypatch):
    from k4bench.provenance import stack

    seen = []

    def read_stack(setup):
        seen.append(setup)
        return Path("/cvmfs/view/manifest.json"), {"DD4hep": "abc"}

    monkeypatch.setattr(stack, "read_stack", read_stack)
    monkeypatch.setenv("KEY4HEP_STACK", "/somewhere/else/setup.sh")
    assert run_info.main(_argv(tmp_path)) == 0
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert seen == [ENV["K4H_STACK_SETUP"]]
    assert info["k4h_stack_manifest"] == "/cvmfs/view/manifest.json"
    assert info["k4h_packages"] == {"DD4hep": "abc"}


def test_unreadable_stack_provenance_does_not_fail_the_job(run_info, nightly_env, tmp_path, monkeypatch):
    from k4bench.provenance import stack

    def read_stack(_setup):
        raise OSError("view vanished")

    monkeypatch.setattr(stack, "read_stack", read_stack)
    assert run_info.main(_argv(tmp_path)) == 0
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert "k4h_packages" not in info


def test_nightly_script_writes_run_info_through_this_script():
    script = (ROOT / ".github" / "scripts" / "nightly_benchmark.sh").read_text()
    assert "python3 .github/scripts/run_info.py" in script
    # Values starting with "--" are only safe from option parsing in the
    # --option=value spelling.
    assert '--ddsim-args="${DDSIM_ARGS}"' in script
