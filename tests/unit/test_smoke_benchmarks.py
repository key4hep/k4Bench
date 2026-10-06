"""Check smoke-test selection and failure handling without a simulation stack."""

import importlib.util
from pathlib import Path
import subprocess

import pytest

SCRIPTS = Path(__file__).parents[2] / ".github/scripts"


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "smoke_benchmarks", SCRIPTS / "smoke_benchmarks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_overrides_are_covered_and_shared_pairs_run_once(smoke, tmp_path):
    config = tmp_path / "IDEA_o1_v04.yml"
    config.write_text("""xml: old.xml
steering_file: common.py
n_events: 100
samples:
  - name: gun
  - name: hepmc
    input_files: root://example.org/events.hepmc
  - name: other_steering
    steering_file: other.py
  - name: other_geometry
    xml: other.xml
""")
    assert smoke.pairs([config]) == {
        ("old.xml", "common.py"): "IDEA_o1_v04/gun",
        ("old.xml", "other.py"): "IDEA_o1_v04/other_steering",
        ("other.xml", "common.py"): "IDEA_o1_v04/other_geometry",
    }


@pytest.fixture
def paths(tmp_path, monkeypatch):
    geometry = tmp_path / "geometry.xml"
    geometry.touch()
    steering = tmp_path / "steering.py"
    steering.touch()
    monkeypatch.setenv("K4GEO", str(tmp_path))
    monkeypatch.setenv("FCCCONFIG", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/existing")
    return geometry, steering


def test_nightly_path_resolution_and_sibling_imports(smoke, paths, tmp_path):
    geometry, steering = paths
    args, env = smoke.command("geometry.xml", "$FCCCONFIG/steering.py", tmp_path / "out.root")
    assert args[args.index("--compactFile") + 1] == str(geometry)
    assert args[args.index("--steeringFile") + 1] == str(steering)
    assert env["PYTHONPATH"] == f"{tmp_path}:/existing"
    assert args[args.index("--numberOfEvents") + 1] == "1"
    assert "--enableGun" in args and "--inputFiles" not in args
    args, env = smoke.command(str(geometry), "", tmp_path / "out.root")
    assert "--steeringFile" not in args
    assert env["PYTHONPATH"] == "/existing"


def test_missing_steering_is_a_failure(smoke, paths, tmp_path):
    with pytest.raises(ValueError, match="Missing steering"):
        smoke.command("geometry.xml", "$FCCCONFIG/missing.py", tmp_path / "out.root")


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing_output", "empty_output"])
def test_failed_simulation_never_passes(smoke, paths, monkeypatch, failure):
    def run(args, **kwargs):
        assert kwargs["timeout"] == 300 and kwargs["check"]
        if failure == "exit":
            raise subprocess.CalledProcessError(1, args)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 300)
        if failure == "empty_output":
            Path(args[args.index("--outputFile") + 1]).touch()

    monkeypatch.setattr(smoke.subprocess, "run", run)
    with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError)):
        smoke.smoke("geometry.xml", "$FCCCONFIG/steering.py")
