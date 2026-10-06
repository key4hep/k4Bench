"""Detector updates against a synthetic nightly, without CVMFS or GitHub."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).parents[2] / ".github/scripts/bump_detectors.py"
spec = importlib.util.spec_from_file_location("bump_detectors", SCRIPT)
bump = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bump
spec.loader.exec_module(bump)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "repo"
    (root / ".github/benchmarks").mkdir(parents=True)
    (root / "k4bench/regression").mkdir(parents=True)
    (root / "k4bench/regression/lineage.py").write_text(
        'DETECTOR_SUCCESSORS: dict[str, str] = {\n    "IDEA_o1_v03": "IDEA_o1_v02",\n}\n'
        "RETIRED_DETECTORS: tuple[str, ...] = ()\n"
    )
    geo = tmp_path / "geo"
    fcc = tmp_path / "fcc"
    fcc.mkdir()

    def config(name):
        (root / f".github/benchmarks/{name}.yml").write_text(
            f"# {name.replace('_', ' ')}\n"
            f"xml: FCCee/IDEA/compact/{name}/{name}.xml\n"
            f"steering_file: $FCCCONFIG/{name}/SteeringFile_{name}.py\n"
        )

    def geometry(name, xml=True):
        path = geo / f"FCCee/IDEA/compact/{name}"
        path.mkdir(parents=True)
        if xml:
            (path / f"{name}.xml").touch()

    config("IDEA_o1_v03")
    for name in ["IDEA_o1_v03", "IDEA_o1_v04", "IDEA_o1_v04_CI", "CLD_o5_v99"]:
        geometry(name)
    geometry("IDEA_o1_v05", xml=False)
    return root, geo, fcc, config, geometry


def test_detection_and_steering_retention(tree):
    root, geo, fcc, _, _ = tree
    plans = bump.detect(root, geo)
    assert [(p.family, p.new) for p in plans] == [("IDEA_o1", "IDEA_o1_v04")]
    assert bump.detect(root, geo, "CLD_o5") == []
    edits, notes = bump.prepare(root, plans[0], fcc)
    text = edits[root / ".github/benchmarks/IDEA_o1_v04.yml"]
    assert "# IDEA o1 v04" in text
    assert "xml: FCCee/IDEA/compact/IDEA_o1_v04/IDEA_o1_v04.xml" in text
    assert "steering_file: $FCCCONFIG/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py" in text
    assert "retained" in notes[0]
    assert '"IDEA_o1_v03":' not in edits[root / "k4bench/regression/lineage.py"]
    assert '"IDEA_o1_v04": "IDEA_o1_v03"' in edits[root / "k4bench/regression/lineage.py"]


def test_steering_upgrade(tree):
    root, geo, fcc, _, _ = tree
    target = fcc / "IDEA_o1_v04/SteeringFile_IDEA_o1_v04.py"
    target.parent.mkdir()
    target.touch()
    plan = bump.detect(root, geo)[0]
    edits, notes = bump.prepare(root, plan, fcc)
    assert (
        "steering_file: $FCCCONFIG/IDEA_o1_v04/"
        in edits[root / ".github/benchmarks/IDEA_o1_v04.yml"]
    )


def test_side_by_side_and_numeric_versions(tree):
    root, geo, _, config, geometry = tree
    for name in ["ILD_FCCee_v01", "ILD_FCCee_v02"]:
        config(name)
    for name in ["ILD_FCCee_v03", "ILD_FCCee_v10"]:
        geometry(name)
    (plan,) = bump.detect(root, geo, "ILD_FCCee")
    assert (plan.old, plan.new) == ("ILD_FCCee_v02", "ILD_FCCee_v10")
    (root / ".github/benchmarks/SiD_v01.yml").write_text("xml: $DD4hepINSTALL/SiD.xml\n")
    assert len(bump.detect(root, geo)) == 2


def test_unexpected_lineage_fails_before_writes(tree):
    root, geo, fcc, _, _ = tree
    lineage = root / "k4bench/regression/lineage.py"
    lineage.write_text("DETECTOR_SUCCESSORS = dict()\n")
    with pytest.raises(ValueError, match="Unexpected"):
        bump.prepare(root, bump.detect(root, geo)[0], fcc)
    assert (root / ".github/benchmarks/IDEA_o1_v03.yml").exists()


def test_cli_dry_run_and_apply(tree, monkeypatch, capsys, tmp_path):
    root, geo, fcc, _, _ = tree
    monkeypatch.setattr(bump, "ROOT", root)
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    args = ["bump_detectors", "--k4geo", str(geo), "--fcc-config", str(fcc)]
    monkeypatch.setattr(sys, "argv", args)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    bump.main()
    assert json.loads(capsys.readouterr().out)[0]["new"] == "IDEA_o1_v04"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    monkeypatch.setattr(sys, "argv", args + ["--apply", "--family", "IDEA_o1"])
    bump.main()
    assert not (root / ".github/benchmarks/IDEA_o1_v03.yml").exists()
    assert (root / ".github/benchmarks/IDEA_o1_v04.yml").exists()
    assert "Geometry trees:" in (tmp_path / "bump-IDEA_o1.md").read_text()
    assert bump.detect(root, geo) == []


def test_first_lineage_entry():
    text = "DETECTOR_SUCCESSORS: dict[str, str] = {}\n"
    assert '"IDEA_o1_v04": "IDEA_o1_v03"' in bump.edit_lineage(text, "IDEA_o1_v03", "IDEA_o1_v04")


def test_dropped_lineage_entry_keeps_its_predecessor_retired():
    text = (
        'DETECTOR_SUCCESSORS: dict[str, str] = {\n    "IDEA_o1_v03": "IDEA_o1_v02",\n}\n'
        'RETIRED_DETECTORS: tuple[str, ...] = (\n    "IDEA_o1_v01",\n)\n'
    )
    namespace = {}
    exec(bump.edit_lineage(text, "IDEA_o1_v03", "IDEA_o1_v04"), namespace)
    assert namespace["DETECTOR_SUCCESSORS"] == {"IDEA_o1_v04": "IDEA_o1_v03"}
    assert namespace["RETIRED_DETECTORS"] == ("IDEA_o1_v01", "IDEA_o1_v02")
    with pytest.raises(ValueError, match="Unexpected RETIRED_DETECTORS block"):
        bump.edit_lineage(text.split("RETIRED")[0], "IDEA_o1_v03", "IDEA_o1_v04")


def test_repository_lineage_retires_on_a_second_bump():
    lineage = Path(__file__).parents[2] / "k4bench/regression/lineage.py"
    namespace = {}
    exec(bump.edit_lineage(lineage.read_text(), "ALLEGRO_o1_v04", "ALLEGRO_o1_v05"), namespace)
    assert namespace["is_detector_replaced"]("ALLEGRO_o1_v03")
    assert namespace["is_detector_replaced"]("ALLEGRO_o1_v04")
    assert not namespace["is_detector_replaced"]("ALLEGRO_o1_v05")


def test_older_steering_and_sample_overrides(tree):
    root, geo, fcc, _, _ = tree
    config = root / ".github/benchmarks/IDEA_o1_v03.yml"
    config.write_text(
        config.read_text().replace(
            "$FCCCONFIG/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py",
            "$FCCCONFIG/IDEA_o1_v02/SteeringFile_IDEA_o1_v02.py",
        )
        + "# steering: https://example.org/IDEA_o1_v02/SteeringFile_IDEA_o1_v02.py\n"
        "samples:\n  - name: sample\n"
        "    steering_file: $FCCCONFIG/IDEA_o1_v03/missing.py\n"
    )
    target = fcc / "IDEA_o1_v04/SteeringFile_IDEA_o1_v04.py"
    target.parent.mkdir()
    target.touch()
    plan = bump.detect(root, geo)[0]
    edits, notes = bump.prepare(root, plan, fcc)
    text = edits[root / ".github/benchmarks/IDEA_o1_v04.yml"]
    assert "steering_file: $FCCCONFIG/IDEA_o1_v04/SteeringFile_IDEA_o1_v04.py" in text
    assert "# steering: https://example.org/IDEA_o1_v04/SteeringFile_IDEA_o1_v04.py" in text
    assert "steering_file: $FCCCONFIG/IDEA_o1_v03/missing.py" in text


def test_bumps_never_edit_workflows(tree):
    root, geo, fcc, _, _ = tree
    workflow = root / ".github/workflows/nightly.yml"
    workflow.parent.mkdir()
    workflow.write_text(
        "# Even a stale example must not require workflow-write permission: IDEA_o1_v03\n"
    )
    script = root / ".github/scripts/nightly_benchmark.sh"
    script.parent.mkdir()
    script.write_text("# config: IDEA_o1_v03\n")
    edits, notes = bump.prepare(root, bump.detect(root, geo)[0], fcc)
    assert workflow not in edits
    assert edits[script] == "# config: IDEA_o1_v04\n"
    assert all(".github/workflows" not in path.as_posix() for path in edits)


@pytest.mark.parametrize(
    "old,new,header,expected",
    [
        ("CLD_o2_v09", "CLD_o2_v10", "CLD o2_v09", "CLD o2_v10"),
        ("ILD_FCCee_v02", "ILD_FCCee_v03", "ILD_FCCee v02", "ILD_FCCee v03"),
        ("IDEA_o1_v04", "IDEA_o1_v05", "IDEA o1 v04", "IDEA o1 v05"),
    ],
)
def test_mixed_header_separators(old, new, header, expected, tmp_path):
    plan = bump.Bump(old.rsplit("_", 1)[0], old, new, "FCCee/compact")
    text, _ = bump.edit_benchmark(f"# {header}\nxml: {old}/{old}.xml\n", plan, tmp_path)
    assert text == f"# {expected}\nxml: {new}/{new}.xml\n"


def test_retained_relative_steering_path_stays_intact(tree):
    root, geo, fcc, _, _ = tree
    plan = bump.detect(root, geo)[0]
    steering = "IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py"
    source = f"xml: IDEA_o1_v03/IDEA_o1_v03.xml\nsteering_file: {steering}\n"
    text, notes = bump.edit_benchmark(source, plan, fcc)
    assert f"steering_file: {steering}\n" in text
    assert "xml: IDEA_o1_v04/IDEA_o1_v04.xml" in text
    assert "retained" in notes[0]
