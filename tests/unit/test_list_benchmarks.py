"""Unit tests for the nightly matrix expansion (``.github/scripts/list_benchmarks.py``)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / ".github" / "scripts" / "list_benchmarks.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def list_benchmarks():
    return _load("list_benchmarks", _SCRIPT)


def _write(bench_dir: Path, name: str, xml: str, *samples: str) -> None:
    entries = "\n".join(f"  - name: {s}\n    n_events: 10" for s in samples)
    (bench_dir / f"{name}.yml").write_text(f"xml: {xml}\nsamples:\n{entries}\n")


def _main(list_benchmarks, bench_dir: Path, monkeypatch, capsys, *args: str) -> list[dict]:
    monkeypatch.setattr(list_benchmarks, "BENCH_DIR", bench_dir)
    monkeypatch.setattr(sys, "argv", ["list_benchmarks.py", *args])
    list_benchmarks.main()
    return json.loads(capsys.readouterr().out)


def test_repository_benchmarks_expand(list_benchmarks, monkeypatch, capsys):
    items = _main(list_benchmarks, ROOT / ".github" / "benchmarks", monkeypatch, capsys)
    assert items
    assert all(item["config"] and item["sample"] for item in items)


def test_repository_records_satisfy_the_benchmark_job_contract(list_benchmarks, monkeypatch, capsys):
    # benchmark_job.py reads a boolean as the string "true"; anything else is false.
    items = _main(list_benchmarks, ROOT / ".github" / "benchmarks", monkeypatch, capsys)
    for item in items:
        assert {item["sweep"], item["verbose"]} <= {"true", "false"}, item["config"]


@pytest.mark.parametrize(
    "lines, expected",
    [
        ("", "false"),
        ("sweep: true\n", "true"),
        ("sweep: False\n", "false"),
        ("sweep: 'true'\n", "true"),
    ],
)
def test_sweep_is_always_true_or_false(list_benchmarks, tmp_path, monkeypatch, capsys, lines, expected):
    _write(tmp_path, "A", "FCCee/A/A.xml", "single_e-_10GeV")
    path = tmp_path / "A.yml"
    path.write_text(lines + path.read_text())
    [item] = _main(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert item["sweep"] == expected
    assert item["verbose"] == "false"


@pytest.mark.parametrize("value", ["maybe", "1", "''"])
def test_non_boolean_sweep_is_refused(list_benchmarks, tmp_path, monkeypatch, capsys, value):
    _write(tmp_path, "A", "FCCee/A/A.xml", "single_e-_10GeV")
    path = tmp_path / "A.yml"
    path.write_text(f"sweep: {value}\n" + path.read_text())
    with pytest.raises(SystemExit):
        _main(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert "sweep must be true or false" in capsys.readouterr().err


def test_configs_on_distinct_geometries_may_share_a_sample(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "A", "FCCee/A/A.xml", "single_e-_10GeV")
    _write(tmp_path, "B", "FCCee/B/B.xml", "single_e-_10GeV")
    items = _main(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert [(i["config"], i["sample"]) for i in items] == [
        ("A", "single_e-_10GeV"), ("B", "single_e-_10GeV"),
    ]


def test_one_geometry_may_be_benchmarked_with_distinct_samples(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "A", "FCCee/A/A.xml", "single_e-_10GeV")
    _write(tmp_path, "A_steered", "FCCee/A/A.xml", "steered_e-_10GeV")
    assert len(_main(list_benchmarks, tmp_path, monkeypatch, capsys)) == 2


def test_two_configs_uploading_to_one_eos_directory_are_refused(list_benchmarks, tmp_path, monkeypatch, capsys):
    # The detector directory is the compact file's basename, so a second config
    # on the same geometry (even via another path) would overwrite the first.
    _write(tmp_path, "A", "FCCee/A/A.xml", "single_e-_10GeV")
    _write(tmp_path, "A_steered", "$OTHER/A.xml", "single_e-_10GeV")
    monkeypatch.setattr(list_benchmarks, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["list_benchmarks.py"])
    with pytest.raises(SystemExit):
        list_benchmarks.main()
    err = capsys.readouterr().err
    assert "'A' and 'A_steered'" in err
    assert "'single_e-_10GeV'" in err


@pytest.mark.parametrize(
    "only, expected",
    [
        ("", [("A", "s1"), ("A", "s2"), ("B", "s1")]),
        ("A", [("A", "s1"), ("A", "s2")]),
        ("A/s2 B", [("A", "s2"), ("B", "s1")]),
        ("  B/s1  ", [("B", "s1")]),
    ],
)
def test_only_keeps_the_named_jobs(list_benchmarks, tmp_path, monkeypatch, capsys, only, expected):
    _write(tmp_path, "A", "FCCee/A/A.xml", "s1", "s2")
    _write(tmp_path, "B", "FCCee/B/B.xml", "s1")
    items = _main(list_benchmarks, tmp_path, monkeypatch, capsys, "--only", only)
    assert [(i["config"], i["sample"]) for i in items] == expected


def test_only_groups_the_selected_jobs(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "A", "FCCee/A/A.xml", "s1", "s2")
    _write(tmp_path, "B", "FCCee/B/B.xml", "s1")
    groups = _main(list_benchmarks, tmp_path, monkeypatch, capsys, "--grouped", "--only", "A/s1")
    assert [(g["detector"], [s["sample"] for s in g["samples"]]) for g in groups] == [("A", ["s1"])]


@pytest.mark.parametrize("only", ["C", "A/s3", "s1"])
def test_only_refuses_a_name_matching_no_job(list_benchmarks, tmp_path, monkeypatch, capsys, only):
    _write(tmp_path, "A", "FCCee/A/A.xml", "s1")
    with pytest.raises(SystemExit):
        _main(list_benchmarks, tmp_path, monkeypatch, capsys, "--only", f"A {only}")
    assert f"no benchmark job matches {only!r}" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# k4run benchmarks
# ---------------------------------------------------------------------------

_K4RUN = """\
tool: k4run
xml: FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml
stage_dir: $CLDCONFIG/share/CLDConfig
options: CLDReconstruction.py
k4run_args: --compactFile $DETECTOR_XML
variants:
  - name: truth_tracking
    k4run_args: --truthTracking
samples:
  - name: p8_ee_Zbb_ecm91
    n_events: 100
    input_files: https://example.org/sim.edm4hep.root
    k4run_args: --cms 91
"""


def _k4run(list_benchmarks, tmp_path, monkeypatch, capsys, text: str = _K4RUN) -> list[dict]:
    (tmp_path / "CLD_o2_v09_reco.yml").write_text(text)
    return _main(list_benchmarks, tmp_path, monkeypatch, capsys)


def test_k4run_record_carries_the_job(list_benchmarks, tmp_path, monkeypatch, capsys):
    (rec,) = _k4run(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert rec["tool"] == "k4run"
    assert rec["options"] == "CLDReconstruction.py"
    assert rec["stage_dir"] == "$CLDCONFIG/share/CLDConfig"
    # Like ddsim_args, k4run_args concatenates top-level and sample-level.
    assert rec["k4run_args"] == "--compactFile $DETECTOR_XML --cms 91"
    assert rec["variants"] == {"truth_tracking": "--truthTracking"}
    assert rec["input_files"] == "https://example.org/sim.edm4hep.root"
    assert rec["ddsim_args"] == "" and rec["sweep"] == "false"


def test_a_config_without_tool_is_ddsim(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "SiD", "SiD.xml", "single_e-_10GeV")
    (rec,) = _main(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert rec["tool"] == "ddsim"
    assert rec["options"] == "" and rec["k4run_args"] == "" and rec["variants"] == {}


def test_sample_level_variants_replace_the_top_level_ones(list_benchmarks, tmp_path, monkeypatch, capsys):
    text = _K4RUN + "    variants:\n      - name: native\n        k4run_args: --native\n"
    (rec,) = _k4run(list_benchmarks, tmp_path, monkeypatch, capsys, text)
    assert rec["variants"] == {"native": "--native"}


@pytest.mark.parametrize(
    "edit",
    [
        lambda t: t.replace("tool: k4run", "tool: gaudirun"),
        # A key of the other tool is an error, not silently ignored.
        lambda t: t.replace("tool: k4run", "tool: k4run\nsweep: true"),
        lambda t: t.replace("tool: k4run", "tool: k4run\nddsim_args: --enableGun"),
        lambda t: t.replace("tool: k4run", "tool: k4run\nsteering_file: steer.py"),
        lambda t: t.replace("options: CLDReconstruction.py\n", ""),
        lambda t: t.replace("xml: FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml\n", ""),
        lambda t: t.replace("    k4run_args: --truthTracking\n", ""),
        lambda t: t.replace("name: truth_tracking", "name: truth tracking"),
        lambda t: t.replace("    k4run_args: --truthTracking\n",
                            "    k4run_args: --truthTracking\n"
                            "  - name: truth_tracking\n    k4run_args: --x\n"),
        lambda t: t.replace("variants:\n", "variants: --truthTracking\nunused:\n"),
    ],
    ids=[
        "unknown-tool", "sweep", "ddsim_args", "steering_file", "no-options", "no-xml",
        "variant-without-args", "variant-name", "duplicate-variant", "variants-not-a-list",
    ],
)
def test_malformed_k4run_config_is_refused(list_benchmarks, tmp_path, monkeypatch, capsys, edit):
    with pytest.raises(SystemExit):
        _k4run(list_benchmarks, tmp_path, monkeypatch, capsys, edit(_K4RUN))


def test_k4run_keys_are_refused_in_a_ddsim_config(list_benchmarks, tmp_path, monkeypatch, capsys):
    (tmp_path / "SiD.yml").write_text(
        "xml: SiD.xml\noptions: Reco.py\nsamples:\n  - name: s\n    n_events: 1\n"
    )
    with pytest.raises(SystemExit):
        _main(list_benchmarks, tmp_path, monkeypatch, capsys)


def test_reco_and_sim_of_one_sample_upload_to_distinct_trees(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "CLD_o2_v09", "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml", "p8_ee_Zbb_ecm91")
    items = _k4run(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert sorted((i["tool"], i["sample"]) for i in items) == [
        ("ddsim", "p8_ee_Zbb_ecm91"), ("k4run", "p8_ee_Zbb_ecm91"),
    ]


def test_two_k4run_configs_uploading_to_one_eos_directory_are_refused(list_benchmarks, tmp_path, monkeypatch, capsys):
    (tmp_path / "CLD_o2_v09_reco2.yml").write_text(_K4RUN)
    with pytest.raises(SystemExit):
        _k4run(list_benchmarks, tmp_path, monkeypatch, capsys)
