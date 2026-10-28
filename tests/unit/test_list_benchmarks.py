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


def _write(bench_dir: Path, name: str, xml: str, *samples: str, suite: str = "sim") -> Path:
    """A config for detector *name*, where list_benchmarks.py looks for it."""
    entries = "\n".join(f"  - name: {s}\n    n_events: 10" for s in samples)
    path = bench_dir / suite / name.split("_", 1)[0] / f"{name}.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"xml: {xml}\nsamples:\n{entries}\n")
    return path


def _main(list_benchmarks, bench_dir: Path, monkeypatch, capsys, *args: str) -> list[dict]:
    monkeypatch.setattr(list_benchmarks, "BENCH_DIR", bench_dir)
    monkeypatch.setattr(sys, "argv", ["list_benchmarks.py", *args])
    list_benchmarks.main()
    return json.loads(capsys.readouterr().out)


def _refused(list_benchmarks, bench_dir: Path, monkeypatch, capsys, *args: str) -> str:
    with pytest.raises(SystemExit):
        _main(list_benchmarks, bench_dir, monkeypatch, capsys, *args)
    return capsys.readouterr().err


def test_repository_benchmarks_expand(list_benchmarks, monkeypatch, capsys):
    items = _main(list_benchmarks, ROOT / ".github" / "benchmarks", monkeypatch, capsys)
    assert items
    assert all(item["config"] and item["sample"] for item in items)


def test_repository_records_satisfy_the_benchmark_job_contract(list_benchmarks, monkeypatch, capsys):
    # benchmark_job.py reads a boolean as the string "true"; anything else is false.
    items = _main(list_benchmarks, ROOT / ".github" / "benchmarks", monkeypatch, capsys)
    for item in items:
        assert {item["sweep"], item["verbose"]} <= {"true", "false"}, item["config"]


def test_a_config_is_identified_by_its_path(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "CLD_o2_v09", "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml", "s1")
    (rec,) = _main(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert (rec["config"], rec["suite"], rec["detector"]) == (
        "sim/CLD/CLD_o2_v09", "sim", "CLD_o2_v09",
    )


def test_a_config_outside_suite_and_concept_is_refused(list_benchmarks, tmp_path, monkeypatch, capsys):
    (tmp_path / "A.yml").write_text("xml: A.xml\nsamples:\n  - name: s\n    n_events: 1\n")
    assert "outside {suite}/{concept}/" in _refused(list_benchmarks, tmp_path, monkeypatch, capsys)


def test_a_config_in_no_suite_is_refused(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "A", "A.xml", "s", suite="ddsim")
    assert "is in no suite" in _refused(list_benchmarks, tmp_path, monkeypatch, capsys)


def test_a_config_under_another_concept_is_refused(list_benchmarks, tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, "IDEA_o1_v04", "IDEA_o1_v04.xml", "s")
    misfiled = tmp_path / "sim" / "CLD" / path.name
    misfiled.parent.mkdir()
    path.rename(misfiled)
    assert "belongs in sim/IDEA/" in _refused(list_benchmarks, tmp_path, monkeypatch, capsys)


def test_a_config_named_after_another_compact_file_is_refused(list_benchmarks, tmp_path, monkeypatch, capsys):
    # The run tree is named after the compact file the job loads.
    _write(tmp_path, "A_steered", "FCCee/A/A.xml", "s")
    assert "must be named after its compact file: A.yml" in _refused(
        list_benchmarks, tmp_path, monkeypatch, capsys
    )


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
    path = _write(tmp_path, "A", "FCCee/A/A.xml", "single_e-_10GeV")
    path.write_text(lines + path.read_text())
    [item] = _main(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert item["sweep"] == expected
    assert item["verbose"] == "false"


@pytest.mark.parametrize("value", ["maybe", "1", "''"])
def test_non_boolean_sweep_is_refused(list_benchmarks, tmp_path, monkeypatch, capsys, value):
    path = _write(tmp_path, "A", "FCCee/A/A.xml", "single_e-_10GeV")
    path.write_text(f"sweep: {value}\n" + path.read_text())
    assert "sweep must be true or false" in _refused(list_benchmarks, tmp_path, monkeypatch, capsys)


def test_configs_on_distinct_geometries_may_share_a_sample(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "A", "FCCee/A/A.xml", "single_e-_10GeV")
    _write(tmp_path, "B", "FCCee/B/B.xml", "single_e-_10GeV")
    items = _main(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert [(i["config"], i["sample"]) for i in items] == [
        ("sim/A/A", "single_e-_10GeV"), ("sim/B/B", "single_e-_10GeV"),
    ]


def test_a_sample_defined_twice_is_refused(list_benchmarks, tmp_path, monkeypatch, capsys):
    # Both would upload into one run directory, the second replacing the first.
    _write(tmp_path, "A", "FCCee/A/A.xml", "single_e-_10GeV", "single_e-_10GeV")
    assert "config 'sim/A/A' defines sample 'single_e-_10GeV' twice" in _refused(
        list_benchmarks, tmp_path, monkeypatch, capsys
    )


@pytest.mark.parametrize(
    "only, expected",
    [
        ("", [("reco/A/A", "s1"), ("sim/A/A", "s1"), ("sim/A/A", "s2"), ("sim/B/B", "s1")]),
        ("sim", [("sim/A/A", "s1"), ("sim/A/A", "s2"), ("sim/B/B", "s1")]),
        ("sim/A", [("sim/A/A", "s1"), ("sim/A/A", "s2")]),
        ("sim/A/A/s2 sim/B", [("sim/A/A", "s2"), ("sim/B/B", "s1")]),
        ("  sim/B/B/s1/  reco ", [("reco/A/A", "s1"), ("sim/B/B", "s1")]),
    ],
)
def test_only_keeps_the_jobs_below_the_named_prefixes(list_benchmarks, tmp_path, monkeypatch, capsys, only, expected):
    _write(tmp_path, "A", "FCCee/A/A.xml", "s1", "s2")
    _write(tmp_path, "B", "FCCee/B/B.xml", "s1")
    path = _write(tmp_path, "A", "FCCee/A/A.xml", "s1", suite="reco")
    path.write_text("options: Reco.py\n" + path.read_text())
    items = _main(list_benchmarks, tmp_path, monkeypatch, capsys, "--only", only)
    assert [(i["config"], i["sample"]) for i in items] == expected


def test_only_groups_the_selected_jobs(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "A", "FCCee/A/A.xml", "s1", "s2")
    _write(tmp_path, "B", "FCCee/B/B.xml", "s1")
    groups = _main(
        list_benchmarks, tmp_path, monkeypatch, capsys, "--grouped", "--only", "sim/A/A/s1",
    )
    assert [(g["config"], [s["sample"] for s in g["samples"]]) for g in groups] == [
        ("sim/A/A", ["s1"]),
    ]


@pytest.mark.parametrize(
    "only, expected",
    [("", ["sim", "reco"]), ("sim/B", ["sim"]), ("reco sim/A/A/s2", ["sim", "reco"]), ("reco", ["reco"])],
)
def test_suites_names_each_suite_the_selected_jobs_benchmark(
    list_benchmarks, tmp_path, monkeypatch, capsys, only, expected,
):
    # The nightly reports on these suites, so a suite none of its jobs
    # benchmarked must not be named: its report would read as an outage.
    _write(tmp_path, "A", "FCCee/A/A.xml", "s1", "s2")
    _write(tmp_path, "B", "FCCee/B/B.xml", "s1")
    path = _write(tmp_path, "A", "FCCee/A/A.xml", "s1", suite="reco")
    path.write_text("options: Reco.py\n" + path.read_text())
    assert _main(list_benchmarks, tmp_path, monkeypatch, capsys, "--suites", "--only", only) == expected


@pytest.mark.parametrize("only", ["sim/C", "sim/A/A/s3", "s1", "A", "sim/A/A/s1/x"])
def test_only_refuses_a_selector_matching_no_job(list_benchmarks, tmp_path, monkeypatch, capsys, only):
    _write(tmp_path, "A", "FCCee/A/A.xml", "s1")
    err = _refused(list_benchmarks, tmp_path, monkeypatch, capsys, "--only", f"sim/A {only}")
    assert f"no benchmark job matches {only!r}" in err


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
    path = tmp_path / "reco" / "CLD" / "CLD_o2_v09.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return _main(list_benchmarks, tmp_path, monkeypatch, capsys)


def test_k4run_record_carries_the_job(list_benchmarks, tmp_path, monkeypatch, capsys):
    (rec,) = _k4run(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert (rec["config"], rec["suite"], rec["tool"]) == ("reco/CLD/CLD_o2_v09", "reco", "k4run")
    assert rec["options"] == "CLDReconstruction.py"
    assert rec["stage_dir"] == "$CLDCONFIG/share/CLDConfig"
    # Like ddsim_args, k4run_args concatenates top-level and sample-level.
    assert rec["k4run_args"] == "--compactFile $DETECTOR_XML --cms 91"
    assert rec["variants"] == {"truth_tracking": "--truthTracking"}
    assert rec["input_files"] == "https://example.org/sim.edm4hep.root"
    assert rec["ddsim_args"] == "" and rec["sweep"] == "false"


def test_k4run_data_files_are_joined_like_input_files(list_benchmarks, tmp_path, monkeypatch, capsys):
    text = _K4RUN.replace(
        "options: CLDReconstruction.py\n",
        "options: CLDReconstruction.py\n"
        "data_files:\n  - https://example.org/a.root\n  - https://example.org/config.tar.gz\n",
    )
    (rec,) = _k4run(list_benchmarks, tmp_path, monkeypatch, capsys, text)
    assert rec["data_files"] == "https://example.org/a.root https://example.org/config.tar.gz"


def test_k4run_memory_events_default_to_none_and_samples_override(list_benchmarks, tmp_path, monkeypatch, capsys):
    (rec,) = _k4run(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert rec["memory_events"] == ""
    text = _K4RUN.replace("options: CLDReconstruction.py\n", "options: CLDReconstruction.py\nmemory_events: 20\n")
    (rec,) = _k4run(list_benchmarks, tmp_path, monkeypatch, capsys, text)
    assert rec["memory_events"] == "20"
    (rec,) = _k4run(list_benchmarks, tmp_path, monkeypatch, capsys, text + "    memory_events: 5\n")
    assert rec["memory_events"] == "5"


def test_a_config_without_tool_runs_its_suites_tool(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "SiD", "SiD.xml", "single_e-_10GeV")
    (sim,) = _main(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert sim["tool"] == "ddsim"
    assert sim["options"] == "" and sim["k4run_args"] == "" and sim["variants"] == {}
    assert sim["data_files"] == ""

    (reco,) = _k4run(list_benchmarks, tmp_path / "reco_only", monkeypatch, capsys,
                     _K4RUN.replace("tool: k4run\n", ""))
    assert reco["tool"] == "k4run"


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
        lambda t: t.replace("tool: k4run", "tool: k4run\nmemory_events: 0"),
        lambda t: t.replace("tool: k4run", "tool: k4run\nmemory_events: ten"),
    ],
    ids=[
        "unknown-tool", "sweep", "ddsim_args", "steering_file", "no-options", "no-xml",
        "variant-without-args", "variant-name", "duplicate-variant", "variants-not-a-list",
        "no-memory-events", "memory-events-not-a-number",
    ],
)
def test_malformed_k4run_config_is_refused(list_benchmarks, tmp_path, monkeypatch, capsys, edit):
    with pytest.raises(SystemExit):
        _k4run(list_benchmarks, tmp_path, monkeypatch, capsys, edit(_K4RUN))


@pytest.mark.parametrize("key", ["options: Reco.py", "memory_events: 10"])
def test_k4run_keys_are_refused_in_a_ddsim_config(list_benchmarks, tmp_path, monkeypatch, capsys, key):
    path = _write(tmp_path, "SiD", "SiD.xml", "s")
    path.write_text(f"{key}\n" + path.read_text())
    with pytest.raises(SystemExit):
        _main(list_benchmarks, tmp_path, monkeypatch, capsys)


def test_reco_and_sim_of_one_detector_are_distinct_configs(list_benchmarks, tmp_path, monkeypatch, capsys):
    _write(tmp_path, "CLD_o2_v09", "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml", "p8_ee_Zbb_ecm91")
    items = _k4run(list_benchmarks, tmp_path, monkeypatch, capsys)
    assert sorted((i["config"], i["tool"], i["sample"]) for i in items) == [
        ("reco/CLD/CLD_o2_v09", "k4run", "p8_ee_Zbb_ecm91"),
        ("sim/CLD/CLD_o2_v09", "ddsim", "p8_ee_Zbb_ecm91"),
    ]
