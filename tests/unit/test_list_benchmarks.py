"""Unit tests for the nightly matrix expansion (``.github/scripts/list_benchmarks.py``)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / ".github" / "scripts" / "list_benchmarks.py"
_RUN_INFO = ROOT / ".github" / "scripts" / "run_info.py"


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


def test_repository_records_satisfy_the_run_info_contract(list_benchmarks, monkeypatch, capsys):
    # nightly_benchmark.sh passes SWEEP straight to run_info.py, whose --sweep
    # only accepts "true" or "false".
    parser = _load("run_info_writer", _RUN_INFO)._parser()
    items = _main(list_benchmarks, ROOT / ".github" / "benchmarks", monkeypatch, capsys)
    for item in items:
        assert {item["sweep"], item["verbose"]} <= {"true", "false"}, item["config"]
        args = parser.parse_args([
            "out", "--detector=d", "--sample=s", "--date=d", "--platform=p",
            "--release=r", "--n-events=1", "--detector-xml=x",
            f"--sweep={item['sweep']}",
        ])
        assert args.sweep == item["sweep"]


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
