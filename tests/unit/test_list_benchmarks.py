"""Unit tests for the nightly matrix expansion (``.github/scripts/list_benchmarks.py``)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / ".github" / "scripts" / "list_benchmarks.py"


@pytest.fixture
def list_benchmarks():
    spec = importlib.util.spec_from_file_location("list_benchmarks", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(bench_dir: Path, name: str, xml: str, *samples: str) -> None:
    entries = "\n".join(f"  - name: {s}\n    n_events: 10" for s in samples)
    (bench_dir / f"{name}.yml").write_text(f"xml: {xml}\nsamples:\n{entries}\n")


def _main(list_benchmarks, bench_dir: Path, monkeypatch, capsys) -> list[dict]:
    monkeypatch.setattr(list_benchmarks, "BENCH_DIR", bench_dir)
    monkeypatch.setattr(sys, "argv", ["list_benchmarks.py"])
    list_benchmarks.main()
    return json.loads(capsys.readouterr().out)


def test_repository_benchmarks_expand(list_benchmarks, monkeypatch, capsys):
    items = _main(list_benchmarks, ROOT / ".github" / "benchmarks", monkeypatch, capsys)
    assert items
    assert all(item["config"] and item["sample"] for item in items)


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
