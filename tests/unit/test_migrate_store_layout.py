"""Unit tests for the flat-to-suite store cutover (``.github/scripts/migrate_store_layout.py``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from k4bench.layout import RECO, SIM, input_path, iter_run_dirs, report_path, reproducer_path

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / ".github" / "scripts" / "migrate_store_layout.py"
_RUN = ("x86_64-el9-gcc14-opt", "key4hep-2026-10-01", "single_e-_10GeV", "2026-10-02")
_INPUT = "p8_ee_Zbb_ecm91_100ev_seed42.edm4hep.root"


@pytest.fixture
def migrate():
    spec = importlib.util.spec_from_file_location("migrate_store_layout", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _file(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def flat_store(tmp_path: Path) -> Path:
    """A store in the flat layout that predates suites."""
    for detector in ("CLD_o2_v09", "ILD_FCCee_v01", "SiD"):
        _file(tmp_path.joinpath(detector, *_RUN, "run_info.json"), detector)
    _file(tmp_path.joinpath("_k4run", "CLD_o2_v09", *_RUN, "run_info.json"), "reco")
    _file(tmp_path / "_reports" / "2026-10-02" / "report.json", "{}")
    _file(tmp_path / "_inputs" / "CLD_o2_v09" / _INPUT)
    _file(tmp_path / "_reproducers" / "recipe.txt", "recipe")
    _file(tmp_path / "_scratch" / "note.txt")
    return tmp_path


def test_a_flat_store_is_refiled_where_the_suite_readers_look(migrate, flat_store):
    assert migrate.main([str(flat_store), "--apply"]) == 0

    sim = list(iter_run_dirs(flat_store, SIM))
    assert [(r.detector, r.platform, r.stack, r.sample, r.date) for r in sim] == [
        (d, *_RUN) for d in ("CLD_o2_v09", "ILD_FCCee_v01", "SiD")
    ]
    assert (sim[0].path / "run_info.json").read_text() == "CLD_o2_v09"
    assert [run.detector for run in iter_run_dirs(flat_store, RECO)] == ["CLD_o2_v09"]
    assert (flat_store / report_path(SIM, "2026-10-02") / "report.json").is_file()
    assert (flat_store / input_path(RECO, "CLD_o2_v09") / _INPUT).is_file()
    assert (flat_store / reproducer_path(SIM) / "recipe.txt").read_text() == "recipe"
    # Recipes stay at their old place too, for the links already posted;
    # unknown reserved names are left alone.
    assert (flat_store / "_reproducers" / "recipe.txt").is_file()
    assert (flat_store / "_scratch" / "note.txt").is_file()
    for moved in ("CLD_o2_v09", "SiD", "_k4run/CLD_o2_v09", "_reports/2026-10-02", "_inputs/CLD_o2_v09"):
        assert not (flat_store / moved).exists()


def test_a_dry_run_changes_nothing(migrate, flat_store, capsys):
    before = sorted(p.relative_to(flat_store) for p in flat_store.rglob("*"))
    assert migrate.main([str(flat_store)]) == 0
    assert sorted(p.relative_to(flat_store) for p in flat_store.rglob("*")) == before
    out = capsys.readouterr().out
    assert "move SiD -> runs/sim/SiD/SiD" in out
    assert "move _inputs/CLD_o2_v09 -> inputs/reco/CLD/CLD_o2_v09" in out
    assert "keep _scratch" in out


def test_a_completed_migration_has_nothing_left_to_do(migrate, flat_store, capsys):
    assert migrate.main([str(flat_store), "--apply"]) == 0
    capsys.readouterr()
    assert migrate.main([str(flat_store), "--apply"]) == 0
    assert "Nothing to migrate." in capsys.readouterr().out


def test_an_existing_destination_refuses_the_whole_plan(migrate, flat_store, capsys):
    _file(flat_store / report_path(SIM, "2026-10-02") / "report.json", "new")
    assert migrate.main([str(flat_store), "--apply"]) == 1
    assert "destination exists" in capsys.readouterr().err
    assert (flat_store / "SiD").is_dir()
    assert (flat_store / report_path(SIM, "2026-10-02") / "report.json").read_text() == "new"


def test_the_reco_config_fetches_its_input_from_where_the_migration_files_it():
    config = yaml.safe_load((ROOT / ".github/benchmarks/reco/CLD/CLD_o2_v09.yml").read_text())
    url = config["samples"][0]["input_files"]
    assert url.endswith(f"/{input_path(RECO, 'CLD_o2_v09')}/{_INPUT}")
