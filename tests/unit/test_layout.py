"""Unit tests for :mod:`k4bench.layout` — the benchmark store's directory contract.

The nightly upload writes this layout and the dashboard, the regression report
and the blame tooling read years of stored runs back through it, so the level
order and the names of every kind of data are pinned here.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from k4bench import layout
from k4bench.layout import (
    LEVELS, RECO, SIM, SUITES, RunRef, concept_of, concept_path, input_path, iter_run_dirs,
    report_path, reproducer_path, run_dir_of, run_path, run_url, stack_dir,
)

ROOT = Path(__file__).resolve().parents[2]

_RUN = ("CLD_o2_v09", "x86_64-almalinux9-gcc14.2.0-opt", "key4hep-2026-10-01",
        "single_e-_10GeV", "2026-10-02")
#: Where :data:`_RUN` lives in the sim suite.
_SIM_DIR = ("runs", "sim", "CLD", *_RUN)


def test_suites_are_sim_and_reco():
    assert SUITES == (SIM, RECO) == ("sim", "reco")


@pytest.mark.parametrize("detector, concept", [
    ("ALLEGRO_o1_v04", "ALLEGRO"),
    ("CLD_o2_v09", "CLD"),
    ("IDEA_o2_v01", "IDEA"),
    ("ILD_FCCee_v01", "ILD"),
    ("SiD", "SiD"),
])
def test_a_detector_belongs_to_the_concept_its_name_starts_with(detector, concept):
    assert concept_of(detector) == concept


def test_run_path_spells_the_levels_outermost_first():
    assert LEVELS == ("suite", "concept", "detector", "platform", "stack", "sample", "date")
    assert run_path(SIM, *_RUN) == "/".join(_SIM_DIR)


def test_run_path_stops_at_the_deepest_level_given():
    assert run_path(SIM) == "runs/sim"
    assert run_path(RECO, *_RUN[:1]) == "runs/reco/CLD/CLD_o2_v09"
    assert run_path(SIM, *_RUN[:3]) == "/".join(_SIM_DIR[:6])


def test_run_path_refuses_a_level_without_its_parents():
    with pytest.raises(ValueError):
        run_path(SIM, "CLD_o2_v09", None, "key4hep-2026-10-01")


def test_run_path_refuses_an_unknown_suite():
    with pytest.raises(ValueError, match="unknown suite"):
        run_path("ddsim", *_RUN)


def test_a_glob_detector_spans_every_concept():
    assert run_path(SIM, "*", _RUN[1]) == f"runs/sim/*/*/{_RUN[1]}"


def test_run_url_joins_below_the_base_whatever_its_trailing_slash():
    expected = "https://example.org/k4bench/runs/sim/CLD/CLD_o2_v09/x86_64-almalinux9-gcc14.2.0-opt"
    assert run_url("https://example.org/k4bench/", SIM, *_RUN[:2]) == expected
    assert run_url("https://example.org/k4bench", SIM, *_RUN[:2]) == expected


def test_every_kind_of_data_is_split_by_suite():
    assert concept_path(SIM, "CLD") == "runs/sim/CLD"
    assert report_path(SIM) == "reports/sim"
    assert report_path(SIM, "2026-10-14") == "reports/sim/2026-10-14"
    assert reproducer_path(SIM) == "reproducers/sim"
    assert input_path(RECO, "CLD_o2_v09") == "inputs/reco/CLD/CLD_o2_v09"


def test_stack_dir_names_a_release():
    assert stack_dir("2026-10-01") == "key4hep-2026-10-01"


def test_run_dir_of_files_a_recorded_run_under_its_suite():
    info = {
        "suite": RECO, "tool": "k4run", "detector": _RUN[0], "platform": _RUN[1],
        "k4h_release": _RUN[2], "sample": _RUN[3], "date": _RUN[4],
    }
    assert run_dir_of(info) == run_path(RECO, *_RUN)


def _mkrun(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    path.mkdir(parents=True)
    return path


def test_iter_run_dirs_yields_every_run_with_its_identity(tmp_path):
    path = _mkrun(tmp_path, *_SIM_DIR)

    (run,) = iter_run_dirs(tmp_path, SIM)

    assert (run.suite, run.detector, run.platform, run.stack, run.sample, run.date) == (
        SIM, *_RUN
    )
    assert run.path == path
    # Its identity, without where this store keeps it.
    assert RunRef(SIM, *_RUN).store_path == "/".join(_SIM_DIR)
    assert isinstance(run, RunRef)


def test_a_run_ref_names_one_run_of_one_night():
    # Every detector and release of a night shares its date: none of them is
    # the same run.
    ref = RunRef(SIM, *_RUN)
    assert ref != RunRef(SIM, "ALLEGRO_o1_v04", *_RUN[1:])
    assert ref != RunRef(SIM, _RUN[0], _RUN[1], "key4hep-2026-09-30", *_RUN[3:])
    assert ref != RunRef(RECO, *_RUN)
    assert len({ref, RunRef(SIM, *_RUN)}) == 1


def test_iter_run_dirs_orders_by_level_not_by_path_string(tmp_path):
    # As one string "ILD-x/…" sorts before "ILD/…" ('-' < '/'); per level,
    # "ILD" comes first.
    _mkrun(tmp_path, "runs", "sim", "ILD-x", "ILD-x", *_RUN[1:])
    _mkrun(tmp_path, "runs", "sim", "ILD", "ILD", *_RUN[1:])

    assert [run.detector for run in iter_run_dirs(tmp_path, SIM)] == ["ILD", "ILD-x"]


def test_iter_run_dirs_reads_one_suite_only(tmp_path):
    _mkrun(tmp_path, *_SIM_DIR)
    _mkrun(tmp_path, "runs", "reco", "CLD", *_RUN[:3], "p8_ee_Zbb_ecm91", _RUN[4])

    assert [run.sample for run in iter_run_dirs(tmp_path, SIM)] == [_RUN[3]]
    assert [run.sample for run in iter_run_dirs(tmp_path, RECO)] == ["p8_ee_Zbb_ecm91"]


def test_iter_run_dirs_skips_staging_dirs_and_files(tmp_path):
    _mkrun(tmp_path, *_SIM_DIR)
    # The download cache stages a run beside its date directory until complete.
    _mkrun(tmp_path, *_SIM_DIR[:-1], ".2026-10-03.tmp-abc")
    (tmp_path.joinpath(*_SIM_DIR[:-1]) / "stray.txt").write_text("")

    assert [run.date for run in iter_run_dirs(tmp_path, SIM)] == [_RUN[4]]


def test_iter_run_dirs_skips_a_detector_filed_under_another_concept(tmp_path, caplog):
    _mkrun(tmp_path, *_SIM_DIR)
    _mkrun(tmp_path, "runs", "sim", "IDEA", *_RUN)

    with caplog.at_level(logging.WARNING, logger="k4bench.layout"):
        runs = list(iter_run_dirs(tmp_path, SIM))

    assert [run.path for run in runs] == [tmp_path.joinpath(*_SIM_DIR)]
    assert "outside its concept CLD" in caplog.text


def test_iter_run_dirs_refuses_a_missing_root_or_suite_tree(tmp_path):
    # A mistyped or unmounted store must not read as an empty one.
    with pytest.raises(NotADirectoryError):
        list(iter_run_dirs(tmp_path / "missing", SIM))
    with pytest.raises(NotADirectoryError):
        list(iter_run_dirs(tmp_path, SIM))


def test_iter_run_dirs_yields_nothing_for_an_empty_suite_tree(tmp_path):
    (tmp_path / "runs" / "sim").mkdir(parents=True)
    assert list(iter_run_dirs(tmp_path, SIM)) == []


def test_nightly_upload_writes_the_layout():
    # The upload path is read off the recorded run by run_dir_of, so the writer
    # cannot spell the layout differently from its readers.
    script = (ROOT / ".github" / "scripts" / "nightly_benchmark.sh").read_text()
    assert "from k4bench.layout import run_dir_of" in script
    assert 'EOS_RUN="${EOS_ROOT}/${RUN_DIR}"' in script


def test_report_upload_writes_the_layout():
    script = (ROOT / ".github" / "scripts" / "regression_report.sh").read_text()
    assert "from k4bench.layout import SIM, report_path, reproducer_path" in script
    # Split from an assignment, so a failed lookup aborts under `set -e`
    # instead of leaving both paths empty.
    assert 'read -r REPORT_DIR REPRODUCER_DIR <<< "${LAYOUT_DIRS}"' in script
    assert 'EOS_REPORT_DIR="${EOS_ROOT}/${REPORT_DIR}"' in script
    assert '--reproducer-dir "root://${EOS_FQDN}/${EOS_ROOT}/${REPRODUCER_DIR}"' in script


def test_layout_imports_only_leaf_modules():
    """Imported by the runner scripts, the remote layer, the report and the
    dashboard alike, so it may depend on no other k4bench layer."""
    tree = ast.parse(Path(layout.__file__).read_text())
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert {name for name in imported if name.startswith("k4bench")} <= {"k4bench.labels"}
