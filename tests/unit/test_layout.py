"""Unit tests for :mod:`k4bench.layout` — the run store's directory contract.

The nightly upload writes this layout and the dashboard, the regression report
and the blame tooling read years of stored runs back through it, so the level
order and the reserved names are pinned here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from k4bench import layout
from k4bench.layout import LEVELS, iter_run_dirs, run_path, run_url, stack_dir

ROOT = Path(__file__).resolve().parents[2]

_RUN = ("CLD_o2_v09", "x86_64-almalinux9-gcc14.2.0-opt", "key4hep-2026-10-01",
        "single_e-_10GeV", "2026-10-02")


def test_run_path_spells_the_levels_outermost_first():
    assert LEVELS == ("detector", "platform", "stack", "sample", "date")
    assert run_path(*_RUN) == "/".join(_RUN)


def test_run_path_stops_at_the_deepest_level_given():
    assert run_path(*_RUN[:1]) == "CLD_o2_v09"
    assert run_path(*_RUN[:3]) == "/".join(_RUN[:3])


def test_run_path_refuses_a_level_without_its_parents():
    with pytest.raises(ValueError):
        run_path("CLD_o2_v09", None, "key4hep-2026-10-01")


def test_run_url_joins_below_the_base_whatever_its_trailing_slash():
    expected = "https://example.org/k4bench/CLD_o2_v09/x86_64-almalinux9-gcc14.2.0-opt"
    assert run_url("https://example.org/k4bench/", *_RUN[:2]) == expected
    assert run_url("https://example.org/k4bench", *_RUN[:2]) == expected


def test_stack_dir_names_a_release():
    assert stack_dir("2026-10-01") == "key4hep-2026-10-01"


def _mkrun(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    path.mkdir(parents=True)
    return path


def test_iter_run_dirs_yields_every_run_with_its_identity(tmp_path):
    path = _mkrun(tmp_path, *_RUN)

    (run,) = iter_run_dirs(tmp_path)

    assert (run.detector, run.platform, run.stack, run.sample, run.date) == _RUN
    assert run.path == path


def test_iter_run_dirs_orders_by_level_not_by_path_string(tmp_path):
    # As one string "ILD-x/…" sorts before "ILD/…" ('-' < '/'); per level,
    # "ILD" comes first.
    _mkrun(tmp_path, "ILD-x", *_RUN[1:])
    _mkrun(tmp_path, "ILD", *_RUN[1:])

    assert [run.detector for run in iter_run_dirs(tmp_path)] == ["ILD", "ILD-x"]


def test_iter_run_dirs_skips_reserved_trees_staging_dirs_and_files(tmp_path):
    _mkrun(tmp_path, *_RUN)
    _mkrun(tmp_path, "_reports", "a", "b", "c", "d")
    _mkrun(tmp_path, ".git", "a", "b", "c", "d")
    # The download cache stages a run beside its date directory until complete.
    _mkrun(tmp_path, *_RUN[:4], ".2026-10-03.tmp-abc")
    (tmp_path.joinpath(*_RUN[:4]) / "stray.txt").write_text("")

    assert [run.date for run in iter_run_dirs(tmp_path)] == [_RUN[4]]


def test_iter_run_dirs_refuses_a_missing_root(tmp_path):
    # A mistyped or unmounted store must not read as an empty one.
    with pytest.raises(NotADirectoryError):
        list(iter_run_dirs(tmp_path / "missing"))


def test_iter_run_dirs_yields_nothing_for_an_empty_store(tmp_path):
    assert list(iter_run_dirs(tmp_path)) == []


def test_nightly_upload_writes_the_layout():
    script = (ROOT / ".github" / "scripts" / "nightly_benchmark.sh").read_text()
    path = run_path(
        "${DETECTOR}", "${K4H_PLATFORM}", stack_dir("${K4H_RELEASE}"), "${SAMPLE}", "${DATE}"
    )
    assert f'EOS_RUN="${{EOS_ROOT}}/{path}"' in script


def test_layout_imports_only_leaf_modules():
    """Imported by the runner scripts, the remote layer, the report and the
    dashboard alike, so it may depend on no other k4bench layer."""
    tree = ast.parse(Path(layout.__file__).read_text())
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert {name for name in imported if name.startswith("k4bench")} <= {"k4bench.labels"}
