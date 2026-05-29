"""Unit tests for .github/scripts/write_run_info.py.

The script is loaded by path (the .github/scripts dir is not an importable
package) and its main() is driven with argv + env, like the workflow does.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "write_run_info.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("write_run_info", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(d: Path, label: str, returncode: int) -> None:
    (d / f"{label}_results.csv").write_text(
        f"label,returncode,n_events\n{label},{returncode},2\n"
    )


def _run(mod, results_dir: Path, monkeypatch) -> dict:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setattr(
        "sys.argv",
        ["write_run_info.py",
         "--results-dir", str(results_dir),
         "--detector", "IDEA_o1_v03", "--sample", "single_e-_10GeV",
         "--date", "2026-05-29", "--platform", "x86_64-el9-gcc14-opt",
         "--release", "2026-05-19", "--n-events", "100",
         "--sweep", "true", "--parallel", "true"],
    )
    assert mod.main() == 0
    return json.loads((results_dir / "run_info.json").read_text())


def test_status_failed_when_a_config_has_nonzero_rc(tmp_path, monkeypatch):
    mod = _load_module()
    _write_csv(tmp_path, "baseline_all", 0)
    _write_csv(tmp_path, "without_EcalBarrel", 1)
    _write_csv(tmp_path, "without_HcalBarrel", 0)

    info = _run(mod, tmp_path, monkeypatch)

    assert info["status"] == "failed"
    assert info["failed_configs"] == ["without_EcalBarrel"]
    assert set(info["configs"]) == {"baseline_all", "without_EcalBarrel", "without_HcalBarrel"}
    assert info["github_run_url"] == "https://github.com/org/repo/actions/runs/42"
    assert info["parallel"] is True


def test_status_ok_when_all_configs_pass(tmp_path, monkeypatch):
    mod = _load_module()
    _write_csv(tmp_path, "baseline_all", 0)
    _write_csv(tmp_path, "without_EcalBarrel", 0)

    info = _run(mod, tmp_path, monkeypatch)

    assert info["status"] == "ok"
    assert info["failed_configs"] == []
    assert info["k4h_release"] == "key4hep-2026-05-19"
