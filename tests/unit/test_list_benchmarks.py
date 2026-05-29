"""Unit tests for .github/scripts/list_benchmarks.py config expansion.

Focus on the timeout key and its validation. The script is loaded by path
since .github/scripts is not an importable package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "list_benchmarks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("list_benchmarks", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_config(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "DET_o1_v01.yml"
    p.write_text(body)
    return p


def test_timeout_round_trip(tmp_path):
    mod = _load_module()
    path = _write_config(tmp_path, """
xml: some/geo.xml
sweep: true
timeout: 600
samples:
  - name: single_e-_10GeV
    n_events: 100
""")
    rec = mod.expand(path)[0]
    assert rec["timeout"] == "600"
    assert rec["sweep"] == "true"


def test_negative_timeout_is_rejected(tmp_path):
    mod = _load_module()
    path = _write_config(tmp_path, """
xml: some/geo.xml
sweep: true
timeout: -5
samples:
  - name: single_e-_10GeV
    n_events: 100
""")
    with pytest.raises(SystemExit):
        mod.expand(path)


def test_defaults_when_keys_absent(tmp_path):
    mod = _load_module()
    path = _write_config(tmp_path, """
xml: some/geo.xml
samples:
  - name: single_e-_10GeV
    n_events: 100
""")
    rec = mod.expand(path)[0]
    assert rec["timeout"] == ""
