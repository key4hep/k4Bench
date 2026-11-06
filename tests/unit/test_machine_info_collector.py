"""Unit tests for the host identity recorded by ``machine_info.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "machine_info.py"


@pytest.fixture
def machine_info():
    spec = importlib.util.spec_from_file_location("machine_info_collector", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_explicit_host_node_is_used_inside_a_container(monkeypatch, machine_info):
    monkeypatch.setenv("K4BENCH_HOST_NODE", "bench01.cern.ch")
    monkeypatch.setattr(machine_info, "_in_container", lambda: True)
    monkeypatch.setattr(machine_info.os, "uname", lambda: pytest.fail("uname fallback used"))
    assert machine_info._hostname() == "bench01.cern.ch"


def test_container_without_a_host_override_reports_host_unknown(monkeypatch, machine_info):
    monkeypatch.delenv("K4BENCH_HOST_NODE", raising=False)
    monkeypatch.setattr(machine_info, "_in_container", lambda: True)
    monkeypatch.setattr(machine_info.os, "uname", lambda: pytest.fail("container id used"))
    assert machine_info._hostname() == ""


def test_bare_metal_falls_back_to_its_own_nodename(monkeypatch, machine_info):
    class _Uname:
        nodename = "bench02"

    monkeypatch.delenv("K4BENCH_HOST_NODE", raising=False)
    monkeypatch.setattr(machine_info, "_in_container", lambda: False)
    monkeypatch.setattr(machine_info.os, "uname", lambda: _Uname())
    assert machine_info._hostname() == "bench02"


_PROC_STAT = """\
cpu  1000 0 100 9000 0 0 0 40 0 0
cpu0 600 0 50 4300 0 0 0 0 0 0
cpu1 100 0 10 870 0 0 0 20 0 0
cpu2 300 0 40 3830 0 0 0 20 0 0
intr 12345
"""


def _proc_stat(machine_info, monkeypatch, text):
    real = machine_info._read
    monkeypatch.setattr(
        machine_info, "_read", lambda path, default="": text if path == "/proc/stat" else real(path, default),
    )


@pytest.mark.parametrize(
    "cpus, expected",
    [([], (40, 10140)), ([1, 2], (40, 5190)), ([0], (0, 4950)), ([7], None)],
)
def test_steal_jiffies_sum_the_requested_cpus(machine_info, monkeypatch, cpus, expected):
    _proc_stat(machine_info, monkeypatch, _PROC_STAT)
    assert machine_info._steal_jiffies(cpus) == expected


@pytest.mark.parametrize(
    "text, expected", [("2-3", [2, 3]), ("0,4-5", [0, 4, 5]), ("", []), ("a-b", [])],
)
def test_parse_cpu_list(machine_info, text, expected):
    assert machine_info._parse_cpu_list(text) == expected


def test_steal_is_the_share_of_the_runners_cpus_over_the_run(machine_info, monkeypatch, tmp_path):
    # The runner's CPUs 1-2 lose 30 of their 1000 jiffies to the hypervisor;
    # CPU 0, another runner's, is not counted.
    monkeypatch.setenv("RUNNER_CPU_SET", "1-2")
    _proc_stat(machine_info, monkeypatch, _PROC_STAT)
    machine_info.cmd_start(tmp_path)
    _proc_stat(machine_info, monkeypatch, _PROC_STAT.replace(
        "cpu1 100 0 10 870 0 0 0 20", "cpu1 400 0 10 1040 0 0 0 50",
    ).replace("cpu0 600 0 50 4300", "cpu0 600 0 50 9300"))
    machine_info.cmd_finalize(tmp_path)
    info = json.loads((tmp_path / machine_info.FINAL_FILE).read_text())
    assert info["cpu_steal_cpus"] == "1,2"
    assert info["cpu_steal_pct"] == pytest.approx(100 * 30 / 500)
    assert "cpu_steal_jiffies_start" not in info and "cpu_total_jiffies_start" not in info
