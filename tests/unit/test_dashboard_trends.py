"""Unit tests for the dashboard trend loaders' run-dir-list interface.

The trend loaders in ``dashboard/data.py`` take an explicit tuple of run
directories (the date-windowed set produced by ``remote.fetch_runs_windowed``)
rather than walking a parent directory. ``data.py`` imports Streamlit, so the
whole module is skipped when Streamlit is unavailable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

_DATA_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "data.py"


def _load_data():
    spec = importlib.util.spec_from_file_location("k4bench_dashboard_data", _DATA_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


data = _load_data()


def _make_run(parent: Path, date: str, k4h_release: str, wall_time_s: float) -> Path:
    """Create a cache-style run dir with run_info.json + one results.csv."""
    run_dir = parent / date
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": date,
        "platform": "PLAT",
        "k4h_release": k4h_release,
        "sample": "single_e",
    }))
    (run_dir / "baseline_results.csv").write_text(
        "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        f"baseline,0,10,{wall_time_s},1024.0,4.0,2.0\n"
    )
    return run_dir


def test_trend_results_loads_from_run_dir_tuple(tmp_path):
    r1 = _make_run(tmp_path / "a", "2026-05-20", "key4hep-2026-05-20", 5.0)
    r2 = _make_run(tmp_path / "b", "2026-05-21", "key4hep-2026-05-21", 6.0)

    df = data.cached_load_trend_results((str(r1), str(r2)))
    assert df is not None
    assert len(df) == 2
    # Per-run metadata columns are attached and x_date is derived.
    for col in ("run_id", "run_date", "k4h_release", "x_date", "wall_time_s"):
        assert col in df.columns
    assert set(df["wall_time_s"]) == {5.0, 6.0}


def test_trend_results_empty_tuple_returns_none():
    assert data.cached_load_trend_results(()) is None


def test_trend_results_skips_missing_dirs(tmp_path):
    r1 = _make_run(tmp_path / "a", "2026-05-20", "key4hep-2026-05-20", 5.0)
    df = data.cached_load_trend_results((str(r1), str(tmp_path / "does-not-exist")))
    assert df is not None and len(df) == 1


def _make_machine_run(parent: Path, date: str, k4h_release: str, hostname: str) -> Path:
    """Create a cache-style run dir with run_info.json + machine_info.json."""
    run_dir = _make_run(parent, date, k4h_release, wall_time_s=1.0)
    (run_dir / "machine_info.json").write_text(json.dumps({
        "hostname": hostname,
        "cpu_physical_cores": 8,
        "cpu_logical_cores": 16,
        "load_avg_1m_start": 1.0,
        "ram_total_gb": 32.0,
        "ram_available_gb_start": 16.0,
    }))
    return run_dir


def test_trend_machine_info_loads_from_run_dir_tuple(tmp_path):
    r1 = _make_machine_run(tmp_path / "a", "2026-05-20", "key4hep-2026-05-20", "host-a")
    r2 = _make_machine_run(tmp_path / "b", "2026-05-21", "key4hep-2026-05-21", "host-b")

    df = data.cached_load_trend_machine_info((str(r1), str(r2)))
    assert df is not None
    assert len(df) == 2
    for col in ("run_id", "run_date", "k4h_release", "x_date", "hostname", "cpu_physical_cores"):
        assert col in df.columns
    assert set(df["hostname"]) == {"host-a", "host-b"}


def test_trend_machine_info_empty_tuple_returns_none():
    assert data.cached_load_trend_machine_info(()) is None


def test_trend_machine_info_skips_dirs_without_machine_info(tmp_path):
    r1 = _make_machine_run(tmp_path / "a", "2026-05-20", "key4hep-2026-05-20", "host-a")
    r2 = _make_run(tmp_path / "b", "2026-05-21", "key4hep-2026-05-21", wall_time_s=1.0)
    df = data.cached_load_trend_machine_info((str(r1), str(r2)))
    assert df is not None and len(df) == 1
