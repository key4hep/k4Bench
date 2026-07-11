"""Unit tests for the pure trend builders in :mod:`k4bench.analysis.trend`.

These are the Streamlit-free extractions behind ``dashboard/data.py``'s cached
trend loaders, used directly by the nightly regression report in CI — so unlike
``test_dashboard_trends.py`` this module must run without Streamlit installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from k4bench.analysis import trend


def test_trend_module_does_not_import_streamlit():
    # The nightly regression report imports this module in a CI venv without
    # Streamlit; a stray dashboard import would break that silently until the
    # first nightly run.
    assert "streamlit" not in sys.modules or "streamlit" not in trend.__dict__


def _make_run(parent: Path, date: str, k4h_release: str, wall_time_s: float) -> Path:
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


def test_build_results_trend(tmp_path):
    r1 = _make_run(tmp_path / "a", "2026-05-20", "key4hep-2026-05-20", 5.0)
    r2 = _make_run(tmp_path / "b", "2026-05-21", "key4hep-2026-05-21", 6.0)
    df = trend.build_results_trend((str(r1), str(r2)))
    assert df is not None and len(df) == 2
    for col in ("run_id", "run_date", "k4h_release", "x_date", "wall_time_s"):
        assert col in df.columns


def test_build_results_trend_empty():
    assert trend.build_results_trend(()) is None


def test_parse_run_dir_reads_run_info(tmp_path):
    r1 = _make_run(tmp_path, "2026-05-20", "key4hep-2026-05-20", 5.0)
    meta = trend.parse_run_dir(r1)
    assert meta["platform"] == "PLAT"
    assert meta["sample"] == "single_e"
    # Release date is inferred from the release name when absent.
    assert str(meta["k4h_release_date"].date()) == "2026-05-20"
