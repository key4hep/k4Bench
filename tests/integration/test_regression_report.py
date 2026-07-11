"""End-to-end smoke test for ``.github/scripts/regression_report.py``.

Builds a tiny synthetic run-dir history with the EOS layout and runs the real
CLI against it (local ``--data-dir`` mode, no network), asserting the shape of
the written ``report.json``/``report.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "regression_report.py"
_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_STACK = "key4hep-2026-01-01"


def _write_run(run_dir: Path, night: str, wall_time_s: float) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "platform": _PLAT, "k4h_release": _STACK, "sample": "single_e",
    }))
    (run_dir / "baseline_results.csv").write_text(
        "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        f"baseline,0,10,{wall_time_s},1024.0,{wall_time_s * 0.98},{10.0 / wall_time_s}\n"
    )
    (run_dir / "machine_info.json").write_text(json.dumps({
        "hostname": "host-a", "cpu_physical_cores": 8, "cpu_logical_cores": 16,
        "load_avg_1m_start": 0.5, "load_avg_1m_end": 0.5,
        "ram_total_gb": 64.0, "ram_available_gb_start": 32.0,
        "ram_available_gb_end": 32.0, "swap_in_pages": 0, "swap_out_pages": 0,
        "thermal_throttle_events": 0,
    }))


def test_regression_report_cli_local_mode(tmp_path):
    # 10 steady nights, then a persisting +20% step on the last two → one
    # confirmed wall-time regression in tonight's report.
    walls = [100.0, 100.4, 99.6, 100.2, 99.8, 100.3, 99.7, 100.1, 99.9, 100.0,
             120.0, 120.5]
    sample_root = tmp_path / "data" / "DET" / _PLAT / _STACK / "single_e"
    d0 = date.fromisoformat("2026-01-01")
    for i, wall in enumerate(walls):
        night = (d0 + timedelta(days=i)).isoformat()
        _write_run(sample_root / night, night, wall)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT),
         "--data-dir", str(tmp_path / "data"), "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    data = json.loads((out_dir / "report.json").read_text())
    summary = data["summary"]
    assert summary["report_night"] == "2026-01-12"
    assert summary["n_detectors"] == 1
    assert summary["has_alertable"] is True
    assert summary["n_regressions"] >= 1
    confirmed = [
        v for g in data["groups"] for v in g["verdicts"]
        if v["severity"] == "CONFIRMED" and v["direction"] == "UP"
    ]
    assert any(v["metric"] == "wall_time_s" for v in confirmed)

    md = (out_dir / "report.md").read_text()
    assert "k4Bench nightly regression report — 2026-01-12" in md
    assert "🔴 Regression" in md
