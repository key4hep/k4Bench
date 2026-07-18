"""Config Impact must compare one selected run, never mixed trend rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from tabs import impact  # noqa: E402


def test_prep_data_uses_only_selected_run_rows_and_needs_no_trend_dates():
    results = pd.DataFrame({
        "label": ["current_a", "current_b"],
        "wall_time_s": [10.0, 8.0],
        "peak_rss_mb": [1000.0, 900.0],
        "user_cpu_s": [9.0, 7.0],
        "events_per_sec": [1.0, 1.25],
    })

    snapshot = impact._prep_data(
        results, ["history_only", "current_a", "current_b"],
    )

    assert list(snapshot["label"]) == ["current_a", "current_b"]
    assert "x_date" not in snapshot.columns

