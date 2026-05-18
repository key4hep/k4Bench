"""Unit tests for dd4bench.analysis.loader."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from dd4bench.analysis.loader import load_event_timing, load_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _minimal_row(label: str, returncode: int = 0) -> dict:
    return {
        "label": label,
        "returncode": returncode,
        "n_events": 10,
        "wall_time_raw": "0:05.00",
        "wall_time_s": 5.0,
        "user_cpu_s": 4.0,
        "sys_cpu_s": 0.5,
        "peak_rss_mb": 1024.0,
        "major_page_faults": 0,
        "voluntary_ctx_switches": 100,
        "involuntary_ctx_switches": 5,
        "output_size_mb": 2.0,
        "events_per_sec": 2.0,
    }


def _write_event_json(path: Path, n_events: int = 5) -> None:
    data = {
        "event_numbers": list(range(n_events)),
        "event_times_s": [0.1 * (i + 1) for i in range(n_events)],
        "event_rss_begin_mb": [500.0] * n_events,
        "event_rss_end_mb": [510.0] * n_events,
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_returns_dataframe(self, tmp_path):
        path = tmp_path / "results.csv"
        _write_csv(path, [_minimal_row("baseline_all")])
        df = load_results(path)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, tmp_path):
        path = tmp_path / "results.csv"
        rows = [_minimal_row("baseline_all"), _minimal_row("without_Ecal")]
        _write_csv(path, rows)
        df = load_results(path)
        assert len(df) == 2

    def test_float_columns_are_float(self, tmp_path):
        path = tmp_path / "results.csv"
        _write_csv(path, [_minimal_row("baseline_all")])
        df = load_results(path)
        assert df["wall_time_s"].dtype == float
        assert df["peak_rss_mb"].dtype == float

    def test_int_columns_are_int64(self, tmp_path):
        path = tmp_path / "results.csv"
        _write_csv(path, [_minimal_row("baseline_all")])
        df = load_results(path)
        assert str(df["n_events"].dtype) == "Int64"
        assert str(df["returncode"].dtype) == "Int64"

    def test_missing_metrics_become_nan(self, tmp_path):
        path = tmp_path / "results.csv"
        row = _minimal_row("failed_run", returncode=1)
        row["wall_time_s"] = ""
        _write_csv(path, [row])
        df = load_results(path)
        assert pd.isna(df["wall_time_s"].iloc[0])

    def test_accepts_string_path(self, tmp_path):
        path = tmp_path / "results.csv"
        _write_csv(path, [_minimal_row("baseline_all")])
        df = load_results(str(path))
        assert len(df) == 1


# ---------------------------------------------------------------------------
# load_event_timing
# ---------------------------------------------------------------------------


class TestLoadEventTiming:
    def test_returns_dict(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        result = load_event_timing(tmp_path)
        assert isinstance(result, dict)

    def test_label_extracted_from_filename(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        result = load_event_timing(tmp_path)
        assert "baseline_all" in result

    def test_dataframe_columns(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json", n_events=3)
        df = load_event_timing(tmp_path)["baseline_all"]
        assert set(df.columns) == {
            "event_number", "event_time_s",
            "rss_begin_mb", "rss_end_mb", "rss_delta_mb",
        }

    def test_rss_delta_computed(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json", n_events=2)
        df = load_event_timing(tmp_path)["baseline_all"]
        assert (df["rss_delta_mb"] == df["rss_end_mb"] - df["rss_begin_mb"]).all()

    def test_multiple_files_loaded(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        result = load_event_timing(tmp_path)
        assert set(result.keys()) == {"baseline_all", "without_Ecal"}

    def test_label_filter(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        result = load_event_timing(tmp_path, labels=["baseline_all"])
        assert list(result.keys()) == ["baseline_all"]

    def test_missing_file_raises_when_labels_explicit(self, tmp_path):
        with pytest.raises(ValueError, match="Missing event files"):
            load_event_timing(tmp_path, labels=["nonexistent"])

    def test_mismatched_array_lengths_raises(self, tmp_path):
        path = tmp_path / "bad_events.json"
        data = {
            "event_numbers": [0, 1, 2],
            "event_times_s": [0.1, 0.2],
            "event_rss_begin_mb": [500.0, 500.0, 500.0],
            "event_rss_end_mb": [510.0, 510.0, 510.0],
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="mismatched"):
            load_event_timing(tmp_path)

    def test_empty_dir_returns_empty_dict(self, tmp_path):
        result = load_event_timing(tmp_path)
        assert result == {}

    def test_accepts_string_path(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        result = load_event_timing(str(tmp_path))
        assert "baseline_all" in result
