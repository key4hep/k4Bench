"""Unit tests for dd4bench.results.reporter."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dd4bench.results.model import RunResult
from dd4bench.results.reporter import print_summary, save_csv


def _make_result(label: str, returncode: int = 0) -> RunResult:
    return RunResult(
        label=label,
        returncode=returncode,
        n_events=2,
        wall_time_raw="0:05.00",
        wall_time_s=5.0,
        user_cpu_s=4.5,
        sys_cpu_s=0.5,
        peak_rss_mb=1024.0,
        output_size_mb=1.0,
        events_per_sec=0.4,
    )


class TestSaveCsv:
    def test_file_is_created(self, tmp_path):
        path = tmp_path / "results.csv"
        save_csv([_make_result("baseline_all")], path)
        assert path.exists()

    def test_parent_dirs_created(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "results.csv"
        save_csv([_make_result("baseline_all")], path)
        assert path.exists()

    def test_csv_has_header_row(self, tmp_path):
        path = tmp_path / "results.csv"
        save_csv([_make_result("baseline_all")], path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 1

    def test_csv_contains_all_results(self, tmp_path):
        path = tmp_path / "results.csv"
        results = [_make_result("baseline_all"), _make_result("without_Ecal")]
        save_csv(results, path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2

    def test_csv_label_field_correct(self, tmp_path):
        path = tmp_path / "results.csv"
        save_csv([_make_result("baseline_all")], path)
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["label"] == "baseline_all"

    def test_empty_results_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_csv([], tmp_path / "results.csv")


class TestPrintSummary:
    def test_prints_without_error(self, capsys):
        results = [_make_result("baseline_all"), _make_result("without_Ecal", returncode=1)]
        print_summary(results)
        out = capsys.readouterr().out
        assert "baseline_all" in out
        assert "without_Ecal" in out

    def test_none_fields_shown_as_na(self, capsys):
        result = RunResult(label="test", returncode=0, n_events=2)
        print_summary([result])
        out = capsys.readouterr().out
        assert "N/A" in out

    def test_summary_header_present(self, capsys):
        print_summary([_make_result("baseline_all")])
        out = capsys.readouterr().out
        assert "SUMMARY" in out
