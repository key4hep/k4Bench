"""Unit tests for dd4bench.benchmark.ddsim.

run_ddsim is monkey-patched throughout so no real ddsim is needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dd4bench.benchmark.ddsim import BenchmarkConfig, SweepMode, run_sweep
from dd4bench.results.model import RunResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures" / "minimal_geometry"
MINIMAL_XML = FIXTURES / "minimal.xml"
ALL_DETECTORS = {"InnerTracker", "OuterTracker", "EcalBarrel", "HcalBarrel"}


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


def _make_config(tmp_path: Path, **kwargs) -> BenchmarkConfig:
    defaults = dict(
        xml_path=MINIMAL_XML,
        n_events=2,
        output_file=tmp_path / "out.root",
        log_dir=tmp_path / "logs",
        mode=SweepMode.FULL,
        detector_names=[],
        setup_script=None,
        extra_args=[],
    )
    return BenchmarkConfig(**{**defaults, **kwargs})


def _mock_run(**kw):
    return _make_result(kw["label"])


# ---------------------------------------------------------------------------
# BenchmarkConfig validation
# ---------------------------------------------------------------------------


class TestBenchmarkConfigValidation:
    def test_include_mode_requires_detector_names(self, tmp_path):
        with pytest.raises(ValueError, match="detector_names"):
            _make_config(tmp_path, mode=SweepMode.INCLUDE_ONLY, detector_names=[])

    def test_exclude_mode_allows_empty_detector_names(self, tmp_path):
        config = _make_config(tmp_path, mode=SweepMode.EXCLUDE_ONLY, detector_names=[])
        assert config.mode == SweepMode.EXCLUDE_ONLY

    def test_full_mode_needs_no_extra_fields(self, tmp_path):
        config = _make_config(tmp_path, mode=SweepMode.FULL)
        assert config.mode == SweepMode.FULL

# ---------------------------------------------------------------------------
# FULL mode
# ---------------------------------------------------------------------------


class TestFullMode:
    @pytest.fixture
    def results(self, tmp_path):
        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            return run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

    def test_baseline_is_first(self, results):
        assert results[0].label == "baseline_all"

    def test_one_result_per_detector_plus_baseline(self, results):
        assert len(results) == 5

    def test_all_detectors_covered(self, results):
        labels = {r.label for r in results}
        assert all(f"without_{d}" in labels for d in ALL_DETECTORS)

    def test_no_tmp_files_left_behind(self, tmp_path):
        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))
        assert list(FIXTURES.glob("_dd4bench_tmp_*")) == []


# ---------------------------------------------------------------------------
# INCLUDE mode
# ---------------------------------------------------------------------------


class TestIncludeMode:
    @pytest.fixture
    def results(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.INCLUDE_ONLY,
            detector_names=["EcalBarrel", "HcalBarrel"],
        )
        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            return run_sweep(config)

    def test_no_baseline(self, results):
        assert not any(r.label == "baseline_all" for r in results)

    def test_single_run(self, results):
        assert len(results) == 1

    def test_label_contains_both_detectors(self, results):
        label = results[0].label
        assert "EcalBarrel" in label
        assert "HcalBarrel" in label

    def test_no_removal_labels(self, results):
        assert not any(r.label.startswith("without_") for r in results)

    def test_unknown_detector_in_include_list_is_skipped(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.INCLUDE_ONLY,
            detector_names=["EcalBarrel", "NonExistent"],
        )
        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            results = run_sweep(config)
        assert len(results) == 1
        assert "EcalBarrel" in results[0].label
        assert "NonExistent" not in results[0].label

    def test_all_unknown_detectors_raises(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.INCLUDE_ONLY,
            detector_names=["NonExistent"],
        )
        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            with pytest.raises(ValueError, match="No valid detectors to keep"):
                run_sweep(config)


# ---------------------------------------------------------------------------
# EXCLUDE mode
# ---------------------------------------------------------------------------


class TestExcludeMode:
    @pytest.fixture
    def results(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.EXCLUDE_ONLY,
            detector_names=["InnerTracker", "OuterTracker"],
        )
        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            return run_sweep(config)

    def test_no_baseline(self, results):
        assert not any(r.label == "baseline_all" for r in results)

    def test_single_run(self, results):
        assert len(results) == 1

    def test_label_starts_with_without(self, results):
        assert results[0].label.startswith("without_")

    def test_label_contains_excluded_detectors(self, results):
        label = results[0].label
        assert "InnerTracker" in label
        assert "OuterTracker" in label

    def test_no_include_only_labels(self, results):
        assert not any(r.label.startswith("only_") for r in results)

    def test_empty_exclude_runs_full_geometry(self, tmp_path):
        config = _make_config(tmp_path, mode=SweepMode.EXCLUDE_ONLY, detector_names=[])
        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            results = run_sweep(config)
        assert len(results) == 1
        assert results[0].label == "baseline_all"

    def test_all_unknown_detectors_raises(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.EXCLUDE_ONLY,
            detector_names=["NonExistent"],
        )
        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            with pytest.raises(ValueError, match="No valid detectors to exclude"):
                run_sweep(config)


# ---------------------------------------------------------------------------
# Failure handling (all removal modes)
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_failed_ddsim_run_included_in_results(self, tmp_path):
        def side_effect(**kw):
            rc = 1 if kw["label"] == "without_EcalBarrel" else 0
            return _make_result(kw["label"], returncode=rc)

        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=side_effect):
            results = run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

        failed = [r for r in results if r.label == "without_EcalBarrel"]
        assert len(failed) == 1
        assert not failed[0].succeeded

    def test_other_runs_continue_after_ddsim_failure(self, tmp_path):
        def side_effect(**kw):
            rc = 1 if kw["label"] == "without_EcalBarrel" else 0
            return _make_result(kw["label"], returncode=rc)

        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=side_effect):
            results = run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

        assert len(results) == 5

    def test_unexpected_exception_skips_run_continues(self, tmp_path):
        def side_effect(**kw):
            if kw["label"] == "without_EcalBarrel":
                raise RuntimeError("unexpected crash")
            return _make_result(kw["label"])

        with patch("dd4bench.benchmark.ddsim.run_ddsim", side_effect=side_effect):
            results = run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

        labels = {r.label for r in results}
        assert "without_EcalBarrel" not in labels
        assert len(results) == 4
