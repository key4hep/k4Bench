"""Smoke tests for dd4bench.analysis.plots.

These tests verify that each plotting function runs without error and
returns a Figure.  Visual correctness is not tested here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dd4bench.analysis.loader import load_results
from dd4bench.analysis.plots import (
    _compute_core_range,
    plot_compare,
    plot_event_timing,
    plot_event_timing_overlay,
    plot_run_overview,
    plot_sweep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _run_row(label: str, wall: float, rss: float, returncode: int = 0) -> dict:
    return {
        "label": label,
        "returncode": returncode,
        "n_events": 10,
        "wall_time_raw": "0:05.00",
        "wall_time_s": wall,
        "user_cpu_s": wall * 0.9,
        "sys_cpu_s": wall * 0.05,
        "peak_rss_mb": rss,
        "major_page_faults": 0,
        "voluntary_ctx_switches": 100,
        "involuntary_ctx_switches": 5,
        "output_size_mb": 2.0,
        "events_per_sec": round(10 / wall, 4),
    }


def _sweep_df(tmp_path: Path) -> pd.DataFrame:
    rows = [
        _run_row("baseline_all",   wall=10.0, rss=2000.0),
        _run_row("without_EcalBarrel", wall=8.0, rss=1800.0),
        _run_row("without_HcalBarrel", wall=9.0, rss=1900.0),
        _run_row("without_InnerTracker", wall=9.5, rss=1950.0),
    ]
    path = tmp_path / "results.csv"
    _write_csv(path, rows)
    return load_results(path)


def _compare_df(tmp_path: Path) -> pd.DataFrame:
    rows = [
        _run_row("geometry_a", wall=10.0, rss=2000.0),
        _run_row("geometry_b", wall=8.5,  rss=1800.0),
    ]
    path = tmp_path / "results.csv"
    _write_csv(path, rows)
    return load_results(path)


def _write_event_json(path: Path, n: int = 5) -> None:
    data = {
        "event_numbers": list(range(n)),
        "event_times_s": [0.1 * (i + 1) for i in range(n)],
        "event_rss_begin_mb": [500.0] * n,
        "event_rss_end_mb": [510.0] * n,
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# _compute_core_range
# ---------------------------------------------------------------------------


class TestComputeCoreRange:
    def test_no_outliers_covers_full_range(self):
        data = np.linspace(0.1, 0.5, 100)
        (lo, hi), n = _compute_core_range(data)
        assert n == 0
        assert lo <= data.min()
        assert hi >= data.max()

    def test_single_extreme_outlier_clipped(self):
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(loc=0.2, scale=0.01, size=99), [50.0]])
        (lo, hi), n = _compute_core_range(data)
        assert n >= 1
        assert hi < 50.0

    def test_returns_nonnegative_lower_bound(self):
        data = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
        (lo, _), _ = _compute_core_range(data)
        assert lo >= 0.0

    def test_identical_values_no_crash(self):
        data = np.full(10, 0.5)
        (lo, hi), n = _compute_core_range(data)
        assert n == 0
        assert lo <= hi


# ---------------------------------------------------------------------------
# plot_sweep
# ---------------------------------------------------------------------------


class TestPlotSweep:
    def test_returns_figure(self, tmp_path):
        df = _sweep_df(tmp_path)
        fig = plot_sweep(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_top_n(self, tmp_path):
        df = _sweep_df(tmp_path)
        fig = plot_sweep(df, top_n=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_baseline_raises(self, tmp_path):
        df = _sweep_df(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            plot_sweep(df, baseline_label="nonexistent")

    def test_warns_on_missing_metrics(self, tmp_path):
        df = _sweep_df(tmp_path)
        df.loc[df["label"] == "without_EcalBarrel", "wall_time_s"] = float("nan")
        with pytest.warns(UserWarning, match="missing"):
            fig = plot_sweep(df)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_event_timing
# ---------------------------------------------------------------------------


class TestPlotEventTiming:
    def test_single_run_returns_figure(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        fig = plot_event_timing(tmp_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_runs_returns_figure(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        fig = plot_event_timing(tmp_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_label_filter(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        fig = plot_event_timing(tmp_path, labels=["baseline_all"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bins_int(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json", n=20)
        fig = plot_event_timing(tmp_path, bins=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bins_auto(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json", n=50)
        fig = plot_event_timing(tmp_path, bins="auto")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_outlier_clipped_silently(self, tmp_path):
        """A single large outlier (<5 % of events) clips without a warning."""
        path = tmp_path / "baseline_all_events.json"
        _write_event_json(path, n=100)
        # Inject one extreme outlier
        import json
        raw = json.loads(path.read_text())
        raw["event_times_s"][-1] = 9999.0
        path.write_text(json.dumps(raw))
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            fig = plot_event_timing(tmp_path)  # should not raise
        plt.close(fig)

    def test_outlier_warns_when_fraction_high(self, tmp_path):
        """More than 5 % outliers triggers a UserWarning."""
        path = tmp_path / "baseline_all_events.json"
        _write_event_json(path, n=20)
        import json
        raw = json.loads(path.read_text())
        # Inject 2/20 = 10 % outliers
        raw["event_times_s"][0]  = 9999.0
        raw["event_times_s"][-1] = 9999.0
        path.write_text(json.dumps(raw))
        with pytest.warns(UserWarning, match="outside plotted range"):
            fig = plot_event_timing(tmp_path)
        plt.close(fig)

    def test_outlier_threshold_respected(self, tmp_path):
        """Raising outlier_threshold keeps more of the tail in view."""
        _write_event_json(tmp_path / "baseline_all_events.json", n=30)
        fig = plot_event_timing(tmp_path, outlier_threshold=10.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No \*_events.json"):
            plot_event_timing(tmp_path)


# ---------------------------------------------------------------------------
# plot_compare
# ---------------------------------------------------------------------------


class TestPlotCompare:
    def test_returns_figure(self, tmp_path):
        df = _compare_df(tmp_path)
        fig = plot_compare(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_labels(self, tmp_path):
        rows = [
            _run_row("v03", wall=10.0, rss=2000.0),
            _run_row("v04", wall=8.5,  rss=1800.0),
        ]
        path = tmp_path / "results.csv"
        _write_csv(path, rows)
        df = load_results(path)
        fig = plot_compare(df, label_a="v03", label_b="v04")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_metrics(self, tmp_path):
        df = _compare_df(tmp_path)
        fig = plot_compare(df, metrics=["wall_time_s"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_label_raises(self, tmp_path):
        df = _compare_df(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            plot_compare(df, label_a="nonexistent")


# ---------------------------------------------------------------------------
# plot_event_timing_overlay
# ---------------------------------------------------------------------------


class TestPlotEventTimingOverlay:
    def test_returns_figure(self, tmp_path):
        _write_event_json(tmp_path / "without_Ecal_events.json")
        _write_event_json(tmp_path / "without_Hcal_events.json")
        fig = plot_event_timing_overlay(tmp_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_accepts_preloaded_dict(self, tmp_path):
        _write_event_json(tmp_path / "without_Ecal_events.json")
        _write_event_json(tmp_path / "without_Hcal_events.json")
        from dd4bench.analysis.loader import load_event_timing
        data = load_event_timing(tmp_path)
        fig = plot_event_timing_overlay(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_label_filter(self, tmp_path):
        _write_event_json(tmp_path / "without_Ecal_events.json")
        _write_event_json(tmp_path / "without_Hcal_events.json")
        _write_event_json(tmp_path / "without_Inner_events.json")
        fig = plot_event_timing_overlay(tmp_path, labels=["without_Ecal", "without_Hcal"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_baseline(self, tmp_path):
        _write_event_json(tmp_path / "without_Ecal_events.json")
        _write_event_json(tmp_path / "without_Hcal_events.json")
        fig = plot_event_timing_overlay(tmp_path, baseline_label="without_Hcal")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_baseline_raises(self, tmp_path):
        _write_event_json(tmp_path / "without_Ecal_events.json")
        _write_event_json(tmp_path / "without_Hcal_events.json")
        with pytest.raises(ValueError, match="baseline_label"):
            plot_event_timing_overlay(tmp_path, baseline_label="nonexistent")

    def test_single_run_raises(self, tmp_path):
        _write_event_json(tmp_path / "without_Ecal_events.json")
        with pytest.raises(ValueError, match="at least 2 runs"):
            plot_event_timing_overlay(tmp_path)

    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No event timing data"):
            plot_event_timing_overlay(tmp_path)

    def test_custom_alpha_and_bins(self, tmp_path):
        _write_event_json(tmp_path / "without_Ecal_events.json", n=50)
        _write_event_json(tmp_path / "without_Hcal_events.json", n=50)
        fig = plot_event_timing_overlay(tmp_path, alpha=0.3, bins=20)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_run_overview
# ---------------------------------------------------------------------------


class TestPlotRunOverview:
    def test_returns_figure(self, tmp_path):
        df = _sweep_df(tmp_path)
        fig = plot_run_overview(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_baseline_highlight(self, tmp_path):
        df = _sweep_df(tmp_path)
        fig = plot_run_overview(df, baseline_label=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_metrics(self, tmp_path):
        df = _sweep_df(tmp_path)
        fig = plot_run_overview(df, metrics=[("wall_time_s", "Wall (s)"), ("peak_rss_mb", "RSS (MB)")])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_compare_results(self, tmp_path):
        df = _compare_df(tmp_path)
        fig = plot_run_overview(df, baseline_label=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
