"""Unit tests for the trimmed timing statistic
(:mod:`k4bench.analysis.trend`) that the regression engine judges with."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from k4bench.analysis import trend
from k4bench.analysis.trend import (
    MAX_SLOPE_EVENTS,
    MIN_SLOPE_EVENTS,
    MIN_TRIM_EVENTS,
    build_event_timing_trend,
    theil_sen_slope,
    upper_trimmed_mean,
)


# ── upper_trimmed_mean ────────────────────────────────────────────────────────

def test_upper_trimmed_mean_drops_the_slow_tail():
    # 95 events at 1.0 s and 5 at 100 s: the mean is dragged far above the
    # typical event, the trimmed mean is not.
    times = np.array([1.0] * 95 + [100.0] * 5)
    assert times.mean() == pytest.approx(5.95)
    assert upper_trimmed_mean(times) == pytest.approx(1.0)


def test_upper_trimmed_mean_trims_only_the_slow_end():
    # One-sided by construction: a fast outlier is kept, which is what makes
    # this an *upper*-trimmed mean rather than the symmetric statistic the bare
    # word usually names.
    times = np.array([0.01] * 5 + [1.0] * 95)
    assert upper_trimmed_mean(times) < 1.0


def test_upper_trimmed_mean_leaves_a_flat_distribution_alone():
    times = np.array([2.0] * 100)
    assert upper_trimmed_mean(times) == pytest.approx(2.0)


def test_upper_trimmed_mean_ignores_event_order():
    rng = np.random.default_rng(0)
    times = rng.lognormal(size=200)
    assert upper_trimmed_mean(times) == pytest.approx(upper_trimmed_mean(times[::-1]))


def test_upper_trimmed_mean_abstains_below_the_event_floor():
    # Under the floor a 5% trim drops less than one event, so a returned value
    # would be the plain mean wearing another name.
    assert upper_trimmed_mean(np.ones(MIN_TRIM_EVENTS - 1)) is None
    assert upper_trimmed_mean(np.ones(MIN_TRIM_EVENTS)) is not None


# ── theil_sen_slope ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("slope", [0.5, -0.25, 0.0])
def test_theil_sen_slope_recovers_a_straight_line(slope):
    x = np.arange(1, 50)
    assert theil_sen_slope(x, 1000.0 + slope * x) == pytest.approx(slope, abs=1e-12)


def test_theil_sen_slope_is_not_moved_by_one_outlier():
    # One spike: least squares and last - first would both be pulled far off.
    x = np.arange(1, 21)
    y = 100.0 + 2.0 * x
    y[-1] += 10_000.0
    assert theil_sen_slope(x, y) == pytest.approx(2.0)
    assert np.polyfit(x, y, 1)[0] > 20.0


def test_theil_sen_slope_uses_actual_x_spacing():
    x = [1, 2, 5, 10, 11]
    assert theil_sen_slope(x, [2.0 * v for v in x]) == pytest.approx(2.0)


def test_theil_sen_slope_skips_non_finite_points_and_equal_x_pairs():
    x = [1, 2, 2, 3, 4, 5, np.nan]
    y = [1.0, 2.0, 2.0, np.nan, 4.0, 5.0, 6.0]
    assert theil_sen_slope(x, y) == pytest.approx(1.0)


def test_theil_sen_slope_abstains_below_the_sample_floor():
    x = np.arange(MIN_SLOPE_EVENTS - 1)
    assert theil_sen_slope(x, x * 1.0) is None
    x = np.arange(MIN_SLOPE_EVENTS)
    assert theil_sen_slope(x, x * 1.0) == pytest.approx(1.0)


def test_theil_sen_slope_abstains_without_distinct_x():
    assert theil_sen_slope([3] * 10, np.arange(10.0)) is None


def test_theil_sen_slope_bounds_its_pairwise_work(monkeypatch):
    calls = []
    real = np.triu_indices

    def spy(n, *args, **kwargs):
        calls.append(n)
        return real(n, *args, **kwargs)

    monkeypatch.setattr(trend.np, "triu_indices", spy)
    rng = np.random.default_rng(1)
    x = np.arange(100_000)
    y = 5000.0 + 0.003 * x + rng.normal(0.0, 2.0, x.size)
    first = theil_sen_slope(x, y)
    second = theil_sen_slope(x, y)
    assert calls and max(calls) <= MAX_SLOPE_EVENTS
    assert first == second
    assert first == pytest.approx(0.003, abs=2e-4)


# ── the trend frame ───────────────────────────────────────────────────────────

def _run_dir(root: Path, night: str, times: list[float]) -> str:
    run_dir = root / night
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "k4h_release": f"key4hep-{night}",
    }))
    n = len(times)
    (run_dir / "baseline_events.json").write_text(json.dumps({
        "event_numbers": list(range(n)),
        "event_times_s": times,
        "event_rss_begin_mb": [1000.0] * n,
        "event_rss_end_mb": [1024.0] * n,
    }))
    return str(run_dir)


def test_event_trend_carries_the_trimmed_column(tmp_path):
    # Event 0 is the warm-up and is excluded before the statistic is taken.
    times = [999.0] + [1.0] * 95 + [50.0] * 5
    df = build_event_timing_trend((_run_dir(tmp_path, "2026-01-01", times),))
    row = df.iloc[0]
    assert row["n_events"] == 100
    assert row["mean_time_s"] == pytest.approx(3.45)
    assert row["trimmed_mean_time_s"] == pytest.approx(1.0)


def test_event_trend_omits_the_column_when_there_are_too_few_events(tmp_path):
    df = build_event_timing_trend((_run_dir(tmp_path, "2026-01-01", [1.0] * 5),))
    row = df.iloc[0]
    assert "trimmed_mean_time_s" not in df.columns or row.isna()["trimmed_mean_time_s"]


def test_event_trend_memory_stats_exclude_warmup(tmp_path):
    run = _run_dir(tmp_path, "2026-01-02", [1.0] * 4)
    path = Path(run) / "baseline_events.json"
    raw = json.loads(path.read_text())
    raw.update(
        {
            "event_rss_anon_begin_mb": [999.0, 99.0, 101.0, 103.0],
            "event_rss_anon_end_mb": [999.0, 100.0, 102.0, 104.0],
            "event_rss_file_end_mb": [999.0, 50.0, 60.0, 70.0],
        }
    )
    path.write_text(json.dumps(raw))
    stats = {
        "mean_rss_anon_mb": 102.0,
        "median_rss_anon_mb": 102.0,
        "std_rss_anon_mb": 2.0,
        "n_events_rss_anon": 3,
        "mean_rss_file_mb": 60.0,
    }
    df = build_event_timing_trend((run,)).set_index("run_id")
    for key, value in stats.items():
        assert df.loc["2026-01-02", key] == pytest.approx(value)


@pytest.mark.parametrize(
    "key, metric",
    [
        ("event_rss_anon_end_mb", "mean_rss_anon_mb"),
        ("event_rss_file_end_mb", "mean_rss_file_mb"),
    ],
)
def test_event_trend_memory_ignores_failed_samples(tmp_path, key, metric):
    run = _run_dir(tmp_path, "2026-01-01", [1.0] * 4)
    path = Path(run) / "baseline_events.json"
    raw = json.loads(path.read_text())
    raw[key] = [999.0, -0.001, None, 100.0]
    path.write_text(json.dumps(raw))
    row = build_event_timing_trend((run,)).iloc[0]
    assert row[metric] == 100.0
    if metric == "mean_rss_anon_mb":
        assert row["std_rss_anon_mb"] == 0.0
        assert row["n_events_rss_anon"] == 1


def _with_anon(run: str, anon_end: list) -> str:
    path = Path(run) / "baseline_events.json"
    raw = json.loads(path.read_text())
    raw["event_rss_anon_end_mb"] = anon_end
    path.write_text(json.dumps(raw))
    return run


def test_event_trend_anon_slope_excludes_warmup_and_failed_samples(tmp_path):
    # Event 0's spike and the failed reads would tilt the slope if kept; the
    # failed reads leave gaps in x, not a shifted sequence.
    anon = [5000.0, 100.0, -0.001, 104.0, None, 108.0, 110.0, 112.0]
    run = _with_anon(_run_dir(tmp_path, "2026-01-03", [1.0] * len(anon)), anon)
    row = build_event_timing_trend((run,)).iloc[0]
    assert row["rss_anon_slope_mb_per_event"] == pytest.approx(2.0)


def test_event_trend_anon_slope_is_absent_with_too_few_samples(tmp_path):
    anon = [5000.0, 100.0, 101.0, -1.0, 102.0, 103.0]
    run = _with_anon(_run_dir(tmp_path, "2026-01-04", [1.0] * len(anon)), anon)
    df = build_event_timing_trend((run,))
    assert df.iloc[0]["n_events_rss_anon"] == 4
    assert "rss_anon_slope_mb_per_event" not in df.columns


def test_event_trend_without_anon_rss_has_no_slope(tmp_path):
    df = build_event_timing_trend((_run_dir(tmp_path, "2026-01-05", [1.0] * 10),))
    assert "rss_anon_slope_mb_per_event" not in df.columns
