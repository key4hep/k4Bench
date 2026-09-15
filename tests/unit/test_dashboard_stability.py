"""Run-to-run variability table shown under the memory trend figures
(:func:`stats.build_stability_table`) and its rendering on old and new data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from stats import MOVEMENT_COL, SAME_RELEASE_SPREAD_COL, build_stability_table  # noqa: E402


def _frame(values, releases, *, label="baseline", metric="peak_vmem_mb") -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "label": [label] * n,
            "run_id": [f"2026-02-{k + 1:02d}" for k in range(n)],
            "x_date": pd.to_datetime(releases),
            metric: values,
        }
    )


_TWO_NIGHTS_EACH = [f"2026-01-{k // 2 + 1:02d}" for k in range(8)]


def test_stable_repeats_give_a_small_relative_spread():
    values = [2000.0, 2000.4, 2100.0, 2099.6, 2200.0, 2200.4, 2300.0, 2299.6]
    table = build_stability_table(_frame(values, _TWO_NIGHTS_EACH), {"peak_vmem_mb": "MB"}, {})
    row = table.loc["baseline"]
    assert row["Metric"] == "Peak virtual memory"
    # Repeats differ by 0.4 MB: 1.4826 * 0.4 / sqrt(2) against a 2150 MB median.
    assert row[SAME_RELEASE_SPREAD_COL] == "0.419 MB (0.02%) · 4 pairs"
    # The releases step by 100 MB, which only the all-runs movement contains.
    assert row[MOVEMENT_COL] == "52.4 MB (2.4%)"


def test_rows_are_put_in_engine_order_before_pairing():
    # Interleaved input would pair no two runs of one release; in (release
    # date, run id) order every release contributes one pair.
    rows = [
        ("2026-01-02", "b", 101.0),
        ("2026-01-01", "a", 100.0),
        ("2026-01-03", "b", 100.0),
        ("2026-01-02", "a", 100.0),
        ("2026-01-01", "b", 101.0),
        ("2026-01-03", "a", 101.0),
    ]
    frame = pd.DataFrame(
        {
            "label": "baseline",
            "x_date": pd.to_datetime([r[0] for r in rows]),
            "run_id": [r[1] for r in rows],
            "peak_vmem_mb": [r[2] for r in rows],
        }
    )
    table = build_stability_table(frame, {"peak_vmem_mb": "MB"}, {})
    assert table.loc["baseline", SAME_RELEASE_SPREAD_COL] == "1.05 MB (1%) · 3 pairs"


def test_unreliable_runs_and_missing_values_are_dropped_not_zeroed():
    values = [100.0, np.nan, 100.0, 100.0, 100.0, 100.0, 100.0, 5000.0]
    frame = _frame(values, _TWO_NIGHTS_EACH)
    reliability = {"2026-02-08": False}
    table = build_stability_table(frame, {"peak_vmem_mb": "MB"}, reliability)
    # The NaN leaves release 01-01 with one run; the unreliable 5000 is gone, so
    # 01-04 has one too. Two same-release pairs remain: too few.
    assert table.loc["baseline", SAME_RELEASE_SPREAD_COL] == "N/A — too few same-release repeats"
    assert table.loc["baseline", MOVEMENT_COL] == "0 MB (0%)"


def test_no_repeats_leaves_the_headline_unavailable_while_movement_shows():
    values = [100.0, 101.0, 99.0, 100.0]
    releases = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    table = build_stability_table(_frame(values, releases), {"peak_vmem_mb": "MB"}, {})
    assert table.loc["baseline", SAME_RELEASE_SPREAD_COL].startswith("N/A")
    assert table.loc["baseline", MOVEMENT_COL] != "N/A"


def test_too_little_history_is_na_never_zero():
    table = build_stability_table(
        _frame([100.0, 100.0], ["2026-01-01"] * 2),
        {"peak_vmem_mb": "MB"},
        {},
    )
    assert table.loc["baseline", SAME_RELEASE_SPREAD_COL].startswith("N/A")
    assert table.loc["baseline", MOVEMENT_COL] == "N/A"


def test_zero_median_shows_no_percent():
    values = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
    table = build_stability_table(_frame(values, _TWO_NIGHTS_EACH), {"peak_vmem_mb": "MB"}, {})
    assert "%" not in table.loc["baseline", SAME_RELEASE_SPREAD_COL]


def test_slope_spread_is_absolute_only():
    metric = "rss_anon_slope_mb_per_event"
    values = [0.02, 0.03, 0.02, 0.01, 0.02, 0.03, 0.02, 0.01]
    table = build_stability_table(
        _frame(values, _TWO_NIGHTS_EACH, metric=metric),
        {metric: "MB/event"},
        {},
    )
    cell = table.loc["baseline", SAME_RELEASE_SPREAD_COL]
    assert "MB/event" in cell and "%" not in cell


def test_frame_without_the_metrics_gives_an_empty_table():
    frame = _frame([1.0, 2.0, 3.0], ["2026-01-01"] * 3, metric="wall_time_s")
    assert build_stability_table(frame, {"peak_vmem_mb": "MB"}, {}).empty


# ── rendering ────────────────────────────────────────────────────────────────
#
# Only these need Streamlit; the table builder above is plain pandas.


def _app_test():
    return pytest.importorskip("streamlit.testing.v1").AppTest


def _memory_app(dashboard_dir, with_new_metrics):
    import sys as _sys

    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import pandas as _pd

    from tabs import event_memory

    n = 6
    df = _pd.DataFrame(
        {
            "label": ["baseline"] * n,
            "run_id": [f"2026-02-{k + 1:02d}" for k in range(n)],
            "run_date": _pd.to_datetime([f"2026-02-{k + 1:02d}" for k in range(n)]),
            "x_date": _pd.to_datetime([f"2026-01-{k // 2 + 1:02d}" for k in range(n)]),
            "k4h_release": [f"key4hep-2026-01-{k // 2 + 1:02d}" for k in range(n)],
            "mean_rss_mb": [150.0 + k for k in range(n)],
            "median_rss_mb": [150.0] * n,
            "std_rss_mb": [1.0] * n,
            "n_events_rss": [10] * n,
        }
    )
    if with_new_metrics:
        df["mean_rss_anon_mb"] = 100.0
        df["n_events_rss_anon"] = 10
        df["rss_anon_slope_mb_per_event"] = 0.01
    event_memory._render_historical(df, {})


@pytest.mark.parametrize("with_new_metrics", [False, True])
def test_event_memory_history_renders_stability_on_old_and_new_data(with_new_metrics):
    at = (
        _app_test()
        .from_function(
            _memory_app,
            args=(str(_DASHBOARD_DIR), with_new_metrics),
            default_timeout=30,
        )
        .run()
    )
    assert not at.exception, at.exception
    assert [e.label for e in at.expander] == ["Run-to-run variability"]
    metrics = set(at.dataframe[0].value["Metric"])
    new = {"Mean event anonymous RSS", "Anonymous RSS growth per event"}
    assert "Mean event RSS" in metrics
    assert (new <= metrics) is with_new_metrics
    assert bool(new & metrics) is with_new_metrics
    # Units come from the dashboard's shared unit table, so a metric missing
    # there would render as a bare number.
    table = at.dataframe[0].value.set_index("Metric")
    for column in (SAME_RELEASE_SPREAD_COL, MOVEMENT_COL):
        assert " MB (" in table.loc["Mean event RSS", column]
        if with_new_metrics:
            assert " MB (" in table.loc["Mean event anonymous RSS", column]
            assert (
                table.loc["Anonymous RSS growth per event", column]
                .split(" · ")[0]
                .endswith(" MB/event")
            )


def _trends_app(dashboard_dir, with_vmem):
    import sys as _sys

    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import pandas as _pd

    from tabs import trends

    trends._cached_fetch_reports = lambda url, ids: {}
    n = 6
    df = _pd.DataFrame(
        {
            "label": ["baseline"] * n,
            "run_id": [f"2026-02-{k + 1:02d}" for k in range(n)],
            "returncode": [0] * n,
            "run_date": _pd.to_datetime([f"2026-02-{k + 1:02d}" for k in range(n)]),
            "x_date": _pd.to_datetime([f"2026-01-{k // 2 + 1:02d}" for k in range(n)]),
            "k4h_release": [f"key4hep-2026-01-{k // 2 + 1:02d}" for k in range(n)],
            "wall_time_s": [5.0] * n,
            "user_cpu_s": [4.0] * n,
            "peak_rss_mb": [1000.0 + k for k in range(n)],
        }
    )
    if with_vmem:
        df["peak_vmem_mb"] = 3000.0
    trends.render(
        df,
        reliability={},
        data_url="https://x.invalid",
        detector="CLD",
        platform="PLAT",
        sample="single_e",
    )


@pytest.mark.parametrize("with_vmem", [False, True])
def test_run_trends_stability_uses_every_run_on_old_and_new_data(with_vmem):
    at = (
        _app_test()
        .from_function(
            _trends_app,
            args=(str(_DASHBOARD_DIR), with_vmem),
            default_timeout=30,
        )
        .run()
    )
    assert not at.exception, at.exception
    assert [e.label for e in at.expander] == ["Run-to-run variability"]
    table = at.dataframe[0].value.set_index("Metric")
    assert ("Peak virtual memory" in table.index) is with_vmem
    # The plotted line collapses each tag to one run; the three same-release
    # pairs are still counted here.
    assert table.loc["Peak RSS", SAME_RELEASE_SPREAD_COL].endswith("· 3 pairs")
