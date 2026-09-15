"""Unit tests for :mod:`k4bench.metrics`, the one description of every
measured column that the dashboard, mail, analysis figures and report share."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from k4bench import metrics
from k4bench.metrics import (
    METRIC_LABELS,
    METRICS,
    is_megabytes,
    metric_label,
    metric_title,
    metric_unit,
    pretty_metric,
)
from k4bench.regression.report_builder import EVENT_VALUE_METRICS, RUN_VALUE_METRICS

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def test_metrics_is_a_leaf_module():
    """Imported by layers that must not import each other (analysis figures,
    regression mail, dashboard), so it imports nothing from k4bench."""
    tree = ast.parse(Path(metrics.__file__).read_text())
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not [name for name in imported if name.startswith("k4bench")]


def test_every_recorded_or_displayed_metric_is_described():
    # One registry instead of per-view tables: anything the report records,
    # or a dashboard view plots, ranks or tabulates, must be named here.
    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    pytest.importorskip("streamlit")
    from tabs import event_memory, impact, trends

    from k4bench.analysis.plots.overview import _OVERVIEW_METRICS

    displayed = (
        set(RUN_VALUE_METRICS)
        | set(EVENT_VALUE_METRICS)
        | {column for column, _ in trends._METRICS}
        | set(trends._STABILITY_METRICS)
        | set(event_memory._STABILITY_METRICS)
        | {metric.column for metric in impact._METRICS}
        | {column for column, _ in _OVERVIEW_METRICS}
    )
    assert displayed <= set(METRICS), sorted(displayed - set(METRICS))


def test_metric_names_are_sentence_case():
    # Callers drop these straight into titles and list items.
    for name in METRIC_LABELS.values():
        assert name[:1] == name[:1].upper()


def test_labels_view_matches_the_registry():
    assert METRIC_LABELS == {name: spec.label for name, spec in METRICS.items()}


def test_a_region_level_metric_carries_its_sub_detector():
    assert pretty_metric("mean_rss_mb", "EMEC_turbine") == "Mean event RSS · EMEC_turbine"
    assert pretty_metric("mean_rss_mb") == "Mean event RSS"


def test_an_unknown_metric_keeps_its_raw_name_and_no_unit():
    assert pretty_metric("some_future_column") == "some_future_column"
    assert metric_label("some_future_column") == "some_future_column"
    assert metric_unit("some_future_column") == ""
    assert metric_title("some_future_column") == "some_future_column"
    assert not is_megabytes("some_future_column")


@pytest.mark.parametrize(
    "metric, title",
    [
        ("wall_time_s", "Wall time (s)"),
        ("mean_rss_mb", "Mean event RSS (MB)"),
        ("rss_anon_slope_mb_per_event", "Anonymous RSS growth per event (MB/event)"),
        ("events_per_sec", "Throughput (ev/s)"),
        ("cpu_efficiency", "CPU efficiency"),
    ],
)
def test_titles_carry_the_stored_unit(metric, title):
    assert metric_title(metric) == title


def test_a_view_can_override_the_unit():
    assert metric_title("peak_vmem_mb", "GB") == "Peak virtual memory (GB)"


@pytest.mark.parametrize(
    "metric", ["peak_rss_mb", "peak_vmem_mb", "mean_rss_mb", "mean_rss_anon_mb", "mean_rss_file_mb"]
)
def test_memory_sizes_are_megabytes(metric):
    assert is_megabytes(metric)
    assert metric_unit(metric) == "MB"


def test_a_growth_rate_is_not_a_size():
    # Stored in MB/event: rescaling it to GB would misstate the rate.
    assert not is_megabytes("rss_anon_slope_mb_per_event")


def test_directions():
    assert METRICS["wall_time_s"].lower_is_better is True
    assert METRICS["events_per_sec"].lower_is_better is False
    assert METRICS["cpu_efficiency"].lower_is_better is False
    assert METRICS["returncode"].lower_is_better is None
