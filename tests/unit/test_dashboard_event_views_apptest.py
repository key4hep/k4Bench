"""Regression tests for current-run Event Timing/Memory widget state."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def _app(dashboard_dir, view, current_labels, selected_labels):
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import pandas as pd
    import plotly.graph_objects as go

    from tabs import event_memory, event_timing

    frame = pd.DataFrame({
        "event_number": [0, 1, 2],
        "event_time_s": [2.0, 1.0, 1.1],
        "rss_end_mb": [100.0, 110.0, 111.0],
    })
    data = {label: frame.copy() for label in current_labels}
    tab = event_timing if view == "timing" else event_memory
    if view == "timing":
        tab.plot_event_timing = lambda *a, **k: go.Figure()
    else:
        tab.plot_event_memory = lambda *a, **k: go.Figure()
    tab.render(data, None, selected_labels)


@pytest.mark.parametrize(
    ("view", "baseline_key"),
    [("timing", "evt_timing_baseline"), ("memory", "evt_memory_baseline")],
)
def test_current_run_excludes_history_only_configurations(view, baseline_key):
    at = AppTest.from_function(
        _app,
        args=(
            str(_DASHBOARD_DIR), view,
            ["current_a", "current_b"],
            ["history_only", "current_a", "current_b"],
        ),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    baseline = at.selectbox(key=baseline_key)
    assert list(baseline.options) == ["current_a", "current_b"]
    assert baseline.value == "current_a"


@pytest.mark.parametrize(
    ("view", "topn_key", "palette_key"),
    [
        ("timing", "evt_timing_topn", "evt_timing_palette"),
        ("memory", "evt_memory_topn", "evt_memory_palette"),
    ],
)
def test_palette_redefaults_when_top_n_crosses_its_capacity(
    view, topn_key, palette_key,
):
    labels = [f"cfg_{i:02d}" for i in range(12)]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), view, labels, labels),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    assert at.selectbox(key=palette_key).value == "Matplotlib"

    at.slider(key=topn_key).set_value(12).run()
    assert not at.exception, at.exception
    assert at.selectbox(key=palette_key).value == "Matplotlib tab20"

