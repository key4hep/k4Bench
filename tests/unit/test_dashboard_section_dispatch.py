"""The sections every page draws alike, and the context a section draws from:
refusing a remote-only read in local mode and reading the window's history
only when a view asks. Also the active section, which the sidebar reads before
the bar is drawn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import section_dispatch  # noqa: E402
import sections  # noqa: E402
from config import Config  # noqa: E402
from data import EVENTS, RUN_BASE, RunData, cached_load_trend_event_timing  # noqa: E402
from sidebar import TrendWindow  # noqa: E402

RECO_RUN = str(Path(__file__).resolve().parents[1] / "fixtures" / "reco_run")


def test_every_shared_section_has_a_shared_renderer_and_scope():
    assert set(section_dispatch.SHARED_RENDERERS) == set(sections.SHARED_SCOPES)


def _local_context() -> section_dispatch.PageContext:
    return section_dispatch.PageContext(
        config=Config(data_dir="run", data_url=None, cache_dir="cache", dashboard_url=""),
        selection=None,
        data_dir="run",
        run_meta=None,
        run=RunData(None, lambda _artifacts: "run"),
        window=TrendWindow(),
    )


def test_local_mode_has_no_trends_and_no_store_selection():
    ctx = _local_context()
    assert not ctx.trends_enabled
    with pytest.raises(RuntimeError, match="remote-only section"):
        ctx.remote


def _remote_context(results, asked: list) -> section_dispatch.PageContext:
    """A context whose window of one run records what it was asked for."""
    def download(artifacts):
        asked.append(artifacts)
        return (RECO_RUN,)

    return section_dispatch.PageContext(
        config=Config(RECO_RUN, "https://store.example", "/tmp", "https://dash.example"),
        selection=None,
        data_dir=RECO_RUN,
        run_meta=None,
        run=RunData(results, lambda _artifacts: RECO_RUN),
        window=TrendWindow(None, (("S", ("2026-10-17",)),), download=download),
    )


def test_the_history_is_read_only_when_a_view_asks_for_it():
    from data import cached_load_results

    asked: list = []
    ctx = _remote_context(cached_load_results(RECO_RUN), asked)
    assert ctx.labels == ["baseline", "variant_truth_tracking"]
    load = section_dispatch.history(ctx, cached_load_trend_event_timing, EVENTS)
    assert asked == []
    trend, reliability = load()
    # The view's own files, then the results that judge them and the hosts.
    assert asked[0] == EVENTS and set(asked[1:]) == {RUN_BASE}
    assert set(trend["label"]) == {"baseline", "variant_truth_tracking"}
    assert reliability == {"reco_run": True}
    load()
    assert len(asked) == 4  # the trend again, the results and reliability kept


def test_a_run_without_results_takes_its_labels_from_the_window():
    asked: list = []
    ctx = _remote_context(None, asked)
    assert ctx.labels == ["baseline", "variant_truth_tracking"]
    assert asked == [RUN_BASE]


def _section_app(dashboard_dir) -> None:
    # No annotations: AppTest.from_function execs this body as a bare script.
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    import section_dispatch

    names = ["Overview", "Run Trends", "Logs"]
    # What the sidebar sees, before the bar is drawn.
    st.session_state["before_bar"] = section_dispatch.current_section(names)
    st.session_state["drawn"] = section_dispatch.pick_section(names)


def _run_sections(tab=None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(
        _section_app, args=(str(_DASHBOARD_DIR),), default_timeout=30,
    )
    if tab is not None:
        at.query_params["tab"] = tab
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_section_is_known_before_the_bar_is_drawn():
    at = _run_sections("run trends")
    assert at.session_state["before_bar"] == at.session_state["drawn"] == "Run Trends"


def test_a_click_on_the_bar_is_known_before_the_bar_is_drawn():
    at = _run_sections()
    assert at.session_state["before_bar"] == "Overview"
    at.button_group[0].set_value("Logs").run()
    assert at.session_state["before_bar"] == at.session_state["drawn"] == "Logs"
