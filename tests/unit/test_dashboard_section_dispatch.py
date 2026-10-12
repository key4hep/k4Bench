"""The section renderer table: one entry per section, in bar order, and the
context it draws from refusing a remote-only read in local mode. Also the
active section, which the sidebar reads before the bar is drawn.

The table is a hand-maintained registry next to the three in ``sections.py``,
so it can drift the same way: a section added to the bar but not here would
raise on selection, one left here after a rename would never be drawn.
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
from data import RunData  # noqa: E402
from sidebar import TrendWindow  # noqa: E402


def test_every_section_has_a_renderer_in_bar_order():
    assert list(section_dispatch.RENDERERS) == sections.SECTION_NAMES


def _local_context() -> section_dispatch.PageContext:
    return section_dispatch.PageContext(
        config=Config(data_dir="run", data_url=None, cache_dir="cache", dashboard_url=""),
        selection=None,
        data_dir="run",
        run_meta=None,
        run=RunData(None, None, None, None, ["baseline"]),
        window=TrendWindow(),
        trend_machine=None,
        reliability={},
    )


def test_local_mode_has_no_trends_and_no_store_selection():
    ctx = _local_context()
    assert not ctx.trends_enabled
    with pytest.raises(RuntimeError, match="remote-only section"):
        ctx.remote


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
