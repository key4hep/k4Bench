"""The section renderer table: one entry per section, in bar order, and the
context it draws from refusing a remote-only read in local mode.

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
