"""Shared helpers and constants for the Region Timing views.

Anything used by more than one of the four region views (attribution,
current-run, historical, step-analysis) lives here so each view module stays
focused on its own figure.
"""
from __future__ import annotations

import streamlit as st

from ui_utils import _PALETTE_NAMES

# Attribution help text — shown verbatim by the current-run selectbox and the
# historical radio. Hoisted to a single constant so the wording stays in sync.
_ATTRIBUTION_HELP = (
    "**At location** — time is charged to the detector region where the "
    "particle *deposited* its energy. Shows which regions are most "
    "expensive to simulate.\n\n"
    "**By birth** — time is charged to the detector region where the "
    "particle was *created*. Shows which regions produce the costliest "
    "secondary particles."
)


def _palette_placeholder(col, key: str) -> None:
    """Render a default ("Matplotlib", index 0) palette selectbox in *col*.

    Used on early-return code paths that bail out *before* the real palette
    selectbox would be created. Registering the widget here keeps its
    ``session_state`` key alive so the selection isn't garbage-collected as
    stale when the view later has data again.
    """
    with col:
        st.selectbox("Colour palette", options=_PALETTE_NAMES, index=0, key=key)
