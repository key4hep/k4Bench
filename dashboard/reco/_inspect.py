"""Opening one component of the selected run in the Components section, from
wherever the Reconstruction page names it.

Every section that draws components offers the same action (:func:`inspect`):
it opens the Components section on that configuration and component, and
remembers the section it came from, which the Components section offers to go
back to. That section's own controls keep their values meanwhile, so going
back finds it as it was left.

A component is named by its name in the run, which the auditor makes unique
within a configuration, never by a label drawn from it or by its place in a
chart.
"""
from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from url_params import COMPONENT

#: The section listing the components and inspecting one.
SECTION = "Components"
#: Its view of the run on screen.
CURRENT_RUN = "Current Run"
#: The Reconstruction page's section bar (``suite_page`` keys it by the
#: page's prefix).
SECTION_KEY = "reco_active_section"
#: The Components section's view, configuration, component and the section it
#: was opened from.
VIEW_KEY = "reco_component_view"
CONFIG_KEY = "reco_component_config"
COMPONENT_KEY = "reco_component_name"
ORIGIN_KEY = "reco_component_origin"
#: Bumped whenever the component is chosen anywhere but in the Components
#: table, which then starts again from that component's row.
TABLE_NONCE_KEY = "reco_component_table_nonce"


def inspect(name: str | None, config: str, origin: str) -> None:
    """Open component *name* of configuration *config* in the Components
    section's Current Run, coming from section *origin*, which shows the run
    on screen; ``None`` opens it on the component it last showed. A widget
    callback: it runs before the rerun draws the section bar."""
    st.session_state[SECTION_KEY] = SECTION
    st.session_state[VIEW_KEY] = CURRENT_RUN
    st.session_state[CONFIG_KEY] = config
    st.session_state[ORIGIN_KEY] = origin
    if name is not None:
        st.session_state[COMPONENT_KEY] = name
        st.query_params[COMPONENT] = name
    st.session_state[TABLE_NONCE_KEY] = st.session_state.get(TABLE_NONCE_KEY, 0) + 1


def go_back() -> None:
    """Return to the section the Components section was opened from."""
    origin = st.session_state.pop(ORIGIN_KEY, None)
    if origin:
        st.session_state[SECTION_KEY] = origin


def inspect_button(
    name: str | None, config: str, origin: str, *, key: str, label: str = "Inspect component",
) -> None:
    """A button opening *name* in the Components section."""
    st.button(
        label, key=key, type="tertiary", icon=":material/manage_search:",
        on_click=inspect, args=(name, config, origin),
        help=(f"Open {name} in Components: its time, counters, allocations, heap, "
              "collections, code and history.") if name else None,
    )


def clicked_name(event, fig) -> str | None:
    """The name a click in *fig* chose: the first of its point's
    ``customdata``, found by the curve and point Streamlit reports; ``None``
    for no click, or a click on a point that carries none (an "Other")."""
    selection = getattr(event, "selection", None) or {}
    for point in selection.get("points", []):
        curve, index = point.get("curve_number"), point.get("point_index")
        if curve is None or index is None or not 0 <= curve < len(fig.data):
            continue
        customdata = fig.data[curve].customdata
        if customdata is not None and 0 <= index < len(customdata) and customdata[index][0]:
            return str(customdata[index][0])
    return None


def clickable_chart(
    fig, key: str, label: str, origin: str, *,
    on_click: Callable[[str], None] | None = None, **kwargs,
) -> None:
    """Draw *fig*, a click on a point carrying a component's name
    (:func:`clicked_name`) opening it, of configuration *label*, in the
    Components section, or handing it to *on_click* instead. After any
    click the chart is drawn anew, without a selection, so a click on the
    same point is a click again. *kwargs* go to ``st.plotly_chart``."""
    nonce_key = f"{key}_nonce"
    widget = f"{key}_{st.session_state.get(nonce_key, 0)}"

    def clicked() -> None:
        st.session_state[nonce_key] = st.session_state.get(nonce_key, 0) + 1
        if (name := clicked_name(st.session_state.get(widget), fig)) is not None:
            if on_click is not None:
                on_click(name)
            else:
                inspect(name, label, origin)

    st.plotly_chart(fig, key=widget, on_select=clicked, selection_mode="points", **kwargs)
