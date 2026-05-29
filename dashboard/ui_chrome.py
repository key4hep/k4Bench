"""Page chrome and small sidebar utilities for the dashboard entry point.

Footers, the stale-selection helper — presentation/plumbing that would otherwise
clutter ``app.main()``.
"""
from __future__ import annotations

import streamlit as st


def _render_footer() -> None:
    """Render a CERN / FCC copyright footer at the bottom of the page."""
    st.markdown(
        """
        <hr style="border:none;border-top:1px solid rgba(128,128,128,0.25);margin:2.5rem 0 0.8rem 0;">
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            gap:1.2rem;
            padding:0.2rem 0 1.2rem 0;
            font-size:0.80rem;
            color:#9a9a9a;
            line-height:1.7;
            text-align:center;
        ">
            <span style="font-size:1.8rem;opacity:0.75;">⚛️</span>
            <div>
                <strong style="color:#c0c0c0;letter-spacing:0.02em;">© 2026 CERN</strong>
                &nbsp;·&nbsp;
                For the benefit of the&nbsp;<a
                    href="https://fcc.web.cern.ch/"
                    target="_blank"
                    style="color:#5b9bd5;text-decoration:none;font-weight:600;"
                >FCC project</a>
                <br>
                Created by <strong style="color:#c0c0c0;">Joshua Falco Beirer</strong>
                &nbsp;<span style="opacity:0.6;">(CERN)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_footer() -> None:
    """Render a compact attribution note at the bottom of the sidebar."""
    st.markdown(
        """
        <hr style="border:none;border-top:1px solid rgba(128,128,128,0.2);margin:1.5rem 0 0.6rem 0;">
        <div style="font-size:0.72rem;color:#888;text-align:center;line-height:1.6;padding-bottom:0.4rem;">
            <strong style="color:#a0a0a0;">© 2026 CERN</strong><br>
            For the benefit of the<br>
            <a href="https://fcc.web.cern.ch/" target="_blank"
               style="color:#5b9bd5;text-decoration:none;">FCC project</a><br>
            <span style="opacity:0.7;">J. F. Beirer (CERN)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _drop_stale_selection(key: str, options: list[str]) -> None:
    """Clear a keyed selectbox's stored value when it's no longer a valid option.

    The dependent dropdowns (Platform → Sample → Stack) rebuild their option lists
    whenever an upstream selection changes, so a value left in ``session_state``
    from the old options can be fed back into ``st.selectbox`` as an invalid
    selection. Popping it *before* the widget is created (the only point at which
    a widget-backed key may be mutated) lets the selectbox re-default cleanly to a
    valid option. A no-op when the stored value is still present in *options*.
    """
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
