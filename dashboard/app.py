"""The dashboard's entry point: one page per suite, chosen in the top bar.

Each page walks its own suite's run tree (see :mod:`suite_page`). Simulation is
the default page, served at ``/``, so the nightly mail's and pull-request
comments' ``/?tab=…`` deep links open it; every other page has its own path.
"""
from __future__ import annotations

import streamlit as st

import sim_page
from reco import page as reco_page
from ui_chrome import configure_page

#: The top bar, in display order.
PAGES = [
    st.Page(sim_page.main, title="Simulation", icon=":material/blur_on:", default=True),
    st.Page(
        reco_page.main, title="Reconstruction", icon=":material/account_tree:",
        url_path="reconstruction",
    ),
]

configure_page()
st.navigation(PAGES, position="top").run()
