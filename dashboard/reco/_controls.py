"""The controls the Reconstruction page's sections share."""
from __future__ import annotations

import streamlit as st


def select_config(labels: list[str], key: str) -> str:
    """The configuration a view draws, chosen among *labels* in a dropdown
    bound to the URL under *key*; the only one, without a dropdown, when there
    is nothing to choose."""
    if len(labels) == 1:
        return labels[0]
    return st.selectbox("Configuration", labels, key=key, bind="query-params")
