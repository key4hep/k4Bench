"""Reconstruction Overview: every reco detector's whole-job time and memory,
side by side, as the Simulation Overview compares the simulated ones.

It draws the Simulation Overview's Performance Trends and Landscape views
(:func:`tabs.detectors_overview.render_figures`). No nightly report judges the
reco suite, so their inputs come from the runs themselves: every detector's
runs of the sidebar's platform and sample in the trend window, read with the
same trend builders Run Trends uses, and their values carry no verdict.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data import (
    cached_load_trend_event_timing,
    cached_load_trend_machine_info,
    cached_load_trend_results,
)
from k4bench.layout import RECO
from k4bench.results.reliability_evidence import run_reliability_map
from remote_cache import (
    cached_fetch_runs_windowed,
    cached_list_detectors,
    cached_list_run_dates,
)
from section_dispatch import PageContext
from tabs import detectors_overview
from tabs._overview_data import run_history_rows, run_overview_inputs, run_reliability_rows
from trend_window import release_day, runs_in_window


def _detector_rows(
    ctx: PageContext, detector: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """*detector*'s history and reliability rows in the window, or ``None``
    when it has no run of the sidebar's platform and sample there."""
    sel, (start, end) = ctx.remote, ctx.window.dates
    try:
        stacks_dates = cached_list_run_dates(
            ctx.config.data_url, RECO, detector, sel.platform, sel.sample,
        )
    except Exception:
        return None
    releases = [
        stack for stack in stacks_dates
        if (day := release_day(stack)) is not None and start <= day <= end
    ]
    windowed = runs_in_window(stacks_dates, releases)
    if not windowed:
        return None
    run_dirs = cached_fetch_runs_windowed(
        ctx.config.data_url, RECO, detector, sel.platform, sel.sample,
        ctx.config.cache_dir, windowed,
    )
    results = cached_load_trend_results(run_dirs) if run_dirs else None
    if results is None:
        return None
    reliability = run_reliability_map(results, cached_load_trend_machine_info(run_dirs))
    return (
        run_history_rows(detector, results, cached_load_trend_event_timing(run_dirs), reliability),
        run_reliability_rows(detector, results, reliability),
    )


def render(ctx: PageContext) -> None:
    """The Overview of every reco detector benchmarked on the sidebar's
    platform and sample."""
    if ctx.window.dates is None:
        st.info("No reconstruction run has been uploaded yet.")
        return
    try:
        detectors = cached_list_detectors(ctx.config.data_url, RECO)
    except Exception as err:
        st.warning(f"Could not list the reconstruction detectors: {err}")
        return
    rows = [r for d in detectors if (r := _detector_rows(ctx, d)) is not None]
    sel = ctx.remote
    detectors_overview.render_figures(
        run_overview_inputs(
            pd.concat([hist for hist, _ in rows], ignore_index=True)
            if rows else run_history_rows("", None, None, {}),
            pd.concat([rel for _, rel in rows], ignore_index=True)
            if rows else run_reliability_rows("", None, {}),
            platform=sel.platform, sample=sel.sample,
        ),
        key_prefix="reco_",
    )
