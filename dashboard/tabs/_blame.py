"""Blame-window overlay for a confirmed regression's drill-down chart.

A ``CONFIRMED`` verdict carries the window the change entered in: ``onset_*``
(the night it first crossed the gates) and ``last_accepted_*`` (the newest
night before that observed at the then-accepted level). Confirmation is a
two-strike rule, so the night a regression is *reported* is one reliable night
after the night it *appeared* — the cause is upstream of ``onset``, never at
the report. This module shades ``(last_accepted, onset]`` on the drill-down and
points the reader at the exact release diff that spans it.

The verdict fields are written by :mod:`k4bench.regression.engine`; reports
written before onset tracking existed carry ``None`` for all four, and this
module then renders nothing — the caller keeps its pre-existing behaviour.
"""

from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from k4bench.regression.models import MetricVerdict, Severity
from tabs._regression_flags import FLAG_MARKS
from ui_utils import _to_rgba

#: Fill for the onset-window band — the same amber the ⚠️ Watch marker uses, at
#: low alpha, so the shaded region reads as "where the ⚠️→🔴 step came from"
#: rather than a second, unrelated highlight.
_WINDOW_FILL = _to_rgba(FLAG_MARKS["WATCH"]["color"], 0.12)


def has_window(verdict: MetricVerdict) -> bool:
    """True when *verdict* is a confirmed regression carrying a recorded onset.

    The onset date is the one field the band and the link both need; a report
    predating onset tracking has it as ``None`` and gets neither.
    """
    return verdict.severity is Severity.CONFIRMED and verdict.onset_run_date is not None


def onset_point(df: pd.DataFrame, verdict: MetricVerdict) -> tuple | None:
    """The plotted ``(x_date, value)`` of the recorded onset night, or ``None``.

    Matches the exact onset *run* first — several runs can share a release, so
    the release date alone is ambiguous — then falls back to the release date
    for a run that predates run-id capture. Returns ``None`` when the onset run
    is not in the fetched window, so the caller draws no marker rather than a
    misplaced one.
    """
    for col, want in (("run_id", verdict.onset_run_id), ("x_date", verdict.onset_run_date)):
        if want is None or col not in df.columns:
            continue
        key = pd.to_datetime(df[col]) if col == "x_date" else df[col].astype(str)
        hits = df[key == (pd.to_datetime(want) if col == "x_date" else str(want))]
        if not hits.empty:
            row = hits.iloc[-1]
            val = row[verdict.metric]
            return None if pd.isna(val) else (row["x_date"], val)
    return None


def add_window_band(fig: go.Figure, df: pd.DataFrame, verdict: MetricVerdict) -> None:
    """Shade ``(last_accepted, onset]`` — the release range the change entered
    in — as a vertical band, mirroring the drill-down's horizontal baseline
    band. When no prior accepted night exists the window is open on the left, so
    the band starts at the earliest plotted release rather than inventing a
    bound.
    """
    onset = pd.to_datetime(verdict.onset_run_date)
    if verdict.last_accepted_run_date is not None:
        x0 = pd.to_datetime(verdict.last_accepted_run_date)
    else:
        x0 = pd.to_datetime(pd.Series(df["x_date"])).min()
    if pd.isna(x0) or x0 >= onset:
        return  # same release (nothing to span) or window collapses to a line
    fig.add_vrect(
        x0=x0, x1=onset, fillcolor=_WINDOW_FILL, line_width=0, layer="below",
        annotation_text="change entered here", annotation_position="top left",
        annotation_font_size=10,
    )


def _release_span(verdict: MetricVerdict) -> str:
    lo = verdict.last_accepted_run_date
    hi = verdict.onset_run_date
    return f"**{lo} → {hi}**" if lo else f"up to **{hi}**"


def render_note(verdict: MetricVerdict) -> None:
    """The below-chart blame note: either "nothing upstream changed" when the
    window's ends are the same release, or a link into the Stack Changes tab
    seeded with the exact release range to inspect.
    """
    onset = verdict.onset_run_date
    baseline = verdict.last_accepted_run_date

    if baseline == onset:
        # Both nights measured the same Key4hep release, so the stack did not
        # move across the step — an upstream commit is ruled out, and that is a
        # sharper answer than any PR list. (The nightly build skips days; a run
        # then re-uses the newest release, so consecutive runs sharing a stack
        # is common, not an error.)
        st.info(
            f"**Nothing upstream changed.** The step appeared between two runs that "
            f"measured the **same** Key4hep release ({onset}), so no package moved "
            "across it: the cause is the host, the sample, or noise — not an upstream "
            "commit.",
            icon="✅",
        )
        return

    st.caption(
        f"The step first appeared on **{onset}**, one reliable night before it "
        f"confirmed — so the cause landed in the shaded window ({_release_span(verdict)}), "
        "not on the reported night."
    )
    params = {"tab": "Stack Changes", "platform": verdict.platform, "to": onset}
    if baseline:
        params["from"] = baseline
    st.link_button(
        "🔍 Compare upstream changes in this window →",
        "?" + urlencode(params),
        help="Open Stack Changes seeded with this release range — every Key4hep "
             "package that moved across the window, with a link to each commit diff."
             + ("" if baseline else " Pick a baseline release there: this step has no "
                "settled level before it to bound the window on."),
    )
