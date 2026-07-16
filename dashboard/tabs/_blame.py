"""Blame-window overlay for a confirmed regression's drill-down chart.

A ``CONFIRMED`` verdict carries the release window the step entered in —
``onset_*`` and ``last_accepted_*``, written by :mod:`k4bench.regression.engine`
and documented on :class:`~k4bench.regression.models.MetricVerdict`. This module
turns that pair into what the reader sees: a shaded band on the drill-down and a
link into the release diff that spans it.

Every "does this verdict have a window, and of what shape?" question routes
through :func:`classify`, so the band, the note and the caller's gate all read
one interpretation rather than re-deriving it. Reports written before onset
tracking carry ``None`` and classify as :attr:`WindowKind.NONE`; the caller then
keeps its pre-existing behaviour.
"""

from __future__ import annotations

from enum import Enum

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
_WINDOW_LABEL = "change entered here"
_WINDOW_LABEL_SIZE = 10


class WindowKind(Enum):
    """The shape of a confirmed regression's blame window.

    ``NONE``       — not a confirmed regression, or a report predating onset
                     tracking: no window to draw.
    ``SAME_STACK`` — onset and baseline are the *same* Key4hep release, so the
                     software stack did not move across the step.
    ``OPEN``       — no settled level precedes the step, so the window is open
                     on the older end (also the safe home for a corrupt window
                     whose baseline is newer than its onset).
    ``BOUNDED``    — a real ``(baseline, onset]`` release range.
    """

    NONE = "none"
    SAME_STACK = "same_stack"
    OPEN = "open"
    BOUNDED = "bounded"


def classify(verdict: MetricVerdict) -> WindowKind:
    """Reduce a verdict's onset/baseline pair to a single :class:`WindowKind`.

    Release date is the stack identity here — the premise the Stack Changes tab
    is built on: a Key4hep release maps to one immutable CVMFS stack, so two
    runs sharing a release measured the same packages. ISO dates order
    chronologically as plain strings, so no parsing is needed to compare them.
    """
    if verdict.severity is not Severity.CONFIRMED or verdict.onset_run_date is None:
        return WindowKind.NONE
    baseline, onset = verdict.last_accepted_run_date, verdict.onset_run_date
    if baseline is None:
        return WindowKind.OPEN
    if baseline == onset:
        return WindowKind.SAME_STACK
    if baseline > onset:
        # Baseline newer than onset is impossible from the engine; treat a
        # corrupt report as open-ended rather than trust the bad bound.
        return WindowKind.OPEN
    return WindowKind.BOUNDED


def has_window(verdict: MetricVerdict) -> bool:
    """True when *verdict* has any blame window to render (band and/or note)."""
    return classify(verdict) is not WindowKind.NONE


def onset_point(df: pd.DataFrame, verdict: MetricVerdict) -> tuple | None:
    """The plotted ``(x_date, value)`` of the recorded onset night, or ``None``.

    Matches the exact onset *run* first — several runs can share a release, so
    the release date alone is ambiguous — then falls back to the release date
    for a run that predates run-id capture. The pick is made by an explicit
    sort, not by the frame's incoming order, so a caller that sorts differently
    cannot move the marker. Returns ``None`` when the onset run is not in the
    fetched window (or its value is missing), so the caller draws no marker
    rather than a misplaced one.
    """
    if verdict.metric not in df.columns:
        return None
    for col, want in (("run_id", verdict.onset_run_id), ("x_date", verdict.onset_run_date)):
        if want is None or col not in df.columns:
            continue
        key = pd.to_datetime(df[col]) if col == "x_date" else df[col].astype(str)
        hits = df[key == (pd.to_datetime(want) if col == "x_date" else str(want))]
        if hits.empty:
            continue
        # run_id is unique per run (one row); the x_date fallback can match
        # several runs sharing a release — order them deterministically and take
        # the newest so the result never depends on the caller's row order.
        order = ["x_date"] + (["run_id"] if "run_id" in hits.columns else [])
        row = hits.sort_values(order).iloc[-1]
        val = row[verdict.metric]
        return None if pd.isna(val) else (row["x_date"], val)
    return None


def add_window_band(fig: go.Figure, df: pd.DataFrame, verdict: MetricVerdict) -> None:
    """Shade the release range the change entered in, mirroring the drill-down's
    horizontal baseline band. Draws nothing for a same-release or absent window;
    an open window starts at the earliest plotted release rather than inventing
    a bound.
    """
    kind = classify(verdict)
    if kind in (WindowKind.NONE, WindowKind.SAME_STACK):
        return
    onset = pd.to_datetime(verdict.onset_run_date)
    if kind is WindowKind.BOUNDED:
        x0 = pd.to_datetime(verdict.last_accepted_run_date)
    else:  # OPEN: no trustworthy baseline — span from the earliest plotted run
        x0 = pd.to_datetime(df["x_date"]).min()
    if pd.isna(x0) or x0 >= onset:
        return  # nothing to span
    fig.add_vrect(
        x0=x0, x1=onset, fillcolor=_WINDOW_FILL, line_width=0, layer="below",
        annotation_text=_WINDOW_LABEL, annotation_position="top left",
        annotation_font_size=_WINDOW_LABEL_SIZE,
    )


def render_note(verdict: MetricVerdict) -> None:
    """The below-chart blame note: "nothing upstream changed" when the window's
    ends are the same release, otherwise a link into the Stack Changes tab
    seeded with the exact release range to inspect.
    """
    from tabs.stack_changes import deep_link  # local: keep this module's import light

    kind = classify(verdict)
    onset = verdict.onset_run_date

    if kind is WindowKind.SAME_STACK:
        # Same release on both ends ⇒ the stack did not move across the step, so
        # an upstream commit is ruled out — a sharper answer than any PR list.
        # (The nightly build skips days; a run then re-uses the newest release,
        # so consecutive runs sharing a stack is common, not an error.)
        st.info(
            f"**Nothing upstream changed.** The step appeared between two runs that "
            f"measured the **same** Key4hep release ({onset}), so no package moved "
            "across it: the cause is the host, the sample, or noise — not an upstream "
            "commit.",
            icon="✅",
        )
        return

    baseline = verdict.last_accepted_run_date if kind is WindowKind.BOUNDED else None
    span = f"**{baseline} → {onset}**" if baseline else f"up to **{onset}**"
    st.caption(
        f"The step first appeared on **{onset}**, one reliable night before it "
        f"confirmed — so the cause landed in the shaded window ({span}), "
        "not on the reported night."
    )
    st.link_button(
        "🔍 Compare upstream changes in this window →",
        deep_link(platform=verdict.platform, head_release=onset, base_release=baseline),
        help="Open Stack Changes seeded with this release range — every Key4hep "
             "package that moved across the window, with a link to each commit diff."
             + ("" if baseline else " Pick a baseline release there: this step has no "
                "settled level before it to bound the window on."),
    )
