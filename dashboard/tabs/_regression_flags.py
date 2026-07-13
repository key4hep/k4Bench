"""Shared regression-flag widgets for the historical trend views.

The Overview and Run Trends tabs both ring the nights the nightly detector
flagged a step against the baseline. Keeping the marker specs, the pills
control and the "nothing flagged in this window" notice here means the two
tabs read identically — same colours, same shapes, same wording — instead of
drifting apart in two hand-maintained copies.

The verdict *severities* themselves come from the precomputed nightly reports
(``_reports/{date}/report.json``); this module only draws them.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ui_utils import _to_rgba

#: Trend-flag marker specs keyed on verdict severity, matching the Regressions
#: tab's colour language (severity = attention level, red = confirmed, amber =
#: first flag). Each flag draws as two layers (see :func:`add_severity_markers`):
#: a soft translucent *halo* — the primary legibility fix, a colour-coded glow
#: that reads at a glance regardless of the line colour or symbol underneath it
#: — and a crisp white-bordered *badge* on top for the precise value, white
#: border chosen so it never blends into whatever line passes behind it (the
#: same white-outline device the Regressions tab's drill-down uses). Shape *and*
#: colour both carry the state, never colour alone.
FLAG_MARKS = {
    "CONFIRMED": dict(symbol="circle", badge_size=13, halo_size=28,
                      color="#d03b3b", label="🔴 Confirmed regression"),
    "WATCH":     dict(symbol="triangle-up", badge_size=12, halo_size=24,
                      color="#fab219", label="⚠️ Watch (unconfirmed)"),
}

_FLAG_HELP = (
    "Ring the nights the nightly detector confirmed a step beyond the "
    "baseline (Confirmed), or first flagged it but hasn't confirmed yet "
    "(Watch), on the trend lines — see the Regressions tab for the verdicts."
)

#: Attention ranking used to reduce same-nightly-tag reruns to their worst
#: verdict: a regression CONFIRMED on one run of a tag must not be masked by a
#: later rerun that re-anchored to OK. OK/UNKNOWN (rank 0) never outrank a flag.
SEVERITY_RANK = {"CONFIRMED": 3, "WATCH": 2, "FAILURE": 1}


def render_flag_pills(key: str) -> tuple[bool, bool]:
    """The Confirmed/Watch flag toggle, both on by default.

    Returns ``(show_confirmed, show_watch)`` — the two booleans the trend
    figures use to decide which severities to overlay.
    """
    flags = st.pills(
        "Regressions", ["🔴 Confirmed", "⚠️ Watch"], selection_mode="multi",
        default=["🔴 Confirmed", "⚠️ Watch"], key=key, help=_FLAG_HELP,
    ) or []
    return "🔴 Confirmed" in flags, "⚠️ Watch" in flags


def add_severity_markers(
    fig: go.Figure,
    flagged,
    *,
    x_col: str,
    y_col: str,
    name_col: str,
    severity: str,
    hover_y: str,
    row: int,
    col: int,
) -> None:
    """Overlay the two-layer flag marker for one *severity* onto *fig*.

    *flagged* is the already-filtered frame of points to ring (columns
    *x_col*/*y_col* for position, *name_col* for the hovered identity — the
    detector on Overview, the config label on Run Trends). *hover_y* is the
    plotly format string for the value line of the tooltip, matching whatever
    the panel's lines use. Draws a soft halo (no hover) then a crisp badge on
    top (carries the tooltip) — see :data:`FLAG_MARKS`.
    """
    mark = FLAG_MARKS[severity]
    fig.add_trace(
        go.Scatter(
            x=flagged[x_col], y=flagged[y_col],
            mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(symbol=mark["symbol"], size=mark["halo_size"],
                        color=_to_rgba(mark["color"], 0.28), line_width=0),
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=flagged[x_col], y=flagged[y_col],
            mode="markers", showlegend=False,
            marker=dict(symbol=mark["symbol"], size=mark["badge_size"],
                        color=mark["color"], line=dict(width=1.5, color="#ffffff")),
            customdata=flagged[name_col],
            hovertemplate=(
                f"{mark['label']}<br><b>%{{customdata}}</b> — "
                f"%{{x|%Y-%m-%d}}<br>{hover_y}<extra></extra>"
            ),
        ),
        row=row, col=col,
    )
