"""The dashboard's section bar: which sections exist, in what order, and which
need remote data.

Two independent lists, deliberately. :data:`SECTION_NAMES` is a presentation
choice — broadest first (across detectors, over time), then narrowing into one
run's internals, then the forensics for judging a number, which is the order
the questions are actually asked in. :data:`REMOTE_ONLY` is a statement of
fact about data sources. Deriving one from the other — hiding a positional
prefix, say — would couple them, so reordering the bar or inserting a section
could silently strand a tab with no data behind it, or hide a working one.

Kept out of ``app.py`` because that module ends in a bare ``main()`` call (a
Streamlit script is exec'd, not imported), so reading the registry from there
would run the entire app — including its network fetches — as a side effect.
"""

from __future__ import annotations

#: Every section, in display order.
SECTION_NAMES = [
    "Overview",
    "Run Trends",
    "Regressions",
    "Stack Changes",
    "Config Impact",
    "Region Timing",
    "Event Timing",
    "Event Memory",
    "Machine Info",
    "Logs",
]

#: Sections that need multi-run (remote) data and cannot work against a single
#: local run directory: the cross-detector views, the trend views, and Stack
#: Changes, which compares two Key4hep releases.
REMOTE_ONLY = frozenset({
    "Overview",
    "Run Trends",
    "Regressions",
    "Stack Changes",
    "Config Impact",
})


def visible_sections(trends_enabled: bool) -> list[str]:
    """The sections to offer, in display order.

    Gated on *remote mode* rather than on whether the current trend window
    happens to have data: that keeps the section set stable as the window
    changes, so the active section survives a window tweak, and an empty window
    shows an in-view "widen the window" message instead of a vanishing tab.
    """
    return [s for s in SECTION_NAMES if trends_enabled or s not in REMOTE_ONLY]
