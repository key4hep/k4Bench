"""The Reconstruction page: the ``reco`` suite's k4run benchmarks.

The k4Bench auditor writes each label's ``_events.json`` in the ddsim event
format, so the per-event timing and memory tabs, Machine Info and Logs draw a
reco run as they draw a sim run, and are reused here as they are.
"""
from __future__ import annotations

from k4bench.layout import RECO
from section_dispatch import RENDERERS
from sections import SECTION_SCOPE
from suite_page import SuitePage, render_page

#: The sim sections this page shares, which read nothing sim-specific.
_SHARED = ("Event Timing", "Event Memory", "Machine Info", "Logs")

RECO_PAGE = SuitePage(
    suite=RECO,
    sections=_SHARED,
    remote_only=frozenset(),
    scopes={name: SECTION_SCOPE[name] for name in _SHARED},
    renderers={name: RENDERERS[name] for name in _SHARED},
    key_prefix="reco_",
    # No nightly report judges the reco suite yet.
    reports=False,
)


def main() -> None:
    render_page(RECO_PAGE)
