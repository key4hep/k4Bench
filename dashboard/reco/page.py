"""The Reconstruction page: the ``reco`` suite's k4run benchmarks.

The k4Bench auditor writes each label's ``_events.json`` in the ddsim event
format, so the per-event timing and memory tabs and Machine Info draw a reco
run as they draw a sim run, and are reused here as they are. Logs is reused
too, reading the log as Gaudi's. The Overview compares the reco detectors as
the Simulation Overview does, and the Breakdown splits the job's time by phase
and by component.
"""
from __future__ import annotations

from k4bench.layout import RECO
from reco import breakdown, overview
from section_dispatch import RENDERERS, PageContext
from sections import SECTION_SCOPE
from suite_page import SuitePage, render_page
from tabs import logs

#: The sim sections this page shares, which read nothing sim-specific.
_SHARED = ("Event Timing", "Event Memory", "Machine Info")


def _logs(ctx: PageContext) -> None:
    """Per-config status and the log explorer, reading Gaudi's messages."""
    logs.render(ctx.run.results, ctx.data_dir, ctx.run_meta, log_format=logs.GAUDI)


RECO_PAGE = SuitePage(
    suite=RECO,
    sections=("Overview", "Breakdown", *_SHARED, "Logs"),
    # Like the Simulation Overview, it compares detectors across the window.
    remote_only=frozenset({"Overview"}),
    scopes={
        "Overview": SECTION_SCOPE["Overview"],
        "Breakdown": SECTION_SCOPE["Region Timing"],
        **{name: SECTION_SCOPE[name] for name in (*_SHARED, "Logs")},
    },
    renderers={
        "Overview": overview.render,
        "Breakdown": breakdown.render,
        **{name: RENDERERS[name] for name in _SHARED},
        "Logs": _logs,
    },
    key_prefix="reco_",
    # No nightly report judges the reco suite yet.
    reports=False,
    components=True,
)


def main() -> None:
    render_page(RECO_PAGE)
