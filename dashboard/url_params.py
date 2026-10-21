"""The page URL's parameters that name what the sidebar selected: the one place
that spells them.

A copied address-bar link, the nightly mail's and the pull-request comments'
deep links are all read back through these names, so a link reopens the same
measurement and the same history:

=============  ================================================================
``tab``        the section, matched case-insensitively
``detector``   the detector, e.g. ``CLD_o2_v09``
``platform``   the platform, e.g. ``x86_64-el9-gcc16-opt``
``sample``     the sample, e.g. ``p8_ee_Zbb_ecm91``
``stack``      the Key4hep release directory, e.g. ``key4hep-2026-10-17``
``run``        the run: the night it was measured, its directory name,
               e.g. ``2026-10-21``
``releases``   the trend window: its oldest and newest release directories,
               ``key4hep-2026-09-22..key4hep-2026-10-17``
``history``    which runs a history draws per release: ``latest`` (the
               default) or ``all``
``timing``     Event Timing's "Timing of" choices, one value each: the
               algorithms' names, or :data:`WHOLE_EVENT`
``component``  reserved: one component of the selected run, by its name
=============  ================================================================

The suite is not a parameter: it is the page's path (Simulation at ``/``,
Reconstruction at ``/reconstruction``). Every value is the store's raw
identifier, never a label drawn from it, so a link survives a change of
wording or of the run's timings. The controls inside a section are bound to
the URL under their widget keys (``bind="query-params"``) and are named there.

Pure, so the parsing is unit-tested without Streamlit.
"""
from __future__ import annotations

from datetime import date

from k4bench.layout import RunRef

TAB = "tab"
DETECTOR = "detector"
PLATFORM = "platform"
SAMPLE = "sample"
STACK = "stack"
RUN = "run"
RELEASES = "releases"
HISTORY = "history"
TIMING = "timing"
#: Reserved for a component of the selected run; nothing reads it yet.
COMPONENT = "component"

#: ``history`` values: each release's latest run, or every run.
HISTORY_LATEST = "latest"
HISTORY_ALL = "all"
#: Each ``history`` value and how the sidebar names it, the default first.
#: Not "All runs", which the views' reliability filter already says for
#: keeping the unreliable runs.
HISTORY_MODES = {HISTORY_LATEST: "Latest run per release", HISTORY_ALL: "Every run"}

#: The ``timing`` value of the event's own time, before any algorithm.
WHOLE_EVENT = "Whole event"

#: What separates the trend window's two ends.
_RANGE = ".."


def run_params(ref: RunRef) -> dict[str, str]:
    """The parameters naming *ref* on its suite's page."""
    return {
        DETECTOR: ref.detector, PLATFORM: ref.platform, SAMPLE: ref.sample,
        STACK: ref.stack, RUN: ref.date,
    }


def is_run_id(value: str | None) -> bool:
    """Whether *value* spells a run: a ``YYYY-MM-DD`` date."""
    if not value or len(value) != 10:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def release_range(lo: str, hi: str) -> str:
    """The ``releases`` value of the window from *lo* to *hi*."""
    return f"{lo}{_RANGE}{hi}"


def parse_release_range(value: str | None) -> tuple[str, str] | None:
    """The two release directories a ``releases`` value names, or ``None``
    when it does not name two."""
    lo, sep, hi = (value or "").partition(_RANGE)
    if not (sep and lo and hi) or _RANGE in hi:
        return None
    return lo, hi
