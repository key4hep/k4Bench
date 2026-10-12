"""Pure trend-window resolution (no Streamlit), so it can be unit-tested.

The sidebar picks a range of Key4hep releases on a slider; this module gives
the range it opens on, and turns a range into the releases the window holds,
the report nights that judged them, and the run listing cut to the runs the
caller downloads for them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, timedelta

import pandas as pd

from k4bench.labels import pretty_release

#: How many of the newest releases the window opens on.
DEFAULT_RELEASES = 10


def _parse_day(raw: str) -> date | None:
    """The date a ``YYYY-MM-DD`` directory name spells, or ``None``."""
    parsed = pd.to_datetime(raw, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def to_dates(raw: Iterable[str]) -> list[date]:
    """Parse ``YYYY-MM-DD`` directory names to sorted, unique dates, dropping
    anything unparseable."""
    return sorted({day for day in map(_parse_day, raw) if day is not None})


def release_day(stack: str) -> date | None:
    """The date a ``key4hep-YYYY-MM-DD`` release directory was tagged, or ``None``."""
    return _parse_day(pretty_release(stack))


def dated_releases(stacks: Iterable[str]) -> list[str]:
    """Release directory names, unique and oldest first, dropping any whose
    name carries no date."""
    return sorted(
        {stack for stack in stacks if release_day(stack) is not None},
        key=lambda stack: (release_day(stack), stack),
    )


def default_releases(
    releases: list[str], anchor: date | None, count: int = DEFAULT_RELEASES,
) -> list[str]:
    """The releases the window opens on, as an oldest-first slice of *releases*.

    *releases* is every release benchmarked on the platform, oldest first (see
    :func:`dated_releases`), not the selected detector's: the cross-detector
    Overview draws every detector over this one window, so it has to open on
    the same releases whichever detector the sidebar has selected.

    That is the *count* newest releases tagged on or before *anchor* — the
    newest nightly **report** night — and every newer release on top. A release
    that tonight's batch is still uploading has no report yet, and counting it
    would push the oldest release, and with it the Overview's oldest night, out
    of the window until the report lands. The window therefore holds one
    release more than *count* while a batch outruns the reports. With no
    *anchor*, the count starts at the newest release.
    """
    reported = [r for r in releases if anchor is None or release_day(r) <= anchor]
    ahead = [r for r in releases if anchor is not None and release_day(r) > anchor]
    return reported[-count:] + ahead


def releases_between(releases: list[str], lo: str, hi: str) -> list[str]:
    """*lo*, *hi* and every release of *releases* between them, oldest first."""
    return releases[releases.index(lo):releases.index(hi) + 1]


def report_span(
    selected: list[str], releases: list[str], last_night: date
) -> tuple[date, date]:
    """The inclusive ``(start, end)`` of the nights whose reports judged the
    *selected* releases.

    A release is benchmarked from the night it is tagged until the next one is:
    a night without a new tag re-runs the newest release. So the span starts on
    the oldest selected release's tag date and ends the night before the next
    newer release in *releases*, or on *last_night* — the newest night the
    store knows of — when the newest release is selected.
    """
    start = release_day(selected[0])
    newer = releases[releases.index(selected[-1]) + 1:]
    end = release_day(newer[0]) - timedelta(days=1) if newer else last_night
    return start, end


def runs_in_window(
    stacks_dates: Mapping[str, Iterable[str]], releases: Iterable[str]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The ``{stack: [date, ...]}`` run listing cut to the window's *releases*
    — every run of each, whichever night it ran on — as the hashable
    ``((stack, (date, ...)), ...)`` the windowed download caches on: stacks and
    dates sorted, run directories without a date and stacks left with no run
    dropped."""
    wanted = set(releases)
    windowed = {
        stack: sorted(raw for raw in dates if _parse_day(raw) is not None)
        for stack, dates in stacks_dates.items()
        if stack in wanted
    }
    return tuple(
        (stack, tuple(dates)) for stack, dates in sorted(windowed.items()) if dates
    )
