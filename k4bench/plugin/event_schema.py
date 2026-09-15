"""Version contract of the event timing plugin's ``<label>_events.json``.

A file without ``schema_version`` is the legacy unversioned event format and
is read as before. Both the analysis loader and the runner validate through
:func:`validate_event_schema`, so a new version is accepted or refused in one
place.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

#: Matches ``kEventSchemaVersion`` in ``plugin/k4BenchTimingAction.cpp``.
EVENT_SCHEMA_VERSION = 1


def validate_event_schema(raw: Mapping, *, source: str | Path | None = None) -> None:
    """Raise ``ValueError`` unless *raw* is an event file this k4bench can read.

    An absent ``schema_version`` is accepted as the legacy unversioned format.
    A present one must be a plain ``int`` (``bool`` and ``float`` are refused)
    between 1 and :data:`EVENT_SCHEMA_VERSION`; a newer version is refused
    rather than parsed as a format it may not be.
    """
    if "schema_version" not in raw:
        return
    where = f"{source}: " if source is not None else ""
    version = raw["schema_version"]
    if type(version) is not int or version < 1:
        raise ValueError(f"{where}malformed schema_version {version!r}")
    if version > EVENT_SCHEMA_VERSION:
        raise ValueError(
            f"{where}unsupported future schema_version {version} "
            f"(this k4bench reads <= {EVENT_SCHEMA_VERSION})"
        )
