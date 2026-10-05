"""Version contracts of the timing plugins' JSON artifacts.

Each plugin stamps its output with a ``schema_version``: the event timing
plugin's ``<label>_events.json`` and the region timing plugin's
``<label>_regions.json``. Every reader validates through this module, so a new
version of either format is accepted or refused in one place.

A file without ``schema_version`` is the legacy unversioned format of its kind
and is read as before.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

#: Matches ``kEventSchemaVersion`` in ``plugin/k4BenchTimingAction.cpp``.
EVENT_SCHEMA_VERSION = 1

#: Matches the ``schema_version`` written by ``plugin/k4BenchRegionTimingAction.cpp``.
REGION_SCHEMA_VERSION = 1


def validate_event_schema(raw: object, *, source: str | Path | None = None) -> None:
    """Raise ``ValueError`` unless *raw* is an event file this k4bench can read.

    See :func:`_validate_schema` for the rules.
    """
    _validate_schema(raw, kind="event", current=EVENT_SCHEMA_VERSION, source=source)


def validate_region_schema(raw: object, *, source: str | Path | None = None) -> None:
    """Raise ``ValueError`` unless *raw* is a region file this k4bench can read.

    See :func:`_validate_schema` for the rules.
    """
    _validate_schema(raw, kind="region", current=REGION_SCHEMA_VERSION, source=source)


def _validate_schema(
    raw: object,
    *,
    kind: str,
    current: int,
    source: str | Path | None,
) -> None:
    """Raise ``ValueError`` unless *raw* is a *kind* file of a readable version.

    *raw* is the parsed JSON document; a root that is not an object is refused.

    An absent ``schema_version`` is accepted as the legacy unversioned format.
    A present one must be a plain ``int`` (``bool`` and ``float`` are refused)
    between 1 and *current*; a newer version is refused rather than parsed as
    a format it may not be.
    """
    where = f"{source}: " if source is not None else ""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}{kind} JSON root must be an object")
    if "schema_version" not in raw:
        return
    version = raw["schema_version"]
    if type(version) is not int or version < 1:
        raise ValueError(f"{where}malformed schema_version {version!r}")
    if version > current:
        raise ValueError(
            f"{where}unsupported future schema_version {version} "
            f"(this k4bench reads <= {current})"
        )
