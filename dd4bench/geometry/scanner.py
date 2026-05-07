"""Scan a DD4hep compact geometry and extract subdetector names.

DD4hep geometries are split across many XML files linked by
``<include ref="..."/>`` tags.  This module resolves the full include
tree and collects every ``<detector name="...">`` element, in
encounter order, deduplicating across files.

"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from xml.dom import minidom
from xml.parsers.expat import ExpatError


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_detector_names(xml_path: Path) -> list[str]:
    """Return the names of all ``<detector>`` elements in the geometry.

    Recursively follows ``<include ref="..."/>`` tags starting from
    *xml_path*, collecting every ``<detector name="...">`` attribute
    found across all reachable files.  Order is encounter order;
    duplicates are suppressed.

    Parameters
    ----------
    xml_path:
        Path to the top-level compact XML file.

    Returns
    -------
    list[str]
        Detector names in the order they are first encountered.
    """
    all_files = resolve_includes(xml_path)
    names: list[str] = []
    seen: set[str] = set()

    for f in all_files:
        for name in _detector_names_in_file(f):
            if name not in seen:
                names.append(name)
                seen.add(name)

    return names


def resolve_includes(
    xml_path: Path,
    _visited: set[Path] | None = None,
) -> list[Path]:
    """Return all XML files reachable from *xml_path* via includes.

    Follows ``<include ref="..."/>`` tags recursively.  The returned
    list is in encounter order and contains no duplicates.  *xml_path*
    itself is always the first element.

    Includes whose ``ref`` attribute contains an unresolved environment
    variable (e.g. ``${DD4hepINSTALL}/...``) are skipped silently —
    ddsim resolves these at runtime using its own search path.

    Parameters
    ----------
    xml_path:
        Absolute or relative path to a DD4hep compact XML file.

    Returns
    -------
    list[Path]
        Resolved, deduplicated paths in encounter order.
    """
    if _visited is None:
        _visited = set()

    xml_path = xml_path.resolve()

    if xml_path in _visited:
        return []
    _visited.add(xml_path)

    collected: list[Path] = [xml_path]

    try:
        doc = minidom.parse(str(xml_path))
    except (ExpatError, OSError) as exc:
        warnings.warn(f"Could not parse {xml_path}: {exc}", stacklevel=2)
        return collected

    for node in doc.getElementsByTagName("include"):
        ref = node.getAttribute("ref")
        if not ref:
            continue

        # Skip refs that contain env vars (e.g. "${DD4hepINSTALL}/...") —
        # ddsim resolves these itself via its own search path.  Check the
        # original ref before expansion so this works regardless of whether
        # the variable happens to be set in the current environment.
        if "$" in ref:
            continue

        candidate = (xml_path.parent / os.path.expandvars(ref)).resolve()

        if not candidate.exists():
            warnings.warn(f"Included file not found: {candidate}", stacklevel=2)
            continue

        collected.extend(resolve_includes(candidate, _visited))

    return collected


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detector_names_in_file(xml_path: Path) -> list[str]:
    """Return ``<detector name="...">`` values found directly in *xml_path*."""
    try:
        doc = minidom.parse(str(xml_path))
    except (ExpatError, OSError):
        return []

    return [
        node.getAttribute("name")
        for node in doc.getElementsByTagName("detector")
        if node.getAttribute("name")
    ]
