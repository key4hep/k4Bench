"""Patch a DD4hep compact geometry to remove a single subdetector.

The patcher writes temporary XML files into the *same directory* as the
original geometry so that all relative includes (elements.xml,
materials.xml, etc.) continue to resolve correctly when ddsim loads the
patched geometry.

Temporary files are prefixed with ``_dd4bench_tmp_`` so they are easy
to identify and clean up.  The recommended usage is via the
:func:`patched_geometry` context manager, which guarantees cleanup even
if the simulation run raises an exception.

"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from xml.dom import minidom
from xml.parsers.expat import ExpatError

from dd4bench.geometry.scanner import resolve_includes

# Prefix for all temporary files written by this module.
_TMP_PREFIX = "_dd4bench_tmp_"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patched_geometry(xml_path: Path, detector_name: str):
    """Context manager that yields a patched geometry path.

    Creates temporary XML files with *detector_name* removed, yields the
    path to the patched top-level XML, then deletes the temp files on
    exit regardless of whether an exception was raised.

    Parameters
    ----------
    xml_path:
        Path to the original top-level compact XML.
    detector_name:
        Name of the ``<detector>`` element to remove.

    Yields
    ------
    Path
        Path to the patched top-level XML file.

    Raises
    ------
    DetectorNotFoundError
        If *detector_name* is not found in any reachable XML file.

    Example
    -------
    ::

        with patched_geometry(Path("ALLEGRO.xml"), "EcalBarrel") as tmp_xml:
            result = run_ddsim(xml_path=tmp_xml, ...)
    """
    top_tmp, sub_tmp = build_patched_xml(xml_path, detector_name)
    try:
        yield top_tmp
    finally:
        for tmp in (top_tmp, sub_tmp):
            if tmp is not None:
                tmp.unlink(missing_ok=True)


def build_patched_xml(
    xml_path: Path, detector_name: str
) -> tuple[Path, Path]:
    """Write patched XML files with *detector_name* removed.

    Locates the file that owns *detector_name*, removes the
    ``<detector>`` node from it, writes a temp copy, then writes a
    patched top-level XML whose include ref points at the temp copy.

    Parameters
    ----------
    xml_path:
        Path to the original top-level compact XML.
    detector_name:
        Name of the ``<detector>`` element to remove.

    Returns
    -------
    tuple[Path, Path]
        ``(top_tmp_path, sub_tmp_path)`` — the caller is responsible for
        deleting both files.  Prefer :func:`patched_geometry` to handle
        cleanup automatically.

    Raises
    ------
    DetectorNotFoundError
        If *detector_name* is not found in any reachable XML file.
    """
    xml_path = xml_path.resolve()
    geo_dir = xml_path.parent

    owner, patched_doc = _find_and_remove_detector(xml_path, detector_name)

    sub_tmp_path = _write_tmp_xml(patched_doc, geo_dir, f"no_{detector_name}_sub_")
    top_tmp_path = _write_patched_top(xml_path, owner, sub_tmp_path, geo_dir, detector_name)

    return top_tmp_path, sub_tmp_path


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DetectorNotFoundError(ValueError):
    """Raised when the requested detector name is not in the geometry."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_and_remove_detector(
    xml_path: Path, detector_name: str
) -> tuple[Path, minidom.Document]:
    """Locate *detector_name* in the include tree and remove its node.

    Returns the owning file path and the modified document.
    Raises :exc:`DetectorNotFoundError` if not found.
    """
    for f in resolve_includes(xml_path):
        try:
            doc = minidom.parse(str(f))
        except (ExpatError, OSError):
            continue

        for node in doc.getElementsByTagName("detector"):
            if node.getAttribute("name") == detector_name:
                node.parentNode.removeChild(node)
                return f, doc

    raise DetectorNotFoundError(
        f"Detector '{detector_name}' not found in any XML reachable from "
        f"{xml_path}."
    )


def _write_tmp_xml(doc: minidom.Document, directory: Path, suffix: str) -> Path:
    """Serialise *doc* to a named temp file in *directory*."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml",
        delete=False,
        mode="w",
        dir=directory,
        prefix=f"{_TMP_PREFIX}{suffix}",
    )
    doc.writexml(tmp)
    tmp.close()
    return Path(tmp.name)


def _write_patched_top(
    original_top: Path,
    owner: Path,
    sub_tmp: Path,
    geo_dir: Path,
    detector_name: str,
) -> Path:
    """Rewrite the top-level XML so the include pointing at *owner*
    is redirected to *sub_tmp*.

    Only the single include ref that resolves to *owner* is changed;
    everything else is left verbatim.
    """
    try:
        top_doc = minidom.parse(str(original_top))
    except (ExpatError, OSError) as exc:
        raise OSError(f"Could not parse top-level XML {original_top}: {exc}") from exc

    for node in top_doc.getElementsByTagName("include"):
        ref = node.getAttribute("ref")
        if not ref or "$" in ref:
            continue
        resolved = (geo_dir / os.path.expandvars(ref)).resolve()
        if resolved == owner:
            # Use just the filename — sub_tmp is in the same directory.
            node.setAttribute("ref", sub_tmp.name)

    return _write_tmp_xml(top_doc, geo_dir, f"no_{detector_name}_top_")
