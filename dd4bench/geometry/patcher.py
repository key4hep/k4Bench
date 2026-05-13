"""Patch a DD4hep compact geometry to remove a single subdetector.

The patcher writes temporary XML files to the system temp directory so
that the original geometry (which may live on a read-only filesystem
such as CVMFS) is never modified.  All relative ``<include ref="...">``
paths in the patched XMLs are rewritten to absolute paths so that
ddsim can resolve them regardless of where the temp files land.

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
def patched_geometry_keep_only(xml_path: Path, keep_names: set[str]):
    """Context manager yielding a geometry with only *keep_names* detectors active.

    All ``<detector>`` elements whose ``name`` attribute is not in *keep_names*
    are removed from every file in the include tree.  Temp files are written
    to the system temp directory and deleted on exit.

    Parameters
    ----------
    xml_path:
        Path to the original top-level compact XML.
    keep_names:
        Detector names to keep.  All others are removed.

    Yields
    ------
    Path
        Path to the patched top-level XML file.
    """
    tmp_files, top_tmp = _build_keep_only_xml(xml_path, keep_names)
    try:
        yield top_tmp
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)


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

    _remove_orphaned_plugins(patched_doc, {detector_name})
    _absolutize_refs(patched_doc, owner.parent)
    sub_tmp_path = _write_tmp_xml(patched_doc, None, f"no_{detector_name}_sub_")
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


def _build_keep_only_xml(xml_path: Path, keep_names: set[str]) -> tuple[list[Path], Path]:
    """Write patched XML files keeping only detectors in *keep_names*.

    Scans every file reachable from *xml_path*, removes all ``<detector>``
    elements not in *keep_names*, writes patched versions of affected files
    to the system temp directory, and returns a patched top-level XML that
    references them.

    Returns
    -------
    tuple[list[Path], Path]
        ``(all_tmp_paths, top_tmp_path)``.  Caller is responsible for
        cleanup; prefer :func:`patched_geometry_keep_only`.
    """
    xml_path = xml_path.resolve()
    geo_dir = xml_path.parent
    all_tmp: list[Path] = []

    try:
        # Resolve once; reused by all three passes to avoid re-traversing the tree.
        all_files = resolve_includes(xml_path)

        # Pass 1: remove unwanted detectors from every reachable file.
        # resolve_includes yields xml_path first, so the top-level is processed too.
        modified: dict[Path, minidom.Document] = {}
        all_removed: set[str] = set()

        for f in all_files:
            try:
                doc = minidom.parse(str(f))
            except (ExpatError, OSError):
                continue

            nodes_to_remove = [
                node
                for node in doc.getElementsByTagName("detector")
                if node.getAttribute("name") and node.getAttribute("name") not in keep_names
            ]
            if not nodes_to_remove:
                continue

            removed_here = {node.getAttribute("name") for node in nodes_to_remove}
            all_removed |= removed_here
            for node in nodes_to_remove:
                node.parentNode.removeChild(node)
            _remove_orphaned_plugins(doc, removed_here)
            modified[f] = doc

        # Pass 2: write tmp files for modified sub-files (not the top-level).
        sub_tmp_map: dict[Path, Path] = {}
        for f, doc in modified.items():
            if f == xml_path:
                continue
            _absolutize_refs(doc, f.parent)
            tmp = _write_tmp_xml(doc, None, "keep_only_sub_")
            sub_tmp_map[f] = tmp
            all_tmp.append(tmp)

        # Pass 3 (fixpoint): create redirect tmps for unmodified sub-files whose
        # <include> refs point to a patched file in sub_tmp_map.  This handles
        # nested include chains (e.g. top → A → B where only B was patched: A
        # must reference B_tmp so ddsim sees the patched sub-tree).
        changed = True
        while changed:
            changed = False
            for f in all_files:
                if f in sub_tmp_map or f == xml_path:
                    continue
                try:
                    doc = minidom.parse(str(f))
                except (ExpatError, OSError):
                    continue
                base = f.parent
                redirected = False
                for node in doc.getElementsByTagName("include"):
                    ref = node.getAttribute("ref")
                    if not ref or "$" in ref:
                        continue
                    resolved = (base / os.path.expandvars(ref)).resolve()
                    if resolved in sub_tmp_map:
                        node.setAttribute("ref", str(sub_tmp_map[resolved]))
                        redirected = True
                if redirected:
                    _absolutize_refs(doc, base)
                    tmp = _write_tmp_xml(doc, None, "keep_only_sub_")
                    sub_tmp_map[f] = tmp
                    all_tmp.append(tmp)
                    changed = True

        # Build the top-level tmp.  If the top-level file itself had detectors
        # removed, use the already-patched doc; otherwise parse fresh from disk.
        top_doc = modified.get(xml_path)
        if top_doc is None:
            try:
                top_doc = minidom.parse(str(xml_path))
            except (ExpatError, OSError) as exc:
                raise OSError(f"Could not parse top-level XML {xml_path}: {exc}") from exc

        for node in top_doc.getElementsByTagName("include"):
            ref = node.getAttribute("ref")
            if not ref or "$" in ref:
                continue
            resolved = (geo_dir / os.path.expandvars(ref)).resolve()
            if resolved in sub_tmp_map:
                node.setAttribute("ref", str(sub_tmp_map[resolved]))

        _remove_orphaned_plugins(top_doc, all_removed)
        _absolutize_refs(top_doc, geo_dir)
        top_tmp = _write_tmp_xml(top_doc, None, "keep_only_top_")
        all_tmp.append(top_tmp)
        return all_tmp, top_tmp

    except:
        for tmp in all_tmp:
            tmp.unlink(missing_ok=True)
        raise


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


def _remove_orphaned_plugins(doc: minidom.Document, removed_names: set[str]) -> None:
    """Remove <plugin> elements whose first <argument value="..."> names a removed detector."""
    for plugin in list(doc.getElementsByTagName("plugin")):
        args = plugin.getElementsByTagName("argument")
        if args and args[0].getAttribute("value") in removed_names:
            plugin.parentNode.removeChild(plugin)


def _absolutize_refs(doc: minidom.Document, base_dir: Path) -> None:
    """Rewrite every relative ref="..." that points to an existing file.

    Walks all XML elements so that <gdmlFile ref="...">, <include ref="...">
    and any other DD4hep node types are covered.  Refs that contain '$' (env
    vars) or already point at absolute paths are left untouched.  Refs that
    do not resolve to an existing file are also left untouched so that
    non-path ref attributes (e.g. detector component names) are not mangled.
    """
    def _walk(node: minidom.Node) -> None:
        if node.nodeType == node.ELEMENT_NODE:
            ref = node.getAttribute("ref")
            if ref and "$" not in ref and not os.path.isabs(ref):
                abs_path = (base_dir / ref).resolve()
                if abs_path.exists():
                    node.setAttribute("ref", str(abs_path))
        for child in node.childNodes:
            _walk(child)

    _walk(doc.documentElement)


def _write_tmp_xml(doc: minidom.Document, directory: Path | None, suffix: str) -> Path:
    """Serialise *doc* to a named temp file.

    *directory* defaults to the system temp dir when ``None``.
    """
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
            node.setAttribute("ref", str(sub_tmp))

    _remove_orphaned_plugins(top_doc, {detector_name})
    _absolutize_refs(top_doc, geo_dir)
    return _write_tmp_xml(top_doc, None, f"no_{detector_name}_top_")
