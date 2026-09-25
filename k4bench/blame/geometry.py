"""Which benchmarked detectors a pull request's diff reaches, and what it changes
in each.

A sweeping geometry pull request touches dozens of files across several
detectors, and its raw diff is a poor carrier of the facts that decide an
attribution: that it changed one constant in the dimensions file of the
detector that moved, that it made the *same* include switch in two detectors
of which only one moved, that its change to a third detector only adds display
settings. Each of those is one line once computed, and none is reliably found
by a model reading a clipped diff.

This module computes them, deterministically, from the pull request's paths
and hunks and the geometry each benchmarked run group loads
(:attr:`~k4bench.regression.models.RunGroupReport.geometry_path`):

* :func:`file_change` — what one compact file's hunk changes: constants
  (``name: old → new``), includes switched, display lines, everything else
  counted;
* :func:`detector_touches` — for every benchmarked detector, the pull
  request's files in the compact directory that detector loads and elsewhere
  in its geometry tree, the structural change inside that directory, and
  which other benchmarked detectors received the identical change.

It judges nothing and renders nothing (:mod:`k4bench.blame.prompt` does).
Absence is never evidence here either: a file whose hunk could not be read is
counted as unread, never as unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from k4bench.blame.github import FilePatch, path_under


def geometry_tree(xml_path: str) -> str:
    """The k4geo subtree a run's compact file lives under —
    ``FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml`` →
    ``FCCee/ALLEGRO/``.

    Two components, not the full directory: a detector's geometry is spread over
    ``compact/``, ``FCCee/ALLEGRO/…`` variants and shared includes, and matching
    the whole path would answer "did this pull request touch this exact file"
    when the useful question is "did it touch this detector at all". Empty when
    the path is unknown or too shallow to name a subtree."""
    parts = [p for p in (xml_path or "").split("/") if p]
    return "/".join(parts[:2]) + "/" if len(parts) >= 3 else ""


def compact_dir(xml_path: str) -> str:
    """The directory a run's compact file sits in —
    ``FCCee/ALLEGRO/compact/ALLEGRO_o2_v01/ALLEGRO_o2_v01.xml`` →
    ``FCCee/ALLEGRO/compact/ALLEGRO_o2_v01/``.

    Narrower than :func:`geometry_tree` on purpose: the tree holds every variant
    of a detector, while this directory holds the one the run loads — its
    dimensions file above all. Empty when :func:`geometry_tree` is."""
    parts = [p for p in (xml_path or "").split("/") if p]
    return "/".join(parts[:-1]) + "/" if len(parts) >= 3 else ""


# ── One file's hunk ───────────────────────────────────────────────────────────

_COMMENT = re.compile(r"<!--.*?-->")
_ELEMENT = re.compile(r"<\s*/?\s*([A-Za-z_][\w:.-]*)")
_CONSTANT = re.compile(r"<constant\b([^>]*)>")
_ATTRIBUTE = re.compile(r"([A-Za-z_][\w:.-]*)\s*=\s*\"([^\"]*)\"")
_INCLUDE = re.compile(r"<(?:include|gdmlFile)\b[^>]*\bref\s*=\s*\"([^\"]+)\"")
_VIS = re.compile(r"<vis\b")
_NEW_FILE = re.compile(r"^@@ -0,0 ")


@dataclass(frozen=True)
class FileChange:
    """What one file's hunk changes, read structurally.

    ``constants_changed`` holds ``(name, old, new)`` for a constant whose value
    moved; ``constants_added``/``constants_removed`` hold ``(name, value)``.
    ``includes_added``/``includes_removed`` are the ``ref`` of ``<include>`` and
    ``<gdmlFile>`` elements. ``display_lines`` counts changed ``<vis>`` lines,
    ``other_lines`` every changed line that is none of these, comments and blank
    lines left out, and ``other_elements`` names the elements those lines hold
    (``""`` for a line continuing an element opened on an unchanged line).
    ``added`` marks a file the pull request creates.

    Every name, value and path here is quoted from the author's diff and
    passed through :func:`quoted`, which keeps it short and plain."""

    path: str
    constants_changed: tuple[tuple[str, str, str], ...] = ()
    constants_added: tuple[tuple[str, str], ...] = ()
    constants_removed: tuple[tuple[str, str], ...] = ()
    includes_added: tuple[str, ...] = ()
    includes_removed: tuple[str, ...] = ()
    display_lines: int = 0
    other_lines: int = 0
    other_elements: tuple[tuple[str, int], ...] = ()
    added: bool = False
    #: Whether the hunk behind this was cut short (GitHub's or the stored cap):
    #: what is listed is then a lower bound.
    clipped: bool = False

    @property
    def geometry_changed(self) -> bool:
        """Whether anything but display settings changed."""
        return bool(
            self.constants_changed or self.constants_added or self.constants_removed
            or self.includes_added or self.includes_removed or self.other_lines
            or self.added
        )


#: The longest quoted name, value or path.
_MAX_QUOTED_CHARS = 60
_UNQUOTABLE = re.compile(r"[^A-Za-z0-9_.,:;+*/()=<>\[\] -]")


def quoted(text: str, limit: int = _MAX_QUOTED_CHARS) -> str:
    """*text* as it may be quoted outside a fence: plain characters only, runs
    of dashes collapsed (so no fence marker can be spelled), whitespace
    collapsed, and at most *limit* characters.

    These strings are the author's — a constant's name and value, an include
    path — and reach the prompt beside k4Bench's own text, like a pull
    request's title and paths do; the system prompts say so. Keeping them to
    identifier-like text keeps them what they are."""
    text = _UNQUOTABLE.sub("", " ".join(str(text).split()))
    text = re.sub(r"-{2,}", "-", text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _code_lines(text: str) -> list[tuple[str, str]]:
    """``(sign, body)`` for every added or removed line of *text* with its XML
    comments stripped; lines that were only comment or whitespace are dropped.
    A comment opened on one line and closed on a later one of the same sign is
    followed across them."""
    out: list[tuple[str, str]] = []
    open_comment = {"+": False, "-": False}
    for line in text.splitlines():
        sign = line[:1]
        if sign not in "+-" or line.startswith(("+++", "---")):
            continue
        body = line[1:]
        if open_comment[sign]:
            if "-->" not in body:
                continue
            body = body.split("-->", 1)[1]
            open_comment[sign] = False
        body = _COMMENT.sub("", body)
        if "<!--" in body:
            body = body.split("<!--", 1)[0]
            open_comment[sign] = True
        body = " ".join(body.split())
        if body:
            out.append((sign, body))
    return out


def file_change(patch: FilePatch) -> FileChange:
    """The structural reading of one file's hunk (see :class:`FileChange`)."""
    removed: dict[str, str] = {}
    added: dict[str, str] = {}
    includes = {"+": [], "-": []}
    display = other = 0
    elements: dict[str, int] = {}
    for sign, body in _code_lines(patch.text):
        constants = [
            dict(_ATTRIBUTE.findall(match.group(1)))
            for match in _CONSTANT.finditer(body)
        ]
        constants = [c for c in constants if c.get("name")]
        refs = [quoted(ref) for ref in _INCLUDE.findall(body)]
        for constant in constants:
            target = added if sign == "+" else removed
            target[quoted(constant["name"])] = quoted(constant.get("value", ""))
        includes[sign].extend(refs)
        if constants or refs:
            continue
        if _VIS.search(body):
            display += 1
        else:
            other += 1
            element = _ELEMENT.search(body)
            name = quoted(element.group(1), 30) if element else ""
            elements[name] = elements.get(name, 0) + 1
    changed = tuple(
        (name, removed[name], added[name])
        for name in sorted(set(removed) & set(added))
        if removed[name] != added[name]
    )
    kept_in, kept_out = set(includes["+"]), set(includes["-"])
    return FileChange(
        path=patch.path,
        constants_changed=changed,
        constants_added=tuple(
            (name, added[name]) for name in sorted(set(added) - set(removed))
        ),
        constants_removed=tuple(
            (name, removed[name]) for name in sorted(set(removed) - set(added))
        ),
        includes_added=tuple(r for r in dict.fromkeys(includes["+"]) if r not in kept_out),
        includes_removed=tuple(r for r in dict.fromkeys(includes["-"]) if r not in kept_in),
        display_lines=display,
        other_lines=other,
        other_elements=tuple(sorted(elements.items(), key=lambda kv: (-kv[1], kv[0]))),
        added=bool(_NEW_FILE.match(patch.text or "")),
        clipped=patch.clipped,
    )


def _fingerprint(patches: Sequence[FilePatch], directory: str) -> frozenset:
    """The changed lines of *patches*, normalised so the same change made to
    two detectors compares equal: paths relative to *directory*, the
    directory's own name replaced by a placeholder, comments and whitespace
    ignored."""
    own = directory.rstrip("/").rsplit("/", 1)[-1]
    lines = set()
    for patch in patches:
        relative = patch.path.removeprefix(directory.rstrip("/") + "/")
        if own:
            relative = relative.replace(own, "{self}")
        for sign, body in _code_lines(patch.text):
            lines.add((relative, sign, body.replace(own, "{self}") if own else body))
    return frozenset(lines)


# ── Every benchmarked detector ────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectorTouch:
    """One benchmarked detector a pull request's files reach.

    ``own_files`` are in the compact directory the detector's runs load
    (:func:`compact_dir`), ``tree_files`` elsewhere in its geometry tree
    (:func:`geometry_tree`). ``changes`` reads each own file's hunk
    (:func:`file_change`); ``unread`` counts own files that had none to read
    (a binary file, a pure rename, or a hunk that could not be fetched).
    ``same_as`` names the other benchmarked detectors whose compact directory
    received exactly the same change."""

    detector: str
    geometry_path: str
    own_dir: str
    own_files: tuple[str, ...] = ()
    tree_files: tuple[str, ...] = ()
    changes: tuple[FileChange, ...] = ()
    unread: int = 0
    same_as: tuple[str, ...] = field(default=())

    @property
    def geometry_changed(self) -> bool | None:
        """Whether the change to the compact directory touches anything but
        display settings — ``None`` when some own file could not be read, since
        an unread file could be the one that does."""
        if any(change.geometry_changed for change in self.changes):
            return True
        if self.unread:
            return None
        return False if self.own_files else None


def detector_touches(
    files: Sequence[str],
    hunks: Sequence[FilePatch],
    geometry: Mapping[str, str],
) -> tuple[DetectorTouch, ...]:
    """Every detector in *geometry* (``detector -> compact file path``) that
    *files* reach, ordered by detector.

    Only a repository-relative path that names a real subtree counts
    (:func:`geometry_tree` is empty for anything shallower), so a run group
    that recorded no geometry, or one that loads an installed file by its
    absolute path, is never matched."""
    by_path = {patch.path: patch for patch in hunks}
    touches = []
    prints: dict[str, frozenset] = {}
    for detector, path in sorted(geometry.items()):
        tree, own_dir = geometry_tree(path), compact_dir(path)
        if not tree or path.startswith("/"):
            continue
        own = tuple(f for f in files if path_under(f, own_dir))
        in_tree = tuple(f for f in files if path_under(f, tree) and f not in own)
        if not own and not in_tree:
            continue
        own_hunks = [by_path[f] for f in own if f in by_path and by_path[f].text]
        touches.append(DetectorTouch(
            detector=detector, geometry_path=path, own_dir=own_dir,
            own_files=own, tree_files=in_tree,
            changes=tuple(file_change(p) for p in own_hunks),
            unread=len(own) - len(own_hunks),
        ))
        if own_hunks and len(own_hunks) == len(own):
            prints[detector] = _fingerprint(own_hunks, own_dir)
    return tuple(
        replace(touch, same_as=tuple(
            other for other, fingerprint in sorted(prints.items())
            if other != touch.detector and fingerprint
            and fingerprint == prints.get(touch.detector)
        ))
        for touch in touches
    )


def benchmarked_geometry(report) -> dict[str, str]:
    """``detector -> compact file`` for every detector *report* benchmarked
    with a recorded geometry — the first recorded, in the report's order."""
    geometry: dict[str, str] = {}
    for group in report.groups:
        if group.geometry_path:
            geometry.setdefault(group.detector, group.geometry_path)
    return geometry
