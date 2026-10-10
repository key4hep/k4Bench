"""Directory layout of the benchmark run store.

Every benchmark run is stored five directory levels below the store root — on
EOS, behind the WebEOS listing, in the dashboard's download cache and in local
mirrors alike::

    {detector}/{platform}/{stack}/{sample}/{date}/

``detector`` is the compact file's basename, ``platform`` the LCG/Spack
triplet, ``stack`` the Key4hep release directory (:func:`stack_dir`),
``sample`` the benchmark sample and ``date`` the run date. The files inside a
run directory are named by :mod:`k4bench.artifacts`.

The nightly upload (``nightly_benchmark.sh``) writes this layout and every
reader walks it back, so it is part of the on-disk data contract. Top-level
names starting with ``_`` or ``.`` (e.g. ``_reports/``) hold data outside the
ddsim run tree and are skipped by every walker. k4run benchmark runs are one
such tree: the same five levels below :data:`K4RUN_ROOT`, so the readers of
ddsim runs never see them. Like :mod:`k4bench.artifacts` this
is a leaf module, so every layer can share the one layout without importing
the others.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from k4bench.labels import RELEASE_PREFIX

#: The directory levels of one run, outermost first.
LEVELS = ("detector", "platform", "stack", "sample", "date")

#: Reserved top-level directory holding the k4run benchmark runs, in the same
#: :data:`LEVELS` as the ddsim runs at the store root.
K4RUN_ROOT = "_k4run"


def stack_dir(release: str) -> str:
    """The ``stack`` directory of a Key4hep release, e.g. ``2026-07-10`` ->
    ``key4hep-2026-07-10``."""
    return f"{RELEASE_PREFIX}{release}"


def is_reserved(name: str) -> bool:
    """Whether a top-level *name* is reserved for data outside the ddsim run tree."""
    return name.startswith(("_", "."))


def run_path(
    detector: str,
    platform: str | None = None,
    stack: str | None = None,
    sample: str | None = None,
    date: str | None = None,
) -> str:
    """The store-relative path down to the deepest level given, e.g.
    ``run_path(det, plat, stack)`` is the directory listing that stack's samples.

    Levels are given outermost first; an inner level without every level above
    it names no directory and is refused.
    """
    levels = (detector, platform, stack, sample, date)
    given = [level for level in levels if level is not None]
    if levels[: len(given)] != tuple(given):
        raise ValueError(f"run_path needs every level above the deepest one given: {levels}")
    return "/".join(given)


def run_url(
    base_url: str,
    detector: str,
    platform: str | None = None,
    stack: str | None = None,
    sample: str | None = None,
    date: str | None = None,
) -> str:
    """:func:`run_path` below *base_url*."""
    return f"{base_url.rstrip('/')}/{run_path(detector, platform, stack, sample, date)}"


@dataclass(frozen=True)
class RunDir:
    """One run directory of a local store and the identity its path spells."""

    detector: str
    platform: str
    stack: str
    sample: str
    date: str
    path: Path


def iter_run_dirs(root: Path) -> Iterator[RunDir]:
    """Every run directory of the local store at *root*, in path order.

    Reserved top-level names are skipped, and so is a hidden directory at any
    level: the download cache stages a run in a hidden sibling of its date
    directory until it is complete. A missing *root* is refused rather than
    walked as an empty store.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"run store root is not a directory: {root}")
    pattern = "/".join("*" * len(LEVELS))
    runs = []
    for path in root.glob(pattern):
        parts = path.relative_to(root).parts
        if (
            path.is_dir()
            and not is_reserved(parts[0])
            and not any(part.startswith(".") for part in parts)
        ):
            runs.append(RunDir(*parts, path=path))
    yield from sorted(runs, key=lambda run: run.path.relative_to(root).parts)
