"""Directory layout of the benchmark store.

The store root holds one directory per kind of data — on EOS, behind the
WebEOS listing, in the dashboard's download cache and in local mirrors alike —
and each kind is split by suite first::

    runs/{suite}/{concept}/{detector}/{platform}/{stack}/{sample}/{date}/
    reports/{suite}/{night}/
    reproducers/{suite}/
    inputs/{suite}/{concept}/{detector}/

A suite is an independent benchmark pipeline: ``sim`` (ddsim) or ``reco``
(k4run reconstruction). Each has its own benchmark configs
(``.github/benchmarks/{suite}/{concept}/{detector}.yml``), jobs, run tree,
report and readers; no suite reads another suite's runs, so every reader names
the suite it reads.

``concept`` is the detector concept a detector config belongs to
(:func:`concept_of`), ``detector`` the compact file's basename, ``platform``
the LCG triplet, ``stack`` the Key4hep release directory (:func:`stack_dir`),
``sample`` the benchmark sample and ``date`` the run date. The files inside a
run directory are named by :mod:`k4bench.artifacts`.

The nightly upload (``nightly_benchmark.sh``) writes this layout through
:func:`run_dir_of` and every reader walks it back, so it is part of the on-disk
data contract. Like :mod:`k4bench.artifacts` this is a leaf module, so every
layer can share the one layout without importing the others.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from k4bench.labels import RELEASE_PREFIX

_log = logging.getLogger(__name__)

#: The benchmark suites, each a tree of its own below every kind of data.
SUITES = ("sim", "reco")
SIM, RECO = SUITES

#: Top-level directories of the store, one per kind of data.
RUNS = "runs"
REPORTS = "reports"
REPRODUCERS = "reproducers"
INPUTS = "inputs"

#: The directory levels of one run below :data:`RUNS`, outermost first.
LEVELS = ("suite", "concept", "detector", "platform", "stack", "sample", "date")


def concept_of(detector: str) -> str:
    """The detector concept *detector* belongs to: its name up to the first
    underscore, e.g. ``ALLEGRO_o1_v04`` -> ``ALLEGRO``, ``ILD_FCCee_v01`` ->
    ``ILD``, ``SiD`` -> ``SiD``.

    Derived rather than configured, so a reader holding only a detector name
    can still spell its path.
    """
    return detector.split("_", 1)[0]


def stack_dir(release: str) -> str:
    """The ``stack`` directory of a Key4hep release, e.g. ``2026-07-10`` ->
    ``key4hep-2026-07-10``."""
    return f"{RELEASE_PREFIX}{release}"


def _checked(suite: str) -> str:
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; expected one of {', '.join(SUITES)}")
    return suite


def run_path(
    suite: str,
    detector: str | None = None,
    platform: str | None = None,
    stack: str | None = None,
    sample: str | None = None,
    date: str | None = None,
) -> str:
    """The store-relative path down to the deepest level given, e.g.
    ``run_path(SIM, det, plat, stack)`` is the directory listing that stack's
    samples, and ``run_path(SIM)`` the suite's run tree.

    The detector's concept directory is inserted for it. Levels are given
    outermost first; an inner level without every level above it names no
    directory and is refused. A ``"*"`` detector spells a glob over every
    concept as well.
    """
    levels = (detector, platform, stack, sample, date)
    given = [level for level in levels if level is not None]
    if levels[: len(given)] != tuple(given):
        raise ValueError(f"run_path needs every level above the deepest one given: {levels}")
    parts = [RUNS, _checked(suite)]
    if detector is not None:
        parts += [concept_of(detector), *given]
    return "/".join(parts)


def run_url(
    base_url: str,
    suite: str,
    detector: str | None = None,
    platform: str | None = None,
    stack: str | None = None,
    sample: str | None = None,
    date: str | None = None,
) -> str:
    """:func:`run_path` below *base_url*."""
    return store_url(base_url, run_path(suite, detector, platform, stack, sample, date))


def concept_path(suite: str, concept: str) -> str:
    """The store-relative directory listing *concept*'s detectors in *suite*."""
    return f"{run_path(suite)}/{concept}"


def run_dir_of(run_info: Mapping[str, str]) -> str:
    """The store-relative run directory a recorded ``run_info.json`` belongs in
    — where the nightly uploads the run."""
    return run_path(
        run_info["suite"], run_info["detector"], run_info["platform"],
        run_info["k4h_release"], run_info["sample"], run_info["date"],
    )


def report_path(suite: str, night: str | None = None) -> str:
    """The store-relative directory of *suite*'s reports, or of one night's."""
    path = f"{REPORTS}/{_checked(suite)}"
    return f"{path}/{night}" if night is not None else path


def reproducer_path(suite: str) -> str:
    """The store-relative directory of *suite*'s published reproducer recipes."""
    return f"{REPRODUCERS}/{_checked(suite)}"


def input_path(suite: str, detector: str) -> str:
    """The store-relative directory of the frozen inputs *suite*'s *detector*
    benchmarks read."""
    return f"{INPUTS}/{_checked(suite)}/{concept_of(detector)}/{detector}"


def store_url(base_url: str, path: str) -> str:
    """The store-relative *path* below *base_url*, whatever its trailing slash."""
    return f"{base_url.rstrip('/')}/{path}"


@dataclass(frozen=True)
class RunDir:
    """One run directory of a local store and the identity its path spells."""

    suite: str
    detector: str
    platform: str
    stack: str
    sample: str
    date: str
    path: Path


def iter_run_dirs(root: Path, suite: str) -> Iterator[RunDir]:
    """Every run directory of *suite* in the local store at *root*, in path order.

    A hidden directory at any level is skipped: the download cache stages a run
    in a hidden sibling of its date directory until it is complete. So is a
    detector filed under another concept than its own, which no reader
    spelling its path would find. A missing *root* or suite tree is refused
    rather than walked as an empty store.
    """
    tree = root / run_path(suite)
    if not tree.is_dir():
        raise NotADirectoryError(f"no {suite} run tree in the store at {root}: {tree}")
    pattern = "/".join("*" * (len(LEVELS) - 1))
    runs = []
    for path in tree.glob(pattern):
        parts = path.relative_to(tree).parts
        if not path.is_dir() or any(part.startswith(".") for part in parts):
            continue
        concept, detector, *levels = parts
        if concept != concept_of(detector):
            _log.warning("iter_run_dirs: skipping %s, filed outside its concept %s",
                         path, concept_of(detector))
            continue
        runs.append(RunDir(suite, detector, *levels, path=path))
    yield from sorted(runs, key=lambda run: run.path.relative_to(tree).parts)
