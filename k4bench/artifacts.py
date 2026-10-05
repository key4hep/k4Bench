"""File names of the artifacts one benchmark run directory holds.

A run directory (``logs/<geometry>/`` locally, ``{detector}/{platform}/{stack}/
{sample}/{date}/`` on EOS) holds two run-wide files and, per benchmark
configuration label, one file per artifact kind::

    run_info.json
    machine_info.json
    {label}_results.csv
    {label}_events.json
    {label}_regions.json
    {label}.log

These names are part of the on-disk data contract: the benchmark writes them,
and the loaders, the regression report, the dashboard and the blame tooling
read them back from years of stored runs. Like :mod:`k4bench.labels` this is a
leaf module, so every layer can share the one spelling without importing the
others.
"""

from __future__ import annotations

from pathlib import Path

#: Run-wide metadata written by the nightly runner.
RUN_INFO = "run_info.json"
#: Host snapshot taken around the benchmark by the nightly runner.
MACHINE_INFO = "machine_info.json"

#: Per-label suffixes. A label's file is ``f"{label}{suffix}"``.
RESULTS_SUFFIX = "_results.csv"
EVENTS_SUFFIX = "_events.json"
REGIONS_SUFFIX = "_regions.json"
LOG_SUFFIX = ".log"


def label_path(run_dir: Path, label: str, suffix: str) -> Path:
    """Path of *label*'s artifact with *suffix* inside *run_dir*."""
    return run_dir / f"{label}{suffix}"


def labelled_files(run_dir: Path, suffix: str) -> list[tuple[Path, str]]:
    """Every ``(path, label)`` in *run_dir* whose name ends in *suffix*,
    sorted by path."""
    return [
        (path, path.name[: -len(suffix)])
        for path in sorted(run_dir.glob(f"*{suffix}"))
    ]
