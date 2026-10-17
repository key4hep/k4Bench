"""File names of the artifacts one benchmark run directory holds.

A run directory (``logs/<geometry>/`` locally, ``{detector}/{platform}/{stack}/
{sample}/{date}/`` on EOS) holds its run-wide files and, per benchmark
configuration label, one file per artifact kind::

    run_info.json
    machine_info.json
    marlin_processors.xml.gz    k4run only, when the job ran Marlin processors
    {label}_results.csv
    {label}_events.json
    {label}_regions.json        ddsim only
    {label}_components.json     k4run only
    {label}_joboptions.opts     k4run only
    {label}_dataflow.json       k4run only
    {label}.log
    {label}_warm-up.log         k4run only, when the warm-up failed

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

#: ``Marlin -x`` of the job's stack, gzipped: every Marlin processor type's
#: parameters and which of them are LCIO input or output collections. Written
#: once per k4run benchmark whose jobs ran Marlin processors.
MARLIN_PROCESSORS = "marlin_processors.xml.gz"

#: Per-label suffixes. A label's file is ``f"{label}{suffix}"``.
RESULTS_SUFFIX = "_results.csv"
EVENTS_SUFFIX = "_events.json"
REGIONS_SUFFIX = "_regions.json"
COMPONENTS_SUFFIX = "_components.json"
#: The resolved Gaudi job options, as JobOptionsSvc dumps them.
JOBOPTIONS_SUFFIX = "_joboptions.opts"
#: What each algorithm read and wrote (:mod:`k4bench.analysis.dataflow`).
DATAFLOW_SUFFIX = "_dataflow.json"
LOG_SUFFIX = ".log"
#: The log of a k4run warm-up that failed, so the measured run may have found
#: a cold cache.
WARM_UP_LOG_SUFFIX = "_warm-up.log"


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
