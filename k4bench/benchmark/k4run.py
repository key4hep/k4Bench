"""Benchmark a k4run job and its variants.

The k4run counterpart of :mod:`k4bench.benchmark.ddsim`. k4Bench knows nothing
about the job: the options files, the directory to stage and the job's own
arguments come from the caller, and everything measured comes from the Gaudi
layer (see :func:`k4bench.runner.k4run.run_k4run`).

A benchmark is the ``baseline`` job plus one run per *variant*, a named set of
arguments appended to the job's own (label ``variant_<name>``). Runs are
sequential, for the same reason as the ddsim sweep: parallel runs would skew
each other's wall time and memory.

Every run gets a fresh working directory, removed after the run: a writable
copy of the stage directory when one is given (reconstruction configurations
such as CLDConfig must run from their own directory), otherwise an empty one.
A job's outputs therefore never land in the caller's directory, and one run's
outputs are never read by the next. Paths in the job's arguments are read from
that working directory, so give them as absolute paths.

Each run is preceded by a warm-up: the same job for one event, unmeasured and
discarded. A job loads its libraries, geometry and conditions from CVMFS, and
on a host whose CVMFS cache does not hold them yet, fetching them dominates its
initialization (tens of seconds for CLD reconstruction, against a few seconds
from a warm cache), which would measure the host's cache rather than the job.
The warm-up event also reads in the code the first measured event runs.
"""

from __future__ import annotations

import re
import shutil
import stat
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from k4bench.labels import BASELINE_LABEL, VARIANT_PREFIX
from k4bench.results.model import RunResult
from k4bench.results.reporter import print_run_result
from k4bench.runner.k4run import run_k4run

#: A variant name becomes part of the run's file names.
_VARIANT_NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")

#: Events of the unmeasured run before each measured one.
WARM_UP_EVENTS = 1


@dataclass
class K4runConfig:
    """All parameters needed to benchmark a k4run job.

    Parameters
    ----------
    options:
        Gaudi options files, loaded in order. Relative paths are relative to
        *stage_dir* when it is given, otherwise to the current directory.
    n_events:
        Number of events per run.
    log_dir:
        Directory where per-run logs and the auditor's files are written.
    stage_dir:
        Directory copied into every run's working directory, for jobs that
        read files relative to where they run.
    extra_args:
        Job arguments passed verbatim to every run.
    variants:
        ``{name: arguments}``. Each variant is one more run, with its arguments
        appended to *extra_args*.
    setup_script:
        Optional shell script sourced before each k4run invocation. A relative
        path is relative to the current directory.
    verbose:
        If True, print k4run output in real time instead of only logging it.
    warm_up:
        If True, precede each run with the same job for
        :data:`WARM_UP_EVENTS` events, unmeasured, so the measured run finds
        what the job reads from CVMFS already cached.
    """

    options: list[str]
    n_events: int
    log_dir: Path
    stage_dir: Path | None = None
    extra_args: list[str] = field(default_factory=list)
    variants: dict[str, list[str]] = field(default_factory=dict)
    setup_script: Path | None = None
    verbose: bool = False
    warm_up: bool = True

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("A k4run benchmark needs at least one options file.")
        if invalid := sorted(n for n in self.variants if not _VARIANT_NAME_RE.match(n)):
            raise ValueError(
                f"Invalid variant names {invalid}: use only letters, digits and '_.+-'."
            )
        if self.stage_dir is not None and not self.stage_dir.is_dir():
            raise ValueError(f"Stage directory not found: {self.stage_dir}")
        base = self.stage_dir if self.stage_dir is not None else Path.cwd()
        if missing := [o for o in self.options if not (base / o).is_file()]:
            raise ValueError(f"Options files not found in {base}: {missing}")
        if self.stage_dir is None:
            # Each run starts in an empty directory, where a relative path
            # would no longer name the caller's file.
            self.options = [str((base / o).resolve()) for o in self.options]
        if self.setup_script is not None:
            # Sourced from the run's working directory, not the caller's.
            self.setup_script = self.setup_script.resolve()


def planned_k4run_labels(variants: list[str]) -> list[str]:
    """The result labels a k4run benchmark with these *variants* produces, in
    run order: ``baseline``, then ``variant_<name>`` for each variant."""
    return [BASELINE_LABEL, *(f"{VARIANT_PREFIX}{name}" for name in variants)]


def run_k4run_benchmark(config: K4runConfig) -> list[RunResult]:
    """Run the baseline job and every variant, and return all results.

    A run that fails is marked with a non-zero return code and included in
    the results; the benchmark always continues to the last variant.
    """
    config.log_dir.mkdir(parents=True, exist_ok=True)
    labels = planned_k4run_labels(list(config.variants))
    job_args = [config.extra_args, *([*config.extra_args, *v] for v in config.variants.values())]
    runs = list(zip(labels, job_args, strict=True))

    results: list[RunResult] = []
    for index, (label, args) in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {label}")
        print(f"         Options: {' '.join(config.options)}")
        if config.warm_up:
            _warm_up(config, label, args)
        try:
            work_dir = _stage(config.stage_dir, label)
        except OSError:
            print(f"  ERROR staging {label}:\n{traceback.format_exc()}")
            result = RunResult(label=label, returncode=1, n_events=config.n_events)
        else:
            try:
                result = run_k4run(
                    options=config.options,
                    label=label,
                    n_events=config.n_events,
                    work_dir=work_dir,
                    log_dir=config.log_dir,
                    setup_script=config.setup_script,
                    extra_args=args,
                    verbose=config.verbose,
                )
            finally:
                _remove(work_dir)
        print_run_result(result)
        results.append(result)
    return results


def _warm_up(config: K4runConfig, label: str, args: list[str]) -> None:
    """Run *label*'s job for :data:`WARM_UP_EVENTS` events in a working and a
    log directory of its own, both discarded.

    A warm-up that fails is reported and otherwise ignored: the measured run
    that follows records the failure in its own log and result.
    """
    try:
        work_dir = _stage(config.stage_dir, f"warm-up-{label}")
    except OSError as exc:
        print(f"         Warm-up skipped: staging failed ({exc})")
        return
    try:
        with tempfile.TemporaryDirectory(prefix=f"k4bench-warm-up-{label}-logs-") as log_dir:
            result = run_k4run(
                options=config.options,
                label=label,
                n_events=WARM_UP_EVENTS,
                work_dir=work_dir,
                log_dir=Path(log_dir),
                setup_script=config.setup_script,
                extra_args=args,
            )
    finally:
        _remove(work_dir)
    if not result.succeeded:
        print(f"         Warm-up failed (return code {result.returncode}), measuring anyway")
    elif result.wall_time_s is not None:
        print(f"         Warm-up: {result.wall_time_s:.1f} s, not measured")


def _stage(stage_dir: Path | None, label: str) -> Path:
    """A fresh working directory for *label*'s run: a writable copy of
    *stage_dir*, or an empty directory without one."""
    work_dir = Path(tempfile.mkdtemp(prefix=f"k4bench-{label}-"))
    if stage_dir is None:
        return work_dir
    try:
        shutil.copytree(stage_dir, work_dir, dirs_exist_ok=True)
        _make_writable(work_dir)
    except BaseException:
        _remove(work_dir)
        raise
    return work_dir


def _make_writable(root: Path) -> None:
    """Give the owner write permission throughout *root*.

    A copy keeps the modes of its source, so a copy of a read-only tree (as on
    CVMFS) would not let the job write next to its own files, nor let the copy
    be removed afterwards.
    """
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _remove(work_dir: Path) -> None:
    """Delete a run's working directory, whatever the job left in it."""
    try:
        _make_writable(work_dir)
    except OSError:
        pass
    shutil.rmtree(work_dir, ignore_errors=True)
