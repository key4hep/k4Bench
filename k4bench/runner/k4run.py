"""Execute a single k4run job and return a :class:`RunResult`.

Design principle
----------------
Like :mod:`k4bench.runner.ddsim`, this module owns *instrumentation* and not
the job's configuration, and here k4Bench knows even less about the job: it is
any set of Gaudi options files. The runner manages only ``--num-events``, to
compute events/sec, and appends the k4BenchAuditor's options file after the
job's own. Input files, geometry and the job's own switches are passed through
verbatim via ``extra_args``.

Per-event and per-component costs
---------------------------------
When available, the k4BenchAuditor is enabled on every algorithm and service.
It writes the per-event JSON in the same format as the DDG4 timing plugin, the
per-component JSON, and the resolved job options, all inside the log
directory. The run-level virtual-memory peak is copied into
:class:`RunResult`.
"""

from __future__ import annotations

import os
from pathlib import Path

from k4bench.artifacts import (
    COMPONENTS_SUFFIX,
    EVENTS_SUFFIX,
    JOBOPTIONS_SUFFIX,
    LOG_SUFFIX,
    label_path,
)
from k4bench.plugin.runtime import auditor_options_file, setup_auditor_environment
from k4bench.results.model import RunResult
from k4bench.runner.process import run_timed, timed_shell_command
from k4bench.runner.result import run_result

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_k4run(
    *,
    options: list[str],
    label: str,
    n_events: int,
    work_dir: Path,
    log_dir: Path,
    setup_script: Path | None = None,
    extra_args: list[str] | None = None,
    verbose: bool = False,
) -> RunResult:
    """Run one k4run job in *work_dir* and return collected metrics.

    The runner injects ``--num-events`` and, when the auditor is available,
    appends its options file after *options*. All other job arguments should
    be supplied via *extra_args*.

    Parameters
    ----------
    options:
        Gaudi options files, loaded in order. Relative paths are resolved by
        k4run against *work_dir*.

    label:
        Human-readable name for this run. Used as the stem of the log and of
        the auditor's output files, and stored in :attr:`RunResult.label`.

    n_events:
        Number of events; passed to ``--num-events`` and used to compute
        :attr:`RunResult.events_per_sec`.

    work_dir:
        Directory the job runs in. The job's own outputs are written there;
        :attr:`RunResult.output_size_mb` is the size of every file it created.

    log_dir:
        Directory where ``<label>.log`` and the auditor's files are written.

    setup_script:
        Optional shell script sourced before k4run.

    extra_args:
        Additional k4run arguments passed through verbatim, after the options
        files, so the options files' own argument parsers read them.

    verbose:
        Stream k4run output live to stdout.

    Returns
    -------
    RunResult
        Process-level timing, memory, and throughput metrics.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Auditor output artifacts. Removed first so a run that fails to write its
    # own is never credited with a previous run's.
    event_json_path = label_path(log_dir, label, EVENTS_SUFFIX)
    components_json_path = label_path(log_dir, label, COMPONENTS_SUFFIX)
    joboptions_path = label_path(log_dir, label, JOBOPTIONS_SUFFIX)
    for path in (event_json_path, components_json_path, joboptions_path):
        path.unlink(missing_ok=True)

    env = os.environ.copy()

    auditor_available = setup_auditor_environment(
        env=env,
        components_json_path=components_json_path,
        event_json_path=event_json_path,
        joboptions_path=joboptions_path,
    )

    cmd = timed_shell_command(
        [
            "k4run",
            *_k4run_args(
                options=options,
                n_events=n_events,
                extra_args=extra_args or [],
                auditor_options=auditor_options_file() if auditor_available else None,
            ),
        ],
        setup_script,
    )

    existing = _files_in(work_dir)
    proc = run_timed(
        cmd,
        env=env,
        log_path=label_path(log_dir, label, LOG_SUFFIX),
        verbose=verbose,
        cwd=work_dir,
    )
    created = _files_in(work_dir) - existing

    return run_result(
        label=label,
        n_events=n_events,
        proc=proc,
        event_json_path=event_json_path,
        output_size_mb=sum(path.stat().st_size for path in created) / 1024**2,
        instrumentation="auditor" if auditor_available else None,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _k4run_args(
    *,
    options: list[str],
    n_events: int,
    extra_args: list[str],
    auditor_options: Path | None,
) -> list[str]:
    """The options files, the auditor's after them, then the managed event
    count and the caller's *extra_args*.

    The auditor's options file is the last options file, so it changes nothing
    the job configured, and it precedes *extra_args*: those are read by the
    options files' own argument parsers, where a trailing file name after, say,
    ``--inputFiles a.root`` would be taken as one more input file.
    """
    files = [*options, *([str(auditor_options)] if auditor_options is not None else [])]
    return [*files, f"--num-events={n_events}", *extra_args]


def _files_in(directory: Path) -> set[Path]:
    """Every regular file below *directory*."""
    return {path for path in directory.rglob("*") if path.is_file()}
