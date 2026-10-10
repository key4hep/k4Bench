"""Execute a single ddsim run and return a :class:`RunResult`.

Design principle
----------------
This module owns *instrumentation*: timing, logging, and metrics
extraction.  It does **not** own physics configuration.  The caller
decides which ddsim arguments to pass; the runner adds the ones it
manages, times the run with :mod:`k4bench.runner.process`, and harvests
the results (:mod:`k4bench.runner.result`).

The only ddsim arguments that the runner needs to know about are:

* ``--compactFile``      — to allow per-run XML patching (geometry sweep)
* ``--numberOfEvents``   — to compute events/sec
* ``--outputFile``       — to measure output file size

Everything else (``--enableGun``, ``--gun.particle``, ``--runType``,
steering files, …) is passed through verbatim via ``extra_args``.

Per-event timing
----------------
When available, the k4Bench C++ timing plugin is loaded
automatically as a DDG4 event action. The plugin writes
per-event timing metrics to JSON files inside the log directory.
Per-event arrays stay in these profiling artifacts. The plugin's run-level
virtual-memory peak is copied into :class:`RunResult`.
"""

from __future__ import annotations

import os
from pathlib import Path

from k4bench.artifacts import EVENTS_SUFFIX, LOG_SUFFIX, REGIONS_SUFFIX, label_path
from k4bench.plugin.runtime import setup_plugin_environment
from k4bench.results.model import RunResult
from k4bench.runner.process import run_timed, timed_shell_command
from k4bench.runner.result import run_result

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ddsim(
    *,
    xml_path: Path,
    label: str,
    n_events: int,
    output_file: Path,
    log_dir: Path,
    setup_script: Path | None = None,
    extra_args: list[str] | None = None,
    verbose: bool = False,
) -> RunResult:
    """Run ddsim for one geometry configuration and return collected metrics.

    The runner injects ``--compactFile``, ``--numberOfEvents``, and
    ``--outputFile`` automatically. All other ddsim options should be
    supplied via *extra_args*.

    Parameters
    ----------
    xml_path:
        Compact XML file passed to ``--compactFile``.

    label:
        Human-readable name for this run. Used as the log filename stem
        and stored in :attr:`RunResult.label`.

    n_events:
        Number of events; passed to ``--numberOfEvents`` and used to
        compute :attr:`RunResult.events_per_sec`.

    output_file:
        EDM4hep ROOT output path passed to ``--outputFile``.
        Its size is recorded after the run.

    log_dir:
        Directory where ``<label>.log`` is written.

    setup_script:
        Optional shell script sourced before ddsim.

    extra_args:
        Additional ddsim arguments passed through verbatim.

    verbose:
        Stream ddsim output live to stdout.

    Returns
    -------
    RunResult
        Process-level timing, memory, and throughput metrics.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Optional plugin output artifacts. Removed first so a run that fails to
    # write its own is never credited with a previous run's.
    event_json_path = label_path(log_dir, label, EVENTS_SUFFIX)
    event_json_path.unlink(missing_ok=True)
    region_json_path = label_path(log_dir, label, REGIONS_SUFFIX)
    region_json_path.unlink(missing_ok=True)

    env = os.environ.copy()

    plugin_available = setup_plugin_environment(
        env=env,
        event_json_path=event_json_path,
        region_json_path=region_json_path,
    )

    cmd = _build_command(
        xml_path=xml_path,
        n_events=n_events,
        output_file=output_file,
        setup_script=setup_script,
        extra_args=extra_args,
        plugin_available=plugin_available,
    )

    proc = run_timed(
        cmd,
        env=env,
        log_path=label_path(log_dir, label, LOG_SUFFIX),
        verbose=verbose,
    )

    output_size_mb: float | None = None

    if output_file.exists():
        output_size_mb = output_file.stat().st_size / 1024**2

    return run_result(
        label=label,
        n_events=n_events,
        proc=proc,
        event_json_path=event_json_path,
        output_size_mb=output_size_mb,
        instrumentation="timing-plugin" if plugin_available else None,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


#: DDG4 actions the timing plugins register, as ``(ddsim flag, action name)``.
_TIMING_ACTIONS = (
    ("--action.event", "k4BenchTimingAction"),
    ("--action.step", "k4BenchRegionTimingAction"),
    ("--action.track", "k4BenchRegionTrackingAction"),
    ("--action.event", "k4BenchRegionEventAction"),
)


def _has_action(args: list[str], action_name: str) -> bool:
    """Return True if action_name is the value of an --action.* flag in args.

    Both spellings argparse accepts count — ``--action.event Foo`` and
    ``--action.event=Foo``. Reading only the separate-argument form would let
    the caller's own ``--action.event=k4BenchTimingAction`` go unseen and the
    same action be injected a second time, registering it twice in DDG4.
    """
    for arg, following in zip(args, [*args[1:], None]):
        if not arg.startswith("--action."):
            continue
        _, sep, inline = arg.partition("=")
        if (inline if sep else following) == action_name:
            return True
    return False


def _build_command(
    *,
    xml_path: Path,
    n_events: int,
    output_file: Path,
    setup_script: Path | None,
    extra_args: list[str] | None,
    plugin_available: bool,
) -> str:
    """Return the shell command used to execute ddsim."""
    return timed_shell_command(
        [
            "ddsim",
            *_ddsim_args(
                xml_path=xml_path,
                n_events=n_events,
                output_file=output_file,
                extra_args=extra_args or [],
                plugin_available=plugin_available,
            ),
        ],
        setup_script,
    )


def _ddsim_args(
    *,
    xml_path: Path,
    n_events: int,
    output_file: Path,
    extra_args: list[str],
    plugin_available: bool,
) -> list[str]:
    """The runner-managed ddsim arguments, then the caller's *extra_args*.

    The timing actions are registered only when the plugins are available and
    the caller has not already registered them, so no action runs twice.
    """
    managed = [
        f"--compactFile={xml_path}",
        f"--numberOfEvents={n_events}",
        f"--outputFile={output_file}",
    ]

    if plugin_available:
        for flag, action in _TIMING_ACTIONS:
            if not _has_action(extra_args, action):
                managed.extend([flag, action])

    return managed + extra_args

