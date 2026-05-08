"""Execute a single ddsim run and return a :class:`RunResult`.

Design principle
----------------
This module owns *instrumentation*: timing, logging, and metrics
extraction.  It does **not** own physics configuration.  The caller
decides which ddsim arguments to pass; the executor wraps them with
``/usr/bin/time -v`` and harvests the results.

The only ddsim arguments that the executor needs to know about are:

* ``--compactFile``      — to allow per-run XML patching (geometry sweep)
* ``--numberOfEvents``   — to compute events/sec
* ``--outputFile``       — to measure output file size

Everything else (``--enableGun``, ``--gun.particle``, ``--runType``,
steering files, …) is passed through verbatim via ``extra_args``.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from pathlib import Path

from dd4bench.results.model import RunResult
from dd4bench.runner.parser import parse_time_output

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
    verbose: bool = False
) -> RunResult:
    """Run ddsim for one geometry configuration and return collected metrics.

    The executor injects ``--compactFile``, ``--numberOfEvents``, and
    ``--outputFile`` automatically.  All other ddsim options should be
    supplied via *extra_args*.

    Parameters
    ----------
    xml_path:
        Compact XML file passed to ``--compactFile``.
    label:
        Human-readable name for this run.  Used as the log filename stem
        and stored in :attr:`RunResult.label`.
    n_events:
        Number of events; passed to ``--numberOfEvents`` and used to
        compute :attr:`RunResult.events_per_sec`.
    output_file:
        EDM4hep ROOT output path passed to ``--outputFile``.  Its size
        is recorded in :attr:`RunResult.output_size_mb` after the run.
    log_dir:
        Directory where ``<label>.log`` is written (created if absent).
    setup_script:
        Optional shell script sourced before ddsim (e.g. a k4geo /
        DD4hep environment setup).  Skipped when *None*.
    extra_args:
        Any additional ddsim arguments, e.g.::

            [
                "--runType=batch",
                "--enableGun",
                "--gun.particle", "e-",
                "--gun.distribution", "uniform",
            ]

        Arguments are shell-quoted before insertion so values with
        spaces are handled correctly.

    Returns
    -------
    RunResult
        Fully populated with timing, memory, and output metrics.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"

    cmd = _build_command(
        xml_path=xml_path,
        n_events=n_events,
        output_file=output_file,
        setup_script=setup_script,
        extra_args=extra_args or [],
    )

    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            start_new_session=True
        )
        lines = []
        for line in proc.stdout:
            if verbose:
                print(line, end="", flush=True)
            lines.append(line)
        stdout = "".join(lines)
        
    except KeyboardInterrupt:
        # Kill the entire process group (bash shell + ddsim child)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise

    completed = subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode, stdout=stdout
    )

    log_path.write_text(completed.stdout)

    metrics = parse_time_output(completed.stdout)

    output_size_mb: float | None = None
    if output_file.exists():
        output_size_mb = output_file.stat().st_size / 1024**2

    events_per_sec: float | None = None
    if metrics["wall_time_s"] is not None and metrics["wall_time_s"] > 0:
        events_per_sec = round(n_events / metrics["wall_time_s"], 4)

    if metrics["wall_time_raw"] is None or metrics["peak_rss_mb"] is None:
        _warn_unparsed(label, log_path)

    return RunResult(
        label=label,
        returncode=completed.returncode,
        n_events=n_events,
        wall_time_raw=metrics["wall_time_raw"],
        wall_time_s=metrics["wall_time_s"],
        user_cpu_s=metrics["user_cpu_s"],
        sys_cpu_s=metrics["sys_cpu_s"],
        peak_rss_mb=metrics["peak_rss_mb"],
        major_page_faults=metrics["major_page_faults"],
        voluntary_ctx_switches=metrics["voluntary_ctx_switches"],
        involuntary_ctx_switches=metrics["involuntary_ctx_switches"],
        output_size_mb=output_size_mb,
        events_per_sec=events_per_sec,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_command(
    *,
    xml_path: Path,
    n_events: int,
    output_file: Path,
    setup_script: Path | None,
    extra_args: list[str],
) -> str:
    """Return the bash command that (optionally) sources the env and runs ddsim.

    The three arguments the executor always controls are placed first so
    they are easy to spot in logs; caller-supplied *extra_args* follow.
    Duplicate flags (e.g. a second ``--compactFile`` in *extra_args*) are
    the caller's responsibility — ddsim will use whichever it sees last.
    """
    source_line = f"source {setup_script}\n" if setup_script is not None else ""

    # Arguments the executor owns — always present.
    managed = [
        f"--compactFile={xml_path}",
        f"--numberOfEvents={n_events}",
        f"--outputFile={output_file}",
    ]

    # Shell-quote each caller-supplied token so values containing spaces
    # (e.g. particle names) are passed correctly.
    caller = [shlex.quote(a) for a in extra_args]

    all_args = " \\\n    ".join(managed + caller)

    return f"{source_line}/usr/bin/time -v ddsim \\\n    {all_args}"
