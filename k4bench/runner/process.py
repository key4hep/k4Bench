"""Run one command under ``/usr/bin/time -v`` and harvest what it measured.

This is the tool-independent half of a benchmark run: building the timed
shell command, streaming the process output into its log, stopping the whole
process group on Ctrl-C, and parsing the ``time -v`` report. It knows nothing
about which program it times; :mod:`k4bench.runner.executor` supplies the
``ddsim`` command line and turns the outcome into a
:class:`~k4bench.results.model.RunResult`.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from k4bench.runner.parser import parse_time_output

#: Output lines kept in memory to parse the ``time -v`` report from. The
#: report is the last thing the timed process prints, so only the tail matters
#: and the full output streams straight to the log file.
_TAIL_LINES = 200


@dataclass(frozen=True)
class TimedProcess:
    """Outcome of one timed process: its exit code and the parsed
    ``time -v`` metrics (see :func:`~k4bench.runner.parser.parse_time_output`
    for the keys; each is ``None`` when the report lacked it)."""

    returncode: int
    metrics: dict


def timed_shell_command(argv: list[str], setup_script: Path | None = None) -> str:
    """Return the bash command that runs *argv* under GNU ``time -v``.

    *setup_script*, when given, is sourced first. Every argument is shell-quoted,
    and one argument goes on each line so the command reads cleanly when it is
    echoed into a log.
    """
    gnu_time = shutil.which("time")
    if gnu_time is None:
        raise RuntimeError(
            "GNU time not found in PATH. Install it (e.g. 'dnf install time' or 'apt install time')."
        )

    source_line = (
        f"source {shlex.quote(str(setup_script))}\n" if setup_script is not None else ""
    )
    program, *args = argv
    joined = " \\\n    ".join(shlex.quote(a) for a in args)
    return f"{source_line}{gnu_time} -v {shlex.quote(program)} \\\n    {joined}"


def run_timed(
    command: str,
    *,
    env: dict[str, str],
    log_path: Path,
    verbose: bool = False,
) -> TimedProcess:
    """Run *command* (from :func:`timed_shell_command`) in bash and wait for it.

    All output goes to *log_path*, and to stdout as well when *verbose*. The
    process runs in its own session, so a Ctrl-C stops it and every child it
    started before the ``KeyboardInterrupt`` is re-raised.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )

        tail: deque[str] = deque(maxlen=_TAIL_LINES)

        with log_path.open("w") as log_file:

            if proc.stdout is None:
                raise RuntimeError(f"Failed to capture the output of {log_path.stem}.")

            for line in proc.stdout:
                if verbose:
                    print(line, end="", flush=True)
                log_file.write(line)
                tail.append(line)

            proc.wait()  # ensure returncode is populated

    except KeyboardInterrupt:
        print(f"\nStopping {log_path.stem}...", flush=True)

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)

        except (subprocess.TimeoutExpired, ProcessLookupError):

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

            except ProcessLookupError:
                pass

        raise

    metrics = parse_time_output("".join(tail))
    if metrics["wall_time_raw"] is None or metrics["peak_rss_mb"] is None:
        _warn_unparsed(log_path)

    return TimedProcess(returncode=proc.returncode, metrics=metrics)


def _warn_unparsed(log_path: Path) -> None:
    """Warn that /usr/bin/time output parsing failed."""

    print(
        f"  WARNING [{log_path.stem}]: "
        f"/usr/bin/time output could not be fully parsed.\n"
        f"           Check {log_path} for the raw output."
    )
