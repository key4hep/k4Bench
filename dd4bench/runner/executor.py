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

Per-event timing
----------------
When available, the DD4bench C++ timing plugin is loaded
automatically as a DDG4 event action. The plugin writes
per-event timing and memory metrics to JSON which are
attached to the RunResult.
"""

from __future__ import annotations

import json
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
    verbose: bool = False,
) -> RunResult:
    """Run ddsim for one geometry configuration and return collected metrics.

    The executor injects ``--compactFile``, ``--numberOfEvents``, and
    ``--outputFile`` automatically.  All other ddsim options should be
    supplied via *extra_args*.

    Per-event timing is collected transparently via the DD4bench C++ plugin
    when available.  The user does not need to configure anything.

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

            ["--enableGun", "--gun.particle", "e-", "--gun.distribution", "uniform"]

        Arguments are shell-quoted before insertion so values with
        spaces are handled correctly.
    verbose:
        Stream ddsim output to stdout in real time.

    Returns
    -------
    RunResult
        Fully populated with timing, memory, and output metrics.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"

    # Per-event timing: prepare a temp JSON path and steering file
    event_json_path = log_dir / f"{label}_events.json"
    event_json_path.unlink(missing_ok=True)
    
    env = os.environ.copy()

    plugin_available = _setup_plugin_environment(
        env=env,
        event_json_path=event_json_path,
    )

    cmd = _build_command(
        xml_path=xml_path,
        n_events=n_events,
        output_file=output_file,
        setup_script=setup_script,
        extra_args=extra_args or [],
        plugin_available=plugin_available,
    )
    
    stdout = ""
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )

        stdout_lines = []

        with log_path.open("w") as log_file:

            if proc.stdout is None:
                raise RuntimeError("Failed to capture ddsim stdout.")

            for line in proc.stdout:

                # Stream to terminal if requested
                if verbose:
                    print(line, end="", flush=True)

                # Stream immediately to logfile
                log_file.write(line)
                log_file.flush()

                # Keep in memory for later parsing
                stdout_lines.append(line)

            proc.wait()  # ensure returncode is populated

        stdout = "".join(stdout_lines)

    except KeyboardInterrupt:
        print("\nStopping ddsim...", flush=True)
        
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

    metrics = parse_time_output(stdout)

    output_size_mb: float | None = None
    if output_file.exists():
        output_size_mb = output_file.stat().st_size / 1024**2

    events_per_sec: float | None = None
    if metrics["wall_time_s"] is not None and metrics["wall_time_s"] > 0:
        events_per_sec = round(n_events / metrics["wall_time_s"], 4)

    event_numbers, event_times, event_rss_peaks, event_rss_deltas = _read_event_data(
        event_json_path
    )

    if metrics["wall_time_raw"] is None or metrics["peak_rss_mb"] is None:
        _warn_unparsed(label, log_path)

    return RunResult(
        label=label,
        returncode=proc.returncode,
        n_events=n_events,
        wall_time_raw=metrics["wall_time_raw"],
        wall_time_s=metrics["wall_time_s"],
        user_cpu_s=metrics["user_cpu_s"],
        sys_cpu_s=metrics["sys_cpu_s"],
        event_numbers=event_numbers,
        event_times_s=event_times,
        event_rss_peak_mb=event_rss_peaks,
        event_rss_delta_mb=event_rss_deltas,
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


def _setup_plugin_environment(
    *,
    env: dict[str, str],
    event_json_path: Path,
) -> bool:
    """Prepare environment variables for the DD4bench timing plugin.

    Returns
    -------
    bool
        True if the plugin is available and enabled.
        False if ddsim should run without per-event timing.
    """
    try:
        from dd4bench.environment.setup import (
            ensure_plugin_built,
            plugin_lib_dir,
        )

        ensure_plugin_built()

        lib_dir = str(plugin_lib_dir())

        existing = env.get("LD_LIBRARY_PATH", "")

        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir

        env["DD4BENCH_EVENT_JSON"] = str(event_json_path.resolve())

        return True

    except Exception as exc:
        print(
            f"NOTE: DD4bench timing plugin unavailable "
            f"({exc}); continuing without per-event timing."
        )

        return False


def _read_event_data(
    json_path: Path,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Read per-event metrics from the plugin JSON output.

    Returns
    -------
    tuple[list[int], list[float], list[float], list[float]]
        (event_numbers, event_times_s, event_rss_peak_mb, event_rss_delta_mb)
    """
    try:
        data = json.loads(json_path.read_text())
        numbers = [int(n) for n in data.get("event_numbers", [])]
        times = [float(t) for t in data.get("event_times_s", [])]
        peaks = [float(r) for r in data.get("event_rss_peak_mb", [])]
        deltas = [float(r) for r in data.get("event_rss_delta_mb", [])]
        return numbers, times, peaks, deltas
    except Exception:
        return [], [], [], []


def _build_command(
    *,
    xml_path: Path,
    n_events: int,
    output_file: Path,
    setup_script: Path | None,
    extra_args: list[str],
    plugin_available: bool,
    # timing_steering: Path | None,
) -> str:
    """Return the bash command that (optionally) sources the env and runs ddsim."""
    source_line = f"source {setup_script}\n" if setup_script is not None else ""

    # Arguments the executor always controls
    managed = [
        f"--compactFile={xml_path}",
        f"--numberOfEvents={n_events}",
        f"--outputFile={output_file}",
    ]

    # Check if user already included the timing plugin in extra_args to avoid double-injection
    has_timing_action = any("DD4benchTimingAction" in arg for arg in extra_args)

    if plugin_available and not has_timing_action:
        managed.extend(
            [
                "--action.event",
                "DD4benchTimingAction",
            ]
        )

    # Shell-quote caller-supplied tokens
    caller = [shlex.quote(a) for a in extra_args]

    all_args = " \\\n    ".join(managed + caller)

    return f"{source_line}/usr/bin/time -v ddsim \\\n    {all_args}"


def _warn_unparsed(label: str, log_path: Path) -> None:
    print(
        f"  WARNING [{label}]: /usr/bin/time output could not be fully parsed.\n"
        f"           Check {log_path} for the raw output."
    )
