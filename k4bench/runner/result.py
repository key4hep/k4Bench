"""Turn one timed benchmark process into a :class:`RunResult`.

Shared by the ddsim and k4run runners (:mod:`k4bench.runner.ddsim`,
:mod:`k4bench.runner.k4run`): both time their tool with
:mod:`k4bench.runner.process`, and both have k4Bench instrumentation (the DDG4
timing plugin, or the Gaudi auditor) write a per-event JSON in one format,
whose run-level virtual-memory peak is copied into the result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from k4bench.plugin.schema import validate_event_schema
from k4bench.results.model import RunResult
from k4bench.runner.process import TimedProcess


def run_result(
    *,
    label: str,
    n_events: int,
    proc: TimedProcess,
    event_json_path: Path,
    output_size_mb: float | None,
    instrumentation: str | None,
) -> RunResult:
    """The :class:`RunResult` of one timed run.

    *instrumentation* names what was set up to write *event_json_path* (e.g.
    ``timing-plugin`` or ``auditor``), or is ``None`` when nothing was; a
    successful instrumented run that left no usable file there is warned about.
    """
    metrics = proc.metrics
    peak_vmem_mb = read_peak_vmem_mb(event_json_path)

    events_per_sec: float | None = None

    if metrics["wall_time_s"] is not None and metrics["wall_time_s"] > 0:
        events_per_sec = round(
            n_events / metrics["wall_time_s"],
            4,
        )

    if instrumentation is not None and proc.returncode == 0 and peak_vmem_mb is None:
        _warn_missing_instrumentation(label, event_json_path, instrumentation)

    return RunResult(
        label=label,
        returncode=proc.returncode,
        n_events=n_events,
        wall_time_raw=metrics["wall_time_raw"],
        wall_time_s=metrics["wall_time_s"],
        user_cpu_s=metrics["user_cpu_s"],
        sys_cpu_s=metrics["sys_cpu_s"],
        peak_rss_mb=metrics["peak_rss_mb"],
        peak_vmem_mb=peak_vmem_mb,
        major_page_faults=metrics["major_page_faults"],
        voluntary_ctx_switches=metrics["voluntary_ctx_switches"],
        involuntary_ctx_switches=metrics["involuntary_ctx_switches"],
        output_size_mb=output_size_mb,
        events_per_sec=events_per_sec,
    )


def read_peak_vmem_mb(path: Path) -> float | None:
    """Read the instrumentation's optional high-water mark, tolerating
    incomplete output and schema versions this k4bench cannot read."""
    try:
        with path.open() as stream:
            raw = json.load(stream)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        validate_event_schema(raw, source=path)
    except ValueError:
        return None
    value = raw.get("peak_vmem_mb")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _warn_missing_instrumentation(
    label: str, event_json_path: Path, instrumentation: str,
) -> None:
    """Warn that the instrumentation was set up but left no usable output.

    Both judged memory metrics are read from this file, so a run that loses it
    still succeeds while contributing no memory judgement at all — a state
    worth distinguishing from a run whose memory simply did not move.
    """

    print(
        f"  WARNING [{label}]: "
        f"{instrumentation} output is missing or unusable.\n"
        f"           Expected {event_json_path}; this run contributes "
        f"no judged memory metrics."
    )
