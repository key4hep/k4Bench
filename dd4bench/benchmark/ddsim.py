"""Benchmark ddsim across geometry configurations.

This module is the top-level orchestrator.  It wires together:

* :mod:`dd4bench.geometry.scanner`  — discover subdetector names
* :mod:`dd4bench.geometry.patcher`  — produce patched XML files
* :mod:`dd4bench.runner.executor`   — time each ddsim run
* :mod:`dd4bench.results.model`     — collect results

All runs are sequential.  Parallel execution would skew wall-time and
RSS metrics because competing processes share CPU, cache, and memory

Sweep modes
-----------
FULL
    Baseline (full geometry) + one run per subdetector with that
    detector removed.
INCLUDE_ONLY
    Single run with only the named detectors active (all others
    removed).  No baseline.
EXCLUDE_ONLY
    One run per non-excluded detector with that detector removed.
    No baseline.
COMPARE
    Baseline of geometry A vs baseline of geometry B — no patching.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dd4bench.geometry.patcher import (
    DetectorNotFoundError,
    patched_geometry,
    patched_geometry_keep_only,
)
from dd4bench.geometry.scanner import get_detector_names
from dd4bench.results.model import RunResult
from dd4bench.runner.executor import run_ddsim


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------


class SweepMode(Enum):
    """Selects which benchmark strategy :func:`run_sweep` executes."""

    BASELINE     = "baseline"     # single baseline run, no detector patching
    FULL         = "full"         # simulate with each detector individually removed
    INCLUDE_ONLY = "include_only" # single run with only the named detectors active
    EXCLUDE_ONLY = "exclude_only" # simulate with all detectors except the named ones
    COMPARE      = "compare"      # baseline of geometry A vs baseline of geometry B


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """All parameters needed to run a benchmark sweep.

    Parameters
    ----------
    xml_path:
        Top-level compact XML for the primary geometry.
    n_events:
        Number of events to simulate per run.
    output_file:
        Temporary EDM4hep ROOT file written by ddsim.  Reused across
        runs (overwritten each time); only its size at run-end matters.
    log_dir:
        Directory where per-run ``.log`` files are written.
    mode:
        Which benchmark strategy to execute; see :class:`SweepMode`.
    detector_names:
        For ``INCLUDE_ONLY`` — simulate with only these detectors active.
        For ``EXCLUDE_ONLY`` — simulate with all detectors except these.
        Ignored for ``FULL`` and ``COMPARE`` modes.
    xml_path_b:
        Second geometry XML, required for ``COMPARE`` mode only.
    setup_script:
        Optional shell script sourced before each ddsim invocation.
    extra_args:
        Additional ddsim arguments passed verbatim to every run
        (e.g. ``["--runType=batch", "--enableGun", "--gun.particle", "e-"]``).
    verbose:
        If True, print ddsim stdout in real time instead of buffering until run-end.
    """

    xml_path: Path
    n_events: int
    output_file: Path
    log_dir: Path
    mode: SweepMode = SweepMode.FULL
    detector_names: list[str] = field(default_factory=list)
    xml_path_b: Path | None = None
    setup_script: Path | None = None
    extra_args: list[str] = field(default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.mode == SweepMode.COMPARE and self.xml_path_b is None:
            raise ValueError("COMPARE mode requires xml_path_b to be set.")
        if self.mode in (SweepMode.INCLUDE_ONLY, SweepMode.EXCLUDE_ONLY) and not self.detector_names:
            raise ValueError(
                f"{self.mode.value} mode requires detector_names to be non-empty."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sweep(config: BenchmarkConfig) -> list[RunResult]:
    """Execute a benchmark sweep and return all results.

    Dispatches to the appropriate strategy based on ``config.mode``.
    Failed ddsim runs are marked with a non-zero return code and
    included in the results; the sweep always continues to completion.

    Parameters
    ----------
    config:
        All parameters for the sweep; see :class:`BenchmarkConfig`.

    Returns
    -------
    list[RunResult]
        Results in execution order.
    """
    config.log_dir.mkdir(parents=True, exist_ok=True)

    if config.mode == SweepMode.BASELINE:
        return _run_baseline(config)
    elif config.mode == SweepMode.COMPARE:
        return _run_compare(config)
    elif config.mode == SweepMode.INCLUDE_ONLY:
        return _run_include_only_sweep(config)
    else:
        return _run_removal_sweep(config)


# ---------------------------------------------------------------------------
# Sweep strategies
# ---------------------------------------------------------------------------


def _run_baseline(config: BenchmarkConfig) -> list[RunResult]:
    """Single baseline run with no detector patching."""
    _print_run_header(1, 1, "baseline_all", config.xml_path)
    result = _timed_run(xml_path=config.xml_path, label="baseline_all", config=config)
    return [result]


def _run_removal_sweep(config: BenchmarkConfig) -> list[RunResult]:
    """Removal runs (FULL / EXCLUDE_ONLY).

    A baseline run (full geometry, no patching) is included only for FULL
    mode.  EXCLUDE_ONLY runs removals for all non-excluded detectors.
    """

    detectors_to_remove = _resolve_detectors(config)
    results: list[RunResult] = []

    if config.mode == SweepMode.FULL:
        total = 1 + len(detectors_to_remove)
        _print_run_header(1, total, "baseline_all", config.xml_path)
        results.append(
            _timed_run(xml_path=config.xml_path, label="baseline_all", config=config)
        )
        removal_start = 2
    else:
        total = len(detectors_to_remove)
        removal_start = 1

    for i, name in enumerate(detectors_to_remove, start=removal_start):
        label = f"without_{name}"
        try:
            with patched_geometry(config.xml_path, name) as tmp_xml:
                _print_run_header(i, total, label, tmp_xml)
                results.append(
                    _timed_run(xml_path=tmp_xml, label=label, config=config)
                )
        except DetectorNotFoundError as exc:
            print(f"  SKIP {label}: {exc}\n")
        except Exception:
            print(f"  ERROR in {label}:\n{traceback.format_exc()}")

    return results


def _run_include_only_sweep(config: BenchmarkConfig) -> list[RunResult]:
    """Single run keeping only the named detectors active.

    All detectors not in ``config.detector_names`` are removed from the
    geometry.  The result is labelled ``only_<name1>_<name2>_...``.
    """
    print("Scanning geometry for subdetectors …")
    all_names = set(get_detector_names(config.xml_path))

    keep = set(config.detector_names)
    unknown = keep - all_names
    if unknown:
        print(f"WARNING: detectors not found in geometry, will be skipped: {sorted(unknown)}")
    keep -= unknown

    label = "only_" + "_".join(sorted(keep))
    print(f"Keeping {len(keep)} detector(s): {sorted(keep)}\n")

    results: list[RunResult] = []
    try:
        with patched_geometry_keep_only(config.xml_path, keep) as tmp_xml:
            _print_run_header(1, 1, label, tmp_xml)
            results.append(_timed_run(xml_path=tmp_xml, label=label, config=config))
    except Exception:
        print(f"  ERROR in {label}:\n{traceback.format_exc()}")

    return results


def _run_compare(config: BenchmarkConfig) -> list[RunResult]:
    """Baseline of geometry A vs baseline of geometry B."""

    assert config.xml_path_b is not None  # guaranteed by __post_init__

    results: list[RunResult] = []

    _print_run_header(1, 2, "geometry_a", config.xml_path)
    results.append(
        _timed_run(xml_path=config.xml_path, label="geometry_a", config=config)
    )

    _print_run_header(2, 2, "geometry_b", config.xml_path_b)
    results.append(
        _timed_run(xml_path=config.xml_path_b, label="geometry_b", config=config)
    )

    return results


# ---------------------------------------------------------------------------
# Detector list resolution
# ---------------------------------------------------------------------------


def _resolve_detectors(config: BenchmarkConfig) -> list[str]:
    """Return the ordered list of detectors to remove for the current mode."""

    print("Scanning geometry for subdetectors …")
    all_names = get_detector_names(config.xml_path)

    if not all_names:
        print("WARNING: no subdetectors found — only baseline will run.\n")
        return []

    if config.mode == SweepMode.FULL:
        selected = all_names
    elif config.mode == SweepMode.EXCLUDE_ONLY:
        selected = [d for d in all_names if d not in set(config.detector_names)]
    else:
        raise ValueError(f"Unexpected mode for removal sweep: {config.mode}")

    print(f"Found {len(all_names)} subdetectors, running {len(selected)}:")
    for name in selected:
        print(f"  - {name}")
    print()

    return selected


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _timed_run(*, xml_path: Path, label: str, config: BenchmarkConfig) -> RunResult:
    """Execute one ddsim run and print a one-line status summary."""
    result = run_ddsim(
        xml_path=xml_path,
        label=label,
        n_events=config.n_events,
        output_file=config.output_file,
        log_dir=config.log_dir,
        setup_script=config.setup_script,
        extra_args=config.extra_args,
        verbose=config.verbose,
    )
    _print_run_result(result)
    return result


def _print_run_header(index: int, total: int, label: str, xml_path: Path) -> None:
    print(f"[{index}/{total}] {label}")
    print(f"         XML: {xml_path}")


def _print_run_result(result: RunResult) -> None:
    status = "ok" if result.succeeded else f"FAILED (rc={result.returncode})"
    wall = f"{result.wall_time_s:.1f}s"        if result.wall_time_s    is not None else "N/A"
    rss  = f"{result.peak_rss_mb:.0f} MB"      if result.peak_rss_mb   is not None else "N/A"
    out  = f"{result.output_size_mb:.2f} MB"   if result.output_size_mb is not None else "N/A"
    eps  = f"{result.events_per_sec:.3f} ev/s" if result.events_per_sec is not None else "N/A"
    print(f"         Status: {status}  |  Wall: {wall}  |  RSS: {rss}  |  Output: {out}  |  {eps}")
    print(f"         Log:    {result.label}.log\n")
