"""Format and persist benchmark results.

Three responsibilities:
* :func:`save_csv`          — write results to a CSV file
* :func:`print_summary`     — print a formatted table to stdout
* :func:`print_run_result`  — print one run's status line as it finishes
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

from k4bench.artifacts import RESULTS_SUFFIX, label_path
from k4bench.results.model import RunResult

# Column format string shared by header and data rows.
_COL = "{:<45} {:>9} {:>10} {:>11} {:>9} {:>8} {:>4}"


def print_summary(results: list[RunResult]) -> None:
    """Print a formatted summary table to stdout.

    Parameters
    ----------
    results:
        Results in the order they should appear in the table.
    """
    header = _COL.format(
        "Label", "Wall(s)", "RSS(MB)", "CPU usr(s)", "Out(MB)", "ev/s", "RC"
    )
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in results:
        wall = f"{r.wall_time_s:.1f}" if r.wall_time_s is not None else "N/A"
        rss = f"{r.peak_rss_mb:.1f}" if r.peak_rss_mb is not None else "N/A"
        cpu = f"{r.user_cpu_s:.1f}" if r.user_cpu_s is not None else "N/A"
        out = f"{r.output_size_mb:.2f}" if r.output_size_mb is not None else "N/A"
        eps = f"{r.events_per_sec:.3f}" if r.events_per_sec is not None else "N/A"
        rc = r.returncode if r.returncode is not None else "N/A"
        print(_COL.format(r.label, wall, rss, cpu, out, eps, rc))

    print(sep)


def print_run_result(result: RunResult) -> None:
    """Print the status line of one finished run."""
    status = "ok" if result.succeeded else f"FAILED (rc={result.returncode})"
    wall = f"{result.wall_time_s:.1f}s"        if result.wall_time_s    is not None else "N/A"
    rss  = f"{result.peak_rss_mb:.0f} MB"      if result.peak_rss_mb   is not None else "N/A"
    out  = f"{result.output_size_mb:.2f} MB"   if result.output_size_mb is not None else "N/A"
    eps  = f"{result.events_per_sec:.3f} ev/s" if result.events_per_sec is not None else "N/A"
    print(f"         Status: {status}  |  Wall: {wall}  |  RSS: {rss}  |  Output: {out}  |  {eps}")
    print(f"         Log:    {result.label}.log\n")


def save_csv(results: list[RunResult], log_dir: Path) -> None:
    """Write one ``{label}_results.csv`` per result into *log_dir*.

    Parameters
    ----------
    results:
        Results to serialise; must be non-empty.
    log_dir:
        Directory to write into.  Created if absent.
    """
    if not results:
        raise ValueError("Cannot write CSV: results list is empty.")

    log_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        row = dataclasses.asdict(result)
        path = label_path(log_dir, result.label, RESULTS_SUFFIX)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        print(f"Results saved to {path}")
