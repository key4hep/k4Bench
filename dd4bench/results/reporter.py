"""Format and persist benchmark results.

Two responsibilities:
* :func:`save_csv`       — write results to a CSV file
* :func:`print_summary`  — print a formatted table to stdout
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

from dd4bench.results.model import RunResult

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


def save_csv(results: list[RunResult], path: Path) -> None:
    """Write *results* to a CSV file at *path*.

    Parameters
    ----------
    results:
        Results to serialise; must be non-empty.
    path:
        Destination path.  Parent directories are created if absent.
    """
    if not results:
        raise ValueError("Cannot write CSV: results list is empty.")

    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [dataclasses.asdict(r) for r in results]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {path}")
