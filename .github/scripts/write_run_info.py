#!/usr/bin/env python3
"""Write ``run_info.json`` for a benchmark run directory.

Called from nightly_benchmark.sh (both the sequential and the parallel-variant
runs use that one script) to record run-level metadata. The per-config status is
derived from the ``returncode`` column of each ``*_results.csv`` already written
by ``dd4bench`` — the dashboard reads the same column — so a failed config is
recorded here and flagged there without extra bookkeeping.

GitHub context (run id/url, commit sha) is read from the environment, matching
the variables the workflow already exports.

Note: in a parallel sweep each variant job runs this against its own one-config
log dir and uploads run_info.json into the shared run dir (last-writer-wins). The
immutable fields agree across variants; the run-level aggregate (configs/status)
therefore reflects one variant and is not relied upon — the dashboard computes
true per-config status from the CSVs.

Usage:
    write_run_info.py --results-dir DIR --detector D --sample S --date YYYY-MM-DD \\
        --platform P --release R --n-events N [--sweep true|false] \\
        [--parallel true|false]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


def _bool(s: str) -> bool:
    return str(s).strip().lower() == "true"


def _config_status(results_dir: Path) -> tuple[list[str], list[str]]:
    """Return ``(configs, failed_configs)`` from ``*_results.csv`` in *results_dir*.

    A config is failed when its CSV has a non-zero (or missing/unparseable)
    ``returncode``. Labels are the CSV filename stems minus the ``_results``
    suffix, sorted for stable output.
    """
    configs: list[str] = []
    failed: list[str] = []
    for csv_path in sorted(results_dir.glob("*_results.csv")):
        label = csv_path.name[: -len("_results.csv")]
        configs.append(label)
        try:
            with csv_path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            rc = int(rows[0]["returncode"]) if rows else 1
        except (OSError, KeyError, ValueError, IndexError):
            rc = 1  # unreadable result counts as a failure
        if rc != 0:
            failed.append(label)
    return configs, failed


def _machine_summary(results_dir: Path) -> tuple[bool, list[str]]:
    """Return ``(machine_consistent, machines)`` from ``machine_info.json``.

    Each run dir carries one ``machine_info.json``; this reports the CPU model so
    the dashboard can show which machine a run used. ``machine_consistent`` is
    True here (a single job sees one machine) — a future cross-variant
    consistency check for parallel sweeps would aggregate these instead.
    """
    mi = results_dir / "machine_info.json"
    if not mi.exists():
        return True, []
    try:
        info = json.loads(mi.read_text())
    except (OSError, json.JSONDecodeError):
        return True, []
    model = info.get("cpu_model", "unknown")
    host = info.get("hostname", "")
    return True, [f"{model}@{host}" if host else model]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--release", required=True, help="Key4hep release date, e.g. 2026-05-19")
    ap.add_argument("--n-events", required=True, type=int)
    ap.add_argument("--sweep", default="false")
    ap.add_argument("--parallel", default="false")
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        return 1

    configs, failed = _config_status(results_dir)
    machine_consistent, machines = _machine_summary(results_dir)

    release = args.release
    m = re.search(r"\d{4}-\d{2}-\d{2}", release)
    release_date = m.group(0) if m else release

    server = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    run_url = f"{server}/{repo}/actions/runs/{run_id}" if (server and repo and run_id) else None

    run_info = {
        "date":               args.date,
        "platform":           args.platform,
        "k4h_release":         f"key4hep-{release_date}",
        "k4h_release_date":    release_date,
        "detector":           args.detector,
        "sample":             args.sample,
        "github_run_id":       run_id,
        "github_run_url":      run_url,
        "commit_sha":          os.environ.get("GITHUB_SHA"),
        "n_events":           args.n_events,
        "sweep":              _bool(args.sweep),
        "parallel":           _bool(args.parallel),
        "configs":            configs,
        "failed_configs":      failed,
        "status":             "failed" if failed else "ok",
        "machine_consistent":  machine_consistent,
        "machines":           machines,
    }

    out = results_dir / "run_info.json"
    out.write_text(json.dumps(run_info, indent=2))
    print(f"Written: {out}  (status={run_info['status']}, "
          f"{len(failed)}/{len(configs)} failed, machine_consistent={machine_consistent})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
