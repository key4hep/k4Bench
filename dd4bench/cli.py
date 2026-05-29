"""Command-line interface for dd4bench.

Entry point registered as ``dd4bench`` in pyproject.toml.

Usage examples
--------------
Single baseline run::

    dd4bench --xml ALLEGRO.xml \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"

Full sweep (baseline + one run per detector removed)::

    dd4bench --xml ALLEGRO.xml --sweep \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"

Simulate with only specific detectors::

    dd4bench --xml ALLEGRO.xml \\
             --include-only ECalBarrel HCalBarrel \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"

Simulate with all detectors except specific ones::

    dd4bench --xml ALLEGRO.xml \\
             --exclude-only ECalBarrel HCalBarrel \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"

Control output::

    dd4bench --xml ALLEGRO.xml \\
             --output-dir logs/ \\
             --pickle results.pkl \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"
"""

from __future__ import annotations

import argparse
import pickle
import shlex
import sys
from pathlib import Path

from dd4bench.benchmark.ddsim import BenchmarkConfig, SweepMode, run_sweep
from dd4bench.results.reporter import print_summary, save_csv

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LOG_ROOT     = Path("logs")
DEFAULT_OUTPUT_FILE  = Path("/tmp/dd4bench_out.edm4hep.root")
DEFAULT_EVENTS      = 2


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the benchmark, save results.

    Returns the exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = DEFAULT_LOG_ROOT / args.xml.stem

    try:
        config = _build_config(args)
    except (ValueError, SystemExit) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    results = run_sweep(config)

    print_summary(results)

    save_csv(results, args.output_dir)

    if args.pickle:
        pickle_path = args.output_dir / args.pickle
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        pickle_path.write_bytes(pickle.dumps(results))
        print(f"Results pickled to {pickle_path}")

    failed = [r for r in results if not r.succeeded]
    if failed:
        print(f"\n{len(failed)} run(s) failed: {[r.label for r in failed]}")
        return 1

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dd4bench",
        description="Benchmark ddsim across DD4hep geometry configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- geometry ---
    parser.add_argument(
        "--xml",
        metavar="PATH",
        type=Path,
        required=True,
        help="Top-level compact XML for the geometry under test.",
    )

    # --- sweep mode (mutually exclusive) ---
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help=(
            "Run a full sweep: baseline + one run per detector with that "
            "detector removed. Without this flag only the baseline is run."
        ),
    )
    sweep.add_argument(
        "--include-only",
        nargs="+",
        metavar="DETECTOR",
        dest="include_only",
        help="Sweep removing each named detector in turn (all others stay active).",
    )
    sweep.add_argument(
        "--exclude-only",
        nargs="+",
        metavar="DETECTOR",
        dest="exclude_only",
        help="Sweep over all detectors except the named ones.",
    )

    # --- simulation ---
    parser.add_argument(
        "--events",
        type=int,
        default=DEFAULT_EVENTS,
        metavar="N",
        help=f"Number of events per run (default: {DEFAULT_EVENTS}).",
    )
    parser.add_argument(
        "--ddsim-args",
        default="",
        metavar="ARGS",
        dest="ddsim_args",
        help=(
            "Additional arguments passed verbatim to ddsim, as a single "
            "quoted string. Use = syntax when the value starts with --: "
            '--ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform".'
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        metavar="PATH",
        dest="output_file",
        help=f"Temporary EDM4hep ROOT output file (default: {DEFAULT_OUTPUT_FILE}).",
    )

    # --- output ---
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        dest="output_dir",
        help=(
            "Directory for logs and results. "
            "Defaults to logs/<xml_stem>/."
        ),
    )
    parser.add_argument(
        "--pickle",
        metavar="FILENAME",
        default=None,
        help="If set, also save results as a pickle file inside --output-dir.",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Stream ddsim output to stdout during each run.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Per-run wall-clock limit in seconds. A run exceeding it is killed "
            "and recorded as failed instead of blocking the sweep (default: no limit)."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Translate parsed CLI arguments into a :class:`BenchmarkConfig`."""

    extra_args = shlex.split(args.ddsim_args) if args.ddsim_args else []

    # --- sweep modes ---
    if args.include_only:
        mode = SweepMode.INCLUDE_ONLY
        detector_names = args.include_only
    elif args.exclude_only:
        mode = SweepMode.EXCLUDE_ONLY
        detector_names = args.exclude_only
    elif args.sweep:
        mode = SweepMode.FULL
        detector_names = []
    else:
        mode = SweepMode.BASELINE
        detector_names = []

    return BenchmarkConfig(
        xml_path=args.xml,
        n_events=args.events,
        output_file=args.output_file,
        log_dir=args.output_dir,
        mode=mode,
        detector_names=detector_names,
        extra_args=extra_args,
        verbose=args.verbose,
        timeout_s=args.timeout,
    )
