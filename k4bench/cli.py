"""Command-line interface for k4bench.

Entry point registered as ``k4bench`` in pyproject.toml.

Usage examples
--------------
Single baseline run::

    k4bench --xml ALLEGRO.xml \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"

Full sweep (baseline + one run per detector removed)::

    k4bench --xml ALLEGRO.xml --sweep \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"

Partial sweep (baseline + one run per named detector removed)::

    k4bench --xml ALLEGRO.xml --sweep-detectors ECalBarrel HCalBarrel \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"

Simulate with only specific detectors::

    k4bench --xml ALLEGRO.xml \\
             --include-only ECalBarrel HCalBarrel \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"

Simulate with all detectors except specific ones::

    k4bench --xml ALLEGRO.xml \\
             --exclude-only ECalBarrel HCalBarrel \\
             --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"

List the detectors available in a geometry (no simulation is run)::

    k4bench --xml ALLEGRO.xml --list-detectors

Control output::

    k4bench --xml ALLEGRO.xml \\
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

from k4bench.benchmark.ddsim import BenchmarkConfig, SweepMode, run_sweep
from k4bench.geometry.scanner import get_detector_names
from k4bench.results.reporter import print_summary, save_csv

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LOG_ROOT     = Path("logs")
DEFAULT_OUTPUT_FILE  = Path("/tmp/k4bench_out.edm4hep.root")
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

    if args.list_detectors:
        return _list_detectors(args.xml)

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
        prog="k4bench",
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
    parser.add_argument(
        "--list-detectors",
        action="store_true",
        default=False,
        dest="list_detectors",
        help=(
            "Print the subdetector names found in --xml, one per line, and "
            "exit without running any simulation. Useful for discovering "
            "valid --include-only/--exclude-only names."
        ),
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
        "--sweep-detectors",
        nargs="+",
        metavar="DETECTOR",
        dest="sweep_detectors",
        help=(
            "Partial sweep: baseline + one run per named detector removed in "
            "turn. Like --sweep but restricted to the named detectors."
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
    
    return parser


# ---------------------------------------------------------------------------
# Detector listing
# ---------------------------------------------------------------------------


def _list_detectors(xml_path: Path) -> int:
    """Print the subdetector names discovered in *xml_path*, one per line."""
    names = get_detector_names(xml_path)

    if not names:
        print(f"No subdetectors found in {xml_path}", file=sys.stderr)
        return 1

    for name in names:
        print(name)

    return 0


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
    elif args.sweep_detectors:
        mode = SweepMode.FULL
        detector_names = args.sweep_detectors
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
        verbose=args.verbose
    )
