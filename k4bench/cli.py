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

Benchmark a k4run job instead (``k4bench k4run --help``), here CLD
reconstruction run from a copy of its configuration directory, with a
truth-tracking variant next to the baseline::

    k4bench k4run CLDReconstruction.py \\
             --stage-dir $CLDCONFIG/share/CLDConfig \\
             --events 30 \\
             --k4run-args="--inputFiles $PWD/sim.edm4hep.root --compactFile $K4GEO/FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml" \\
             --variant truth_tracking=--truthTracking
"""

from __future__ import annotations

import argparse
import pickle
import shlex
import sys
from pathlib import Path

from k4bench.benchmark.ddsim import BenchmarkConfig, run_sweep, select_sweep
from k4bench.benchmark.k4run import K4runConfig, run_k4run_benchmark
from k4bench.geometry.scanner import get_detector_names
from k4bench.results.model import RunResult
from k4bench.results.reporter import print_summary, save_csv

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LOG_ROOT     = Path("logs")
DEFAULT_OUTPUT_FILE  = Path("/tmp/k4bench_out.edm4hep.root")
DEFAULT_EVENTS      = 2

#: First argument that selects the k4run benchmark instead of ddsim.
K4RUN_COMMAND = "k4run"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the benchmark, save results.

    ``k4bench k4run ...`` benchmarks a k4run job; any other command line is a
    ddsim benchmark.

    Returns the exit code (0 = success, 1 = error).
    """
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == [K4RUN_COMMAND]:
        return _main_k4run(argv[1:])

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

    return _report(results, args)


def _main_k4run(argv: list[str]) -> int:
    """``k4bench k4run``: benchmark a k4run job and its variants."""
    args = _build_k4run_parser().parse_args(argv)

    if args.output_dir is None:
        args.output_dir = DEFAULT_LOG_ROOT / Path(args.options[0]).stem

    try:
        config = _build_k4run_config(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return _report(run_k4run_benchmark(config), args)


def _report(results: list[RunResult], args: argparse.Namespace) -> int:
    """Print and save *results*; the exit code is 1 if any run failed."""
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
    _add_output_args(parser, program="ddsim", default_dir="logs/<xml_stem>/")

    return parser


def _add_output_args(parser: argparse.ArgumentParser, *, program: str, default_dir: str) -> None:
    """The output flags both benchmarks share: where results go, an optional
    pickle of them, and whether *program*'s output is streamed."""
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        dest="output_dir",
        help=f"Directory for logs and results. Defaults to {default_dir}.",
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
        help=f"Stream {program} output to stdout during each run.",
    )


def _build_k4run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"k4bench {K4RUN_COMMAND}",
        description=(
            "Benchmark a k4run job, measured per component by the k4Bench "
            "Gaudi auditor. Each run starts in a fresh working directory, so "
            "give paths in --k4run-args as absolute paths."
        ),
    )
    parser.add_argument(
        "options",
        nargs="+",
        metavar="OPTIONS",
        help=(
            "Gaudi options files, loaded in order; relative to --stage-dir "
            "when it is given."
        ),
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        metavar="DIR",
        dest="stage_dir",
        help=(
            "Directory copied into every run's working directory, for jobs "
            "that must run from their own configuration directory."
        ),
    )
    parser.add_argument(
        "--events",
        type=int,
        default=DEFAULT_EVENTS,
        metavar="N",
        help=f"Number of events per run (default: {DEFAULT_EVENTS}).",
    )
    parser.add_argument(
        "--k4run-args",
        default="",
        metavar="ARGS",
        dest="k4run_args",
        help=(
            "Arguments passed verbatim to k4run after the options files, as a "
            'single quoted string: --k4run-args="--inputFiles /data/sim.root".'
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=ARGS",
        dest="variants",
        help=(
            "One more run, labelled variant_NAME, with ARGS appended to "
            "--k4run-args. Repeat for several variants."
        ),
    )
    _add_output_args(parser, program="k4run", default_dir="logs/<options stem>/")
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

    mode, detector_names = select_sweep(
        sweep=args.sweep,
        sweep_detectors=args.sweep_detectors or (),
        include_only=args.include_only or (),
        exclude_only=args.exclude_only or (),
    )

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


def _build_k4run_config(args: argparse.Namespace) -> K4runConfig:
    """Translate parsed ``k4bench k4run`` arguments into a :class:`K4runConfig`."""
    variants: dict[str, list[str]] = {}
    for spec in args.variants:
        name, sep, variant_args = spec.partition("=")
        if not sep or not name:
            raise ValueError(f"--variant takes NAME=ARGS, got {spec!r}")
        if name in variants:
            raise ValueError(f"--variant {name!r} is given twice")
        variants[name] = shlex.split(variant_args)

    return K4runConfig(
        options=args.options,
        n_events=args.events,
        log_dir=args.output_dir,
        stage_dir=args.stage_dir,
        extra_args=shlex.split(args.k4run_args) if args.k4run_args else [],
        variants=variants,
        verbose=args.verbose,
    )
