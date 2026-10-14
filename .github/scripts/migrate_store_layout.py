#!/usr/bin/env python3
"""
Re-file a benchmark store from the flat layout into the suite layout of
k4bench/layout.py. A one-time cutover, run on the EOS FUSE mount (e.g.
lxplus) before the first nightly that writes the suite layout:

  {detector}/...              -> runs/sim/{concept}/{detector}/...
  _k4run/{detector}/...       -> runs/reco/{concept}/{detector}/...
  _reports/{night}/           -> reports/sim/{night}/
  _inputs/{detector}/         -> inputs/reco/{concept}/{detector}/
  _reproducers/{recipe}       -> reproducers/sim/{recipe}   (copied)

Without it every reader of the suite layout finds no history: the dashboard
shows no runs, the regression report has no baselines and the reco benchmark
fetches its frozen input from a URL that does not exist.

Recipes are copied rather than moved, so the links standing pull-request
comments already carry keep working. Everything else is moved, so the run
files and their metadata are carried over unchanged. Prints the plan and
changes nothing unless --apply is given; refuses the whole plan if any move's
destination already exists, so it never overwrites, and re-running it after a
completed migration finds nothing to do (a recipe already copied is skipped). Other top-level names starting with
"_" or "." are left in place and listed.

Usage: migrate_store_layout.py ROOT [--apply]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Executing a file below ``.github/scripts`` otherwise puts that directory—not
# the checkout root—first on sys.path, so the layout is read from the checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    sys.path.remove(str(_REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPO_ROOT))

from k4bench.layout import (  # noqa: E402
    INPUTS, RECO, REPORTS, REPRODUCERS, RUNS, SIM,
    input_path, report_path, reproducer_path, run_path,
)

#: Top-level names of the suite layout, left alone.
NEW_KINDS = (RUNS, REPORTS, REPRODUCERS, INPUTS)


def plan(root: Path) -> tuple[list[tuple[Path, Path, bool]], list[Path]]:
    """The ``(source, destination, copy)`` steps that re-file *root*, and the
    top-level entries left where they are."""
    steps, kept = [], []

    def children(name: str) -> list[Path]:
        tree = root / name
        return sorted(tree.iterdir()) if tree.is_dir() else []

    for entry in sorted(root.iterdir()):
        name = entry.name
        if name in NEW_KINDS:
            continue
        if name == "_k4run":
            steps += [(d, root / run_path(RECO, d.name), False) for d in children(name)]
        elif name == "_reports":
            steps += [(n, root / report_path(SIM, n.name), False) for n in children(name)]
        elif name == "_inputs":
            steps += [(d, root / input_path(RECO, d.name), False) for d in children(name)]
        elif name == "_reproducers":
            # A recipe already copied is done; it stays at its source too.
            copies = [(f, root / reproducer_path(SIM) / f.name) for f in children(name)]
            steps += [(src, dst, True) for src, dst in copies if not dst.exists()]
        elif name.startswith(("_", ".")) or not entry.is_dir():
            kept.append(entry)
        else:
            steps.append((entry, root / run_path(SIM, name), False))
    return steps, kept


def apply(steps: list[tuple[Path, Path, bool]]) -> None:
    for src, dst, copy in steps:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(src, dst)
        else:
            src.rename(dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="store root, e.g. /eos/user/j/jbeirer/k4bench")
    parser.add_argument("--apply", action="store_true", help="carry the plan out (default: print it)")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"ERROR: store root is not a directory: {args.root}", file=sys.stderr)
        return 1
    steps, kept = plan(args.root)
    for src, dst, copy in steps:
        print(f"{'copy' if copy else 'move'} {src.relative_to(args.root)} -> {dst.relative_to(args.root)}")
    for entry in kept:
        print(f"keep {entry.relative_to(args.root)}")
    if clashes := [dst for _, dst, _ in steps if dst.exists()]:
        for dst in clashes:
            print(f"ERROR: destination exists: {dst}", file=sys.stderr)
        return 1
    if not steps:
        print("Nothing to migrate.")
    elif args.apply:
        apply(steps)
        print(f"Migrated {len(steps)} entries.")
    else:
        print("Dry run; re-run with --apply to carry it out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
