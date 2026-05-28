#!/usr/bin/env python3
"""
Expand .github/benchmarks/*.yml into a flat list of benchmark jobs and print
it as JSON to stdout. The output feeds the nightly workflow matrix; each
record is a fully-merged config that can be consumed by the runner as plain
env vars — no YAML parsing happens downstream.

Top-level keys in a benchmark file are detector-wide defaults; keys inside a
samples[] entry override them for that sample only. The single exception is
ddsim_args: top-level and sample-level strings are concatenated (top first),
which lets shared ddsim flags live at the detector level. Lists are joined
to space-separated strings so they round-trip through env vars unchanged.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

BENCH_DIR = Path(".github/benchmarks")
CONFIG_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SAMPLE_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")

SCALAR_KEYS = ("xml", "n_events", "ddsim_args", "verbose", "sweep", "steering_file")
LIST_KEYS   = ("input_files", "include_only", "exclude_only")


def _scalar(v) -> str:
    if v is None:           return ""
    if isinstance(v, bool): return str(v).lower()
    return str(v).strip()


def _list(v) -> str:
    if v is None:           return ""
    if isinstance(v, list): return " ".join(str(x) for x in v)
    return str(v).strip()


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def expand(path: Path) -> list[dict]:
    config = path.stem
    if not CONFIG_RE.match(config):
        _die(f"invalid config name {config!r} (from {path})")

    cfg = yaml.safe_load(path.read_text()) or {}
    samples = cfg.get("samples") or []
    if not samples:
        _die(f"no samples defined in {path}")

    records = []
    for s in samples:
        name = s.get("name") if isinstance(s, dict) else None
        if not name:
            _die(f"sample entry in {path} is missing 'name'")
        if not SAMPLE_RE.match(name):
            _die(f"invalid sample name {name!r} in {path}")

        def merge(k):
            # ddsim_args concatenates (top + sample); everything else overrides.
            if k == "ddsim_args":
                parts = [v for v in (cfg.get(k), s.get(k)) if v]
                return " ".join(str(p).strip() for p in parts) if parts else None
            return s[k] if k in s else cfg.get(k)

        rec = {"config": config, "sample": name}
        for k in SCALAR_KEYS: rec[k] = _scalar(merge(k))
        for k in LIST_KEYS:   rec[k] = _list(merge(k))

        loc = f"{path}::{name}"
        if not rec["n_events"].isdigit() or int(rec["n_events"]) <= 0:
            _die(f"n_events must be a positive integer ({loc})")
        if rec["input_files"] and "--enableGun" in rec["ddsim_args"]:
            _die(f"input_files and '--enableGun' in ddsim_args are mutually exclusive ({loc})")
        modes = (rec["sweep"] == "true") + bool(rec["include_only"]) + bool(rec["exclude_only"])
        if modes > 1:
            _die(f"sweep / include_only / exclude_only are mutually exclusive ({loc})")

        records.append(rec)
    return records


def main() -> None:
    paths = sorted(BENCH_DIR.glob("*.yml"))
    if not paths:
        _die(f"no benchmark configs found in {BENCH_DIR}/")
    items = [r for p in paths for r in expand(p)]
    print(json.dumps(items))


if __name__ == "__main__":
    main()
