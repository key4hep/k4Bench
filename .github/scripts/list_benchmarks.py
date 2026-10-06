#!/usr/bin/env python3
"""
Expand .github/benchmarks/*.yml into benchmark jobs and print them as JSON
on stdout. Two output shapes:

  default  — flat list, one record per (detector, sample)
  --grouped — list of {detector, samples: [...]} so the nightly workflow can
              fan out one reusable-workflow call per detector and nest the
              sample matrix beneath it in the Actions UI

--only "A B/sample" keeps just the named jobs: a bare config name selects all
of its samples, "config/sample" one of them. A name matching no job is an error.

Each sample record is a fully-merged config consumed downstream as plain env
vars — no YAML parsing happens after this script runs.

Top-level keys in a benchmark file are detector-wide defaults; keys inside a
samples[] entry override them for that sample only. The single exception is
ddsim_args: top-level and sample-level strings are concatenated (top first),
which lets shared ddsim flags live at the detector level. Lists are joined
to space-separated strings so they round-trip through env vars unchanged, and
boolean keys are always "true" or "false" (an absent key is "false").
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import yaml

BENCH_DIR = Path(".github/benchmarks")
CONFIG_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SAMPLE_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")

SCALAR_KEYS = ("xml", "n_events", "ddsim_args", "steering_file", "timeout")
BOOL_KEYS   = ("verbose", "sweep")
LIST_KEYS   = ("input_files", "sweep_detectors", "include_only", "exclude_only")


def _scalar(v) -> str:
    if v is None:           return ""
    if isinstance(v, bool): return str(v).lower()
    return str(v).strip()


def _bool(v, key: str, loc: str) -> str:
    if v is None:           return "false"
    if isinstance(v, bool): return str(v).lower()
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower()
    _die(f"{key} must be true or false, got {v!r} ({loc})")


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
    if not isinstance(cfg, dict):
        _die(f"top-level YAML must be a mapping in {path}")
    samples = cfg.get("samples") or []
    if not isinstance(samples, list):
        _die(f"'samples' must be a list in {path}")
    if not samples:
        _die(f"no samples defined in {path}")

    records = []
    for s in samples:
        if not isinstance(s, dict):
            _die(f"sample entries in {path} must be mappings, got {type(s).__name__}")
        name = s.get("name")
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

        loc = f"{path}::{name}"
        rec = {"config": config, "sample": name}
        for k in SCALAR_KEYS: rec[k] = _scalar(merge(k))
        for k in BOOL_KEYS:   rec[k] = _bool(merge(k), k, loc)
        for k in LIST_KEYS:   rec[k] = _list(merge(k))

        if not rec["n_events"].isdigit() or int(rec["n_events"]) <= 0:
            _die(f"n_events must be a positive integer ({loc})")
        if rec["timeout"] and (not rec["timeout"].isdigit() or int(rec["timeout"]) <= 0):
            _die(f"timeout must be a positive integer of minutes ({loc})")
        if rec["input_files"] and "--enableGun" in shlex.split(rec["ddsim_args"]):
            _die(f"input_files and '--enableGun' in ddsim_args are mutually exclusive ({loc})")
        modes = (
            (rec["sweep"] == "true")
            + bool(rec["sweep_detectors"])
            + bool(rec["include_only"])
            + bool(rec["exclude_only"])
        )
        if modes > 1:
            _die(f"sweep / sweep_detectors / include_only / exclude_only are mutually exclusive ({loc})")

        records.append(rec)
    return records


def _eos_directory(rec: dict) -> tuple[str, str]:
    """The ``(detector, sample)`` part of the EOS directory *rec* uploads to.

    nightly_benchmark.sh names the detector directory after the compact file
    (``basename "${DETECTOR_XML}" .xml``), not after the benchmark config.
    """
    return Path(rec["xml"]).name.removesuffix(".xml"), rec["sample"]


def _check_unique_destinations(items: list[dict]) -> None:
    """Refuse two jobs that would upload into the same EOS run directory.

    Two benchmark configs naming compact files with one basename, and sharing a
    sample name, write ``run_info.json`` and every ``{label}_results.csv`` to the
    same ``{detector}/{platform}/{release}/{sample}/{date}/`` — whichever job
    uploads last silently replaces the other's results.
    """
    seen: dict[tuple[str, str], str] = {}
    for rec in items:
        where = _eos_directory(rec)
        if where in seen:
            _die(
                f"configs {seen[where]!r} and {rec['config']!r} both upload sample "
                f"{rec['sample']!r} to EOS detector directory {where[0]!r}; "
                "rename the sample in one of them"
            )
        seen[where] = rec["config"]


def _select(items: list[dict], only: str) -> list[dict]:
    """The jobs named by *only*, or every job when it is blank."""
    names = only.split()
    if not names:
        return items
    unknown = [
        n for n in names
        if not any(n in (i["config"], f"{i['config']}/{i['sample']}") for i in items)
    ]
    if unknown:
        _die(f"no benchmark job matches {', '.join(map(repr, unknown))}")
    return [
        i for i in items
        if i["config"] in names or f"{i['config']}/{i['sample']}" in names
    ]


def _group_by_detector(items: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in items:
        grouped.setdefault(item["config"], []).append(item)
    return [{"detector": d, "samples": s} for d, s in grouped.items()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--only", default="",
                        help='space-separated "config" or "config/sample" names')
    args = parser.parse_args()
    paths = sorted(BENCH_DIR.glob("*.yml"))
    if not paths:
        _die(f"no benchmark configs found in {BENCH_DIR}/")
    items = [r for p in paths for r in expand(p)]
    _check_unique_destinations(items)
    items = _select(items, args.only)
    print(json.dumps(_group_by_detector(items) if args.grouped else items))


if __name__ == "__main__":
    main()
