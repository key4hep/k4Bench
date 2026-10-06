#!/usr/bin/env python3
"""Run one particle-gun event per configured geometry/steering pair.

Run inside the Key4hep nightly environment. This checks simulation startup and
output, using a standard gun rather than the full benchmark samples or sweeps.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from list_benchmarks import expand


def pairs(configs: list[Path]) -> dict[tuple[str, str], str]:
    """Deduplicate pairs after applying the nightly's sample-level overrides."""
    selected = {}
    for config in configs:
        for sample in expand(config):
            pair = sample["xml"], sample["steering_file"]
            selected.setdefault(pair, f"{sample['config']}/{sample['sample']}")
    return selected


def command(xml: str, steering: str, output: Path) -> tuple[list[str], dict[str, str]]:
    geometry = Path(os.path.expandvars(xml))
    if not geometry.is_absolute():
        geometry = Path(os.environ["K4GEO"]) / geometry
    if not geometry.is_file():
        raise ValueError(f"Missing geometry: {geometry}")
    env = dict(os.environ)
    args = ["ddsim"]
    if steering:
        path = Path(os.path.expandvars(steering)).resolve()
        if not path.is_file():
            raise ValueError(f"Missing steering file: {path}")
        args += ["--steeringFile", str(path)]
        # Match nightly_benchmark.sh: CLD steering imports a sibling module.
        env["PYTHONPATH"] = str(path.parent) + os.pathsep + env.get("PYTHONPATH", "")
    args += [
        "--compactFile",
        str(geometry),
        "--outputFile",
        str(output),
        "--runType",
        "batch",
        "--numberOfEvents",
        "1",
        "--enableGun",
        "--gun.particle",
        "e-",
        "--gun.energy",
        "10*GeV",
        "--gun.distribution",
        "uniform",
        "--random.seed",
        "42",
        "--random.enableEventSeed",
    ]
    return args, env


def smoke(xml: str, steering: str) -> None:
    with tempfile.TemporaryDirectory(prefix="k4bench-smoke-") as directory:
        output = Path(directory) / "smoke.edm4hep.root"
        args, env = command(xml, steering, output)
        print(shlex.join(args), flush=True)
        # Leave simulator output in the CI log, including when it times out.
        subprocess.run(args, env=env, cwd=directory, check=True, timeout=300)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"ddsim produced no output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path, help="benchmark YAML paths")
    args = parser.parse_args()
    configs = args.configs or [
        Path(path) for path in json.loads(os.environ.get("K4BENCH_SMOKE_CONFIGS", "[]"))
    ]
    if not configs:
        parser.error("specify benchmark YAMLs or set K4BENCH_SMOKE_CONFIGS to their JSON list")
    for (xml, steering), label in pairs(configs).items():
        print(f"::group::Smoke simulation: {label}", flush=True)
        try:
            smoke(xml, steering)
        finally:
            print("::endgroup::", flush=True)


if __name__ == "__main__":
    main()
