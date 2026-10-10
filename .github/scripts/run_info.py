"""
The run_info.json record of one nightly benchmark job.

Written by benchmark_job.py once the benchmark has run (:func:`write_run_info`),
from the job it resolved, the CI identity (GITHUB_*), the stack the nightly
resolved (K4H_STACK_SETUP, K4H_PLATFORM, K4H_RELEASE) and the runner's CPU set
(RUNNER_CPU_SET) in the environment.

Besides the job parameters, the record holds:

  tool               what the job ran: ddsim, or k4run (absent on runs that
                     predate k4run benchmarks, which were all ddsim)
  configs            labels that produced a result CSV in <output_dir>
  configured_labels  labels the benchmark config was supposed to produce,
                     resolved against the geometry the job loaded
  random_seed        the ddsim seed the run simulated with
  k4h_packages       upstream commit of every git-built package in the stack

A k4run job also records its options files, stage directory, k4run arguments
and variants.

Resolving the roster or the stack provenance never fails the job: the
measurements are the deliverable, so either is recorded as unknown instead.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Executing a file below ``.github/scripts`` otherwise puts that directory—not
# the checkout root—first on sys.path. Prefer the checkout over a stale k4bench
# installation in long-lived CI/dev virtual environments.
_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    sys.path.remove(str(_REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPO_ROOT))

# Leaf modules that import only the standard library, so these imports cannot
# fail on a broken dependency the way the benchmark modules below could.
from k4bench.artifacts import RESULTS_SUFFIX, RUN_INFO, labelled_files  # noqa: E402
from k4bench.labels import RELEASE_PREFIX  # noqa: E402

if TYPE_CHECKING:
    from benchmark_job import Job


def random_seed(ddsim_args: str) -> int | None:
    """The ddsim seed *ddsim_args* sets, or ``None`` if it sets none.

    Timing is a function of which events were simulated, so a report comparing
    two nights is only comparing software if the seed is the same on both —
    recording it is what lets a report state that rather than assume it. None
    means the run drew a fresh seed, i.e. the workload is not reproducible.

    The *last* occurrence wins, because that is what argparse gives ddsim and
    the benchmark configs concatenate detector-level args before sample-level
    ones — so a sample overriding the detector's seed would otherwise be
    recorded as the seed it replaced, and the record would name a workload that
    never ran.
    """
    tokens = shlex.split(ddsim_args)
    seed = None
    for i, token in enumerate(tokens):
        if token.startswith("--random.seed="):
            raw = token.partition("=")[2]
        elif token == "--random.seed" and i + 1 < len(tokens):
            raw = tokens[i + 1]
        else:
            continue
        try:
            seed = int(raw)
        except ValueError:
            seed = None
    return seed


def produced_configs(output_dir: Path) -> list[str]:
    """Labels that wrote a result CSV into *output_dir*."""
    return [label for _, label in labelled_files(output_dir, RESULTS_SUFFIX)]


def configured_labels(job: Job) -> list[str] | None:
    """The labels the benchmark config was supposed to produce, or ``None``.

    Kept separate from :func:`produced_configs`: that is what produced a CSV,
    while this is what was supposed to produce one. If the benchmark process
    was killed part-way through a sweep, the difference is the missing-config
    failure the nightly report needs to surface. ``None`` — the legacy metadata
    the report already understands — when the roster cannot be resolved, for
    any reason including a benchmark module that fails to import.
    """
    try:
        if job.tool == "k4run":
            from k4bench.benchmark.k4run import planned_k4run_labels

            return planned_k4run_labels(list(job.variants))

        from k4bench.benchmark.ddsim import planned_config_labels, select_sweep

        rec = job.record
        mode, names = select_sweep(
            sweep=rec["sweep"] == "true",
            sweep_detectors=shlex.split(rec["sweep_detectors"]),
            include_only=shlex.split(rec["include_only"]),
            exclude_only=shlex.split(rec["exclude_only"]),
        )
        return planned_config_labels(job.detector_xml, mode, names)
    except Exception as exc:
        print(f"WARNING: could not resolve configured labels: {exc}", file=sys.stderr)
        return None


def build_run_info(
    job: Job,
    env: dict[str, str],
    *,
    date: str,
    configs: list[str],
    labels: list[str] | None,
) -> dict:
    """The run_info.json record of one job, without stack provenance."""
    rec = job.record
    run_id = env["GITHUB_RUN_ID"]
    release = env["K4H_RELEASE"]
    record = {
        "tool":             job.tool,
        "date":             date,
        "platform":         env["K4H_PLATFORM"],
        "k4h_release":      f"{RELEASE_PREFIX}{release}",
        "k4h_release_date": release,
        # The resolved LCG view, never whatever a sourced stack left in its own
        # variables: it is what the release label and the EOS path came from.
        "k4h_stack_setup":  env["K4H_STACK_SETUP"],
        "detector":         job.detector,
        "sample":           rec["sample"],
        # The compact file this run loaded, relative to $K4GEO when it came from
        # there. Recorded so attribution can state as a *fact* which pull
        # requests touch the geometry this run actually reads, instead of
        # inferring it from path names.
        "xml_path":         job.xml_path,
        "configured_xml_path": rec["xml"],
        "github_run_id":    run_id,
        "github_run_url": (
            f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}/actions/runs/{run_id}"
        ),
        "commit_sha":       env["GITHUB_SHA"],
        "n_events":         int(rec["n_events"]),
        "sweep":            rec["sweep"] == "true",
        "ddsim_args":       job.ddsim_args,
        "random_seed":      random_seed(job.ddsim_args),
        # How the benchmark was invoked, beyond its arguments.  Both move a timing
        # measurement -- --verbose streams ddsim's output while it is being timed,
        # and the runner pins the process to a fixed CPU set -- so a reproducer that
        # does not know them cannot say it ran the same measurement.
        "verbose":          rec["verbose"] == "true",
        "runner_cpu_set":   env.get("RUNNER_CPU_SET", ""),
        # Preserve the configured source values.  ddsim_args above names the /tmp
        # copy actually read by ddsim; a reproducer also needs the xrootd URL from
        # which that ephemeral file was obtained.
        "input_files":      shlex.split(rec["input_files"]),
        "steering_file":    rec["steering_file"],
        "resolved_steering_file": job.steering_path,
        "configs":          configs,
        "configured_labels": labels,
    }
    if job.tool == "k4run":
        record |= {
            "options":            rec["options"].split(),
            # As configured, and as the job found it once $VARs were expanded.
            "stage_dir":          rec["stage_dir"],
            "resolved_stage_dir": job.stage_path,
            # With $VARs expanded: the arguments the job actually read.
            "k4run_args":         job.k4run_args,
            "variants":           job.variants,
        }
    return record


def add_stack_provenance(run_info: dict, stack_setup: str) -> None:
    """Record the upstream commit of every package the stack built from git.

    A regression found weeks from now can then still be traced to the PRs in
    its blame window. This is the only moment the answer exists: LCG overwrites
    each weekday slot a week later, after which the view that produced these
    numbers is gone. Never fatal — the measurements are the deliverable,
    provenance is metadata.
    """
    try:
        from k4bench.provenance.stack import read_stack
        manifest, packages = read_stack(stack_setup)
        if manifest is None:
            print("WARNING: no stack provenance metadata found")
        else:
            run_info["k4h_stack_manifest"] = str(manifest)
            run_info["k4h_packages"] = packages
            print(f"Stack provenance: {len(packages)} git-built package(s)")
    except Exception as exc:
        print(f"WARNING: stack provenance not recorded: {exc}")


def write_run_info(output_dir: Path, job: Job, env: dict[str, str], *, date: str) -> Path:
    """Write *job*'s run_info.json into *output_dir*, once the benchmark has run."""
    run_info = build_run_info(
        job,
        env,
        date=date,
        configs=produced_configs(output_dir),
        labels=configured_labels(job),
    )
    add_stack_provenance(run_info, env["K4H_STACK_SETUP"])

    path = output_dir / RUN_INFO
    path.write_text(json.dumps(run_info, indent=2))
    print(f"Written: {path}")
    return path
