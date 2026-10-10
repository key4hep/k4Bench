#!/usr/bin/env python3
"""
Run one nightly benchmark job: resolve its inputs, run k4bench, record the run.

  benchmark_job.py <output_dir>

Called by nightly_benchmark.sh once the Key4hep stack is sourced and k4bench is
installed. The job is its record from list_benchmarks.py, as JSON in
BENCHMARK_JOB; the stack it runs on is the one nightly_benchmark.sh resolved
(K4H_STACK_SETUP, K4H_PLATFORM, K4H_RELEASE). Into <output_dir> it writes the
benchmark's results and logs, machine_info.json and run_info.json.

The exit code is k4bench's: a sweep with one failed configuration still
records its run, so the script that uploads the results can turn the job red
afterwards. A job that cannot start (its geometry, steering file or input is
missing) fails before anything is measured and records no run_info.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import machine_info
import run_info


class JobError(RuntimeError):
    """The job cannot run as configured."""


@dataclass(frozen=True)
class Job:
    """A benchmark job's record, resolved against the sourced Key4hep stack.

    *xml_path* is the configured geometry with its ``$VAR``s expanded, still
    relative to ``$K4GEO`` when it came from there; *detector_xml* is the
    compact file the job loads. *ddsim_args* are the arguments as ddsim reads
    them, with the steering file and the local copy of the input in front of
    the configured ones. *pythonpath* is the ``PYTHONPATH`` ddsim runs with.
    """

    record: dict[str, str]
    xml_path: str
    detector_xml: Path
    steering_path: str
    ddsim_args: str
    pythonpath: str | None

    @property
    def detector(self) -> str:
        """The EOS detector directory: the compact file's basename."""
        return self.detector_xml.name.removesuffix(".xml")

    def k4bench_argv(self, output_dir: Path) -> list[str]:
        """The k4bench command line that benchmarks this job into *output_dir*."""
        rec = self.record
        argv = [
            "--xml", str(self.detector_xml),
            "--events", rec["n_events"],
            "--output-dir", str(output_dir),
        ]
        if rec["sweep"] == "true":
            argv.append("--sweep")
        for flag, key in (
            ("--sweep-detectors", "sweep_detectors"),
            ("--include-only", "include_only"),
            ("--exclude-only", "exclude_only"),
        ):
            if names := rec[key].split():
                argv += [flag, *names]
        if rec["verbose"] == "true":
            argv.append("--verbose")
        if self.ddsim_args:
            argv.append(f"--ddsim-args={self.ddsim_args}")
        return argv


def resolve(record: dict[str, str], env: dict[str, str]) -> Job:
    """*record* resolved against the sourced stack's *env*, fetching its input.

    Paths may reference Key4hep variables (``$K4GEO``, ``$FCCCONFIG``,
    ``$DD4hepINSTALL``, ...), so they are expanded here, after the stack is
    sourced; a geometry path that is not absolute is relative to ``$K4GEO``.
    """
    xml_path = os.path.expandvars(record["xml"])
    detector_xml = Path(xml_path) if xml_path.startswith("/") else Path(env["K4GEO"]) / xml_path
    if not detector_xml.is_file():
        raise JobError(f"XML not found: {detector_xml}")

    ddsim_args = record["ddsim_args"]
    steering_path = ""
    pythonpath = None
    if record["steering_file"]:
        steering_path = os.path.expandvars(record["steering_file"])
        if not Path(steering_path).is_file():
            raise JobError(f"steering file not found: {steering_path}")
        # In front, so a sample-level --steeringFile in ddsim_args overrides it.
        ddsim_args = f"--steeringFile {steering_path} {ddsim_args}"
        # ddsim exec()s the steering file without putting its directory on
        # sys.path, so a steering file importing a sibling module (CLDConfig's
        # cld_arc_steer.py does `from cld_steer import *`) needs it there.
        pythonpath = f"{os.path.dirname(steering_path) or '.'}:{env.get('PYTHONPATH', '')}"

    if record["input_files"]:
        # HepMC inputs can't be streamed over xrootd (ROOT mis-parses the text
        # as a ROOT file and crashes), so they are fetched to a local path.
        local_input = f"/tmp/{os.path.basename(record['input_files'])}"
        subprocess.run(["xrdcp", "--force", record["input_files"], local_input], check=True)
        ddsim_args = f"--inputFiles {local_input} {ddsim_args}"

    job = Job(record, xml_path, detector_xml, steering_path, ddsim_args, pythonpath)
    print(f"Detector : {job.detector}")
    print(f"XML      : {job.detector_xml}")
    print(f"Steering : {job.steering_path or '<none>'}")
    print(f"ddsim    : {job.ddsim_args or '<none>'}")
    return job


def run(job: Job, output_dir: Path, env: dict[str, str]) -> int:
    """Benchmark *job* into *output_dir* with k4bench and return its exit code.

    The runner's CPU set, when it has one, pins k4bench and everything it starts.
    """
    pin = ["taskset", "-c", env["RUNNER_CPU_SET"]] if env.get("RUNNER_CPU_SET") else []
    command = [*pin, "k4bench", *job.k4bench_argv(output_dir)]
    job_env = dict(env)
    if job.pythonpath is not None:
        job_env["PYTHONPATH"] = job.pythonpath
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, env=job_env).returncode


def _group(title: str) -> None:
    print(f"::endgroup::\n::group::{title}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <output_dir>")
    output_dir = Path(args[0])
    env = dict(os.environ)
    record = json.loads(env["BENCHMARK_JOB"])

    print("::group::5. Resolve inputs", flush=True)
    try:
        job = resolve(record, env)
    except (JobError, subprocess.CalledProcessError) as exc:
        print("::endgroup::")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    # Once, so run_info.json and the upload path agree even if the benchmark
    # runs across midnight.
    date = time.strftime("%Y-%m-%d")

    _group("6. Collect machine info (start)")
    machine_info.cmd_start(output_dir)
    _group("7. Run benchmark")
    returncode = run(job, output_dir, env)
    _group("8. Write run metadata")
    run_info.write_run_info(output_dir, job, env, date=date)
    machine_info.cmd_finalize(output_dir)
    print("::endgroup::", flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
