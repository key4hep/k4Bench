"""The recipe of a ddsim run: ``k4bench`` on the run's geometry, swept as the
row's label was."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from k4bench.blame.reproduce.records import _files, _integer, _pct, _tokens, _window
from k4bench.blame.reproduce.shell import (
    _continued,
    _differed,
    _fetch_stage,
    _half_setup,
    _note,
    _safe,
    _subshell,
)
from k4bench.labels import BASELINE_LABEL, INCLUDE_PREFIX, REMOVAL_PREFIX


@dataclass(frozen=True)
class ReproducerFacts:
    detector: str
    platform: str
    sample: str
    label: str
    metric: str
    sub_detector: str | None
    base_xml_path: str
    onset_xml_path: str
    base_configured_xml_path: str
    onset_configured_xml_path: str
    sweep_option: str
    sweep_value: str | None
    pct_change: str
    base_release: str
    onset_release: str
    base_stack_setup: str
    onset_stack_setup: str
    base_run_id: str
    onset_run_id: str
    base_actions_url: str
    onset_actions_url: str
    base_commit: str
    onset_commit: str
    base_n_events: int
    onset_n_events: int
    base_ddsim_args: str
    onset_ddsim_args: str
    base_verbose: bool
    onset_verbose: bool
    base_cpu_set: str
    onset_cpu_set: str
    base_seed: int | None
    onset_seed: int | None
    base_input_files: tuple[str, ...]
    onset_input_files: tuple[str, ...]
    base_steering_file: str
    onset_steering_file: str
    base_resolved_steering_file: str
    onset_resolved_steering_file: str
    parity_diffs: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        """Canonical, JSON-ready facts included in the comment digest."""
        return asdict(self)


def sweep_flag(label: str) -> str | None:
    """Invert a result label into its CLI sweep fragment.

    The empty string represents the unswept baseline. Multi-detector labels are
    intentionally not reversible: their hash records identity, not the roster.
    """
    if label == BASELINE_LABEL:
        return ""
    if label.startswith(REMOVAL_PREFIX):
        name = label.removeprefix(REMOVAL_PREFIX)
        if not name or re.fullmatch(r"\d+_detectors_[0-9a-fA-F]+", name):
            return None
        return f"--sweep-detectors {name}"
    if label.startswith(INCLUDE_PREFIX):
        name = label.removeprefix(INCLUDE_PREFIX)
        return f"--include-only {name}" if name else None
    return None


def _option_value(tokens: tuple[str, ...], option: str) -> str:
    value = ""
    for index, token in enumerate(tokens):
        if token.startswith(f"{option}="):
            value = token.partition("=")[2]
        elif token == option and index + 1 < len(tokens):
            value = tokens[index + 1]
    return value


def _workload_args(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Drop transport paths that are compared from their source metadata."""
    result = []
    skip = False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token in ("--inputFiles", "--steeringFile"):
            skip = True
            continue
        if token.startswith("--inputFiles=") or token.startswith("--steeringFile="):
            continue
        result.append(token)
    return tuple(result)


def facts(
    verdict: Any,
    base_info: dict[str, Any],
    onset_info: dict[str, Any],
    *,
    input_files: object = None,
) -> ReproducerFacts | None:
    """:func:`~k4bench.blame.reproduce.recipe.facts_from` for two ddsim runs."""
    inversion = sweep_flag(str(getattr(verdict, "label", "")))
    if inversion is None:
        return None
    window = _window(verdict, base_info, onset_info)
    if window is None:
        return None

    base_xml = str(base_info.get("xml_path") or "")
    onset_xml = str(onset_info.get("xml_path") or "")
    if not base_xml or not onset_xml:
        return None
    base_events, onset_events = window.base_events, window.onset_events
    base_commit, onset_commit = window.base_commit, window.onset_commit

    base_args = str(base_info.get("ddsim_args") or "")
    onset_args = str(onset_info.get("ddsim_args") or "")
    base_tokens = _tokens(base_args)
    onset_tokens = _tokens(onset_args)
    if base_tokens is None or onset_tokens is None:
        return None
    base_seed = _integer(base_info.get("random_seed"))
    onset_seed = _integer(onset_info.get("random_seed"))
    # Recorded from the nightly's own invocation. Both change what is measured:
    # ``--verbose`` streams ddsim's output, and the runner pins the benchmark to
    # a CPU set. Absent from records written before they were recorded, which
    # reads as the off/unpinned default rather than as a mismatch.
    base_verbose = bool(base_info.get("verbose"))
    onset_verbose = bool(onset_info.get("verbose"))
    base_cpu_set = str(base_info.get("runner_cpu_set") or "")
    onset_cpu_set = str(onset_info.get("runner_cpu_set") or "")
    fallback_files = _files(input_files)
    base_files = _files(base_info.get("input_files")) or fallback_files
    onset_files = _files(onset_info.get("input_files")) or fallback_files
    base_configured_xml = str(base_info.get("configured_xml_path") or "")
    onset_configured_xml = str(onset_info.get("configured_xml_path") or "")
    base_steering = str(base_info.get("steering_file") or "")
    onset_steering = str(onset_info.get("steering_file") or "")
    base_resolved_steering = str(
        base_info.get("resolved_steering_file")
        or _option_value(base_tokens, "--steeringFile")
    )
    onset_resolved_steering = str(
        onset_info.get("resolved_steering_file")
        or _option_value(onset_tokens, "--steeringFile")
    )
    parity = []
    for name, before, after in (
        ("n_events", base_events, onset_events),
        ("ddsim_args", _workload_args(base_tokens), _workload_args(onset_tokens)),
        ("random_seed", base_seed, onset_seed),
        ("commit_sha", base_commit, onset_commit),
        ("input_files", base_files, onset_files),
        ("steering_file", base_steering, onset_steering),
        ("verbose", base_verbose, onset_verbose),
    ):
        if before != after:
            parity.append(name)
    if (
        (base_configured_xml or onset_configured_xml)
        and base_configured_xml != onset_configured_xml
    ):
        parity.append("xml_path")

    option, _, value = inversion.partition(" ")
    return ReproducerFacts(
        detector=window.detector,
        platform=window.platform,
        sample=window.sample,
        label=str(verdict.label),
        metric=str(getattr(verdict, "metric", "")),
        sub_detector=getattr(verdict, "sub_detector", None),
        base_xml_path=base_xml,
        onset_xml_path=onset_xml,
        base_configured_xml_path=base_configured_xml,
        onset_configured_xml_path=onset_configured_xml,
        sweep_option=option,
        sweep_value=value or None,
        pct_change=_pct(getattr(verdict, "pct_change", None)),
        base_release=window.base_release,
        onset_release=window.onset_release,
        base_stack_setup=str(base_info.get("k4h_stack_setup") or ""),
        onset_stack_setup=str(onset_info.get("k4h_stack_setup") or ""),
        base_run_id=window.base_run,
        onset_run_id=window.onset_run,
        base_actions_url=window.base_url,
        onset_actions_url=window.onset_url,
        base_commit=base_commit,
        onset_commit=onset_commit,
        base_n_events=base_events,
        onset_n_events=onset_events,
        base_ddsim_args=base_args,
        onset_ddsim_args=onset_args,
        base_verbose=base_verbose,
        onset_verbose=onset_verbose,
        base_cpu_set=base_cpu_set,
        onset_cpu_set=onset_cpu_set,
        base_seed=base_seed,
        onset_seed=onset_seed,
        base_input_files=base_files,
        onset_input_files=onset_files,
        base_steering_file=base_steering,
        onset_steering_file=onset_steering,
        base_resolved_steering_file=base_resolved_steering,
        onset_resolved_steering_file=onset_resolved_steering,
        parity_diffs=tuple(parity),
    )


def parity_note(facts: ReproducerFacts) -> str:
    """Whether the two runs simulated the same workload, as the recipe says it."""
    if facts.parity_diffs:
        return _differed(facts.parity_diffs)
    if facts.base_seed is None:
        return _note(
            "WARNING: neither run recorded a fixed random seed, so their exact "
            "generated workloads cannot be reproduced or shown to be identical."
        )
    return _note(
        f"Both nightly runs used k4Bench {facts.base_commit[:12]} and the "
        f"same workload: {facts.base_n_events} events, seed "
        f"{facts.base_seed}."
    )


def _xml_arg(path: str) -> str:
    if path.startswith("/"):
        return _safe(path)
    return f'"$K4GEO"/{_safe(path)}'


def _local_input(source: str) -> str:
    """Where a fetched input lands. Derived from the source's own basename,
    because the recorded ``--ddsim-args`` reads it under exactly that name."""
    return "/tmp/" + source.rsplit("/", 1)[-1]


def _fetch(input_files: tuple[str, ...]) -> list[str]:
    return [
        f"xrdcp --force {_safe(source)} {_safe(_local_input(source))}"
        for source in input_files
    ]


def _shared_input(facts: ReproducerFacts) -> bool:
    """Whether both runs read the same sources, which are then fetched once.

    One fetch when both runs read the same sources, one per half when they do
    not: differing inputs are a real workload difference (recorded in
    ``parity_diffs``), and each half has to run against the sources its own
    nightly used. Each lands under the basename its recorded ``--ddsim-args``
    reads, so a shared name is re-fetched rather than renamed — the halves are
    sequential, and the second run's copy replaces one the first has already
    consumed.
    """
    return facts.base_input_files == facts.onset_input_files


def fetch_stage(facts: ReproducerFacts) -> tuple[str, str] | None:
    """The titled fetch both halves read from, when they read the same sources."""
    if not _shared_input(facts) or not facts.base_input_files:
        return None
    return (
        "input (fetched once; both runs read the same sources)",
        _fetch_stage(
            facts.base_release, facts.base_stack_setup, _fetch(facts.base_input_files)
        ),
    )


def _command(
    facts: ReproducerFacts,
    *,
    release: str,
    stack_setup: str,
    n_events: int,
    ddsim_args: str,
    xml_path: str,
    resolved_steering_file: str,
    verbose: bool,
    output: str,
    worktree: str,
    fetch: list[str],
) -> str:
    """One half of the recipe (:func:`~k4bench.blame.reproduce.shell._half_setup`)."""
    lines = _half_setup(worktree, release, stack_setup)
    if resolved_steering_file:
        directory = resolved_steering_file.rpartition("/")[0] or "."
        lines.append(f"export PYTHONPATH={_safe(directory)}:\"${{PYTHONPATH:-}}\"")
    lines.extend(fetch)
    command = [f"k4bench --xml {_xml_arg(xml_path)}", f"--events {_safe(n_events)}"]
    if verbose:
        command.append("--verbose")
    if facts.sweep_option and facts.sweep_value:
        command.append(f"{facts.sweep_option} {_safe(facts.sweep_value)}")
    command.append(f"--output-dir {_safe(output)}")
    command.append(f"--ddsim-args={_safe(ddsim_args)}")
    lines.extend(_continued(command))
    return _subshell(lines)


def halves(
    facts: ReproducerFacts,
    *,
    outputs: tuple[str, str],
    worktrees: tuple[str, str],
) -> tuple[str, str]:
    """The BEFORE and AFTER halves, writing into *outputs* from *worktrees*."""
    shared = _shared_input(facts)
    before = _command(
        facts,
        release=facts.base_release,
        stack_setup=facts.base_stack_setup,
        n_events=facts.base_n_events,
        ddsim_args=facts.base_ddsim_args,
        xml_path=facts.base_xml_path,
        resolved_steering_file=facts.base_resolved_steering_file,
        verbose=facts.base_verbose,
        output=outputs[0],
        worktree=worktrees[0],
        fetch=[] if shared else _fetch(facts.base_input_files),
    )
    after = _command(
        facts,
        release=facts.onset_release,
        stack_setup=facts.onset_stack_setup,
        n_events=facts.onset_n_events,
        ddsim_args=facts.onset_ddsim_args,
        xml_path=facts.onset_xml_path,
        resolved_steering_file=facts.onset_resolved_steering_file,
        verbose=facts.onset_verbose,
        output=outputs[1],
        worktree=worktrees[1],
        fetch=[] if shared else _fetch(facts.onset_input_files),
    )
    return before, after
