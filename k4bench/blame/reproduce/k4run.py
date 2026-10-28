"""The recipe of a k4run job: ``k4bench k4run`` on the job's frozen input.

A k4run benchmark is the ``baseline`` job plus one run per variant. The recipe
reruns the baseline and, for a variant's row, that one variant, each half with
the arguments its nightly run recorded. Those already name the local copies of
the job's inputs and data files, so the recipe fetches every source to the path
the nightly put it at (:mod:`k4bench.staging`) and passes the arguments on as
they are.

The untimed memory pass behind ``<label>_heap.json`` is not rerun: no judged
metric comes from it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from k4bench.blame.reproduce.records import _files, _pct, _tokens, _window
from k4bench.blame.reproduce.shell import (
    _continued,
    _differed,
    _fetch_stage,
    _half_setup,
    _note,
    _safe,
    _subshell,
)
from k4bench.labels import BASELINE_LABEL, VARIANT_PREFIX
from k4bench.staging import (
    ARCHIVE_SUFFIXES,
    LOCAL_DATA_DIR,
    LOCAL_INPUT_DIR,
    REMOTE_PREFIXES,
    data_key,
    download_argv,
    input_key,
)


@dataclass(frozen=True)
class K4runReproducerFacts:
    """The facts a k4run recipe is rendered from; *variant* is empty for the
    baseline."""

    detector: str
    platform: str
    sample: str
    label: str
    metric: str
    sub_detector: str | None
    variant: str
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
    base_options: tuple[str, ...]
    onset_options: tuple[str, ...]
    base_stage_dir: str
    onset_stage_dir: str
    base_k4run_args: str
    onset_k4run_args: str
    base_variant_args: str
    onset_variant_args: str
    base_verbose: bool
    onset_verbose: bool
    base_cpu_set: str
    onset_cpu_set: str
    base_input_files: tuple[str, ...]
    onset_input_files: tuple[str, ...]
    base_data_files: tuple[str, ...]
    onset_data_files: tuple[str, ...]
    parity_diffs: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        """Canonical, JSON-ready facts included in the comment digest."""
        return asdict(self)


def _variant_of(label: str) -> str | None:
    """The variant a k4run result label names: ``""`` for the baseline,
    ``<name>`` for ``variant_<name>``, ``None`` for any other label."""
    if label == BASELINE_LABEL:
        return ""
    name = label.removeprefix(VARIANT_PREFIX)
    return name if name != label and name else None


def _workload(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """*tokens* with every absolute path reduced to its file name.

    Where a path lies says which stack or which local copy a run read, both
    compared from their source metadata (geometry, stage directory, inputs and
    data files); left whole, the two runs' arguments would differ on every
    window, since each half reads its own release's files."""
    def reduced(token: str) -> str:
        if token.startswith("/"):
            return PurePosixPath(token).name
        option, equals, value = token.partition("=")
        if equals and option.startswith("-") and value.startswith("/"):
            return f"{option}={PurePosixPath(value).name}"
        return token

    return tuple(reduced(token) for token in tokens)


def facts(
    verdict: Any, base_info: dict[str, Any], onset_info: dict[str, Any]
) -> K4runReproducerFacts | None:
    """:func:`~k4bench.blame.reproduce.recipe.facts_from` for two runs of one
    k4run job."""
    variant = _variant_of(str(getattr(verdict, "label", "")))
    if variant is None:
        return None
    window = _window(verdict, base_info, onset_info)
    if window is None:
        return None

    sides = []
    for info in (base_info, onset_info):
        options = _files(info.get("options"))
        args = str(info.get("k4run_args") or "")
        variants = info.get("variants")
        variants = variants if isinstance(variants, dict) else {}
        variant_args = str(variants.get(variant) or "") if variant else ""
        tokens = _tokens(args)
        variant_tokens = _tokens(variant_args)
        if not options or tokens is None or variant_tokens is None:
            return None
        if variant and variant not in variants:
            return None
        inputs = _files(info.get("input_files"))
        data = _files(info.get("data_files"))
        # Every source is fetched to where the nightly put it; a path on the
        # runner is not there for the recipe to fetch.
        if not all(source.startswith(REMOTE_PREFIXES) for source in (*inputs, *data)):
            return None
        stage_dir = str(info.get("resolved_stage_dir") or "")
        if info.get("stage_dir") and not stage_dir:
            return None
        sides.append({
            "options": options,
            "stage_dir": stage_dir,
            "k4run_args": args,
            "variant_args": variant_args,
            "inputs": inputs,
            "data": data,
            "verbose": bool(info.get("verbose")),
            "cpu_set": str(info.get("runner_cpu_set") or ""),
            "workload": _workload(tokens),
            "variant_workload": _workload(variant_tokens),
        })
    base, onset = sides

    parity = []
    for name, before, after in (
        ("n_events", window.base_events, window.onset_events),
        ("options", base["options"], onset["options"]),
        ("stage_dir", base_info.get("stage_dir") or "", onset_info.get("stage_dir") or ""),
        ("k4run_args", base["workload"], onset["workload"]),
        ("variant_args", base["variant_workload"], onset["variant_workload"]),
        ("commit_sha", window.base_commit, window.onset_commit),
        ("input_files", base["inputs"], onset["inputs"]),
        ("data_files", base["data"], onset["data"]),
        ("verbose", base["verbose"], onset["verbose"]),
    ):
        if before != after:
            parity.append(name)
    base_xml = str(base_info.get("configured_xml_path") or "")
    onset_xml = str(onset_info.get("configured_xml_path") or "")
    if (base_xml or onset_xml) and base_xml != onset_xml:
        parity.append("xml_path")

    return K4runReproducerFacts(
        detector=window.detector,
        platform=window.platform,
        sample=window.sample,
        label=str(verdict.label),
        metric=str(getattr(verdict, "metric", "")),
        sub_detector=getattr(verdict, "sub_detector", None),
        variant=variant,
        pct_change=_pct(getattr(verdict, "pct_change", None)),
        base_release=window.base_release,
        onset_release=window.onset_release,
        base_stack_setup=str(base_info.get("k4h_stack_setup") or ""),
        onset_stack_setup=str(onset_info.get("k4h_stack_setup") or ""),
        base_run_id=window.base_run,
        onset_run_id=window.onset_run,
        base_actions_url=window.base_url,
        onset_actions_url=window.onset_url,
        base_commit=window.base_commit,
        onset_commit=window.onset_commit,
        base_n_events=window.base_events,
        onset_n_events=window.onset_events,
        base_options=base["options"],
        onset_options=onset["options"],
        base_stage_dir=base["stage_dir"],
        onset_stage_dir=onset["stage_dir"],
        base_k4run_args=base["k4run_args"],
        onset_k4run_args=onset["k4run_args"],
        base_variant_args=base["variant_args"],
        onset_variant_args=onset["variant_args"],
        base_verbose=base["verbose"],
        onset_verbose=onset["verbose"],
        base_cpu_set=base["cpu_set"],
        onset_cpu_set=onset["cpu_set"],
        base_input_files=base["inputs"],
        onset_input_files=onset["inputs"],
        base_data_files=base["data"],
        onset_data_files=onset["data"],
        parity_diffs=tuple(parity),
    )


def parity_note(facts: K4runReproducerFacts) -> str:
    """Whether the two runs reconstructed the same workload, as the recipe
    says it."""
    if facts.parity_diffs:
        return _differed(facts.parity_diffs)
    source = " of the same frozen input" if facts.base_input_files else ""
    return _note(
        f"Both nightly runs used k4Bench {facts.base_commit[:12]} and the "
        f"same workload: {facts.base_n_events} events{source}."
    )


def _download(source: str, local: str) -> str:
    return " ".join(_safe(word) for word in download_argv(source, local))


def _name(source: str) -> str:
    return source.rsplit("/", 1)[-1]


def fetch_stage(facts: K4runReproducerFacts) -> tuple[str, str] | None:
    """The titled fetch of every input and data file either run read.

    One fetch serves both halves even when their sources differ: every source
    has a directory of its own, named for it, so one half's copy never replaces
    the other's, and each half's recorded arguments find exactly the copy its
    nightly read.
    """
    lines = []
    for source in dict.fromkeys((*facts.base_input_files, *facts.onset_input_files)):
        directory = LOCAL_INPUT_DIR / input_key(source)
        lines.append(f"mkdir -p {_safe(directory)}")
        lines.append(_download(source, str(directory / _name(source))))
    for data in dict.fromkeys((facts.base_data_files, facts.onset_data_files)):
        if not data:
            continue
        directory = LOCAL_DATA_DIR / data_key(list(data))
        lines.append(f"mkdir -p {_safe(directory)}")
        for source in data:
            local = str(directory / _name(source))
            lines.append(_download(source, local))
            # Unpacked where it lands, and not kept, as the nightly does.
            if local.endswith(ARCHIVE_SUFFIXES):
                lines.append(f"tar -xzf {_safe(local)} -C {_safe(directory)}")
                lines.append(f"rm -f {_safe(local)}")
    if not lines:
        return None
    return (
        "inputs (fetched to where each nightly run read them)",
        _fetch_stage(facts.base_release, facts.base_stack_setup, lines),
    )


def _command(
    facts: K4runReproducerFacts,
    *,
    release: str,
    stack_setup: str,
    options: tuple[str, ...],
    stage_dir: str,
    n_events: int,
    k4run_args: str,
    variant_args: str,
    verbose: bool,
    output: str,
    worktree: str,
) -> str:
    """One half of the recipe (:func:`~k4bench.blame.reproduce.shell._half_setup`)."""
    command = ["k4bench k4run " + " ".join(_safe(option) for option in options)]
    if stage_dir:
        command.append(f"--stage-dir {_safe(stage_dir)}")
    command.append(f"--events {_safe(n_events)}")
    if verbose:
        command.append("--verbose")
    if facts.variant:
        command.append(f"--variant {_safe(f'{facts.variant}={variant_args}')}")
    command.append(f"--output-dir {_safe(output)}")
    if k4run_args:
        command.append(f"--k4run-args={_safe(k4run_args)}")
    return _subshell([*_half_setup(worktree, release, stack_setup), *_continued(command)])


def halves(
    facts: K4runReproducerFacts,
    *,
    outputs: tuple[str, str],
    worktrees: tuple[str, str],
) -> tuple[str, str]:
    """The BEFORE and AFTER halves, writing into *outputs* from *worktrees*."""
    before = _command(
        facts,
        release=facts.base_release,
        stack_setup=facts.base_stack_setup,
        options=facts.base_options,
        stage_dir=facts.base_stage_dir,
        n_events=facts.base_n_events,
        k4run_args=facts.base_k4run_args,
        variant_args=facts.base_variant_args,
        verbose=facts.base_verbose,
        output=outputs[0],
        worktree=worktrees[0],
    )
    after = _command(
        facts,
        release=facts.onset_release,
        stack_setup=facts.onset_stack_setup,
        options=facts.onset_options,
        stage_dir=facts.onset_stage_dir,
        n_events=facts.onset_n_events,
        k4run_args=facts.onset_k4run_args,
        variant_args=facts.onset_variant_args,
        verbose=facts.onset_verbose,
        output=outputs[1],
        worktree=worktrees[1],
    )
    return before, after
