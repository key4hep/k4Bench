"""From two run records to the recipe file one comment row links."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from k4bench.blame.reproduce import ddsim, k4run
from k4bench.blame.reproduce.ddsim import ReproducerFacts
from k4bench.blame.reproduce.k4run import K4runReproducerFacts
from k4bench.blame.reproduce.shell import _note, _rule, _safe

#: The rendered recipe is an uploaded artifact rather than comment body, so the
#: cap is only there to bound what a malformed run record can produce.
_MAX_BYTES = 65536

#: Either recipe's facts: a ddsim run's, or a k4run job's.
RecipeFacts = ReproducerFacts | K4runReproducerFacts


def facts_from(
    row: Any,
    base_info: dict[str, Any] | None,
    onset_info: dict[str, Any] | None,
    *,
    input_files: object = None,
) -> RecipeFacts | None:
    """Return frozen reproducer facts for *row*, or ``None`` if unsafe.

    Two ddsim runs give :class:`ReproducerFacts`, two runs of a k4run job
    :class:`K4runReproducerFacts`; a window whose runs ran different tools has
    no one command to give. Workload differences are not failures: they are
    recorded in ``parity_diffs`` and called out by :func:`render_text`.
    Identity/release mismatches, missing command essentials, and labels that
    cannot be inverted are irreconcilable and suppress only the reproducer.
    """
    if not isinstance(base_info, dict) or not isinstance(onset_info, dict):
        return None
    verdict = getattr(row, "verdict", row)
    tools = {info.get("tool", "ddsim") for info in (base_info, onset_info)}
    if tools == {"ddsim"}:
        return ddsim.facts(verdict, base_info, onset_info, input_files=input_files)
    if tools == {"k4run"}:
        return k4run.facts(verdict, base_info, onset_info)
    return None


def artifact_name(facts: RecipeFacts) -> str:
    """The published recipe's file name — stable for one measurement in one
    change window, and unique across every other.

    Readable, because it is the last thing shown in a browser's address bar
    before someone reads a script they are about to run, and suffixed with a
    digest of the full identity (platform and sub-detector included) so two
    measurements that differ only in a part the readable stem drops can never
    land on one name. The two run ids are in that digest because the two
    releases alone do not identify a window: k4Bench supports two changes
    inside one release and tells those windows apart by their runs, and without
    the ids both would publish over one recipe. Nothing nightly is in it — the
    window's own endpoints do not move — so re-publishing the same window
    replaces the same file and the link a standing comment already carries keeps
    working rather than accumulating a copy a night.
    """
    identity = "|".join((
        facts.detector, facts.platform, facts.sample, facts.label,
        facts.metric, facts.sub_detector or "",
        facts.base_release, facts.onset_release,
        facts.base_run_id, facts.onset_run_id,
    ))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    stem = _slug("-".join(
        (facts.detector, facts.sample, facts.label, facts.metric)
    ))[:96].strip("-")
    return f"{stem}-{_slug(facts.base_release)}-{_slug(facts.onset_release)}-{digest}.txt"


def _slug(value: str) -> str:
    """*value* reduced to the characters a file name and a URL path both carry
    without quoting. Names here are detector, sample and metric identifiers
    from the report, so this is a bound on untrusted input, not a nicety."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unnamed"


def _pinning(facts: RecipeFacts) -> str:
    """What the runner's CPU pinning means for the reader, or ``""`` when
    neither run was pinned."""
    if facts.base_cpu_set and facts.base_cpu_set == facts.onset_cpu_set:
        return _note(
            f"Both nightly runs were pinned to CPUs {facts.base_cpu_set} "
            "(taskset -c) on the runner. The recipe does not pin: those cores "
            "are the runner's, not yours. For timings as quiet as the "
            "nightly's, run both halves pinned to one idle set of your own."
        )
    if facts.base_cpu_set or facts.onset_cpu_set:
        return _note(
            "WARNING: the two nightly runs were pinned to different CPU sets "
            f"({facts.base_cpu_set or 'unpinned'} vs "
            f"{facts.onset_cpu_set or 'unpinned'}), which moves a timing "
            "measurement on its own. Pin both halves to one idle set of your "
            "own and treat the nightly percentage as unconfirmed."
        )
    return ""


def render_text(facts: RecipeFacts) -> str:
    """The recipe as the standalone text file a comment links to, or ``""``
    above its byte cap.

    Plain text on purpose: it is read in a browser tab and pasted into a shell,
    so it carries no markup to strip and nothing that renders differently from
    what runs. Every word of it is a ``#`` comment, so the whole file is one
    paste rather than a page of prose the reader has to pick commands out of —
    it opens by naming the measurement, since a link out of a table is read with
    none of the surrounding comment's context, and closes with the two nightly
    runs it was derived from.

    The harness is cloned once, and each half runs in a subshell over its own
    worktree, sourcing its own Key4hep release
    (:func:`~k4bench.blame.reproduce.shell._half_setup`); what the halves run,
    and what is fetched before them, is the tool's (:mod:`.ddsim`,
    :mod:`.k4run`). Everything executable sits inside one outer ``set -e``
    subshell, so any stage failing — the clone, a fetch, either half — ends the
    whole reproduction with a non-zero status instead of letting a successful
    AFTER half report a run that has nothing to be compared against. It is a
    subshell rather than a top-level ``set -e`` for the same reason the halves
    are: nothing here may alter the shell of a reader who pasted it, errexit and
    working directory alike.
    """
    tool = k4run if isinstance(facts, K4runReproducerFacts) else ddsim
    heading = "k4Bench: reproduce this measurement"
    scope = " / ".join((facts.detector, facts.sample, facts.label))
    parity = tool.parity_note(facts)
    pinning = _pinning(facts)
    fetch = tool.fetch_stage(facts)
    # Named for the runs, never for the releases: a window can begin and end
    # inside one release, and two directories named after it would be one
    # directory, with the AFTER half overwriting the results it is compared
    # against.
    before_dir, after_dir = (
        f"logs/before-{_slug(facts.base_run_id)}",
        f"logs/after-{_slug(facts.onset_run_id)}",
    )
    before_tree, after_tree = "../k4Bench-before", "../k4Bench-after"
    before, after = tool.halves(
        facts,
        outputs=(before_dir, after_dir),
        worktrees=(before_tree, after_tree),
    )
    check = max(1, min(100, facts.onset_n_events // 10))
    metric = facts.metric + (
        f" ({facts.sub_detector})" if facts.sub_detector else ""
    )
    text = "\n".join((
        "#!/usr/bin/env bash",
        f"# {heading}",
        "# " + "=" * len(heading),
        "#",
        f"# {scope}",
        f"# Metric:            {metric}",
        f"# Platform:          {facts.platform}",
        f"# Change window:     {facts.base_release} -> {facts.onset_release} "
        "(Key4hep releases)",
        f"# Nightly measured:  {facts.pct_change}",
        "#",
        parity,
        *([pinning, "#"] if pinning else []),
        _note(
            f"The nightly measured this on {facts.platform}, inside the "
            "Key4hep container; the recipe runs on your host. Compare its two "
            "halves against each other, not against the nightly's absolute "
            "numbers."
        ),
        "#",
        _note(
            "Run it with `bash <this file>`, or paste the whole thing into one "
            "shell. Everything runs inside subshells: the two Key4hep releases "
            "never share an environment, any failed stage ends the whole "
            "reproduction, and neither your shell nor your working directory "
            "is touched."
        ),
        "",
        # The one place errexit can cover the stages between the halves — the
        # clone, the worktrees, a shared fetch — as well as the halves.
        "(",
        "set -e",
        "",
        _rule("harness (one clone, one worktree per half)"),
        "git clone https://github.com/key4hep/k4Bench",
        "cd k4Bench",
        # Detached, and per half: setup.sh reuses an existing py-venv and
        # plugin/build.sh an up-to-date .so, so sharing one working copy would
        # run the second release against the first release's build.
        f"git worktree add --detach {before_tree} {_safe(facts.base_commit)}",
        f"git worktree add --detach {after_tree} {_safe(facts.onset_commit)}",
        *(["", _rule(fetch[0]), fetch[1]] if fetch else []),
        "",
        _rule(f"BEFORE: Key4hep {facts.base_release}"),
        before,
        "",
        _rule(f"AFTER: Key4hep {facts.onset_release}"),
        after,
        ")",
        "",
        _note(
            f"Then compare {facts.metric} for {facts.label} between "
            f"k4Bench-before/{before_dir} and k4Bench-after/{after_dir}: the "
            f"nightly measured {facts.pct_change}."
        ),
        _note(
            f"Quick directional check: re-run with --events {check}. It "
            "reproduces the direction, not the percentage."
        ),
        "#",
        _note(
            "Key4hep nightlies are kept on CVMFS for about three weeks; past "
            "that this window cannot be re-run."
        ),
        "#",
        "# Nightly runs this recipe was derived from:",
        f"#   {facts.base_run_id}  {facts.base_actions_url}",
        f"#   {facts.onset_run_id}  {facts.onset_actions_url}",
        "",
    ))
    return text if len(text.encode("utf-8")) <= _MAX_BYTES else ""
