"""The vocabulary both blame prompts are written in.

:mod:`k4bench.blame.rank` asks "which of these pull requests caused this
configuration's regressions"; :mod:`k4bench.blame.attribute` asks "which of this
window's regressions did this one pull request cause". Different questions, but
they describe the same world to the model — a platform, a physics sample, the
direction a metric moved, a bounded diff — and they must describe it *identically*
or the second pass will read the first pass's priors against a different
vocabulary than it was given them in.

Nothing here talks to a model or knows a request shape; it turns k4Bench's own
identifiers into the phrasing a model can act on.
"""

from __future__ import annotations

import math
import textwrap

from k4bench.labels import describe_platform, pretty_sample


def direction_phrase(direction: str, pct_change: float | None) -> str:
    """``"up +20.0%"`` / ``"down -5.0%"`` / ``"changed"`` — the mechanical sign
    of a step, never a good/bad judgement (the report's own convention)."""
    word = {"UP": "up", "DOWN": "down"}.get((direction or "").upper(), "changed")
    if pct_change is not None and math.isfinite(pct_change):
        return f"{word} {pct_change * 100:+.1f}%"
    return word


def sample_line(sample: str, *, prefix: str = "- Sample: ") -> str:
    """``"- Sample: p8_ee_Zbb_ecm91 — Pythia8: e⁺e⁻ → Z → bb (91 GeV)"``.

    The raw directory name is the identity the rest of the report uses; the
    readable form tells the model what physics is actually being simulated,
    which is what decides whether a diff can plausibly touch it."""
    pretty = pretty_sample(sample)
    return f"{prefix}{sample}" + (f" — {pretty}" if pretty != sample else "")


def platform_line(platform: str, *, prefix: str = "- Platform: ") -> str:
    """``"- Platform: <slug> — x86_64 · AlmaLinux 9 · GCC 14.2.0 (optimized)"``.

    The slug alone under-reads: a codegen- or build-flag-sensitive change lands
    differently under ``opt`` than ``dbg`` and across compiler versions, so the
    architecture, OS, compiler and build type are spelled out — all four from
    :func:`~k4bench.labels.describe_platform`, which owns the layout. An
    unrecognized platform degrades to the raw slug."""
    label = describe_platform(platform)
    if label is None:
        return f"{prefix}{platform}"
    return (
        f"{prefix}{platform} — {label.architecture} · {label.os} · "
        f"{label.compiler} ({label.build_type})"
    )


def format_files(files: tuple[str, ...], limit: int) -> str:
    """A PR's changed paths, capped, with the overflow counted rather than hidden."""
    shown = list(files[:limit])
    suffix = f", … (+{len(files) - len(shown)} more)" if len(files) > len(shown) else ""
    return ", ".join(shown) + suffix


def allocate_diff_budget(needs: list[int], total: int) -> list[int]:
    """Chars of diff each item may render, waterfilled from *total*.

    When everything fits, everyone gets their full patch. Under pressure the
    budget is shared evenly: small diffs stay whole and the largest ones split
    the remainder — an item's *position* in the prompt never decides whether its
    diff survives."""
    alloc = [0] * len(needs)
    remaining = total
    active = [i for i, n in enumerate(needs) if n > 0]
    while active and remaining >= len(active):
        share = remaining // len(active)
        satisfied = []
        for i in active:
            take = min(needs[i] - alloc[i], share)
            alloc[i] += take
            remaining -= take
            if alloc[i] >= needs[i]:
                satisfied.append(i)
        if not satisfied:
            break  # everyone consumed a full share; nothing left to rebalance
        active = [i for i in active if i not in satisfied]
    return alloc


def diff_block(patch: str, budget: int, *, indent: str = "    ") -> list[str]:
    """A patch as indented prompt lines, clipped to *budget*, or nothing at all.

    Truncation is marked rather than silent: a model that cannot see the end of
    a hunk should know that is why."""
    if not patch or budget <= 0:
        return []
    if len(patch) > budget:
        patch = patch[:budget] + "\n… (truncated)"
    return ["  diff:", textwrap.indent(patch, indent)]
