"""Which platforms replace which.

Regression history is scoped to ``(detector, platform, sample)``: results are
filed per platform, and a platform is an identity, not a label for "the current
one". Two platforms benchmarked side by side are therefore two independent
series, and need nothing from this module.

A platform that *replaces* another names it here. Its series then continue the
predecessor's: the predecessor's nights come first in the history the engine
walks, as ordinary baseline points, so a shift caused by the migration is judged
like any other step — watch, then regression, with a window bounded by the
predecessor's last night — and the baseline re-anchors onto the new level. The
successor keeps its own directory, metadata, provenance and report.

A replaced platform is not expected to run once its successor has: its absence
from a report is silence rather than a missing-run failure, and it sorts after
the platforms still running.

Succession is one hop and chains are not allowed: a successor continues its
predecessor's own runs only, so a platform that replaced another and is itself
replaced would hand its successor a history cut short at its own first night.
When the next platform replaces the current one, remove the current one's own
entry before adding the new one; a later replay of the current platform's
first nights then judges them cold.
"""

from __future__ import annotations

#: ``{platform: the platform it replaced}``. Retain entries for historical
#: replay: a successor's early verdicts were judged against its predecessor's
#: nights.
PLATFORM_SUCCESSORS: dict[str, str] = {
    # The nightly moved from the Key4hep Spack stack to the LCG devkey-head
    # views: same machines, same benchmarks, new compiler (gcc 14.2 → 16).
    "x86_64-el9-gcc16-opt": "x86_64-almalinux9-gcc14.2.0-opt",
}


def predecessor_of(platform: str) -> str | None:
    """The platform *platform* replaced, or ``None``."""
    predecessor = PLATFORM_SUCCESSORS.get(platform)
    return None if predecessor == platform else predecessor


def successors_of(platform: str) -> tuple[str, ...]:
    """The platforms that replaced *platform*, name-ascending."""
    return tuple(sorted(
        successor for successor, predecessor in PLATFORM_SUCCESSORS.items()
        if predecessor == platform and successor != platform
    ))


def is_replaced(platform: str) -> bool:
    """Whether some platform has replaced *platform*."""
    return bool(successors_of(platform))
