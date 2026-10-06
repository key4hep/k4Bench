"""Which platforms and detector configs replace which.

Regression history is scoped to ``(detector, platform, sample)``: results are
filed per detector config and platform, and each is an identity, not a label for
"the current one". Two platforms — or two geometry versions — benchmarked side
by side are therefore independent series, and need nothing from this module.

A platform that *replaces* another names it here, and so does a detector config
that replaces another (a geometry version bump). The successor's series then
continue the predecessor's: the predecessor's nights come first in the history
the engine walks, as ordinary baseline points, so a shift caused by the switch
is judged like any other step — watch, then regression, with a window bounded
by the predecessor's last night — and the baseline re-anchors onto the new
level. The successor keeps its own directory, metadata, provenance and report.

A replaced platform or detector is not expected to run once its successor has:
its absence from a report is silence rather than a missing-run failure, and it
sorts after the ones still running.

Succession is one hop and chains are not allowed: a successor continues its
predecessor's own runs only, so one that replaced another and is itself replaced
would hand its successor a history cut short at its own first night. When the
next platform (or version) replaces the current one, remove the current one's
own entry before adding the new one; a later replay of the current one's first
nights then judges them cold. A detector config whose entry is removed that way
moves to :data:`RETIRED_DETECTORS` with the config that replaced it, so it stays
replaced — not expected to run once that successor, or any config that later
replaced it, has — without its history being continued.
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

#: ``{detector config: the detector config it replaced}``, on the same
#: platform. Retained for historical replay, like :data:`PLATFORM_SUCCESSORS`.
DETECTOR_SUCCESSORS: dict[str, str] = {
    "ALLEGRO_o1_v04": "ALLEGRO_o1_v03",
    "CLD_o2_v09":     "CLD_o2_v08",
    "IDEA_o1_v04":    "IDEA_o1_v03",
}

#: ``{detector config: the detector config that replaced it}`` for configs
#: whose successor was itself replaced. Their :data:`DETECTOR_SUCCESSORS` entry
#: is gone to keep succession one hop, but they are still replaced; this map
#: answers only that, never whose history a series continues.
RETIRED_DETECTORS: dict[str, str] = {}


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


def detector_predecessor_of(detector: str) -> str | None:
    """The detector config *detector* replaced, or ``None``."""
    predecessor = DETECTOR_SUCCESSORS.get(detector)
    return None if predecessor == detector else predecessor


def _direct_detector_successors(detector: str) -> set[str]:
    return {
        *(
            successor for successor, predecessor in DETECTOR_SUCCESSORS.items()
            if predecessor == detector
        ),
        *(
            successor for retired, successor in RETIRED_DETECTORS.items()
            if retired == detector
        ),
    } - {detector}


def detector_successors_of(detector: str) -> tuple[str, ...]:
    """The detector configs that replaced *detector*, name-ascending: the one
    that replaced it, and whatever replaced that one in turn. A retired config
    is thus no longer expected once any later version has run, even one whose
    direct successor never produced a run."""
    found: set[str] = set()
    frontier = [detector]
    while frontier:
        for successor in _direct_detector_successors(frontier.pop()):
            if successor != detector and successor not in found:
                found.add(successor)
                frontier.append(successor)
    return tuple(sorted(found))


def is_detector_replaced(detector: str) -> bool:
    """Whether some detector config has replaced *detector*."""
    return bool(detector_successors_of(detector))


def predecessor_series(detector: str, platform: str) -> tuple[tuple[str, str], ...]:
    """The ``(detector, platform)`` series the *detector* / *platform* series
    may continue, nearest first: the detector config it replaced on this
    platform, then this detector config on the platform this one replaced.

    A series continues the first candidate that has runs. Both can exist at
    once — a version bump made after a platform migration, say — and then
    only one of them normally ran: the new version never ran on the platform
    retired before it existed.
    """
    candidates = []
    if (previous := detector_predecessor_of(detector)) is not None:
        candidates.append((previous, platform))
    if (previous := predecessor_of(platform)) is not None:
        candidates.append((detector, previous))
    return tuple(candidates)


def successor_series(detector: str, platform: str) -> tuple[tuple[str, str], ...]:
    """The ``(detector, platform)`` series whose runs mean the *detector* /
    *platform* series is no longer expected: the inverse of
    :func:`predecessor_series`, plus later detector configs that replaced a
    retired one."""
    return (
        *((successor, platform) for successor in detector_successors_of(detector)),
        *((detector, successor) for successor in successors_of(platform)),
    )
