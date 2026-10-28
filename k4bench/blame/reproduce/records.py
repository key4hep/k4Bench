"""The two run records a recipe is built from, read and checked.

A recipe reruns the nightly runs on either side of a change window, so it is
built only when both ``run_info.json`` records are the runs the verdict names
(:func:`_window`). Fields are read tolerantly: a malformed one reads as
missing, which costs the recipe and never the comment.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from typing import Any

from k4bench.labels import pretty_release


def _release(info: dict[str, Any]) -> str:
    value = info.get("k4h_release_date") or info.get("k4h_release") or ""
    return pretty_release(str(value))


def _files(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            return tuple(shlex.split(value))
        except ValueError:
            return ()
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(value)
    return ()


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _pct(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "-"
    return f"{value:+.1%}"


def _tokens(args: str) -> tuple[str, ...] | None:
    try:
        return tuple(shlex.split(args))
    except ValueError:
        return None


@dataclass(frozen=True)
class _Window:
    """What both tools' recipes need of the two runs, once checked."""

    detector: str
    platform: str
    sample: str
    base_release: str
    onset_release: str
    base_run: str
    onset_run: str
    base_url: str
    onset_url: str
    base_commit: str
    onset_commit: str
    base_events: int
    onset_events: int


def _window(
    verdict: Any, base_info: dict[str, Any], onset_info: dict[str, Any]
) -> _Window | None:
    """The two runs of *verdict*'s window, or ``None`` when either record is
    not that run, or lacks what every recipe is built from."""
    expected = {
        "detector": str(getattr(verdict, "detector", "")),
        "platform": str(getattr(verdict, "platform", "")),
        "sample": str(getattr(verdict, "sample", "")),
    }
    # A window opened on a replaced platform has its base (or onset) run there.
    platforms = (
        str(getattr(verdict, "last_accepted_platform", None) or expected["platform"]),
        str(getattr(verdict, "onset_platform", None) or expected["platform"]),
    )
    for info, platform in zip((base_info, onset_info), platforms):
        wanted = {**expected, "platform": platform}
        if any(str(info.get(key, "")) != value for key, value in wanted.items()):
            return None

    base_release = _release(base_info)
    onset_release = _release(onset_info)
    wanted_base = str(getattr(verdict, "last_accepted_run_date", "") or "")
    wanted_onset = str(getattr(verdict, "onset_run_date", "") or "")
    if base_release != wanted_base or onset_release != wanted_onset:
        return None

    base_events = _integer(base_info.get("n_events"))
    onset_events = _integer(onset_info.get("n_events"))
    base_commit = str(base_info.get("commit_sha") or "")
    onset_commit = str(onset_info.get("commit_sha") or "")
    base_url = str(base_info.get("github_run_url") or "")
    onset_url = str(onset_info.get("github_run_url") or "")
    base_run = str(getattr(verdict, "last_accepted_run_id", "") or "")
    onset_run = str(getattr(verdict, "onset_run_id", "") or "")
    if base_events is None or onset_events is None or not all(
        (base_commit, onset_commit, base_url, onset_url, base_run, onset_run)
    ):
        return None
    return _Window(
        **expected,
        base_release=base_release,
        onset_release=onset_release,
        base_run=base_run,
        onset_run=onset_run,
        base_url=base_url,
        onset_url=onset_url,
        base_commit=base_commit,
        onset_commit=onset_commit,
        base_events=base_events,
        onset_events=onset_events,
    )
