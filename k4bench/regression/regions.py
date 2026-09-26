"""Where inside the detector a timing step landed.

A confirmed run-level regression is one number, and one number names no
mechanism: "ALLEGRO got 21% slower" and "the HCAL barrel got fourteen times
slower while everything else stood still" are the same measurement, but only the
second can be matched against a diff. The ``k4BenchRegionTimingAction`` plugin
records per-event time per top-level detector region on every run, so the second
form is already measured — it has simply never been read by anything that
attributes a regression.

This module reads it: for one benchmark configuration and one change window, how
each region's per-event time differs between the two ends the change entered
between — the two releases, or the window's two runs when one release holds
both. It judges nothing (the engine has already decided *that* the metric
stepped) and it introduces no thresholds of its own; it reports the
decomposition, largest movement first, and leaves the reading to whoever asked.

Two costs shape the implementation. Region files are per *configuration* and hold
per-event arrays, so loading a whole trend window across every label is
expensive — this loads exactly the two ends of one window, for one label, and
only when something actually regressed there. And a release that recorded no
region file is *absent*, never zero: a region that appears on one side of a
window only is a real event (a detector added, removed or renamed) and must stay
distinguishable from one that stood still.

The same files also carry each event's wall time and how much of it was spent
in stepping, which is the other half of the question: whether the step is in
the typical event or in a handful of long ones (:func:`region_evidence`). Both
readings come from one read of each file.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from k4bench.analysis.loader import load_region_timing
from k4bench.analysis.trend import EXPECTED_LOAD_ERRORS, parse_run_dir
from k4bench.regression.engine import release_key
from k4bench.regression.models import (
    EventProfile,
    EventSample,
    LongEvent,
    MatchedEvent,
    RegionDelta,
)

_log = logging.getLogger(__name__)

#: Regions carried on a verdict. The tail of a detector's region list is dozens
#: of near-zero entries; the largest few are where a step actually went, and the
#: rest would cost prompt space to say "nothing happened here" many times over.
MAX_REGIONS = 6

#: Longest events listed per window end. Enough to show whether one event or a
#: few carry the tail, and to see the same event numbers recur.
MAX_LONG_EVENTS = 3

#: Which attribution the decomposition uses. ``at_location`` charges time to the
#: region the Geant4 step physically occurred in, which is the question a
#: regression asks ("where is the time being spent now?"). ``by_birth`` answers a
#: different one — where the track came from — and mixing the two in one figure
#: would be a category error.
_ATTRIBUTION = "at_location"

#: Event 0 is the warm-up event every other consumer of this data drops (see
#: :func:`k4bench.analysis.trend.build_region_timing_trend`); it carries geometry
#: initialisation and would dominate a per-event median.
_WARMUP_EVENT = 0


@dataclass(frozen=True)
class _Night:
    """One night's region timing for one configuration, warm-up excluded.

    ``regions`` holds the per-event time charged to each region, indexed by
    event number; ``wall`` and ``stepping`` each event's wall time and the part
    of it spent in stepping, when the run recorded them."""

    regions: pd.DataFrame
    wall: pd.Series | None = None
    stepping: pd.Series | None = None


def _read_night(run_dir: Path, label: str) -> _Night | None:
    """One run's region timing for one configuration, or ``None``.

    ``None`` when this run recorded no region file for that configuration —
    which is the ordinary case for a run predating the plugin, and is why the
    caller must treat "no regions" as unknown rather than as zero."""
    if not (run_dir / f"{label}_regions.json").is_file():
        return None
    try:
        data = load_region_timing(run_dir, labels=[label])
    except EXPECTED_LOAD_ERRORS as exc:
        _log.debug("regions: unreadable region timing in %s — %s", run_dir, exc)
        return None
    except Exception:
        _log.exception("regions: failed reading region timing in %s", run_dir)
        return None

    reading = data.get(label) or {}
    frame = reading.get(_ATTRIBUTION)
    if frame is None or frame.empty:
        return None
    frame = frame[frame.index != _WARMUP_EVENT]
    if frame.empty:
        return None
    wall = stepping = None
    events = reading.get("events")
    if events is not None and not events.empty:
        events = events[events["event_number"] != _WARMUP_EVENT].set_index("event_number")
        if not events.empty:
            wall = events["event_wall_s"].astype(float)
            stepping = events["event_region_sum_s"].astype(float)
    return _Night(regions=frame, wall=wall, stepping=stepping)


def _region_medians(night: _Night) -> dict[str, float]:
    """Per-event median time per region for one night."""
    medians = {}
    for region in night.regions.columns:
        values = night.regions[region].dropna().to_numpy()
        if len(values):
            medians[str(region)] = float(np.median(values))
    return medians


def _read_nights(
    run_dirs: Sequence[Path],
    label: str,
    judgeable_configs: set[tuple[str, str]] | None = None,
) -> list[_Night]:
    """The readable nights of one window end, oldest first. Config-nights absent
    from *judgeable_configs* (failed or orphaned) are gaps and are skipped."""
    nights = []
    for run_dir in sorted(run_dirs, key=lambda d: d.name):
        if (
            judgeable_configs is not None
            and (str(run_dir.name), str(label)) not in judgeable_configs
        ):
            continue
        night = _read_night(run_dir, label)
        if night is not None:
            nights.append(night)
    return nights


def _release_medians(nights: Sequence[_Night]) -> dict[str, float]:
    """Per-region medians for one release, pooled across its nights.

    A release measured on several nights is several measurements of one software
    state, exactly as the engine treats it: the nights are combined into one
    figure per region (the median of their medians) rather than left to whichever
    night happened to be last."""
    per_night: dict[str, list[float]] = {}
    for night in nights:
        for region, median in _region_medians(night).items():
            per_night.setdefault(region, []).append(median)
    return {
        region: float(np.median(np.asarray(values)))
        for region, values in per_night.items()
        if values
    }


def _median(values: Sequence[float]) -> float | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(np.asarray(finite))) if finite else None


def _event_sample(nights: Sequence[_Night]) -> EventSample | None:
    """One window end's per-event wall times, summarised (see
    :class:`~k4bench.regression.models.EventSample`), or ``None`` when no night
    recorded them.

    Every figure is computed night by night and the nights are combined by
    their median. The longest events are ranked by their median time over the
    nights that simulated them, among the events most of the nights simulated,
    and each names the region with the largest median time in it over those
    nights — the same nights its own time is the median of."""
    usable = [n for n in nights if n.wall is not None and not n.wall.dropna().empty]
    if not usable:
        return None
    means, medians, stepping, without = [], [], [], []
    times: dict[int, list[float]] = {}
    for night in usable:
        wall = night.wall.dropna()
        means.append(float(wall.mean()))
        medians.append(float(wall.median()))
        if night.stepping is not None and not night.stepping.dropna().empty:
            stepping.append(float(night.stepping.dropna().mean()))
        if len(wall) > 1:
            without.append(float(wall.drop(wall.idxmax()).mean()))
        for event, seconds in wall.items():
            times.setdefault(int(event), []).append(float(seconds))
    ranked = sorted(
        (event for event in times if 2 * len(times[event]) > len(usable)),
        key=lambda event: (-float(np.median(times[event])), event),
    )[:MAX_LONG_EVENTS]
    longest = []
    for event in ranked:
        spent: dict[str, list[float]] = {}
        for night in usable:
            if event in night.wall.index and event in night.regions.index:
                for name, seconds in night.regions.loc[event].dropna().items():
                    spent.setdefault(str(name), []).append(float(seconds))
        typical = {name: float(np.median(values)) for name, values in spent.items()}
        region = max(sorted(typical), key=typical.get, default="")
        region_seconds = typical.get(region)
        if region_seconds is None or region_seconds <= 0:
            region, region_seconds = "", None
        longest.append(LongEvent(
            event=event, seconds=float(np.median(times[event])),
            region=region, region_seconds=region_seconds,
        ))
    return EventSample(
        nights=len(usable),
        n_events=int(np.median([len(n.wall.dropna()) for n in usable])),
        mean=_median(means),
        median=_median(medians),
        stepping_mean=_median(stepping),
        mean_without_longest=_median(without),
        longest=tuple(longest),
    )


def dirs_by_release(run_dirs: Sequence[str]) -> dict[str, list[Path]]:
    """Group run directories by the release they measured, keyed exactly as the
    engine keys releases (:func:`~k4bench.regression.engine.release_key`), so a
    window's ends match the verdict that named them."""
    grouped: dict[str, list[Path]] = {}
    for path in run_dirs:
        run_dir = Path(path)
        if not run_dir.is_dir():
            continue
        meta = parse_run_dir(run_dir)
        # `pd.NaT` is *truthy*, so an `or` here would keep the missing release
        # date instead of falling back to the run date — the same fallback
        # `x_date` makes for the frame the engine walked. Test for absence
        # explicitly, or a run predating release-date capture keys on something
        # the verdict's window never names, and its regions read as unmeasured.
        release_date = meta.get("k4h_release_date")
        if release_date is None or pd.isna(release_date):
            release_date = meta.get("run_date")
        key = release_key(release_date, run_dir.name)
        grouped.setdefault(key, []).append(run_dir)
    return grouped


def _run_dir_named(run_dirs: Sequence[Path], run_id: str | None) -> Path | None:
    """The one directory in *run_dirs* that is the run *run_id*, if it is there."""
    if not run_id:
        return None
    return next((d for d in run_dirs if d.name == str(run_id)), None)


def region_evidence(
    run_dirs: Sequence[str],
    *,
    label: str,
    base_release: str,
    onset_release: str,
    base_run_id: str | None = None,
    onset_run_id: str | None = None,
    limit: int = MAX_REGIONS,
    judgeable_configs: set[tuple[str, str]] | None = None,
) -> tuple[tuple[RegionDelta, ...], EventProfile | None]:
    """How each region's per-event time moved across ``(base, onset]``, largest
    movement first, and the per-event wall times at both ends
    (:class:`~k4bench.regression.models.EventProfile`).

    Returns ``((), None)`` when either end recorded no region timing at all —
    with only one side measured there is no comparison to make, and inventing
    one (treating the missing side as zero) would report every region of the
    detector as newly appearing. The profile is likewise ``None`` unless both
    ends recorded per-event wall times. When *judgeable_configs* is supplied,
    failed or orphaned config-nights absent from that set are gaps and do not
    enter either end.

    A window whose ends name one release is two runs of that release, and
    *base_run_id* / *onset_run_id* are what tell them apart — the same pair the
    verdict, the email's window token and the blame range are identified by. The
    ends are then those two runs rather than the release's pool, because one
    pool measured against itself is not a comparison and would report every
    region as having stood still. Without a resolvable run on each side there is
    again nothing to compare.
    """
    grouped = dirs_by_release(run_dirs)
    base_dirs, onset_dirs = grouped.get(base_release, []), grouped.get(onset_release, [])
    if not base_dirs or not onset_dirs:
        return (), None

    if base_release == onset_release:
        base_dir = _run_dir_named(base_dirs, base_run_id)
        onset_dir = _run_dir_named(onset_dirs, onset_run_id)
        if base_dir is None or onset_dir is None or base_dir == onset_dir:
            return (), None
        base_dirs, onset_dirs = [base_dir], [onset_dir]

    base_nights = _read_nights(base_dirs, label, judgeable_configs)
    onset_nights = _read_nights(onset_dirs, label, judgeable_configs)
    base = _release_medians(base_nights)
    onset = _release_medians(onset_nights)
    if not base or not onset:
        return (), None

    deltas = []
    for region in sorted(set(base) | set(onset)):
        before, after = base.get(region), onset.get(region)
        # A region present on one side only genuinely appeared or disappeared;
        # its whole time is the movement, and saying so is the point.
        delta = (after or 0.0) - (before or 0.0)
        if not math.isfinite(delta):
            continue
        deltas.append(RegionDelta(region=region, base=before, onset=after, delta=delta))
    deltas.sort(key=lambda d: (-abs(d.delta), d.region))

    base_sample, onset_sample = _event_sample(base_nights), _event_sample(onset_nights)
    profile = None
    if base_sample is not None and onset_sample is not None:
        events = dict.fromkeys(
            e.event for e in (*base_sample.longest, *onset_sample.longest)
        )
        matched = sorted(
            (
                MatchedEvent(
                    event=event,
                    base=_event_time(base_nights, event),
                    onset=_event_time(onset_nights, event),
                )
                for event in events
            ),
            key=lambda m: (-max(m.base or 0.0, m.onset or 0.0), m.event),
        )
        profile = EventProfile(
            base=base_sample, onset=onset_sample, matched=tuple(matched),
        )
    return tuple(deltas[:limit]), profile


def _event_time(nights: Sequence[_Night], event: int) -> float | None:
    """*event*'s median wall time over the *nights* that simulated it."""
    return _median([
        float(night.wall.loc[event]) for night in nights
        if night.wall is not None and event in night.wall.index
    ])


def region_deltas(
    run_dirs: Sequence[str],
    *,
    label: str,
    base_release: str,
    onset_release: str,
    base_run_id: str | None = None,
    onset_run_id: str | None = None,
    limit: int = MAX_REGIONS,
    judgeable_configs: set[tuple[str, str]] | None = None,
) -> tuple[RegionDelta, ...]:
    """The region half of :func:`region_evidence` alone."""
    deltas, _profile = region_evidence(
        run_dirs, label=label,
        base_release=base_release, onset_release=onset_release,
        base_run_id=base_run_id, onset_run_id=onset_run_id,
        limit=limit, judgeable_configs=judgeable_configs,
    )
    return deltas
