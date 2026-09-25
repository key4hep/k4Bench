"""Unit tests for :mod:`k4bench.regression.regions` — where inside the detector a
timing step landed.

This is the decomposition that turns a number into a mechanism, so what it must
never do is invent one: a release that recorded no region timing is not a
release where every region read zero, and a region seen on only one side of a
window genuinely appeared or disappeared rather than "moved from zero".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from k4bench.regression.models import LongEvent, MatchedEvent
from k4bench.regression.regions import MAX_REGIONS, region_deltas, region_evidence


def _write_run(
    root: Path, night: str, release: str, per_region: dict[str, float] | None,
    *, label: str = "baseline",
) -> str:
    """One run directory measuring *release*, with or without region timing."""
    run_dir = root / night
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "platform": "x86_64-almalinux9-gcc14.2.0-opt",
        "k4h_release": f"key4hep-{release}", "k4h_release_date": release,
        "sample": "single_e",
    }))
    if per_region is None:
        return str(run_dir)
    # Three events: event 0 is the warm-up every consumer drops, so the two that
    # follow are what the median is taken over.
    events = [0, 1, 2]
    (run_dir / f"{label}_regions.json").write_text(json.dumps({
        "event_numbers": events,
        "event_wall_seconds": [9.9, 1.0, 1.0],
        "event_region_sum_seconds": [9.9, 1.0, 1.0],
        "event_unaccounted_seconds": [0.0, 0.0, 0.0],
        "indexed_top_level_detectors": sorted(per_region),
        "at_location_seconds": [
            {region: 99.0 for region in per_region},  # warm-up: must be ignored
            dict(per_region),
            dict(per_region),
        ],
        "by_birth_seconds": [dict(per_region) for _ in events],
    }))
    return str(run_dir)


def test_the_region_that_absorbed_the_step_comes_first(tmp_path):
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 0.30, "ECAL": 1.00}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.50, "ECAL": 1.01}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert [d.region for d in deltas] == ["HCAL", "ECAL"]
    assert deltas[0].base == 0.30 and deltas[0].onset == 4.50
    assert round(deltas[0].delta, 2) == 4.20


def test_the_warm_up_event_is_excluded_from_the_level(tmp_path):
    # Event 0 carries geometry initialisation; counting it would put every
    # region's median an order of magnitude out.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 2.0}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert (deltas[0].base, deltas[0].onset) == (1.0, 2.0)


def test_a_release_measured_twice_is_one_level_not_two(tmp_path):
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0}),
        _write_run(tmp_path, "2026-07-15", "2026-07-14", {"HCAL": 1.2}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 3.0}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert deltas[0].base == 1.1  # the median of the release's two nights


def test_failed_rerun_does_not_contaminate_release_region_level(tmp_path):
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0}),
        _write_run(tmp_path, "2026-07-15", "2026-07-14", {"HCAL": 101.0}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.0}),
    ]

    deltas = region_deltas(
        dirs,
        label="baseline",
        base_release="2026-07-14",
        onset_release="2026-07-18",
        judgeable_configs={
            ("2026-07-14", "baseline"),
            ("2026-07-18", "baseline"),
        },
    )

    assert [(d.base, d.onset, d.delta) for d in deltas] == [(1.0, 4.0, 3.0)]


def test_a_region_present_on_one_side_only_says_so(tmp_path):
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 1.0, "MUON": 0.5}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    muon = next(d for d in deltas if d.region == "MUON")
    # "Appeared" and "went from zero" are different events, and only the first
    # one happened: the base side has no measurement at all.
    assert muon.base is None and muon.onset == 0.5


def test_no_region_timing_on_either_end_yields_nothing(tmp_path):
    # A run predating the plugin. With one side unmeasured there is no
    # comparison, and treating the missing side as zero would report the entire
    # detector as newly appearing.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", None),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.5}),
    ]
    assert region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    ) == ()


def test_a_window_end_that_was_never_run_yields_nothing(tmp_path):
    dirs = [_write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.5})]
    assert region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    ) == ()


def test_another_configurations_regions_are_never_read(tmp_path):
    # Region files are per benchmark configuration; a removal sweep's
    # no_HCAL run must not answer for the baseline.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0},
                   label="no_HCAL"),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.0},
                   label="no_HCAL"),
    ]
    assert region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    ) == ()
    assert region_deltas(
        dirs, label="no_HCAL",
        base_release="2026-07-14", onset_release="2026-07-18",
    )


def _write_undated_run(
    root: Path, dir_name: str, run_date: str, per_region: dict[str, float],
) -> str:
    """A run that recorded no Key4hep release date, in a directory whose name is
    not simply that run's date."""
    run_dir = root / dir_name
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": run_date, "platform": "x86_64-almalinux9-gcc14.2.0-opt",
        "sample": "single_e",
    }))
    (run_dir / "baseline_regions.json").write_text(json.dumps({
        "event_numbers": [0, 1, 2],
        "event_wall_seconds": [9.9, 1.0, 1.0],
        "event_region_sum_seconds": [9.9, 1.0, 1.0],
        "event_unaccounted_seconds": [0.0, 0.0, 0.0],
        "indexed_top_level_detectors": sorted(per_region),
        "at_location_seconds": [
            {region: 99.0 for region in per_region},
            dict(per_region),
            dict(per_region),
        ],
        "by_birth_seconds": [dict(per_region) for _ in range(3)],
    }))
    return str(run_dir)


def test_a_release_with_no_date_falls_back_to_the_run_date(tmp_path):
    # The engine keys such a run on its run date (``x_date`` is the release date
    # with a run-date fallback), so this must key it the same way — a window's
    # ends are named by the verdict, and a run grouped under a key the verdict
    # never names reads as a release that recorded no regions at all.
    dirs = [
        _write_undated_run(tmp_path, "2026-07-14-rerun", "2026-07-14", {"HCAL": 1.0}),
        _write_undated_run(tmp_path, "2026-07-18-rerun", "2026-07-18", {"HCAL": 4.0}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert [(d.region, d.base, d.onset) for d in deltas] == [("HCAL", 1.0, 4.0)]


def test_the_list_is_bounded_by_the_largest_movements(tmp_path):
    many = {f"REGION_{i}": float(i) for i in range(MAX_REGIONS + 4)}
    moved = {region: value * 2 for region, value in many.items()}
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", many),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", moved),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert len(deltas) == MAX_REGIONS
    assert deltas[0].region == f"REGION_{MAX_REGIONS + 3}"  # the biggest mover


# ── Same-release windows ──────────────────────────────────────────────────────

def test_two_runs_of_one_release_are_compared_against_each_other(tmp_path):
    # The engine's two-strike rule confirms a step within one release: one night
    # flags, the next re-measures the same software state and confirms. Both
    # ends of that window key on the same release, so the release's pool is the
    # same set on both sides and measuring it against itself would report every
    # region as having stood still. The runs are what tell the ends apart.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 0.30, "ECAL": 1.00}),
        _write_run(tmp_path, "2026-07-15", "2026-07-14", {"HCAL": 4.50, "ECAL": 1.01}),
    ]
    deltas = region_deltas(
        dirs, label="baseline",
        base_release="2026-07-14", onset_release="2026-07-14",
        base_run_id="2026-07-14", onset_run_id="2026-07-15",
    )
    assert [d.region for d in deltas] == ["HCAL", "ECAL"]
    assert deltas[0].base == 0.30 and deltas[0].onset == 4.50
    assert deltas[0].delta == pytest.approx(4.20)
    assert deltas[1].delta == pytest.approx(0.01)


def test_a_same_release_window_without_runs_yields_nothing(tmp_path):
    # A report predating run-id capture names such a window by its releases
    # alone, which cannot say which two nights it spans. That is a window whose
    # ends are unknown, not one where every region held still.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 0.30}),
        _write_run(tmp_path, "2026-07-15", "2026-07-14", {"HCAL": 4.50}),
    ]
    assert region_deltas(
        dirs, label="baseline",
        base_release="2026-07-14", onset_release="2026-07-14",
    ) == ()
    # Likewise a run that is not in this corpus: one end is unmeasured.
    assert region_deltas(
        dirs, label="baseline",
        base_release="2026-07-14", onset_release="2026-07-14",
        base_run_id="2026-07-13", onset_run_id="2026-07-15",
    ) == ()


def test_a_same_release_window_whose_ends_are_one_run_yields_nothing(tmp_path):
    # One night against itself is not a comparison, and every delta it produced
    # would be exactly zero — the reading "nothing moved anywhere in the
    # detector", stated about a window that was never measured across.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 0.30}),
        _write_run(tmp_path, "2026-07-15", "2026-07-14", {"HCAL": 4.50}),
    ]
    assert region_deltas(
        dirs, label="baseline",
        base_release="2026-07-14", onset_release="2026-07-14",
        base_run_id="2026-07-15", onset_run_id="2026-07-15",
    ) == ()


def test_a_cross_release_window_still_pools_every_night_of_its_ends(tmp_path):
    # Runs are how a same-release window's ends are told apart; they must not
    # narrow a cross-release one, whose ends are whole releases however many
    # nights measured them.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.00}),
        _write_run(tmp_path, "2026-07-15", "2026-07-14", {"HCAL": 3.00}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 10.00}),
    ]
    deltas = region_deltas(
        dirs, label="baseline",
        base_release="2026-07-14", onset_release="2026-07-18",
        base_run_id="2026-07-15", onset_run_id="2026-07-18",
    )
    # The base is the median of both nights of its release (2.0), not the run
    # named as the window's end (3.0).
    assert [(d.region, d.base, d.onset) for d in deltas] == [("HCAL", 2.0, 10.0)]


# ── The per-event wall times at both ends ─────────────────────────────────────

def _write_events_run(
    root: Path, night: str, release: str, walls: list[float],
    *, long_region: tuple[int, str, float] | None = None, outside: float = 0.003,
) -> str:
    """One run with per-event walls (event 0 first, the warm-up), each event's
    stepping time its wall minus *outside*, and *long_region* charging most of
    one event to one region."""
    run_dir = root / night
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "platform": "x86_64-almalinux9-gcc14.2.0-opt",
        "k4h_release": f"key4hep-{release}", "k4h_release_date": release,
        "sample": "single_e",
    }))
    at_location = []
    for event, wall in enumerate(walls):
        regions = {"TPC": wall - outside, "SET": 0.0}
        if long_region is not None and long_region[0] == event:
            _event, region, seconds = long_region
            regions = {"TPC": wall - outside - seconds, region: seconds}
        at_location.append(regions)
    (run_dir / "baseline_regions.json").write_text(json.dumps({
        "event_numbers": list(range(len(walls))),
        "event_wall_seconds": walls,
        "event_region_sum_seconds": [w - outside for w in walls],
        "event_unaccounted_seconds": [outside] * len(walls),
        "indexed_top_level_detectors": ["TPC", "SET"],
        "at_location_seconds": at_location,
        "by_birth_seconds": at_location,
    }))
    return str(run_dir)


def test_the_event_profile_names_the_long_event_that_carried_the_mean(tmp_path):
    # The 09-25 ILD_FCCee_v02 shape: event 3 took 35.5 s at the base, most of it
    # in SET, and is gone at the onset while the typical event held.
    typical = [0.43, 0.44, 0.42, 0.45, 0.43, 0.44]
    base_walls = [50.0, *typical[:3], 35.5, *typical[3:]]
    onset_walls = [50.0, *typical[:3], 0.40, *typical[3:]]
    dirs = [
        _write_events_run(tmp_path, "2026-09-23", "2026-09-23", base_walls,
                          long_region=(4, "SET", 30.1)),
        _write_events_run(tmp_path, "2026-09-24", "2026-09-24", onset_walls),
    ]
    _deltas, profile = region_evidence(
        dirs, label="baseline", base_release="2026-09-23", onset_release="2026-09-24",
    )
    base, onset = profile.base, profile.onset
    assert base.nights == onset.nights == 1 and base.n_events == 7
    assert base.longest[0] == LongEvent(event=4, seconds=35.5, region="SET",
                                        region_seconds=pytest.approx(30.1))
    assert onset.longest[0].seconds == 0.45
    # The typical event held: the median moves by one place in a sorted sample.
    assert (base.median, onset.median) == (0.44, 0.43)
    assert base.mean == pytest.approx((sum(typical) + 35.5) / 7)
    # Without the one long event the base is the typical sample again.
    assert base.mean_without_longest == pytest.approx(sum(typical) / 6)
    # Stepping is every region together; the rest is time outside it.
    assert base.mean - base.stepping_mean == pytest.approx(0.003)


def test_the_warm_up_event_never_counts_as_a_long_event(tmp_path):
    dirs = [
        _write_events_run(tmp_path, "2026-09-23", "2026-09-23", [99.0, 1.0, 2.0]),
        _write_events_run(tmp_path, "2026-09-24", "2026-09-24", [99.0, 1.0, 3.0]),
    ]
    _deltas, profile = region_evidence(
        dirs, label="baseline", base_release="2026-09-23", onset_release="2026-09-24",
    )
    assert [e.event for e in profile.base.longest] == [2, 1]
    assert profile.base.mean == 1.5 and profile.onset.mean == 2.0


def test_a_release_measured_on_several_nights_is_summarised_by_their_median(tmp_path):
    # With a fixed seed each night simulates the same events: event 1 is the
    # long one on every night, and its time is the median of its nights.
    dirs = [
        _write_events_run(tmp_path, "2026-09-23", "2026-09-23", [9.0, 30.0, 1.0, 1.0]),
        _write_events_run(tmp_path, "2026-09-24", "2026-09-24", [9.0, 8.0, 1.0, 1.0]),
        _write_events_run(tmp_path, "2026-09-25", "2026-09-24", [9.0, 9.0, 1.2, 1.2]),
        _write_events_run(tmp_path, "2026-09-26", "2026-09-24", [9.0, 7.0, 1.1, 1.1]),
    ]
    _deltas, profile = region_evidence(
        dirs, label="baseline", base_release="2026-09-23", onset_release="2026-09-24",
    )
    onset = profile.onset
    assert onset.nights == 3
    assert onset.longest[0] == LongEvent(event=1, seconds=8.0, region="TPC",
                                         region_seconds=pytest.approx(7.997))
    assert onset.median == 1.1  # the median of the nights' medians
    assert onset.mean == pytest.approx((8.0 + 2.0) / 3)  # likewise their means


def test_no_profile_without_both_ends(tmp_path):
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", None),
        _write_events_run(tmp_path, "2026-07-18", "2026-07-18", [9.0, 1.0, 1.0]),
    ]
    assert region_evidence(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    ) == ((), None)


def test_the_longest_events_are_followed_to_the_other_end(tmp_path):
    # The same event numbers at both ends: event 1 changed from 35 s to 0.4 s,
    # event 2 held — with a fixed seed, the sample is otherwise the same events.
    dirs = [
        _write_events_run(tmp_path, "2026-09-23", "2026-09-23", [9.0, 35.0, 3.7, 0.4]),
        _write_events_run(tmp_path, "2026-09-24", "2026-09-24", [9.0, 0.4, 3.8, 8.8]),
    ]
    _deltas, profile = region_evidence(
        dirs, label="baseline", base_release="2026-09-23", onset_release="2026-09-24",
    )
    assert profile.matched == (
        MatchedEvent(event=1, base=35.0, onset=0.4),
        MatchedEvent(event=3, base=0.4, onset=8.8),
        MatchedEvent(event=2, base=3.7, onset=3.8),
    )
