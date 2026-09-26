"""Unit tests for :mod:`k4bench.regression.history` — the release-level tail a
confirmed verdict carries.

The tail exists so a reader can tell a step out of a quiet series from a step
out of a series that does this by itself. That only works if the points say
exactly what was measured and what was *judged*: a release recorded on a
contended host has a level and no verdict, and rendering it as a flat night
would invent the very evidence the tail is there to supply.
"""

from __future__ import annotations

import pandas as pd

from k4bench.regression.history import (
    HISTORY_RELEASES,
    history_tail,
    host_facts,
    release_points,
)
from k4bench.regression.models import (
    Direction,
    HostFact,
    HostLevel,
    MetricVerdict,
    Severity,
)


def _frame(rows) -> pd.DataFrame:
    """``(run_id, release, value, reliable)`` rows as the engine's history frame."""
    return pd.DataFrame({
        "run_id":   [r[0] for r in rows],
        "run_date": pd.to_datetime([r[1] for r in rows]),
        "value":    [r[2] for r in rows],
        "reliable": [r[3] for r in rows],
    })


def _verdict(run_id, run_date, value, severity, direction=Direction.NONE) -> MetricVerdict:
    return MetricVerdict(
        detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e", label="baseline", metric_family="time",
        metric="wall_time_s", sub_detector=None,
        run_id=run_id, run_date=run_date, value=value,
        baseline_median=12.0, baseline_mad=0.06, pct_change=0.0, z_score=0.0,
        severity=severity, direction=direction, reason="",
    )


def test_nights_of_one_release_collapse_into_one_point():
    # Two nights re-measuring one release are two measurements of one software
    # state, not two data points: rendering them separately would show a metric
    # moving under a stack that never changed.
    frame = _frame([
        ("2026-07-01", "2026-07-01", 12.0, True),
        ("2026-07-02", "2026-07-01", 12.2, True),
    ])
    verdicts = [
        _verdict("2026-07-01", "2026-07-01", 12.0, Severity.OK),
        _verdict("2026-07-02", "2026-07-01", 12.2, Severity.OK),
    ]
    points = release_points(frame, verdicts)
    assert len(points) == 1
    assert points[0].run_date == "2026-07-01"
    assert points[0].value == 12.1  # the median of the judged nights
    assert (points[0].n_runs, points[0].n_judged) == (2, 2)


def test_a_release_nobody_judged_keeps_its_level_and_says_so():
    # An unreliable host is skipped by the engine, so it produces no verdict at
    # all. The release must still appear — a gap in the tail reads as a stack
    # that was never benchmarked — but it must never read as a flat night.
    frame = _frame([("2026-07-04", "2026-07-04", 19.0, False)])
    points = release_points(frame, [])
    assert len(points) == 1
    assert points[0].value == 19.0
    assert points[0].n_judged == 0
    assert points[0].severity is Severity.UNKNOWN


def test_a_release_that_recorded_nothing_is_not_a_point():
    frame = _frame([("2026-07-04", "2026-07-04", float("nan"), True)])
    assert release_points(frame, []) == ()


def test_the_worst_night_speaks_for_its_release():
    frame = _frame([
        ("2026-07-05", "2026-07-05", 12.0, True),
        ("2026-07-06", "2026-07-05", 14.6, True),
    ])
    verdicts = [
        _verdict("2026-07-05", "2026-07-05", 12.0, Severity.OK),
        _verdict("2026-07-06", "2026-07-05", 14.6, Severity.CONFIRMED, Direction.UP),
    ]
    point = release_points(frame, verdicts)[0]
    assert point.severity is Severity.CONFIRMED
    assert point.direction is Direction.UP


def test_an_unjudged_night_never_outranks_a_judged_one():
    # UNKNOWN is the absence of a verdict, not a verdict of its own: a release
    # with one judged and one unjudged night is as judged as its judged night.
    frame = _frame([
        ("2026-07-05", "2026-07-05", 12.0, True),
        ("2026-07-06", "2026-07-05", 12.1, True),
    ])
    verdicts = [
        _verdict("2026-07-05", "2026-07-05", 12.0, Severity.OK),
        _verdict("2026-07-06", "2026-07-05", 12.1, Severity.UNKNOWN),
    ]
    point = release_points(frame, verdicts)[0]
    assert point.severity is Severity.OK
    assert (point.n_runs, point.n_judged) == (2, 1)


def test_unjudged_values_do_not_set_a_release_level_that_has_judged_ones():
    # The contended night's 19.0 must not drag the level: the engine refused to
    # read that night, and so does the tail.
    frame = _frame([
        ("2026-07-07", "2026-07-07", 12.0, True),
        ("2026-07-08", "2026-07-07", 19.0, False),
    ])
    verdicts = [_verdict("2026-07-07", "2026-07-07", 12.0, Severity.OK)]
    point = release_points(frame, verdicts)[0]
    assert point.value == 12.0
    assert (point.n_runs, point.n_judged) == (2, 1)


def test_points_are_chronological_whatever_order_the_frame_arrives_in():
    frame = _frame([
        ("2026-07-08", "2026-07-08", 12.0, True),
        ("2026-07-01", "2026-07-01", 11.0, True),
        ("2026-07-04", "2026-07-04", 13.0, True),
    ])
    dates = [p.run_date for p in release_points(frame, [])]
    assert dates == ["2026-07-01", "2026-07-04", "2026-07-08"]


def test_hosts_are_recorded_per_release():
    frame = _frame([
        ("2026-07-01", "2026-07-01", 12.0, True),
        ("2026-07-04", "2026-07-04", 14.0, True),
    ])
    machine = pd.DataFrame({
        "run_id": ["2026-07-01", "2026-07-04"],
        "hostname": ["bench01", "bench02"],
        "cpu_logical_cores": [64, 128],
    })
    points = release_points(frame, [], hosts=host_facts(machine))
    assert points[0].hosts == (HostFact("bench01", 64),)
    assert points[1].hosts == (HostFact("bench02", 128),)


def _machines(pairs) -> pd.DataFrame:
    """``(run_id, hostname)`` pairs as a machine-info trend, 64 cores each."""
    return pd.DataFrame({
        "run_id": [p[0] for p in pairs],
        "hostname": [p[1] for p in pairs],
        "cpu_logical_cores": [64] * len(pairs),
    })


def test_each_host_keeps_its_own_level_in_first_sighting_order():
    # One release measured on two machines has one level but two measurements,
    # and only the per-machine view can say whether the machine that measured
    # both sides of a step moved with it.
    frame = _frame([
        ("2026-09-24a", "2026-09-24", 5850.0, True),
        ("2026-09-24b", "2026-09-24", 5840.0, True),
        ("2026-09-24c", "2026-09-24", 5860.0, True),
    ])
    verdicts = [
        _verdict("2026-09-24a", "2026-09-24", 5850.0, Severity.OK),
        _verdict("2026-09-24b", "2026-09-24", 5840.0, Severity.OK),
        _verdict("2026-09-24c", "2026-09-24", 5860.0, Severity.OK),
    ]
    hosts = host_facts(_machines([
        ("2026-09-24a", "fcc-ironic-03"),
        ("2026-09-24b", "fcc-ironic-01"),
        ("2026-09-24c", "fcc-ironic-03"),
    ]))
    point = release_points(frame, verdicts, hosts=hosts)[0]
    assert point.hosts == (HostFact("fcc-ironic-03", 64), HostFact("fcc-ironic-01", 64))
    assert point.host_levels == (
        HostLevel(HostFact("fcc-ironic-03", 64), 5855.0),
        HostLevel(HostFact("fcc-ironic-01", 64), 5840.0),
    )


def test_a_release_split_across_machines_sits_between_their_own_levels():
    # The release level is the median of every night, so machines measuring
    # two levels put it between them — near neither — while each machine's own
    # level stays where it measured.
    frame = _frame([
        ("2026-09-24a", "2026-09-24", 6618.0, True),
        ("2026-09-24b", "2026-09-24", 5850.0, True),
    ])
    verdicts = [
        _verdict("2026-09-24a", "2026-09-24", 6618.0, Severity.OK),
        _verdict("2026-09-24b", "2026-09-24", 5850.0, Severity.WATCH, Direction.DOWN),
    ]
    hosts = host_facts(_machines([
        ("2026-09-24a", "fcc-ironic-01"), ("2026-09-24b", "fcc-ironic-03"),
    ]))
    point = release_points(frame, verdicts, hosts=hosts)[0]
    assert point.value == 6234.0
    assert point.direction is Direction.DOWN
    assert point.host_levels == (
        HostLevel(HostFact("fcc-ironic-01", 64), 6618.0),
        HostLevel(HostFact("fcc-ironic-03", 64), 5850.0),
    )


def test_a_host_whose_nights_were_not_judged_has_no_level_beside_judged_ones():
    # The release level ignores an unjudged night when a judged one exists, and
    # so does each machine: the contended host is listed, but a level the
    # engine refused to read is not one of its measurements.
    frame = _frame([
        ("2026-07-07", "2026-07-07", 12.0, True),
        ("2026-07-08", "2026-07-07", 19.0, False),
    ])
    verdicts = [_verdict("2026-07-07", "2026-07-07", 12.0, Severity.OK)]
    hosts = host_facts(_machines([("2026-07-07", "bench01"), ("2026-07-08", "bench02")]))
    point = release_points(frame, verdicts, hosts=hosts)[0]
    assert point.hosts == (HostFact("bench01", 64), HostFact("bench02", 64))
    assert point.host_levels == (HostLevel(HostFact("bench01", 64), 12.0),)


def test_host_levels_fall_back_to_recorded_values_when_nothing_was_judged():
    frame = _frame([
        ("2026-07-07", "2026-07-07", 12.0, False),
        ("2026-07-08", "2026-07-07", float("nan"), False),
        ("2026-07-09", "2026-07-07", 13.0, False),
    ])
    hosts = host_facts(_machines([
        ("2026-07-07", "bench01"), ("2026-07-08", "bench02"), ("2026-07-09", "bench01"),
    ]))
    point = release_points(frame, [], hosts=hosts)[0]
    # bench02 recorded no finite value, so it has no level rather than a gap.
    assert point.host_levels == (HostLevel(HostFact("bench01", 64), 12.5),)


def test_without_host_facts_there_are_no_host_levels():
    frame = _frame([("2026-07-07", "2026-07-07", 12.0, True)])
    verdicts = [_verdict("2026-07-07", "2026-07-07", 12.0, Severity.OK)]
    assert release_points(frame, verdicts)[0].host_levels == ()


def test_hex_shaped_names_are_preserved_for_context_aware_comparison():
    # A container's nodename is its own id, new on every `docker run`, so two
    # nights on the same machine can record different ids. Shape alone cannot
    # distinguish those from valid hex hostnames, so the reader preserves them;
    # the blame evidence compares adjacent ids together with their core counts.
    machine = pd.DataFrame({
        "run_id": ["2026-07-01", "2026-07-04"],
        "hostname": ["de6b89cdaf2a", "2034eae0e208"],
        "cpu_logical_cores": [64, 64],
    })
    hosts = host_facts(machine)
    assert hosts["2026-07-01"] == HostFact("de6b89cdaf2a", 64)
    assert hosts["2026-07-04"] == HostFact("2034eae0e208", 64)


def test_a_real_host_name_is_left_alone():
    machine = pd.DataFrame({
        "run_id": ["2026-07-01", "2026-07-04", "2026-07-08"],
        # Shape alone is never enough to erase a hostname: the middle value is
        # exactly container-id shaped but remains a real machine identity here.
        "hostname": ["bench01.cern.ch", "deadbeefcafe", "abcdef123456789"],
        "cpu_logical_cores": [64, 64, 64],
    })
    hosts = host_facts(machine)
    assert hosts["2026-07-01"].name == "bench01.cern.ch"
    assert hosts["2026-07-04"].name == "deadbeefcafe"
    assert hosts["2026-07-08"].name == "abcdef123456789"


def test_a_run_with_no_machine_info_simply_has_no_host():
    # "We do not know which machine ran this" and "the host never changed" are
    # different claims, and only the first one is true here.
    machine = pd.DataFrame({
        "run_id": ["2026-07-01", "2026-07-02"],
        "hostname": [None, pd.NA],
        "cpu_logical_cores": [None, None],
    })
    assert host_facts(machine) == {}
    assert host_facts(None) == {}


def test_the_tail_is_cut_at_the_verdicts_own_release():
    # A nightly run can re-benchmark an older release; that verdict must not
    # carry history from releases measured after the state it judged.
    frame = _frame([
        (d, d, 12.0, True)
        for d in ("2026-07-01", "2026-07-04", "2026-07-08", "2026-07-11")
    ])
    points = release_points(frame, [])
    tail = history_tail(points, upto="2026-07-04")
    assert [p.run_date for p in tail] == ["2026-07-01", "2026-07-04"]


def test_the_tail_is_bounded_and_keeps_the_newest_releases():
    frame = _frame([
        (f"2026-06-{day:02d}", f"2026-06-{day:02d}", 12.0, True)
        for day in range(1, 21)
    ])
    tail = history_tail(release_points(frame, []))
    assert len(tail) == HISTORY_RELEASES
    assert tail[-1].run_date == "2026-06-20"
