"""Unit tests for :mod:`k4bench.blame.evidence` — the evidence about a
regression that is not a diff.

Two things are asserted here above all. First, that the derived readings say
what they mean: "the step persisted", "this series does this by itself" and "we
cannot tell yet" are three different conclusions and the third is the common
one. Second, that unknown evidence never turns into negative evidence — an
unread release boundary is not a quiet one, and a configuration that did not run
is not a configuration that stayed flat.
"""

from __future__ import annotations

import dataclasses

import pytest

from k4bench.blame.evidence import (
    HistoryPoint,
    HostReading,
    MetricHistory,
    history_from_verdict,
    outcomes_for_window,
    steps_in_window,
)
from k4bench.blame.prompt import history_block
from k4bench.regression.models import (
    Direction,
    HostFact,
    HostLevel,
    MetricVerdict,
    NightlyReport,
    ReleasePoint,
    RunGroupReport,
    Severity,
    Unjudged,
)

_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"


def _point(release, value, *, judged=True, severity="OK", packages=None,
           hosts=(), levels=None) -> HistoryPoint:
    """*levels* maps a host to its own level; its hosts become the point's."""
    if levels is not None:
        hosts = tuple(levels)
    return HistoryPoint(
        release=release, value=value, n_runs=1, n_judged=1 if judged else 0,
        severity=severity, direction="NONE", hosts=hosts,
        packages_changed=packages,
        host_levels=tuple(
            HostLevel(host, level) for host, level in (levels or {}).items()
        ),
    )


def _history(points, *, base="2026-07-14", onset="2026-07-18",
             median=12.0, mad=0.06) -> MetricHistory:
    return MetricHistory(
        points=tuple(points), baseline_median=median, baseline_mad=mad,
        base_release=base, onset_release=onset,
    )


# ── Reading a series ──────────────────────────────────────────────────────────

def test_the_noise_band_is_the_baselines_own_spread():
    assert _history(()).noise_band == 0.005


def test_a_zero_baseline_has_no_band_rather_than_an_infinite_one():
    assert _history((), median=0.0).noise_band is None


def test_a_move_across_an_unchanged_stack_is_the_series_own_noise():
    # Identical software either side of the boundary, so whatever the metric did
    # there, it did on its own. This is the number that tells a 0.4% step from a
    # 21% one.
    history = _history([
        _point("2026-07-01", 12.0, packages=3),
        _point("2026-07-04", 12.6, packages=0),
    ])
    assert history.quiet_boundary_move == pytest.approx(0.05)


def test_an_unread_boundary_is_not_a_quiet_one():
    # packages_changed=None means nobody looked. Counting it as "nothing changed"
    # would manufacture a noise measurement out of missing provenance.
    history = _history([
        _point("2026-07-01", 12.0, packages=None),
        _point("2026-07-04", 20.0, packages=None),
    ])
    assert history.quiet_boundary_move is None
    assert history.quiet_boundaries == 0


def test_an_unjudged_release_cannot_bound_the_noise():
    history = _history([
        _point("2026-07-01", 12.0, packages=1),
        _point("2026-07-04", 20.0, judged=False, severity="UNKNOWN", packages=0),
    ])
    assert history.quiet_boundary_move is None


def test_a_level_that_held_reads_as_persisted():
    history = _history([
        _point("2026-07-14", 12.0),
        _point("2026-07-18", 14.5, severity="CONFIRMED"),
        _point("2026-07-22", 14.6, severity="CONFIRMED"),
    ])
    assert history.persistence == "persisted"


def test_a_level_that_fell_back_reads_as_returned():
    history = _history([
        _point("2026-07-14", 12.0),
        _point("2026-07-18", 14.5, severity="CONFIRMED"),
        _point("2026-07-22", 12.05),
    ])
    assert history.persistence == "returned"


def test_a_step_with_nothing_measured_after_it_is_simply_unknown():
    # The ordinary state on the night a regression is confirmed. It must not
    # read as either of the other two: "we cannot tell yet" is the answer.
    history = _history([
        _point("2026-07-14", 12.0),
        _point("2026-07-18", 14.5, severity="CONFIRMED"),
    ])
    assert history.persistence == "unknown"


def test_prior_flags_count_only_releases_before_the_window():
    history = _history([
        _point("2026-07-01", 14.0, severity="WATCH"),
        _point("2026-07-04", 12.0),
        _point("2026-07-14", 12.0),
        _point("2026-07-18", 14.5, severity="CONFIRMED"),
    ])
    assert history.prior_flags == 1
    assert len(history.before_window) == 3


def test_a_host_change_is_reported_only_when_it_lands_on_the_onset():
    bench01, bench02 = HostFact("bench01", 64), HostFact("bench02", 64)
    at_onset = _history([
        _point("2026-07-14", 12.0, hosts=(bench01,)),
        _point("2026-07-18", 14.5, severity="CONFIRMED", hosts=(bench02,)),
    ])
    assert at_onset.host_change_at_onset == (bench01, bench02)

    # A machine swapped three releases ago explains nothing about a step that
    # appeared now, and offering it as an alternative story would let any past
    # fleet change compete with the diff.
    earlier = _history([
        _point("2026-07-01", 12.0, hosts=(bench01,)),
        _point("2026-07-14", 12.0, hosts=(bench02,)),
        _point("2026-07-18", 14.5, severity="CONFIRMED", hosts=(bench02,)),
    ])
    assert earlier.host_change_at_onset is None


def test_rotating_container_ids_do_not_claim_the_host_changed():
    old = HostFact("de6b89cdaf2a", 64)
    new = HostFact("2034eae0e208", 64)
    history = _history([
        _point("2026-07-14", 12.0, hosts=(old,)),
        _point("2026-07-18", 14.5, severity="CONFIRMED", hosts=(new,)),
    ])
    assert history.host_change_at_onset is None

    # A changed core count is concrete host evidence even when both names have
    # container-id shape, so that transition must remain visible.
    changed_hardware = _history([
        _point("2026-07-14", 12.0, hosts=(old,)),
        _point(
            "2026-07-18", 14.5, severity="CONFIRMED",
            hosts=(HostFact("2034eae0e208", 128),),
        ),
    ])
    assert changed_hardware.host_change_at_onset == (
        old, HostFact("2034eae0e208", 128),
    )


# ── What each machine measured ────────────────────────────────────────────────
#
# Numbers from the 09-25 ALLEGRO_o2_v01 VmPeak step: the baseline sat at 6620 MB
# on fcc-ironic-01, and the onset release measured 5850 MB on both fcc-ironic-03
# and fcc-ironic-01.

_IRONIC01 = HostFact("fcc-ironic-01", 64)
_IRONIC02 = HostFact("fcc-ironic-02", 64)
_IRONIC03 = HostFact("fcc-ironic-03", 64)


def _step(previous: dict, onset: dict, *, onset_value=5850.0, earlier=None,
          judged=(True, True, True)):
    """A two-release window, 09-23 -> 09-24, with the given per-host levels.
    *earlier* adds a 09-10 release before it; *judged* says, oldest first,
    which of the (up to three) releases were judged."""
    earlier_judged, previous_judged, onset_judged = judged
    points = []
    if earlier is not None:
        points.append(_point("2026-09-10", 6620.0, levels=earlier, judged=earlier_judged))
    points += [
        _point("2026-09-23", 6620.0, levels=previous, judged=previous_judged),
        _point("2026-09-24", onset_value, severity="CONFIRMED", levels=onset,
               judged=onset_judged),
    ]
    return _history(points, base="2026-09-23", onset="2026-09-24", median=6620.0, mad=6.0)


def test_a_machine_that_measured_both_releases_is_not_a_host_change():
    # The 09-25 incident: the onset added fcc-ironic-03 beside fcc-ironic-01,
    # and the old set comparison reported "01 -> 03" although 01 measured the
    # new level too.
    history = _step(
        {_IRONIC01: 6620.0}, {_IRONIC03: 5850.0, _IRONIC01: 5850.0},
    )
    assert history.host_change_at_onset is None
    assert history.host_reading == HostReading(
        kind="reproduced", moved=(_IRONIC01,), at_new=(_IRONIC03, _IRONIC01),
    )


def test_a_step_only_a_new_machine_measured_is_confined_to_it():
    history = _step(
        {_IRONIC01: 6620.0}, {_IRONIC01: 6618.0, _IRONIC03: 5850.0},
    )
    assert history.host_change_at_onset is None
    assert history.host_reading == HostReading(
        kind="confined", stayed=(_IRONIC01,), at_new=(_IRONIC03,),
        at_old=(_IRONIC01,),
    )


def test_a_new_machine_still_at_the_old_level_makes_the_reading_mixed():
    # fcc-ironic-01 moved, but fcc-ironic-03 measured the same software at the
    # old level: that is evidence against "reproduced", common host or not.
    history = _step(
        {_IRONIC01: 6620.0}, {_IRONIC01: 5850.0, _IRONIC03: 6615.0},
    )
    reading = history.host_reading
    assert reading.kind == "mixed"
    assert (reading.at_new, reading.at_old) == ((_IRONIC01,), (_IRONIC03,))


def test_one_common_machine_moving_and_one_staying_is_mixed():
    history = _step(
        {_IRONIC01: 6620.0, _IRONIC02: 6622.0},
        {_IRONIC01: 6619.0, _IRONIC02: 5850.0},
    )
    reading = history.host_reading
    assert reading.kind == "mixed"
    assert (reading.moved, reading.stayed) == ((_IRONIC02,), (_IRONIC01,))


def test_a_level_exactly_halfway_is_undecided_and_the_reading_unknown():
    # Step 12 -> 14: 13.0 is exactly half the step from both levels.
    history = _history([
        _point("2026-07-14", 12.0, levels={_IRONIC01: 12.0}),
        _point("2026-07-18", 14.0, severity="CONFIRMED",
               levels={_IRONIC01: 14.0, _IRONIC03: 13.0}),
    ])
    assert history._level_side(13.0) is None
    assert history._level_side(13.01) == "new"
    assert history._level_side(12.99) == "old"
    assert history.host_reading is None


def test_a_disjoint_change_of_machine_is_still_claimed_without_a_reading():
    history = _step({_IRONIC01: 6620.0}, {_IRONIC03: 5850.0})
    assert history.host_change_at_onset == (_IRONIC01, _IRONIC03)
    assert history.host_reading is None
    assert history.new_host_seen_at_old_level is None


def test_a_new_machine_that_measured_the_old_level_before_is_counter_evidence():
    history = _step(
        {_IRONIC01: 6620.0}, {_IRONIC03: 5850.0}, earlier={_IRONIC03: 6617.0},
    )
    assert history.host_change_at_onset == (_IRONIC01, _IRONIC03)
    assert history.new_host_seen_at_old_level == (_IRONIC03, "2026-09-10")


def test_the_counter_evidence_needs_a_host_change_to_answer():
    history = _step(
        {_IRONIC01: 6620.0}, {_IRONIC01: 5850.0, _IRONIC03: 5850.0},
        earlier={_IRONIC03: 6617.0},
    )
    assert history.host_change_at_onset is None
    assert history.new_host_seen_at_old_level is None


def test_an_old_report_with_overlapping_hosts_gives_no_reading_at_all():
    # Reports written before per-host levels existed: the overlap no longer
    # claims a change, and nothing is claimed in its place.
    history = _history([
        _point("2026-07-14", 12.0, hosts=(_IRONIC01,)),
        _point("2026-07-18", 14.5, severity="CONFIRMED", hosts=(_IRONIC03, _IRONIC01)),
    ])
    assert history.host_change_at_onset is None
    assert history.host_reading is None


def test_rotating_container_ids_give_no_reading_either():
    old, new = HostFact("de6b89cdaf2a", 64), HostFact("2034eae0e208", 64)
    history = _step({old: 6620.0}, {new: 5850.0})
    assert history.host_change_at_onset is None
    assert history.host_reading is None


def test_an_unjudged_release_before_the_onset_is_not_host_evidence():
    # The release's levels are recorded, but the engine refused to read them —
    # an unreliable host. They cannot testify that fcc-ironic-01 moved.
    history = _step(
        {_IRONIC01: 6620.0}, {_IRONIC01: 5850.0}, judged=(True, False, True),
    )
    assert history.host_reading is None


def test_an_unjudged_earlier_release_is_not_counter_evidence():
    history = _step(
        {_IRONIC01: 6620.0}, {_IRONIC03: 5850.0}, earlier={_IRONIC03: 6617.0},
        judged=(False, True, True),
    )
    assert history.host_change_at_onset == (_IRONIC01, _IRONIC03)
    assert history.new_host_seen_at_old_level is None


def test_unnamed_machines_cannot_establish_that_one_machine_measured_both_sides():
    # Two runs that recorded a core count but no hostname compare equal without
    # being known to be the same machine.
    unnamed = HostFact("", 64)
    history = _step({unnamed: 6620.0}, {unnamed: 5850.0})
    assert history.host_reading is None
    named_beside_it = _step({_IRONIC01: 6620.0}, {_IRONIC01: 5850.0, unnamed: 5851.0})
    assert named_beside_it.host_reading is None
    change = _step({_IRONIC01: 6620.0}, {unnamed: 5850.0}, earlier={unnamed: 6617.0})
    assert change.new_host_seen_at_old_level is None


def test_per_host_levels_reach_the_blame_view_from_a_verdict():
    level = HostLevel(_IRONIC01, 5850.0)
    verdict = _verdict(history=(
        ReleasePoint("2026-07-18", 5850.0, 1, 1, Severity.CONFIRMED, Direction.DOWN,
                     (_IRONIC01,), host_levels=(level,)),
    ))
    assert history_from_verdict(verdict).points[0].host_levels == (level,)


# ── Building the view from a verdict ──────────────────────────────────────────

def _verdict(**kw) -> MetricVerdict:
    base = dict(
        detector="ALLEGRO_o1_v03", platform=_PLAT, sample="single_e",
        label="baseline", metric_family="time", metric="wall_time_s",
        sub_detector=None, run_id="2026-07-22", run_date="2026-07-22",
        value=14.6, baseline_median=12.0, baseline_mad=0.06,
        pct_change=0.21, z_score=42.0, severity=Severity.CONFIRMED,
        direction=Direction.UP, reason="step",
        onset_run_id="2026-07-18", onset_run_date="2026-07-18",
        last_accepted_run_id="2026-07-14", last_accepted_run_date="2026-07-14",
    )
    return MetricVerdict(**{**base, **kw})


def test_a_verdict_without_history_yields_no_view():
    # Every report written before histories were recorded. The prompts render
    # without the block rather than with an empty one.
    assert history_from_verdict(_verdict()) is None


def test_the_view_carries_the_verdicts_baseline_and_window():
    verdict = _verdict(history=(
        ReleasePoint("2026-07-14", 12.0, 1, 1, Severity.OK, Direction.NONE),
        ReleasePoint("2026-07-18", 14.5, 1, 1, Severity.CONFIRMED, Direction.UP),
    ))
    view = history_from_verdict(verdict, packages_changed={"2026-07-18": 4})
    assert view is not None
    assert view.baseline_median == 12.0 and view.noise_band == 0.005
    assert (view.base_release, view.onset_release) == ("2026-07-14", "2026-07-18")
    assert view.points[1].packages_changed == 4
    # A release the caller said nothing about stays unread, not unchanged.
    assert view.points[0].packages_changed is None


# ── The controls ──────────────────────────────────────────────────────────────

def _group(detector, *, reliable=True, verdicts=(), failures=(),
           release="key4hep-2026-07-22", sample="single_e") -> RunGroupReport:
    return RunGroupReport(
        detector=detector, platform=_PLAT, sample=sample,
        k4h_release=release, run_date="2026-07-22", run_id="2026-07-22",
        verdicts=list(verdicts), job_failures=list(failures), reliable=reliable,
    )


def _flat(label="baseline", severity=Severity.OK, metric="wall_time_s",
          **kw) -> MetricVerdict:
    return _verdict(
        label=label, metric=metric, severity=severity, direction=Direction.NONE,
        onset_run_id=None, onset_run_date=None,
        last_accepted_run_id=None, last_accepted_run_date=None,
        **kw,
    )


def _outcomes(groups, **kw):
    report = NightlyReport(generated_at="2026-07-22T00:00:00", groups=list(groups))
    return outcomes_for_window(
        report,
        base_release=kw.get("base", "2026-07-14"),
        onset_release=kw.get("onset", "2026-07-18"),
        stacks=kw.get("stacks", {"key4hep-2026-07-22"}),
        regressed_scopes=kw.get("scopes", {("ALLEGRO_o1_v03", _PLAT, "single_e")}),
    )


def test_a_flat_configuration_is_a_control():
    outcomes = _outcomes([_group("IDEA_o1_v03", verdicts=[_flat()])])
    assert [(o.detector, o.status) for o in outcomes] == [("IDEA_o1_v03", "clean")]


def test_a_configuration_that_stepped_in_this_window_is_not_a_control():
    stepped = _verdict(onset_run_date="2026-07-18", last_accepted_run_date="2026-07-14")
    assert _outcomes([_group("IDEA_o1_v03", verdicts=[stepped])]) == ()


def test_a_configuration_still_re_anchoring_is_not_a_control():
    # It stepped, the step was confirmed and accepted, and the baseline was
    # re-seated on the new level. Tonight reads OK because it has not moved
    # again — which is not evidence that it held still across the window.
    settling = _flat(reanchor_run_date="2026-07-18")
    assert _outcomes([_group("IDEA_o1_v03", verdicts=[settling])]) == ()
    # One re-anchoring metric disqualifies the whole configuration.
    assert _outcomes([_group("IDEA_o1_v03", verdicts=[_flat(), settling])]) == ()


def test_a_control_carries_how_far_it_actually_moved():
    # "Clean" only means no detection was flagged. A configuration sitting 4.7%
    # below its baseline is clean and is not flat, so the shift is carried and
    # the largest one wins.
    outcomes = _outcomes([_group("IDEA_o1_v03", verdicts=[
        _flat(pct_change=-0.047),
        _flat(metric="mean_time_s", pct_change=0.008),
    ])])
    assert outcomes[0].status == "clean"
    assert outcomes[0].max_shift == pytest.approx(-0.047)


def test_a_later_step_is_not_folded_into_an_earlier_window_drift():
    # A step after the requested window does not disqualify the configuration
    # as evidence about that window, but its percentage describes the later
    # window and must not look like contemporaneous common-mode drift.
    later = _verdict(
        severity=Severity.CONFIRMED,
        pct_change=0.20,
        onset_run_date="2026-07-19",
        last_accepted_run_date="2026-07-18",
    )
    outcomes = _outcomes([_group("IDEA_o1_v03", verdicts=[later])])
    assert outcomes[0].status == "clean"
    assert outcomes[0].max_shift is None

    # If the configuration also has a current WATCH, the displayed shift comes
    # from that non-confirming metric, never from the unrelated confirmation.
    watching = _flat(
        metric="mean_time_s", severity=Severity.WATCH, pct_change=-0.04,
    )
    outcomes = _outcomes([_group("IDEA_o1_v03", verdicts=[later, watching])])
    assert outcomes[0].status == "watch"
    assert outcomes[0].max_shift == pytest.approx(-0.04)


def test_a_control_with_nothing_measurable_carries_no_shift():
    outcomes = _outcomes([_group("IDEA_o1_v03", verdicts=[
        _flat(pct_change=None),
        _flat(metric="mean_time_s", pct_change=float("nan")),
    ])])
    assert outcomes[0].max_shift is None


def test_an_unreliable_or_failed_run_is_silence_not_a_clean_result():
    assert _outcomes([_group("IDEA_o1_v03", reliable=False, verdicts=[_flat()])]) == ()
    assert _outcomes([_group("IDEA_o1_v03", reliable=None, verdicts=[_flat()])]) == ()
    assert _outcomes([
        _group("IDEA_o1_v03", verdicts=[_flat()], failures=["no run uploaded"])
    ]) == ()


def test_a_configuration_with_nothing_judged_never_becomes_a_control():
    unread = _flat(
        severity=Severity.UNKNOWN,
        unjudged=Unjudged.INSUFFICIENT_HISTORY,
    )
    assert _outcomes([_group("IDEA_o1_v03", verdicts=[unread])]) == ()


def test_partial_coverage_is_offered_as_the_partial_evidence_it_is():
    outcomes = _outcomes([_group("IDEA_o1_v03", verdicts=[
        _flat(),
        _flat(
            metric="peak_rss_mb",
            severity=Severity.UNKNOWN,
            unjudged=Unjudged.INSUFFICIENT_HISTORY,
        ),
    ])])
    assert outcomes[0].unjudged == 1


def test_reported_only_metric_does_not_reduce_control_coverage():
    outcomes = _outcomes([_group("IDEA_o1_v03", verdicts=[
        _flat(),
        _flat(
            metric="user_cpu_s",
            severity=Severity.UNKNOWN,
            unjudged=Unjudged.REPORTED_ONLY,
        ),
    ])])
    assert len(outcomes) == 1
    assert outcomes[0].unjudged == 0


def test_only_reported_only_metrics_never_become_a_control():
    reported_only = _flat(
        metric="user_cpu_s",
        severity=Severity.UNKNOWN,
        unjudged=Unjudged.REPORTED_ONLY,
    )
    assert _outcomes([
        _group("IDEA_o1_v03", verdicts=[reported_only])
    ]) == ()


def test_sub_threshold_movement_is_weak_agreement_not_disagreement():
    outcomes = _outcomes([_group("IDEA_o1_v03", verdicts=[
        _flat(severity=Severity.WATCH),
    ])])
    assert outcomes[0].status == "watch"
    assert outcomes[0].watched == ("wall_time_s",)


def test_a_group_running_another_release_is_not_a_like_for_like_control():
    assert _outcomes([
        _group("IDEA_o1_v03", verdicts=[_flat()], release="key4hep-2026-05-01")
    ]) == ()


def test_the_within_group_control_survives_and_sorts_first():
    # baseline stepped while no_HCAL did not, same detector/sample/platform:
    # the sharpest control the suite produces, and the one a prompt cap must keep.
    groups = [
        _group("ALLEGRO_o1_v03", verdicts=[_flat(label="no_HCAL")]),
        _group("IDEA_o1_v03", verdicts=[_flat()]),
    ]
    outcomes = _outcomes(groups)
    assert (outcomes[0].detector, outcomes[0].label) == ("ALLEGRO_o1_v03", "no_HCAL")


def test_a_step_that_cannot_be_dated_counts_as_inside_the_window():
    # A step nobody can place is not evidence of flatness.
    undated = _verdict(onset_run_date=None)
    assert steps_in_window(undated, ("2026-07-14", "2026-07-18")) is True


# ── Gaps in the evidence are breaks, not things to reach across ───────────────
#
# Both readings below are stated to a model as facts, and either one can flip a
# likely_noise verdict. Bridging an unjudged release or an unknown host turns
# "we do not know" into a claim, which is the one failure mode this module
# exists to prevent.

def test_a_quiet_boundary_is_never_claimed_across_an_unjudged_release():
    # A judged -> B unjudged (a package moved entering it) -> C judged (none
    # moved entering it). The 0 describes B→C only; pairing A with C would
    # attribute a two-boundary move to it and call the software identical.
    history = _history([
        _point("2026-07-01", 12.0, packages=None),
        _point("2026-07-04", 20.0, judged=False, severity="UNKNOWN", packages=1),
        _point("2026-07-08", 18.0, packages=0),
    ])
    assert history.quiet_boundary_move is None
    assert history.quiet_boundaries == 0


def test_a_quiet_boundary_between_two_judged_releases_still_counts():
    history = _history([
        _point("2026-07-01", 12.0, packages=2),
        _point("2026-07-04", 12.6, packages=0),
        _point("2026-07-08", 12.0, packages=0),
    ])
    assert history.quiet_boundary_move == pytest.approx(0.05)
    assert history.quiet_boundaries == 2


def test_an_onset_host_change_needs_the_release_immediately_before_it():
    old, new = HostFact("bench01", 64), HostFact("bench02", 64)
    # The release before the onset recorded no host: whether the machine changed
    # at the onset is unknown, and saying it did would hand the model a rival
    # explanation nobody measured.
    unknown_between = _history([
        _point("2026-07-01", 12.0, hosts=(old,)),
        _point("2026-07-14", 12.0),
        _point("2026-07-18", 18.0, severity="CONFIRMED", hosts=(new,)),
    ])
    assert unknown_between.host_change_at_onset is None

    adjacent = _history([
        _point("2026-07-14", 12.0, hosts=(old,)),
        _point("2026-07-18", 18.0, severity="CONFIRMED", hosts=(new,)),
    ])
    assert adjacent.host_change_at_onset == (old, new)


def test_an_onset_with_no_release_before_it_claims_nothing():
    history = _history([
        _point("2026-07-18", 18.0, severity="CONFIRMED", hosts=(HostFact("bench01"),)),
    ])
    assert history.host_change_at_onset is None


def test_a_boundary_where_the_build_platform_changed_is_not_a_noise_measurement():
    # No tracked package moved, but the compiler and stack did.
    migrated = _history((
        dataclasses.replace(_point("2026-07-14", 12.0), platform=_PLAT),
        _point("2026-07-18", 14.0, packages=0),
    ))
    assert migrated.quiet_boundaries == 0
    assert migrated.quiet_boundary_move is None


def test_history_points_keep_the_platform_that_measured_them():
    verdict = MetricVerdict(
        detector="DET", platform="x86_64-el9-gcc16-opt", sample="s", label="baseline",
        metric_family="time", metric="wall_time_s", sub_detector=None,
        run_id="2026-07-18", run_date="2026-07-18", value=14.0,
        baseline_median=12.0, baseline_mad=0.06, pct_change=0.17, z_score=30.0,
        severity=Severity.CONFIRMED, direction=Direction.UP, reason="step",
        history=(
            ReleasePoint(run_date="2026-07-14", value=12.0, platform=_PLAT),
            ReleasePoint(run_date="2026-07-18", value=14.0),
        ),
    )
    history = history_from_verdict(verdict)
    assert [p.platform for p in history.points] == [_PLAT, None]
    assert f"[measured on {_PLAT}]" in "\n".join(history_block(history))


def test_the_host_spread_is_what_the_machines_disagree_by_on_one_release():
    # 09-24 ran on fcc-ironic-03 and -01 at 5849 and 5851 MB: identical software,
    # 2 MB apart on a 6620 MB baseline — what switching machines does to VmPeak.
    history = _step({_IRONIC01: 6620.0}, {_IRONIC03: 5849.0, _IRONIC01: 5851.0})
    spread, releases = history.host_spread
    assert releases == 1
    assert spread == pytest.approx(2.0 / 6620.0)


def test_no_host_spread_without_a_release_measured_on_two_named_machines():
    assert _step({_IRONIC01: 6620.0}, {_IRONIC01: 5850.0}).host_spread is None
    unnamed = HostFact("", 64)
    assert _step({_IRONIC01: 6620.0}, {_IRONIC01: 5850.0, unnamed: 5000.0}).host_spread is None
    # An unjudged release's levels were never read, so they measure nothing.
    unjudged = _step({_IRONIC01: 6620.0, _IRONIC03: 6000.0}, {_IRONIC01: 5850.0},
                     judged=(True, False, True))
    assert unjudged.host_spread is None
