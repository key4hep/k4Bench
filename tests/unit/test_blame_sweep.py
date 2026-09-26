"""Unit tests for :mod:`k4bench.blame.sweep` — every configuration of a scope,
read as one picture.

The fixtures are reduced from the 2026-09-25 night: ILD_FCCee_v02's sweep, where
nineteen configurations stepped in mean event time while their median and
trimmed mean held; ILD_FCCee_v01's, where two configurations stepped in opposite
directions on a baseline that did not move; and ALLEGRO_o2_v01's, where every
configuration's VmPeak stepped and removing the silicon wrapper shrank it.
"""

from __future__ import annotations

import pytest

from k4bench.blame.sweep import (
    ELSEWHERE,
    OK,
    SETTLING,
    STEPPED,
    UNJUDGED,
    WATCH,
    scope_sweep,
    time_shape,
    window_sweeps,
)
from k4bench.regression.models import (
    Direction,
    EventProfile,
    EventSample,
    MetricVerdict,
    NightlyReport,
    RegionDelta,
    RunGroupReport,
    Severity,
    Unjudged,
)

_PLAT = "x86_64-el9-gcc16-opt"
_WINDOW = ("2026-09-23", "2026-09-24")


def _v(label, metric, pct, severity=Severity.OK, *, detector="ILD_FCCee_v02",
       onset="2026-09-24", base="2026-09-23", value=None, baseline=None, **kw):
    confirmed = severity is Severity.CONFIRMED
    return MetricVerdict(
        detector=detector, platform=_PLAT, sample="single_e-_10GeV",
        label=label, metric_family="memory" if "mb" in metric else "time",
        metric=metric, sub_detector=None, run_id="2026-09-25",
        run_date="2026-09-24",
        value=value if value is not None else 1.0 + (pct or 0.0),
        baseline_median=baseline if baseline is not None else 1.0,
        baseline_mad=0.01, pct_change=pct, z_score=5.0,
        severity=severity,
        direction=Direction.NONE if not confirmed else (
            Direction.UP if (pct or 0) > 0 else Direction.DOWN
        ),
        reason="",
        onset_run_id=onset if confirmed else None,
        onset_run_date=onset if confirmed else None,
        last_accepted_run_id=base if confirmed else None,
        last_accepted_run_date=base if confirmed else None,
        **kw,
    )


def _timing(label, mean, median, trimmed, wall, stepped=False, **kw):
    """One configuration's four time metrics; *stepped* confirms mean and wall."""
    s = Severity.CONFIRMED if stepped else Severity.OK
    return [
        _v(label, "mean_time_s", mean, s, **kw),
        _v(label, "median_time_s", median, **kw),
        _v(label, "trimmed_mean_time_s", trimmed, **kw),
        _v(label, "wall_time_s", wall, s, **kw),
    ]


def _group(verdicts, *, detector="ILD_FCCee_v02", reliable=True, **kw):
    return RunGroupReport(
        detector=detector, platform=_PLAT, sample="single_e-_10GeV",
        k4h_release="key4hep-2026-09-24", run_date="2026-09-25",
        run_id="2026-09-25", verdicts=list(verdicts), reliable=reliable,
        geometry_path=f"FCCee/ILD_FCCee/compact/{detector}/{detector}.xml", **kw,
    )


def _ild_v02():
    return scope_sweep(_group([
        *_timing("baseline", -0.072, -0.012, -0.013, -0.071, stepped=True),
        *_timing("no_ScreenSol", -0.120, -0.011, -0.018, -0.115, stepped=True),
        *_timing("no_TPC", +0.108, -0.007, -0.005, +0.100, stepped=True),
        *_timing("no_Gold_STL", +0.061, +0.003, -0.005, +0.055, stepped=True),
        *_timing("no_LumiCal", -0.090, -0.011, -0.010, -0.085, stepped=True),
        *_timing("no_Vertex", -0.031, -0.012, -0.016, -0.029),
        *_timing("no_VertexBarrel", -0.018, -0.016, -0.015, -0.018),
        *_timing("no_InnerTrackers", -0.010, -0.020, -0.019, -0.009),
        *_timing("no_EcalBarrel", -0.011, -0.004, -0.013, -0.010),
        _v("baseline", "peak_vmem_mb", 0.001),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])


def test_a_step_the_median_and_trimmed_mean_did_not_follow_is_carried_by_long_events():
    reading = _ild_v02().reading("time")
    assert reading.lead == "mean_time_s"
    assert dict(reading.shapes)["baseline"].kind == "tail"
    assert set(reading.tail) == {
        "baseline", "no_ScreenSol", "no_TPC", "no_Gold_STL", "no_LumiCal",
    }


def test_the_pattern_names_the_opposite_rows_and_the_configurations_without_the_step():
    reading = _ild_v02().reading("time")
    assert reading.stepped[0] == "no_ScreenSol"  # largest move first
    assert (reading.down, reading.up) == (3, 2)
    assert set(reading.opposite) == {"no_TPC", "no_Gold_STL"}
    # Under a third of the typical step (median of |9.0, 7.2, 12.0, 10.8, 6.1|
    # is 9.0%): the step is absent there.
    assert set(reading.absent) == {"no_VertexBarrel", "no_InnerTrackers", "no_EcalBarrel"}
    assert reading.unconfirmed == ("no_Vertex",)
    assert reading.isolated is False
    assert dict(reading.smaller) == {"no_Vertex": pytest.approx(0.031 / 0.072)}
    assert dict(reading.larger) == {"no_ScreenSol": pytest.approx(0.120 / 0.072)}


def test_isolated_opposite_rows_on_a_flat_baseline_are_named_as_such():
    # ILD_FCCee_v01 on 09-25: 36 configurations within ±3%, two outliers.
    sweep = scope_sweep(_group([
        *_timing("baseline", +0.016, -0.008, +0.011, +0.012),
        *_timing("no_HcalEndcap", -0.112, -0.013, -0.002, -0.107, stepped=True),
        *_timing("no_CompSol", +0.111, +0.009, +0.016, +0.107, stepped=True),
        *(v for i in range(10) for v in _timing(f"no_X{i}", 0.01 * (i % 3), 0.0, 0.0, 0.0)),
    ], detector="ILD_FCCee_v01"), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    reading = sweep.reading("time")
    assert reading.isolated is True
    assert set(reading.opposite) == set(reading.stepped) == {"no_HcalEndcap", "no_CompSol"}
    assert "baseline" in reading.absent
    assert reading.tail == ("no_HcalEndcap", "no_CompSol")


def test_a_baseline_that_moved_without_confirming_is_not_a_flat_one():
    # ALLEGRO_o2_v01 Z→bb: the baseline's wall time moved -5.8% (watched) and two
    # configurations confirmed — the whole sweep moved, nothing is isolated.
    sweep = scope_sweep(_group([
        _v("baseline", "wall_time_s", -0.058, Severity.WATCH),
        _v("no_LumiCalCooling", "wall_time_s", -0.086, Severity.CONFIRMED),
        _v("no_HCalBarrel", "wall_time_s", -0.070, Severity.CONFIRMED),
        _v("no_Vertex", "wall_time_s", -0.041),
    ], detector="ALLEGRO_o2_v01"), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    reading = sweep.reading("time")
    assert reading.lead == "wall_time_s"
    assert reading.isolated is False
    assert reading.baseline.status == WATCH
    assert set(reading.unconfirmed) == {"baseline", "no_Vertex"}


def test_a_memory_step_localised_by_its_size_against_the_baseline():
    # 09-25 ALLEGRO_o2_v01: -760 MB on the baseline, -315 MB without SiWrB,
    # -496 MB without SiWrD. Configurations differ in size, so the comparison is
    # in MB, never in percent (no_EMEC_turbine's -19.5% is the same -760 MB).
    def mem(label, pct, mb, base, stepped=True):
        return _v(label, "peak_vmem_mb", pct,
                  Severity.CONFIRMED if stepped else Severity.OK,
                  detector="ALLEGRO_o2_v01", value=base + mb, baseline=base)
    sweep = scope_sweep(_group([
        mem("baseline", -0.117, -760.0, 6620.0),
        mem("no_EMEC_turbine", -0.195, -760.0, 3900.0),
        mem("no_SiWrB", -0.049, -315.0, 6420.0, stepped=False),
        mem("no_SiWrD", -0.082, -496.0, 6050.0),
    ], detector="ALLEGRO_o2_v01"), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    reading = sweep.reading("memory")
    assert reading.lead == "peak_vmem_mb"
    assert dict(reading.smaller) == {
        "no_SiWrB": pytest.approx(315 / 760), "no_SiWrD": pytest.approx(496 / 760),
    }
    assert reading.larger == ()


def test_unknown_is_never_flat():
    sweep = scope_sweep(_group([
        _v("baseline", "mean_time_s", -0.07, Severity.CONFIRMED),
        _v("no_A", "mean_time_s", None, Severity.UNKNOWN,
           unjudged=Unjudged.INSUFFICIENT_HISTORY),
        _v("no_B", "mean_time_s", -0.01, reanchor_run_date="2026-09-10"),
        _v("no_C", "mean_time_s", -0.08, Severity.CONFIRMED,
           onset="2026-09-10", base="2026-09-08"),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    statuses = {row.label: row.cell("mean_time_s").status for row in sweep.rows}
    assert statuses == {
        "baseline": STEPPED, "no_A": UNJUDGED, "no_B": SETTLING, "no_C": ELSEWHERE,
    }
    reading = sweep.reading("time")
    # None of the three can say "the step is absent here".
    assert reading.absent == () and reading.judged == 1
    assert dict(reading.unjudged) == {
        "no_A": "insufficient_history", "no_B": SETTLING, "no_C": ELSEWHERE,
    }


def test_a_wall_step_the_mean_event_time_did_not_follow_is_outside_the_event_loop():
    # ALLEGRO_o2_v01 no_EMEC_turbine on 09-25: wall -5.6%, mean -1.6% — a much
    # smaller geometry builds faster; the events did not change.
    sweep = scope_sweep(_group([
        _v("no_EMEC_turbine", "wall_time_s", -0.056, Severity.CONFIRMED),
        _v("no_EMEC_turbine", "mean_time_s", -0.016),
        _v("no_EMEC_turbine", "median_time_s", -0.013),
        _v("no_EMEC_turbine", "trimmed_mean_time_s", -0.015),
    ], detector="ALLEGRO_o2_v01"), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    assert time_shape(sweep.row("no_EMEC_turbine")).kind == "outside"


def test_a_step_the_typical_event_followed_is_typical():
    sweep = scope_sweep(_group(
        _timing("no_LumiCalCooling", -0.071, -0.077, -0.068, -0.086, stepped=True),
        detector="ALLEGRO_o2_v01",
    ), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    assert time_shape(sweep.row("no_LumiCalCooling")).kind == "typical"


def test_no_shape_is_claimed_without_a_judged_typical_view():
    sweep = scope_sweep(_group([
        _v("baseline", "mean_time_s", -0.07, Severity.CONFIRMED),
        _v("baseline", "median_time_s", None, Severity.UNKNOWN,
           unjudged=Unjudged.INSUFFICIENT_HISTORY),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    assert time_shape(sweep.row("baseline")) is None


def test_the_event_profile_of_the_stepped_time_verdict_reaches_the_row():
    profile = EventProfile(
        base=EventSample(nights=1, n_events=999, mean=0.5277, median=0.4302,
                         stepping_mean=0.5248, mean_without_longest=0.4927),
        onset=EventSample(nights=2, n_events=999, mean=0.4939, median=0.4279,
                          stepping_mean=0.4909, mean_without_longest=0.4855),
    )
    sweep = scope_sweep(_group([
        _v("baseline", "mean_time_s", -0.072, Severity.CONFIRMED, event_profile=profile),
        _v("baseline", "median_time_s", -0.012),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    shape = time_shape(sweep.row("baseline"))
    assert shape.profile is profile
    assert shape.mean_move == pytest.approx(-0.0641, abs=1e-4)
    assert shape.move_without_longest == pytest.approx(-0.0146, abs=1e-4)
    # Geant4 stepping carries the whole move: the time outside it held.
    assert shape.stepping_share == pytest.approx(1.0, abs=0.01)


def test_window_sweeps_keep_an_unread_scope_as_unread():
    report = NightlyReport(generated_at="x", groups=[
        _group([_v("baseline", "mean_time_s", -0.07, Severity.CONFIRMED)]),
        _group([_v("baseline", "mean_time_s", 0.2)], detector="ILD_FCCee_v01",
               reliable=False),
        _group([_v("baseline", "mean_time_s", 0.0)], detector="SiD",
               job_failures=["no run uploaded for SiD"]),
    ])
    report.groups.append(RunGroupReport(
        detector="IDEA_o2_v01", platform=_PLAT, sample="single_e-_10GeV",
        k4h_release="key4hep-2026-09-20", run_date="2026-09-25", run_id="2026-09-25",
        reliable=True,
    ))
    sweeps = window_sweeps(
        report, base_release=_WINDOW[0], onset_release=_WINDOW[1],
        stacks={"key4hep-2026-09-24"},
    )
    by_detector = {sweep.detector: sweep for sweep in sweeps}
    # Another release measured another window; a missing run measured nothing.
    assert set(by_detector) == {"ILD_FCCee_v01", "ILD_FCCee_v02"}
    assert "reliability check" in by_detector["ILD_FCCee_v01"].unread
    assert by_detector["ILD_FCCee_v02"].unread == ""


def test_a_failed_configuration_costs_itself_and_not_the_sweep():
    group = _group([_v("baseline", "mean_time_s", -0.07, Severity.CONFIRMED)],
                   job_failures=["config no_TPC produced no results"])
    sweep = scope_sweep(group, base_release=_WINDOW[0], onset_release=_WINDOW[1])
    assert sweep.unread == ""
    assert sweep.failures == ("config no_TPC produced no results",)
    assert sweep.reading("time").stepped == ("baseline",)


def test_an_ok_cell_keeps_its_move_and_a_failed_configuration_reads_failed():
    sweep = scope_sweep(_group([
        _v("baseline", "mean_time_s", 0.02),
        _v("no_A", "mean_time_s", 0.0, Severity.FAILURE),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    assert sweep.row("baseline").cell("mean_time_s").status == OK
    assert sweep.row("baseline").cell("mean_time_s").pct == 0.02
    assert all(c.status == "failed" for c in sweep.row("no_A").cells)


# ── One metric per reading ────────────────────────────────────────────────────
# A family is read on its lead metric alone. A configuration where another
# metric of the family stepped is kept apart, never counted, sized or pointed
# as a step of the lead.

def _mixed():
    # The baseline stepped in mean event time only, no_ECAL in wall time only,
    # no_HCAL in neither.
    return scope_sweep(_group([
        _v("baseline", "mean_time_s", +0.100, Severity.CONFIRMED),
        _v("baseline", "median_time_s", +0.005),
        _v("baseline", "trimmed_mean_time_s", +0.006),
        _v("baseline", "wall_time_s", +0.020),
        _v("no_ECAL", "mean_time_s", +0.010),
        _v("no_ECAL", "median_time_s", +0.002),
        _v("no_ECAL", "trimmed_mean_time_s", +0.003),
        _v("no_ECAL", "wall_time_s", +0.120, Severity.CONFIRMED),
        *_timing("no_HCAL", +0.005, +0.001, +0.002, +0.004),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])


def test_a_configuration_that_stepped_in_another_metric_is_not_a_step_of_the_lead():
    reading = _mixed().reading("time")
    assert reading.lead == "mean_time_s"
    assert reading.stepped == ("baseline",)
    assert (reading.judged, reading.up, reading.down) == (3, 1, 0)
    assert reading.also == (("wall_time_s", ("no_ECAL",)),)
    # In the lead metric no_ECAL is without the step, which is what it measured.
    assert set(reading.absent) == {"no_ECAL", "no_HCAL"}
    assert reading.unconfirmed == ()
    assert [label for label, _s in reading.shapes] == ["baseline"]
    assert dict(reading.shapes)["baseline"].lead == "mean_time_s"


def test_a_lead_metric_nobody_judged_keeps_the_row_out_of_the_lead_counts():
    sweep = scope_sweep(_group([
        _v("baseline", "mean_time_s", +0.100, Severity.CONFIRMED),
        _v("no_X", "mean_time_s", None, Severity.UNKNOWN,
           unjudged=Unjudged.INSUFFICIENT_HISTORY),
        _v("no_X", "wall_time_s", +0.090, Severity.CONFIRMED),
        _v("no_Y", "mean_time_s", +0.002),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    reading = sweep.reading("time")
    assert reading.stepped == ("baseline",)
    assert reading.judged == 2
    assert dict(reading.unjudged) == {"no_X": "insufficient_history"}
    assert reading.also == (("wall_time_s", ("no_X",)),)
    assert reading.absent == ("no_Y",)


def test_directions_are_the_lead_metrics_own():
    # The baseline's events got slower while its job got faster (a quicker
    # initialisation); no_B's wall time fell with its events unmoved.
    sweep = scope_sweep(_group([
        _v("baseline", "mean_time_s", +0.100, Severity.CONFIRMED),
        _v("baseline", "wall_time_s", -0.060, Severity.CONFIRMED),
        _v("no_A", "mean_time_s", +0.090, Severity.CONFIRMED),
        _v("no_A", "wall_time_s", +0.080, Severity.CONFIRMED),
        _v("no_B", "mean_time_s", +0.005),
        _v("no_B", "wall_time_s", -0.090, Severity.CONFIRMED),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    reading = sweep.reading("time")
    assert (reading.up, reading.down) == (2, 0)
    assert reading.opposite == ()
    assert reading.stepped == ("baseline", "no_A")
    assert reading.also == (("wall_time_s", ("no_B",)),)


def test_metrics_stepping_together_are_counted_once_under_the_lead():
    sweep = scope_sweep(_group([
        *_timing("baseline", +0.100, +0.004, +0.005, +0.095, stepped=True),
        *_timing("no_A", +0.080, +0.003, +0.002, +0.078, stepped=True),
        *_timing("no_B", +0.002, +0.001, +0.001, +0.003),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    reading = sweep.reading("time")
    assert reading.stepped == ("baseline", "no_A")
    assert reading.judged == 3
    assert reading.also == ()
    assert reading.absent == ("no_B",)


def test_region_movements_ride_on_the_row_that_stepped_in_time():
    regions = (RegionDelta("HCAL", 0.31, 0.46, 0.15), RegionDelta("ECAL", 0.12, 0.121, 0.001))
    sweep = scope_sweep(_group([
        _v("baseline", "mean_time_s", +0.100, Severity.CONFIRMED, region_deltas=regions),
        _v("no_A", "mean_time_s", +0.002, region_deltas=regions),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    assert sweep.row("baseline").regions == regions
    # A configuration that did not step carries no step to decompose.
    assert sweep.row("no_A").regions == ()


def test_a_confirmed_step_of_unknown_size_is_still_a_step():
    # A non-finite change reaches the sweep as no percentage at all: the row
    # stepped, and nothing is said about how far or which way.
    sweep = scope_sweep(_group([
        _v("baseline", "mean_time_s", +0.100, Severity.CONFIRMED),
        _v("no_A", "mean_time_s", float("inf"), Severity.CONFIRMED),
        _v("no_B", "mean_time_s", +0.002),
    ]), base_release=_WINDOW[0], onset_release=_WINDOW[1])
    assert sweep.row("no_A").cell("mean_time_s").pct is None
    reading = sweep.reading("time")
    assert set(reading.stepped) == {"baseline", "no_A"}
    assert (reading.judged, reading.up, reading.down) == (3, 1, 0)
    assert reading.unjudged == ()
    assert reading.absent == ("no_B",)
