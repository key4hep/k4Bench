"""Unit tests for :mod:`k4bench.blame.summary` — the evidence summary both blame
prompts open with.

Snapshot-style: each test renders one section from a 2026-09-25 shape and pins
the sentences a model has to be able to read off it — which configurations
stepped and which did not, whether the typical event followed, which long events
carried a mean, and what a candidate changed in the detectors it reaches.
"""

from __future__ import annotations

import dataclasses

from k4bench.blame.evidence import HistoryPoint, MetricHistory
from k4bench.blame.geometry import DetectorTouch, FileChange
from k4bench.blame.summary import (
    event_record_line,
    family_reading_lines,
    file_change_phrase,
    history_summary,
    scope_line,
    sweep_table_lines,
    touch_lines,
)
from k4bench.blame.sweep import scope_sweep, time_shape
from k4bench.regression.models import (
    Direction,
    EventProfile,
    EventSample,
    HostFact,
    HostLevel,
    LongEvent,
    MatchedEvent,
    MetricVerdict,
    RunGroupReport,
    Severity,
)

_PLAT = "x86_64-el9-gcc16-opt"


def _v(label, metric, pct, severity=Severity.OK, **kw):
    confirmed = severity is Severity.CONFIRMED
    return MetricVerdict(
        detector=kw.pop("detector", "ILD_FCCee_v02"), platform=_PLAT,
        sample="single_e-_10GeV", label=label,
        metric_family="memory" if "mb" in metric else "time", metric=metric,
        sub_detector=None, run_id="2026-09-25", run_date="2026-09-24",
        value=None if pct is None else 1.0 + pct, baseline_median=1.0,
        baseline_mad=0.01, pct_change=pct, z_score=5.0, severity=severity,
        direction=(Direction.UP if pct > 0 else Direction.DOWN) if confirmed else Direction.NONE,
        reason="", onset_run_id="2026-09-24" if confirmed else None,
        onset_run_date="2026-09-24" if confirmed else None,
        last_accepted_run_id="2026-09-23" if confirmed else None,
        last_accepted_run_date="2026-09-23" if confirmed else None, **kw,
    )


def _timing(label, mean, median, trimmed, wall, stepped=False, **kw):
    s = Severity.CONFIRMED if stepped else Severity.OK
    return [
        _v(label, "mean_time_s", mean, s, **kw),
        _v(label, "median_time_s", median),
        _v(label, "trimmed_mean_time_s", trimmed),
        _v(label, "wall_time_s", wall, s),
    ]


_PROFILE = EventProfile(
    base=EventSample(
        nights=1, n_events=999, mean=0.5277, median=0.4302, stepping_mean=0.5248,
        mean_without_longest=0.4927,
        longest=(LongEvent(37, 35.53, "SET", 30.08), LongEvent(240, 3.71, "TPC", 1.79)),
    ),
    onset=EventSample(
        nights=2, n_events=999, mean=0.4939, median=0.4279, stepping_mean=0.4909,
        mean_without_longest=0.4855,
        longest=(LongEvent(949, 8.81, "unattributed", 5.31), LongEvent(240, 3.80, "TPC", 1.89)),
    ),
    matched=(
        MatchedEvent(37, 35.53, 0.38), MatchedEvent(949, 3.14, 8.81),
        MatchedEvent(240, 3.71, 3.80),
    ),
)


def _sweep(verdicts, **kw):
    return scope_sweep(RunGroupReport(
        detector=kw.pop("detector", "ILD_FCCee_v02"), platform=_PLAT,
        sample="single_e-_10GeV", k4h_release="key4hep-2026-09-24",
        run_date="2026-09-25", run_id="2026-09-25", verdicts=list(verdicts),
        reliable=kw.pop("reliable", True),
    ), base_release="2026-09-23", onset_release="2026-09-24")


def _ild_v02():
    return _sweep([
        *_timing("baseline", -0.072, -0.012, -0.013, -0.071, stepped=True,
                 event_profile=_PROFILE),
        *_timing("no_TPC", +0.108, -0.007, -0.005, +0.100, stepped=True),
        *_timing("no_LumiCal", -0.090, -0.011, -0.010, -0.085, stepped=True),
        *_timing("no_Vertex", -0.018, -0.012, -0.016, -0.017),
        _v("baseline", "peak_vmem_mb", 0.001),
    ])


def test_the_table_lists_every_configuration_baseline_first_with_its_status():
    lines = sweep_table_lines(_ild_v02())
    assert lines[0].startswith("  All 4 configurations of ILD_FCCee_v02 · single_e-_10GeV")
    header, first = lines[1], lines[2]
    assert header.split() == [
        "configuration", "mean", "median", "trimmed", "wall", "|", "VmPeak", "anonRSS",
    ]
    assert first.split()[:5] == ["baseline", "-7.2%S", "-1.2%·", "-1.3%·", "-7.1%S"]
    # Then the largest move.
    assert lines[3].split()[0] == "no_TPC"


def test_a_long_sweep_keeps_the_pattern_and_summarises_the_quiet_remainder():
    sweep = _sweep([
        *_timing("baseline", -0.072, -0.012, -0.013, -0.071, stepped=True),
        *_timing("no_TPC", +0.108, -0.007, -0.005, +0.100, stepped=True),
        *(v for i in range(8) for v in _timing(f"no_Q{i}", 0.001 * i, 0.0, 0.0, 0.0)),
    ])
    lines = sweep_table_lines(sweep, max_rows=4)
    labels = [line.split()[0] for line in lines[2:-1]]
    assert labels[:2] == ["baseline", "no_TPC"]
    assert len(labels) == 4
    assert lines[-1].strip().startswith("… 6 further configuration(s), none stepped")


def test_row_ids_ride_on_the_cells_they_score():
    sweep = _ild_v02()
    row = sweep.row("no_TPC")
    tagged = dataclasses.replace(sweep, rows=tuple(
        dataclasses.replace(r, cells=tuple(
            dataclasses.replace(c, fact_id="r7") if r is row and c.metric == "mean_time_s" else c
            for c in r.cells
        ))
        for r in sweep.rows
    ))
    assert "+10.8%S[r7]" in "\n".join(sweep_table_lines(tagged))


def test_the_reading_states_the_pattern_the_table_holds():
    text = "\n".join(family_reading_lines(_ild_v02(), "time"))
    assert "time: mean event time stepped on 3 of 4 judged configurations (2 down, 1 up)" in text
    assert "The baseline stepped: mean event time -7.2%." in text
    assert "Against the direction of the rest: no_TPC +10.8%." in text
    assert "Without the step — judged and moved less than a third of it (1): no_Vertex -1.8%." in text
    assert (
        "On 3 of 3 stepped configurations the typical event did not follow — the "
        "median and trimmed mean moved less than a third of the step, so a few long "
        "events carry it (e.g. baseline: mean event time -7.2%, median -1.2%, trimmed "
        "mean -1.3%)."
    ) in text


def test_isolated_opposite_configurations_are_said_to_be_that():
    sweep = _sweep([
        *_timing("baseline", +0.016, -0.008, +0.011, +0.012),
        *_timing("no_HcalEndcap", -0.112, -0.013, -0.002, -0.107, stepped=True),
        *_timing("no_CompSol", +0.111, +0.009, +0.016, +0.107, stepped=True),
        *(v for i in range(10) for v in _timing(f"no_X{i}", 0.01, 0.0, 0.0, 0.0)),
    ], detector="ILD_FCCee_v01")
    text = "\n".join(family_reading_lines(sweep, "time"))
    assert "The baseline did not step: mean event time +1.6%." in text
    assert "They stepped in opposite directions." in text
    assert "These are isolated configurations on a baseline that did not move." in text


def test_an_event_moving_with_a_typical_step_is_not_one_that_changed():
    # k4geo#607 on 2026-06-25 slowed every ALLEGRO event by about 12%: an event
    # going 6.0 s -> 7.1 s moved with the typical event, one going 4.6 s -> 6.3 s
    # moved well beyond it.
    profile = EventProfile(
        base=EventSample(nights=1, n_events=99, mean=3.35, median=3.40,
                         longest=(LongEvent(57, 6.0),)),
        onset=EventSample(nights=1, n_events=99, mean=3.70, median=3.80,
                          longest=(LongEvent(57, 7.1),)),
        matched=(MatchedEvent(57, 6.0, 7.1), MatchedEvent(65, 4.6, 6.3)),
    )
    sweep = _sweep([
        _v("baseline", "mean_time_s", 0.105, Severity.CONFIRMED, event_profile=profile),
        _v("baseline", "median_time_s", 0.117),
    ])
    line = event_record_line("baseline", time_shape(sweep.row("baseline")), matched=True)
    assert (
        "event 65 went 4.6 s → 6.3 s, while event 57 (6.0 s → 7.1 s) moved with "
        "the typical event"
    ) in line


def test_the_event_records_name_the_long_event_and_what_the_mean_does_without_it():
    shape = time_shape(_ild_v02().row("baseline"))
    line = event_record_line("baseline", shape, matched=True)
    assert line == (
        "baseline: longest event 35.5 s (event 37, 30.1 s of it in SET) at the base, "
        "8.8 s (event 949, 5.3 s of it in unattributed) at the onset; the same "
        "event numbers at both ends: event 37 went 35.5 s → 0.4 s, event 949 went "
        "3.1 s → 8.8 s, while event 240 (3.7 s → 3.8 s) moved with the typical "
        "event; the mean moved -6.4%, -1.5% without each end's longest event, the "
        "median -0.5%; Geant4 stepping carries 100% of the mean's move."
    )


def test_one_line_per_scope_never_calls_an_unread_or_watched_scope_flat():
    unread = _sweep([_v("baseline", "mean_time_s", 0.2)], reliable=False)
    assert scope_line(unread).endswith(
        "no reading — tonight's run failed the host reliability check, so nothing "
        "was judged; unknown, not flat."
    )
    watched = _sweep([
        _v("baseline", "mean_time_s", 0.072),
        _v("no_VertexBarrelSupports", "mean_time_s", 0.267, Severity.WATCH),
        _v("no_A", "mean_time_s", None, Severity.UNKNOWN),
    ], detector="SiD")
    line = scope_line(watched)
    assert "no time step (mean event time +7.2% to +26.7% on 2 configurations" in line
    assert "not judged on 1" in line
    assert "1 moved past the gates once without confirming — no_VertexBarrelSupports +26.7%" in line
    assert "memory not judged" in line


def test_the_history_summary_names_the_machines_and_what_switching_them_does():
    host01, host03 = HostFact("fcc-ironic-01", 64), HostFact("fcc-ironic-03", 64)

    def point(release, value, levels, severity="OK"):
        return HistoryPoint(
            release=release, value=value, n_runs=len(levels), n_judged=len(levels),
            severity=severity, hosts=tuple(h for h, _ in levels),
            host_levels=tuple(HostLevel(h, v) for h, v in levels),
        )

    history = MetricHistory(
        points=(
            point("2026-09-21", 6620.0, [(host01, 6620.0), (host03, 6620.6)]),
            point("2026-09-23", 6621.0, [(host01, 6621.0)]),
            point("2026-09-24", 5850.0, [(host01, 5851.0), (host03, 5849.0)], "CONFIRMED"),
        ),
        baseline_median=6620.0, baseline_mad=0.5,
        base_release="2026-09-23", onset_release="2026-09-24",
    )
    text = "\n".join(history_summary(history, "peak_vmem_mb"))
    assert "History of baseline's VmPeak: this series normally varies by ±0.0076%" in text
    assert (
        "Hosts: fcc-ironic-01 measured both the release before the onset and the "
        "onset release and moved with the step; machines measuring the same "
        "release differed by up to 0.03% on this series (2 release(s) measured on "
        "several machines)."
    ) in text


def test_a_switched_include_is_paired_by_its_versioned_name():
    change = FileChange(
        path="FCCee/ILD_FCCee/compact/ILD_FCCee_v02/ILD_FCCee_v02.xml",
        includes_added=(
            "../ILD_common_FCCee/materials.xml",
            "../../../CLD/compact/CLD_o2_v09/Vertex_o4_v08_smallBP.xml",
        ),
        includes_removed=("../../../CLD/compact/CLD_o2_v07/Vertex_o4_v07_smallBP.xml",),
    )
    assert file_change_phrase(change) == (
        "ILD_FCCee_v02.xml: include switched "
        "../../../CLD/compact/CLD_o2_v07/Vertex_o4_v07_smallBP.xml → "
        "../../../CLD/compact/CLD_o2_v09/Vertex_o4_v08_smallBP.xml; include added "
        "../ILD_common_FCCee/materials.xml"
    )


def test_the_geometry_map_leads_with_this_run_and_says_what_each_detector_measured():
    same = ("ILD_FCCee_v01",)
    touches = (
        DetectorTouch(
            detector="ILD_FCCee_v01",
            geometry_path="FCCee/ILD_FCCee/compact/ILD_FCCee_v01/ILD_FCCee_v01.xml",
            own_dir="FCCee/ILD_FCCee/compact/ILD_FCCee_v01/",
            own_files=("FCCee/ILD_FCCee/compact/ILD_FCCee_v01/ILD_FCCee_v01.xml",),
            changes=(FileChange(path="ILD_FCCee_v01.xml", includes_added=("m.xml",)),),
            same_as=("ILD_FCCee_v02",),
        ),
        DetectorTouch(
            detector="ILD_FCCee_v02",
            geometry_path="FCCee/ILD_FCCee/compact/ILD_FCCee_v02/ILD_FCCee_v02.xml",
            own_dir="FCCee/ILD_FCCee/compact/ILD_FCCee_v02/",
            own_files=("FCCee/ILD_FCCee/compact/ILD_FCCee_v02/ILD_FCCee_v02.xml",),
            changes=(FileChange(path="ILD_FCCee_v02.xml", includes_added=("m.xml",)),),
            same_as=same,
        ),
        DetectorTouch(
            detector="CLD_o3_v01", geometry_path="FCCee/CLD/compact/CLD_o3_v01/CLD_o3_v01.xml",
            own_dir="FCCee/CLD/compact/CLD_o3_v01/",
            tree_files=("FCCee/CLD/compact/CLD_o2_v09/Vertex_o4_v08_smallBP.xml",),
        ),
    )
    v01 = _sweep([
        *_timing("baseline", +0.016, -0.008, +0.011, +0.012),
        *_timing("no_CompSol", +0.111, +0.009, +0.016, +0.107, stepped=True),
    ], detector="ILD_FCCee_v01")
    text = "\n".join(touch_lines(
        touches, {"ILD_FCCee_v01": [v01]}, this_detector="ILD_FCCee_v02",
    ))
    lines = text.splitlines()
    assert lines[1].startswith("    - ILD_FCCee_v02 — this run: its compact directory")
    assert "The same change as ILD_FCCee_v01." in lines[1]
    assert "this window: none of its runs measured this window." in lines[2]
    v01_line = lines.index(next(line for line in lines if line.startswith("    - ILD_FCCee_v01:")))
    assert "mean event time stepped on 1 of 2" in lines[v01_line + 1]
    assert (
        "    - CLD_o3_v01: nothing in its compact directory; 1 file(s) elsewhere "
        "under FCCee/CLD/ (compact/CLD_o2_v09/Vertex_o4_v08_smallBP.xml), which it "
        "loads only if its compact files include them."
    ) in lines


def test_a_baseline_that_stepped_in_another_metric_is_not_called_a_mean_step():
    # k4geo#607's ALLEGRO_o1_v03 single_e- baseline: wall time confirmed +10.1%
    # while its mean event time moved +6.2% without confirming, and the sweep's
    # time reading leads with the mean because other configurations stepped in it.
    sweep = _sweep([
        _v("baseline", "wall_time_s", 0.101, Severity.CONFIRMED),
        _v("baseline", "mean_time_s", 0.062),
        _v("no_SiWrB", "mean_time_s", 0.121, Severity.CONFIRMED),
    ])
    text = "\n".join(family_reading_lines(sweep, "time"))
    assert (
        "The baseline stepped in wall time +10.1%; its mean event time moved "
        "+6.2% without confirming."
    ) in text
