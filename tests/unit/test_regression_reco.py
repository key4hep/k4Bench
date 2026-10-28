"""The reco suite's regression report: each job, and each of its components,
judged on what :mod:`k4bench.regression.catalog` lists for the suite."""

from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest

from k4bench.layout import RECO, SIM
from k4bench.regression.catalog import COMPONENT_METRICS, JOB_COST_METRICS, sources
from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.heap import MB, NO_COMPONENT
from k4bench.regression.components import (
    DECOMPOSED_AS,
    EVENT,
    HEAP,
    JOB,
    MAX_COMPONENTS,
    TYPICAL_EVENT,
    _timed,
    component_deltas,
)
from k4bench.regression.models import Direction, Severity
from k4bench.regression.report_builder import group_report_from_run_dirs

_RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"
_NIGHTS = [(date(2026, 10, 1) + timedelta(days=i)).isoformat() for i in range(12)]
_GROUP = ("CLD_o2_v09", "x86_64-el9-gcc16-opt", "p8_ee_Zbb_ecm91")
_STEPPED = "MyConformalTracking"


def _history(
    root: Path, *, step_from: int | None = None, factor: float = 1.5, event_step: float = 0.0,
) -> tuple[str, ...]:
    """The fixture run on each of :data:`_NIGHTS`, each night measuring a
    release of its own; from night *step_from* on, the baseline's
    :data:`_STEPPED` takes *factor* times as long in every event, and the
    baseline's events *event_step* seconds longer."""
    dirs = []
    for i, night in enumerate(_NIGHTS):
        run = root / night
        shutil.copytree(_RECO_RUN, run)
        info = json.loads((run / "run_info.json").read_text())
        info.update(date=night, k4h_release=f"key4hep-{night}", k4h_release_date=night)
        (run / "run_info.json").write_text(json.dumps(info))
        if step_from is not None and i >= step_from:
            path = run / "baseline_components.json"
            timing = json.loads(path.read_text())
            column = [c["name"] for c in timing["components"]].index(_STEPPED)
            for row in timing["execute"]["wall_s"]:
                if row[column] is not None:
                    row[column] *= factor
            path.write_text(json.dumps(timing))
            events_path = run / "baseline_events.json"
            events = json.loads(events_path.read_text())
            events["event_times_s"] = [t + event_step for t in events["event_times_s"]]
            events_path.write_text(json.dumps(events))
        dirs.append(str(run))
    return tuple(dirs)


def test_a_step_in_one_algorithm_is_confirmed_as_that_algorithms(tmp_path):
    group = group_report_from_run_dirs(RECO, *_GROUP, _history(tmp_path, step_from=10))
    by_series = {(v.label, v.metric, v.sub_detector): v for v in group.verdicts}
    stepped = by_series[("baseline", "seconds_per_event", _STEPPED)]
    assert stepped.severity is Severity.CONFIRMED
    assert stepped.direction is Direction.UP
    assert stepped.metric_family == "component_time"
    assert stepped.pct_change == pytest.approx(0.5)
    assert (stepped.last_accepted_run_id, stepped.onset_run_id) == (_NIGHTS[9], _NIGHTS[10])
    # Nothing else moved: no other algorithm, no other config, and not the
    # job's event time, which the fixture's event file keeps as it was.
    others = [v for v in group.regressions if v is not stepped]
    assert others == []
    assert by_series[("baseline", "mean_time_s", None)].severity is Severity.OK


def test_a_confirmed_step_names_the_components_it_moved(tmp_path):
    group = group_report_from_run_dirs(
        RECO, *_GROUP, _history(tmp_path, step_from=10, event_step=1.2),
    )
    by_series = {(v.label, v.metric, v.sub_detector): v for v in group.verdicts}
    job = by_series[("baseline", "mean_time_s", None)]
    assert job.severity is Severity.CONFIRMED
    lead, *rest = job.component_deltas
    assert lead.component == _STEPPED
    assert lead.delta == pytest.approx(0.5 * lead.base)
    # Its time moved and its work did not: the instructions it retired held.
    assert lead.instructions_change == pytest.approx(0.0)
    assert len(job.component_deltas) == MAX_COMPONENTS
    assert all(abs(d.delta) < 1e-9 for d in rest)
    # A component's verdict carries its own movement alone.
    own = by_series[("baseline", "seconds_per_event", _STEPPED)]
    assert own.component_deltas == (lead,)
    # Verdicts that did not confirm carry none.
    assert all(not v.component_deltas for v in group.verdicts if not v.flagged)


def test_a_job_and_each_component_are_series_of_their_own(tmp_path):
    group = group_report_from_run_dirs(RECO, *_GROUP, _history(tmp_path))
    baseline = [v for v in group.verdicts if v.label == "baseline"]
    job = {v.metric: v for v in baseline if v.sub_detector is None}
    # The sim suite's job metrics, and the job's initialize.
    assert {"wall_time_s", "peak_vmem_mb", "mean_time_s", "initialize_s"} <= set(job)
    assert job["initialize_s"].metric_family == JOB_COST_METRICS["initialize_s"]
    per_component: dict[str, set[str]] = {}
    for v in baseline:
        if v.sub_detector is not None:
            assert v.metric_family == COMPONENT_METRICS[v.metric]
            per_component.setdefault(v.sub_detector, set()).add(v.metric)
    # An algorithm is judged per event and in initialize; a service only there.
    assert per_component[_STEPPED] == {"seconds_per_event", "initialize_s"}
    assert per_component["GeoSvc"] == {"initialize_s"}
    # A sequencer's own cost is bookkeeping, so it is no series.
    assert not {"k4FWCore__Sequencer", "k4FWCore__Algs"} & set(per_component)
    assert all(v.severity is Severity.OK for v in group.verdicts if v.sub_detector)


def test_the_sim_suite_judges_no_component(tmp_path):
    group = group_report_from_run_dirs(SIM, *_GROUP, _history(tmp_path, step_from=10))
    assert group.verdicts
    assert all(v.sub_detector is None for v in group.verdicts)
    assert "initialize_s" not in {v.metric for v in group.verdicts}


def test_every_judged_reco_metric_is_taken_apart_by_component():
    judged = {
        metric for source in sources(RECO)
        for metric in (*source.judged, *source.component_judged)
    }
    # The peak virtual memory alone is not: no component cost measures the
    # address space, and the heap held would name components that did not
    # cause its step.
    assert "peak_vmem_mb" in judged and "peak_vmem_mb" not in DECOMPOSED_AS
    judged.discard("peak_vmem_mb")
    assert judged <= set(DECOMPOSED_AS), sorted(judged - set(DECOMPOSED_AS))


def _timing(wall: list[list[float | None]], *, initialize: list[float]) -> ComponentTiming:
    """Algorithms ``A`` and ``B`` and the service ``Svc``, each event's own
    times *wall* (instructions the same), and their own times in initialize."""
    names = [("A", "algorithm"), ("B", "algorithm"), ("Svc", "service")]
    zeros = [[0.0] * len(names) for _ in wall]
    return ComponentTiming.from_json({
        "components": [{"name": n, "category": c} for n, c in names],
        "event_numbers": list(range(len(wall))),
        "execute": {
            "wall_s": wall, "cpu_s": wall, "peak_rss_increase_mb": zeros, "instructions": wall,
        },
        "lifecycle": {"initialize": {
            "wall_s": initialize, "cpu_s": initialize,
            "peak_rss_increase_mb": [0.0] * len(names),
        }},
        "phases": {},
    })


def test_the_typical_event_time_is_taken_apart_over_the_typical_events():
    # B costs 0.1 s in every event but the two slowest of 40, where it takes
    # 50 s: a tail the trimmed mean and the median do not see.
    wall = [[1.0, 0.1, None] for _ in range(41)]
    wall[7][1] = wall[20][1] = 50.0
    timing = _timing(wall, initialize=[0.0, 0.0, 0.0])
    assert DECOMPOSED_AS["mean_time_s"] == EVENT
    assert DECOMPOSED_AS["median_time_s"] == DECOMPOSED_AS["trimmed_mean_time_s"] == TYPICAL_EVENT
    mean, mean_work = _timed(timing, EVENT)
    assert mean["B"] == pytest.approx((38 * 0.1 + 100.0) / 40)
    typical, typical_work = _timed(timing, TYPICAL_EVENT)
    assert typical.to_dict() == pytest.approx({"A": 1.0, "B": 0.1})
    assert typical_work.to_dict() == pytest.approx({"A": 1.0, "B": 0.1})


def test_a_service_initialized_during_an_event_is_not_summed_into_one_component():
    # Svc initializes on first use, in event 3: the auditor lists its 5 s under
    # initialize and leaves them in A's execute time too.
    wall = [[1.0, 0.5, None] for _ in range(6)]
    wall[3][0] += 5.0
    cost, work = _timed(_timing(wall, initialize=[0.2, 0.1, 5.0]), JOB)
    assert cost.to_dict() == pytest.approx({
        "A in initialize": 0.2, "B in initialize": 0.1, "Svc in initialize": 5.0,
        "A in execute": 11.0, "B in execute": 3.0,
    })
    # Every entry is one phase of one component, never a sum over phases.
    assert not {"A", "B", "Svc"} & set(cost.index)
    # Initialize counted no instructions, which costs its own entries alone.
    assert set(work.index) == {"A in execute", "B in execute"}


def _write_heap(run: Path, held_mb: dict[str, float]) -> None:
    """The memory pass's ``baseline_heap.json`` for *run*: each component of
    *held_mb* holding that much at the start and end of each of three events."""
    names = list(held_mb)
    held = [int(held_mb[name] * MB) for name in names]
    events = [0, 1, 2]
    (run / "baseline_heap.json").write_text(json.dumps({
        "schema_version": 1,
        "producer": "k4BenchAuditor",
        "components": names,
        "before_first_event": {
            "held_bytes": held, "peak_held_bytes": held,
            "malloc_in_use_bytes": sum(held), "rss_anon_mb": 400.0,
        },
        "event_numbers": events,
        "execute": {
            "start_held_bytes": [held for _ in events],
            "held_bytes": [held for _ in events],
            "peak_held_bytes": [held for _ in events],
            "malloc_in_use_bytes": [sum(held) for _ in events],
            "rss_anon_mb": [400.0 for _ in events],
        },
        "at_end": {"held_bytes": held, "malloc_in_use_bytes": sum(held), "rss_anon_mb": 400.0},
        "table": {"peak_entries": 1000, "mapped_bytes": 1 << 20},
    }))


def test_an_anonymous_memory_step_is_taken_apart_into_the_heap_each_component_holds(tmp_path):
    assert DECOMPOSED_AS["mean_rss_anon_mb"] == HEAP
    base, onset = (Path(d) for d in _history(tmp_path)[:2])
    held = {NO_COMPONENT: 50.0, "GeoSvc": 300.0, "k4FWCore__Sequencer": 0.0}
    _write_heap(base, {**held, _STEPPED: 5.0, "MyAIDAProcessor": 0.0, "OldReader": 2.0})
    _write_heap(onset, {**held, _STEPPED: 25.0, "MyAIDAProcessor": 3.0, "NewReader": 4.0})

    def deltas():
        return component_deltas(
            [str(base), str(onset)], label="baseline", quantity=HEAP,
            base_release=_NIGHTS[0], onset_release=_NIGHTS[1],
        )

    by_name = {d.component: d for d in deltas()}
    lead = deltas()[0]
    assert (lead.component, lead.base, lead.onset) == (_STEPPED, 5.0, 25.0)
    assert lead.delta == pytest.approx(20.0)
    # The heap says nothing of instructions.
    assert lead.instructions_change is None
    # What one end did not hold is absent there, never zero.
    assert (by_name["NewReader"].base, by_name["NewReader"].onset) == (None, 4.0)
    assert (by_name["OldReader"].base, by_name["OldReader"].onset) == (2.0, None)
    # A component that ran at both ends but held nothing at one is a zero
    # there: it started holding memory, it was not added.
    assert (by_name["MyAIDAProcessor"].base, by_name["MyAIDAProcessor"].onset) == (0.0, 3.0)
    # A sequencer that held nothing is no evidence at all.
    assert "k4FWCore__Sequencer" not in by_name
    # What no component allocated is a holder too, and here it stood still.
    assert by_name[NO_COMPONENT].delta == 0.0 and by_name["GeoSvc"].delta == 0.0
    # An end without a memory pass leaves nothing to compare.
    (onset / "baseline_heap.json").unlink()
    assert deltas() == ()


def test_a_phase_without_instruction_counts_costs_only_its_own_entries():
    from k4bench.analysis.loader import load_component_timing
    from k4bench.regression.components import _job_phases

    timing = load_component_timing(_RECO_RUN, labels=["baseline"])["baseline"]
    timing.lifecycle["initialize"] = timing.lifecycle["initialize"].drop(columns="instructions")
    work = _job_phases(timing, "instructions")
    assert f"{_STEPPED} in execute" in work.index
    assert not any(entry.endswith(" in initialize") for entry in work.index)
    # The time of every phase is still there.
    assert "GeoSvc in initialize" in _job_phases(timing, "wall_s").index
