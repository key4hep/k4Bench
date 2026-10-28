"""The reco suite's regression report: each job, and each of its components,
judged on what :mod:`k4bench.regression.catalog` lists for the suite."""

from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest

from k4bench.layout import RECO, SIM
from k4bench.regression.catalog import COMPONENT_METRICS, JOB_COST_METRICS
from k4bench.regression.models import Direction, Severity
from k4bench.regression.report_builder import group_report_from_run_dirs

_RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"
_NIGHTS = [(date(2026, 10, 1) + timedelta(days=i)).isoformat() for i in range(12)]
_GROUP = ("CLD_o2_v09", "x86_64-el9-gcc16-opt", "p8_ee_Zbb_ecm91")
_STEPPED = "MyConformalTracking"


def _history(root: Path, *, step_from: int | None = None, factor: float = 1.5) -> tuple[str, ...]:
    """The fixture run on each of :data:`_NIGHTS`, each night measuring a
    release of its own; from night *step_from* on, the baseline's
    :data:`_STEPPED` takes *factor* times as long in every event."""
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
