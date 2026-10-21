"""Unit tests for the k4BenchAuditor's per-component files
(:mod:`k4bench.analysis.components` and ``load_component_timing``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from k4bench.analysis.components import (
    ALLOCATION_METRICS,
    COMPONENT_METRICS,
    HARDWARE_METRICS,
    INCLUSIVE_METRICS,
    THREAD_METRICS,
    ComponentTiming,
    event_loop_time,
)
from k4bench.analysis.loader import load_component_timing, load_each_component_timing

S = 1_000_000_000  # ns per s


def _raw(**overrides) -> dict:
    """A job with one sequencer ``Top`` running ``Fast`` and ``Slow``, and a
    service ``Geo``. ``Slow`` was filtered out of event 2."""
    raw = {
        "schema_version": 1,
        "producer": "k4BenchAuditor",
        "measurement_overhead_ns": 800.0,
        "threads": 1,
        "process_start_epoch_ns": 100 * S,
        "phases": {
            "initialize": {"begin_epoch_ns": 102 * S, "end_epoch_ns": 105 * S},
            "execute": {"begin_epoch_ns": 105 * S, "end_epoch_ns": 111 * S},
        },
        "components": [
            {
                "name": "Geo",
                "category": "service",
                "type": "GeoSvc",
                "impl": "GeoSvc",
                "library": "/cvmfs/x/k4geo/lib/libGeo.so",
                "parent": None,
                "execute_calls": 0,
            },
            {
                "name": "Top",
                "category": "algorithm",
                "type": "Gaudi::Sequencer",
                "impl": "Gaudi::Sequencer",
                "library": "/cvmfs/x/Gaudi/lib/libGaudiKernel.so",
                "parent": None,
                "execute_calls": 3,
            },
            {
                "name": "Fast",
                "category": "algorithm",
                "type": "Wrapper",
                "impl": "FastImpl",
                "library": "/cvmfs/x/wrap/lib/libWrap.so",
                "parent": 1,
                "execute_calls": 3,
            },
            {
                "name": "Slow",
                "category": "algorithm",
                "type": "Slow",
                "impl": "Slow",
                "library": "/cvmfs/x/slow/lib/libSlow.so",
                "parent": 1,
                "execute_calls": 2,
            },
        ],
        "lifecycle": {
            "initialize": {
                "wall_s": [2.0, 0.001, 0.5, 0.25],
                "cpu_s": [1.9, 0.001, 0.4, 0.25],
                "peak_rss_increase_mb": [400.0, 0.0, 10.0, 0.0],
            },
        },
        "event_numbers": [0, 1, 2],
        "execute": {
            "wall_s": [[None, 0.01, 0.2, 1.5], [None, 0.01, 0.1, 1.0], [None, 0.01, 0.1, None]],
            "cpu_s": [[None, 0.01, 0.2, 1.4], [None, 0.01, 0.1, 1.0], [None, 0.01, 0.1, None]],
            "peak_rss_increase_mb": [
                [None, 0.0, 5.0, 20.0],
                [None, 0.0, 0.0, 0.0],
                [None, 0.0, 0.0, None],
            ],
        },
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def timing() -> ComponentTiming:
    return ComponentTiming.from_json(_raw())


class TestFromJson:
    def test_components_keep_order_and_resolve_parents_to_names(self, timing):
        assert timing.components.index.tolist() == ["Geo", "Top", "Fast", "Slow"]
        assert timing.components["parent"].tolist() == [None, None, "Top", "Top"]
        assert timing.components.loc["Fast", "impl"] == "FastImpl"

    def test_a_file_without_impl_libraries_names_only_unwrapped_ones(self, timing):
        assert timing.components["impl_library"].tolist() == [
            "/cvmfs/x/k4geo/lib/libGeo.so", "/cvmfs/x/Gaudi/lib/libGaudiKernel.so",
            "", "/cvmfs/x/slow/lib/libSlow.so",
        ]

    def test_the_recorded_impl_library_is_kept(self):
        raw = _raw()
        raw["components"][2]["impl_library"] = "/cvmfs/x/fast/lib/libFast.so"
        components = ComponentTiming.from_json(raw).components
        assert components.loc["Fast", "impl_library"] == "/cvmfs/x/fast/lib/libFast.so"
        assert components.loc["Fast", "library"] == "/cvmfs/x/wrap/lib/libWrap.so"

    def test_execute_frames_are_events_by_components_per_metric(self, timing):
        assert set(timing.execute) == set(COMPONENT_METRICS)
        wall = timing.execute["wall_s"]
        assert wall.index.tolist() == [0, 1, 2]
        assert wall.index.name == "event_number"
        assert wall.loc[1, "Slow"] == 1.0

    def test_a_component_that_did_not_run_is_nan_not_zero(self, timing):
        wall = timing.execute["wall_s"]
        assert math.isnan(wall.loc[2, "Slow"])
        assert wall["Geo"].isna().all()

    def test_lifecycle_frames_are_components_by_metrics(self, timing):
        init = timing.lifecycle["initialize"]
        assert init.loc["Geo", "wall_s"] == 2.0
        assert init.loc["Geo", "peak_rss_increase_mb"] == 400.0

    def test_metadata(self, timing):
        assert timing.threads == 1
        assert timing.measurement_overhead_ns == 800.0
        assert timing.phases["execute"] == (105 * S, 111 * S)

    def test_hardware_counters_error_says_why_they_are_missing(self):
        reason = "perf_event_open: Function not implemented"
        timing = ComponentTiming.from_json(_raw(hardware_counters_error=reason))
        assert timing.hardware_counters_error == reason

    def test_hardware_counters_error_is_none_without_the_key(self, timing):
        assert timing.hardware_counters_error is None

    @pytest.mark.parametrize("field", ["process_start_epoch_ns", "measurement_overhead_ns"])
    def test_negative_sentinels_read_as_unknown(self, field):
        assert getattr(ComponentTiming.from_json(_raw(**{field: -1})), field) is None

    @pytest.mark.parametrize(
        "key", ["components", "event_numbers", "execute", "lifecycle", "phases"]
    )
    def test_missing_key_is_refused(self, key):
        raw = _raw()
        del raw[key]
        with pytest.raises(ValueError, match="missing keys"):
            ComponentTiming.from_json(raw, source="x_components.json")

    def test_execute_row_of_wrong_width_is_refused(self):
        raw = _raw()
        raw["execute"]["cpu_s"][1] = [0.1, 0.2]
        with pytest.raises(ValueError, match="execute.cpu_s is not 3 events × 4 components"):
            ComponentTiming.from_json(raw)

    def test_execute_with_wrong_event_count_is_refused(self):
        with pytest.raises(ValueError, match="execute.wall_s"):
            ComponentTiming.from_json(_raw(event_numbers=[0, 1]))

    @pytest.mark.parametrize("metric", COMPONENT_METRICS)
    def test_execute_without_a_metric_is_refused(self, metric):
        raw = _raw()
        del raw["execute"][metric]
        with pytest.raises(ValueError, match=f"execute is missing metrics: \\['{metric}'\\]"):
            ComponentTiming.from_json(raw)

    @pytest.mark.parametrize("metric", COMPONENT_METRICS)
    def test_lifecycle_phase_without_a_metric_is_refused(self, metric):
        raw = _raw()
        del raw["lifecycle"]["initialize"][metric]
        with pytest.raises(
            ValueError, match=f"lifecycle.initialize is missing metrics: \\['{metric}'\\]"
        ):
            ComponentTiming.from_json(raw)

    def test_lifecycle_of_wrong_length_is_refused(self):
        raw = _raw()
        raw["lifecycle"]["initialize"]["wall_s"] = [1.0]
        with pytest.raises(ValueError, match="lifecycle.initialize"):
            ComponentTiming.from_json(raw)

    def test_parent_index_out_of_range_is_refused(self):
        raw = _raw()
        raw["components"][2]["parent"] = 9
        with pytest.raises(ValueError, match="parent index 9 out of range"):
            ComponentTiming.from_json(raw)

    def test_component_that_is_its_own_parent_is_refused(self):
        raw = _raw()
        raw["components"][1]["parent"] = 1
        with pytest.raises(ValueError, match="'Top' is its own ancestor"):
            ComponentTiming.from_json(raw)

    def test_parent_cycle_is_refused(self):
        raw = _raw()
        raw["components"][1]["parent"] = 2
        with pytest.raises(ValueError, match="is its own ancestor"):
            ComponentTiming.from_json(raw)

    def test_duplicate_component_names_are_refused(self):
        raw = _raw()
        raw["components"][3]["name"] = "Fast"
        with pytest.raises(ValueError, match="not unique"):
            ComponentTiming.from_json(raw)

    def test_duplicate_event_numbers_are_refused(self):
        with pytest.raises(ValueError, match="duplicates"):
            ComponentTiming.from_json(_raw(event_numbers=[0, 0, 1]))


class TestTree:
    def test_children_and_descendants(self, timing):
        assert timing.children("Top") == ["Fast", "Slow"]
        assert timing.descendants("Top") == ["Fast", "Slow"]
        assert timing.descendants("Fast") == []

    def test_inclusive_adds_descendants_to_own_cost(self, timing):
        inclusive = timing.inclusive("wall_s")
        assert inclusive["Top"].tolist() == pytest.approx([1.71, 1.11, 0.11])
        assert inclusive["Fast"].tolist() == pytest.approx([0.2, 0.1, 0.1])

    def test_inclusive_keeps_nan_where_the_component_did_not_run(self, timing):
        inclusive = timing.inclusive("wall_s")
        assert math.isnan(inclusive.loc[2, "Slow"])
        assert inclusive["Geo"].isna().all()


def _with_optional_metrics(raw: dict) -> dict:
    """*raw* as the auditor writes it with every optional metric: the thread's
    faults and context switches, hardware counts (unreadable once, for ``Slow``
    in event 1) and allocations (``Slow`` frees more than it allocates in event
    0)."""
    raw["execute"] |= {
        "minor_page_faults": [[None, 0, 12, 300], [None, 0, 0, 2], [None, 0, 0, None]],
        "major_page_faults": [[None, 0, 0, 7], [None, 0, 0, 0], [None, 0, 0, None]],
        "voluntary_context_switches": [[None, 0, 0, 3], [None, 0, 0, 0], [None, 0, 0, None]],
        "involuntary_context_switches": [[None, 0, 1, 2], [None, 0, 0, 1], [None, 0, 0, None]],
        "instructions": [[None, 900, 4e8, 3e9], [None, 900, 2e8, None], [None, 900, 2e8, None]],
        "cycles": [[None, 800, 2e8, 1e9], [None, 800, 1e8, None], [None, 800, 1e8, None]],
        "allocations": [[None, 0, 40, 9], [None, 0, 40, 9], [None, 0, 40, None]],
        "allocated_bytes": [[None, 0, 4096, 512], [None, 0, 4096, 512], [None, 0, 4096, None]],
        "net_allocated_bytes": [[None, 0, 640, -128], [None, 0, 640, 0], [None, 0, 640, None]],
        # Inclusive: the sequencer held at least what each child did.
        "peak_heap_bytes": [
            [None, 1024, 1024, 384],
            [None, 1024, 1024, 256],
            [None, 1024, 1024, None],
        ],
        "largest_allocation_bytes": [
            [None, 512, 512, 256],
            [None, 512, 512, 256],
            [None, 512, 512, None],
        ],
    }
    raw["lifecycle"]["initialize"] |= {
        "minor_page_faults": [90_000, 0, 500, 20],
        "major_page_faults": [1200, 0, 3, 0],
        "voluntary_context_switches": [40, 0, 1, 0],
        "involuntary_context_switches": [5, 0, 0, 0],
        "instructions": [5e9, 1e4, 1e8, 1e6],
        "cycles": [4e9, 1e4, 9e7, 1e6],
        "allocations": [12000, 0, 300, 5],
        "allocated_bytes": [400_000_000, 0, 65536, 256],
        "net_allocated_bytes": [350_000_000, 0, 32768, 256],
        "peak_heap_bytes": [380_000_000, 65536, 65536, 256],
        "largest_allocation_bytes": [200_000_000, 32768, 32768, 256],
    }
    return raw


OPTIONAL_METRICS = {*THREAD_METRICS, *HARDWARE_METRICS, *ALLOCATION_METRICS}


class TestOptionalMetrics:
    def test_read_like_every_other_metric(self):
        timing = ComponentTiming.from_json(_with_optional_metrics(_raw()))
        assert set(timing.execute) == {*COMPONENT_METRICS, *OPTIONAL_METRICS}
        assert timing.execute["net_allocated_bytes"].loc[0, "Slow"] == -128
        assert timing.execute["major_page_faults"].loc[0, "Slow"] == 7
        assert math.isnan(timing.execute["allocations"].loc[2, "Slow"])
        assert timing.lifecycle["initialize"].loc["Geo", "allocations"] == 12000

    def test_unreadable_hardware_counts_are_nan_although_the_component_ran(self):
        timing = ComponentTiming.from_json(_with_optional_metrics(_raw()))
        assert timing.execute["wall_s"].loc[1, "Slow"] == 1.0
        assert math.isnan(timing.execute["instructions"].loc[1, "Slow"])

    def test_absent_from_files_without_them(self, timing):
        assert not OPTIONAL_METRICS & set(timing.execute)
        assert not OPTIONAL_METRICS & set(timing.lifecycle["initialize"])

    @pytest.mark.parametrize("metric", ["allocations", "major_page_faults"])
    def test_inclusive_adds_descendants_self_costs(self, metric):
        timing = ComponentTiming.from_json(_with_optional_metrics(_raw()))
        execute = timing.execute[metric]
        expected = execute[["Top", "Fast", "Slow"]].fillna(0).sum(axis=1)
        assert timing.inclusive(metric)["Top"].tolist() == expected.tolist()

    def test_inclusive_is_nan_where_a_descendant_ran_without_a_value(self):
        timing = ComponentTiming.from_json(_with_optional_metrics(_raw()))
        inclusive = timing.inclusive("instructions")
        assert math.isnan(inclusive.loc[1, "Top"])
        assert inclusive.loc[1, "Fast"] == 2e8
        assert inclusive.loc[2, "Top"] == 900 + 2e8

    @pytest.mark.parametrize("metric", INCLUSIVE_METRICS)
    def test_inclusive_refuses_what_already_includes_nested_calls(self, metric):
        timing = ComponentTiming.from_json(_with_optional_metrics(_raw()))
        with pytest.raises(ValueError, match=f"{metric} already includes nested calls"):
            timing.inclusive(metric)


def test_phase_seconds_adds_configure_from_process_start(timing):
    assert timing.phase_seconds() == {"initialize": 3.0, "execute": 6.0, "configure": 2.0}


def test_phase_seconds_without_process_start_has_no_configure():
    timing = ComponentTiming.from_json(_raw(process_start_epoch_ns=-1))
    assert "configure" not in timing.phase_seconds()


class TestLoadComponentTiming:
    def _write(self, directory: Path, label: str, raw: dict) -> None:
        (directory / f"{label}_components.json").write_text(json.dumps(raw))

    def test_loads_every_label(self, tmp_path):
        self._write(tmp_path, "default", _raw())
        self._write(tmp_path, "truthTracking", _raw())
        loaded = load_component_timing(tmp_path)
        assert sorted(loaded) == ["default", "truthTracking"]
        assert isinstance(loaded["default"], ComponentTiming)

    def test_directory_without_component_files_is_empty(self, tmp_path):
        assert load_component_timing(tmp_path) == {}

    def test_requested_label_without_a_file_is_refused(self, tmp_path):
        self._write(tmp_path, "default", _raw())
        with pytest.raises(ValueError, match="Missing component files for labels: \\['native'\\]"):
            load_component_timing(tmp_path, labels=["default", "native"])

    def test_future_schema_is_refused_with_the_file_named(self, tmp_path):
        self._write(tmp_path, "default", _raw(schema_version=2))
        with pytest.raises(ValueError, match="default_components.json: unsupported future"):
            load_component_timing(tmp_path)


def _short_matrix(path: Path) -> None:
    raw = _raw()
    raw["execute"]["wall_s"] = raw["execute"]["wall_s"][:-1]
    path.write_text(json.dumps(raw))


#: A component file broken each way a run can leave one, and what its reason names.
_BROKEN = {
    "malformed JSON": (lambda path: path.write_text("{"), "JSONDecodeError: Expecting"),
    "future schema": (
        lambda path: path.write_text(json.dumps(_raw(schema_version=2))),
        "ValueError: unsupported future schema_version 2",
    ),
    "matrix shape": (_short_matrix, "ValueError: execute.wall_s is not"),
    "unreadable file": (lambda path: path.mkdir(), "IsADirectoryError: Is a directory"),
}


class TestLoadEachComponentTiming:
    @pytest.mark.parametrize("broken", list(_BROKEN))
    def test_a_broken_file_hides_only_its_own_configuration(self, tmp_path, broken):
        (tmp_path / "default_components.json").write_text(json.dumps(_raw()))
        corrupt, reason = _BROKEN[broken]
        corrupt(tmp_path / "native_components.json")
        loaded, unreadable = load_each_component_timing(tmp_path)
        assert list(loaded) == ["default"]
        assert isinstance(loaded["default"], ComponentTiming)
        assert list(unreadable) == ["native"]
        # The file by its name, never by where the run was cached.
        assert unreadable["native"].startswith(f"native_components.json: {reason}")
        assert str(tmp_path) not in unreadable["native"]

    def test_a_requested_broken_file_still_raises(self, tmp_path):
        (tmp_path / "native_components.json").write_text("{")
        with pytest.raises(ValueError):
            load_component_timing(tmp_path, labels=["native"])

    def test_an_unexpected_error_is_logged_and_hides_only_its_configuration(
        self, tmp_path, caplog,
    ):
        (tmp_path / "default_components.json").write_text(json.dumps(_raw()))
        raw = _raw()
        del raw["components"][0]["name"]
        (tmp_path / "native_components.json").write_text(json.dumps(raw))
        loaded, unreadable = load_each_component_timing(tmp_path)
        assert list(loaded) == ["default"]
        assert unreadable == {"native": "native_components.json: unexpected error (KeyError)"}
        (record,) = [r for r in caplog.records if r.levelname == "ERROR"]
        assert "native_components.json" in record.getMessage()
        assert record.exc_info is not None

    def test_a_directory_without_component_files_has_nothing_unreadable(self, tmp_path):
        assert load_each_component_timing(tmp_path) == ({}, {})


def test_sequencers_are_the_components_others_are_nested_in(timing):
    assert timing.sequencers() == ["Top"]


def test_algorithms_are_those_doing_work_of_their_own(timing):
    # Geo is a service and Top a sequencer.
    assert timing.algorithms() == ["Fast", "Slow"]


def test_phase_component_seconds_sums_the_event_loop_over_its_events(timing):
    loop = timing.phase_component_seconds("execute")
    # Slow did not run in event 2, which costs it nothing.
    assert loop["Slow"] == pytest.approx(2.5)
    assert loop["Geo"] == 0.0
    assert timing.phase_component_seconds("initialize")["Geo"] == 2.0
    assert timing.phase_component_seconds("finalize").empty


def _wall(columns: dict[str, list], events: list[int]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.Index(events, name="event_number"), dtype=float)


def test_event_loop_time_tells_the_share_of_every_event_from_the_time_when_run():
    # Event 0 warms up, a hundred times slower; Filtered runs in event 1 only.
    wall = _wall({"Always": [0.1, 1e-3, 1e-3, 1e-3], "Filtered": [0.6, 6e-3, None, None]},
                 [0, 1, 2, 3])
    time = event_loop_time(wall)
    filtered, always = time.loc["Filtered"], time.loc["Always"]
    assert filtered["events"] == 3
    assert filtered["executed_events"] == 1
    assert filtered["coverage"] == pytest.approx(1 / 3)
    assert filtered["seconds"] == pytest.approx(6e-3)
    assert filtered["seconds_per_event"] == pytest.approx(2e-3)
    assert filtered["seconds_when_run"] == pytest.approx(6e-3)
    # Run in every event, the two agree.
    assert always["seconds_per_event"] == pytest.approx(always["seconds_when_run"])
    assert always["coverage"] == 1.0


def test_event_loop_time_leaves_out_the_warm_up_by_its_number_not_its_row():
    # Events 5, 7, 8 (a gap at 6): no event 0, so the first is kept.
    wall = _wall({"Algo": [3.0, 1.0, 2.0]}, [5, 7, 8])
    time = event_loop_time(wall)
    assert time.loc["Algo", "events"] == 3
    assert time.loc["Algo", "seconds_per_event"] == pytest.approx(2.0)
    # And the warm-up counts when asked to.
    with_warm_up = event_loop_time(_wall({"Algo": [9.0, 1.0]}, [0, 1]), warm_up=None)
    assert with_warm_up.loc["Algo", "seconds_per_event"] == pytest.approx(5.0)


def test_event_loop_time_has_no_time_without_an_event_run_in():
    wall = _wall({"WarmUpOnly": [1.0, None], "Never": [None, None], "Free": [0.0, 0.0]}, [0, 1])
    time = event_loop_time(wall)
    # Run only in the warm-up, or never: unknown, not free.
    for name in ("WarmUpOnly", "Never"):
        assert time.loc[name, "executed_events"] == 0
        assert np.isnan(time.loc[name, ["seconds", "seconds_per_event",
                                        "seconds_when_run"]].astype(float)).all()
    # A measured zero stays zero.
    assert time.loc["Free", "seconds_per_event"] == 0.0
    assert time.loc["Free", "seconds_when_run"] == 0.0


@pytest.mark.parametrize("events", [[0], []])
def test_event_loop_time_without_an_event_past_the_warm_up(events):
    time = event_loop_time(_wall({"Algo": [1.0] * len(events)}, events))
    assert time.loc["Algo", "events"] == 0
    assert np.isnan(time.loc["Algo", ["coverage", "seconds_per_event",
                                      "seconds_when_run"]].astype(float)).all()
