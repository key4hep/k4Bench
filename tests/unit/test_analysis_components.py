"""Unit tests for the k4BenchAuditor's per-component files
(:mod:`k4bench.analysis.components` and ``load_component_timing``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from k4bench.analysis.components import COMPONENT_METRICS, ComponentTiming
from k4bench.analysis.loader import load_component_timing

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
