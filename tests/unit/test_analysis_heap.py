"""Unit tests for the memory pass's heap files
(:mod:`k4bench.analysis.heap` and ``load_component_heap``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from k4bench.analysis.heap import (
    EVENT_HEAP_METRICS,
    HEAP_METRICS,
    NO_COMPONENT,
    PROCESS_METRICS,
    ComponentHeap,
)
from k4bench.analysis.loader import load_component_heap

MB = 1_000_000


def _raw(**overrides) -> dict:
    """A job whose geometry service keeps 300 MB from initialize, whose reader
    holds each event's data until the writer frees it, and whose sequencer
    holds nothing. The anonymous RSS could not be read after event 1."""
    raw = {
        "schema_version": 1,
        "producer": "k4BenchAuditor",
        "components": [NO_COMPONENT, "Geo", "Top", "Reader"],
        "before_first_event": {
            "held_bytes": [50 * MB, 300 * MB, 0, 1 * MB],
            "peak_held_bytes": [60 * MB, 310 * MB, 0, 1 * MB],
            "malloc_in_use_bytes": 352 * MB,
            "rss_anon_mb": 420.0,
        },
        "event_numbers": [0, 1],
        "execute": {
            "start_held_bytes": [[50 * MB, 300 * MB, 0, 1 * MB], [50 * MB, 300 * MB, 0, 2 * MB]],
            "held_bytes": [[50 * MB, 300 * MB, 0, 5 * MB], [50 * MB, 300 * MB, 0, 6 * MB]],
            "peak_held_bytes": [[50 * MB, 300 * MB, 0, 8 * MB], [51 * MB, 300 * MB, 0, 9 * MB]],
            "malloc_in_use_bytes": [356 * MB, 357 * MB],
            "rss_anon_mb": [430.0, -1.0],
        },
        "at_end": {"held_bytes": [40 * MB, 0, 0, 1 * MB], "malloc_in_use_bytes": 42 * MB, "rss_anon_mb": 400.0},
        "table": {"peak_entries": 2_000_000, "mapped_bytes": 64 << 20},
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def heap() -> ComponentHeap:
    return ComponentHeap.from_json(_raw())


def test_each_event_is_a_row_and_each_component_a_column(heap):
    assert set(heap.execute) == set(EVENT_HEAP_METRICS)
    held = heap.execute["held_bytes"]
    assert held.index.tolist() == [0, 1] and held.index.name == "event_number"
    assert held.columns.tolist() == [NO_COMPONENT, "Geo", "Top", "Reader"]
    assert held.loc[1, "Reader"] == 6 * MB
    assert heap.execute["peak_held_bytes"].loc[1, NO_COMPONENT] == 51 * MB


def test_the_working_set_is_the_peak_beyond_what_the_event_started_with(heap):
    working = heap.working_set()
    assert working.loc[0].tolist() == [0, 0, 0, 7 * MB]
    assert working.loc[1, "Reader"] == 7 * MB
    assert working.loc[1, NO_COMPONENT] == 1 * MB


def test_a_job_that_ran_events_at_once_has_no_peaks_per_event():
    execute = {**_raw()["execute"], "peak_held_bytes": [[None] * 4, [None] * 4]}
    heap = ComponentHeap.from_json(_raw(execute=execute))
    assert heap.execute["peak_held_bytes"].isna().all().all()
    assert heap.working_set().isna().all().all()
    assert heap.execute["held_bytes"].loc[1, "Reader"] == 6 * MB


def test_before_the_first_event_and_at_the_end_are_components_by_metric(heap):
    assert heap.before_first_event.columns.tolist() == list(HEAP_METRICS)
    assert heap.before_first_event.loc["Geo"].tolist() == [300 * MB, 310 * MB]
    assert heap.at_end.columns.tolist() == ["held_bytes"]
    assert heap.at_end["held_bytes"].tolist() == [40 * MB, 0, 0, 1 * MB]


def test_the_process_heap_is_read_at_the_same_moments(heap):
    assert heap.process.columns.tolist() == list(PROCESS_METRICS)
    assert heap.process["malloc_in_use_bytes"].tolist() == [356 * MB, 357 * MB]
    assert heap.process_before_first_event.to_dict() == {"malloc_in_use_bytes": 352 * MB, "rss_anon_mb": 420.0}
    assert heap.process_at_end["malloc_in_use_bytes"] == 42 * MB
    assert (heap.table_peak_entries, heap.table_mapped_bytes) == (2_000_000, 64 << 20)


def test_an_rss_that_could_not_be_read_is_missing(heap):
    assert heap.process.loc[0, "rss_anon_mb"] == 430.0
    assert math.isnan(heap.process.loc[1, "rss_anon_mb"])
    failed = ComponentHeap.from_json(_raw(at_end={**_raw()["at_end"], "rss_anon_mb": -1.0}))
    assert math.isnan(failed.process_at_end["rss_anon_mb"])
    assert failed.process_at_end["malloc_in_use_bytes"] == 42 * MB


def test_a_job_that_ran_no_event_has_nothing_before_it():
    heap = ComponentHeap.from_json(_raw(
        before_first_event=None,
        event_numbers=[],
        execute={key: [] for key in (*EVENT_HEAP_METRICS, *PROCESS_METRICS)},
    ))
    assert heap.execute["held_bytes"].empty
    assert heap.before_first_event.index.tolist() == [NO_COMPONENT, "Geo", "Top", "Reader"]
    assert heap.before_first_event.isna().all().all()
    assert heap.process_before_first_event.isna().all()
    assert heap.at_end.loc[NO_COMPONENT, "held_bytes"] == 40 * MB


def test_the_holders_held_heap_at_some_moment_the_warm_up_events_included(heap):
    # The sequencer never holds anything.
    assert heap.holders().tolist() == [NO_COMPONENT, "Geo", "Reader"]
    execute = _raw()["execute"]
    execute["peak_held_bytes"][0][2] = 500 * MB
    spiked = ComponentHeap.from_json(_raw(execute=execute))
    assert spiked.holders().tolist() == [NO_COMPONENT, "Geo", "Top", "Reader"]


# ── Summaries ────────────────────────────────────────────────────────────────

MIB = 1024**2

#: The components of :func:`_job`.
_JOB = [NO_COMPONENT, "Geo", "Leaky", "Writer", "Arena", "Busy"]


def _job(n_events: int = 20) -> ComponentHeap:
    """A job over *n_events* events, in MB: no component keeps 50 and the
    geometry 300 throughout; a leak gains 2 per event and keeps it after
    finalize; the writer gains 2 per event and frees it at the end; an arena
    grows by 500 once, at event 3, and keeps it; a busy component needs 10
    in each event (100 in the warm-up event) and frees it before the event
    ends."""
    events = list(range(n_events))

    def end(e):
        return [50, 300, 2 * e, 2 * e, 500 if e >= 3 else 0, 0]

    def start(e):
        return end(e - 1) if e else [50, 300, 0, 0, 0, 0]

    def peak(e):
        return [max(a, b) for a, b in zip(start(e), end(e))][:5] + [100 if e == 0 else 10]

    def mib(rows):
        return [[v * MIB for v in row] for row in rows]

    last = end(events[-1]) if events else [0] * 6
    return ComponentHeap.from_json({
        "schema_version": 1,
        "producer": "k4BenchAuditor",
        "components": _JOB,
        "before_first_event": {
            "held_bytes": [v * MIB for v in start(0)],
            "peak_held_bytes": [v * MIB for v in start(0)],
            "malloc_in_use_bytes": 350 * MIB,
            "rss_anon_mb": 400.0,
        } if events else None,
        "event_numbers": events,
        "execute": {
            "start_held_bytes": mib(start(e) for e in events),
            "held_bytes": mib(end(e) for e in events),
            "peak_held_bytes": mib(peak(e) for e in events),
            "malloc_in_use_bytes": [sum(end(e)) * MIB for e in events],
            "rss_anon_mb": [400.0] * n_events,
        },
        "at_end": {
            "held_bytes": [v * MIB for v in (40, 0, last[2], 0, last[4], 0)],
            "malloc_in_use_bytes": 0,
            "rss_anon_mb": 300.0,
        },
        "table": {"peak_entries": 1, "mapped_bytes": 1},
    })


def test_the_summary_gives_each_component_its_heap_over_the_job():
    summary = _job().summary()
    assert summary.index.tolist() == _JOB
    geo = summary.loc["Geo"]
    assert (geo.before_first_mb, geo.typical_mb, geo.largest_mb, geo.after_finalize_mb) == (300, 300, 300, 0)
    assert geo.growth_mb_per_event == 0
    # Events 1..19: the warm-up event is left out.
    leaky = summary.loc["Leaky"]
    assert (leaky.typical_mb, leaky.largest_mb) == (20, 38)
    assert leaky.growth_mb_per_event == pytest.approx(2.0)


def test_the_working_set_leaves_the_warm_up_event_out():
    busy = _job().summary().loc["Busy"]
    assert (busy.working_set_median_mb, busy.working_set_p95_mb, busy.working_set_max_mb) == (10, 10, 10)
    assert _job().summary(warm_up=None).loc["Busy", "working_set_max_mb"] == 100


def test_growth_kept_after_finalize_is_suspected_of_leaking():
    summary = _job().summary()
    assert summary["leak"].to_dict() == {
        NO_COMPONENT: False, "Geo": False,
        # 2 MB per event over 18 events, all of it kept.
        "Leaky": True,
        # Grows as fast, but frees it at the end.
        "Writer": False,
        # One step: the slope between most pairs of events is nought.
        "Arena": False, "Busy": False,
    }
    assert summary.loc["Arena", "growth_mb_per_event"] == 0


def _one(held_mb: list[float], after_mb: float, before_mb: float = 0.0) -> ComponentHeap:
    """A job of one component holding *held_mb* at its events' ends (the
    first being the warm-up), *before_mb* before them and *after_mb* after
    finalize."""
    n = len(held_mb)
    ends = [[0, round(v * MIB)] for v in held_mb]
    starts = [[0, round(before_mb * MIB)], *ends[:-1]]
    return ComponentHeap.from_json({
        "schema_version": 1,
        "producer": "k4BenchAuditor",
        "components": [NO_COMPONENT, "Algo"],
        "before_first_event": {
            "held_bytes": [0, round(before_mb * MIB)], "peak_held_bytes": [0, round(before_mb * MIB)],
            "malloc_in_use_bytes": 0, "rss_anon_mb": 0.0,
        },
        "event_numbers": list(range(n)),
        "execute": {
            "start_held_bytes": starts,
            "held_bytes": ends,
            "peak_held_bytes": [[max(a, b) for a, b in zip(s, e)] for s, e in zip(starts, ends)],
            "malloc_in_use_bytes": [0] * n,
            "rss_anon_mb": [0.0] * n,
        },
        "at_end": {"held_bytes": [0, round(after_mb * MIB)], "malloc_in_use_bytes": 0, "rss_anon_mb": 0.0},
        "table": {"peak_entries": 1, "mapped_bytes": 1},
    })


def test_growth_freed_at_finalize_is_not_suspected_however_much_was_kept_from_the_start():
    # 1 GB from initialize, 2 MB more per event, the gain freed at finalize.
    algo = _one([1000 + 2 * e for e in range(21)], after_mb=1000, before_mb=1000).summary().loc["Algo"]
    assert algo.growth_mb_per_event == pytest.approx(2.0)
    assert not algo.leak
    # Keeping the gain too is suspected.
    assert _one([1000 + 2 * e for e in range(21)], after_mb=1040, before_mb=1000).summary().loc["Algo", "leak"]


def test_growth_freed_in_the_last_event_is_not_suspected():
    # 1 MB more per event, all but 1 MB of it freed in the last event and
    # nothing more at finalize: the slope still reads 1 MB per event.
    held = [10.0 + e for e in range(100)] + [12.0]
    algo = _one(held, after_mb=12).summary().loc["Algo"]
    assert algo.growth_mb_per_event == pytest.approx(1.0, rel=0.05)
    assert not algo.leak


def test_one_step_halfway_through_is_not_growth():
    # 50 MB more from event 10 on, kept after finalize: a table filled once.
    algo = _one([100.0] * 10 + [150.0] * 11, after_mb=150).summary().loc["Algo"]
    assert not algo.leak


def test_growth_that_slows_down_is_still_suspected():
    # A cache filling as new cells are seen: fast at first, then slower.
    held = [40 * (1 - np.exp(-e / 60)) * 3 for e in range(101)]
    assert _one(held, after_mb=held[-1]).summary().loc["Algo", "leak"]


def test_growth_too_small_to_tell_from_the_heap_moving_is_not_suspected():
    # 2 MB per event over 4 events is 8 MB, below LEAK_MIN_MB.
    summary = _job(n_events=6).summary()
    assert summary.loc["Leaky", "growth_mb_per_event"] == pytest.approx(2.0)
    assert not summary.loc["Leaky", "leak"]


def test_too_few_events_have_no_growth_and_no_suspect():
    summary = _job(n_events=3).summary()
    # What moved has no slope; what held still grew by nothing.
    assert summary.loc[["Leaky", "Writer"], "growth_mb_per_event"].isna().all()
    assert summary.loc["Geo", "growth_mb_per_event"] == 0
    assert not summary["leak"].any()


def test_the_totals_are_the_whole_jobs_heap():
    totals = _job().totals()
    assert totals.before_first_mb == 350
    assert totals.per_event_mb.index.tolist() == list(range(20))
    assert totals.per_event_mb[19] == 50 + 300 + 38 + 38 + 500
    assert totals.typical_mb == 50 + 300 + 20 + 20 + 500
    assert (totals.largest_mb, totals.largest_event) == (926, 19)
    assert totals.growth_mb_per_event == pytest.approx(4.0)
    assert totals.after_finalize_mb == 40 + 38 + 500


def test_a_job_without_events_has_totals_but_no_statistics():
    totals = _job(n_events=0).totals()
    assert math.isnan(totals.before_first_mb) and math.isnan(totals.typical_mb)
    assert totals.largest_event is None and math.isnan(totals.growth_mb_per_event)
    assert totals.after_finalize_mb == 40


class TestInconsistentFilesAreRefused:
    def test_missing_keys(self):
        raw = _raw()
        del raw["table"]
        with pytest.raises(ValueError, match="missing keys: \\['table'\\]"):
            ComponentHeap.from_json(raw)

    def test_components_not_led_by_what_no_component_holds(self):
        with pytest.raises(ValueError, match="must start with"):
            ComponentHeap.from_json(_raw(components=["Geo", NO_COMPONENT, "Top", "Reader"]))

    def test_component_names_that_are_not_unique(self):
        with pytest.raises(ValueError, match="not unique"):
            ComponentHeap.from_json(_raw(components=[NO_COMPONENT, "Geo", "Geo", "Reader"]))

    def test_duplicate_events(self):
        execute = _raw()["execute"]
        with pytest.raises(ValueError, match="duplicates"):
            ComponentHeap.from_json(_raw(event_numbers=[0, 0], execute=execute))

    def test_a_table_that_is_not_events_by_components(self):
        execute = {**_raw()["execute"], "held_bytes": [[1, 2, 3, 4], [1, 2, 3]]}
        with pytest.raises(ValueError, match="execute.held_bytes is not 2 events × 4 components"):
            ComponentHeap.from_json(_raw(execute=execute))

    @pytest.mark.parametrize("metric", [*EVENT_HEAP_METRICS, *PROCESS_METRICS])
    def test_a_missing_metric(self, metric):
        execute = {key: value for key, value in _raw()["execute"].items() if key != metric}
        with pytest.raises(ValueError, match=f"execute is missing \\['{metric}'\\]"):
            ComponentHeap.from_json(_raw(execute=execute))

    def test_a_process_series_without_one_value_per_event(self):
        execute = {**_raw()["execute"], "malloc_in_use_bytes": [1]}
        with pytest.raises(ValueError, match="execute.malloc_in_use_bytes does not have one value per event"):
            ComponentHeap.from_json(_raw(execute=execute))

    def test_a_moment_missing_a_metric(self):
        before = {key: value for key, value in _raw()["before_first_event"].items() if key != "peak_held_bytes"}
        with pytest.raises(ValueError, match="before_first_event is missing \\['peak_held_bytes'\\]"):
            ComponentHeap.from_json(_raw(before_first_event=before))

    def test_a_moment_without_one_value_per_component(self):
        at_end = {**_raw()["at_end"], "held_bytes": [1, 2]}
        with pytest.raises(ValueError, match="at_end does not have one value per component"):
            ComponentHeap.from_json(_raw(at_end=at_end))


class TestLoadComponentHeap:
    def _write(self, directory: Path, label: str, raw: dict) -> None:
        (directory / f"{label}_heap.json").write_text(json.dumps(raw))

    def test_loads_every_label(self, tmp_path):
        self._write(tmp_path, "baseline", _raw())
        self._write(tmp_path, "variant_truth_tracking", _raw())
        loaded = load_component_heap(tmp_path)
        assert sorted(loaded) == ["baseline", "variant_truth_tracking"]
        assert isinstance(loaded["baseline"], ComponentHeap)

    def test_directory_without_heap_files_is_empty(self, tmp_path):
        (tmp_path / "baseline_components.json").write_text("{}")
        assert load_component_heap(tmp_path) == {}

    def test_requested_label_without_a_file_is_refused(self, tmp_path):
        self._write(tmp_path, "baseline", _raw())
        with pytest.raises(ValueError, match="Missing heap files for labels: \\['variant_native'\\]"):
            load_component_heap(tmp_path, labels=["baseline", "variant_native"])

    def test_future_schema_is_refused_with_the_file_named(self, tmp_path):
        self._write(tmp_path, "baseline", _raw(schema_version=2))
        with pytest.raises(ValueError, match="baseline_heap.json: unsupported future"):
            load_component_heap(tmp_path)
