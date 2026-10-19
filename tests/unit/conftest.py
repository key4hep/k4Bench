"""Fixtures shared by the unit tests."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"

_MIB = 1024**2


def write_reco_run_with_heap(parent: Path, name: str = "run", date: str | None = None) -> Path:
    """The ``reco_run`` fixture run in *parent*/*name* with a heap file for its
    baseline, in MB: no component keeps 49, the geometry 397 and the reader
    150; LumiCalReco gains 2 per event and VertexFinder 1.2, and both keep it
    after finalize; the writer gains 1.5 per event and frees it at the end;
    the reader needs 5 in each event. With *date*, the run is that night's:
    its run date, its nightly tag and the release named after it."""
    run = parent / name
    shutil.copytree(RECO_RUN, run)
    info = {**json.loads((RECO_RUN / "run_info.json").read_text()), "suite": "reco"}
    if date is not None:
        info.update(date=date, k4h_release=f"key4hep-{date}", k4h_release_date=date)
    (run / "run_info.json").write_text(json.dumps(info))
    timing = json.loads((RECO_RUN / "baseline_components.json").read_text())
    names = ["(no component)", *(c["name"] for c in timing["components"])]
    events = timing["event_numbers"]

    def held(event: int | None) -> list[int]:
        values = dict.fromkeys(names, 0.0)
        values.update({"(no component)": 49, "GeoSvc": 397, "k4FWCore__Reader": 150})
        if event is not None:
            values.update({
                "LumiCalReco": 2.0 * (event + 1), "VertexFinder": 1.2 * (event + 1),
                "k4FWCore__Writer": 1.5 * (event + 1),
            })
        return [int(values[n] * _MIB) for n in names]

    def peak(event: int) -> list[int]:
        values = held(event)
        values[names.index("k4FWCore__Reader")] += 5 * _MIB
        return values

    final = dict(zip(names, held(events[-1])))
    final.update({"GeoSvc": 0, "k4FWCore__Reader": 0, "k4FWCore__Writer": 0})
    (run / "baseline_heap.json").write_text(json.dumps({
        "schema_version": 1,
        "producer": "k4BenchAuditor",
        "components": names,
        "before_first_event": {
            "held_bytes": held(None), "peak_held_bytes": held(None),
            "malloc_in_use_bytes": sum(held(None)), "rss_anon_mb": 700.0,
        },
        "event_numbers": events,
        "execute": {
            "start_held_bytes": [held(e - 1) if e else held(None) for e in events],
            "held_bytes": [held(e) for e in events],
            "peak_held_bytes": [peak(e) for e in events],
            "malloc_in_use_bytes": [sum(held(e)) for e in events],
            "rss_anon_mb": [700.0] * len(events),
        },
        "at_end": {
            "held_bytes": [final[n] for n in names],
            "malloc_in_use_bytes": sum(final.values()), "rss_anon_mb": 500.0,
        },
        "table": {"peak_entries": 1000, "mapped_bytes": 1 << 20},
    }))
    return run


@pytest.fixture
def reco_run_with_heap(tmp_path) -> Callable[..., Path]:
    """:func:`write_reco_run_with_heap` in ``tmp_path``: called with a
    directory name (``"run"`` by default) and, optionally, a night's date."""
    def make(name: str = "run", date: str | None = None) -> Path:
        return write_reco_run_with_heap(tmp_path, name, date)
    return make
