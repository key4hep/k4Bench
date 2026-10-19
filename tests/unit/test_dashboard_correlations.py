"""The Correlations section, rendered on the Reconstruction page from the
``reco_run`` fixture with an event source added, and on the Simulation page
from a ddsim run."""

from __future__ import annotations

import base64
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from k4bench.analysis.loader import load_component_timing  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"

#: The fixture's auditor files hold 12 events (its events files 100).
_EVENTS = 12


def _local_env(monkeypatch, run_dir: Path, cache_dir: Path) -> None:
    monkeypatch.delenv("K4BENCH_DATA_URL", raising=False)
    monkeypatch.setenv("K4BENCH_DATA_DIR", str(run_dir))
    monkeypatch.setenv("K4BENCH_CACHE_DIR", str(cache_dir))


def _reco_app(dashboard_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from reco import page

    page.main()


def _write_source(
    run: Path, events: int = _EVENTS, origin: str = "recorded", follow: str = "VXDBarrelDigitiser",
) -> list[float]:
    """An event source whose input's vertex hits follow the time of
    algorithm *follow* event by event, and whose events hold 1 to 4 charged
    pions; that algorithm's times."""
    wall = load_component_timing(RECO_RUN)["baseline"].execute["wall_s"][follow]
    times = wall.iloc[:events].tolist()
    ranks = {t: i for i, t in enumerate(sorted(times))}
    hits = [100 + 10 * ranks[t] for t in times]
    per_event = [i % 4 + 1 for i in range(events)]
    n = sum(per_event)
    record = {
        "schema_version": 1, "origin": origin,
        "source": {"file": "sim.edm4hep.root", "first_entry": 0, "events": events},
        "generator": {
            "offsets": [0, *(sum(per_event[:i + 1]) for i in range(events))],
            "pdg": [211] * n, "gen_status": [1] * n, "charge": [1.0] * n, "mass": [0.13957] * n,
            "px": [1.0] * n, "py": [0.0] * n, "pz": [0.5] * n,
            "vx": [0.0] * n, "vy": [0.0] * n, "vz": [0.0] * n,
        },
        "input_content": {"collections": [
            {"name": "VertexBarrelCollection", "type": "edm4hep::SimTrackerHitCollection",
             "entries": hits},
        ]},
    }
    (run / "event_source.json.gz").write_bytes(gzip.compress(json.dumps(record).encode()))
    return times


def _reco_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    (run / "run_info.json").write_text(json.dumps(
        {**json.loads((RECO_RUN / "run_info.json").read_text()), "suite": "reco"}
    ))
    return run


def _reco_page(monkeypatch, tmp_path, run_dir: Path) -> AppTest:
    _local_env(monkeypatch, run_dir, tmp_path / "cache")
    at = AppTest.from_function(_reco_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params["tab"] = "Correlations"
    at.run()
    assert not at.exception, at.exception
    return at


def _charts(at: AppTest) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def _scatter(at: AppTest) -> dict:
    """The cost against the expression: the last chart, below the matrix."""
    return _charts(at)[-1]


def _values(array) -> list[float]:
    """A chart spec's numeric array, which Plotly writes as typed binary."""
    if isinstance(array, dict):
        return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"]).tolist()
    return list(array)


def test_an_algorithm_is_drawn_against_what_it_reads(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    times = _write_source(run)
    at = _reco_page(monkeypatch, tmp_path, run)
    assert at.session_state["reco_active_section"] == "Correlations"
    # The baseline alone: no configuration to choose.
    with pytest.raises(KeyError):
        at.selectbox(key="corr_config")
    assert at.selectbox(key="corr_metric").options == ["time", "CPU time", "instructions",
                                                              "allocations"]
    # The whole event's 100 events are not the 12 the source holds.
    assert at.selectbox(key="corr_of").value == "Whole event"
    assert "profiling counted 100 events" in at.warning[0].value

    at.selectbox(key="corr_of").set_value("VXDBarrelDigitiser").run()
    assert not at.exception, at.exception
    # The expression box is on show, filled with a quantity from the start.
    assert at.text_input(key="corr_x").value
    at.text_input(key="corr_x").set_value("n_VertexBarrelCollection").run()
    points = _scatter(at)["data"][0]
    # The warm-up left out, in ms.
    assert _values(points["y"]) == pytest.approx([t * 1e3 for t in times[1:]])
    assert _scatter(at)["layout"]["xaxis"]["title"]["text"] == "VertexBarrelCollection elements"


def test_a_canned_quantitys_expression_is_named_on_the_axis(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    at = _reco_page(monkeypatch, tmp_path, run)
    at.selectbox(key="corr_of").set_value("Refit").run()
    at.text_input(key="corr_x").set_value("count(final & charged)").run()
    assert _scatter(at)["layout"]["xaxis"]["title"]["text"] == "Charged final-state particles"


def test_every_name_an_expression_can_use_is_listed(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    at = _reco_page(monkeypatch, tmp_path, run)
    at.selectbox(key="corr_of").set_value("Refit").run()
    assert at.expander[0].label == "What can I write?"
    quantities, fields, grammar = (d.value for d in at.expander[0].dataframe)
    row = quantities.set_index("Name").loc["n_VertexBarrelCollection"]
    assert (row["From"], row["Least"], row["Most"]) == ("the event's collections", 100, 210)
    assert quantities.set_index("Name").loc["n_charged", "Written as"] == "count(final & charged)"
    assert "abs_cos_theta" in fields["Name"].to_list()
    assert "count(mask)" in grammar["Write"].to_list()


def test_an_example_is_tried_by_a_click(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    at = _reco_page(monkeypatch, tmp_path, run)
    at.selectbox(key="corr_of").set_value("Refit").run()
    # Each example by what it gives, and only those the run can compute: its
    # input has one collection, so none adds two.
    labels = at.button_group(key="corr_example").options
    assert labels[0] == "Charged particles above 1 GeV"
    assert "Two collections' elements together" not in labels
    at.button_group(key="corr_example").set_value("count(final & charged & (pt > 1))").run()
    assert not at.exception, at.exception
    assert at.text_input(key="corr_x").value == "count(final & charged & (pt > 1))"


def test_an_expression_is_drawn_and_a_wrong_one_explained(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    at = _reco_page(monkeypatch, tmp_path, run)
    at.selectbox(key="corr_of").set_value("MyConformalTracking").run()
    at.text_input(key="corr_x").set_value("count(final & charged) * 2").run()
    assert not at.exception, at.exception
    assert _values(_scatter(at)["data"][0]["x"]) == [i % 4 * 2 + 2 for i in range(1, _EVENTS)]
    assert _scatter(at)["layout"]["xaxis"]["title"]["text"] == "count(final & charged) * 2"
    at.text_input(key="corr_x").set_value("count(pt)").run()
    assert "count needs a condition" in at.error[0].value
    at.text_input(key="corr_x").set_value("").run()
    assert "Type an expression" in at.info[0].value


def test_the_matrix_sets_each_cost_against_its_strongest_quantities(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run, follow="Refit")
    at = _reco_page(monkeypatch, tmp_path, run)
    matrix = _charts(at)[0]["data"][0]
    # Every particle is the same charged pion: the charged count, summed pT
    # and visible energy move with the final-state count and share its column.
    titles = {c[1]: c[4] for c in matrix["customdata"]}
    assert "n_charged" not in titles
    assert titles["n_final"] == (
        "Final-state particles<br>Moves with it: n_charged, sum_pt_charged, visible_E"
    )
    cells = {(c[0], c[1]): (c[2], c[3]) for c in matrix["customdata"]}
    rho, n = cells[("Refit", "n_VertexBarrelCollection")]
    assert rho == pytest.approx(1.0)
    assert n == _EVENTS - 1
    # The costliest first; the whole event, unmatched, has no row.
    rows = list(dict.fromkeys(c[0] for c in matrix["customdata"]))
    assert rows[:3] == ["MyConformalTracking", "Refit", "VertexFinder"]
    assert "Whole event" not in rows


def test_a_click_on_the_matrix_plots_its_pair(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run, follow="Refit")
    at = _reco_page(monkeypatch, tmp_path, run)
    # What a click leaves for the rerun it starts.
    at.session_state["corr_pending"] = ("Refit", "n_VertexBarrelCollection")
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="corr_of").value == "Refit"
    assert at.text_input(key="corr_x").value == "n_VertexBarrelCollection"
    # The pair is ringed on the matrix.
    ring = _charts(at)[0]["data"][1]
    assert ring["marker"]["symbol"] == "square-open"


def test_an_inferred_source_is_said_to_be(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run, origin="backfill-inferred")
    at = _reco_page(monkeypatch, tmp_path, run)
    at.selectbox(key="corr_of").set_value("Refit").run()
    assert any("inferred" in c.value for c in at.caption)


def test_a_run_without_what_its_events_hold_says_so(monkeypatch, tmp_path):
    at = _reco_page(monkeypatch, tmp_path, _reco_run(tmp_path))
    assert "recorded nothing of what its events hold" in at.info[0].value


def test_a_source_that_cannot_be_read_is_warned_about(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    (run / "event_source.json.gz").write_bytes(b"not gzip")
    at = _reco_page(monkeypatch, tmp_path, run)
    assert "generator particles could not be read" in at.warning[0].value


_RESULTS_HEADER = (
    "label,returncode,n_events,wall_time_raw,wall_time_s,user_cpu_s,sys_cpu_s,"
    "peak_rss_mb,peak_vmem_mb,major_page_faults,voluntary_ctx_switches,"
    "involuntary_ctx_switches,output_size_mb,events_per_sec"
)


_ECAL = {
    "name": "ECalBarrelCollection", "type": "edm4hep::SimCalorimeterHitCollection",
    "subset": False, "compressed_bytes": 5, "uncompressed_bytes": 9,
    "entries": [900, 100, 200, 300, 400, 500],
    "energy_gev": [9.0, 1.0, 2.0, 3.0, 4.0, 5.0],
}


def _sim_page(monkeypatch, tmp_path, collection: dict) -> AppTest:
    """The Simulation page of a ddsim run of 6 events, its output holding
    *collection*, without an event source."""
    run = tmp_path / "sim"
    run.mkdir()
    (run / "baseline_results.csv").write_text(
        _RESULTS_HEADER + "\nbaseline,0,6,0:50.00,50.0,49.0,1.0,1400,3400,0,10,1,5.96,2.0\n"
    )
    events = list(range(6))
    (run / "baseline_events.json").write_text(json.dumps({
        "schema_version": 1, "event_numbers": events,
        "event_times_s": [9.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "event_rss_begin_mb": [1.0] * 6, "event_rss_end_mb": [1.0] * 6,
    }))
    (run / "baseline_regions.json").write_text(json.dumps({
        "schema_version": 1, "event_numbers": events,
        "event_wall_seconds": [9.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "event_region_sum_seconds": [9.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "event_unaccounted_seconds": [0.0] * 6,
        "at_location_seconds": [{"ECal": t} for t in [9.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
        "by_birth_seconds": [{"ECal": t} for t in [9.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
        "interval_counts": [{"ECal": s} for s in [90, 10, 20, 30, 40, 50]],
    }))
    (run / "baseline_event_size.json").write_text(json.dumps({"schema_version": 1, "files": [{
        "name": "out.edm4hep.root", "bytes": 10, "events": 6, "compression": 505,
        "collections": [collection],
        "parameters": {"compressed_bytes": 1, "uncompressed_bytes": 1},
    }]}))
    _local_env(monkeypatch, run, tmp_path / "cache")
    at = AppTest.from_file(str(_DASHBOARD_DIR / "app.py"), default_timeout=60)
    at.query_params["tab"] = "Correlations"
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_simulation_page_draws_a_detectors_steps_against_its_hits(monkeypatch, tmp_path):
    at = _sim_page(monkeypatch, tmp_path, _ECAL)
    assert at.selectbox(key="corr_metric").options == ["time", "steps"]
    at.selectbox(key="corr_metric").set_value("steps").run()
    assert at.selectbox(key="corr_of").value == "ECal"
    assert at.text_input(key="corr_x").value == "n_ECalBarrelCollection"
    points = _scatter(at)["data"][0]
    assert _values(points["x"]) == [100, 200, 300, 400, 500]
    assert _values(points["y"]) == [10, 20, 30, 40, 50]


def test_a_collections_energy_alone_is_what_its_events_hold(monkeypatch, tmp_path):
    energy_only = {k: v for k, v in _ECAL.items() if k != "entries"}
    at = _sim_page(monkeypatch, tmp_path, energy_only)
    assert not any("recorded nothing" in i.value for i in at.info)
    at.text_input(key="corr_x").set_value("E_ECalBarrelCollection").run()
    assert not at.exception, at.exception
    assert _values(_scatter(at)["data"][0]["x"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
