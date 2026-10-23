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
    numbered: bool = False, first_number: int = 0,
) -> list[float]:
    """An event source whose input's vertex hits follow the time of
    algorithm *follow* event by event, and whose events hold 1 to 4 charged
    pions, each with its EventHeader if *numbered*, numbered from
    *first_number*; that algorithm's times."""
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
    if numbered:
        record["event_headers"] = {
            "event_number": list(range(first_number, first_number + events)),
            "run_number": [0] * events,
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
    # Both configurations on offer, the baseline drawn first.
    assert at.selectbox(key="corr_config").options == ["baseline", "variant_truth_tracking"]
    assert at.selectbox(key="corr_config").value == "baseline"
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


_VARIANT = "variant_truth_tracking"


def _wall(label: str, algorithm: str) -> list[float]:
    """*label*'s time of *algorithm* in each of the fixture's events."""
    return load_component_timing(RECO_RUN)[label].execute["wall_s"][algorithm].tolist()


def _write_outputs(run: Path, label: str, collections: dict[str, list[int]]) -> None:
    """*label*'s output file, its events numbered as the source's and
    holding *collections*' elements."""
    (run / f"{label}_event_size.json").write_text(json.dumps({"schema_version": 1, "files": [{
        "name": f"{label}.edm4hep.root", "bytes": 10, "events": _EVENTS,
        "event_headers": {"event_number": list(range(_EVENTS)), "run_number": [0] * _EVENTS},
        "collections": [
            {"name": name, "type": "edm4hep::ReconstructedParticleCollection", "subset": False,
             "compressed_bytes": 5, "uncompressed_bytes": 9, "entries": entries}
            for name, entries in collections.items()
        ],
    }]}))


def _caption_text(at: AppTest) -> str:
    return " ".join(c.value for c in at.caption)


def test_a_run_of_one_configuration_offers_no_choice(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    for path in run.glob(f"{_VARIANT}*"):
        path.unlink()
    _write_source(run)
    at = _reco_page(monkeypatch, tmp_path, run)
    with pytest.raises(KeyError):
        at.selectbox(key="corr_config")
    assert at.selectbox(key="corr_of").value == "Whole event"


def test_the_chosen_configurations_costs_are_drawn(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    at = _reco_page(monkeypatch, tmp_path, run)
    at.selectbox(key="corr_of").set_value("Refit").run()
    at.text_input(key="corr_x").set_value("n_VertexBarrelCollection").run()
    baseline, variant = _wall("baseline", "Refit"), _wall(_VARIANT, "Refit")
    assert baseline[1:] != variant[1:]
    assert _values(_scatter(at)["data"][0]["y"]) == pytest.approx([t * 1e3 for t in baseline[1:]])

    at.selectbox(key="corr_config").set_value(_VARIANT).run()
    assert not at.exception, at.exception
    # The algorithm and the expression, both the variant's too, are kept.
    assert at.selectbox(key="corr_of").value == "Refit"
    assert at.text_input(key="corr_x").value == "n_VertexBarrelCollection"
    assert _values(_scatter(at)["data"][0]["y"]) == pytest.approx([t * 1e3 for t in variant[1:]])
    # The variant's own algorithms, in the matrix as in the plot row.
    options = at.selectbox(key="corr_of").options
    assert "MyTruthTrackFinder" in options and "MyConformalTracking" not in options
    rows = {c[0] for c in _charts(at)[0]["data"][0]["customdata"]}
    assert "MyTruthTrackFinder" in rows and "MyConformalTracking" not in rows


def test_a_link_opens_the_configuration_it_names(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    _local_env(monkeypatch, run, tmp_path / "cache")
    at = AppTest.from_function(_reco_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params.update({"tab": "Correlations", "corr_config": _VARIANT, "corr_of": "Refit",
                            "corr_x": "n_VertexBarrelCollection"})
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="corr_config").value == _VARIANT
    assert not any("shown instead" in w.value for w in at.warning)
    assert _values(_scatter(at)["data"][0]["y"]) == pytest.approx(
        [t * 1e3 for t in _wall(_VARIANT, "Refit")[1:]]
    )


def test_an_algorithm_the_chosen_configuration_lacks_is_said_to_be(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    at = _reco_page(monkeypatch, tmp_path, run)
    at.selectbox(key="corr_metric").set_value("instructions").run()
    at.selectbox(key="corr_of").set_value("MyConformalTracking").run()
    at.selectbox(key="corr_config").set_value(_VARIANT).run()
    assert not at.exception, at.exception
    shown = at.selectbox(key="corr_of").value
    assert shown != "MyConformalTracking"
    assert at.selectbox(key="corr_metric").value == "instructions"
    assert (f"**MyConformalTracking** has no instructions in **{_VARIANT}**: **{shown}** is shown."
            in _caption_text(at))


def test_each_configuration_reads_its_own_outputs(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run, numbered=True)
    _write_outputs(run, "baseline", {"PandoraPFOs": list(range(10, 22)),
                                     "SiTracks": list(range(_EVENTS))})
    _write_outputs(run, _VARIANT, {"PandoraPFOs": list(range(50, 62))})
    at = _reco_page(monkeypatch, tmp_path, run)
    at.selectbox(key="corr_of").set_value("Refit").run()
    at.text_input(key="corr_x").set_value("out_n_PandoraPFOs").run()
    assert _values(_scatter(at)["data"][0]["x"]) == list(range(11, 22))

    at.selectbox(key="corr_config").set_value(_VARIANT).run()
    assert not at.exception, at.exception
    assert at.text_input(key="corr_x").value == "out_n_PandoraPFOs"
    assert _values(_scatter(at)["data"][0]["x"]) == list(range(51, 62))
    names = at.expander[0].dataframe[0].value["Name"].to_list()
    assert "out_n_PandoraPFOs" in names and "out_n_SiTracks" not in names


def test_an_expression_the_chosen_configuration_cannot_give_is_replaced(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run, numbered=True)
    _write_outputs(run, "baseline", {"SiTracks": list(range(_EVENTS))})
    at = _reco_page(monkeypatch, tmp_path, run)
    at.selectbox(key="corr_of").set_value("Refit").run()
    at.text_input(key="corr_x").set_value("out_n_SiTracks * 2").run()
    assert _values(_scatter(at)["data"][0]["x"]) == [2 * i for i in range(1, _EVENTS)]

    at.selectbox(key="corr_config").set_value(_VARIANT).run()
    assert not at.exception, at.exception
    replaced = at.text_input(key="corr_x").value
    assert replaced and "out_n_SiTracks" not in replaced
    assert not at.error
    assert f"`out_n_SiTracks * 2` cannot be computed for **{_VARIANT}**'s events" in (
        _caption_text(at)
    )


def test_a_pending_click_is_not_carried_to_another_configuration(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    at = _reco_page(monkeypatch, tmp_path, run)
    at.session_state["corr_pending"] = ("MyConformalTracking", "n_VertexBarrelCollection")
    at.selectbox(key="corr_config").set_value(_VARIANT).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="corr_of").value != "MyConformalTracking"
    assert "corr_pending" not in at.session_state


def test_stale_query_params_are_dropped_safely(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    _local_env(monkeypatch, run, tmp_path / "cache")
    at = AppTest.from_function(_reco_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params.update({
        "tab": "Correlations", "corr_config": "no_such_config", "corr_metric": "nonsense",
        "corr_of": "NoSuchAlgorithm", "corr_x": "out_n_Nothing",
    })
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="corr_config").value == "baseline"
    assert ("**no_such_config** is not in this run, so **baseline** is shown instead."
            in [w.value for w in at.warning])
    assert at.selectbox(key="corr_metric").value == "time"
    assert at.selectbox(key="corr_of").value == "Whole event"
    # The user's expression is kept, and why it draws nothing said.
    assert at.text_input(key="corr_x").value == "out_n_Nothing"
    at.selectbox(key="corr_of").set_value("Refit").run()
    assert "cannot be computed" in at.error[0].value
    # The warning stays while the baseline is shown in its place.
    assert any("no_such_config" in w.value for w in at.warning)


def test_a_failed_configuration_is_not_offered(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    results = run / f"{_VARIANT}_results.csv"
    header, row = results.read_text().splitlines()[:2]
    results.write_text(f"{header}\n{row.replace(f'{_VARIANT},0,', f'{_VARIANT},1,', 1)}\n")
    _local_env(monkeypatch, run, tmp_path / "cache")
    at = AppTest.from_function(_reco_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params.update({"tab": "Correlations", "corr_config": _VARIANT})
    at.run()
    assert not at.exception, at.exception
    with pytest.raises(KeyError):
        at.selectbox(key="corr_config")
    assert (f"**{_VARIANT}** failed in this run, so **baseline** is shown instead."
            in [w.value for w in at.warning])
    at.selectbox(key="corr_of").set_value("MyConformalTracking").run()
    assert _values(_scatter(at)["data"][0]["y"]) == pytest.approx(
        [t * 1e3 for t in _wall("baseline", "MyConformalTracking")[1:]]
    )


def test_a_configuration_whose_component_file_is_unreadable_has_its_event_time_alone(
    monkeypatch, tmp_path,
):
    run = _reco_run(tmp_path)
    _write_source(run)
    (run / f"{_VARIANT}_components.json").write_text("{not json")
    at = _reco_page(monkeypatch, tmp_path, run)
    assert any(_VARIANT in w.value for w in at.warning)
    at.selectbox(key="corr_config").set_value(_VARIANT).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="corr_metric").options == ["time"]
    assert at.selectbox(key="corr_of").options == ["Whole event"]
    # Its 100 events, not the source's 12: no matrix or plot, and why.
    assert any("profiling counted 100 events" in w.value for w in at.warning)
    assert not _charts(at)


def _clicking_app(dashboard_dir) -> None:
    """The Reconstruction page, whose charts can be clicked: a
    ``session_state["fake_click"]`` of ``(chart key, curve, point)`` calls
    that chart's selection callback before the page is drawn, as Streamlit
    calls it after a click, with the click on that point of that curve of
    the figure drawn."""
    import sys as _sys
    from types import SimpleNamespace
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st
    from reco import _inspect, page

    if (click := st.session_state.pop("fake_click", None)) is not None:
        chart, curve, point = click
        event = SimpleNamespace(selection={"points": [{"curve_number": curve, "point_index": point}]})
        real_clicked = _inspect.clicked_name
        _inspect.clicked_name = lambda _event, fig: real_clicked(event, fig)
        try:
            st.session_state["callbacks"][chart]()
        finally:
            _inspect.clicked_name = real_clicked

    real_chart = st.plotly_chart

    def plotly_chart(fig, *args, **kwargs):
        if callable(kwargs.get("on_select")):
            chart = kwargs["key"].rsplit("_", 1)[0]
            st.session_state.setdefault("callbacks", {})[chart] = kwargs["on_select"]
        return real_chart(fig, *args, **kwargs)

    st.plotly_chart = plotly_chart
    try:
        page.main()
    finally:
        st.plotly_chart = real_chart


#: Real for the events with 3 or 4 of the source's 1 to 4 pions, the
#: others missing from the plot: a point's place is not its event.
_GAPPED = "sqrt(count(final) - 2.5)"
_PLOTTED = [2, 3, 6, 7, 10, 11]


def _audited_events(run: Path, label: str) -> list[float]:
    """Make *label*'s events file hold the auditor's events, as a real job's
    does (the fixture's holds 100); their times."""
    path = run / f"{label}_events.json"
    record = json.loads(path.read_text())
    record = {k: v[:_EVENTS] if isinstance(v, list) else v for k, v in record.items()}
    path.write_text(json.dumps(record))
    return record["event_times_s"]


def _clicking_page(monkeypatch, tmp_path, run: Path, params=None) -> AppTest:
    _local_env(monkeypatch, run, tmp_path / "cache")
    at = AppTest.from_function(_clicking_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params["tab"] = "Correlations"
    for key, value in (params or {}).items():
        at.query_params[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _click(at: AppTest, chart: str, curve: int, point: int) -> AppTest:
    at.session_state["fake_click"] = (chart, curve, point)
    at.run()
    assert not at.exception, at.exception
    return at


def _chart(at: AppTest, key: str) -> dict:
    """The chart drawn under *key*, whatever its nonce."""
    return next(json.loads(c.proto.spec) for c in at.get("plotly_chart")
                if f"-{key}_" in c.proto.id)


def _refit_against_gaps(monkeypatch, tmp_path, **source) -> tuple[AppTest, Path]:
    run = _reco_run(tmp_path)
    _write_source(run, **source)
    at = _clicking_page(monkeypatch, tmp_path, run, {"corr_of": "Refit", "corr_x": _GAPPED})
    assert [int(e) for e in _values(_chart(at, "corr_chart")["data"][0]["customdata"])] == _PLOTTED
    return at, run


def test_a_click_on_a_point_shows_its_event_not_the_one_at_its_place(monkeypatch, tmp_path):
    at, _ = _refit_against_gaps(monkeypatch, tmp_path)
    assert at.selectbox(key="corr_event").value is None
    with pytest.raises(StopIteration):
        _chart(at, "corr_event_chart")
    # The third point is event 6.
    _click(at, "corr_chart", 0, 2)
    assert at.selectbox(key="corr_event").value == 6
    assert "Event **6** of **baseline**, entry 6 of `sim.edm4hep.root`" in _caption_text(at)
    # Ringed on the plot, at its point.
    ring = _chart(at, "corr_chart")["data"][-1]
    assert ring["marker"]["symbol"] == "circle-open"
    assert _values(ring["y"]) == pytest.approx([_wall("baseline", "Refit")[6] * 1e3])
    # Each algorithm's own time in that event, of that configuration, the
    # largest first, the one plotted in red.
    bars = _chart(at, "corr_event_chart")["data"][0]
    wall = load_component_timing(RECO_RUN)["baseline"].execute["wall_s"].loc[6]
    names = [row[0] for row in bars["customdata"]]
    assert _values(bars["x"]) == pytest.approx([wall[n] * 1e3 for n in names])
    assert _values(bars["x"]) == sorted(_values(bars["x"]), reverse=True)
    assert names[0] == wall.dropna().idxmax()
    assert [c for n, c in zip(names, bars["marker"]["color"]) if c == "#d62728"] == ["#d62728"]
    assert bars["marker"]["color"][names.index("Refit")] == "#d62728"


def test_a_click_on_the_trend_line_or_a_ring_shows_no_event(monkeypatch, tmp_path):
    at, _ = _refit_against_gaps(monkeypatch, tmp_path)
    for curve in range(1, len(_chart(at, "corr_chart")["data"])):
        _click(at, "corr_chart", curve, 0)
        assert at.selectbox(key="corr_event").value is None
    # Nor does the ring of the event shown: it stays shown.
    _click(at, "corr_chart", 0, 2)
    assert _chart(at, "corr_chart")["data"][-1]["name"] == "Selected"
    _click(at, "corr_chart", len(_chart(at, "corr_chart")["data"]) - 1, 0)
    assert at.selectbox(key="corr_event").value == 6


def test_an_event_is_chosen_by_its_number_and_a_link_names_it(monkeypatch, tmp_path):
    run = _reco_run(tmp_path)
    _write_source(run)
    times = _audited_events(run, "baseline")
    at = _clicking_page(monkeypatch, tmp_path, run,
                        {"corr_of": "Refit", "corr_x": "n_VertexBarrelCollection",
                         "corr_event": "4"})
    # Every event the job counted but the warm-up.
    assert at.selectbox(key="corr_event").options == [str(e) for e in range(1, _EVENTS)]
    assert at.selectbox(key="corr_event").value == 4
    # The event's time beside the algorithm's, from the same job.
    text = _caption_text(at)
    assert f"Event time: {times[4]:.3g} s" in text
    # Over a second, in seconds.
    refit = _wall("baseline", "Refit")[4]
    assert refit > 1 and f"Refit time: {refit:.3g} s" in text
    at.selectbox(key="corr_event").set_value(9).run()
    assert "Event **9** of **baseline**" in _caption_text(at)


@pytest.mark.parametrize("asked", ["0", "99", "x"])
def test_an_event_the_plot_has_not_is_not_shown(monkeypatch, tmp_path, asked):
    run = _reco_run(tmp_path)
    _write_source(run)
    at = _clicking_page(monkeypatch, tmp_path, run,
                        {"corr_of": "Refit", "corr_x": "n_VertexBarrelCollection",
                         "corr_event": asked})
    assert at.selectbox(key="corr_event").value is None
    with pytest.raises(StopIteration):
        _chart(at, "corr_event_chart")


def test_another_configuration_shows_the_same_event_with_its_own_costs(monkeypatch, tmp_path):
    at, _ = _refit_against_gaps(monkeypatch, tmp_path)
    _click(at, "corr_chart", 0, 2)
    at.selectbox(key="corr_config").set_value(_VARIANT).run()
    assert not at.exception, at.exception
    # Every configuration reads the same input: event 6 is the same event.
    assert at.selectbox(key="corr_event").value == 6
    assert f"Event **6** of **{_VARIANT}**" in _caption_text(at)
    bars = _chart(at, "corr_event_chart")["data"][0]
    wall = load_component_timing(RECO_RUN)[_VARIANT].execute["wall_s"].loc[6]
    names = [row[0] for row in bars["customdata"]]
    assert "MyTruthTrackFinder" in names and "MyConformalTracking" not in names
    assert _values(bars["x"]) == pytest.approx([wall[n] * 1e3 for n in names])


def test_a_click_on_an_algorithm_opens_it_in_components_and_goes_back(monkeypatch, tmp_path):
    at, _ = _refit_against_gaps(monkeypatch, tmp_path)
    at.selectbox(key="corr_config").set_value(_VARIANT).run()
    _click(at, "corr_chart", 0, 2)
    names = [row[0] for row in _chart(at, "corr_event_chart")["data"][0]["customdata"]]
    _click(at, "corr_event_chart", 0, names.index("MyTruthTrackFinder"))
    assert at.session_state["reco_active_section"] == "Components"
    assert at.selectbox(key="reco_component_config").value == _VARIANT
    assert at.session_state["reco_component_name"] == "MyTruthTrackFinder"
    at.button(key="reco_component_back").click().run()
    assert not at.exception, at.exception
    assert at.session_state["reco_active_section"] == "Correlations"
    assert at.selectbox(key="corr_config").value == _VARIANT
    assert at.selectbox(key="corr_of").value == "Refit"
    assert at.text_input(key="corr_x").value == _GAPPED
    assert at.selectbox(key="corr_event").value == 6


def test_the_events_entry_and_its_eventheader_are_told_apart(monkeypatch, tmp_path):
    at, _ = _refit_against_gaps(monkeypatch, tmp_path, numbered=True, first_number=1000)
    _click(at, "corr_chart", 0, 2)
    assert ("Event **6** of **baseline**, entry 6 of `sim.edm4hep.root` "
            "(EventHeader event 1006, run 0)") in _caption_text(at)


def test_an_event_without_a_value_of_the_expression_says_so(monkeypatch, tmp_path):
    at, _ = _refit_against_gaps(monkeypatch, tmp_path)
    # Event 5 holds 2 pions: it is no point, but can be chosen.
    at.selectbox(key="corr_event").set_value(5).run()
    assert f"{_GAPPED}: no value" in _caption_text(at)
    assert _chart(at, "corr_event_chart")


def test_a_configuration_without_audited_algorithms_shows_the_events_time_alone(
    monkeypatch, tmp_path,
):
    run = _reco_run(tmp_path)
    _write_source(run)
    times = _audited_events(run, _VARIANT)
    (run / f"{_VARIANT}_components.json").write_text("{not json")
    at = _clicking_page(monkeypatch, tmp_path, run, {"corr_config": _VARIANT, "corr_event": "3"})
    assert at.selectbox(key="corr_of").options == ["Whole event"]
    text = _caption_text(at)
    assert f"Event time: {times[3]:.3g} s" in text
    assert f"No algorithm of **{_VARIANT}** was audited" in text
    with pytest.raises(StopIteration):
        _chart(at, "corr_event_chart")


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
