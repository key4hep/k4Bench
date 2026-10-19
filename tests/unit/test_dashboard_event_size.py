"""The Event Size section, rendered: on the Reconstruction page from the
``reco_run`` fixture with event sizes added, on the Simulation page from a
ddsim run, and its history across nights."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"

_RESULTS_HEADER = (
    "label,returncode,n_events,wall_time_raw,wall_time_s,user_cpu_s,sys_cpu_s,"
    "peak_rss_mb,peak_vmem_mb,major_page_faults,voluntary_ctx_switches,"
    "involuntary_ctx_switches,output_size_mb,events_per_sec"
)


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


def _reco_page(monkeypatch, tmp_path, run_dir: Path) -> AppTest:
    _local_env(monkeypatch, run_dir, tmp_path / "cache")
    at = AppTest.from_function(_reco_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params["tab"] = "Event Size"
    at.run()
    assert not at.exception, at.exception
    return at


def _reco_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    (run / "run_info.json").write_text(json.dumps(
        {**json.loads((RECO_RUN / "run_info.json").read_text()), "suite": "reco"}
    ))
    return run


def test_the_reco_page_gives_the_event_size_of_each_file_the_job_wrote(
    monkeypatch, tmp_path, event_sizes,
):
    run = _reco_run(tmp_path)
    event_sizes.write(run, files=[
        event_sizes.file("reco_DST.edm4hep.root", scale=0.25),
        event_sizes.file("reco_REC.edm4hep.root"),
        {"name": "reco_aida.root", "bytes": 1024**2},
    ])
    event_sizes.write(run, "variant_truth_tracking")
    at = _reco_page(monkeypatch, tmp_path, run)
    assert at.session_state["reco_active_section"] == "Event Size"
    assert at.selectbox(key="evt_size_config").options == ["baseline", "variant_truth_tracking"]
    # The largest file first.
    assert at.selectbox(key="evt_size_file").options == [
        "reco_REC.edm4hep.root", "reco_DST.edm4hep.root",
    ]
    assert [(m.label, m.value) for m in at.metric] == [
        ("Event size", "60.5 kB"),
        ("File size", "5.96 MB"),
        ("Compression", "2.13×"),
        ("Largest collection", "40%"),
    ]
    captions = [c.value for c in at.caption]
    assert "ZSTD level 5" in captions
    assert "100 events, metadata included" in captions
    assert "ECALBarrel, 24 kB" in captions
    assert any("`reco_aida.root` (1 MB)" in c for c in captions)
    # The event by collection, the largest six and the rest, and by type.
    by_collection, by_type = (json.loads(c.proto.spec)["data"][0] for c in at.get("plotly_chart"))
    assert by_collection["labels"] == [
        "ECALBarrel", "MCParticles", "PandoraPFOs", "SiTracks", "TightSelectedPandoraPFOs",
        "Frame parameters",
    ]
    assert by_type["labels"] == [
        "edm4hep::CalorimeterHit", "edm4hep::MCParticle", "edm4hep::ReconstructedParticle",
        "edm4hep::Track", "edm4hep::ReconstructedParticle (subset)", "Frame parameters",
    ]
    table = at.dataframe[0].value
    assert table["Collection"].to_list() == [
        "ECALBarrel", "MCParticles", "PandoraPFOs", "SiTracks", "TightSelectedPandoraPFOs",
        "Frame parameters",
    ]
    assert table["Type"].to_list()[-2:] == ["edm4hep::ReconstructedParticle (subset)", "—"]
    assert table["Compressed"].iloc[0] == pytest.approx(24.0)
    assert table["Uncompressed"].iloc[0] == pytest.approx(60.0)
    assert table["Compression"].iloc[0] == pytest.approx(2.5)
    assert table["Share"].sum() == pytest.approx(100.0)

    at.selectbox(key="evt_size_file").set_value("reco_DST.edm4hep.root").run()
    assert at.metric[0].value == "15.2 kB"
    # A configuration that wrote one file has no file to choose.
    at.selectbox(key="evt_size_config").set_value("variant_truth_tracking").run()
    with pytest.raises(KeyError):
        at.selectbox(key="evt_size_file")
    assert at.metric[0].value == "60.5 kB"


def test_a_malformed_record_hides_no_other_configuration(monkeypatch, tmp_path, event_sizes):
    run = _reco_run(tmp_path)
    event_sizes.write(run)
    (run / "variant_truth_tracking_event_size.json").write_text("{")
    at = _reco_page(monkeypatch, tmp_path, run)
    assert at.selectbox(key="evt_size_config").options == ["baseline"]
    assert at.warning[0].value.startswith(
        "The event sizes of **variant_truth_tracking** could not be read, so it is not offered:"
    )


def test_a_run_without_event_sizes_says_so(monkeypatch, tmp_path):
    at = _reco_page(monkeypatch, tmp_path, _reco_run(tmp_path))
    assert "No event size in this run" in at.info[0].value
    # Local mode has no history to offer.
    with pytest.raises(KeyError):
        at.radio(key="evt_size_view")


def test_the_simulation_page_gives_the_ddsim_outputs_event_size(monkeypatch, tmp_path, event_sizes):
    run = tmp_path / "sim"
    run.mkdir()
    (run / "baseline_results.csv").write_text(
        _RESULTS_HEADER + "\nbaseline,0,100,0:50.00,50.0,49.0,1.0,1400,3400,0,10,1,5.96,2.0\n"
    )
    event_sizes.write(run, files=[event_sizes.file("out.edm4hep.root", compression=101)])
    _local_env(monkeypatch, run, tmp_path / "cache")
    at = AppTest.from_file(str(_DASHBOARD_DIR / "app.py"), default_timeout=60)
    at.query_params["tab"] = "Event Size"
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["active_section"] == "Event Size"
    assert at.metric[0].value == "60.5 kB"
    assert "ZLIB level 1" in [c.value for c in at.caption]


def _history_app(dashboard_dir, run_dirs) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from k4bench.analysis.trend import build_event_size_trend
    from tabs import event_size

    event_size.render(run_dirs[-1], None, lambda: build_event_size_trend(run_dirs))


def _history(tmp_path: Path, event_sizes) -> AppTest:
    runs = []
    for night, scale in (("2026-10-20", 1.0), ("2026-10-21", 1.5)):
        run = tmp_path / night
        run.mkdir()
        (run / "run_info.json").write_text(json.dumps({
            "date": night, "k4h_release": f"key4hep-{night}", "k4h_release_date": night,
        }))
        event_sizes.write(run, files=[event_sizes.file("reco_REC.edm4hep.root", scale=scale)])
        runs.append(str(run))
    at = AppTest.from_function(_history_app, args=(str(_DASHBOARD_DIR), tuple(runs)),
                               default_timeout=60)
    at.run()
    at.radio(key="evt_size_view").set_value("Historical Trends").run()
    assert not at.exception, at.exception
    return at


def _traces(at: AppTest) -> list[dict]:
    return json.loads(at.get("plotly_chart")[0].proto.spec)["data"]


def test_the_history_stacks_each_nights_largest_collections(tmp_path, event_sizes):
    at = _history(tmp_path, event_sizes)
    assert at.selectbox(key="evt_size_hist_config").options == ["baseline"]
    traces = _traces(at)
    bars = [t for t in traces if t["type"] == "bar"]
    assert [b["name"] for b in bars] == [
        "ECALBarrel", "MCParticles", "PandoraPFOs", "SiTracks", "TightSelectedPandoraPFOs",
        "Frame parameters",
    ]
    # Each column carries the night's whole event.
    totals = next(t for t in traces if t["type"] == "scatter")
    assert totals["text"] == ["60.5 kB", "90.7 kB"]

    at.slider(key="evt_size_hist_topn").set_value(3).run()
    assert [b["name"] for b in _traces(at) if b["type"] == "bar"] == [
        "ECALBarrel", "MCParticles", "PandoraPFOs", "Other (2)", "Frame parameters",
    ]


def test_the_history_draws_a_line_per_collection(tmp_path, event_sizes):
    at = _history(tmp_path, event_sizes)
    at.segmented_control(key="evt_size_hist_chart_type").set_value("Lines").run()
    traces = _traces(at)
    assert {t["type"] for t in traces} == {"scatter"}
    assert traces[0]["name"] == "ECALBarrel"


def test_a_window_without_event_sizes_says_so(tmp_path):
    at = AppTest.from_function(
        _history_app, args=(str(_DASHBOARD_DIR), (str(tmp_path),)), default_timeout=60,
    )
    at.run()
    at.radio(key="evt_size_view").set_value("Historical Trends").run()
    assert "No event sizes in the selected window" in at.info[0].value
