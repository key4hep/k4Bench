"""Unit tests for the pure trend builders in :mod:`k4bench.analysis.trend`.

These are the Streamlit-free extractions behind ``dashboard/data.py``'s cached
trend loaders, used directly by the nightly regression report in CI — so unlike
``test_dashboard_trends.py`` this module must run without Streamlit installed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from k4bench.analysis.heap import SUMMARY_COLUMNS
from k4bench.analysis.loader import failed_config_mask, load_component_heap
from k4bench.analysis import trend


def test_trend_module_does_not_import_streamlit():
    # The nightly regression report imports this module in a CI venv without
    # Streamlit; a stray dashboard import would break that silently until the
    # first nightly run.
    assert "streamlit" not in sys.modules or "streamlit" not in trend.__dict__


def _make_run(parent: Path, date: str, k4h_release: str, wall_time_s: float) -> Path:
    run_dir = parent / date
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": date,
        "platform": "PLAT",
        "k4h_release": k4h_release,
        "sample": "single_e",
        "github_run_url": f"https://ci.example/runs/{date}",
    }))
    (run_dir / "baseline_results.csv").write_text(
        "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        f"baseline,0,10,{wall_time_s},1024.0,4.0,2.0\n"
    )
    return run_dir


def test_build_results_trend(tmp_path):
    r1 = _make_run(tmp_path / "a", "2026-05-20", "key4hep-2026-05-20", 5.0)
    r2 = _make_run(tmp_path / "b", "2026-05-21", "key4hep-2026-05-21", 6.0)
    df = trend.build_results_trend((str(r1), str(r2)))
    assert df is not None and len(df) == 2
    for col in ("run_id", "run_date", "k4h_release", "x_date", "wall_time_s", "github_run_url"):
        assert col in df.columns
    assert set(df["github_run_url"]) == {
        "https://ci.example/runs/2026-05-20", "https://ci.example/runs/2026-05-21",
    }


def test_results_trend_preserves_old_schema_returncode_provenance(tmp_path):
    legacy = _make_run(
        tmp_path / "a", "2026-05-20", "key4hep-2026-05-20", 5.0,
    )
    (legacy / "baseline_results.csv").write_text(
        "label,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        "baseline,10,5.0,1024.0,4.0,2.0\n"
    )
    modern = _make_run(
        tmp_path / "b", "2026-05-21", "key4hep-2026-05-21", 6.0,
    )

    df = trend.build_results_trend((str(legacy), str(modern)))
    assert df is not None
    failed_by_run = pd.Series(
        failed_config_mask(df).to_numpy(),
        index=df["run_id"],
    )

    assert not bool(failed_by_run["2026-05-20"])
    assert not bool(failed_by_run["2026-05-21"])


def test_build_results_trend_empty():
    assert trend.build_results_trend(()) is None


def test_parse_run_dir_reads_run_info(tmp_path):
    r1 = _make_run(tmp_path, "2026-05-20", "key4hep-2026-05-20", 5.0)
    info_path = r1 / "run_info.json"
    info = json.loads(info_path.read_text())
    info["configured_labels"] = ["baseline", "no_ECal"]
    info_path.write_text(json.dumps(info))
    meta = trend.parse_run_dir(r1)
    assert meta["platform"] == "PLAT"
    assert meta["sample"] == "single_e"
    # Release date is inferred from the release name when absent.
    assert str(meta["k4h_release_date"].date()) == "2026-05-20"
    assert meta["configured_labels"] == ["baseline", "no_ECal"]


def test_parse_run_dir_reads_stack_packages(tmp_path):
    run_dir = _make_run(tmp_path, "2026-07-10", "key4hep-2026-07-10", 5.0)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": "2026-07-10",
        "k4h_release": "key4hep-2026-07-10",
        "k4h_packages": {
            "k4geo": {"commit": "a" * 40, "version": "develop",
                      "repo_url": "https://github.com/key4hep/k4geo.git"},
        },
    }))
    assert trend.parse_run_dir(run_dir)["k4h_packages"]["k4geo"]["commit"] == "a" * 40


def test_parse_run_dir_defaults_stack_packages_to_empty(tmp_path):
    # Runs predating provenance capture must load cleanly — an empty map reads
    # as "unknown", and callers must never mistake it for "nothing changed".
    r1 = _make_run(tmp_path, "2026-05-20", "key4hep-2026-05-20", 5.0)
    assert trend.parse_run_dir(r1)["k4h_packages"] == {}
    assert trend.parse_run_dir(tmp_path / "does-not-exist")["k4h_packages"] == {}


def test_parse_run_dir_rejects_empty_or_blank_config_rosters(tmp_path):
    r1 = _make_run(tmp_path, "2026-05-20", "key4hep-2026-05-20", 5.0)
    info_path = r1 / "run_info.json"
    for labels in ([], [""], ["baseline", " "]):
        info = json.loads(info_path.read_text())
        info["configured_labels"] = labels
        info_path.write_text(json.dumps(info))
        assert trend.parse_run_dir(r1)["configured_labels"] is None


def test_parse_run_dir_deduplicates_config_roster(tmp_path):
    r1 = _make_run(tmp_path, "2026-05-20", "key4hep-2026-05-20", 5.0)
    info_path = r1 / "run_info.json"
    info = json.loads(info_path.read_text())
    info["configured_labels"] = ["baseline", "no_ECal", "baseline"]
    info_path.write_text(json.dumps(info))
    assert trend.parse_run_dir(r1)["configured_labels"] == [
        "baseline",
        "no_ECal",
    ]


# ── build_phase_trend ─────────────────────────────────────────────────────────

_RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"

def test_build_phase_trend_gives_phases_and_their_components():
    df = trend.build_phase_trend((str(_RECO_RUN),))
    totals = df[df["component"].isna() & (df["label"] == "baseline")].set_index("phase")
    # The fixture keeps the initialize and execute spans; configure is derived.
    assert set(totals.index) == {"configure", "initialize", "execute"}
    init = df[(df["phase"] == "initialize") & df["component"].notna()
              & (df["label"] == "baseline")]
    assert init["seconds"].sum() == pytest.approx(totals.loc["initialize", "seconds"], rel=1e-3)
    assert (init["seconds"] > 0).all()
    assert init.sort_values("seconds")["component"].iloc[-1] == "GeoSvc"
    loop = df[(df["phase"] == "execute") & df["component"].notna()
              & (df["label"] == "baseline")]
    assert loop.sort_values("seconds")["component"].iloc[-1] == "MyConformalTracking"


def test_build_algorithm_timing_trend_gives_each_algorithms_event_time_statistics():
    df = trend.build_algorithm_timing_trend((str(_RECO_RUN),))
    baseline = df[df["label"] == "baseline"].set_index("component")
    tracking = baseline.loc["MyConformalTracking"]
    # The fixture keeps 12 events; the warm-up is left out.
    assert tracking["n_events"] == 11
    assert tracking["median_time_s"] <= tracking["p95_time_s"]
    assert tracking["mean_time_s"] > 1.0
    assert "GeoSvc" not in baseline.index
    # The truth-tracking variant runs no pattern recognition.
    variant = df[df["label"] == "variant_truth_tracking"]
    assert "MyConformalTracking" not in set(variant["component"])


def test_build_component_cost_trend_gives_what_each_component_and_the_job_cost():
    df = trend.build_component_cost_trend((str(_RECO_RUN),))
    baseline = df[df["label"] == "baseline"]
    (job,) = (row for _, row in baseline[baseline["component"].isna()].iterrows())
    costs = baseline[baseline["component"].notna()].set_index("component")
    phases = trend.build_phase_trend((str(_RECO_RUN),))
    span = phases[phases["component"].isna() & (phases["label"] == "baseline")
                  & (phases["phase"] == "initialize")]["seconds"].item()
    assert job["initialize_s"] == pytest.approx(span)
    assert pd.isna(job["seconds_per_event"])
    # An algorithm costs per event and in initialize; a service only there.
    assert costs.loc["MyConformalTracking", "seconds_per_event"] > 1.0
    assert pd.isna(costs.loc["GeoSvc", "seconds_per_event"])
    assert costs.loc["GeoSvc", "initialize_s"] > 0
    # Sequencers' own costs are bookkeeping, so they are not listed.
    assert not {"k4FWCore__Sequencer", "k4FWCore__Algs"} & set(costs.index)
    # Self costs: the algorithms' per-event times add up to the event loop's.
    loop = trend.build_algorithm_timing_trend((str(_RECO_RUN),))
    loop = loop[loop["label"] == "baseline"].set_index("component")
    assert costs.loc[loop.index, "seconds_per_event"].sum() == pytest.approx(
        (loop["mean_time_s"] * loop["n_events"]).sum() / loop["n_events"].max(),
    )


def test_build_component_cost_trend_skips_runs_without_component_files(tmp_path):
    sim = _make_run(tmp_path, "2026-10-01", "key4hep-2026-10-01", 50.0)
    assert trend.build_component_cost_trend((str(sim),)) is None


def test_build_algorithm_timing_trend_skips_runs_without_component_files(tmp_path):
    sim = _make_run(tmp_path, "2026-10-01", "key4hep-2026-10-01", 50.0)
    assert trend.build_algorithm_timing_trend((str(sim),)) is None


def test_build_phase_trend_skips_runs_without_component_files(tmp_path):
    sim = _make_run(tmp_path, "2026-10-01", "key4hep-2026-10-01", 50.0)
    assert trend.build_phase_trend((str(sim),)) is None


@pytest.mark.parametrize("build", [
    trend.build_phase_trend, trend.build_component_summary_trend,
    trend.build_algorithm_timing_trend, trend.build_component_cost_trend,
], ids=["phase", "component_summary", "algorithm_timing", "component_cost"])
def test_component_trends_keep_a_runs_other_configs_beside_a_malformed_file(tmp_path, build):
    broken, valid = (tmp_path / "broken", tmp_path / "valid")
    for run in (broken, valid):
        shutil.copytree(_RECO_RUN, run)
    (broken / "variant_truth_tracking_components.json").write_text("{")
    df = build((str(broken), str(valid)))
    labels = df.groupby("run_id")["label"].unique().map(sorted).to_dict()
    # The malformed file gives no row at all, so neither a zero nor a gap filled in.
    assert labels == {
        "broken": ["baseline"], "valid": ["baseline", "variant_truth_tracking"],
    }


# ── build_heap_trend ──────────────────────────────────────────────────────────

def _heap_rows(df: pd.DataFrame, label: str = "baseline") -> tuple[pd.DataFrame, pd.Series]:
    """*label*'s component rows, by component, and its whole-job row."""
    rows = df[df["label"] == label]
    (job,) = (row for _, row in rows[rows["component"].isna()].iterrows())
    return rows[rows["component"].notna()].set_index("component"), job


def test_build_heap_trend_gives_each_component_and_the_whole_job(reco_run_with_heap):
    df = trend.build_heap_trend((str(reco_run_with_heap()),))
    assert list(df.columns) == [
        "run_id", "run_date", "k4h_release_date", "k4h_release", "label", "component",
        *SUMMARY_COLUMNS, "x_date",
    ]
    components, job = _heap_rows(df)
    # The components that held heap at some moment, no component among them.
    assert set(components.index) == {
        "(no component)", "GeoSvc", "k4FWCore__Reader", "LumiCalReco", "VertexFinder",
        "k4FWCore__Writer",
    }
    assert components.loc["GeoSvc", "before_first_mb"] == pytest.approx(397)
    assert components.loc["LumiCalReco", "growth_mb_per_event"] == pytest.approx(2.0)
    assert components.loc["k4FWCore__Reader", "working_set_median_mb"] == pytest.approx(5.0)
    assert components["leak"].eq(True).to_dict() == {
        name: name in ("LumiCalReco", "VertexFinder") for name in components.index
    }
    # The whole job's before the first event is its components' together.
    assert job["before_first_mb"] == pytest.approx(components["before_first_mb"].sum())
    assert pd.isna(job["working_set_median_mb"]) and pd.isna(job["leak"])
    assert df["x_date"].notna().all()


def test_build_heap_trend_gives_the_whole_job_as_its_totals(reco_run_with_heap):
    run = reco_run_with_heap()
    totals = load_component_heap(run, ["baseline"])["baseline"].totals(warm_up=0)
    _, job = _heap_rows(trend.build_heap_trend((str(run),)))
    for column in trend.HEAP_TOTAL_COLUMNS:
        assert job[column] == pytest.approx(getattr(totals, column)), column


def test_build_heap_trend_gives_each_night_on_its_tag(reco_run_with_heap):
    runs = [reco_run_with_heap(date, date) for date in ("2026-10-19", "2026-10-20")]
    df = trend.build_heap_trend(tuple(map(str, runs)))
    jobs = df[df["component"].isna()]
    assert jobs["run_id"].tolist() == ["2026-10-19", "2026-10-20"]
    assert jobs["x_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-19", "2026-10-20"]
    assert jobs["k4h_release"].tolist() == ["key4hep-2026-10-19", "key4hep-2026-10-20"]


def test_build_heap_trend_skips_runs_without_heap_files(reco_run_with_heap, tmp_path):
    sim = _make_run(tmp_path / "sim", "2026-10-01", "key4hep-2026-10-01", 50.0)
    df = trend.build_heap_trend((str(_RECO_RUN), str(sim), str(reco_run_with_heap())))
    assert df["run_id"].unique().tolist() == ["run"]


def test_build_heap_trend_keeps_a_runs_other_configs_beside_a_malformed_file(
    reco_run_with_heap,
):
    run = reco_run_with_heap()
    (run / "variant_truth_tracking_heap.json").write_text("{")
    df = trend.build_heap_trend((str(run),))
    assert df["label"].unique().tolist() == ["baseline"]


def test_build_heap_trend_is_none_without_heap_files(tmp_path):
    sim = _make_run(tmp_path, "2026-10-01", "key4hep-2026-10-01", 50.0)
    assert trend.build_heap_trend((str(_RECO_RUN), str(sim))) is None
    assert trend.build_heap_trend(()) is None


# ── build_event_size_trend ────────────────────────────────────────────────────

def test_build_event_size_trend_gives_each_collection_and_the_whole_event(tmp_path, event_sizes):
    first = _make_run(tmp_path / "a", "2026-10-20", "key4hep-2026-10-20", 5.0)
    event_sizes.write(first, files=[event_sizes.file("out.root"), {"name": "aida.root", "bytes": 9}])
    second = _make_run(tmp_path / "b", "2026-10-21", "key4hep-2026-10-21", 5.0)
    event_sizes.write(second, files=[event_sizes.file("out.root", scale=2.0)])

    df = trend.build_event_size_trend((str(first), str(second)))
    assert list(df.columns) == [*trend.EVENT_SIZE_TREND_COLUMNS, "x_date"]
    # A file without events has no rows.
    assert set(df["file"]) == {"out.root"}
    rows = df[df["run_id"] == "2026-10-20"].set_index("collection", drop=False)
    collections = rows[rows["collection"].notna()]
    assert collections.index.to_list() == [
        "ECALBarrel", "MCParticles", "PandoraPFOs", "SiTracks", "TightSelectedPandoraPFOs",
        "(frame parameters)",
    ]
    assert collections.loc["ECALBarrel", "compressed_kb"] == pytest.approx(24.0)
    assert collections.loc["ECALBarrel", "type"] == "edm4hep::CalorimeterHitCollection"
    (whole,) = (row for _, row in rows[rows["collection"].isna()].iterrows())
    assert whole["compressed_kb"] == pytest.approx(24 + 16 + 12 + 8 + 0.4 + 0.1)
    assert whole["uncompressed_kb"] == pytest.approx(collections["uncompressed_kb"].sum())
    assert whole["events"] == 100
    # The events' 6,195,200 bytes and 50,000 of metadata.
    assert whole["file_mb"] == pytest.approx(6_245_200 / 1024**2)
    later = df[(df["run_id"] == "2026-10-21") & (df["collection"] == "ECALBarrel")]
    assert later["compressed_kb"].item() == pytest.approx(48.0)
    assert df["x_date"].notna().all()


def test_build_event_size_trend_skips_what_it_cannot_read(tmp_path, event_sizes):
    run = _make_run(tmp_path / "a", "2026-10-20", "key4hep-2026-10-20", 5.0)
    event_sizes.write(run, "variant_a")
    (run / "baseline_event_size.json").write_text("{")
    event_sizes.write(run, "variant_b", files=[{**event_sizes.file("out.root"), "events": "invalid"}])
    df = trend.build_event_size_trend((str(run),))
    assert set(df["label"]) == {"variant_a"}
    # A run from before event sizes were recorded has none.
    old = _make_run(tmp_path / "b", "2026-10-01", "key4hep-2026-10-01", 5.0)
    assert trend.build_event_size_trend((str(old),)) is None


# ── build_event_size_metrics_trend ────────────────────────────────────────────

def test_build_event_size_metrics_trend_gives_one_row_per_run_and_config(tmp_path, event_sizes):
    first = _make_run(tmp_path / "a", "2026-10-20", "key4hep-2026-10-20", 5.0)
    event_sizes.write(first, files=[event_sizes.file("out.root"), {"name": "aida.root", "bytes": 9}])
    event_sizes.write(first, "variant_a", files=[event_sizes.file("out.root", scale=2.0)])
    second = _make_run(tmp_path / "b", "2026-10-21", "key4hep-2026-10-21", 5.0)
    event_sizes.write(second, files=[event_sizes.file("out.root", scale=2.0)])

    df = trend.build_event_size_metrics_trend((str(first), str(second)))
    assert list(df.columns) == [
        "run_id", "run_date", "k4h_release_date", "k4h_release", "label",
        *trend.EVENT_SIZE_METRIC_COLUMNS, "x_date",
    ]
    rows = df.set_index(["run_id", "label"])
    assert len(rows) == 3
    baseline = rows.loc[("2026-10-20", "baseline")]
    # The collections' 60.4 kB and the frame parameters' 0.1 kB, per event;
    # a file without events adds nothing.
    assert baseline["event_size_kb"] == pytest.approx(60.5)
    uncompressed = (6_144_000 + 3_276_800 + 2_457_600 + 1_228_800 + 81_920 + 20_480) / 100 / 1024
    assert baseline["uncompressed_event_size_kb"] == pytest.approx(uncompressed)
    # The events' 6,195,200 bytes and 50,000 of metadata, over 100 events.
    assert baseline["file_size_per_event_kb"] == pytest.approx(6_245_200 / 100 / 1024)
    assert baseline["compression_ratio"] == pytest.approx(uncompressed / 60.5)
    assert rows.loc[("2026-10-21", "baseline"), "event_size_kb"] == pytest.approx(
        rows.loc[("2026-10-20", "variant_a"), "event_size_kb"]
    )
    assert df["x_date"].notna().all()


def test_build_event_size_metrics_trend_sums_a_jobs_podio_files(tmp_path, event_sizes):
    run = _make_run(tmp_path, "2026-10-20", "key4hep-2026-10-20", 5.0)
    event_sizes.write(run, files=[
        event_sizes.file("reco_REC.edm4hep.root"),
        event_sizes.file("reco_DST.edm4hep.root", scale=0.5),
    ])
    (row,) = trend.build_event_size_metrics_trend((str(run),)).to_dict("records")
    one = trend.build_event_size_trend((str(run),))
    whole = one[one["collection"].isna()]
    assert row["event_size_kb"] == pytest.approx(whole["compressed_kb"].sum())
    assert row["file_size_per_event_kb"] == pytest.approx(
        (whole["file_mb"] * 1024 / whole["events"]).sum()
    )
    assert row["compression_ratio"] == pytest.approx(
        whole["uncompressed_kb"].sum() / whole["compressed_kb"].sum()
    )


def test_build_event_size_metrics_trend_reads_the_records_of_its_labels_alone(
    tmp_path, event_sizes, monkeypatch,
):
    run = _make_run(tmp_path, "2026-10-20", "key4hep-2026-10-20", 5.0)
    for label in ("baseline", "no_ECal", "no_HCal"):
        event_sizes.write(run, label)
    read = []
    original = trend.read_event_size
    monkeypatch.setattr(trend, "read_event_size", lambda path: read.append(path.name) or original(path))

    df = trend.build_event_size_metrics_trend((str(run),), frozenset({"baseline"}))
    assert df["label"].tolist() == ["baseline"]
    assert read == ["baseline_event_size.json"]
    # A run with none of them has no row.
    assert trend.build_event_size_metrics_trend((str(run),), frozenset({"variant"})) is None


def test_build_event_size_metrics_trend_is_none_without_event_sizes(tmp_path):
    old = _make_run(tmp_path, "2026-10-01", "key4hep-2026-10-01", 5.0)
    assert trend.build_event_size_metrics_trend((str(old),)) is None
    assert trend.build_event_size_metrics_trend(()) is None
