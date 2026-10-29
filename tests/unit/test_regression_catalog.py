"""What each suite's report judges (:mod:`k4bench.regression.catalog`)."""

from __future__ import annotations

import pandas as pd
import pytest

from k4bench.analysis.trend import (
    build_event_size_metrics_trend,
    build_event_timing_trend,
    build_results_trend,
)
from k4bench.artifacts import COMPONENTS_SUFFIX, EVENT_SIZE_SUFFIX, EVENTS_SUFFIX, RESULTS_SUFFIX
from k4bench.labels import BASELINE_LABEL
from k4bench.layout import RECO, SIM, SUITES
from k4bench.metrics import METRICS
from k4bench.regression.catalog import (
    EVENT_METRICS,
    EVENT_REPORTED_ONLY_METRICS,
    EVENT_SIZE_METRICS,
    REPORTED_ONLY_METRICS,
    REPORTED_ONLY_REASONS,
    RESULTS,
    RUN_METRICS,
    source_of,
    sources,
)
from k4bench.regression.engine import ABS_DELTA_FLOOR, EFFECT_FLOOR


@pytest.mark.parametrize("suite", SUITES)
def test_every_suite_reads_its_results_and_names_each_source_once(suite):
    names = [source.name for source in sources(suite)]
    assert names[0] == RESULTS
    assert len(set(names)) == len(names)


def test_the_sim_suite_is_judged_on_its_results_and_events():
    results, events, event_size = sources(SIM)
    assert (results.build, results.judged, results.reported_only) == (
        build_results_trend, RUN_METRICS, REPORTED_ONLY_METRICS,
    )
    assert (events.build, events.judged, events.reported_only) == (
        build_event_timing_trend, EVENT_METRICS, EVENT_REPORTED_ONLY_METRICS,
    )
    assert not results.component_judged and not events.component_judged
    # Recorded without being judged, for the baseline config alone, whose
    # records alone are read.
    assert (event_size.build.func, event_size.judged, event_size.reported_only) == (
        build_event_size_metrics_trend, {}, EVENT_SIZE_METRICS,
    )
    assert not event_size.component_judged
    assert event_size.configs == {BASELINE_LABEL}
    assert event_size.build.keywords == {"labels": event_size.configs}
    assert results.configs is None and events.configs is None


def test_a_source_reads_the_rows_of_its_configs_alone():
    frame = pd.DataFrame({
        "label": ["baseline", "no_ECal", "baseline", "no_ECal"],
        "component": [None, None, "Tracking", "Tracking"],
    })
    _, _, event_size = sources(SIM)
    assert event_size.job_rows(frame)["label"].tolist() == ["baseline"]
    assert event_size.component_rows(frame)["label"].tolist() == ["baseline"]
    results = sources(SIM)[0]
    assert results.job_rows(frame)["label"].tolist() == ["baseline", "no_ECal"]


def test_event_size_says_why_it_is_not_judged():
    assert all("event size" in REPORTED_ONLY_REASONS[metric] for metric in EVENT_SIZE_METRICS)


def test_the_reco_suite_adds_its_components_to_the_sim_suites_sources():
    reco = sources(RECO)
    assert reco[: len(sources(SIM))] == sources(SIM)
    assert any(source.component_judged for source in reco)


def test_each_source_names_the_file_its_frame_is_built_from():
    assert {source.name: source.suffix for source in sources(RECO)} == {
        RESULTS: RESULTS_SUFFIX, "events": EVENTS_SUFFIX, "event_size": EVENT_SIZE_SUFFIX,
        "components": COMPONENTS_SUFFIX,
    }


def test_a_metric_is_read_from_the_source_recording_it_for_the_job_or_a_component():
    def name(suite, metric, component=None):
        source = source_of(suite, metric, component)
        return source.name if source is not None else None

    assert name(SIM, "wall_time_s") == RESULTS
    assert name(SIM, "user_cpu_s") == RESULTS
    assert name(SIM, "mean_rss_mb") == "events"
    assert name(SIM, "event_size_kb") == "event_size"
    assert name(RECO, "compression_ratio") == "event_size"
    assert name(RECO, "initialize_s") == "components"
    assert name(RECO, "initialize_s", "GeoSvc") == "components"
    assert name(RECO, "seconds_per_event", "MyConformalTracking") == "components"
    # Judged for the whole job only, or by another suite only, or never.
    assert name(RECO, "wall_time_s", "GeoSvc") is None
    assert name(RECO, "seconds_per_event") is None
    assert name(SIM, "initialize_s") is None
    assert name(SIM, "returncode") is None


@pytest.mark.parametrize("suite", SUITES)
def test_every_judged_family_has_its_floors(suite):
    families = {
        family
        for source in sources(suite)
        for family in (*source.judged.values(), *source.component_judged.values())
    }
    assert families <= set(EFFECT_FLOOR) & set(ABS_DELTA_FLOOR)


@pytest.mark.parametrize("suite", SUITES)
def test_every_recorded_metric_has_a_name(suite):
    recorded = {
        metric
        for source in sources(suite)
        for metric in (*source.judged, *source.reported_only, *source.component_judged)
    }
    assert recorded <= set(METRICS), sorted(recorded - set(METRICS))


def test_an_unknown_suite_is_refused():
    with pytest.raises(ValueError, match="unknown suite"):
        sources("physics")
