"""What each suite's report judges (:mod:`k4bench.regression.catalog`)."""

from __future__ import annotations

import pytest

from k4bench.analysis.trend import build_event_timing_trend, build_results_trend
from k4bench.layout import RECO, SIM, SUITES
from k4bench.metrics import METRICS
from k4bench.regression.catalog import (
    EVENT_METRICS,
    EVENT_REPORTED_ONLY_METRICS,
    REPORTED_ONLY_METRICS,
    RESULTS,
    RUN_METRICS,
    sources,
)
from k4bench.regression.engine import ABS_DELTA_FLOOR, EFFECT_FLOOR


@pytest.mark.parametrize("suite", SUITES)
def test_every_suite_reads_its_results_and_names_each_source_once(suite):
    names = [source.name for source in sources(suite)]
    assert names[0] == RESULTS
    assert len(set(names)) == len(names)


def test_the_sim_suite_is_judged_on_its_results_and_events():
    results, events = sources(SIM)
    assert (results.build, results.judged, results.reported_only) == (
        build_results_trend, RUN_METRICS, REPORTED_ONLY_METRICS,
    )
    assert (events.build, events.judged, events.reported_only) == (
        build_event_timing_trend, EVENT_METRICS, EVENT_REPORTED_ONLY_METRICS,
    )
    assert not results.component_judged and not events.component_judged


def test_the_reco_suite_adds_its_components_to_the_sim_suites_sources():
    reco = sources(RECO)
    assert reco[: len(sources(SIM))] == sources(SIM)
    assert any(source.component_judged for source in reco)


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
