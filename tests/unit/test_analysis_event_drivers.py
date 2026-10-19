"""What an event's cost goes with (:mod:`k4bench.analysis.event_drivers`):
the table of what each event holds, the costs, how they are matched, ranked
and fitted."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from k4bench.analysis.event_drivers import (
    FEATURES,
    association_matrix,
    PRIMARY_FEATURES,
    WHOLE_EVENT,
    Measure,
    Trend,
    above_trend,
    alias,
    align,
    associations,
    event_table,
    paired,
    reconstruction_measures,
    simulation_measures,
    spearman,
    trend,
)
from k4bench.analysis.event_size import CollectionSize, OutputFile
from k4bench.analysis.event_source import GENERATOR_COLUMNS, EventSource


def _source(per_event: list[list[dict]], counts: dict | None = None) -> EventSource:
    """A source whose events hold *per_event* particles, each a dict of
    columns over a final, neutral, massless particle at rest at the origin."""
    base = dict.fromkeys(GENERATOR_COLUMNS, 0.0) | {"gen_status": 1, "pdg": 22}
    particles = [base | p for event in per_event for p in event]
    columns = {
        name: np.array([p[name] for p in particles],
                       dtype=np.int64 if name in ("pdg", "gen_status") else float)
        for name in GENERATOR_COLUMNS
    }
    n = len(per_event)
    index = pd.RangeIndex(n, name="entry")
    return EventSource(
        origin="recorded", file="sim.root", sha256=None, label=None, first_entry=0, events=n,
        event_numbers=tuple(range(n)), run_numbers=None,
        offsets=np.array([0, *np.cumsum([len(e) for e in per_event])]),
        columns=columns,
        input_counts=pd.DataFrame(counts or {}, index=index, dtype=float),
        input_energies=pd.DataFrame(index=index),
    )


def _output(name: str, entries: dict[str, list], energies: dict[str, list] | None = None,
            events: int = 3) -> OutputFile:
    energies = energies or {}
    return OutputFile(name=name, bytes=1, events=events, collections=tuple(
        CollectionSize(c, "edm4hep::SimCalorimeterHitCollection", False, 1, 1,
                       entries=tuple(v), energy_gev=tuple(energies[c]) if c in energies else None)
        for c, v in entries.items()
    ))


PION = {"pdg": 211, "charge": 1.0, "px": 3.0, "py": 4.0}
DISPLACED = PION | {"vx": 3.0, "vy": 4.0}
NEUTRINO = {"pdg": 12, "pz": 7.0}


class TestEventTable:
    def test_the_canned_quantities_come_from_the_generator_particles(self):
        table = event_table(_source([[PION, PION | {"gen_status": 2}], [], [DISPLACED, NEUTRINO]]))
        frame = table.frame
        assert frame["n_final"].tolist() == [1, 0, 2]
        assert frame["n_charged"].tolist() == [1, 0, 1]
        assert frame["n_displaced"].tolist() == [0, 0, 1]
        assert frame["max_pt"].tolist()[0] == pytest.approx(5.0)
        assert np.isnan(frame["max_pt"].tolist()[1])
        # The neutrino is not visible.
        assert frame["visible_E"].tolist()[2] == pytest.approx(5.0, rel=1e-3)
        assert [c for c in table.columns if table.columns[c].kind == "truth"] == [
            f.name for f in FEATURES
        ]
        assert table.columns["n_charged"].expression == "count(final & charged)"

    def test_a_sample_of_one_primary_per_event_gets_the_primarys_quantities(self):
        table = event_table(_source([[PION], [DISPLACED], [PION | {"pz": 5.0}]]))
        for feature in PRIMARY_FEATURES:
            assert feature.name in table.columns
        assert table.frame["primary_abs_cos_theta"].tolist() == pytest.approx(
            [0.0, 0.0, np.sqrt(0.5)],
        )
        # A periodic quantity is plotted, never ranked.
        assert "primary_phi" not in table.ranked()
        assert "primary_phi" not in event_table(_source([[PION, PION]])).columns

    def test_what_each_collection_holds_has_its_own_column(self):
        source = _source([[PION]] * 3, counts={"VXDHits": [5, 6, 7]})
        output = _output("rec.root", {"Tracks": [1, 2, 3]}, {"Tracks": [0.5, 1.0, 1.5]})
        table = event_table(source, outputs=[output])
        assert table.frame["n_VXDHits"].tolist() == [5, 6, 7]
        assert table.frame["out_n_Tracks"].tolist() == [1, 2, 3]
        assert table.frame["out_E_Tracks"].tolist() == [0.5, 1.0, 1.5]
        assert table.columns["out_n_Tracks"].kind == "output"
        assert (table.columns["n_VXDHits"].collection, table.columns["n_VXDHits"].file) == (
            "VXDHits", "sim.root",
        )

    def test_a_simulation_output_is_the_events_own_content(self):
        table = event_table(None, content=[_output("out.edm4hep.root", {"ECal": [1, 2, 3]})])
        assert table.frame["n_ECal"].tolist() == [1, 2, 3]
        assert table.columns["n_ECal"].kind == "content"
        assert not table.has_truth

    def test_a_file_measured_without_events_is_left_out(self):
        unmeasured = OutputFile(name="out.root", bytes=1, events=3, collections=(
            CollectionSize("ECal", "t", False, 1, 1),
        ))
        assert event_table(None, content=[unmeasured]) is None
        assert event_table(None) is None

    def test_files_that_disagree_on_their_events_cannot_be_tabulated(self):
        with pytest.raises(ValueError, match="disagree"):
            event_table(_source([[PION]] * 2), outputs=[_output("rec.root", {"T": [1, 2, 3]})])

    def test_files_whose_headers_name_other_events_cannot_be_tabulated(self):
        source = _source([[PION]] * 3)
        output = _output("rec.root", {"T": [1, 2, 3]})
        reordered = replace(output, event_numbers=(0, 2, 1))
        with pytest.raises(ValueError, match="different events at entry 1"):
            event_table(source, outputs=[reordered])
        # Entries without a header, and files without any, are not compared.
        partial = replace(output, event_numbers=(0, None, 2))
        assert event_table(source, outputs=[partial]).events == 3
        assert event_table(source, outputs=[output]).events == 3

    def test_a_file_without_headers_between_two_hides_no_disagreement(self):
        source = _source([[PION]] * 3)
        output = _output("rec.root", {"T": [1, 2, 3]})
        unnumbered = replace(output, name="mid.root", event_numbers=(None, None, None))
        other = replace(output, name="last.root", event_numbers=(0, 1, 5))
        with pytest.raises(ValueError, match="sim.root and last.root .* entry 2"):
            event_table(source, outputs=[unnumbered, other])

    def test_an_expression_reads_the_particles_and_the_columns(self):
        table = event_table(_source([[PION], [PION, PION]]), )
        assert table.evaluate("count(final) * n_charged").tolist() == [1.0, 4.0]


class TestAlias:
    def test_a_name_is_made_one_an_expression_can_use(self):
        assert alias("n_", "Hits-Barrel.v2", "f.root", set()) == "n_Hits_Barrel_v2"
        assert alias("", "2ndHits", "f.root", set()) == "_2ndHits"

    def test_a_collision_gets_the_file_then_a_number(self):
        taken = {"n_final"}
        assert alias("n_", "final", "out/rec.edm4hep.root", taken) == "n_final_rec"
        taken |= {"n_final_rec"}
        assert alias("n_", "final", "other/rec.edm4hep.root", taken) == "n_final_rec_2"

    def test_a_collection_never_takes_a_canned_quantitys_name(self):
        source = _source([[PION]] * 3, counts={"charged": [1, 2, 3]})
        table = event_table(source)
        assert table.frame["n_charged"].tolist() == [1, 1, 1]
        assert table.frame["n_charged_sim"].tolist() == [1, 2, 3]


def _regions(at_location: dict, steps: dict | None = None, events=(0, 1, 2, 3)) -> dict:
    index = pd.Index(events, name="event_number")
    return {
        "at_location": pd.DataFrame(at_location, index=index),
        "steps": pd.DataFrame(steps, index=index) if steps else None,
    }


class TestMeasures:
    def test_a_simulation_run_costs_its_event_and_each_detectors_time_and_steps(self):
        events = pd.DataFrame({"event_number": [0, 1, 2, 3], "event_time_s": [9.0, 1.0, 2.0, 3.0]})
        measures = simulation_measures(events, _regions(
            {"Tracker": [5.0, 0.1, 0.1, 0.1], "ECal": [1.0, 1.0, 2.0, 3.0]},
            {"Tracker": [10, 20, 30, 40], "ECal": [1, 2, 3, 4]},
        ))
        # Each cost's detectors are ordered by that cost.
        assert [(m.name, m.metric) for m in measures] == [
            (WHOLE_EVENT, "time"), ("ECal", "time"), ("Tracker", "time"),
            ("Tracker", "steps"), ("ECal", "steps"),
        ]
        # The warm-up is left out of every cost, but was counted.
        assert measures[0].values.tolist() == [1.0, 2.0, 3.0]
        assert measures[0].profiled == (0, 1, 2, 3)

    def test_the_detectors_are_ordered_without_the_warm_up(self):
        measures = simulation_measures(None, _regions(
            {"Tracker": [100.0, 0.1, 0.1, 0.1], "ECal": [1.0, 1.0, 1.0, 1.0]},
        ))
        assert [m.name for m in measures] == ["ECal", "Tracker"]

    def test_a_reconstruction_run_costs_each_algorithms_metrics(self):
        index = pd.Index([0, 1, 2], name="event")
        timing = SimpleNamespace(
            algorithms=lambda: ["Fast", "Slow"],
            execute={
                "wall_s": pd.DataFrame({"Fast": [9, 1, 1], "Slow": [9, 5, 5], "Seq": [9, 9, 9]},
                                       index=index, dtype=float),
                "instructions": pd.DataFrame({"Slow": [1e9, 2e9, 3e9]}, index=index),
            },
        )
        measures = reconstruction_measures(None, timing)
        assert [(m.name, m.metric) for m in measures] == [
            ("Slow", "time"), ("Fast", "time"), ("Slow", "instructions"),
        ]
        assert measures[0].values.tolist() == [5.0, 5.0]
        assert reconstruction_measures(None, None) == []

    def test_each_metrics_algorithms_are_ordered_by_that_metric(self):
        index = pd.Index([0, 1, 2], name="event")
        timing = SimpleNamespace(
            algorithms=lambda: ["Fast", "Slow"],
            execute={
                "wall_s": pd.DataFrame({"Fast": [9, 1, 1], "Slow": [9, 5, 5]},
                                       index=index, dtype=float),
                "allocations": pd.DataFrame({"Fast": [1, 1000, 1000], "Slow": [1000, 1, 1]},
                                            index=index, dtype=float),
            },
        )
        measures = reconstruction_measures(None, timing)
        assert [(m.name, m.metric) for m in measures] == [
            ("Slow", "time"), ("Fast", "time"), ("Fast", "allocations"), ("Slow", "allocations"),
        ]


def _measure(values: list[float], events=None) -> Measure:
    events = list(events if events is not None else range(1, len(values) + 1))
    return Measure("Tracking", "time", "s", pd.Series(values, index=events),
                   tuple([0, *events]))


class TestMatching:
    def test_a_measure_matches_the_table_when_profiling_counted_its_entries(self):
        table = event_table(_source([[PION]] * 4))
        assert align(_measure([1.0, 2.0, 3.0]), table) is None

    @pytest.mark.parametrize(("profiled", "why"), [
        ((0, 1, 1, 2), "twice"),
        ((0, 1, 2), "counted 3 events, numbered 0–2"),
        ((10, 11, 12, 13), "numbered 10–13"),
    ])
    def test_events_that_are_not_the_tables_entries_are_not_matched(self, profiled, why):
        table = event_table(_source([[PION]] * 4))
        measure = Measure("Tracking", "time", "s", pd.Series(dtype=float), profiled)
        assert why in align(measure, table)

    def test_pairs_keep_the_events_both_have(self):
        pairs = paired(_measure([1.0, 2.0, np.nan]), pd.Series([0.0, 5.0, np.nan, 7.0]))
        assert pairs.index.tolist() == [1]
        assert pairs.loc[1].tolist() == [5.0, 1.0]


class TestAssociations:
    @staticmethod
    def _table(columns: dict[str, list]) -> SimpleNamespace:
        frame = pd.DataFrame(columns, index=pd.RangeIndex(len(next(iter(columns.values())))))
        return SimpleNamespace(frame=frame, ranked=lambda: list(columns))

    def test_columns_rank_by_the_size_of_their_correlation_sign_kept(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        table = self._table({
            "noise": [0, 3, 1, 5, 2, 9, 4],
            "falls": [0, 60, 50, 40, 30, 20, 10],
            "rises": [0, 1, 2, 4, 3, 5, 6],
            "constant": [1] * 7,
            "sparse": [0, 1, np.nan, np.nan, np.nan, np.nan, 2],
        })
        ranked = associations(_measure(y), table)
        assert ranked.index.tolist() == ["falls", "rises", "noise"]
        assert ranked.loc["falls", "rho"] == pytest.approx(-1.0)
        assert ranked.loc["rises", "rho"] == pytest.approx(0.942857, rel=1e-5)
        assert ranked["n"].tolist() == [6, 6, 6]

    def test_a_correlation_needs_enough_varied_events(self):
        assert spearman(pd.Series([1, 2, 3, 4]), pd.Series([1, 2, 3, 4])) is None
        assert spearman(pd.Series([1] * 6), pd.Series(range(6))) is None
        assert spearman(pd.Series(range(6)), pd.Series(range(6))) == pytest.approx(1.0)


class TestAssociationMatrix:
    def test_each_cost_against_the_quantities_most_associated_with_any(self):
        table = event_table(None, content=[_output("out.root", {
            "A": [0, 1, 2, 3, 4, 5, 6], "B": [6, 5, 4, 3, 2, 1, 0], "C": [0, 1, 4, 2, 6, 3, 5],
            "Flat": [1] * 7,
        }, events=7)])
        rising = _measure([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        falling = Measure("Refit", "time", "s", pd.Series([6.0, 5, 4, 3, 2, 1], index=range(1, 7)),
                          tuple(range(7)))
        unmatched = Measure("Other", "time", "s", pd.Series([1.0]), (0, 5))
        rho, n, similar = association_matrix([rising, falling, unmatched], table, columns=2,
                                             per_row=3)
        # The unmatched cost has no row; each row's three strongest columns,
        # the strongest overall first, at most two.
        assert rho.index.tolist() == ["Tracking", "Refit"]
        # B mirrors A, so it is folded into A's column; C takes the second.
        assert rho.columns.tolist() == ["n_A", "n_C"]
        assert similar == {"n_A": ["n_B"], "n_C": []}
        assert rho.loc["Tracking", "n_A"] == pytest.approx(1.0)
        assert rho.loc["Refit", "n_A"] == pytest.approx(-1.0)
        assert n.loc["Refit", "n_C"] == 6


class TestTrend:
    def test_a_line_is_fitted_robustly(self):
        x = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
        y = 2 * x + 1
        y[7] = 100.0
        line = trend(x, y)
        assert (line.slope, line.intercept) == pytest.approx((2.0, 1.0))

    @pytest.mark.parametrize("x", [[1.0, 2, 3, 4], [3.0] * 8])
    def test_no_line_through_too_few_or_undistinguished_events(self, x):
        assert trend(pd.Series(x), pd.Series(range(len(x)), dtype=float)) is None

    def test_the_events_well_above_the_line_are_listed_furthest_first(self):
        x = np.arange(1.0, 21.0)
        y = 2 * x + np.tile([0.1, -0.1], 10)
        y[4], y[9] = 30.0, 60.0
        pairs = pd.DataFrame({"x": x, "y": y}, index=pd.Index(range(1, 21), name="event"))
        above = above_trend(pairs, trend(pairs["x"], pairs["y"]))
        assert above.index.tolist() == [10, 5]
        assert above.loc[10, "ratio"] == pytest.approx(3.0, rel=1e-2)
        assert above.loc[10, "residual"] == pytest.approx(40.0, rel=1e-2)

    def test_none_is_above_a_line_nothing_spreads_around(self):
        pairs = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)})
        assert above_trend(pairs, Trend(1.0, 0.0)).empty

    def test_an_event_above_a_line_most_lie_exactly_on_is_listed(self):
        x = np.arange(20.0)
        y = 2 * x + 1
        y[-1] = 1000.0
        pairs = pd.DataFrame({"x": x, "y": y})
        above = above_trend(pairs, trend(pairs["x"], pairs["y"]))
        assert above.index.tolist() == [19]
        assert above.loc[19, "residual"] == pytest.approx(961.0)

    def test_an_event_above_a_count_that_never_changes_is_listed(self):
        pairs = pd.DataFrame({"x": np.arange(10.0, 60.0), "y": np.full(50, 120.0)})
        pairs.loc[7, "y"] = 9000.0
        above = above_trend(pairs, trend(pairs["x"], pairs["y"]))
        assert above.index.tolist() == [7]

    def test_no_ratio_where_the_line_is_near_zero(self):
        x = np.arange(10.0)
        y = np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 0.01, 5.0]) + np.tile([0.001, -0.001], 5)
        pairs = pd.DataFrame({"x": x, "y": y})
        above = above_trend(pairs, Trend(0.0, 0.0))
        assert 9 in above.index
        assert np.isnan(above.loc[9, "ratio"])
