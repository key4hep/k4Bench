"""Unit tests for k4bench.analysis.plots.dataflow: grouping, layout and figure."""

from __future__ import annotations

import pytest

from k4bench.analysis.dataflow import DataFlow
from k4bench.analysis.plots.dataflow import (
    dataflow_view,
    group_id,
    layered_positions,
    plot_dataflow,
)


def _algorithm(name, impl, *, reads=(), writes=(), framework="marlin", type_="MarlinProcessorWrapper"):
    store = "lcio" if framework == "marlin" else "gaudi"
    return {
        "name": name, "type": type_, "impl": impl, "framework": framework,
        "inputs": [{"name": c, "store": store} for c in reads],
        "outputs": [{"name": c, "store": store} for c in writes],
        "converters": [],
    }


@pytest.fixture
def flow() -> DataFlow:
    """Two digitisers of one class feed tracking, whose tracks are converted to
    EDM4hep for a Gaudi analysis; a status monitor declares nothing; the Writer
    saves what the analysis wrote."""
    return DataFlow.from_json({
        "schema_version": 1, "handles_recorded": True, "marlin_recorded": True,
        "algorithms": [
            _algorithm("DigiA", "Digi", writes=["HitsA"]),
            _algorithm("DigiB", "Digi", writes=["HitsB"]),
            {**_algorithm("Tracking", "Tracking", reads=["HitsA", "HitsB"], writes=["Tracks"]),
             "converters": [{"direction": "lcio_to_edm4hep", "convert_all": True, "mapping": {}}]},
            _algorithm("Status", "Status"),
            _algorithm("Analysis", "Analysis", reads=["Tracks"], writes=["Summary"],
                       framework="gaudi", type_="Analysis"),
            {**_algorithm("Writer", "Writer", framework="gaudi", type_="Writer"),
             "output_commands": ["keep Summary"]},
        ],
    })


SECONDS = {"DigiA": 0.1, "DigiB": 0.3, "Tracking": 2.0, "Analysis": 0.01, "Writer": 0.02}


class TestView:
    def test_a_class_with_several_instances_is_one_node(self, flow):
        view = dataflow_view(flow, SECONDS)
        ids = [n.id for n in view.nodes]
        assert ids == [group_id("Digi"), "Tracking", "Status", "Analysis", "Writer"]
        digi = view.nodes[0]
        assert digi.is_group and digi.members == ("DigiA", "DigiB")
        assert digi.label == "Digi ×2" and digi.seconds == pytest.approx(0.4)
        (link_a, link_b) = sorted(view.edges[(group_id("Digi"), "Tracking")], key=lambda link: link.name)
        assert (link_a.name, link_b.name) == ("HitsA", "HitsB")

    def test_an_expanded_class_shows_each_instance(self, flow):
        view = dataflow_view(flow, SECONDS, {"Digi"})
        assert [n.id for n in view.nodes][:2] == ["DigiA", "DigiB"]
        assert ("DigiA", "Tracking") in view.edges and ("DigiB", "Tracking") in view.edges
        assert view.groups == {"Digi": ("DigiA", "DigiB")}

    def test_the_writers_reads_can_be_left_out(self, flow):
        assert ("Analysis", "Writer") in dataflow_view(flow).edges
        view = dataflow_view(flow, writer_edges=False)
        assert all(consumer != "Writer" for _, consumer in view.edges)
        assert view.sink == "Writer"

    def test_nodes_without_edges_can_be_left_out_but_not_the_writer(self, flow):
        assert "Status" in [n.id for n in dataflow_view(flow).nodes]
        ids = [n.id for n in dataflow_view(flow, writer_edges=False, isolated=False).nodes]
        assert "Status" not in ids and "Writer" in ids

    def test_a_class_another_algorithm_sits_between_is_not_collapsed(self):
        # MergeA feeds Refit, which feeds MergeB: as one node they would loop.
        flow = DataFlow.from_json({
            "schema_version": 1, "handles_recorded": True, "marlin_recorded": True,
            "algorithms": [
                _algorithm("MergeA", "Merge", writes=["Hits"]),
                _algorithm("Refit", "Refit", reads=["Hits"], writes=["Tracks"]),
                _algorithm("MergeB", "Merge", reads=["Tracks"], writes=["AllTracks"]),
            ],
        })
        view = dataflow_view(flow)
        assert [n.id for n in view.nodes] == ["MergeA", "Refit", "MergeB"]
        assert view.groups == {}
        x = {node: p[0] for node, p in layered_positions(view).items()}
        assert x["MergeA"] < x["Refit"] < x["MergeB"]

    def test_a_class_one_instance_of_which_reads_another_is_not_collapsed(self):
        # As one node, RecoA's hits read by RecoB would vanish with the pair.
        flow = DataFlow.from_json({
            "schema_version": 1, "handles_recorded": True, "marlin_recorded": True,
            "algorithms": [
                _algorithm("RecoA", "Reco", writes=["Hits"]),
                _algorithm("RecoB", "Reco", reads=["Hits"], writes=["Tracks"]),
            ],
        })
        view = dataflow_view(flow, isolated=False)
        assert [n.id for n in view.nodes] == ["RecoA", "RecoB"]
        assert view.groups == {} and ("RecoA", "RecoB") in view.edges

    def test_a_node_only_the_writer_reads_from_is_kept_without_the_writers_reads(self):
        flow = DataFlow.from_json({
            "schema_version": 1, "handles_recorded": True, "marlin_recorded": True,
            "algorithms": [
                _algorithm("Producer", "Producer", writes=["Summary"], framework="gaudi", type_="Producer"),
                {**_algorithm("Writer", "Writer", framework="gaudi", type_="Writer"),
                 "output_commands": ["keep Summary"]},
            ],
        })
        view = dataflow_view(flow, writer_edges=False, isolated=False)
        assert [n.id for n in view.nodes] == ["Producer", "Writer"]
        assert view.edges == {}
        # Drawn as a node with collections, in the graph rather than below it.
        positions = layered_positions(view)
        assert positions["Producer"][1] == positions["Writer"][1]
        fig = plot_dataflow(view, positions)
        assert "rgba(0,0,0,0)" not in fig.data[-1].marker.color
        assert not fig.layout.annotations

    def test_unknown_times_stay_unknown(self, flow):
        assert dataflow_view(flow).nodes[1].seconds is None

    def test_a_group_with_an_untimed_member_has_no_total(self, flow):
        assert dataflow_view(flow, {"DigiA": 0.1}).nodes[0].seconds is None


class TestLayout:
    def test_each_node_is_right_of_what_it_reads_and_the_writer_is_last(self, flow):
        view = dataflow_view(flow, SECONDS)
        x = {node: p[0] for node, p in layered_positions(view).items()}
        assert x[group_id("Digi")] < x["Tracking"] < x["Analysis"] < x["Writer"]

    def test_a_node_without_edges_sits_below_the_graph(self, flow):
        positions = layered_positions(dataflow_view(flow, SECONDS))
        assert positions["Status"][1] < min(p[1] for n, p in positions.items() if n != "Status")


class TestFigure:
    def test_the_last_trace_is_the_nodes_carrying_their_ids(self, flow):
        view = dataflow_view(flow, SECONDS)
        nodes = plot_dataflow(view).data[-1]
        assert [row[0] for row in nodes.customdata] == [n.id for n in view.nodes]
        assert list(nodes.marker.symbol) == ["square", "circle", "circle", "circle", "circle"]

    def test_a_node_without_edges_is_hollow_but_the_writer_is_not(self, flow):
        view = dataflow_view(flow, SECONDS, writer_edges=False)
        nodes = plot_dataflow(view).data[-1]
        colors = dict(zip((row[0] for row in nodes.customdata), nodes.marker.color))
        assert colors["Status"] == "rgba(0,0,0,0)"
        assert colors["Writer"] != "rgba(0,0,0,0)"

    def test_hovering_along_an_edge_lists_its_collections(self, flow):
        view = dataflow_view(flow, SECONDS)
        fig = plot_dataflow(view)
        hover = fig.data[-2]
        along = [i for i, t in enumerate(hover.text) if t.startswith("<b>Digi ×2 → Tracking")]
        assert len(along) > 10, "the whole curve answers, not one point"
        assert "HitsA" in hover.text[along[0]] and "HitsB" in hover.text[along[0]]
        # Its ends are left to the nodes.
        (x0, _), (x1, _) = (layered_positions(view)[n] for n in (group_id("Digi"), "Tracking"))
        xs = [hover.x[i] for i in along]
        assert min(xs) > x0 + 0.05 and max(xs) < x1 - 0.05

    def test_the_selected_node_is_outlined(self, flow):
        nodes = plot_dataflow(dataflow_view(flow, SECONDS), selected="Tracking").data[-1]
        assert nodes.marker.line.width[1] > nodes.marker.line.width[0]


def test_dragging_pans_so_a_click_still_picks_a_node(flow):
    layout = plot_dataflow(dataflow_view(flow, SECONDS)).layout
    assert layout.dragmode == "pan" and layout.clickmode == "event+select"
