"""Unit tests for k4bench.analysis.dataflow: building a job's data flow from
what it recorded, reading it back, and joining its algorithms into a graph."""

from __future__ import annotations

import json

import pytest

from k4bench.analysis.dataflow import (
    GAUDI,
    LCIO,
    Collection,
    DataFlow,
    build_dataflow,
    load_dataflow,
    marlin_collection_parameters,
    parse_joboptions,
)
from k4bench.plugin.schema import DATAFLOW_SCHEMA_VERSION

MARLIN_XML = """<?xml version="1.0" encoding="us-ascii"?>
<!-- Loading shared library : libExample.so -->
<marlin>
 <processor name="MyDigi" type="Digi">
  <parameter name="SimHits" type="string" lcioInType="SimTrackerHit">SimDefault </parameter>
  <parameter name="Hits" type="string" lcioOutType="TrackerHitPlane">DigiHits </parameter>
  <parameter name="Resolution" type="float">0.004 </parameter>
 </processor>
 <processor name="MyTracking" type="Tracking">
  <parameter name="HitCollections" type="StringVec" lcioInType="TrackerHitPlane">DigiHits </parameter>
  <parameter name="Truth" type="string" lcioInType="MCParticle">MCParticle </parameter>
  <parameter name="Tracks" type="string" lcioOutType="Track">Tracks </parameter>
 </processor>
 <processor name="MyStatus" type="Status">
  <parameter name="HowOften" type="int">1 </parameter>
 </processor>
</marlin>
"""

JOBOPTIONS = """\
Digi.Parameters = { 'SimHits' : [ 'VertexCollection' ] , 'Resolution' : [ '0.003' ] };
Digi.ProcessorType = 'Digi';
Digi.EDM4hep2LcioTool = EDM4hep2LcioTool/InputConversion;
Digi.InputConversion.convertAll = True;
Digi.InputConversion.collNameMapping = { 'MCParticles' : 'MCParticle' };
Tracking.Parameters = { 'Steps' : [ '[VXD]; MaxCellAngle : 0.01;' ] };
Tracking.ProcessorType = 'Tracking';
Tracking.Lcio2EDM4hepTool = Lcio2EDM4hepTool/OutputConversion;
Tracking.OutputConversion.convertAll = True;
Tracking.OutputConversion.collNameMapping = { 'MCParticle' : 'MCParticles' };
Status.ProcessorType = 'Status';
IOSvc.outputCommands = [ 'keep *' , 'drop Vertex*' ];
ToolSvc.InputConversion.convertAll = True; // unused
"""


def _component(name, type_, *, category="algorithm", impl=None, inputs=None, outputs=None,
               parent=None):
    entry = {"name": name, "category": category, "type": type_, "impl": impl or type_,
             "parent": parent}
    if inputs is not None or outputs is not None:
        entry["inputs"], entry["outputs"] = inputs, outputs
    return entry


def _components(*, handles: bool = True) -> dict:
    """A sequencer running a Reader, two Marlin processors, a Gaudi algorithm
    and the Writer, and one service."""
    h = (lambda ins, outs: {"inputs": ins, "outputs": outs}) if handles else (lambda ins, outs: {})
    return {"components": [
        _component("Top", "Gaudi::Sequencer"),
        _component("Reader", "Reader", parent=0,
                   **h([], ["/Event/MCParticles", "VertexCollection"])),
        _component("Digi", "MarlinProcessorWrapper", impl="Digi", parent=0, **h([], [])),
        _component("Tracking", "MarlinProcessorWrapper", impl="Tracking", parent=0, **h([], [])),
        _component("Status", "MarlinProcessorWrapper", impl="Status", parent=0, **h([], [])),
        _component("Counter", "TrackCounter", parent=0, **h(["Tracks"], ["NTracks"])),
        _component("Writer", "Writer", parent=0, **h([], [])),
        _component("GeoSvc", "GeoSvc", category="service"),
    ]}


@pytest.fixture
def flow() -> DataFlow:
    return DataFlow.from_json(build_dataflow(_components(), JOBOPTIONS, MARLIN_XML))


class TestParsing:
    def test_job_options_are_parsed_as_literals_where_they_are(self):
        options = parse_joboptions(JOBOPTIONS)
        assert options["Digi"]["Parameters"]["SimHits"] == ["VertexCollection"]
        assert options["Digi"]["EDM4hep2LcioTool"] == "EDM4hep2LcioTool/InputConversion"
        assert options["Digi.InputConversion"]["collNameMapping"] == {"MCParticles": "MCParticle"}
        # A ';' inside a value does not end it, and a trailing comment is not part of it.
        assert options["Tracking"]["Parameters"]["Steps"] == ["[VXD]; MaxCellAngle : 0.01;"]
        assert options["ToolSvc.InputConversion"]["convertAll"] is True

    def test_marlin_lists_only_the_collection_parameters_with_their_direction(self):
        params = marlin_collection_parameters(MARLIN_XML)
        assert params["Digi"] == {
            "SimHits": {"direction": "input", "type": "SimTrackerHit", "default": ["SimDefault"]},
            "Hits": {"direction": "output", "type": "TrackerHitPlane", "default": ["DigiHits"]},
        }
        assert params["Status"] == {}


class TestBuild:
    def test_algorithms_in_order_without_sequencers_or_services(self, flow):
        assert [a.name for a in flow.algorithms] == [
            "Reader", "Digi", "Tracking", "Status", "Counter", "Writer",
        ]
        assert flow.handles_recorded and flow.marlin_recorded

    def test_gaudi_handles_are_collections_of_the_event_store(self, flow):
        reader, counter = flow.algorithms[0], flow.algorithms[4]
        assert reader.outputs == (Collection("MCParticles", GAUDI), Collection("VertexCollection", GAUDI))
        assert counter.inputs == (Collection("Tracks", GAUDI),)

    def test_a_marlin_parameter_is_its_job_value_or_its_default(self, flow):
        digi = flow.algorithms[1]
        assert digi.inputs == (Collection("VertexCollection", LCIO, "SimTrackerHit", "SimHits", True),)
        assert digi.outputs == (Collection("DigiHits", LCIO, "TrackerHitPlane", "Hits", False),)
        assert [c.direction for c in digi.converters] == ["edm4hep_to_lcio"]
        assert digi.converters[0].mapping == {"MCParticles": "MCParticle"}

    def test_a_processor_without_collection_parameters_declares_nothing(self, flow):
        assert flow.algorithms[3].inputs == () and flow.algorithms[3].outputs == ()

    def test_the_writer_keeps_its_rules(self, flow):
        assert flow.algorithms[-1].output_commands == ("keep *", "drop Vertex*")

    def test_the_writer_keeps_everything_by_default(self):
        flow = DataFlow.from_json(build_dataflow(_components(), "", MARLIN_XML))
        assert flow.algorithms[-1].output_commands == ("keep *",)

    def test_without_recorded_handles_gaudi_collections_are_unknown(self):
        flow = DataFlow.from_json(build_dataflow(_components(handles=False), JOBOPTIONS, MARLIN_XML))
        assert not flow.handles_recorded
        assert flow.algorithms[0].outputs is None and flow.algorithms[4].inputs is None
        # Marlin's side does not depend on the auditor.
        assert flow.algorithms[1].inputs is not None

    def test_the_reader_can_be_given_its_collections(self):
        flow = DataFlow.from_json(build_dataflow(
            _components(handles=False), JOBOPTIONS, MARLIN_XML, reader_collections=["MCParticles"],
        ))
        assert flow.algorithms[0].outputs == (Collection("MCParticles", GAUDI),)

    def test_an_unreadable_marlin_dump_costs_only_the_marlin_side(self):
        flow = DataFlow.from_json(build_dataflow(_components(), JOBOPTIONS, "<marlin><oops & </marlin>"))
        assert not flow.marlin_recorded and flow.algorithms[1].inputs is None
        assert flow.algorithms[0].outputs  # the Gaudi side is untouched

    def test_without_marlin_the_processors_collections_are_unknown(self):
        flow = DataFlow.from_json(build_dataflow(_components(), JOBOPTIONS, None))
        assert not flow.marlin_recorded
        assert flow.algorithms[1].inputs is None
        assert flow.algorithms[1].converters  # the converters come from the job options


class TestGraph:
    def test_a_collection_converted_to_lcio_links_its_writer_to_its_reader(self, flow):
        graph = flow.graph()
        (link,) = graph.edges[("Reader", "Digi")]
        assert (link.name, link.store, link.source, link.source_store) == (
            "VertexCollection", LCIO, "VertexCollection", GAUDI,
        )
        # Renamed on the way, and read under the new name.
        (truth,) = [x for x in graph.edges[("Reader", "Tracking")] if x.name == "MCParticle"]
        assert truth.source == "MCParticles"

    def test_marlin_processors_link_through_the_lcio_event(self, flow):
        (link,) = flow.graph().edges[("Digi", "Tracking")]
        assert link.name == "DigiHits" and link.type == "TrackerHitPlane"

    def test_a_collection_converted_back_reaches_gaudi_algorithms(self, flow):
        (link,) = flow.graph().edges[("Tracking", "Counter")]
        assert (link.name, link.store, link.source_store) == ("Tracks", GAUDI, LCIO)

    def test_the_writer_reads_what_its_rules_keep(self, flow):
        graph = flow.graph()
        written = {link.name for (p, c), links in graph.edges.items() if c == "Writer" for link in links}
        assert written == {"MCParticles", "DigiHits", "Tracks", "NTracks"}

    def test_a_read_nobody_writes_is_unresolved(self):
        components = _components()
        components["components"][5]["inputs"] = ["Tracks", "Missing"]
        graph = DataFlow.from_json(build_dataflow(components, JOBOPTIONS, MARLIN_XML)).graph()
        assert [(c, r.name) for c, r in graph.unresolved] == [("Counter", "Missing")]

    def test_a_read_before_its_write_falls_back_to_the_writer(self):
        components = _components()
        components["components"][1]["inputs"] = ["NTracks"]
        graph = DataFlow.from_json(build_dataflow(components, JOBOPTIONS, MARLIN_XML)).graph()
        assert ("Counter", "Reader") in graph.edges

    def test_an_algorithm_reading_what_it_writes_has_no_edge_to_itself(self):
        components = _components()
        components["components"][5]["inputs"] = ["NTracks"]
        graph = DataFlow.from_json(build_dataflow(components, JOBOPTIONS, MARLIN_XML)).graph()
        assert ("Counter", "Counter") not in graph.edges
        assert [(c, r.name) for c, r in graph.unresolved] == [("Counter", "NTracks")]


def test_load_reads_every_label(tmp_path):
    for label in ("baseline", "variant_x"):
        doc = build_dataflow(_components(), JOBOPTIONS, MARLIN_XML)
        (tmp_path / f"{label}_dataflow.json").write_text(json.dumps(doc))
    flows = load_dataflow(tmp_path)
    assert sorted(flows) == ["baseline", "variant_x"]
    assert flows["baseline"].algorithms[0].name == "Reader"


def test_keys_a_newer_writer_adds_are_ignored():
    doc = build_dataflow(_components(), JOBOPTIONS, MARLIN_XML)
    expected = DataFlow.from_json(doc)
    for entry in doc["algorithms"]:
        for collection in (entry["inputs"] or []) + (entry["outputs"] or []):
            collection["added"] = 1
        for converter in entry.get("converters", []):
            converter["added"] = 1
    assert any(entry.get("converters") for entry in doc["algorithms"])
    assert DataFlow.from_json(doc) == expected


def test_a_newer_schema_is_refused(tmp_path):
    doc = build_dataflow(_components(), JOBOPTIONS, MARLIN_XML)
    doc["schema_version"] = DATAFLOW_SCHEMA_VERSION + 1
    (tmp_path / "baseline_dataflow.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="future schema_version"):
        load_dataflow(tmp_path)
