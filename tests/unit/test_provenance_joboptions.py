"""Unit tests for :mod:`k4bench.provenance.joboptions` — a k4run job's
resolved Gaudi options, and what changed between two jobs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from k4bench.provenance.joboptions import (
    ABSENT,
    OptionChange,
    diff_joboptions,
    parse_joboptions,
    stack_aliases,
)

NIGHTS = Path(__file__).resolve().parents[1] / "fixtures" / "reco_nights"

DUMP = """\
ApplicationMgr.EvtMax = 100;
ApplicationMgr.ExtSvc = [ 'AuditorSvc/AuditorSvc' , 'GeoSvc/GeoSvc' ];
Tracking.Parameters = { 'MaxChi2' : [ '1000' ] , 'Smooth' : [ 'false' ] };
Tracking.OutputLevel = 4; // unused
RndmGenSvc.Engine.Seeds = [ 1234567 , 0 ];
Odd.Value = not a literal;
"""


@pytest.fixture
def dump(tmp_path) -> Path:
    path = tmp_path / "baseline_joboptions.opts"
    path.write_text(DUMP)
    return path


def test_a_dump_is_read_as_literals_by_component(dump):
    options = parse_joboptions(dump)
    assert options["ApplicationMgr"] == {
        "EvtMax": 100, "ExtSvc": ["AuditorSvc/AuditorSvc", "GeoSvc/GeoSvc"],
    }
    assert options["Tracking"]["Parameters"] == {"MaxChi2": ["1000"], "Smooth": ["false"]}
    # A trailing "// unused" is the dump's remark, not the value.
    assert options["Tracking"]["OutputLevel"] == 4
    # A tool or engine belongs to the component it is named after.
    assert options["RndmGenSvc.Engine"] == {"Seeds": [1234567, 0]}
    assert options["Odd"]["Value"] == "not a literal"


def _options(**components) -> dict:
    return {name: dict(props) for name, props in components.items()}


def test_an_unchanged_configuration_has_no_diff(dump):
    options = parse_joboptions(dump)
    diff = diff_joboptions(options, options)
    assert (diff.added, diff.removed, diff.changed) == ([], [], [])


def test_a_mapping_property_changes_entry_by_entry():
    a = _options(Tracking={"Parameters": {"MaxChi2": ["1000"], "Smooth": ["false"]}})
    b = _options(Tracking={"Parameters": {"MaxChi2": ["500"], "Kalman": ["true"]}})
    assert diff_joboptions(a, b).changed == [
        OptionChange("Tracking", "Parameters['MaxChi2']", ["1000"], ["500"]),
        OptionChange("Tracking", "Parameters['Smooth']", ["false"], ABSENT),
        OptionChange("Tracking", "Parameters['Kalman']", ABSENT, ["true"]),
    ]


def test_components_only_one_side_configures_are_listed_not_diffed():
    a = _options(Reader={"OutputLevel": 4}, Tracking={"OutputLevel": 4})
    b = _options(Reader={"OutputLevel": 3}, Truth={"OutputLevel": 4})
    diff = diff_joboptions(a, b)
    assert (diff.added, diff.removed) == (["Truth"], ["Tracking"])
    assert diff.changed == [OptionChange("Reader", "OutputLevel", 4, 3)]


def test_k4benchs_own_settings_and_paths_are_not_configuration():
    a = _options(
        k4BenchAuditor={"ComponentsOutput": "/logs/baseline_components.json"},
        JobOptionsSvc={"DUMPFILE": "/logs/baseline_joboptions.opts", "OutputLevel": 3},
        IOSvc={"Input": ["/tmp/k4bench-inputs/aa2c432059f291bb/zbb.edm4hep.root"]},
        Pandora={"Settings": "/tmp/k4bench-baseline-gd3mmyej/Pandora/Default.xml"},
    )
    b = _options(
        k4BenchAuditor={"ComponentsOutput": "/logs/variant_components.json"},
        JobOptionsSvc={"DUMPFILE": "/logs/variant_joboptions.opts", "OutputLevel": 3},
        IOSvc={"Input": ["/tmp/k4bench-inputs/12d4d8ce61c32f56/zbb.edm4hep.root"]},
        Pandora={"Settings": "/tmp/k4bench-variant-f_hspakq/Pandora/Default.xml"},
    )
    diff = diff_joboptions(a, b)
    assert (diff.added, diff.removed, diff.changed) == ([], [], [])


def test_another_input_file_still_shows():
    a = _options(IOSvc={"Input": ["/tmp/k4bench-inputs/aa2c432059f291bb/zbb.root"]})
    b = _options(IOSvc={"Input": ["/tmp/k4bench-inputs/aa2c432059f291bb/zcc.root"]})
    assert len(diff_joboptions(a, b).changed) == 1


def test_stack_aliases_name_the_nightlys_install_tree_and_view():
    run_info = json.loads((NIGHTS / "2026-10-17" / "run_info.json").read_text())
    assert stack_aliases(run_info) == {
        "/cvmfs/sft-nightlies.cern.ch/lcg/nightlies/devkey-head/Sat": "<stack>",
        "/cvmfs/sft-nightlies.cern.ch/lcg/views/devkey-head/Sat/x86_64-el9-gcc16-opt": "<view>",
    }
    assert stack_aliases({}) == {}


def _night(night: str, label: str = "baseline"):
    run_info = json.loads((NIGHTS / night / "run_info.json").read_text())
    return parse_joboptions(NIGHTS / night / f"{label}_joboptions.opts"), stack_aliases(run_info)


def test_cld_two_nights_of_one_configuration_are_the_same():
    (a, aliases_a), (b, aliases_b) = _night("2026-10-16"), _night("2026-10-17")
    # The stack is mounted under the weekday of its nightly …
    assert a["GeoSvc"]["detectors"] != b["GeoSvc"]["detectors"]
    # … which is where it was installed, not what was configured.
    diff = diff_joboptions(a, b, aliases_a=aliases_a, aliases_b=aliases_b)
    assert (diff.added, diff.removed, diff.changed) == ([], [], [])
    assert diff_joboptions(a, b).changed[0].option == "detectors"


def test_cld_truth_tracking_swaps_the_track_finding():
    (a, aliases), (b, _) = _night("2026-10-17"), _night("2026-10-17", "variant_truth_tracking")
    diff = diff_joboptions(a, b, aliases_a=aliases, aliases_b=aliases)
    assert diff.added == ["MyTruthTrackFinder"]
    assert diff.removed == ["ClonesAndSplitTracksFinder", "MyConformalTracking"]
    assert [(c.component, c.option) for c in diff.changed] == [("k4FWCore__Algs", "Members")]
