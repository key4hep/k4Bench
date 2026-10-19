"""What each output file holds on disk (:mod:`k4bench.analysis.event_size`),
measured from stand-ins for uproot's files and branches."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from k4bench.analysis import event_size
from k4bench.analysis.event_size import (
    FRAME_PARAMETERS,
    OutputFile,
    collection_sizes,
    describe_compression,
    load_event_size,
    measure_outputs,
    read_event_size,
    record_event_size,
)


def _branch(name: str, compressed: int = 0, uncompressed: int = 0, *subs) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, compressed_bytes=compressed, uncompressed_bytes=uncompressed, branches=list(subs),
    )


#: An events tree as podio splits it: hits with their data members as
#: sub-branches, a relation and a vector member, a subset collection, a
#: collection whose name begins another's, and the frame parameters.
BRANCHES = [
    _branch("Hits", 10, 20, _branch("Hits.cellID", 300, 600), _branch("Hits.energy", 200, 400)),
    _branch("_Hits_contributions", 5, 10, _branch("_Hits_contributions.index", 100, 300)),
    _branch("HitsContributions", 8, 16, _branch("HitsContributions.energy", 1000, 1200)),
    _branch("_HitsContributions_particle", 4, 8, _branch("_HitsContributions_particle.index", 50, 90)),
    _branch("MCParticles", 7, 14, _branch("MCParticles.PDG", 60, 200)),
    _branch("_EventHeader_weights", 30, 40),
    _branch("EventHeader", 9, 18),
    _branch("Selected_objIdx", 3, 6, _branch("Selected_objIdx.index", 12, 24)),
    _branch("GPIntKeys", 2, 4),
    _branch("GPIntValues", 1, 2),
]

COLLECTIONS = {
    "Hits": ("edm4hep::SimCalorimeterHitCollection", False),
    "HitsContributions": ("edm4hep::CaloHitContributionCollection", False),
    "MCParticles": ("edm4hep::MCParticleCollection", False),
    "EventHeader": ("edm4hep::EventHeaderCollection", False),
    "Selected": ("edm4hep::MCParticleCollection", True),
}


def test_a_collection_counts_its_branches_relations_and_members_with_their_sub_branches():
    collections, parameters = collection_sizes(BRANCHES, COLLECTIONS)
    sizes = {c["name"]: (c["compressed_bytes"], c["uncompressed_bytes"]) for c in collections}
    assert sizes == {
        # Its own branch and members, and its relation's.
        "Hits": (10 + 300 + 200 + 5 + 100, 20 + 600 + 400 + 10 + 300),
        # ``_HitsContributions_particle`` is not ``Hits``' though it begins "_Hits".
        "HitsContributions": (8 + 1000 + 4 + 50, 16 + 1200 + 8 + 90),
        "MCParticles": (7 + 60, 14 + 200),
        "EventHeader": (9 + 30, 18 + 40),
        "Selected": (3 + 12, 6 + 24),
    }
    # The branches of no collection are the frame parameters.
    assert parameters == {"compressed_bytes": 3, "uncompressed_bytes": 6}
    assert [c["name"] for c in collections] == list(COLLECTIONS)
    selected = next(c for c in collections if c["name"] == "Selected")
    assert selected["subset"] is True
    assert selected["type"] == "edm4hep::MCParticleCollection"


def test_a_relation_belongs_to_the_longest_collection_name_it_begins_with():
    collections, _ = collection_sizes(
        [_branch("_Jets_rel_from", 5, 5), _branch("Jets_rel", 1, 1), _branch("Jets", 2, 2)],
        {"Jets": ("t", False), "Jets_rel": ("t", False)},
    )
    assert {c["name"]: c["compressed_bytes"] for c in collections} == {"Jets": 2, "Jets_rel": 6}


class _Array:
    def __init__(self, value):
        self._value = value

    def array(self, library):
        assert library == "np"
        return np.array([np.array(self._value)], dtype=object)


class _File:
    """An open uproot file: what each top-level object is, and the objects."""

    def __init__(self, kinds: dict[str, str], objects: dict | None = None, compress: int = 101):
        self._kinds = kinds
        self._objects = objects or {}
        self.file = SimpleNamespace(fCompress=compress)

    def classnames(self, recursive, cycle):
        assert (recursive, cycle) == (False, False)
        return self._kinds

    def __getitem__(self, name):
        return self._objects[name]


def _podio_file(branches=BRANCHES, collections=COLLECTIONS, *, events=4, compress=505) -> _File:
    names = list(collections)
    metadata = {
        f"{event_size._TYPE_INFO}name": _Array(names),
        f"{event_size._TYPE_INFO}dataType": _Array([collections[n][0] for n in names]),
        f"{event_size._TYPE_INFO}isSubset": _Array([collections[n][1] for n in names]),
    }
    return _File(
        {"events": "TTree", "podio_metadata": "TTree", "runs": "TTree"},
        {"events": SimpleNamespace(branches=branches, num_entries=events),
         "podio_metadata": metadata},
        compress,
    )


@pytest.fixture
def opener(monkeypatch):
    """Stand in for uproot: each path opens as the file in ``files[path.name]``,
    and one not there fails to open as uproot fails on a file it cannot read."""
    files: dict = {}

    @contextmanager
    def open_root(path):
        if path.name not in files:
            raise ValueError(f"not a ROOT file: {path}")
        yield files[path.name]

    monkeypatch.setattr(event_size, "_root_opener", lambda: open_root)
    return files


def test_measuring_records_each_files_size_and_a_podio_files_events(tmp_path, opener):
    reco = tmp_path / "reco_REC.edm4hep.root"
    reco.write_bytes(b"x" * 5000)
    aida = tmp_path / "reco_aida.root"
    aida.write_bytes(b"x" * 300)
    opener[reco.name] = _podio_file()
    opener[aida.name] = _File({"MyHistogram": "TH1D"})
    (tmp_path / "cdb.log").write_text("x" * 7)

    record = measure_outputs({
        "reco_REC.edm4hep.root": reco, "reco_aida.root": aida, "cdb.log": tmp_path / "cdb.log",
    })
    assert record["schema_version"] == 1
    rec, histograms, log = record["files"]
    assert (rec["name"], rec["bytes"], rec["events"], rec["compression"]) == (
        "reco_REC.edm4hep.root", 5000, 4, 505,
    )
    assert [c["name"] for c in rec["collections"]] == list(COLLECTIONS)
    assert rec["parameters"] == {"compressed_bytes": 3, "uncompressed_bytes": 6}
    # Neither a file of histograms nor a log has events.
    assert histograms == {"name": "reco_aida.root", "bytes": 300}
    assert log == {"name": "cdb.log", "bytes": 7}


def test_a_file_that_cannot_be_read_keeps_its_size_and_hides_no_other(tmp_path, opener, capsys):
    broken = tmp_path / "broken.root"
    broken.write_bytes(b"not root")
    good = tmp_path / "good.root"
    good.write_bytes(b"x" * 10)
    opener[good.name] = _podio_file()
    record = measure_outputs({"broken.root": broken, "good.root": good})
    assert record["files"][0] == {"name": "broken.root", "bytes": 8}
    assert record["files"][1]["events"] == 4
    assert "could not read the events of broken.root" in capsys.readouterr().out


def test_without_uproot_every_file_keeps_its_size(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "uproot", None)
    out = tmp_path / "out.edm4hep.root"
    out.write_bytes(b"x" * 42)
    assert measure_outputs({"out.edm4hep.root": out})["files"] == [
        {"name": "out.edm4hep.root", "bytes": 42},
    ]
    assert "uproot is not available" in capsys.readouterr().out


def test_a_failed_measurement_costs_only_the_record(tmp_path, monkeypatch, capsys):
    def fail(files, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(event_size, "measure_outputs", fail)
    path = tmp_path / "baseline_event_size.json"
    record_event_size(path, {})
    assert not path.exists()
    assert "no event size recorded" in capsys.readouterr().out


def test_a_record_that_cannot_be_written_costs_only_the_record(tmp_path, capsys):
    # Its directory is missing, so writing it fails as a full or read-only disk would.
    path = tmp_path / "missing" / "baseline_event_size.json"
    record_event_size(path, {})
    assert not path.exists()
    assert "no event size recorded" in capsys.readouterr().out


def _write(path: Path, files: list[dict], version: int = 1) -> Path:
    path.write_text(json.dumps({"schema_version": version, "files": files}))
    return path


def _measured(tmp_path, opener, events: int = 4) -> dict:
    out = tmp_path / "out.edm4hep.root"
    out.write_bytes(b"x" * 4096)
    opener[out.name] = _podio_file(events=events)
    return measure_outputs({out.name: out})["files"][0]


def test_per_event_gives_each_collection_in_kb_per_event_largest_first(tmp_path, opener):
    (output,) = read_event_size(
        _write(tmp_path / "baseline_event_size.json", [_measured(tmp_path, opener)]),
    )
    assert output.has_events
    per_event = output.per_event()
    assert per_event.index.to_list() == [
        "HitsContributions", "Hits", "MCParticles", "EventHeader", "Selected", FRAME_PARAMETERS,
    ]
    # 1062 bytes over 4 events.
    assert per_event.loc["HitsContributions", "compressed_kb"] == pytest.approx(1062 / 1024 / 4)
    assert per_event.loc["Hits", "uncompressed_kb"] == pytest.approx(1330 / 1024 / 4)
    assert per_event["share"].sum() == pytest.approx(1.0)
    assert output.event_bytes == 615 + 1062 + 67 + 39 + 15 + 3


def test_a_file_without_events_has_no_collections(tmp_path):
    (output,) = read_event_size(
        _write(tmp_path / "baseline_event_size.json", [{"name": "reco_aida.root", "bytes": 300}]),
    )
    assert output == OutputFile(name="reco_aida.root", bytes=300)
    assert not output.has_events
    assert output.per_event().empty


def test_one_unreadable_record_hides_no_other_configuration(tmp_path, opener):
    _write(tmp_path / "baseline_event_size.json", [_measured(tmp_path, opener)])
    _write(tmp_path / "variant_a_event_size.json", [], version=3)
    (tmp_path / "variant_b_event_size.json").write_text('{"files": [{"bytes": 3}]}')
    files, unreadable = load_event_size(tmp_path)
    assert list(files) == ["baseline"]
    assert set(unreadable) == {"variant_a", "variant_b"}
    assert "future schema_version 3" in unreadable["variant_a"]
    assert "malformed" in unreadable["variant_b"]


def test_a_record_with_a_count_that_is_no_number_is_malformed(tmp_path, opener):
    measured = _measured(tmp_path, opener)
    path = _write(tmp_path / "baseline_event_size.json", [{**measured, "events": "invalid"}])
    with pytest.raises(ValueError, match="malformed"):
        read_event_size(path)


def test_a_run_without_records_has_none(tmp_path):
    assert load_event_size(tmp_path) == ({}, {})


@pytest.mark.parametrize(("setting", "words"), [
    (101, "ZLIB level 1"),
    (505, "ZSTD level 5"),
    (404, "LZ4 level 4"),
    (207, "LZMA level 7"),
    (0, "uncompressed"),
    (None, "unknown"),
    (909, "algorithm 9 level 9"),
])
def test_compression_settings_are_written_out(setting, words):
    assert describe_compression(setting) == words


# ---------------------------------------------------------------------------
# What each event holds
# ---------------------------------------------------------------------------


class _Leaf:
    """A branch of a stand-in events tree: read with the count interpretation
    it gives each entry's element count, otherwise each entry's values."""

    def __init__(self, per_entry: list):
        self._per_entry = per_entry

    def array(self, *, library, entry_stop=None, interpretation=None):
        assert library == "np"
        entries = self._per_entry[:entry_stop]
        if interpretation is not None:
            assert interpretation == "count"
            return np.array([len(e) for e in entries])
        return np.array([np.array(e) for e in entries] + [None], dtype=object)[:-1]


class _Tree:
    def __init__(self, branches, num_entries: int, leaves: dict[str, list]):
        self.branches = branches
        self.num_entries = num_entries
        self._leaves = leaves

    def __getitem__(self, name):
        leaf = self._leaves[name]
        return leaf if isinstance(leaf, _Leaf) else _Leaf(leaf)


#: Four events: the hits and their energies, the header, and a subset
#: collection; MCParticles' branch cannot be read.
LEAVES = {
    "Hits": [[0, 0], [], [0], [0, 0, 0]],
    "Hits/Hits.energy": [[0.5, 0.25], [], [1.0], [0.1, 0.2, 0.3]],
    "HitsContributions": [[0] * 3, [], [0], [0] * 5],
    "EventHeader": [[0]] * 4,
    "EventHeader/EventHeader.eventNumber": [[10], [11], [12], [13]],
    "EventHeader/EventHeader.runNumber": [[7]] * 4,
    "Selected_objIdx": [[0], [0, 0], [], [0]],
}


def _content_file(leaves=LEAVES, events=4) -> _File:
    podio = _podio_file(events=events)
    podio._objects["events"] = _Tree(BRANCHES, events, leaves)
    return podio


@pytest.fixture
def counting(monkeypatch):
    monkeypatch.setattr(event_size, "_count_interpretation", lambda: "count")


def test_per_event_measuring_records_each_collections_entries_energies_and_the_headers(
    tmp_path, opener, counting, capsys,
):
    out = tmp_path / "out.edm4hep.root"
    out.write_bytes(b"x" * 10)
    opener[out.name] = _content_file()
    (record,) = measure_outputs({out.name: out}, per_event=True)["files"]
    collections = {c["name"]: c for c in record["collections"]}
    assert collections["Hits"]["entries"] == [2, 0, 1, 3]
    assert collections["Hits"]["energy_gev"] == [0.75, 0.0, 1.0, 0.6]
    assert collections["HitsContributions"]["entries"] == [3, 0, 1, 5]
    # Not a calorimeter hit collection: no energies.
    assert "energy_gev" not in collections["HitsContributions"]
    # A subset collection is counted from its index branch.
    assert collections["Selected"]["entries"] == [1, 2, 0, 1]
    assert record["event_headers"] == {"event_number": [10, 11, 12, 13], "run_number": [7] * 4}
    # One that cannot be read has no counts, never zero ones, and keeps its bytes.
    assert "entries" not in collections["MCParticles"]
    assert collections["MCParticles"]["compressed_bytes"] == 67
    assert "no per-event counts for MCParticles" in capsys.readouterr().out


def test_without_per_event_nothing_per_event_is_read(tmp_path, opener, counting):
    out = tmp_path / "out.edm4hep.root"
    out.write_bytes(b"x")
    opener[out.name] = _content_file()
    (record,) = measure_outputs({out.name: out})["files"]
    assert "event_headers" not in record
    assert not any({"entries", "energy_gev"} & set(c) for c in record["collections"])


def test_a_negative_count_is_no_count(tmp_path, opener, counting, capsys):
    class _Negative(_Leaf):
        def array(self, **kwargs):
            return np.array([1, -1, 0, 2])

    out = tmp_path / "out.edm4hep.root"
    out.write_bytes(b"x")
    opener[out.name] = _content_file({**LEAVES, "Hits": _Negative([])})
    (record,) = measure_outputs({out.name: out}, per_event=True)["files"]
    hits = next(c for c in record["collections"] if c["name"] == "Hits")
    assert "entries" not in hits and "energy_gev" in hits
    assert "negative element count" in capsys.readouterr().out


def test_what_each_event_holds_reads_back_as_frames_by_entry(tmp_path, opener, counting):
    out = tmp_path / "out.edm4hep.root"
    out.write_bytes(b"x")
    opener[out.name] = _content_file()
    files = measure_outputs({out.name: out}, per_event=True)["files"]
    (output,) = read_event_size(_write(tmp_path / "baseline_event_size.json", files))
    counts = output.event_counts()
    assert counts.index.name == "entry" and counts.index.to_list() == [0, 1, 2, 3]
    assert counts["Hits"].to_list() == [2, 0, 1, 3]
    assert "MCParticles" not in counts
    assert output.event_energies().columns.to_list() == ["Hits"]
    assert output.event_numbers == (10, 11, 12, 13)
    assert output.run_numbers == (7, 7, 7, 7)


def test_a_record_without_per_event_content_has_empty_frames(tmp_path, opener):
    (output,) = read_event_size(
        _write(tmp_path / "baseline_event_size.json", [_measured(tmp_path, opener)]),
    )
    assert output.event_counts().empty and len(output.event_counts()) == 4
    assert output.event_numbers is None


@pytest.mark.parametrize("key", ["entries", "energy_gev"])
def test_a_per_event_list_of_the_wrong_length_is_malformed(tmp_path, opener, key):
    measured = _measured(tmp_path, opener)
    measured["collections"][0][key] = [1, 2, 3]
    with pytest.raises(ValueError, match=f"malformed.*{key} of Hits has 3 values for 4 events"):
        read_event_size(_write(tmp_path / "baseline_event_size.json", [measured]))


def test_event_content_reads_a_podio_files_first_entries(counting):
    content = event_size.event_content(_content_file(), entry_stop=2)
    assert content["collections"]["Hits"] == {"entries": [2, 0], "energy_gev": [0.75, 0.0]}
    assert content["event_headers"] == {"event_number": [10, 11], "run_number": [7, 7]}
    assert event_size.event_content(_File({"MyHistogram": "TH1D"})) == {}


def test_an_entry_without_a_header_has_no_numbers(counting):
    leaves = {**LEAVES, "EventHeader/EventHeader.eventNumber": [[10], [], [12], [13]],
              "EventHeader/EventHeader.runNumber": [[7], [], [7], [7]]}
    content = event_size.event_content(_content_file(leaves))
    assert content["event_headers"]["event_number"] == [10, None, 12, 13]
