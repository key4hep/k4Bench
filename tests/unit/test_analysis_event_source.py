"""What went into a run's events (:mod:`k4bench.analysis.event_source`),
recorded from stand-ins for uproot's files and read back."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from k4bench.analysis import event_size
from k4bench.analysis.event_source import (
    GENERATOR_COLUMNS,
    PARTICLE_FIELDS,
    EventSource,
    describe_source,
    load_event_source,
    read_event_source,
    record_event_source,
)


class _Leaf:
    def __init__(self, per_entry: list):
        self._per_entry = per_entry

    def array(self, *, library, entry_stop=None, interpretation=None):
        assert library == "np"
        entries = self._per_entry[:entry_stop]
        if interpretation is not None:
            return np.array([len(e) for e in entries])
        return np.array([np.array(e) for e in entries] + [None], dtype=object)[:-1]


class _Array:
    def __init__(self, value):
        self._value = value

    def array(self, library):
        return np.array([np.array(self._value)], dtype=object)


class _File:
    def __init__(self, collections: dict[str, tuple[str, bool]], leaves: dict, events: int):
        info = event_size._TYPE_INFO
        names = list(collections)
        self._objects = {
            "podio_metadata": {
                f"{info}name": _Array(names),
                f"{info}dataType": _Array([collections[n][0] for n in names]),
                f"{info}isSubset": _Array([collections[n][1] for n in names]),
            },
            "events": _Tree(leaves, events),
        }

    def classnames(self, recursive, cycle):
        return {"events": "TTree", "podio_metadata": "TTree"}

    def __getitem__(self, name):
        return self._objects[name]


class _Tree:
    def __init__(self, leaves: dict, events: int):
        self._leaves = leaves
        self.num_entries = events
        self.branches = []

    def __getitem__(self, name):
        return _Leaf(self._leaves[name])


COLLECTIONS = {
    "MCParticles": ("edm4hep::MCParticleCollection", False),
    "EventHeader": ("edm4hep::EventHeaderCollection", False),
    "ECalHits": ("edm4hep::SimCalorimeterHitCollection", False),
}


def _mc(member: str, per_entry: list) -> tuple[str, list]:
    return f"MCParticles/MCParticles.{member}", per_entry


#: Three events. Event 0: a generator Z (status 2) decaying into two
#: generator muons, and one particle Geant4 made (status 0). Event 1: a
#: photon. Event 2: one charged pion at rest, born 3-4 mm off the axis.
LEAVES = dict([
    _mc("PDG", [[23, 13, -13, 11], [22], [211]]),
    _mc("generatorStatus", [[2, 1, 1, 0], [1], [1]]),
    _mc("charge", [[0.0, -1.0, 1.0, -1.0], [0.0], [1.0]]),
    _mc("mass", [[91.1876, 0.1056583745, 0.1056583745, 0.000511], [0.0], [0.13957]]),
    _mc("momentum.x", [[0.0, 3.0, -3.0, 0.01], [0.0], [0.0]]),
    _mc("momentum.y", [[0.0, 4.0, -4.0, 0.0], [0.0], [0.0]]),
    _mc("momentum.z", [[0.0, 0.0, 0.0, 0.0], [5.0], [0.0]]),
    _mc("vertex.x", [[0.0, 0.0, 0.0, 10.0], [0.0], [3.0]]),
    _mc("vertex.y", [[0.0, 0.0, 0.0, 10.0], [0.0], [4.0]]),
    _mc("vertex.z", [[0.0, 0.0, 0.0, 10.0], [0.0], [0.0]]),
    ("EventHeader/EventHeader.eventNumber", [[5], [6], [7]]),
    ("EventHeader/EventHeader.runNumber", [[1], [1], [1]]),
    ("MCParticles", [[0] * 4, [0], [0]]),
    ("EventHeader", [[0], [0], [0]]),
    ("ECalHits", [[0, 0], [], [0]]),
    ("ECalHits/ECalHits.energy", [[0.5, 1.5], [], [2.0]]),
])


@pytest.fixture
def source(tmp_path, monkeypatch) -> Path:
    """A podio file of :data:`LEAVES` that the stand-in uproot opens."""
    path = tmp_path / "input.edm4hep.root"
    path.write_bytes(b"frozen input")
    files = {path.name: _File(COLLECTIONS, LEAVES, events=3)}

    @contextmanager
    def open_root(p):
        if p.name not in files:
            raise ValueError(f"not a ROOT file: {p}")
        yield files[p.name]

    monkeypatch.setattr(event_size, "_root_opener", lambda: open_root)
    monkeypatch.setattr(event_size, "_count_interpretation", lambda: "count")
    return path


def test_the_generator_particles_of_the_events_read_are_kept(source):
    record = describe_source(source, n_events=3, label="baseline")
    assert record["schema_version"] == 1
    assert record["origin"] == "recorded"
    assert record["source"] == {
        "file": "input.edm4hep.root", "first_entry": 0, "events": 3, "label": "baseline",
    }
    generator = record["generator"]
    # The particle Geant4 made is left out.
    assert generator["offsets"] == [0, 3, 4, 5]
    assert generator["pdg"] == [23, 13, -13, 22, 211]
    assert generator["gen_status"] == [2, 1, 1, 1, 1]
    # Six significant digits.
    assert generator["mass"][1] == 0.105658
    assert set(generator) == {"offsets", *GENERATOR_COLUMNS}
    assert record["event_headers"] == {"event_number": [5, 6, 7], "run_number": [1, 1, 1]}
    assert "input_content" not in record and "sha256" not in record["source"]


def test_only_the_events_the_job_processed_are_kept(source):
    record = describe_source(source, n_events=2)
    assert record["source"]["events"] == 2
    assert record["generator"]["offsets"] == [0, 3, 4]
    assert record["event_headers"]["event_number"] == [5, 6]


def test_a_job_that_processed_more_events_than_the_file_holds_has_no_source(source):
    with pytest.raises(ValueError, match="holds 3 events, the job processed 4"):
        describe_source(source, n_events=4)


def test_an_input_also_has_its_content_and_checksum(source):
    record = describe_source(source, n_events=3, input_content=True, checksum=True)
    assert record["source"]["sha256"] == hashlib.sha256(b"frozen input").hexdigest()
    content = {c["name"]: c for c in record["input_content"]["collections"]}
    assert content["ECalHits"] == {
        "name": "ECalHits", "type": "edm4hep::SimCalorimeterHitCollection",
        "entries": [2, 0, 1], "energy_gev": [2.0, 0.0, 2.0],
    }
    assert content["MCParticles"]["entries"] == [4, 1, 1]


def test_recording_writes_a_gzipped_record_that_reads_back(source, tmp_path):
    out = tmp_path / "event_source.json.gz"
    assert record_event_source(out, source, n_events=3, input_content=True, checksum=True)
    raw = json.loads(gzip.decompress(out.read_bytes()))
    assert raw["origin"] == "recorded"

    read = load_event_source(tmp_path)
    assert (read.file, read.events, read.first_entry, read.label) == (
        "input.edm4hep.root", 3, 0, None,
    )
    assert read.sha256 == hashlib.sha256(b"frozen input").hexdigest()
    assert read.event_numbers == (5, 6, 7) and read.run_numbers == (1, 1, 1)
    assert read.offsets.tolist() == [0, 3, 4, 5]
    assert read.event_index.tolist() == [0, 0, 0, 1, 2]
    assert read.columns["pdg"].dtype == np.int64
    assert read.input_counts["ECalHits"].to_list() == [2, 0, 1]
    assert read.input_energies.columns.to_list() == ["ECalHits"]
    assert read.input_counts.index.name == "entry"


def test_what_follows_from_the_generator_columns(source, tmp_path):
    record_event_source(tmp_path / "event_source.json.gz", source, n_events=3)
    p = load_event_source(tmp_path).particles()
    # Every field is described, in the order given.
    assert list(p) == list(PARTICLE_FIELDS)
    # The Z at rest, the muons back to back in the transverse plane.
    assert p["pt"].tolist() == pytest.approx([0.0, 5.0, 5.0, 0.0, 0.0])
    assert p["p"][3] == pytest.approx(5.0)
    assert p["E"][0] == pytest.approx(91.1876)
    assert p["final"].tolist() == [False, True, True, True, True]
    assert p["charged"].tolist() == [False, True, True, False, True]
    assert p["theta"][1] == pytest.approx(np.pi / 2)
    assert p["eta"][1] == pytest.approx(0.0)
    assert p["phi"][2] == pytest.approx(np.arctan2(-4.0, -3.0))
    assert p["vr"][4] == pytest.approx(5.0)
    # Undefined along the beam axis and at rest.
    assert np.isnan(p["eta"][3]) and np.isnan(p["eta"][0])
    assert np.isnan(p["abs_cos_theta"][0])
    assert p["abs_cos_theta"][3] == pytest.approx(1.0)


def test_a_failed_recording_costs_only_the_file(source, tmp_path, capsys):
    out = tmp_path / "event_source.json.gz"
    out.write_bytes(b"a previous run's")
    assert not record_event_source(out, source, n_events=9)
    assert not out.exists()
    assert "no event source recorded" in capsys.readouterr().out


def test_without_uproot_nothing_is_recorded(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "uproot", None)
    (tmp_path / "out.root").write_bytes(b"x")
    assert not record_event_source(tmp_path / "s.json.gz", tmp_path / "out.root", n_events=1)
    assert "uproot is not available" in capsys.readouterr().out


def test_a_file_without_mc_particles_has_no_source(tmp_path, monkeypatch):
    path = tmp_path / "out.root"
    path.write_bytes(b"x")

    @contextmanager
    def open_root(p):
        yield _File({"EventHeader": COLLECTIONS["EventHeader"]}, LEAVES, events=3)

    monkeypatch.setattr(event_size, "_root_opener", lambda: open_root)
    with pytest.raises(ValueError, match="no MCParticle collection"):
        describe_source(path, n_events=1)


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"origin": "guessed"}, "unknown origin"),
    ({"n_events": 0}, "must be positive"),
])
def test_a_source_needs_a_known_origin_and_events(source, kwargs, match):
    with pytest.raises(ValueError, match=match):
        describe_source(source, **{"n_events": 3, **kwargs})


def _raw(source: Path) -> dict:
    return describe_source(source, n_events=3, input_content=True)


@pytest.mark.parametrize("corrupt", [
    lambda r: r.update(origin="guessed"),
    lambda r: r["generator"].update(offsets=[0, 3, 4]),
    lambda r: r["generator"].update(offsets=[0, 4, 3, 5]),
    lambda r: r["generator"].update(pdg=[23]),
    lambda r: r["event_headers"].update(event_number=[5]),
    lambda r: r["input_content"]["collections"][0].update(entries=[1]),
    lambda r: r.pop("generator"),
])
def test_a_malformed_source_is_refused(source, corrupt):
    raw = _raw(source)
    corrupt(raw)
    with pytest.raises(ValueError, match="malformed event source"):
        EventSource.from_json(raw)


def test_a_future_or_unreadable_file_is_refused(tmp_path):
    path = tmp_path / "event_source.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"schema_version": 2}).encode()))
    with pytest.raises(ValueError, match="future schema_version 2"):
        read_event_source(path)
    path.write_bytes(b"not gzip")
    with pytest.raises(ValueError, match="not a gzipped JSON event source"):
        read_event_source(path)


def test_a_run_without_a_source_has_none(tmp_path):
    assert load_event_source(tmp_path) is None
