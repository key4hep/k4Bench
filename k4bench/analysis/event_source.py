"""What went into a run's events: ``event_source.json.gz``.

One file per run, shared by all its configurations, since every configuration
of a run processes the same events. It holds, entry by entry, the generator
particles of each event and, for a job that reads a file (reconstruction on a
frozen simulated input), what that file's collections hold. The per-event
costs (:mod:`k4bench.analysis.components`, the timing plugins' files) can then
be set against what each event is.

Where it comes from
-------------------
:func:`record_event_source` reads a podio file with uproot: a ddsim benchmark's
own output file after its first run, a k4run benchmark's input file before its
first run. Its ``origin`` says how it was made:

- ``recorded``: by the benchmark, from the run's own file;
- ``backfill-verified``: later, from the very file the run read, matched by
  its checksum;
- ``backfill-inferred``: later, from a file that should hold the same
  generator particles, e.g. another detector's simulation of the same sample.

Its ``source`` names the file, its checksum when one was taken, the
configuration whose output it is (simulation only) and the entries read:
``events`` of them from ``first_entry``.

Which event is which
--------------------
The entries are those the job processed, in order: entry *i* is the *i*-th
event the job's profiling counted (``G4Event`` ID in simulation,
``EventContext::evt()`` in reconstruction), as a job reads its input from its
first entry on. The EventHeader's event and run numbers of each entry are kept
beside them (``None`` for an entry without one) but are not the key.

The generator particles
-----------------------
The particles of the generator record (``generatorStatus`` not 0)
(:data:`GENERATOR_COLUMNS`), floats to six significant digits; edm4hep's
units: GeV, mm, and the elementary charge. What transport made of them
(``simulatorStatus``, end points) is not kept: it depends on the detector, so
it would differ between the configurations sharing the file. Neither is the
record wholly the generator's: simulation boosts it by the crossing angle its
steering sets, and the products of a decay the generator assigned but Geant4
performs (a b hadron's, say) start where and as transport left the parent, so
in Z→bb̄ about 1% of the particles, all born off the interaction point, differ
slightly from detector to detector.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from k4bench.analysis import event_size as _event_size
from k4bench.analysis.event_size import (
    event_content,
    podio_collections,
    read_event_headers,
    round_significant,
)
from k4bench.artifacts import EVENT_SOURCE
from k4bench.plugin.schema import EVENT_SOURCE_SCHEMA_VERSION, validate_event_source_schema

#: How an event-source file was made; see the module docstring.
ORIGINS = ("recorded", "backfill-verified", "backfill-inferred")

#: Each generator particle's columns, and the MCParticle member each is read from.
GENERATOR_MEMBERS = {
    "pdg": "PDG",
    "gen_status": "generatorStatus",
    "charge": "charge",
    "mass": "mass",
    "px": "momentum.x",
    "py": "momentum.y",
    "pz": "momentum.z",
    "vx": "vertex.x",
    "vy": "vertex.y",
    "vz": "vertex.z",
}
GENERATOR_COLUMNS = tuple(GENERATOR_MEMBERS)
#: The columns that are whole numbers.
_INTEGER_COLUMNS = frozenset({"pdg", "gen_status"})

#: Every field :meth:`EventSource.particles` gives, in its order: its unit and
#: what it is. The expression help lists them from here.
PARTICLE_FIELDS = {
    "pdg": ("", "PDG code: 11 electron, 13 muon, 22 photon, 211 charged pion, ..., "
                "negative for antiparticles"),
    "gen_status": ("", "generator status: 1 final state, 2 decayed, ..."),
    "charge": ("e", "electric charge"),
    "mass": ("GeV", "mass"),
    "px": ("GeV", "momentum along x"),
    "py": ("GeV", "momentum along y"),
    "pz": ("GeV", "momentum along the beam axis"),
    "vx": ("mm", "production vertex, x"),
    "vy": ("mm", "production vertex, y"),
    "vz": ("mm", "production vertex, along the beam axis"),
    "p": ("GeV", "momentum"),
    "pt": ("GeV", "transverse momentum"),
    "E": ("GeV", "energy"),
    "theta": ("rad", "polar angle from the beam axis, 0 to π"),
    "phi": ("rad", "azimuthal angle, −π to π"),
    "eta": ("", "pseudorapidity; NaN along the beam axis"),
    "abs_cos_theta": ("", "|cos θ|: 0 sideways, 1 along the beams"),
    "vr": ("mm", "production vertex's distance from the beam axis"),
    "final": ("condition", "in the final state (generator status 1)"),
    "charged": ("condition", "charge is not 0"),
}

_MC_PARTICLE_TYPE = "edm4hep::MCParticleCollection"
#: The MCParticle collection's usual name, preferred when a file has several.
_MC_PARTICLES = "MCParticles"


@dataclass(frozen=True)
class EventSource:
    """What went into a run's events, read back from ``event_source.json.gz``."""

    #: One of :data:`ORIGINS`.
    origin: str
    #: The source file's name, its sha256 when one was taken, and the
    #: configuration whose output it is (simulation only).
    file: str
    sha256: str | None
    label: str | None
    #: The source file's entries the events are: *events* of them from
    #: *first_entry*.
    first_entry: int
    events: int
    #: Each event's EventHeader event and run number, ``None`` for one
    #: without; ``None`` when the file had no EventHeader.
    event_numbers: tuple[int | None, ...] | None
    run_numbers: tuple[int | None, ...] | None
    #: Where each event's generator particles begin in :attr:`columns`, and
    #: where the last ends: ``events + 1`` values.
    offsets: np.ndarray
    #: The generator particles, one array per :data:`GENERATOR_COLUMNS`.
    columns: Mapping[str, np.ndarray]
    #: For a job that read the file, its collections' elements and its
    #: calorimeter hit collections' summed energy (GeV) in each event, one
    #: column per collection indexed by ``entry``; empty otherwise.
    input_counts: pd.DataFrame
    input_energies: pd.DataFrame

    @property
    def event_index(self) -> np.ndarray:
        """The event of each generator particle."""
        return np.repeat(np.arange(self.events), np.diff(self.offsets))

    def particles(self) -> dict[str, np.ndarray]:
        """Every field of :data:`PARTICLE_FIELDS` per generator particle: its
        columns and what follows from them. A quantity undefined for a
        particle (``eta`` along the beam axis, ``abs_cos_theta`` at rest) is
        NaN."""
        c = {name: np.asarray(values, dtype=float) for name, values in self.columns.items()}
        with np.errstate(divide="ignore", invalid="ignore"):
            pt = np.hypot(c["px"], c["py"])
            p = np.sqrt(pt**2 + c["pz"] ** 2)
            eta = np.arcsinh(c["pz"] / pt)
            abs_cos_theta = np.abs(c["pz"]) / p
        return {
            **c,
            "p": p,
            "pt": pt,
            "E": np.sqrt(p**2 + c["mass"] ** 2),
            "theta": np.arctan2(pt, c["pz"]),
            "phi": np.arctan2(c["py"], c["px"]),
            "eta": np.where(np.isfinite(eta), eta, np.nan),
            "abs_cos_theta": abs_cos_theta,
            "vr": np.hypot(c["vx"], c["vy"]),
            "final": c["gen_status"] == 1,
            "charged": c["charge"] != 0,
        }

    @classmethod
    def from_json(cls, raw: Mapping, *, source: str | Path | None = None) -> EventSource:
        where = f"{source}: " if source is not None else ""
        try:
            origin = raw["origin"]
            if origin not in ORIGINS:
                raise ValueError(f"unknown origin {origin!r}")
            info = raw["source"]
            events, first_entry = int(info["events"]), int(info["first_entry"])
            generator = raw["generator"]
            offsets = np.asarray(generator["offsets"], dtype=np.int64)
            if len(offsets) != events + 1 or offsets[0] != 0 or (np.diff(offsets) < 0).any():
                raise ValueError(f"offsets do not bound {events} events")
            columns = {
                name: np.asarray(generator[name],
                                 dtype=np.int64 if name in _INTEGER_COLUMNS else float)
                for name in GENERATOR_COLUMNS
            }
            if any(len(values) != offsets[-1] for values in columns.values()):
                raise ValueError(f"generator columns do not all hold {offsets[-1]} particles")
            headers = raw.get("event_headers") or {}
            numbers = {
                key: _optional_ints(headers.get(key), events)
                for key in ("event_number", "run_number")
            }
            counts, energies = _content_frames(raw.get("input_content"), events)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where}malformed event source ({exc})") from exc
        return cls(
            origin=origin, file=str(info["file"]), sha256=info.get("sha256"),
            label=info.get("label"), first_entry=first_entry, events=events,
            event_numbers=numbers["event_number"], run_numbers=numbers["run_number"],
            offsets=offsets, columns=columns,
            input_counts=counts, input_energies=energies,
        )


def load_event_source(run_dir: str | Path) -> EventSource | None:
    """The ``event_source.json.gz`` of *run_dir*, or ``None`` when it has none.

    Raises ``ValueError`` for an unsupported or malformed file.
    """
    path = Path(run_dir) / EVENT_SOURCE
    if not path.is_file():
        return None
    return read_event_source(path)


def read_event_source(path: Path) -> EventSource:
    """The event source in *path*; ``ValueError`` for an unsupported
    ``schema_version`` or a malformed file."""
    try:
        raw = json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, EOFError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: not a gzipped JSON event source ({exc})") from exc
    validate_event_source_schema(raw, source=path)
    return EventSource.from_json(raw, source=path)


def record_event_source(
    out_path: Path,
    source_file: Path,
    *,
    n_events: int,
    origin: str = "recorded",
    label: str | None = None,
    input_content: bool = False,
    checksum: bool = False,
) -> bool:
    """Write what went into the first *n_events* events of the podio
    *source_file* to *out_path*: their generator particles and, with
    *input_content*, what the file's collections hold in each; with
    *checksum*, the file's sha256. *label* names the configuration whose
    output the file is. Whether it was written: a failure is warned about and
    costs only this file, never the run."""
    try:
        record = describe_source(
            source_file, n_events=n_events, origin=origin, label=label,
            input_content=input_content, checksum=checksum,
        )
        out_path.write_bytes(gzip.compress(
            json.dumps(record, separators=(",", ":")).encode(), mtime=0,
        ))
    except Exception:  # noqa: BLE001 - the event source is an extra, never the run's failure
        out_path.unlink(missing_ok=True)
        print(f"  WARNING: no event source recorded:\n{traceback.format_exc()}")
        return False
    return True


def describe_source(
    source_file: Path,
    *,
    n_events: int,
    origin: str = "recorded",
    label: str | None = None,
    input_content: bool = False,
    checksum: bool = False,
) -> dict:
    """The ``event_source.json.gz`` record of *source_file*'s first
    *n_events* events; see :func:`record_event_source`."""
    if origin not in ORIGINS:
        raise ValueError(f"unknown origin {origin!r}, not one of {ORIGINS}")
    if n_events < 1:
        raise ValueError(f"n_events must be positive, got {n_events}")
    open_root = _event_size._root_opener()
    if open_root is None:
        raise RuntimeError("uproot is not available")
    with open_root(source_file) as root_file:
        collections = podio_collections(root_file)
        if collections is None:
            raise ValueError(f"{source_file.name} is not a podio file")
        tree = root_file["events"]
        available = int(tree.num_entries)
        if available < n_events:
            raise ValueError(
                f"{source_file.name} holds {available} events, the job processed {n_events}"
            )
        generator = _generator_particles(tree, _mc_particles(collections), n_events)
        headers = read_event_headers(tree, collections, n_events)
        content = (
            event_content(root_file, entry_stop=n_events)["collections"] if input_content else None
        )
    source = {"file": source_file.name, "first_entry": 0, "events": n_events}
    if checksum:
        source["sha256"] = _sha256(source_file)
    if label is not None:
        source["label"] = label
    record = {
        "schema_version": EVENT_SOURCE_SCHEMA_VERSION,
        "origin": origin,
        "source": source,
        "generator": generator,
    }
    if headers:
        record["event_headers"] = headers
    if content is not None:
        record["input_content"] = {
            "collections": [
                {"name": name, "type": collections[name][0], **measured}
                for name, measured in content.items()
            ],
        }
    return record


def _mc_particles(collections: Mapping[str, tuple[str, bool]]) -> str:
    """The name of the file's MCParticle collection: ``MCParticles`` when it
    has one, else its first."""
    names = [n for n, (t, subset) in collections.items() if t == _MC_PARTICLE_TYPE and not subset]
    if not names:
        raise ValueError("the file has no MCParticle collection")
    return _MC_PARTICLES if _MC_PARTICLES in names else names[0]


def _generator_particles(tree, name: str, n_events: int) -> dict:
    """The generator particles of the first *n_events* entries of collection
    *name*, as ``offsets`` and one list per :data:`GENERATOR_COLUMNS`."""
    members = {
        column: tree[f"{name}/{name}.{member}"].array(entry_stop=n_events, library="np")
        for column, member in GENERATOR_MEMBERS.items()
    }
    status = members["gen_status"]
    keep = [np.asarray(s) != 0 for s in status]
    out: dict = {"offsets": [0, *np.cumsum([int(k.sum()) for k in keep]).tolist()]}
    for column, per_entry in members.items():
        values = np.concatenate([np.asarray(v)[k] for v, k in zip(per_entry, keep, strict=True)])
        out[column] = (
            [int(v) for v in values] if column in _INTEGER_COLUMNS
            else [round_significant(float(v)) for v in values]
        )
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_ints(values, events: int) -> tuple[int | None, ...] | None:
    if values is None:
        return None
    if len(values) != events:
        raise ValueError(f"{len(values)} event headers for {events} events")
    return tuple(None if v is None else int(v) for v in values)


def _content_frames(raw: Mapping | None, events: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The input's per-event counts and energies, each a frame of one column
    per collection measured, indexed by ``entry``."""
    index = pd.RangeIndex(events, name="entry")
    counts: dict[str, list] = {}
    energies: dict[str, list] = {}
    for collection in (raw or {}).get("collections", ()):
        for key, out in (("entries", counts), ("energy_gev", energies)):
            values = collection.get(key)
            if values is None:
                continue
            if len(values) != events:
                raise ValueError(
                    f"{key} of {collection['name']} has {len(values)} values for {events} events"
                )
            out[collection["name"]] = values
    return (
        pd.DataFrame(counts, index=index, dtype=float) if counts else pd.DataFrame(index=index),
        pd.DataFrame(energies, index=index, dtype=float) if energies else pd.DataFrame(index=index),
    )
