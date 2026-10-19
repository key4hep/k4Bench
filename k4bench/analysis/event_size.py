"""What each output file a job wrote holds on disk: ``{label}_event_size.json``.

The runners measure every file a successful run wrote, right after it, while
the files still exist (:func:`record_event_size`): ddsim's output file, and
every file a k4run job created in its working directory. Each file has its
size on disk; a podio file, an EDM4hep ROOT file whose ``events`` tree podio
wrote, also has its events, its ROOT compression setting and, for every
collection of its events, the bytes its branches take compressed, as they lie
in the file, and uncompressed. :func:`load_event_size` reads the files back
as :class:`OutputFile` records, whose :meth:`OutputFile.per_event` gives each
collection's bytes per event.

How a collection's bytes are counted
------------------------------------
podio writes each collection of the ``events`` tree as a branch of its name
(``<name>_objIdx`` for a subset collection) and one ``_<name>_<member>``
branch for each of its relations and vector members; split, each of those has
a sub-branch per data member. A collection's bytes are those of all its
branches and their sub-branches together, TKey headers included, as uproot
counts them per branch (``compressed_bytes``, ``uncompressed_bytes``). The
collections are those podio's own metadata lists for the ``events`` category,
with their types. The branches of no collection, podio's frame parameters,
are :data:`FRAME_PARAMETERS`. The collections and the frame parameters
together are the events tree; what the file holds besides, the metadata and
run trees and ROOT's own records, is in the file's size only.

What the numbers do not say:

- what a production file of the same events would take: the benchmark writes
  tens to a few thousand events with the job's own compression setting, and a
  file with more events fills ROOT's baskets further, which can compress a
  little differently;
- the size of a file in any other format: an LCIO (``.slcio``) or RNTuple
  file has its size on disk only.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from k4bench.artifacts import EVENT_SIZE_SUFFIX, labelled_files
from k4bench.plugin.schema import EVENT_SIZE_SCHEMA_VERSION, validate_event_size_schema

#: The events tree's branches of no collection: podio's frame parameters.
FRAME_PARAMETERS = "(frame parameters)"

#: Bytes in a kB, as the event files count memory (kB / 1024 = MB).
KB = 1024

#: The suffix of a subset collection's branch.
_SUBSET_SUFFIX = "_objIdx"

#: podio's metadata branch listing the ``events`` category's collections.
_TYPE_INFO = "events___CollectionTypeInfo/events___CollectionTypeInfo."

#: ROOT's compression algorithms, by the hundreds of a compression setting.
_ALGORITHMS = {0: "ROOT's default", 1: "ZLIB", 2: "LZMA", 3: "old ZLIB", 4: "LZ4", 5: "ZSTD"}


@dataclass(frozen=True)
class CollectionSize:
    """One collection's branches in a file's events tree."""

    name: str
    #: The collection type podio recorded, e.g. ``edm4hep::MCParticleCollection``.
    type: str
    subset: bool
    compressed_bytes: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class OutputFile:
    """One file a job wrote."""

    #: Its path relative to the directory the job wrote it in.
    name: str
    #: Its size on disk.
    bytes: int
    #: The entries of its podio ``events`` tree; ``None`` for a file that is
    #: not a podio file.
    events: int | None = None
    #: Its ROOT compression setting (:func:`describe_compression`).
    compression: int | None = None
    #: Its events tree's collections, :data:`FRAME_PARAMETERS` among them.
    collections: tuple[CollectionSize, ...] = ()

    @property
    def has_events(self) -> bool:
        """Whether the file is a podio file holding events."""
        return bool(self.events) and bool(self.collections)

    @property
    def event_bytes(self) -> int:
        """The compressed bytes of its events tree."""
        return sum(c.compressed_bytes for c in self.collections)

    def per_event(self) -> pd.DataFrame:
        """Each collection's bytes per event, largest first: ``type``,
        ``subset``, ``compressed_kb`` and ``uncompressed_kb`` per event, and
        its ``share`` of the event's compressed bytes, indexed by
        ``collection``. Empty for a file without events."""
        if not self.has_events:
            return pd.DataFrame(
                columns=["type", "subset", "compressed_kb", "uncompressed_kb", "share"],
                index=pd.Index([], name="collection"),
            )
        df = pd.DataFrame(
            [(c.name, c.type, c.subset, c.compressed_bytes, c.uncompressed_bytes)
             for c in self.collections],
            columns=["collection", "type", "subset", "compressed_kb", "uncompressed_kb"],
        ).set_index("collection")
        df[["compressed_kb", "uncompressed_kb"]] /= KB * self.events
        total = df["compressed_kb"].sum()
        df["share"] = df["compressed_kb"] / total if total > 0 else 0.0
        return df.sort_values("compressed_kb", ascending=False, kind="stable")

    @classmethod
    def from_json(cls, raw: Mapping) -> OutputFile:
        parameters = raw.get("parameters")
        events, compression = raw.get("events"), raw.get("compression")
        collections = [
            CollectionSize(
                c["name"], c["type"], bool(c["subset"]),
                int(c["compressed_bytes"]), int(c["uncompressed_bytes"]),
            )
            for c in raw.get("collections") or ()
        ]
        if parameters and collections:
            collections.append(CollectionSize(
                FRAME_PARAMETERS, "", False,
                int(parameters["compressed_bytes"]), int(parameters["uncompressed_bytes"]),
            ))
        return cls(
            name=raw["name"], bytes=int(raw["bytes"]),
            events=None if events is None else int(events),
            compression=None if compression is None else int(compression),
            collections=tuple(collections),
        )


def describe_compression(setting: int | None) -> str:
    """A ROOT compression setting in words: 101 is "ZLIB level 1"."""
    if setting is None:
        return "unknown"
    algorithm, level = divmod(setting, 100)
    if level == 0:
        return "uncompressed"
    return f"{_ALGORITHMS.get(algorithm, f'algorithm {algorithm}')} level {level}"


def load_event_size(
    run_dir: str | Path,
) -> tuple[dict[str, tuple[OutputFile, ...]], dict[str, str]]:
    """Every label's output files in *run_dir* that could be read, and why
    each other could not: each record is read on its own, so one that is
    malformed hides no other configuration. Both empty when it has none."""
    files: dict[str, tuple[OutputFile, ...]] = {}
    unreadable: dict[str, str] = {}
    for path, label in labelled_files(Path(run_dir), EVENT_SIZE_SUFFIX):
        try:
            files[label] = read_event_size(path)
        except (OSError, ValueError) as exc:
            unreadable[label] = str(exc)
    return files, unreadable


def read_event_size(path: Path) -> tuple[OutputFile, ...]:
    """The output files one ``_event_size.json`` records.

    Raises ``ValueError`` for an unsupported ``schema_version``
    (:func:`~k4bench.plugin.schema.validate_event_size_schema`) or a file
    that is not an event-size record.
    """
    raw = json.loads(path.read_text())
    validate_event_size_schema(raw, source=path)
    try:
        return tuple(OutputFile.from_json(f) for f in raw["files"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed event size record ({exc!r})") from exc


def record_event_size(json_path: Path, files: Mapping[str, Path]) -> None:
    """Write what each of *files* (name to path) holds to *json_path*, warning
    about a file that could not be read. A failure costs only this record,
    never the run."""
    try:
        record = measure_outputs(files)
        json_path.write_text(json.dumps(record, separators=(",", ":")))
    except Exception:  # noqa: BLE001 - the event size is an extra, never the run's failure
        print(f"  WARNING: no event size recorded:\n{traceback.format_exc()}")


def measure_outputs(files: Mapping[str, Path]) -> dict:
    """The ``_event_size.json`` record of *files*, name to path: each one's
    size on disk and, for a podio file, what its events hold."""
    open_root = _root_opener() if any(p.suffix == ".root" for p in files.values()) else None
    records = []
    for name, path in files.items():
        record: dict = {"name": name, "bytes": path.stat().st_size}
        if open_root is not None and path.suffix == ".root":
            try:
                with open_root(path) as root_file:
                    record.update(_podio_events(root_file))
            except Exception as exc:  # noqa: BLE001 - one unreadable file hides no other
                print(f"  WARNING: could not read the events of {name}: {exc}")
        records.append(record)
    return {"schema_version": EVENT_SIZE_SCHEMA_VERSION, "files": records}


def collection_sizes(
    branches: Iterable, collections: Mapping[str, tuple[str, bool]],
) -> tuple[list[dict], dict]:
    """The bytes of each of *collections* (name to type and whether it is a
    subset) among an events tree's top-level *branches* (uproot's), and of the
    branches of none, as ``(collections, parameters)``. Collections are in
    *collections*' order."""
    names = set(collections)
    sizes = {name: [0, 0] for name in collections}
    rest = [0, 0]
    for branch in branches:
        owner = _owner(branch.name, names)
        total = sizes[owner] if owner is not None else rest
        compressed, uncompressed = _branch_bytes(branch)
        total[0] += compressed
        total[1] += uncompressed
    return (
        [
            {"name": name, "type": type_, "subset": subset,
             "compressed_bytes": sizes[name][0], "uncompressed_bytes": sizes[name][1]}
            for name, (type_, subset) in collections.items()
        ],
        {"compressed_bytes": rest[0], "uncompressed_bytes": rest[1]},
    )


def _root_opener():
    """uproot's ``open``, or ``None``, warned about, when uproot is missing."""
    try:
        import uproot
    except ImportError:
        print("  WARNING: uproot is not available, so no output file's events are measured")
        return None
    return uproot.open


def _podio_events(root_file) -> dict:
    """What the podio events tree of an open uproot file holds; empty when it
    has none."""
    kinds = root_file.classnames(recursive=False, cycle=False)
    if kinds.get("events") != "TTree" or kinds.get("podio_metadata") != "TTree":
        return {}
    metadata = root_file["podio_metadata"]
    names, types, subsets = (
        metadata[_TYPE_INFO + leaf].array(library="np")[0]
        for leaf in ("name", "dataType", "isSubset")
    )
    tree = root_file["events"]
    collections, parameters = collection_sizes(
        tree.branches,
        {str(n): (str(t), bool(s)) for n, t, s in zip(names, types, subsets, strict=True)},
    )
    return {
        "events": int(tree.num_entries),
        "compression": int(root_file.file.fCompress),
        "collections": collections,
        "parameters": parameters,
    }


def _owner(branch: str, names: set[str]) -> str | None:
    """The collection of *names* a top-level events branch belongs to, or
    ``None`` for a branch of none."""
    if branch in names:
        return branch
    if branch.endswith(_SUBSET_SUFFIX) and (name := branch.removesuffix(_SUBSET_SUFFIX)) in names:
        return name
    if branch.startswith("_"):
        # A relation's or vector member's ``_<name>_<member>``: one collection's
        # name can begin another's, so the longest that fits names it.
        return max((n for n in names if branch.startswith(f"_{n}_")), key=len, default=None)
    return None


def _branch_bytes(branch) -> tuple[int, int]:
    """The compressed and uncompressed bytes of *branch* and all its
    sub-branches: uproot counts each branch's own baskets only."""
    compressed, uncompressed = int(branch.compressed_bytes), int(branch.uncompressed_bytes)
    for sub in branch.branches:
        c, u = _branch_bytes(sub)
        compressed += c
        uncompressed += u
    return compressed, uncompressed
