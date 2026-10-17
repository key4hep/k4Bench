"""What each algorithm of a k4run job read and wrote: ``{label}_dataflow.json``.

The k4run benchmark builds the file after each job (:func:`build_dataflow`)
from what the job recorded, and :func:`load_dataflow` reads it back as a
:class:`DataFlow`, whose :meth:`DataFlow.graph` joins the algorithms by the
collections one writes and a later one reads.

Where each algorithm's collections come from
--------------------------------------------
* A Gaudi algorithm, k4FWCore's included: the keys of the data handles it
  declared, which the k4BenchAuditor records (``inputs`` and ``outputs`` in
  ``_components.json``). These are exact. The k4FWCore Reader declares one
  output per collection of its input file.
* A Marlin processor run by a ``MarlinProcessorWrapper``: its LCIO collection
  parameters. Marlin marks a parameter registered as an input or output
  collection with its LCIO type, and ``Marlin -x`` lists those for every
  processor type; the collection names are the parameter's value in the job
  options, or its default where the job leaves it unset. The wrapper's
  EDM4hep-to-LCIO and LCIO-to-EDM4hep converters come from the job options.
* The k4FWCore Writer, which reads whatever its keep/drop rules
  (``IOSvc.outputCommands``) select from the event store.

Limitations
-----------
* A Marlin processor that takes a collection name as a plain string parameter,
  rather than registering it as a collection, declares nothing: what it reads
  and writes is missing, and a collection only it writes has no producer.
* A declared collection need not be one the algorithm uses: a Marlin input
  whose default the job never set is listed all the same.
* The converters are modelled, not observed: one converting everything
  (``convertAll``) is taken to convert every collection the other store holds
  at that point, under its ``collNameMapping`` name, unless the target store
  already holds one of that name.
* Gaudi handles carry a key, not a type; only LCIO collections have a type.
* Algorithms are in the order the auditor first saw them, which is the order
  they were initialized and, in a sequential job, the order they run.
* A run whose auditor predates recording data handles has none: its Gaudi
  algorithms' reads and writes are unknown (``None``).
"""

from __future__ import annotations

import ast
import fnmatch
import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from k4bench.artifacts import DATAFLOW_SUFFIX, MARLIN_PROCESSORS, labelled_files
from k4bench.plugin.schema import DATAFLOW_SCHEMA_VERSION, validate_dataflow_schema

#: The two stores collections live in: Gaudi's event store, where k4FWCore
#: keeps EDM4hep collections, and the LCIO event Marlin processors share.
GAUDI = "gaudi"
LCIO = "lcio"

#: The k4MarlinWrapper converter properties, by the direction they convert.
_CONVERTERS = {"EDM4hep2LcioTool": "edm4hep_to_lcio", "Lcio2EDM4hepTool": "lcio_to_edm4hep"}

#: k4FWCore's Writer keeps everything unless told otherwise.
_DEFAULT_OUTPUT_COMMANDS = ("keep *",)

_OPTION_RE = re.compile(r"^(?P<owner>[\w.:]+)\.(?P<prop>\w+) = (?P<value>.*?);(?:\s*//.*)?$")


@dataclass(frozen=True)
class Collection:
    """One collection an algorithm reads or writes."""

    name: str
    store: str
    #: The LCIO type a Marlin processor declares; ``None`` for a Gaudi handle.
    type: str | None = None
    #: The Marlin parameter naming it, and whether the job set that parameter
    #: (``False``: its default).
    parameter: str | None = None
    explicit: bool | None = None


@dataclass(frozen=True)
class Converter:
    """A k4MarlinWrapper converter run before (EDM4hep to LCIO) or after
    (LCIO to EDM4hep) its processor."""

    direction: str
    convert_all: bool
    #: Source name to target name.
    mapping: Mapping[str, str]


@dataclass(frozen=True)
class Algorithm:
    """One algorithm, with what it read and wrote; ``None`` where unknown."""

    name: str
    type: str
    #: The class doing the work: the type, or the Marlin processor type.
    impl: str
    #: ``"marlin"`` for a wrapped Marlin processor, else ``"gaudi"``.
    framework: str
    inputs: tuple[Collection, ...] | None
    outputs: tuple[Collection, ...] | None
    converters: tuple[Converter, ...] = ()
    #: The Writer's keep/drop rules; ``None`` for any other algorithm.
    output_commands: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Link:
    """One collection passed from a producer to a consumer."""

    #: The name the consumer reads it by, in the consumer's store.
    name: str
    store: str
    type: str | None
    #: The name the producer wrote it under, in its own store: different
    #: when a converter carried it across.
    source: str
    source_store: str


@dataclass(frozen=True)
class Graph:
    """Algorithms joined by the collections one writes and another reads."""

    #: ``(producer, consumer)`` to what passes between them.
    edges: Mapping[tuple[str, str], tuple[Link, ...]]
    #: ``(consumer, collection)`` for every read with no recorded producer.
    unresolved: tuple[tuple[str, Collection], ...]


@dataclass(frozen=True)
class DataFlow:
    """One label's ``_dataflow.json``."""

    algorithms: tuple[Algorithm, ...]
    #: Whether the job's Gaudi algorithms had their data handles recorded.
    handles_recorded: bool
    #: Whether Marlin's collection parameters were available.
    marlin_recorded: bool

    @classmethod
    def from_json(cls, raw: Mapping, *, source: str | Path | None = None) -> DataFlow:
        validate_dataflow_schema(raw, source=source)

        def collections(entries: list | None) -> tuple[Collection, ...] | None:
            return None if entries is None else tuple(Collection(**entry) for entry in entries)

        algorithms = tuple(
            Algorithm(
                name=entry["name"], type=entry["type"], impl=entry["impl"],
                framework=entry["framework"],
                inputs=collections(entry["inputs"]), outputs=collections(entry["outputs"]),
                converters=tuple(Converter(**c) for c in entry.get("converters", [])),
                output_commands=(None if entry.get("output_commands") is None
                                 else tuple(entry["output_commands"])),
            )
            for entry in raw["algorithms"]
        )
        return cls(algorithms, raw["handles_recorded"], raw["marlin_recorded"])

    def graph(self) -> Graph:
        """Replay the job in order: each read is linked to the latest earlier
        write of that collection in that store, a converter carrying
        collections from one store to the other, and the Writer reading what
        its rules keep. A read with no earlier write falls back to any
        algorithm that writes that collection directly."""
        # name -> (producer, name it was written under, store it was written to)
        stores: dict[str, dict[str, tuple[str, str, str]]] = {GAUDI: {}, LCIO: {}}
        writers = {
            (out.store, out.name): algorithm.name
            for algorithm in self.algorithms for out in algorithm.outputs or ()
        }
        edges: dict[tuple[str, str], list[Link]] = {}
        unresolved: list[tuple[str, Collection]] = []
        for algorithm in self.algorithms:
            for converter in algorithm.converters:
                if converter.direction == "edm4hep_to_lcio":
                    _convert(stores[GAUDI], stores[LCIO], converter)
            if algorithm.output_commands is not None:
                for name, origin in stores[GAUDI].items():
                    if _kept(name, algorithm.output_commands):
                        _link(edges, origin, algorithm.name, Collection(name, GAUDI))
            for read in algorithm.inputs or ():
                origin = stores[read.store].get(read.name)
                if origin is None and (writer := writers.get((read.store, read.name))):
                    origin = (writer, read.name, read.store)
                if origin is None or origin[0] == algorithm.name:
                    unresolved.append((algorithm.name, read))
                else:
                    _link(edges, origin, algorithm.name, read)
            for out in algorithm.outputs or ():
                stores[out.store][out.name] = (algorithm.name, out.name, out.store)
            for converter in algorithm.converters:
                if converter.direction == "lcio_to_edm4hep":
                    _convert(stores[LCIO], stores[GAUDI], converter)
        return Graph({pair: tuple(links) for pair, links in edges.items()}, tuple(unresolved))


def _convert(source: dict, target: dict, converter: Converter) -> None:
    names = source if converter.convert_all else converter.mapping
    for name in list(names):
        if name in source:
            target.setdefault(converter.mapping.get(name, name), source[name])


def _kept(name: str, commands: Iterable[str]) -> bool:
    """Whether k4FWCore keep/drop *commands* keep collection *name*: the last
    rule whose pattern matches decides, and nothing is kept unmatched."""
    kept = False
    for command in commands:
        action, _, pattern = command.strip().partition(" ")
        if fnmatch.fnmatchcase(name, pattern.strip()):
            kept = action == "keep"
    return kept


def _link(edges: dict, origin: tuple[str, str, str], consumer: str, read: Collection) -> None:
    producer, source, source_store = origin
    edges.setdefault((producer, consumer), []).append(
        Link(read.name, read.store, read.type, source, source_store)
    )


# -- building ------------------------------------------------------------------


def parse_joboptions(text: str) -> dict[str, dict[str, object]]:
    """``{owner: {property: value}}`` of a JobOptionsSvc dump. Values that are
    Python literals are parsed; others, such as tool references, kept as text."""
    options: dict[str, dict[str, object]] = {}
    for line in text.splitlines():
        if not (match := _OPTION_RE.match(line.strip())):
            continue
        raw = match["value"].strip()
        try:
            value = ast.literal_eval(raw.replace("set()", "[]"))
        except (ValueError, SyntaxError):
            value = raw
        options.setdefault(match["owner"], {})[match["prop"]] = value
    return options


def marlin_collection_parameters(xml: str) -> dict[str, dict[str, dict]]:
    """``{processor type: {parameter: {direction, type, default}}}`` for the
    LCIO collection parameters ``Marlin -x`` lists in *xml*."""
    root = ET.fromstring(xml[xml.index("<marlin"):])
    out: dict[str, dict[str, dict]] = {}
    for processor in root.iter("processor"):
        parameters = {}
        for parameter in processor.iter("parameter"):
            if lcio := parameter.get("lcioInType"):
                direction = "input"
            elif lcio := parameter.get("lcioOutType"):
                direction = "output"
            else:
                continue
            parameters[parameter.get("name")] = {
                "direction": direction, "type": lcio,
                "default": (parameter.text or "").split(),
            }
        out.setdefault(processor.get("type"), parameters)
    return out


def _gaudi_key(key: str) -> str:
    """A handle's key as the collection's name: Gaudi may root it at ``/Event``."""
    return key.removeprefix("/Event/").lstrip("/")


def _marlin_algorithm(name: str, processor_type: str, options: dict,
                      marlin: Mapping[str, Mapping[str, dict]] | None) -> dict:
    entry: dict = {"framework": "marlin"}
    if marlin is None or processor_type not in marlin:
        entry["inputs"] = entry["outputs"] = None
    else:
        values = options.get(name, {}).get("Parameters", {})
        values = values if isinstance(values, dict) else {}
        entry["inputs"], entry["outputs"] = [], []
        for parameter, spec in sorted(marlin[processor_type].items()):
            explicit = parameter in values
            for collection in (values[parameter] if explicit else spec["default"]):
                if collection:
                    entry[f"{spec['direction']}s"].append({
                        "name": collection, "store": LCIO, "type": spec["type"],
                        "parameter": parameter, "explicit": explicit,
                    })
    converters = []
    for prop, direction in _CONVERTERS.items():
        tool = options.get(name, {}).get(prop)
        if isinstance(tool, str) and tool:
            settings = options.get(f"{name}.{tool.split('/')[-1]}", {})
            mapping = settings.get("collNameMapping") or {}
            converters.append({
                "direction": direction,
                "convert_all": bool(settings.get("convertAll", False)),
                "mapping": dict(mapping) if isinstance(mapping, dict) else {},
            })
    entry["converters"] = converters
    return entry


def build_dataflow(
    components: Mapping,
    joboptions: str,
    marlin_xml: str | None,
    *,
    reader_collections: Iterable[str] | None = None,
) -> dict:
    """The ``_dataflow.json`` document of one job.

    *components* is the job's parsed ``_components.json``, *joboptions* its
    resolved job options and *marlin_xml* ``Marlin -x`` of its stack (``None``
    when the job ran no Marlin processor, and an unreadable one counts as
    none). *reader_collections* stands in for
    the Reader's outputs where the auditor recorded no data handles.
    """
    options = parse_joboptions(joboptions)
    try:
        marlin = marlin_collection_parameters(marlin_xml) if marlin_xml else None
    except (ValueError, ET.ParseError):
        # An unreadable dump costs the Marlin side only.
        marlin = None
    entries = components["components"]
    parents = {entry.get("parent") for entry in entries if entry.get("parent") is not None}
    algorithms = [
        entry for index, entry in enumerate(entries)
        if entry.get("category") == "algorithm" and index not in parents
    ]
    handles = any("inputs" in entry and entry["inputs"] is not None for entry in algorithms)
    out = []
    for entry in algorithms:
        record = {"name": entry["name"], "type": entry["type"], "impl": entry.get("impl") or entry["type"]}
        if entry["type"] == "MarlinProcessorWrapper":
            record.update(_marlin_algorithm(entry["name"], record["impl"], options, marlin))
        else:
            record["framework"] = "gaudi"
            for direction in ("inputs", "outputs"):
                keys = entry.get(direction)
                record[direction] = None if keys is None else [
                    {"name": _gaudi_key(key), "store": GAUDI} for key in keys
                ]
            if entry["type"] == "Reader" and record["outputs"] is None and reader_collections is not None:
                record["inputs"] = []
                record["outputs"] = [{"name": n, "store": GAUDI} for n in reader_collections]
            if entry["type"] == "Writer":
                commands = options.get("IOSvc", {}).get("outputCommands")
                record["output_commands"] = list(commands) if commands else list(_DEFAULT_OUTPUT_COMMANDS)
        out.append(record)
    return {
        "schema_version": DATAFLOW_SCHEMA_VERSION,
        "handles_recorded": handles,
        "marlin_recorded": marlin is not None,
        "algorithms": out,
    }


def load_dataflow(run_dir: str | Path) -> dict[str, DataFlow]:
    """Every label's data flow in *run_dir*; empty when it has none."""
    flows = {}
    for path, label in labelled_files(Path(run_dir), DATAFLOW_SUFFIX):
        flows[label] = DataFlow.from_json(json.loads(path.read_text()), source=path)
    return flows


def read_marlin_processors(run_dir: str | Path) -> str | None:
    """The run's ``Marlin -x`` dump, or ``None`` when it has none."""
    path = Path(run_dir) / MARLIN_PROCESSORS
    return gzip.decompress(path.read_bytes()).decode() if path.exists() else None
