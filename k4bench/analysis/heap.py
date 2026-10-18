"""The heap each component of a k4run job held: ``{label}_heap.json``.

Written by the k4BenchAuditor in the k4run benchmark's memory pass, a run of
the job of its own after the timed one, whose allocation counter records which
component allocated every heap block. A block is charged to the component
whose audited call allocated it, at any depth (a service initialized from
within an algorithm's call is a call of its own), and stays charged to it until
freed, wherever and whenever that happens. What no component allocated — the
Python configuration, ROOT's start-up, the framework between calls — belongs to
:data:`NO_COMPONENT`.

For every component, in bytes of heap (malloc's 8-byte header and rounding to
16 bytes included, as ``_components.json``'s allocation metrics count them):

- held before the first event, which is what initialize and start left, and
  the most it held at once until then;
- held at the start and the end of each event, and the most it held at once
  during the event (from its first top-level call to its last). The most it
  held beyond what it held at the start is its working set, the memory the
  event needed of it (:meth:`ComponentHeap.working_set`);
- held at the auditor's finalize: what survived the job's own finalize.

Beside them, at the same moments, the process's heap as malloc counts it
(``mallinfo2``: the chunks in use, those it mapped on their own included, and
the free ones it caches per thread) and its anonymous RSS. The components and
:data:`NO_COMPONENT` add up to nearly malloc's count (to 0.06% in CLD
reconstruction); the rest is the blocks allocated while the process was still
loading, before the counter started, and the auditor's own memory. The
anonymous RSS counts something else: it also holds what libraries map
themselves, the free heap malloc has not returned and the counter's own table
(``table_mapped_bytes``), but not heap that was allocated and never touched,
which takes no pages. In CLD reconstruction that is about a sixth of the heap,
so held bytes are memory allocated, not memory resident.

What the numbers do not say:

- who *uses* memory: a block is charged to the component that created it, so
  the event data the input reader creates stays the reader's until the output
  writer frees it, and the memory a library takes as it loads (its static
  constructors, ROOT dictionaries) is that of the component whose call loaded
  it;
- what lies outside malloc's heap: memory a library maps itself (pool
  allocators, CPython's arenas, large buffers) and file-backed memory (code,
  CVMFS), which ``_events.json``'s file-backed RSS shows;
- the timed run's values: the pass is another, shorter run of the same job.
  It reads the same input, so its events are the timed run's first ones, but
  recording owners slows every allocation, so its numbers are comparable in
  kind, not identical in value;
- what a service first used during an event keeps is held from that event on,
  not before the first: the service is initialized then.

In a job running calls concurrently (``threads`` > 1 in ``_components.json``),
attribution still follows each thread's running component, but a snapshot
taken while other threads run catches them mid-call. A job that had several
events in flight at once has no per-event peaks: the owners' peaks are the
process's.

A missing value is ``NaN``: before the first event of a job that ran none, the
per-event peaks of a job that ran events at once, and an RSS that could not be
read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: The column of what no component allocated.
NO_COMPONENT = "(no component)"

#: Per-component metrics before the first event, and at each event's end.
HEAP_METRICS = ("held_bytes", "peak_held_bytes")

#: Per-component metrics of each event: :data:`HEAP_METRICS`, and what each
#: component held as the event started.
EVENT_HEAP_METRICS = ("start_held_bytes", *HEAP_METRICS)

#: The process's heap at each moment.
PROCESS_METRICS = ("malloc_in_use_bytes", "rss_anon_mb")


@dataclass(frozen=True)
class ComponentHeap:
    """One label's ``_heap.json``, as frames.

    Components are :data:`NO_COMPONENT` first, then the auditor's components
    in its order, by name, as in the timed run's
    :class:`~k4bench.analysis.components.ComponentTiming`, which they join on.

    ``execute`` maps each of :data:`EVENT_HEAP_METRICS` to an events ×
    components frame indexed by event number: what each component held as the
    event started and at its end, and the most it held during it.
    ``before_first_event`` is components ×
    :data:`HEAP_METRICS`, ``at_end`` components × ``held_bytes``. ``process``
    is events × :data:`PROCESS_METRICS`, and ``process_before_first_event``
    and ``process_at_end`` the same at those two moments. ``table_peak_entries``
    is the most blocks the counter's table held at once, ``table_mapped_bytes``
    the memory it mapped for them.
    """

    execute: dict[str, pd.DataFrame]
    before_first_event: pd.DataFrame
    at_end: pd.DataFrame
    process: pd.DataFrame
    process_before_first_event: pd.Series
    process_at_end: pd.Series
    table_peak_entries: int
    table_mapped_bytes: int

    @classmethod
    def from_json(cls, raw: Mapping, *, source: str | Path | None = None) -> ComponentHeap:
        """Build from a parsed ``_heap.json``, refusing an inconsistent one.

        The schema version is checked by the caller
        (:func:`~k4bench.plugin.schema.validate_heap_schema`).
        """
        where = f"{source}: " if source is not None else ""
        missing = [
            key
            for key in ("components", "before_first_event", "event_numbers", "execute", "at_end", "table")
            if key not in raw
        ]
        if missing:
            raise ValueError(f"{where}missing keys: {missing}")

        names = list(raw["components"])
        if not names or names[0] != NO_COMPONENT:
            raise ValueError(f"{where}components must start with {NO_COMPONENT!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"{where}component names are not unique")
        components = pd.Index(names, name="component")
        n = len(names)

        events = pd.Index(raw["event_numbers"], name="event_number")
        if events.has_duplicates:
            raise ValueError(f"{where}event_numbers contains duplicates")
        execute = raw["execute"]
        if absent := [key for key in (*EVENT_HEAP_METRICS, *PROCESS_METRICS) if key not in execute]:
            raise ValueError(f"{where}execute is missing {absent}")
        frames = {}
        for metric in EVENT_HEAP_METRICS:
            rows = execute[metric]
            if len(rows) != len(events) or any(len(row) != n for row in rows):
                raise ValueError(f"{where}execute.{metric} is not {len(events)} events × {n} components")
            frames[metric] = pd.DataFrame(rows, index=events, columns=components, dtype=float)
        for metric in PROCESS_METRICS:
            if len(execute[metric]) != len(events):
                raise ValueError(f"{where}execute.{metric} does not have one value per event")
        process = pd.DataFrame(
            {metric: execute[metric] for metric in PROCESS_METRICS}, index=events, dtype=float
        )
        # The auditor writes an RSS it could not read as negative.
        process = process.mask(process < 0)

        def moment(key: str, metrics: tuple[str, ...]) -> tuple[pd.DataFrame, pd.Series]:
            """The components × *metrics* and process values at moment *key*,
            all missing when the file has none."""
            snapshot = raw[key]
            if snapshot is None:
                return (
                    pd.DataFrame(float("nan"), index=components, columns=list(metrics)),
                    pd.Series(float("nan"), index=list(PROCESS_METRICS)),
                )
            if absent := [name for name in (*metrics, *PROCESS_METRICS) if name not in snapshot]:
                raise ValueError(f"{where}{key} is missing {absent}")
            if any(len(snapshot[metric]) != n for metric in metrics):
                raise ValueError(f"{where}{key} does not have one value per component")
            values = pd.Series({metric: snapshot[metric] for metric in PROCESS_METRICS}, dtype=float)
            return (
                pd.DataFrame({metric: snapshot[metric] for metric in metrics}, index=components, dtype=float),
                values.mask(values < 0),
            )

        before_first_event, process_before = moment("before_first_event", HEAP_METRICS)
        at_end, process_at_end = moment("at_end", ("held_bytes",))

        table = raw["table"]
        return cls(
            execute=frames,
            before_first_event=before_first_event,
            at_end=at_end,
            process=process,
            process_before_first_event=process_before,
            process_at_end=process_at_end,
            table_peak_entries=int(table["peak_entries"]),
            table_mapped_bytes=int(table["mapped_bytes"]),
        )

    def working_set(self) -> pd.DataFrame:
        """Events × components: the most each component held during each event
        beyond what it held as the event started, the heap the event needed of
        it. Its persistent state is left out, and so is the data others created
        that it read."""
        return self.execute["peak_held_bytes"] - self.execute["start_held_bytes"]
