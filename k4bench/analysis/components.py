"""Per-component costs of one k4run job, as the k4BenchAuditor recorded them.

A Gaudi job is a set of components — algorithms and services — and the auditor
times every audited call of each: algorithm executions per event, and the
initialize/start/stop/finalize of algorithms and services. Every recorded cost
but those in :data:`INCLUSIVE_METRICS` is a *self* cost, what the call spent
outside the audited calls nested inside it, so costs add up: a sequencer's own
cost is its bookkeeping, and the costs of an event's components sum to the cost
of its audited calls. The event time in
``_events.json`` spans the first to the last top-level algorithm, so with
several top-level algorithms it exceeds the components' summed ``wall_s`` by
the time spent between them, which no component is charged for.

Metrics recorded for every call:

- ``wall_s`` — wall-clock time;
- ``cpu_s`` — CPU time of the calling thread;
- ``peak_rss_increase_mb`` — how much the call raised the process's peak RSS.
  Zero for most calls once the job has warmed up; nonzero names the component
  that pushed the high-water mark, which is what a peak-memory regression needs.

Recorded for every call too, but absent from files k4bench 0.0.47's auditor
wrote (:data:`THREAD_METRICS`):

- ``minor_page_faults`` — pages mapped in from memory: first touches of fresh
  memory, or of files already cached;
- ``major_page_faults`` — pages the thread waited to read from disk or CVMFS;
- ``voluntary_context_switches`` — times the thread blocked, on I/O or a lock;
  together with the major faults, why ``wall_s`` exceeds ``cpu_s``;
- ``involuntary_context_switches`` — times it was preempted: a sign the host
  was busy and the call's times are noisier.

Where the hardware performance counters could be read (:data:`HARDWARE_METRICS`):

- ``instructions`` — user-space instructions executed. They depend on the work
  done, not on the host's load, so they show changes too small for time to;
- ``cycles`` — user-space CPU cycles; instructions per cycle tell more work
  apart from the same work run slower.

When the job preloaded the auditor's allocation counter, as ``k4bench k4run``
does (:data:`ALLOCATION_METRICS`):

- ``allocations`` — heap blocks allocated (malloc and its relatives, and C++
  new); a resized block counts as one;
- ``allocated_bytes`` — the heap memory they take, malloc's 8-byte header and
  rounding to 16 bytes included (a mapped page multiple for large blocks); the
  byte metrics below count the same way;
- ``net_allocated_bytes`` — allocated minus freed, what the call kept on the
  heap. Negative for a call that freed more than it allocated;
- ``peak_heap_bytes`` — the most heap the call held at once beyond what its
  thread held when it started: the memory the component needs;
- ``largest_allocation_bytes`` — its largest single block.

The last two are not self costs (:data:`INCLUSIVE_METRICS`): they include
nested calls, whose memory was held during the call too, so they do not add up,
and a component called several times in one event or phase is charged its
largest. Memory a library maps from the kernel itself, as pool allocators do,
is not counted.

Nesting is tracked per thread, so the above holds for a serial job. When calls
ran concurrently (``threads`` > 1), ``cpu_s`` still sums to the event's work,
but the ``wall_s`` of calls that overlapped on different threads sums to more
than the event's elapsed time (the event time is in ``_events.json``), and since
the peak RSS is the process's, a call's ``peak_rss_increase_mb`` may include
memory allocated by other threads meanwhile. Everything else is counted per
thread and stays exact.

A missing value is ``NaN``, never zero: an algorithm that did not run in an
event (a filtered sequence) has no cost there, which is different from a cost
too small to resolve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Metrics the auditor records for every call.
COMPONENT_METRICS = ("wall_s", "cpu_s", "peak_rss_increase_mb")

#: Metrics recorded for every call, absent from files k4bench 0.0.47's auditor wrote.
THREAD_METRICS = (
    "minor_page_faults",
    "major_page_faults",
    "voluntary_context_switches",
    "involuntary_context_switches",
)

#: Metrics recorded where the hardware performance counters could be read.
HARDWARE_METRICS = ("instructions", "cycles")

#: Metrics recorded when the job preloaded the allocation counter.
ALLOCATION_METRICS = (
    "allocations",
    "allocated_bytes",
    "net_allocated_bytes",
    "peak_heap_bytes",
    "largest_allocation_bytes",
)

#: Metrics that already include nested calls, and do not add up.
INCLUSIVE_METRICS = ("peak_heap_bytes", "largest_allocation_bytes")


@dataclass(frozen=True)
class ComponentTiming:
    """One label's ``_components.json``, as frames.

    ``components`` is indexed by component name, in the order the auditor first
    saw each component, with columns ``category`` (``algorithm``, ``service`` or
    ``other``), ``type`` (the Gaudi type), ``impl`` (the class doing the work: the
    type, or the class a wrapper delegates to), ``library`` (the shared object
    defining the type, symlinks resolved), ``parent`` (the enclosing sequencer's
    name, or ``None``, also for an algorithm several sequencers share, which no
    sequencer's inclusive cost then claims) and ``execute_calls``.

    ``execute`` maps each metric to an events × components frame indexed by
    event number. ``lifecycle`` maps each non-event phase (``initialize``,
    ``start``, ``stop``, ``finalize``, or a custom audited section) to a
    components × metrics frame. A non-event call made during an execution, such
    as a service initialized on first use, appears there and also stays in the
    executing component's ``execute`` cost. ``phases`` holds each phase's
    ``(begin, end)`` in ns since the Unix epoch, ``execute`` included, spanning
    only the calls made outside event processing.
    """

    components: pd.DataFrame
    execute: dict[str, pd.DataFrame]
    lifecycle: dict[str, pd.DataFrame]
    phases: dict[str, tuple[int, int]]
    process_start_epoch_ns: int | None
    measurement_overhead_ns: float | None
    threads: int

    @classmethod
    def from_json(cls, raw: Mapping, *, source: str | Path | None = None) -> ComponentTiming:
        """Build from a parsed ``_components.json``, refusing an inconsistent one.

        The schema version is checked by the caller
        (:func:`~k4bench.plugin.schema.validate_component_schema`).
        """
        where = f"{source}: " if source is not None else ""
        missing = [
            key
            for key in ("components", "event_numbers", "execute", "lifecycle", "phases")
            if key not in raw
        ]
        if missing:
            raise ValueError(f"{where}missing keys: {missing}")

        entries = raw["components"]
        names = [entry["name"] for entry in entries]
        if len(set(names)) != len(names):
            raise ValueError(f"{where}component names are not unique")
        n = len(names)
        parents = []
        for entry in entries:
            parent = entry.get("parent")
            if parent is not None and not 0 <= parent < n:
                raise ValueError(
                    f"{where}component {entry['name']!r} has parent index {parent} out of range"
                )
            parents.append(None if parent is None else names[parent])
        # A parent cycle would make descendants() recurse without end.
        parent_of = dict(zip(names, parents))
        for name in names:
            seen = {name}
            ancestor = parent_of[name]
            while ancestor is not None:
                if ancestor in seen:
                    raise ValueError(f"{where}component {name!r} is its own ancestor")
                seen.add(ancestor)
                ancestor = parent_of[ancestor]
        components = pd.DataFrame(
            {
                "category": [entry.get("category", "other") for entry in entries],
                "type": [entry.get("type", "") for entry in entries],
                "impl": [entry.get("impl", "") for entry in entries],
                "library": [entry.get("library", "") for entry in entries],
                "execute_calls": [entry.get("execute_calls", 0) for entry in entries],
            },
            index=pd.Index(names, name="component"),
        )
        # Object dtype keeps None as None; an inferred string dtype would make it NaN.
        components.insert(4, "parent", pd.Series(parents, index=components.index, dtype=object))

        events = pd.Index(raw["event_numbers"], name="event_number")
        if events.has_duplicates:
            raise ValueError(f"{where}event_numbers contains duplicates")
        absent = [metric for metric in COMPONENT_METRICS if metric not in raw["execute"]]
        if absent:
            raise ValueError(f"{where}execute is missing metrics: {absent}")
        execute = {}
        for metric, rows in raw["execute"].items():
            if len(rows) != len(events) or any(len(row) != n for row in rows):
                raise ValueError(
                    f"{where}execute.{metric} is not {len(events)} events × {n} components"
                )
            execute[metric] = pd.DataFrame(
                rows, index=events, columns=components.index, dtype=float
            )

        lifecycle = {}
        for phase, table in raw["lifecycle"].items():
            absent = [metric for metric in COMPONENT_METRICS if metric not in table]
            if absent:
                raise ValueError(f"{where}lifecycle.{phase} is missing metrics: {absent}")
            if any(len(column) != n for column in table.values()):
                raise ValueError(f"{where}lifecycle.{phase} does not have one value per component")
            lifecycle[phase] = pd.DataFrame(table, index=components.index, dtype=float)

        phases = {
            phase: (int(span["begin_epoch_ns"]), int(span["end_epoch_ns"]))
            for phase, span in raw["phases"].items()
        }
        start = raw.get("process_start_epoch_ns")
        overhead = raw.get("measurement_overhead_ns")
        return cls(
            components=components,
            execute=execute,
            lifecycle=lifecycle,
            phases=phases,
            process_start_epoch_ns=start if start is not None and start >= 0 else None,
            measurement_overhead_ns=overhead if overhead is not None and overhead >= 0 else None,
            threads=int(raw.get("threads", 1)),
        )

    def children(self, component: str) -> list[str]:
        """Components whose enclosing sequencer is *component*."""
        return self.components.index[self.components["parent"] == component].tolist()

    def descendants(self, component: str) -> list[str]:
        """Every component nested under *component*, at any depth."""
        out = []
        for child in self.children(component):
            out += [child, *self.descendants(child)]
        return out

    def inclusive(self, metric: str = "wall_s") -> pd.DataFrame:
        """Per-event *metric* of each component including its descendants — a
        sequencer's cost with everything it ran. A component that did not run
        in an event stays ``NaN`` there; its descendants' missing values count
        as zero. For ``wall_s`` in a job with ``threads`` > 1, descendants that
        ran concurrently make this more than the elapsed time. Refused for a
        metric that already includes nested calls (:data:`INCLUSIVE_METRICS`)."""
        if metric in INCLUSIVE_METRICS:
            raise ValueError(f"{metric} already includes nested calls; read it from execute")
        own = self.execute[metric]
        filled = own.fillna(0.0)
        out = {}
        for component in own.columns:
            total = filled[[component, *self.descendants(component)]].sum(axis=1)
            out[component] = total.where(own[component].notna())
        return pd.DataFrame(out, index=own.index)

    def phase_seconds(self) -> dict[str, float]:
        """Duration of each recorded phase, plus ``configure``: from process
        start to the first audited initialize, which is the Python options
        processing and everything before the auditor existed."""
        out = {phase: (end - begin) / 1e9 for phase, (begin, end) in self.phases.items()}
        if self.process_start_epoch_ns is not None and "initialize" in self.phases:
            out["configure"] = (self.phases["initialize"][0] - self.process_start_epoch_ns) / 1e9
        return out
