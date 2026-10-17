"""A k4run job's resolved Gaudi options, and what changed between two jobs.

The k4Bench auditor's options fragment sets ``JobOptionsSvc.DUMPFILE``, so
every k4run job writes ``{label}_joboptions.opts``: each property of every
configured component after the options files ran, defaults included, one per
line as Python literals::

    MyConformalTracking.OutputLevel = 4;
    MyConformalTracking.Parameters = { 'MaxHitInvertedFit' : [ '0' ] , ... };
    ApplicationMgr.ExtSvc = [ 'AuditorSvc/AuditorSvc' , ... ];
    MyHitResiduals.Asynchronous = False; // unused

A property's component is everything before its last dot, since a tool or an
engine is named after its owner (``RndmGenSvc.Engine.Seeds``).

Two runs of an unchanged configuration still differ in what k4Bench itself put
there — the paths of the job's working copy and input cache, which are random
or hashed per run, and the settings of k4Bench's own instrumentation — and in
where the stack was mounted (an LCG nightly view lives under the day of the
week). :func:`diff_joboptions` leaves the former out and, given each run's
:func:`stack_aliases`, replaces the latter by a placeholder, so that what it
reports is configuration.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

#: Stands for a property one side of a diff does not have.
ABSENT = type("Absent", (), {"__repr__": lambda self: "ABSENT"})()

#: Components k4Bench configures for its own measurement.
K4BENCH_COMPONENTS = frozenset({"k4BenchAuditor"})
#: Properties k4Bench sets on the job's own components.
K4BENCH_PROPERTIES = frozenset({("JobOptionsSvc", "DUMPFILE")})

#: k4Bench's per-run paths: the input cache, named by a hash of the input's
#: URL (whether the input changed is the run record's to say), and the job's
#: working copy, named by its label and a random suffix.
_K4BENCH_PATHS = (
    (re.compile(r"/tmp/k4bench-inputs/[0-9a-f]+/"), "<input cache>/"),
    (re.compile(r"/tmp/k4bench-[^/]+/"), "<work dir>/"),
)

_LINE = re.compile(r"^(?P<key>[^=]+?) = (?P<value>.*?);\s*(?://.*)?$")


def parse_joboptions(path: str | Path) -> dict[str, dict[str, object]]:
    """Each component's properties in a ``_joboptions.opts`` dump, by name.

    Values are read as the Python literals the dump writes; one that is not a
    literal is kept as its text.
    """
    options: dict[str, dict[str, object]] = {}
    for line in Path(path).read_text().splitlines():
        match = _LINE.match(line)
        if match is None:
            continue
        component, _, prop = match["key"].rpartition(".")
        try:
            value = ast.literal_eval(match["value"])
        except (ValueError, SyntaxError):
            value = match["value"]
        options.setdefault(component, {})[prop] = value
    return options


def stack_aliases(run_info: Mapping) -> dict[str, str]:
    """Where a run's stack was mounted, as ``{prefix: placeholder}`` for
    :func:`diff_joboptions`: the nightly's install tree (the directory of its
    package manifest) and its view (the directory of its setup script)."""
    aliases = {}
    for key, placeholder in (("k4h_stack_manifest", "<stack>"), ("k4h_stack_setup", "<view>")):
        path = run_info.get(key)
        if path:
            aliases[str(PurePosixPath(path).parent)] = placeholder
    return aliases


def _normalised(value: object, aliases: Mapping[str, str]) -> object:
    if isinstance(value, str):
        for pattern, placeholder in _K4BENCH_PATHS:
            value = pattern.sub(placeholder, value)
        # The longest prefix first, so a nested root is not cut short.
        for prefix in sorted(aliases, key=len, reverse=True):
            value = value.replace(prefix, aliases[prefix])
        return value
    if isinstance(value, list):
        return [_normalised(item, aliases) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalised(item, aliases) for item in value)
    if isinstance(value, dict):
        return {key: _normalised(item, aliases) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class OptionChange:
    """One property whose value differs."""

    component: str
    #: The property, or ``Property['key']`` for one entry of a mapping-valued
    #: property (a wrapped Marlin processor's ``Parameters``).
    option: str
    #: :data:`ABSENT` on the side that lacks it.
    before: object
    after: object


@dataclass(frozen=True)
class JobOptionsDiff:
    """How the second job's options differ from the first's."""

    #: Components only the second job configures.
    added: list[str]
    #: Components only the first job configures.
    removed: list[str]
    #: Changed properties of the components both configure.
    changed: list[OptionChange]


def diff_joboptions(
    a: Mapping[str, Mapping[str, object]],
    b: Mapping[str, Mapping[str, object]],
    *,
    aliases_a: Mapping[str, str] | None = None,
    aliases_b: Mapping[str, str] | None = None,
) -> JobOptionsDiff:
    """What changed from *a* to *b*, two :func:`parse_joboptions` results.

    k4Bench's own components and settings are left out, and every string has
    k4Bench's per-run paths and each side's *aliases* (see
    :func:`stack_aliases`) replaced first. A mapping-valued property is compared
    entry by entry.
    """
    aliases_a, aliases_b = aliases_a or {}, aliases_b or {}
    names_a = [name for name in a if name not in K4BENCH_COMPONENTS]
    names_b = [name for name in b if name not in K4BENCH_COMPONENTS]
    changed = []
    for component in (name for name in names_a if name in b):
        props_a, props_b = a[component], b[component]
        for prop in [*props_a, *(p for p in props_b if p not in props_a)]:
            if (component, prop) in K4BENCH_PROPERTIES:
                continue
            before = _normalised(props_a.get(prop, ABSENT), aliases_a)
            after = _normalised(props_b.get(prop, ABSENT), aliases_b)
            if before == after:
                continue
            if isinstance(before, dict) and isinstance(after, dict):
                for key in [*before, *(k for k in after if k not in before)]:
                    if before.get(key, ABSENT) != after.get(key, ABSENT):
                        changed.append(OptionChange(
                            component, f"{prop}[{key!r}]",
                            before.get(key, ABSENT), after.get(key, ABSENT),
                        ))
            else:
                changed.append(OptionChange(component, prop, before, after))
    return JobOptionsDiff(
        added=[name for name in names_b if name not in a],
        removed=[name for name in names_a if name not in b],
        changed=changed,
    )
