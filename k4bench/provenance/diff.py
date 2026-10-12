"""Diff two Key4hep stacks — which packages moved, and to what.

The unit of comparison is the **release** (the nightly tag), not the day a
benchmark ran. The nightly build lags: on a day with no new nightly the
benchmark sources the newest one available, so several consecutive run dates
routinely share one identical stack. Comparing run dates would invent changes
that never happened.

An empty diff means no recorded changes. Callers distinguish complete captures
from missing provenance before claiming that the endpoint stacks are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from k4bench.provenance.stack import RepoRef, StackSnapshot, parse_repo

#: A package present in only one of the two stacks. Added and removed packages
#: are rare (a dependency enters or leaves the stack) but are real changes, and
#: silently dropping them would understate a diff.
ADDED = "added"
REMOVED = "removed"
CHANGED = "changed"


@dataclass(frozen=True)
class PackageChange:
    """One package's difference between two stacks.

    A recorded version or commit establishes presence at an endpoint. HEAD
    revisions can be unknown even when a manifest records the package.
    """

    name: str
    base_commit: str | None
    head_commit: str | None
    version: str = ""
    repo_url: str | None = None
    base_version: str | None = None
    head_version: str | None = None
    head_githash: str | None = None
    kind: str = "head"
    loaded: bool | None = None

    @property
    def base_value(self) -> str | None:
        return self._value(self.base_version, self.base_commit)

    @property
    def head_value(self) -> str | None:
        return self._value(self.head_version, self.head_commit)

    def _value(self, version: str | None, commit: str | None) -> str | None:
        if self.kind == "head":
            return commit
        # An unchanged pin only appears because its upstream revision moved.
        if self.kind == "pin" and commit and self.base_version == self.head_version:
            return f"{version} ({commit[:12]})"
        return version

    @property
    def status(self) -> str:
        if self.base_version is None and self.base_commit is None:
            return ADDED
        if self.head_version is None and self.head_commit is None:
            return REMOVED
        return CHANGED

    @property
    def repo(self) -> RepoRef | None:
        """The upstream repository, or ``None`` on an unrecognized forge."""
        return parse_repo(self.repo_url)

    @property
    def compare_url(self) -> str | None:
        """HEAD forge comparison or loaded pin's filtered build history."""
        if self.status != CHANGED:
            return None
        if self.kind == "pin":
            if self.loaded and self.head_githash:
                return ("https://gitlab.cern.ch/sft/stacks/lcgcmake/-/commits/"
                        f"{self.head_githash}?search={quote(self.name, safe='')}")
            return None
        if (self.kind != "head" or not self.base_commit or not self.head_commit
                or not (repo := self.repo)):
            return None
        return repo.compare_url(self.base_commit, self.head_commit)


def _package_key(name: str) -> str:
    """A package name as both stacks can agree on it.

    Spack lower-cases and hyphenates (``dd4hep``, ``fcc-config``,
    ``k4projecttemplate``) where LCG spells names as upstream does (``DD4hep``,
    ``fcc_config``, ``k4_project_template``); a window spanning a platform
    migration compares one of each."""
    return name.lower().replace("-", "").replace("_", "")


def _same_commit(a: str | None, b: str | None) -> bool:
    """Whether two recorded commits name the same revision. Spack records the
    full sha and LCG an abbreviated one, so a prefix match is the same commit."""
    if not a or not b:
        return a == b
    return a.startswith(b) or b.startswith(a)


def _code_changed(before: dict, after: dict) -> bool | None:
    """Compare versions and revisions; None means a HEAD revision is missing.

    A pin's recorded upstream revision still counts when both sides have one,
    so a same-version rebuild from different code is not a hash-only rebuild."""
    if before.get("version") != after.get("version"):
        return True
    base_commit, head_commit = before.get("commit"), after.get("commit")
    if not base_commit or not head_commit:
        return None if before.get("version") == "HEAD" else False
    return not _same_commit(base_commit, head_commit)


def _paired(base: dict, head: dict) -> list[tuple[str, dict | None, dict | None]]:
    """``(name, base entry, head entry)`` per package across both maps, matched
    on :func:`_package_key` and named as the head spells it."""
    base_by_key = {_package_key(name): name for name in base}
    head_by_key = {_package_key(name): name for name in head}
    return [
        (
            head_by_key.get(key) or base_by_key[key],
            base.get(base_by_key[key]) if key in base_by_key else None,
            head.get(head_by_key[key]) if key in head_by_key else None,
        )
        for key in set(base_by_key) | set(head_by_key)
    ]


def diff_packages(base: dict, head: dict) -> list[PackageChange]:
    """Packages that differ between two ``k4h_packages`` maps.

    Changed packages come first (they are what a regression hunt is looking
    for), then added, then removed; alphabetical within each group. An empty
    result means the tracked git commits are identical.
    """
    changes: list[PackageChange] = []
    for name, before, after in _paired(base, head):
        base_commit = (before or {}).get("commit")
        head_commit = (after or {}).get("commit")
        if _same_commit(base_commit, head_commit):
            continue
        # Prefer the head's metadata: it describes the stack as it is now.
        meta = after or before or {}
        changes.append(PackageChange(
            name=name,
            base_commit=base_commit,
            head_commit=head_commit,
            version=str(meta.get("version") or ""),
            repo_url=(after or {}).get("repo_url") or (before or {}).get("repo_url"),
        ))
    order = {CHANGED: 0, ADDED: 1, REMOVED: 2}
    return sorted(changes, key=lambda c: (order[c.status], c.name))


def unchanged_packages(base: dict, head: dict) -> list[str]:
    """Packages present in both stacks at the same commit.

    Reported as a count rather than a table: "11 repos unchanged" is context a
    reader needs to size the diff, but eleven identical rows are not.
    """
    return sorted(
        name for name, before, after in _paired(base, head)
        if before is not None and after is not None
        and _same_commit((before or {}).get("commit"), (after or {}).get("commit"))
    )


def diff_stacks(base: StackSnapshot, head: StackSnapshot) -> list[PackageChange]:
    """Version/commit changes only; build hashes alone never manufacture a row.

    Loaded means observed at either endpoint in this benchmark scope. If either
    capture is absent, an unobserved package's load status remains unknown.
    """
    loaded_names = (base.loaded or frozenset()) | (head.loaded or frozenset())
    loaded = {_package_key(name) for name in loaded_names}
    unobserved = False if base.loaded is not None and head.loaded is not None else None
    changes = []
    for name, before, after in _paired(base.packages, head.packages):
        if before is not None and after is not None:
            if _code_changed(before, after) is not True:
                continue
        # A HEAD/pin transition is represented by versions and any known commit.
        versions = {p.get("version") for p in (before, after) if p is not None}
        b, h = before or {}, after or {}
        changes.append(PackageChange(
            name=name, base_commit=b.get("commit"), head_commit=h.get("commit"),
            version=h.get("version", b.get("version", "")),
            repo_url=h.get("repo_url") or b.get("repo_url"),
            base_version=b.get("version"), head_version=h.get("version"),
            head_githash=h.get("githash"), kind="head" if versions == {"HEAD"} else "pin",
            loaded=True if _package_key(name) in loaded else unobserved,
        ))
    if base.datasets is not None and head.datasets is not None:
        for name in sorted(base.datasets.keys() | head.datasets.keys()):
            before, after = base.datasets.get(name), head.datasets.get(name)
            if before != after:
                changes.append(PackageChange(
                    name, None, None, base_version=before, head_version=after, kind="dataset",
                    loaded=True if "geant4" in loaded else unobserved,
                ))
    if base.os and head.os and base.os != head.os:
        changes.append(PackageChange(
            "OS", None, None, base_version=base.os, head_version=head.os, kind="os", loaded=True,
        ))
    order = {CHANGED: 0, ADDED: 1, REMOVED: 2}
    return sorted(changes, key=lambda c: (c.loaded is not True, order[c.status], c.name.lower()))


def hash_only_rebuilds(base: StackSnapshot, head: StackSnapshot) -> int:
    """Count changed build hashes only when the recorded code stayed the same."""
    count = 0
    for _, before, after in _paired(base.packages, head.packages):
        if before is None or after is None:
            continue
        base_hash, head_hash = before.get("hash"), after.get("hash")
        if not base_hash or not head_hash or base_hash == head_hash:
            continue
        if _code_changed(before, after) is False:
            count += 1
    return count
