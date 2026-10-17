"""Where a run's components are defined: the stack package whose install holds
each component's library, and that package's repository at the revision the
stack built.

Links point at the commit the package's build recorded. A HEAD package's
repository is the one the stack's metadata names. A package LCG builds from a
release tarball has none recorded, so it is linked only where
:data:`RELEASE_REPOS` names its repository, and otherwise named with its
version alone.
"""

from __future__ import annotations

import gzip
import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

from k4bench.provenance.stack import STACK_MANIFESTS, RepoRef, parse_repo, stack_snapshot

#: Repositories of release-built packages that define components, keyed by
#: manifest name. Only the location is assumed: the revision is the commit
#: the build recorded. Gaudi's is Key4hep's public GitHub mirror, as upstream
#: GitLab needs a CERN login.
RELEASE_REPOS = {"Gaudi": "https://github.com/key4hep/Gaudi"}


@dataclass(frozen=True)
class PackageSource:
    """One stack package: its name, the commit built and its repository."""

    package: str
    #: The commit the build recorded.
    ref: str | None
    #: ``None`` on an unrecognized forge, or where none is known.
    repo: RepoRef | None
    #: A release's version, which the label names in place of the commit.
    version: str | None = None

    @property
    def label(self) -> str:
        shown = self.version or self.ref
        return f"{self.package} @ {shown}" if shown else self.package

    @property
    def tree_url(self) -> str | None:
        return self.repo.tree_url(self.ref) if self.repo and self.ref else None

    def search_url(self, cls: str) -> str | None:
        """Where class *cls* (namespaced or templated alike) is defined."""
        if not self.repo or not self.ref:
            return None
        return self.repo.code_search_url(class_symbol(cls), self.ref)


def class_symbol(cls: str) -> str:
    """The unqualified name of class *cls*, its template arguments dropped."""
    return re.split(r"<", cls, maxsplit=1)[0].rsplit("::", 1)[-1].strip()


def load_run_packages(run_dir: str | Path) -> dict[str, dict]:
    """The packages *run_dir*'s stack holds, with install paths where its
    manifests were saved; empty for a run without ``run_info.json``."""
    run_dir = Path(run_dir)
    try:
        info = json.loads((run_dir / "run_info.json").read_text())
    except (OSError, ValueError):
        return {}
    manifests = []
    for name in STACK_MANIFESTS:
        try:
            manifests.append(gzip.decompress((run_dir / name).read_bytes()).decode())
        except (OSError, EOFError, UnicodeDecodeError, zlib.error):
            continue
    return stack_snapshot(info, {}, manifests).packages


def package_of(library: str, packages: dict[str, dict]) -> PackageSource | None:
    """The package whose install holds *library*, by its longest install path.

    The auditor records *library* resolved through symlinks, so it is matched
    against the install path the manifest names and against the directory that
    path resolved to when the run was captured. Paths are compared as
    recorded: LCG's rotating view names may point elsewhere now.
    """
    if not library:
        return None
    match = max(
        (
            (len(prefix), name)
            for name, package in packages.items()
            for key in ("install_path", "install_realpath")
            if (prefix := package.get(key, "").rstrip("/") + "/") != "/"
            and library.startswith(prefix)
        ),
        default=None,
    )
    if match is None:
        return None
    name = match[1]
    package = packages[name]
    if package.get("version") == "HEAD":
        return PackageSource(name, package.get("commit"), parse_repo(package.get("repo_url")))
    return PackageSource(
        name, package.get("commit"), parse_repo(RELEASE_REPOS.get(name)), package.get("version"),
    )
