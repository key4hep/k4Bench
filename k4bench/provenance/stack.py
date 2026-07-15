"""Read a Key4hep release's git provenance off CVMFS.

Spack records the exact upstream commit of every package it builds from git, in
three independent places that this module reads in order of preference:

1. ``{release}/.spack-db/index.json`` — the install database, one file for the
   whole stack (``spec.parameters.commit``).
2. the install directory name, ``{release}/{pkg}/{commit}_{version}-{hash}`` —
   used when the database is unreadable or its schema moved. Coarser (no
   dependency info) but essentially immune to schema churn.
3. ``{install}/.spack/repos/**/packages/{pkg}/package.py`` — the recipe Spack
   ships beside each install, whose ``git =`` attribute is the upstream URL.
   Every git-built package in the stack carries one, including the ones from
   Spack's own builtin repo (those nest one level deeper, under
   ``repos/spack_repo/builtin/``).

Nothing here raises: provenance is metadata *about* a benchmark, and a stack
this module cannot parse must degrade to an empty record rather than fail the
benchmark that produced the measurements.
"""

from __future__ import annotations

import glob
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

#: Spack install-database schema this reader was written against. A different
#: version is not fatal — the fields we read (``name``/``version``/
#: ``parameters.commit``) have been stable across schema bumps — but it is
#: logged, and any parse failure then falls back to the directory-name scan.
SPACK_DB_VERSION = "8"

#: Install directory name for a git-built package: ``{commit}_{version}-{hash}``.
#: The 40-hex prefix is what distinguishes it from a release-tarball install
#: (``{version}-{hash}``), which has no upstream commit to record.
_INSTALL_DIR_RE = re.compile(r"^(?P<commit>[0-9a-f]{40})_(?P<version>.+)-[a-z0-9]+$")

#: ``git = "https://github.com/key4hep/k4geo.git"`` in a Spack recipe.
_GIT_ATTR_RE = re.compile(r"""^\s*git\s*=\s*["']([^"']+)["']""", re.M)

#: Both URL spellings Spack recipes use, with the ``.git`` suffix optional —
#: recipes are inconsistent about it (k4geo has it, fcc-config does not).
_URL_RE = re.compile(
    r"^(?:https?://(?:www\.)?|git@)(?P<host>[^/:]+)[:/](?P<slug>.+?)(?:\.git)?/?$"
)

#: Path shape for the compare view, keyed on forge. GitLab nests its repo views
#: under ``/-/``; GitHub does not. Everything else keeps its commit but gets no
#: link — a wrong link is worse than none.
_FORGE_INFIX = {"github": "", "gitlab": "/-"}

#: How far up from ``$KEY4HEP_STACK`` to look for the release root. The setup
#: script sits 3 levels below it today; the extra slack absorbs a layout change
#: without letting a bad input walk to the CVMFS root.
_MAX_WALK_UP = 6


def find_release_root(stack_setup: str | Path) -> Path | None:
    """Return the release root for a ``$KEY4HEP_STACK`` path, or ``None``.

    ``$KEY4HEP_STACK`` points at the stack's own ``setup.sh``, several levels
    below the release root::

        /cvmfs/sw-nightlies.hsf.org/key4hep/releases/{date}/{platform}
            └── key4hep-stack/{version}/setup.sh   ← $KEY4HEP_STACK

    We walk up looking for the ``.spack-db`` the caller actually needs, rather
    than matching the ``releases/{date}/{platform}`` layout with a regex: the
    walk tests for the artifact itself, so it keeps working if upstream renests
    the tree, and it fails loudly (``None``) instead of silently returning a
    plausible-looking wrong directory.
    """
    path = Path(stack_setup)
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents)[:_MAX_WALK_UP]:
        if (candidate / ".spack-db" / "index.json").is_file():
            return candidate
    _log.warning("find_release_root: no .spack-db above '%s'", stack_setup)
    return None


@dataclass(frozen=True)
class RepoRef:
    """A package's upstream repository, parsed far enough to build links."""

    forge: str
    host: str
    slug: str

    @property
    def url(self) -> str:
        """The repository's landing page."""
        return f"https://{self.host}/{self.slug}"

    def compare_url(self, base: str, head: str) -> str:
        return f"{self.url}{_FORGE_INFIX[self.forge]}/compare/{base}...{head}"


def parse_repo(url: str | None) -> RepoRef | None:
    """Parse a recipe's ``git`` URL, or ``None`` if we cannot build links for it.

    Recognizes GitHub and GitLab (including the self-hosted ``gitlab.cern.ch``
    and ``gitlab.desy.de`` instances two Key4hep packages live on). An
    unrecognized forge returns ``None``: the package keeps its commit, but
    guessing a URL layout would produce links that quietly 404.
    """
    if not url:
        return None
    match = _URL_RE.match(url.strip())
    if not match:
        return None
    host, slug = match.group("host"), match.group("slug")
    if host == "github.com":
        forge = "github"
        # GitHub repos are exactly owner/repo; anything deeper is not a repo
        # root, so a compare link built from it would not resolve.
        if slug.count("/") != 1:
            return None
    elif "gitlab" in host:
        forge = "gitlab"  # nested groups are allowed, so no depth check
    else:
        return None
    return RepoRef(forge=forge, host=host, slug=slug)


def _repo_url(install_path: str, name: str) -> str | None:
    """The upstream git URL from the recipe Spack ships beside *install_path*.

    Recipes live at ``repos/{namespace}/packages/{pkg}/package.py``, except
    Spack's builtin ones which nest a level deeper under
    ``repos/spack_repo/builtin/``. Both shapes are probed explicitly rather
    than with a recursive glob: a package's shipped repo can carry hundreds of
    sibling recipes, and walking all of them over CVMFS for every package is
    needlessly slow.
    """
    for pattern in (
        f"{install_path}/.spack/repos/*/packages/{name}/package.py",
        f"{install_path}/.spack/repos/*/*/packages/{name}/package.py",
    ):
        for recipe in glob.glob(pattern):
            try:
                match = _GIT_ATTR_RE.search(Path(recipe).read_text())
            except OSError:
                continue
            if match:
                return match.group(1)
    return None


def _from_database(release_root: Path) -> dict[str, dict]:
    """Packages from ``.spack-db/index.json``, or ``{}`` if it cannot be read."""
    index = release_root / ".spack-db" / "index.json"
    try:
        data = json.loads(index.read_text())
        database = data["database"]
        installs = database["installs"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        _log.warning("read_stack_packages: unusable '%s' (%s)", index, exc)
        return {}

    version = str(database.get("version"))
    if version != SPACK_DB_VERSION:
        _log.info(
            "read_stack_packages: spack db version %s (reader written for %s)",
            version, SPACK_DB_VERSION,
        )

    packages: dict[str, dict] = {}
    for record in installs.values():
        try:
            spec = record["spec"]
            commit = spec.get("parameters", {}).get("commit")
            if not commit:
                continue  # built from a release tarball: no upstream commit
            name = spec["name"]
            # ``version`` is "{commit}={branch}" for a git build — keep the branch.
            packages[name] = {
                "commit": commit,
                "version": str(spec.get("version", "")).split("=")[-1],
                "repo_url": _repo_url(record.get("path", ""), name),
            }
        except (KeyError, TypeError, AttributeError) as exc:
            _log.warning("read_stack_packages: skipping malformed record (%s)", exc)
    return packages


def _from_install_dirs(release_root: Path) -> dict[str, dict]:
    """Packages parsed from install directory names — the schema-proof fallback."""
    packages: dict[str, dict] = {}
    try:
        package_dirs = [p for p in release_root.iterdir() if p.is_dir()]
    except OSError as exc:
        _log.warning("read_stack_packages: cannot list '%s' (%s)", release_root, exc)
        return {}

    for package_dir in package_dirs:
        if package_dir.name.startswith("."):
            continue
        try:
            installs = list(package_dir.iterdir())
        except OSError:
            continue
        for install in installs:
            match = _INSTALL_DIR_RE.match(install.name)
            if not match:
                continue
            packages[package_dir.name] = {
                "commit": match.group("commit"),
                "version": match.group("version"),
                "repo_url": _repo_url(str(install), package_dir.name),
            }
            break
    return packages


def read_stack_packages(release_root: str | Path) -> dict[str, dict]:
    """Return ``{package: {"commit", "version", "repo_url"}}`` for a release.

    Only packages Spack built from git are included — a release-tarball install
    has no upstream commit, so it can never be the subject of a blame window.
    ``repo_url`` is the recipe's URL verbatim (``None`` when the recipe is
    missing); use :func:`github_slug` to normalize it.

    Returns ``{}`` rather than raising when *release_root* cannot be read at
    all, so a caller can record "no provenance" and carry on.
    """
    root = Path(release_root)
    packages = _from_database(root)
    if packages:
        return packages
    _log.info("read_stack_packages: falling back to install-dir scan of '%s'", root)
    return _from_install_dirs(root)
