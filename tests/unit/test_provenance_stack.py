"""Unit tests for :mod:`k4bench.provenance.stack`.

The reader runs inside the nightly benchmark job, in a CI venv without
Streamlit, against a real CVMFS tree. These tests build a synthetic release
with the same layout so they stay offline and fast; the shapes asserted here
(``.spack-db/index.json`` schema, install-dir naming, the two recipe nestings)
were taken from a live ``sw-nightlies.hsf.org`` release.
"""

from __future__ import annotations

import json
from pathlib import Path

from k4bench.provenance import stack


def test_provenance_module_does_not_import_streamlit():
    # Imported by nightly_benchmark.sh in a venv without Streamlit; a stray
    # dashboard import would break the nightly rather than a test.
    assert "streamlit" not in stack.__dict__


def _add_package(
    release: Path,
    name: str,
    commit: str,
    version: str = "develop",
    *,
    git_url: str | None = "https://github.com/key4hep/thing.git",
    builtin: bool = False,
) -> Path:
    """Create one git-built install and return its path."""
    install = release / name / f"{commit}_{version}-abcdef"
    install.mkdir(parents=True)
    if git_url is not None:
        # Spack's builtin recipes nest a level deeper than namespaced ones.
        rel = f"spack_repo/builtin/packages/{name}" if builtin else f"k4/packages/{name}"
        recipe = install / ".spack" / "repos" / rel / "package.py"
        recipe.parent.mkdir(parents=True)
        recipe.write_text(
            f'class {name.capitalize()}(CMakePackage):\n'
            f'    homepage = "https://example.invalid/{name}"\n'
            f'    git = "{git_url}"\n'
        )
    return install


def _write_db(release: Path, packages: list[dict], version: str = "8") -> None:
    installs = {}
    for i, pkg in enumerate(packages):
        spec = {"name": pkg["name"], "version": pkg["spec_version"]}
        if pkg.get("commit"):
            spec["parameters"] = {"commit": pkg["commit"]}
        installs[f"hash{i}"] = {"path": pkg["path"], "spec": spec}
    db = release / ".spack-db"
    db.mkdir(exist_ok=True)
    (db / "index.json").write_text(
        json.dumps({"database": {"version": version, "installs": installs}})
    )


def _release(tmp_path: Path) -> Path:
    """A release tree with a namespaced package, a builtin one, and a tarball one."""
    release = tmp_path / "releases" / "2026-07-10" / "x86_64-almalinux9-gcc14.2.0-opt"
    release.mkdir(parents=True)
    k4geo = _add_package(
        release, "k4geo", "a" * 40, git_url="https://github.com/key4hep/k4geo.git"
    )
    dd4hep = _add_package(
        release, "dd4hep", "b" * 40,
        git_url="https://github.com/AIDASoft/DD4hep.git", builtin=True,
    )
    # Built from a release tarball: no upstream commit, must never be recorded.
    boost = release / "boost" / "1.85.0-zzzzzz"
    boost.mkdir(parents=True)
    _write_db(release, [
        {"name": "k4geo", "path": str(k4geo), "commit": "a" * 40,
         "spec_version": f"{'a' * 40}=develop"},
        {"name": "dd4hep", "path": str(dd4hep), "commit": "b" * 40,
         "spec_version": f"{'b' * 40}=develop"},
        {"name": "boost", "path": str(boost), "spec_version": "1.85.0"},
    ])
    return release


# ── parse_repo ───────────────────────────────────────────────────────────────

def test_parse_repo_handles_every_recipe_spelling():
    # Real recipes are inconsistent about the .git suffix: k4geo has it,
    # fcc-config does not.
    for url in (
        "https://github.com/key4hep/k4geo.git",
        "https://github.com/key4hep/k4geo",
        "https://github.com/key4hep/k4geo/",
        "git@github.com:key4hep/k4geo.git",
    ):
        repo = stack.parse_repo(url)
        assert (repo.forge, repo.host, repo.slug) == ("github", "github.com", "key4hep/k4geo")


def test_parse_repo_builds_github_links():
    repo = stack.parse_repo("https://github.com/key4hep/k4geo.git")
    assert repo.url == "https://github.com/key4hep/k4geo"
    assert repo.compare_url("a" * 40, "c" * 40) == (
        f"https://github.com/key4hep/k4geo/compare/{'a' * 40}...{'c' * 40}"
    )


def test_parse_repo_builds_gitlab_links():
    # The only two non-GitHub packages in the stack are on self-hosted GitLab,
    # which nests its repo views under /-/.
    repo = stack.parse_repo("https://gitlab.cern.ch/acts/OpenDataDetector.git")
    assert (repo.forge, repo.host, repo.slug) == ("gitlab", "gitlab.cern.ch", "acts/OpenDataDetector")
    assert repo.compare_url("a" * 40, "c" * 40) == (
        f"https://gitlab.cern.ch/acts/OpenDataDetector/-/compare/{'a' * 40}...{'c' * 40}"
    )


def test_parse_repo_allows_nested_gitlab_groups():
    # GitLab groups nest arbitrarily; GitHub's owner/repo does not.
    repo = stack.parse_repo("https://gitlab.desy.de/ilcsoft/sub/Thing.git")
    assert repo.slug == "ilcsoft/sub/Thing"
    assert repo.url == "https://gitlab.desy.de/ilcsoft/sub/Thing"


def test_parse_repo_rejects_what_it_cannot_link():
    # An unknown forge keeps its commit but gets no links: guessing a URL
    # layout would produce links that quietly 404.
    assert stack.parse_repo("https://git.example.com/a/b.git") is None
    assert stack.parse_repo("https://notgithub.com/a/b") is None
    # Not a GitHub repo root, so a compare built from it would not resolve.
    assert stack.parse_repo("https://github.com/key4hep/k4geo/tree/main") is None
    assert stack.parse_repo(None) is None
    assert stack.parse_repo("") is None


# ── find_release_root ────────────────────────────────────────────────────────

def test_find_release_root_walks_up_from_setup_sh(tmp_path):
    release = _release(tmp_path)
    setup = release / "key4hep-stack" / "2026-07-10-miqtr5" / "setup.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text("# $KEY4HEP_STACK points here\n")
    assert stack.find_release_root(setup) == release


def test_find_release_root_returns_none_without_a_database(tmp_path):
    # Better to record no provenance than to return a plausible wrong root.
    lost = tmp_path / "a" / "b" / "setup.sh"
    lost.parent.mkdir(parents=True)
    lost.write_text("")
    assert stack.find_release_root(lost) is None


def test_find_release_root_does_not_walk_past_the_bound(tmp_path):
    release = _release(tmp_path)
    deep = release.joinpath(*["nest"] * (stack._MAX_WALK_UP + 1)) / "setup.sh"
    deep.parent.mkdir(parents=True)
    deep.write_text("")
    assert stack.find_release_root(deep) is None


# ── read_stack_packages ──────────────────────────────────────────────────────

def test_read_stack_packages_from_database(tmp_path):
    packages = stack.read_stack_packages(_release(tmp_path))

    assert set(packages) == {"k4geo", "dd4hep"}, "tarball installs must be excluded"
    assert packages["k4geo"] == {
        "commit": "a" * 40,
        "version": "develop",
        "repo_url": "https://github.com/key4hep/k4geo.git",
    }
    # Builtin recipes nest one level deeper — the shallow glob alone misses them.
    assert packages["dd4hep"]["repo_url"] == "https://github.com/AIDASoft/DD4hep.git"


def test_read_stack_packages_records_missing_recipe_as_none(tmp_path):
    release = tmp_path / "rel"
    release.mkdir()
    install = _add_package(release, "mystery", "c" * 40, git_url=None)
    _write_db(release, [{"name": "mystery", "path": str(install), "commit": "c" * 40,
                         "spec_version": f"{'c' * 40}=develop"}])
    packages = stack.read_stack_packages(release)
    # The commit is the load-bearing part; a missing URL costs only the PR link.
    assert packages["mystery"]["commit"] == "c" * 40
    assert packages["mystery"]["repo_url"] is None


def test_read_stack_packages_tolerates_unknown_db_version(tmp_path):
    release = _release(tmp_path)
    _write_db(
        release,
        [{"name": "k4geo", "path": str(release / "k4geo" / f"{'a' * 40}_develop-abcdef"),
          "commit": "a" * 40, "spec_version": f"{'a' * 40}=develop"}],
        version="99",
    )
    # A schema bump must not silently drop provenance: the fields we read have
    # been stable across versions, so parse anyway and log.
    assert stack.read_stack_packages(release)["k4geo"]["commit"] == "a" * 40


def test_read_stack_packages_falls_back_to_install_dirs(tmp_path):
    release = _release(tmp_path)
    (release / ".spack-db" / "index.json").write_text("{not json")

    packages = stack.read_stack_packages(release)

    assert set(packages) == {"k4geo", "dd4hep"}, "tarball dir must not parse as git-built"
    assert packages["k4geo"]["commit"] == "a" * 40
    assert packages["k4geo"]["version"] == "develop"
    assert packages["dd4hep"]["repo_url"] == "https://github.com/AIDASoft/DD4hep.git"


def test_install_dir_fallback_keeps_hyphenated_versions(tmp_path):
    # "{commit}_{version}-{hash}" is ambiguous when the version itself contains
    # a hyphen; the split must take the *last* one. No database here, so this
    # also covers a release tree with no .spack-db at all.
    release = tmp_path / "rel"
    release.mkdir()
    _add_package(release, "thing", "d" * 40, version="v1.2-rc1")
    packages = stack.read_stack_packages(release)
    assert packages["thing"] == {
        "commit": "d" * 40,
        "version": "v1.2-rc1",
        "repo_url": "https://github.com/key4hep/thing.git",
    }


def test_read_stack_packages_survives_a_missing_release(tmp_path):
    # An empty record, never an exception: provenance must not fail a benchmark.
    assert stack.read_stack_packages(tmp_path / "nope") == {}


def test_read_stack_packages_skips_malformed_records(tmp_path):
    release = _release(tmp_path)
    data = json.loads((release / ".spack-db" / "index.json").read_text())
    data["database"]["installs"]["broken"] = {"path": "/x"}  # no "spec"
    (release / ".spack-db" / "index.json").write_text(json.dumps(data))
    # One bad record must not cost the other 62 packages their provenance.
    assert set(stack.read_stack_packages(release)) == {"k4geo", "dd4hep"}
