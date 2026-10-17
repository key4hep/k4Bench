"""Unit tests for linking a run's components to their code
(:mod:`k4bench.provenance.sources`)."""

from __future__ import annotations

import gzip
import json

from k4bench.provenance import sources
from k4bench.provenance.stack import STACK_MANIFESTS, parse_repo

NIGHTLY = "/cvmfs/sft-nightlies.cern.ch/lcg/nightlies/devkey-head/Sat"
PLATFORM = "x86_64-el9-gcc16-opt"

PACKAGES = {
    "k4fwcore": {
        "version": "HEAD", "install_path": f"{NIGHTLY}/k4fwcore/HEAD/{PLATFORM}",
        "commit": "767ff81", "repo_url": "https://github.com/key4hep/k4FWCore.git",
    },
    "k4fwcore_extra": {
        "version": "HEAD", "install_path": f"{NIGHTLY}/k4fwcore/HEAD/{PLATFORM}/extra",
        "commit": "1234567", "repo_url": "https://github.com/key4hep/Extra.git",
    },
    "Gaudi": {"version": "v41r0", "install_path": f"{NIGHTLY}/Gaudi/v41r0/{PLATFORM}",
              "commit": "bdbe774"},
    "ROOT": {"version": "6.40.04", "install_path": f"{NIGHTLY}/ROOT/6.40.04/{PLATFORM}",
             "commit": "1111111"},
    "lctuple": {"version": "HEAD", "install_path": f"{NIGHTLY}/lctuple/HEAD/{PLATFORM}",
                "commit": "aaaaaaa", "repo_url": None},
}


def test_a_head_package_links_at_its_built_commit():
    source = sources.package_of(
        f"{NIGHTLY}/k4fwcore/HEAD/{PLATFORM}/lib/gaudi-plugins/libk4FWCorePlugins.so", PACKAGES,
    )
    assert source.label == "k4fwcore @ 767ff81"
    assert source.tree_url == "https://github.com/key4hep/k4FWCore/tree/767ff81"
    assert source.search_url("Reader") == (
        "https://github.com/search?q=repo%3Akey4hep%2Fk4FWCore%20symbol%3AReader&type=code"
    )


def test_the_longest_install_path_wins_and_a_prefix_must_end_at_a_directory():
    assert sources.package_of(
        f"{NIGHTLY}/k4fwcore/HEAD/{PLATFORM}/extra/lib/libX.so", PACKAGES,
    ).package == "k4fwcore_extra"
    assert sources.package_of(f"{NIGHTLY}/k4fwcore/HEAD/{PLATFORM}2/lib/libX.so", PACKAGES) is None


def test_gaudi_links_to_its_github_mirror_at_its_built_commit():
    source = sources.package_of(f"{NIGHTLY}/Gaudi/v41r0/{PLATFORM}/lib/libGaudiKernel.so", PACKAGES)
    assert source.label == "Gaudi @ v41r0"
    assert source.tree_url == "https://github.com/key4hep/Gaudi/tree/bdbe774"
    assert source.search_url("Gaudi::Sequencer") == (
        "https://github.com/search?q=repo%3Akey4hep%2FGaudi%20symbol%3ASequencer&type=code"
    )


def test_gaudi_without_a_recorded_commit_is_not_linked():
    packages = {**PACKAGES, "Gaudi": {k: v for k, v in PACKAGES["Gaudi"].items() if k != "commit"}}
    source = sources.package_of(f"{NIGHTLY}/Gaudi/v41r0/{PLATFORM}/lib/libGaudiKernel.so", packages)
    assert source.label == "Gaudi @ v41r0"
    assert source.tree_url is None
    assert source.search_url("Gaudi::Sequencer") is None


def test_a_package_without_a_known_repository_is_named_but_not_linked():
    for library in (f"{NIGHTLY}/ROOT/6.40.04/{PLATFORM}/lib/libCore.so",
                    f"{NIGHTLY}/lctuple/HEAD/{PLATFORM}/lib/libLCTuple.so"):
        source = sources.package_of(library, PACKAGES)
        assert source is not None
        assert source.tree_url is None
        assert source.search_url("MergeCollections") is None


def test_an_unrecorded_library_has_no_package():
    assert sources.package_of("", PACKAGES) is None
    assert sources.package_of("/usr/lib64/libc.so.6", PACKAGES) is None
    assert sources.package_of(f"{NIGHTLY}/k4fwcore/HEAD/{PLATFORM}/lib/libX.so", {}) is None


def test_class_symbol_drops_namespaces_and_template_arguments():
    assert sources.class_symbol("marlin::AIDAProcessor") == "AIDAProcessor"
    assert sources.class_symbol("HepRndm::Engine<CLHEP::RanluxEngine>") == "Engine"
    assert sources.class_symbol("DDPlanarDigiProcessor") == "DDPlanarDigiProcessor"


def test_tree_urls_follow_the_forge():
    assert parse_repo("https://gitlab.cern.ch/a/b.git").tree_url("v1") == (
        "https://gitlab.cern.ch/a/b/-/tree/v1"
    )
    assert parse_repo("https://github.com/a/b").tree_url("abc") == "https://github.com/a/b/tree/abc"


def test_a_runs_packages_combine_its_manifests_and_recorded_commits(tmp_path):
    (tmp_path / "run_info.json").write_text(json.dumps({"k4h_packages": {
        "k4fwcore": {"commit": "767ff81", "version": "HEAD",
                     "repo_url": "https://github.com/key4hep/k4FWCore.git"},
    }}))
    (tmp_path / STACK_MANIFESTS[0]).write_bytes(gzip.compress(
        f"k4fwcore; 98b4e; HEAD; {NIGHTLY}/k4fwcore/HEAD/{PLATFORM}; Gaudi-242b8\n".encode()
    ))
    packages = sources.load_run_packages(tmp_path)
    source = sources.package_of(f"{NIGHTLY}/k4fwcore/HEAD/{PLATFORM}/lib/libk4FWCorePlugins.so",
                                packages)
    assert source.label == "k4fwcore @ 767ff81"


def test_a_library_under_a_symlinked_install_matches_its_recorded_real_path(tmp_path):
    """The auditor resolves symlinks; the manifest names the linked path."""
    real = f"/cvmfs/sft-nightlies.cern.ch/lcg/releases/k4fwcore/HEAD-98b4e/{PLATFORM}"
    (tmp_path / "run_info.json").write_text(json.dumps({"k4h_loaded_packages": {
        "k4fwcore": {"version": "HEAD", "commit": "767ff81", "install_realpath": real,
                     "repo_url": "https://github.com/key4hep/k4FWCore.git"},
    }}))
    (tmp_path / STACK_MANIFESTS[0]).write_bytes(gzip.compress(
        f"k4fwcore; 98b4e; HEAD; {NIGHTLY}/k4fwcore/HEAD/{PLATFORM}; Gaudi-242b8\n".encode()
    ))
    packages = sources.load_run_packages(tmp_path)
    source = sources.package_of(f"{real}/lib/libk4FWCorePlugins.so", packages)
    assert source.label == "k4fwcore @ 767ff81"
    assert sources.package_of(f"{real}2/lib/libX.so", packages) is None


def test_a_run_without_run_info_has_no_packages(tmp_path):
    assert sources.load_run_packages(tmp_path) == {}
