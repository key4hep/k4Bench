"""Recorded LCG fixtures exercise capture, measured scope and full-stack diffs."""
from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

from k4bench.provenance import stack
from k4bench.provenance.diff import diff_stacks, hash_only_rebuilds

FIXTURES = Path(__file__).parents[1] / "fixtures/stack"


def manifests():
    return [gzip.decompress((FIXTURES / name).read_bytes()).decode() for name in stack.STACK_MANIFESTS]


def snapshots():
    loaded = json.loads((FIXTURES / "loaded.json").read_text())
    probe = json.loads(gzip.decompress((FIXTURES / "probe.json.gz").read_bytes()))
    datasets = {k: Path(v).name for k, v in probe["g4data"].items()}
    out = []
    for day, version in (("02", "11.4.2"), ("03", "11.4.3")):
        info = {
            "k4h_packages": json.loads((FIXTURES / f"HEAD-2026-10-{day}.json").read_text()),
            "k4h_loaded_packages": {n: dict(p) for n, p in loaded.items()},
            "g4_datasets": datasets,
        }
        info["k4h_loaded_packages"]["Geant4"]["version"] = version
        # The saved probe's HEAD metadata belongs to Oct 12, not these endpoints.
        for name in info["k4h_loaded_packages"]:
            if name in info["k4h_packages"]:
                info["k4h_loaded_packages"][name].update(info["k4h_packages"][name])
        out.append(stack.stack_snapshot(info, {"os": "AlmaLinux 9.8 (Olive Jaguar)"}, manifests()))
    return out


def test_real_manifest_headers_head_pins_and_contrib():
    external, contrib = map(stack.parse_manifest, manifests())
    assert len(external) == 992
    assert sum(p["version"] == "HEAD" for p in external.values()) == 64
    assert external["Geant4"]["version"] == "11.4.3"
    assert contrib["gcc"]["install_path"] == "/cvmfs/sft.cern.ch/lcg/contrib/gcc/16/x86_64-el9"
    assert contrib["gcc"]["version"] == "16.1.0"


def test_probe_mapping_is_not_tied_to_cvmfs_availability(monkeypatch):
    packages = json.loads((FIXTURES / "loaded.json").read_text())
    probe = json.loads(gzip.decompress((FIXTURES / "probe.json.gz").read_bytes()))
    # Both fixture sources already resolved their paths while CVMFS was available.
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    loaded, unknown = stack.mapped_packages(probe["maps"], packages)
    assert loaded == set(packages)
    assert not [path for path in unknown if path.startswith("/cvmfs/")]


def test_mapping_symlinks_longest_prefix_external_install_and_unknown(tmp_path):
    package = stack.parse_manifest(manifests()[0])["Geant4"]
    release = tmp_path / "stable/Geant4"
    nested = release / "plugins"
    nested.mkdir(parents=True)
    view = tmp_path / "view"
    view.symlink_to(release, target_is_directory=True)
    packages = {
        "Geant4": {**package, "install_path": str(release)},
        "nested": {**package, "install_path": str(nested)},
    }
    paths = [str(view / "lib.so"), str(view / "plugins/lib.so"), str(release) + "-other/lib.so", "/usr/lib/libc.so"]
    loaded, unknown = stack.mapped_packages(paths, packages)
    assert loaded == {"Geant4", "nested"}
    assert unknown == sorted(paths[2:])


def test_full_diff_keeps_head_compare_and_filters_pin_history():
    base, head = snapshots()
    changes = {c.name: c for c in diff_stacks(base, head)}
    assert changes["Geant4"].base_value == "11.4.2"
    assert changes["Geant4"].compare_url == "https://gitlab.cern.ch/sft/stacks/lcgcmake/-/commits/e411040b8?search=Geant4"
    assert changes["acts"].compare_url == "https://github.com/acts-project/acts/compare/80b66a5...6f2940a"
    assert list(changes)[0] == "Geant4"
    assert changes["acts"].loaded is False


def test_rebuilds_added_removed_datasets_os_and_unknown_loads():
    base, _ = snapshots()
    packages = {n: dict(p) for n, p in base.packages.items()}
    packages["Geant4"]["hash"] = "changed-build-hash"
    packages.pop("acts")
    packages["new"] = packages["Geant4"]
    head = replace(base, packages=packages, datasets={**base.datasets, "G4LEDATA": "G4EMLOW8.7"}, os="AlmaLinux 9.7", loaded=None)
    changes = {c.name: c for c in diff_stacks(base, head)}
    assert "Geant4" not in changes
    assert hash_only_rebuilds(base, head) == 1
    assert changes["acts"].status == "removed"
    assert changes["new"].status == "added"
    assert changes["new"].loaded is None
    assert changes["new"].compare_url is None
    assert changes["G4LEDATA"].loaded is True
    assert changes["OS"].compare_url is None


def test_old_run_info_is_not_an_empty_loaded_set(tmp_path):
    info = {"k4h_packages": json.loads((FIXTURES / "HEAD-2026-10-02.json").read_text())}
    (tmp_path / "run_info.json").write_text(json.dumps(info))
    snapshot = stack.stack_snapshot(json.loads((tmp_path / "run_info.json").read_text()), {})
    assert snapshot.packages == info["k4h_packages"]
    assert snapshot.loaded is None and not snapshot.complete


@pytest.mark.parametrize("missing", [False, True])
def test_capture_unions_configs_and_removes_private_paths(tmp_path, monkeypatch, missing):
    setup = tmp_path / "lcg/views/devkey-head/Mon/platform/setup.sh"
    setup.parent.mkdir(parents=True)
    setup.touch()
    nightly = tmp_path / "lcg/nightlies/devkey-head/Mon"
    nightly.mkdir(parents=True)
    for source, kind in zip(manifests(), ("externals", "contrib")):
        (nightly / f"LCG_{kind}_platform.txt").write_text(source)
    run = tmp_path / "run"
    run.mkdir()
    for label in ("baseline", "no_TPC"):
        if missing and label == "no_TPC":
            continue
        stack.loaded_paths_file(run, label).write_text(f"# k4bench loaded paths v1\n/{label}/library.so\nG4LEDATA=/data/G4EMLOW8.8\n")
    seen = []
    def mapping(paths, packages):
        seen.extend(paths)
        return {"Geant4", "gcc"}, []
    monkeypatch.setattr(stack, "mapped_packages", mapping)
    monkeypatch.setattr(stack, "build_info", lambda install: {"githash": "e411040b8"})
    info = {"configs": ["baseline", "no_TPC"]}
    stack.capture_stack(run, info, {"K4H_STACK_SETUP": str(setup)})
    assert info["k4h_stack_capture"] == 1
    assert set(p.name for p in run.iterdir()) == set(stack.STACK_MANIFESTS)
    if missing:
        assert "k4h_loaded_packages" not in info
    else:
        assert set(seen) == {"/baseline/library.so", "/no_TPC/library.so"}
        assert set(info["k4h_loaded_packages"]) == {"Geant4", "gcc"}
        assert info["g4_datasets"] == {"G4LEDATA": "G4EMLOW8.8"}


def test_missing_head_revision_is_not_a_change_or_a_meaningful_link():
    base, _ = snapshots()
    packages = {n: dict(p) for n, p in base.packages.items()}
    packages["acts"].pop("commit")
    packages["acts"]["hash"] = "unknown-build"
    head = replace(base, packages=packages)
    assert diff_stacks(base, head) == []
    assert hash_only_rebuilds(base, head) == 0


def test_unloaded_pinned_version_has_no_history_link():
    base, head = snapshots()
    base = replace(base, loaded=frozenset())
    head = replace(head, loaded=frozenset())
    geant4 = next(c for c in diff_stacks(base, head) if c.name == "Geant4")
    assert geant4.compare_url is None


@pytest.mark.parametrize("name", ["Geant4", "acts"])
def test_rebuild_count_distinguishes_same_code_from_a_version_or_commit_change(name):
    base, changed = snapshots()
    packages = {n: dict(p) for n, p in base.packages.items()}
    packages[name]["hash"] += "-rebuilt"
    rebuilt = replace(base, packages=packages)
    assert diff_stacks(base, rebuilt) == []
    assert hash_only_rebuilds(base, rebuilt) == 1

    packages[name] = {**changed.packages[name], "hash": packages[name]["hash"]}
    assert [c.name for c in diff_stacks(base, rebuilt)] == [name]
    assert hash_only_rebuilds(base, rebuilt) == 0


def test_pinned_version_with_changed_upstream_revision_is_a_code_change():
    base, _ = snapshots()
    before = {n: dict(p) for n, p in base.packages.items()}
    before["Geant4"]["commit"] = "aaaaaaa"
    after = {n: dict(p) for n, p in before.items()}
    after["Geant4"].update(commit="bbbbbbb", hash=before["Geant4"]["hash"] + "-rebuilt")
    base, head = replace(base, packages=before), replace(base, packages=after)
    change, = diff_stacks(base, head)
    assert (change.name, change.kind, change.status) == ("Geant4", "pin", "changed")
    assert (change.base_value, change.head_value) == ("11.4.2 (aaaaaaa)", "11.4.2 (bbbbbbb)")
    assert hash_only_rebuilds(base, head) == 0


@pytest.mark.parametrize("added", [True, False])
def test_head_package_presence_is_known_without_its_revision(added):
    base, _ = snapshots()
    packages = {n: dict(p) for n, p in base.packages.items()}
    packages["acts"].pop("commit")
    present = replace(base, packages=packages)
    absent = replace(base, packages={n: p for n, p in packages.items() if n != "acts"})
    before, after = (absent, present) if added else (present, absent)
    change, = diff_stacks(before, after)
    assert change.kind == "head"
    assert change.status == ("added" if added else "removed")
    assert change.compare_url is None
