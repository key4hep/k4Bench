"""Tests for the pure PR-comment reproducer builder."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from k4bench import staging
from k4bench.blame import reproduce
from k4bench.blame.reproduce import (
    K4runReproducerFacts,
    artifact_name,
    facts_from,
    render_text,
    sweep_flag,
)


def _row(**overrides):
    values = {
        "detector": "ILD_FCCee_v01",
        "platform": "x86_64-almalinux9-gcc14.2.0-opt",
        "sample": "single_e-_10GeV",
        "label": "no_TPC",
        "metric": "mean_time_s",
        "sub_detector": None,
        "pct_change": 0.3614,
        "last_accepted_run_date": "2026-08-27",
        "last_accepted_run_id": "run-before",
        "onset_run_date": "2026-08-28",
        "onset_run_id": "run-after",
    }
    values.update(overrides)
    return SimpleNamespace(verdict=SimpleNamespace(**values))


def _info(release: str, run: str, **overrides):
    values = {
        "detector": "ILD_FCCee_v01",
        "platform": "x86_64-almalinux9-gcc14.2.0-opt",
        "sample": "single_e-_10GeV",
        "k4h_release_date": release,
        "xml_path": "FCCee/ILD_FCCee/compact/ILD_FCCee_v01/ILD_FCCee_v01.xml",
        "github_run_url": f"https://github.test/actions/runs/{run}",
        "commit_sha": "c2f5766" * 5,
        "n_events": 1000,
        "ddsim_args": "--random.seed 42 --enableGun --gun.particle e-",
        "random_seed": 42,
        "input_files": [],
        "steering_file": "",
    }
    values.update(overrides)
    return values


def _facts(**after_overrides):
    return facts_from(
        _row(),
        _info("2026-08-27", "1"),
        _info("2026-08-28", "2", **after_overrides),
    )


def _halves(body: str) -> list[str]:
    """The two per-release subshells of a rendered recipe, in order.

    They are the last two subshells in the file: the outer fail-fast wrapper
    around every stage opens first, and a shared input fetch — itself a
    subshell, since it has to source a release for ``xrdcp`` — comes between
    that wrapper and the halves whenever both runs read the same sources.
    """
    halves = body.split("\n(\n")[-2:]
    return [half.split("\n)\n", 1)[0] for half in halves]


def _prose(body: str) -> str:
    """The recipe's comment prose as one line: an assertion about what it says
    should not also be an assertion about where textwrap broke a sentence."""
    return " ".join(
        line.lstrip("#").strip()
        for line in body.splitlines()
        if line.startswith("#")
    )


def _facts_with_inputs(before: list[str], after: list[str]):
    facts = facts_from(
        _row(),
        _info("2026-08-27", "1", input_files=before),
        _info("2026-08-28", "2", input_files=after),
    )
    assert facts is not None
    return facts


def test_sweep_flag_inverts_supported_labels_and_rejects_hashed_multi_sweep():
    assert sweep_flag("baseline") == ""
    assert sweep_flag("no_TPC") == "--sweep-detectors TPC"
    assert sweep_flag("only_VertexBarrel") == "--include-only VertexBarrel"
    assert sweep_flag("no_3_detectors_12ab90ef") is None
    assert sweep_flag("custom") is None


def test_missing_record_or_identity_mismatch_returns_none():
    assert facts_from(_row(), None, _info("2026-08-28", "2")) is None
    assert (
        facts_from(
            _row(),
            _info("2026-08-27", "1", detector="another"),
            _info("2026-08-28", "2"),
        )
        is None
    )


def test_parity_differences_are_named_instead_of_suppressing_recipe():
    facts = _facts(n_events=100, random_seed=7, commit_sha="d" * 40)
    assert facts is not None
    assert facts.parity_diffs == ("n_events", "random_seed", "commit_sha")
    body = render_text(facts)
    assert "did NOT measure the same workload" in body
    assert "n_events" in body
    assert "same workload: 1000 events" not in body


def test_input_sources_are_part_of_workload_parity_even_with_same_tmp_name():
    before = _info(
        "2026-08-27", "1",
        input_files=["root://old.example/events.hepmc"],
        ddsim_args="--inputFiles /tmp/events.hepmc --random.seed 42",
    )
    after = _info(
        "2026-08-28", "2",
        input_files=["root://new.example/events.hepmc"],
        ddsim_args="--inputFiles /tmp/events.hepmc --random.seed 42",
    )
    facts = facts_from(_row(), before, after)
    assert facts is not None
    assert facts.parity_diffs == ("input_files",)


def test_release_specific_steering_paths_are_compared_logically():
    before = _info(
        "2026-08-27", "1",
        steering_file="$CLDCONFIG/share/CLDConfig/cld_arc_steer.py",
        resolved_steering_file="/cvmfs/nightly-27/CLDConfig/cld_arc_steer.py",
        ddsim_args="--steeringFile /cvmfs/nightly-27/CLDConfig/cld_arc_steer.py --random.seed 42",
    )
    after = _info(
        "2026-08-28", "2",
        steering_file="$CLDCONFIG/share/CLDConfig/cld_arc_steer.py",
        resolved_steering_file="/cvmfs/nightly-28/CLDConfig/cld_arc_steer.py",
        ddsim_args="--steeringFile /cvmfs/nightly-28/CLDConfig/cld_arc_steer.py --random.seed 42",
    )
    facts = facts_from(_row(), before, after)
    assert facts is not None
    assert facts.parity_diffs == ()
    body = render_text(facts)
    assert body.count("export PYTHONPATH=") == 2
    assert "/cvmfs/nightly-27/CLDConfig" in body
    assert "/cvmfs/nightly-28/CLDConfig" in body


def test_different_logical_steering_or_real_ddsim_option_breaks_parity():
    facts = _facts(
        steering_file="$FCCCONFIG/other.py",
        ddsim_args="--random.seed 42 --enableGun --gun.particle mu-",
    )
    assert facts is not None
    assert facts.parity_diffs == ("ddsim_args", "steering_file")


def test_release_specific_sid_xml_paths_are_kept_for_each_command():
    configured = "$DD4hepINSTALL/DDDetectors/compact/SiD.xml"
    before = _info(
        "2026-08-27", "1", detector="SiD",
        xml_path="/cvmfs/nightly-27/DDDetectors/compact/SiD.xml",
        configured_xml_path=configured,
    )
    after = _info(
        "2026-08-28", "2", detector="SiD",
        xml_path="/cvmfs/nightly-28/DDDetectors/compact/SiD.xml",
        configured_xml_path=configured,
    )
    facts = facts_from(_row(detector="SiD"), before, after)
    assert facts is not None
    assert "xml_path" not in facts.parity_diffs
    body = render_text(facts)
    assert "/cvmfs/nightly-27/DDDetectors/compact/SiD.xml" in body
    assert "/cvmfs/nightly-28/DDDetectors/compact/SiD.xml" in body


def test_missing_fixed_seed_never_claims_the_workloads_were_identical():
    facts = _facts(random_seed=None)
    assert facts is not None
    body = render_text(replace(facts, base_seed=None, parity_diffs=()))
    assert "neither run recorded a fixed random seed" in body
    assert "and the same workload" not in body


def test_render_shell_quotes_untrusted_arguments_into_one_word():
    facts = _facts()
    assert facts is not None
    hostile = "--foo 'two words'\necho injected"
    body = render_text(replace(facts, onset_ddsim_args=hostile))
    # The whole hostile value reaches the shell as one quoted word, so the
    # embedded newline cannot start a command of its own.
    argument = body.rsplit("--ddsim-args=", 1)[1].split("\n)", 1)[0]
    assert shlex.split(argument) == [hostile]


def test_render_contains_two_full_commands_and_only_display_precision_pct():
    facts = _facts()
    assert facts is not None
    body = render_text(facts)
    assert body.count("git clone https://github.com/key4hep/k4Bench") == 1
    assert body.count("k4bench --xml ") == 2
    assert "--sweep-detectors TPC" in body
    assert "Nightly measured:  +36.1%" in body
    assert "0.3614" not in body


def test_one_clone_gives_each_half_its_own_worktree_at_its_own_commit():
    # One clone (a second would only abort with "destination path already
    # exists"), but never one working copy: setup.sh reuses an existing py-venv
    # and plugin/build.sh an up-to-date .so, so a shared checkout would run the
    # AFTER release against the BEFORE release's venv and timing plugin.
    facts = _facts()
    assert facts is not None
    body = render_text(replace(facts, onset_commit="a" * 40))
    lines = body.splitlines()

    assert sum(
        line.startswith("git clone https://github.com/key4hep/k4Bench")
        for line in lines
    ) == 1
    assert "cd k4Bench" in lines
    # Two halves plus the outer fail-fast wrapper.
    assert lines.count("(") == 3 and lines.count(")") == 3
    assert (
        f"git worktree add --detach ../k4Bench-before {facts.base_commit}" in lines
    )
    assert f"git worktree add --detach ../k4Bench-after {'a' * 40}" in lines
    # Detached, so the common case of one k4Bench commit for both nightlies
    # still yields two worktrees rather than a "already checked out" refusal.
    same = render_text(facts).splitlines()
    assert sum(line.startswith("git worktree add --detach") for line in same) == 2


def test_each_half_enters_its_own_worktree_before_setup_without_duplicate_build():
    facts = _facts()
    assert facts is not None
    body = render_text(facts)
    for half, worktree, release in zip(
        _halves(body), ("before", "after"), ("2026-08-27", "2026-08-28")
    ):
        enter = half.index(f"cd ../k4Bench-{worktree}")
        nightly = half.index(
            f"source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r {release}"
        )
        historical_setup = half.index("source setup.sh")
        assert enter < nightly < historical_setup
    assert "KEY4HEP_REPO=" not in body
    assert "bash plugin/build.sh" not in body
    assert "git checkout" not in body


def test_lcg_reproducer_sources_recorded_views_and_guards_rotating_slots():
    before = "/cvmfs/sft-nightlies.cern.ch/lcg/views/devkey-head/Wed/platform/setup.sh"
    after = "/cvmfs/sft-nightlies.cern.ch/lcg/views/devkey-head/Thu/platform/setup.sh"
    facts = facts_from(
        _row(),
        _info("2026-08-27", "1", k4h_stack_setup=before),
        _info("2026-08-28", "2", k4h_stack_setup=after),
    )
    assert facts is not None
    body = render_text(facts)
    assert before in body and after in body
    assert body.count("Recorded LCG view is no longer available") == 2
    assert "/key4hep/setup.sh" not in body


def test_a_shared_input_is_fetched_once_and_a_differing_one_per_half():
    source = "root://example/events.hepmc"
    shared = render_text(
        _facts_with_inputs([source], [source])
    )
    assert shared.count("xrdcp --force") == 1
    # Fetched outside both subshells, since both runs read the same file.
    assert all("xrdcp" not in half for half in _halves(shared))
    # xrdcp is a Key4hep tool, so the shared fetch sources a release of its
    # own before reaching for it — in a subshell, so that release cannot
    # follow it into either half.
    fetch = shared.split("\n(\n")[2].split("\n)\n", 1)[0]
    setup = "source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-08-27"
    assert fetch.index(setup) < fetch.index("xrdcp --force")

    # Differing sources are a workload difference, and each half has to run
    # against the one its own nightly used.
    split = render_text(
        _facts_with_inputs(
            ["root://old.example/before.hepmc"],
            ["root://new.example/after.hepmc"],
        )
    )
    assert split.count("xrdcp --force") == 2
    halves = _halves(split)
    assert "xrdcp --force root://old.example/before.hepmc /tmp/before.hepmc" \
        in halves[0]
    assert "xrdcp --force root://new.example/after.hepmc /tmp/after.hepmc" \
        in halves[1]
    assert "/tmp/after.hepmc" not in halves[0]
    assert "/tmp/before.hepmc" not in halves[1]


def test_the_recipe_names_the_measurement_it_reproduces():
    facts = _facts()
    assert facts is not None
    body = render_text(facts)
    assert body.startswith("#!/usr/bin/env bash\n# k4Bench: reproduce this measurement")
    assert "ILD_FCCee_v01" in body and "no_TPC" in body
    assert "2026-08-27 -> 2026-08-28" in body
    # Read on its own, it still says which runs it came from.
    assert "https://github.test/actions/runs/1" in body
    assert "https://github.test/actions/runs/2" in body


def test_the_recipe_is_pure_ascii():
    # The file is served as plain text with no charset declared, so a browser
    # guessing latin-1 renders every multi-byte character as mojibake — which is
    # what an em dash in a section rule and a prettified sample name produced.
    facts = _facts()
    assert facts is not None
    body = render_text(facts)
    assert body.isascii()
    assert "single_e-_10GeV" in body


def test_the_recipe_runs_as_a_script_and_fails_fast_in_every_stage():
    # "Paste it" and "run it" should both work: a shebang costs a comment line
    # in the paste, and every errexit lives inside a subshell — the outer one
    # covering the stages between the halves — so a failure aborts the whole
    # reproduction without ever touching the reader's own shell.
    facts = _facts()
    assert facts is not None
    body = render_text(facts)
    lines = body.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    # Every command is inside the outer subshell, which opens with errexit.
    opened = lines.index("(")
    assert lines[opened + 1] == "set -e"
    assert all(
        line.startswith("#") or not line.strip()
        for line in lines[:opened]
    )
    for half in _halves(body):
        assert half.splitlines()[0].strip() == "set -e"
        assert half.index("set -e") < half.index("cd ../k4Bench-")


def test_a_failed_before_half_fails_the_whole_recipe(tmp_path, monkeypatch):
    # The half that matters is the one that can be silently skipped: without an
    # outer errexit, a BEFORE that never produced results is followed by an
    # AFTER that succeeds, and `bash recipe.txt` exits 0 with nothing to compare.
    facts = _facts()
    assert facts is not None
    # Point the halves at a release root that cannot exist, so the BEFORE half
    # fails on its own first line rather than on whatever this machine has
    # mounted under /cvmfs.
    absent = tmp_path / "absent-nightlies"
    monkeypatch.setattr(reproduce.shell, "_NIGHTLY_REPO", str(absent))
    recipe = tmp_path / "recipe.txt"
    recipe.write_text(render_text(facts))

    # git is stubbed so the test neither clones nor reaches the network; the
    # BEFORE half then fails for real on the absent CVMFS nightly.
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "git").write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  clone) mkdir -p k4Bench ;;\n"
        '  worktree) mkdir -p "$4" ;;\n'
        "esac\n"
    )
    (stub / "git").chmod(0o755)

    done = subprocess.run(
        ["bash", str(recipe)],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{stub}:/usr/bin:/bin"},
        capture_output=True, text=True,
    )
    assert done.returncode != 0
    # Exactly one half attempted its release: the AFTER half never ran.
    assert done.stderr.count(f"{absent}/key4hep/setup.sh") == 1


def test_recorded_execution_settings_are_replayed_or_stated():
    # A performance measurement is not defined by its ddsim arguments alone:
    # --verbose streams output while the run is being timed, and the nightly
    # pins the benchmark to a CPU set.
    facts = facts_from(
        _row(),
        _info("2026-08-27", "1", verbose=True, runner_cpu_set="2-5"),
        _info("2026-08-28", "2", verbose=True, runner_cpu_set="2-5"),
    )
    assert facts is not None
    assert facts.parity_diffs == ()
    body = render_text(facts)
    assert body.count("--verbose") == 2
    # Stated rather than replayed: the runner's cores are not the reader's, and
    # a taskset over CPUs they do not have would only fail the recipe.
    assert "pinned to CPUs 2-5" in _prose(body)
    assert "taskset -c 2-5" not in body

    # An unpinned, non-verbose nightly says neither.
    quiet = render_text(_facts())
    assert "--verbose" not in quiet and "pinned to CPUs" not in quiet


def test_settings_that_differed_between_the_runs_are_flagged_not_hidden():
    facts = facts_from(
        _row(),
        _info("2026-08-27", "1", verbose=True, runner_cpu_set="2-5"),
        _info("2026-08-28", "2", verbose=False, runner_cpu_set="6-9"),
    )
    assert facts is not None
    assert facts.parity_diffs == ("verbose",)
    body = render_text(facts)
    assert body.count("--verbose") == 1
    assert "different CPU sets (2-5 vs 6-9)" in _prose(body)


def test_artifact_name_is_stable_per_measurement_and_window():
    facts = _facts()
    assert facts is not None
    assert artifact_name(facts) == artifact_name(_facts())
    assert artifact_name(facts).endswith(".txt")
    assert artifact_name(facts).startswith(
        "ILD_FCCee_v01-single_e-_10GeV-no_TPC-mean_time_s-"
    )
    # A different window is a different recipe, and so is a different platform
    # even though the readable stem cannot show it.
    assert artifact_name(replace(facts, onset_release="2026-08-29")) != \
        artifact_name(facts)
    assert artifact_name(replace(facts, platform="aarch64-el9-gcc14-opt")) != \
        artifact_name(facts)
    # Two changes inside one release are two windows, told apart only by their
    # runs — without the run ids in the digest they would publish over one file.
    same_release = replace(facts, base_release="2026-08-28")
    assert artifact_name(replace(same_release, base_run_id="x", onset_run_id="y")) \
        != artifact_name(replace(same_release, base_run_id="w", onset_run_id="z"))


def test_a_same_release_window_writes_its_halves_to_separate_directories():
    # The releases are equal here, so anything named after them collides — and
    # a colliding output directory means AFTER overwrites the results BEFORE is
    # supposed to be compared against.
    facts = _facts()
    assert facts is not None
    body = render_text(replace(facts, base_release="2026-08-28"))
    assert "--output-dir logs/before-run-before" in body
    assert "--output-dir logs/after-run-after" in body
    assert "k4Bench-before/logs/before-run-before" in body
    assert "k4Bench-after/logs/after-run-after" in body


def test_artifact_name_never_leaves_the_published_directory():
    facts = _facts()
    assert facts is not None
    hostile = artifact_name(replace(facts, label="../../etc/passwd", detector="a b"))
    assert "/" not in hostile and " " not in hostile


_INPUT = (
    "https://k4bench-data.web.cern.ch/inputs/reco/CLD/CLD_o2_v09/"
    "p8_ee_Zbb_ecm91_100ev_seed42.edm4hep.root"
)


def _slot(day: str, package: str) -> str:
    """A package's install in one LCG weekday slot, as the nightly resolved it."""
    return (
        f"/cvmfs/sft-nightlies.cern.ch/lcg/nightlies/devkey-head/{day}/{package}"
        "/HEAD/x86_64-el9-gcc16-opt"
    )


def _local_input(source: str) -> str:
    return str(staging.LOCAL_INPUT_DIR / staging.input_key(source) / source.rsplit("/", 1)[-1])


def _k4run_row(**overrides):
    return _row(**{
        "detector": "CLD_o2_v09", "platform": "x86_64-el9-gcc16-opt",
        "sample": "p8_ee_Zbb_ecm91", "label": "baseline",
        "metric": "seconds_per_event", "sub_detector": "ConformalTracking",
        "pct_change": 0.12,
        "last_accepted_run_date": "2026-10-19", "onset_run_date": "2026-10-26",
        **overrides,
    })


def _k4run_info(release: str, run: str, day: str, **overrides):
    """A CLD reco run_info.json as the 2026-10 nightlies write it, in *day*'s slot."""
    geometry = (
        f"{_slot(day, 'k4geo')}/share/k4geo/FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml"
    )
    values = {
        "suite": "reco", "tool": "k4run",
        "detector": "CLD_o2_v09", "platform": "x86_64-el9-gcc16-opt",
        "sample": "p8_ee_Zbb_ecm91", "k4h_release_date": release,
        "k4h_stack_setup": (
            f"/cvmfs/sft-nightlies.cern.ch/lcg/views/devkey-head/{day}"
            "/x86_64-el9-gcc16-opt/setup.sh"
        ),
        "xml_path": "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml",
        "configured_xml_path": "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml",
        "github_run_url": f"https://github.test/actions/runs/{run}",
        "commit_sha": "4c40c67" * 5 + "4c40c",
        "n_events": 100, "ddsim_args": "", "random_seed": None,
        "verbose": False, "runner_cpu_set": "24-31",
        "input_files": [_INPUT], "steering_file": "", "resolved_steering_file": "",
        "options": ["CLDReconstruction.py"],
        "stage_dir": "$CLDCONFIG/share/CLDConfig",
        "resolved_stage_dir": f"{_slot(day, 'cldconfig')}/share/CLDConfig",
        "k4run_args": (
            f"--compactFile {geometry} --inputFiles {_local_input(_INPUT)} "
            "--outputBasename reco --cms 91"
        ),
        "variants": {"truth_tracking": "--truthTracking", "other": "--other"},
        "data_files": [], "memory_events": 100,
    }
    values.update(overrides)
    return values


def _k4run_facts(row=None, **after_overrides):
    return facts_from(
        row or _k4run_row(),
        _k4run_info("2026-10-19", "1", "Sun"),
        _k4run_info("2026-10-26", "2", "Mon", **after_overrides),
    )


def _option(half: str, option: str) -> str:
    """The one shell word *option* is given in *half*'s command."""
    return shlex.split(half.split(f"{option}", 1)[1].split(" \\\n", 1)[0])[0]


def test_a_k4run_job_reruns_its_recorded_command_in_each_half():
    facts = _k4run_facts()
    assert isinstance(facts, K4runReproducerFacts)
    # Each half reads its own weekday slot's geometry and configuration, which
    # is the stack changing, not the workload.
    assert facts.parity_diffs == ()
    body = render_text(facts)
    assert body.count("k4bench k4run CLDReconstruction.py") == 2
    assert "k4bench --xml" not in body
    for half, day in zip(_halves(body), ("Sun", "Mon")):
        info = _k4run_info("", "", day)
        assert _option(half, "--k4run-args=") == info["k4run_args"]
        assert _option(half, "--stage-dir ") == info["resolved_stage_dir"]
        assert f"views/devkey-head/{day}/" in half
    # The baseline alone, and no memory pass: no judged metric comes from it.
    assert "--variant" not in body and "--memory-events" not in body
    assert "# Metric:            seconds_per_event (ConformalTracking)" in body
    assert "same workload: 100 events of the same frozen input" in _prose(body)


def test_a_variant_row_reruns_the_baseline_and_that_variant_alone():
    facts = _k4run_facts(_k4run_row(label="variant_truth_tracking"))
    assert facts is not None and facts.variant == "truth_tracking"
    body = render_text(facts)
    for half in _halves(body):
        assert _option(half, "--variant ") == "truth_tracking=--truthTracking"
    assert "--other" not in body


@pytest.mark.parametrize(
    "row, after",
    [
        # Labels a k4run benchmark cannot produce, or a variant one run lacks.
        ({"label": "no_TPC"}, {}),
        ({"label": "variant_truth_tracking"}, {"variants": {}}),
        # A record without its command, or with a source the recipe cannot fetch.
        ({}, {"options": []}),
        ({}, {"resolved_stage_dir": ""}),
        ({}, {"input_files": ["/data/sim.root"]}),
        # A window across a change of tool has no one command to give.
        ({}, {"tool": "ddsim"}),
    ],
)
def test_k4run_records_without_a_rerunnable_command_get_no_recipe(row, after):
    assert _k4run_facts() is not None
    assert _k4run_facts(_k4run_row(**row), **after) is None


@pytest.mark.parametrize(
    "row, after, diffs",
    [
        ({}, {"n_events": 50}, ("n_events",)),
        ({}, {"stage_dir": "$CLDCONFIG/share/Other"}, ("stage_dir",)),
        (
            {},
            {"k4run_args": _k4run_info("", "", "Mon")["k4run_args"].replace(
                "--cms 91", "--cms 240"
            )},
            ("k4run_args",),
        ),
        (
            {"label": "variant_truth_tracking"},
            {"variants": {"truth_tracking": "--truthTracking --extra"}},
            ("variant_args",),
        ),
        ({}, {"data_files": ["https://a.cern.ch/noise.root"]}, ("data_files",)),
    ],
)
def test_k4run_parity_names_what_was_configured_differently(row, after, diffs):
    facts = _k4run_facts(_k4run_row(**row), **after)
    assert facts is not None and facts.parity_diffs == diffs
    assert "did NOT measure the same workload" in render_text(facts)


def test_a_moved_input_is_an_input_difference_not_an_argument_one():
    moved = _INPUT.replace("k4bench-data.web.cern.ch", "mirror.example")
    args = _k4run_info("", "", "Mon")["k4run_args"].replace(
        _local_input(_INPUT), _local_input(moved)
    )
    facts = _k4run_facts(input_files=[moved], k4run_args=args)
    assert facts is not None and facts.parity_diffs == ("input_files",)


def test_every_k4run_source_is_fetched_once_to_where_its_nightly_read_it():
    data = ("https://a.cern.ch/noise.root", "https://b.cern.ch/Config-abc.tar.gz")
    moved = "root://eosuser.cern.ch//eos/inputs/other.edm4hep.root"
    before = _k4run_info("2026-10-19", "1", "Sun", data_files=list(data))
    after = _k4run_info(
        "2026-10-26", "2", "Mon", data_files=list(data), input_files=[_INPUT, moved],
    )
    facts = facts_from(_k4run_row(), before, after)
    assert facts is not None
    body = render_text(facts)
    # Ahead of both halves, in a subshell sourcing the BEFORE release first.
    fetch = body.split("\n(\n")[2].split("\n)\n", 1)[0]
    assert fetch.index("views/devkey-head/Sun/") < fetch.index("curl ")
    assert all("curl " not in half and "xrdcp " not in half for half in _halves(body))
    # Each input in its own directory, as the recorded arguments read it.
    assert fetch.count(f"-o {_local_input(_INPUT)} {_INPUT}") == 1
    assert f"xrdcp --force {moved} {_local_input(moved)}" in fetch
    # The data files share one directory; an archive is unpacked there.
    folder = staging.LOCAL_DATA_DIR / staging.data_key(list(data))
    # Emptied first, as the nightly does, so nothing left there joins the job.
    assert f"rm -rf {folder}\n  mkdir -p {folder}" in fetch
    assert f"-o {folder}/noise.root https://a.cern.ch/noise.root" in fetch
    assert f"tar -xzf {folder}/Config-abc.tar.gz -C {folder}" in fetch
    assert f"rm -f {folder}/Config-abc.tar.gz" in fetch


def test_k4run_recipe_quotes_recorded_arguments_and_parses_as_bash(tmp_path):
    hostile = "--cms 91 'two words'\necho injected; rm -rf ~"
    facts = _k4run_facts(k4run_args=hostile)
    assert facts is not None
    body = render_text(facts)
    assert _option(_halves(body)[1], "--k4run-args=") == hostile
    recipe = tmp_path / "recipe.txt"
    recipe.write_text(body)
    assert subprocess.run(["bash", "-n", str(recipe)]).returncode == 0


def test_k4run_artifact_name_tells_components_apart():
    facts = _k4run_facts()
    assert facts is not None
    assert artifact_name(facts) == artifact_name(_k4run_facts())
    assert artifact_name(facts) != artifact_name(replace(facts, sub_detector="GeoSvc"))
    # Readably too: a window confirms many components under one metric.
    assert "ConformalTracking" in artifact_name(facts)
    # A measurement of no component keeps the name it always had.
    assert artifact_name(replace(facts, sub_detector=None)).startswith(
        "CLD_o2_v09-p8_ee_Zbb_ecm91-baseline-seconds_per_event-"
    )


def test_a_recipe_says_how_long_its_stacks_stay():
    lcg = render_text(_k4run_facts())
    assert "LCG reuses each weekday's nightly view after about a week" in lcg
    assert "three weeks" not in lcg
    spack = render_text(replace(_k4run_facts(), base_stack_setup="", onset_stack_setup=""))
    assert "Key4hep nightlies are kept on CVMFS for about three weeks" in spack
