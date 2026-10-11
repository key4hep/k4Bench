"""Unit tests for k4bench.benchmark.k4run: the config, the run roster, and the
working directory every run gets. The k4run process itself is mocked out."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from k4bench.benchmark import k4run as k4run_module
from k4bench.benchmark.k4run import K4runConfig, planned_k4run_labels, run_k4run_benchmark
from k4bench.results.model import RunResult


@pytest.fixture
def stage_dir(tmp_path: Path) -> Path:
    """A read-only configuration directory, as CVMFS serves one."""
    stage = tmp_path / "Config"
    (stage / "Tracking").mkdir(parents=True)
    (stage / "Reco.py").write_text("# job options\n")
    (stage / "Tracking" / "settings.xml").write_text("<settings/>\n")
    for path in [*stage.rglob("*"), stage]:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    yield stage
    for path in [stage, *stage.rglob("*")]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _config(tmp_path: Path, stage_dir: Path | None, **kwargs) -> K4runConfig:
    """A config without warm-ups, so each run is one call; TestWarmUp covers them."""
    defaults = dict(
        options=["Reco.py"], n_events=3, log_dir=tmp_path / "logs", stage_dir=stage_dir,
        warm_up=False,
    )
    return K4runConfig(**{**defaults, **kwargs})


class TestConfig:
    def test_needs_an_options_file(self, tmp_path, stage_dir):
        with pytest.raises(ValueError, match="options file"):
            _config(tmp_path, stage_dir, options=[])

    def test_refuses_a_missing_options_file(self, tmp_path, stage_dir):
        with pytest.raises(ValueError, match="Missing.py"):
            _config(tmp_path, stage_dir, options=["Missing.py"])

    def test_refuses_a_missing_stage_dir(self, tmp_path):
        with pytest.raises(ValueError, match="Stage directory"):
            _config(tmp_path, tmp_path / "absent")

    @pytest.mark.parametrize("name", ["", "has space", "a/b"])
    def test_refuses_a_variant_name_unfit_for_a_file_name(self, tmp_path, stage_dir, name):
        with pytest.raises(ValueError, match="variant"):
            _config(tmp_path, stage_dir, variants={name: ["--x"]})

    def test_options_stay_relative_to_the_stage_dir(self, tmp_path, stage_dir):
        assert _config(tmp_path, stage_dir).options == ["Reco.py"]

    def test_without_a_stage_dir_options_are_made_absolute(self, tmp_path, monkeypatch):
        (tmp_path / "job.py").write_text("")
        monkeypatch.chdir(tmp_path)
        config = _config(tmp_path, None, options=["job.py"])
        assert config.options == [str(tmp_path / "job.py")]

    def test_setup_script_is_made_absolute(self, tmp_path, stage_dir, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _config(tmp_path, stage_dir, setup_script=Path("setup.sh"))
        assert config.setup_script == tmp_path / "setup.sh"


def test_planned_labels_are_the_baseline_then_each_variant():
    assert planned_k4run_labels([]) == ["baseline"]
    assert planned_k4run_labels(["truth_tracking", "native"]) == [
        "baseline", "variant_truth_tracking", "variant_native",
    ]


def _recording_run_k4run(calls: list[dict], returncode: int = 0):
    """A stand-in for run_k4run that records what each run saw."""

    def fake(**kwargs):
        work_dir = kwargs["work_dir"]
        calls.append({
            **kwargs,
            "files": sorted(p.relative_to(work_dir).as_posix() for p in work_dir.rglob("*")),
            "writable": all(os.access(p, os.W_OK) for p in [work_dir, *work_dir.rglob("*")]),
        })
        (work_dir / "output.root").write_text("job output")
        (kwargs["log_dir"] / f"{kwargs['label']}.log").write_text("job log")
        return RunResult(label=kwargs["label"], returncode=returncode, n_events=kwargs["n_events"])

    return fake


def test_each_run_starts_in_a_fresh_writable_copy_of_the_stage_dir(tmp_path, stage_dir):
    calls: list[dict] = []
    config = _config(tmp_path, stage_dir, variants={"truth_tracking": ["--truthTracking"]})
    with patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)):
        results = run_k4run_benchmark(config)

    assert [r.label for r in results] == ["baseline", "variant_truth_tracking"]
    for call in calls:
        # The previous run's output.root is not there.
        assert call["files"] == ["Reco.py", "Tracking", "Tracking/settings.xml"]
        assert call["writable"]
        assert not call["work_dir"].exists(), "the working directory is removed after the run"
    assert calls[0]["work_dir"] != calls[1]["work_dir"]


def test_variant_args_are_appended_to_the_job_args(tmp_path, stage_dir):
    calls: list[dict] = []
    config = _config(
        tmp_path, stage_dir,
        extra_args=["--cms", "91"],
        variants={"truth_tracking": ["--truthTracking"], "native": ["--native"]},
    )
    with patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)):
        run_k4run_benchmark(config)

    assert [c["extra_args"] for c in calls] == [
        ["--cms", "91"],
        ["--cms", "91", "--truthTracking"],
        ["--cms", "91", "--native"],
    ]
    assert {c["label"] for c in calls} == {"baseline", "variant_truth_tracking", "variant_native"}


def test_without_a_stage_dir_each_run_starts_empty(tmp_path, monkeypatch):
    (tmp_path / "job.py").write_text("")
    monkeypatch.chdir(tmp_path)
    calls: list[dict] = []
    with patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)):
        run_k4run_benchmark(_config(tmp_path, None, options=["job.py"]))
    assert calls[0]["files"] == []
    assert not (tmp_path / "output.root").exists(), "job output never lands in the caller's directory"


def test_a_failed_run_does_not_stop_the_variants(tmp_path, stage_dir):
    calls: list[dict] = []
    config = _config(tmp_path, stage_dir, variants={"native": ["--native"]})
    with patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls, returncode=1)):
        results = run_k4run_benchmark(config)
    assert [r.returncode for r in results] == [1, 1]
    assert len(calls) == 2


def test_a_run_that_cannot_be_staged_is_recorded_as_failed(tmp_path, stage_dir):
    calls: list[dict] = []
    config = _config(tmp_path, stage_dir, variants={"native": ["--native"]})
    real_stage = k4run_module._stage

    def stage(stage_dir, label):
        if label == "baseline":
            raise OSError("disk full")
        return real_stage(stage_dir, label)

    with (
        patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)),
        patch.object(k4run_module, "_stage", stage),
    ):
        results = run_k4run_benchmark(config)

    assert [(r.label, r.returncode) for r in results] == [("baseline", 1), ("variant_native", 0)]
    assert [c["label"] for c in calls] == ["variant_native"]


class TestWarmUp:
    """Each run is preceded by the same job for one event, unmeasured."""

    def test_each_run_is_preceded_by_a_one_event_run_of_its_own_job(self, tmp_path, stage_dir):
        calls: list[dict] = []
        config = _config(
            tmp_path, stage_dir, warm_up=True,
            extra_args=["--cms", "91"], variants={"truth_tracking": ["--truthTracking"]},
        )
        with patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)):
            results = run_k4run_benchmark(config)

        assert [(c["label"], c["n_events"], c["extra_args"]) for c in calls] == [
            ("baseline", k4run_module.WARM_UP_EVENTS, ["--cms", "91"]),
            ("baseline", 3, ["--cms", "91"]),
            ("variant_truth_tracking", k4run_module.WARM_UP_EVENTS, ["--cms", "91", "--truthTracking"]),
            ("variant_truth_tracking", 3, ["--cms", "91", "--truthTracking"]),
        ]
        assert [(r.label, r.n_events) for r in results] == [("baseline", 3), ("variant_truth_tracking", 3)]

    def test_the_warm_up_leaves_nothing_behind(self, tmp_path, stage_dir):
        calls: list[dict] = []
        config = _config(tmp_path, stage_dir, warm_up=True)
        with patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)):
            run_k4run_benchmark(config)

        warm_up, measured = calls
        # The measured run starts from a clean copy, without the warm-up's output.
        assert measured["files"] == ["Reco.py", "Tracking", "Tracking/settings.xml"]
        assert warm_up["work_dir"] != measured["work_dir"]
        assert not warm_up["work_dir"].exists()
        assert warm_up["log_dir"] != config.log_dir and not warm_up["log_dir"].exists()
        assert sorted(p.name for p in config.log_dir.iterdir()) == ["baseline.log"]

    def test_a_failed_warm_up_still_measures_the_run(self, tmp_path, stage_dir, capsys):
        calls: list[dict] = []
        recording = _recording_run_k4run(calls)

        def warm_up_fails(**kwargs):
            result = recording(**kwargs)
            if kwargs["n_events"] == k4run_module.WARM_UP_EVENTS:
                result.returncode = 1
            return result

        config = _config(tmp_path, stage_dir, warm_up=True)
        with patch.object(k4run_module, "run_k4run", warm_up_fails):
            (result,) = run_k4run_benchmark(config)

        assert len(calls) == 2 and result.succeeded
        assert "Warm-up failed (return code 1)" in capsys.readouterr().out
        # Its log stays with the run's results, marking a measurement that may
        # have found a cold cache.
        assert sorted(p.name for p in config.log_dir.iterdir()) == [
            "baseline.log", "baseline_warm-up.log",
        ]

    def test_a_previous_failed_warm_up_does_not_mark_the_run(self, tmp_path, stage_dir):
        config = _config(tmp_path, stage_dir, warm_up=True)
        config.log_dir.mkdir()
        (config.log_dir / "baseline_warm-up.log").write_text("an earlier benchmark's warm-up")
        with patch.object(k4run_module, "run_k4run", _recording_run_k4run([])):
            run_k4run_benchmark(config)

        assert sorted(p.name for p in config.log_dir.iterdir()) == ["baseline.log"]

    def test_a_warm_up_that_cannot_be_staged_is_skipped(self, tmp_path, stage_dir, capsys):
        calls: list[dict] = []
        real_stage = k4run_module._stage

        def stage(stage_dir, label):
            if label.startswith("warm-up-"):
                raise OSError("disk full")
            return real_stage(stage_dir, label)

        config = _config(tmp_path, stage_dir, warm_up=True)
        with (
            patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)),
            patch.object(k4run_module, "_stage", stage),
        ):
            (result,) = run_k4run_benchmark(config)

        assert [c["n_events"] for c in calls] == [3] and result.succeeded
        assert "Warm-up skipped" in capsys.readouterr().out

    def test_warm_up_is_on_by_default(self, tmp_path, stage_dir):
        config = K4runConfig(options=["Reco.py"], n_events=3, log_dir=tmp_path, stage_dir=stage_dir)
        assert config.warm_up
