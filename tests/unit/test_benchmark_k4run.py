"""Unit tests for k4bench.benchmark.k4run: the config, the run roster, and the
working directory every run gets. The k4run process itself is mocked out."""

from __future__ import annotations

import gzip
import json
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

    @pytest.mark.parametrize("events", [0, -1])
    def test_refuses_a_memory_pass_without_events(self, tmp_path, stage_dir, events):
        with pytest.raises(ValueError, match="memory_events must be a positive number"):
            _config(tmp_path, stage_dir, memory_events=events)

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
        if heap := kwargs.get("heap_json_path"):
            heap.write_text("{}")
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


class TestMemoryPass:
    """With memory_events, each run is followed by its job again, untimed,
    recording the heap each component holds."""

    def test_each_run_is_followed_by_a_memory_pass_of_its_own_job(self, tmp_path, stage_dir):
        calls: list[dict] = []
        config = _config(
            tmp_path, stage_dir, memory_events=2,
            extra_args=["--cms", "91"], variants={"truth_tracking": ["--truthTracking"]},
        )
        with patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)):
            results = run_k4run_benchmark(config)

        assert [(c["label"], c["n_events"], c["extra_args"]) for c in calls] == [
            ("baseline", 3, ["--cms", "91"]),
            ("baseline", 2, ["--cms", "91"]),
            ("variant_truth_tracking", 3, ["--cms", "91", "--truthTracking"]),
            ("variant_truth_tracking", 2, ["--cms", "91", "--truthTracking"]),
        ]
        # Only the memory pass records owners, into its own log directory.
        heaps = [c.get("heap_json_path") for c in calls]
        assert heaps[0] is None and heaps[2] is None
        assert heaps[1] == calls[1]["log_dir"] / "baseline_heap.json"
        assert heaps[3] == calls[3]["log_dir"] / "variant_truth_tracking_heap.json"
        assert [(r.label, r.n_events) for r in results] == [("baseline", 3), ("variant_truth_tracking", 3)]
        assert (config.log_dir / "variant_truth_tracking_heap.json").is_file()

    def test_the_memory_pass_leaves_only_its_heap_file(self, tmp_path, stage_dir):
        calls: list[dict] = []
        config = _config(tmp_path, stage_dir, memory_events=2)
        with patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)):
            run_k4run_benchmark(config)

        measured, memory = calls
        # A fresh copy of the stage directory, without the timed run's output.
        assert memory["files"] == ["Reco.py", "Tracking", "Tracking/settings.xml"]
        assert memory["work_dir"] != measured["work_dir"] and not memory["work_dir"].exists()
        assert memory["log_dir"] != config.log_dir and not memory["log_dir"].exists()
        assert sorted(p.name for p in config.log_dir.iterdir()) == ["baseline.log", "baseline_heap.json"]

    @pytest.mark.parametrize(
        "returncode, writes_heap, warning",
        [(1, True, "memory pass failed (return code 1)"), (0, False, "memory pass wrote no heap file")],
    )
    def test_a_failed_memory_pass_never_fails_the_benchmark(
        self, tmp_path, stage_dir, capsys, returncode, writes_heap, warning
    ):
        recording = _recording_run_k4run([])

        def memory_pass_fails(**kwargs):
            heap = kwargs.get("heap_json_path")
            result = recording(**{**kwargs, "heap_json_path": heap if writes_heap else None})
            if heap is not None:
                result.returncode = returncode
            return result

        config = _config(tmp_path, stage_dir, memory_events=2)
        with patch.object(k4run_module, "run_k4run", memory_pass_fails):
            (result,) = run_k4run_benchmark(config)

        assert result.succeeded
        assert warning in capsys.readouterr().out
        # No heap file, and its log stays with the run's results to say why.
        assert sorted(p.name for p in config.log_dir.iterdir()) == ["baseline.log", "baseline_heap.log"]

    def test_a_failed_run_gets_no_memory_pass(self, tmp_path, stage_dir):
        calls: list[dict] = []
        recording = _recording_run_k4run(calls)

        def baseline_fails(**kwargs):
            result = recording(**kwargs)
            if kwargs["label"] == "baseline":
                result.returncode = 1
            return result

        config = _config(tmp_path, stage_dir, memory_events=2, variants={"native": ["--native"]})
        with patch.object(k4run_module, "run_k4run", baseline_fails):
            results = run_k4run_benchmark(config)

        assert [r.succeeded for r in results] == [False, True]
        assert [(c["label"], c["n_events"]) for c in calls] == [
            ("baseline", 3), ("variant_native", 3), ("variant_native", 2),
        ]
        assert sorted(p.name for p in config.log_dir.iterdir()) == [
            "baseline.log", "variant_native.log", "variant_native_heap.json",
        ]

    def test_a_memory_pass_that_raises_never_fails_the_benchmark(self, tmp_path, stage_dir, capsys):
        recording = _recording_run_k4run([])

        def memory_pass_raises(**kwargs):
            if kwargs.get("heap_json_path") is not None:
                raise OSError("no space left")
            return recording(**kwargs)

        config = _config(tmp_path, stage_dir, memory_events=2, variants={"native": ["--native"]})
        with patch.object(k4run_module, "run_k4run", memory_pass_raises):
            results = run_k4run_benchmark(config)

        assert [r.succeeded for r in results] == [True, True]
        assert "WARNING: no heap file for baseline" in capsys.readouterr().out

    @pytest.mark.parametrize("failing_copy", ["the heap file", "the failure log"])
    def test_a_copy_cut_short_leaves_no_heap_file(self, tmp_path, stage_dir, capsys, failing_copy):
        recording = _recording_run_k4run([])

        def memory_pass(**kwargs):
            result = recording(**kwargs)
            if kwargs.get("heap_json_path") is not None and failing_copy == "the failure log":
                result.returncode = 1
            return result

        real_copy = k4run_module.shutil.copyfile

        def copy_cut_short(source, target, **kwargs):
            if Path(source).name.endswith(("_heap.json", ".log")) and Path(target).parent == config.log_dir:
                Path(target).write_text('{"schema_versi')
                raise OSError("No space left on device")
            return real_copy(source, target, **kwargs)

        config = _config(tmp_path, stage_dir, memory_events=2)
        with (
            patch.object(k4run_module, "run_k4run", memory_pass),
            patch.object(k4run_module.shutil, "copyfile", copy_cut_short),
        ):
            (result,) = run_k4run_benchmark(config)

        assert result.succeeded
        assert "WARNING: no heap file for baseline" in capsys.readouterr().out
        assert not (config.log_dir / "baseline_heap.json").exists()

    def test_a_previous_memory_pass_is_never_credited(self, tmp_path, stage_dir):
        config = _config(tmp_path, stage_dir)
        config.log_dir.mkdir()
        for name in ("baseline_heap.json", "baseline_heap.log"):
            (config.log_dir / name).write_text("an earlier benchmark's memory pass")
        with patch.object(k4run_module, "run_k4run", _recording_run_k4run([])):
            run_k4run_benchmark(config)

        assert sorted(p.name for p in config.log_dir.iterdir()) == ["baseline.log"]

    def test_memory_pass_is_off_by_default(self, tmp_path, stage_dir):
        calls: list[dict] = []
        config = K4runConfig(options=["Reco.py"], n_events=3, log_dir=tmp_path / "logs", stage_dir=stage_dir)
        assert config.memory_events is None
        with patch.object(k4run_module, "run_k4run", _recording_run_k4run(calls)):
            run_k4run_benchmark(_config(tmp_path, stage_dir))
        assert [c.get("heap_json_path") for c in calls] == [None]


class TestDataFlow:
    """After the last run, each run's data flow is built from its auditor file
    and job options, with Marlin asked once for its processors."""

    MARLIN_XML = (
        '<?xml version="1.0"?>\n<marlin><processor name="P" type="Digi">'
        '<parameter name="Hits" type="string" lcioOutType="TrackerHit">Hits</parameter>'
        "</processor></marlin>\n"
    )

    @staticmethod
    def _auditing_run_k4run(wrapped: bool = True):
        """A run_k4run stand-in whose job leaves an auditor file and job options."""

        def fake(**kwargs):
            log_dir, label = kwargs["log_dir"], kwargs["label"]
            component = {"name": "Digi", "category": "algorithm", "parent": None,
                         "type": "MarlinProcessorWrapper" if wrapped else "Digi",
                         "impl": "Digi", "inputs": [], "outputs": []}
            (log_dir / f"{label}_components.json").write_text(json.dumps({"components": [component]}))
            (log_dir / f"{label}_joboptions.opts").write_text("Digi.ProcessorType = 'Digi';\n")
            return RunResult(label=label, returncode=0, n_events=kwargs["n_events"])

        return fake

    def test_each_run_gets_its_data_flow_and_marlin_is_asked_once(self, tmp_path, stage_dir):
        config = _config(tmp_path, stage_dir, variants={"x": ["--x"]})
        dumps = []
        with patch.object(k4run_module, "run_k4run", self._auditing_run_k4run()), \
             patch.object(k4run_module, "marlin_processor_dump",
                          lambda setup: dumps.append(setup) or self.MARLIN_XML):
            run_k4run_benchmark(config)
        assert len(dumps) == 1
        assert gzip.decompress((config.log_dir / "marlin_processors.xml.gz").read_bytes()).decode() \
            == self.MARLIN_XML
        for label in ("baseline", "variant_x"):
            flow = json.loads((config.log_dir / f"{label}_dataflow.json").read_text())
            assert flow["marlin_recorded"]
            assert flow["algorithms"][0]["outputs"][0]["name"] == "Hits"

    def test_a_job_without_marlin_processors_never_runs_marlin(self, tmp_path, stage_dir):
        config = _config(tmp_path, stage_dir)
        with patch.object(k4run_module, "run_k4run", self._auditing_run_k4run(wrapped=False)), \
             patch.object(k4run_module, "marlin_processor_dump", pytest.fail):
            run_k4run_benchmark(config)
        assert (config.log_dir / "baseline_dataflow.json").exists()
        assert not (config.log_dir / "marlin_processors.xml.gz").exists()

    def test_without_marlin_the_processors_are_left_unknown(self, tmp_path, stage_dir, capsys):
        config = _config(tmp_path, stage_dir)
        with patch.object(k4run_module, "run_k4run", self._auditing_run_k4run()), \
             patch.object(k4run_module, "marlin_processor_dump", lambda setup: None):
            run_k4run_benchmark(config)
        flow = json.loads((config.log_dir / "baseline_dataflow.json").read_text())
        assert not flow["marlin_recorded"] and flow["algorithms"][0]["inputs"] is None
        assert "Marlin -x failed" in capsys.readouterr().out

    def test_a_failure_recording_it_never_fails_the_benchmark(self, tmp_path, stage_dir, capsys):
        config = _config(tmp_path, stage_dir)

        def broken(setup):
            raise RuntimeError("disk full")

        with patch.object(k4run_module, "run_k4run", self._auditing_run_k4run()), \
             patch.object(k4run_module, "marlin_processor_dump", broken):
            results = run_k4run_benchmark(config)
        assert [r.label for r in results] == ["baseline"]
        assert "no data flow recorded" in capsys.readouterr().out

    def test_a_run_without_auditor_files_gets_none_and_old_ones_go(self, tmp_path, stage_dir):
        config = _config(tmp_path, stage_dir)
        config.log_dir.mkdir()
        (config.log_dir / "baseline_dataflow.json").write_text("{}")
        (config.log_dir / "marlin_processors.xml.gz").write_bytes(b"")
        with patch.object(k4run_module, "run_k4run", _recording_run_k4run([])):
            run_k4run_benchmark(config)
        assert not (config.log_dir / "baseline_dataflow.json").exists()
        assert not (config.log_dir / "marlin_processors.xml.gz").exists()
