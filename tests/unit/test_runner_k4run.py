"""Unit tests for k4bench.runner.k4run.

_k4run_args is a pure function, tested directly. run_k4run is run against a
mocked process: what it hands k4run, where the job runs, and what it reads
back afterwards.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from k4bench.plugin.runtime import ALLOC_COUNTER_PRELOAD
from k4bench.runner.k4run import _k4run_args, run_k4run

AUDITOR = Path("/k4bench/plugin/auditor/k4BenchAuditorOptions.py")


class TestK4runArgs:
    def _args(self, **kwargs) -> list[str]:
        defaults = dict(
            options=["Reco.py"], n_events=5, extra_args=[], auditor_options=AUDITOR,
        )
        return _k4run_args(**{**defaults, **kwargs})

    def test_auditor_options_follow_the_job_options(self):
        args = self._args(options=["a.py", "b.py"])
        assert args[:3] == ["a.py", "b.py", str(AUDITOR)]

    def test_no_auditor_options_without_the_auditor(self):
        assert str(AUDITOR) not in self._args(auditor_options=None)

    def test_event_count_is_managed(self):
        assert "--num-events=7" in self._args(n_events=7)

    def test_extra_args_come_last(self):
        # A trailing options file would be read as one more --inputFiles value.
        args = self._args(extra_args=["--inputFiles", "a.root", "b.root"])
        assert args[-3:] == ["--inputFiles", "a.root", "b.root"]
        assert args.index(str(AUDITOR)) < args.index("--inputFiles")


def _finished_process(on_wait=lambda: None):
    proc = MagicMock(stdout=iter([]), returncode=None)

    def wait():
        on_wait()
        proc.returncode = 0

    proc.wait.side_effect = wait
    return proc


def _run(tmp_path: Path, proc, *, auditor: bool = True, setup_env=None, **kwargs):
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    defaults = dict(
        options=["Reco.py"], label="baseline", n_events=4,
        work_dir=work_dir, log_dir=tmp_path / "logs",
    )
    with (
        patch("k4bench.runner.process.subprocess.Popen", return_value=proc) as popen,
        patch(
            "k4bench.runner.k4run.setup_auditor_environment",
            setup_env or MagicMock(return_value=auditor),
        ),
        patch("k4bench.runner.k4run.auditor_options_file", return_value=AUDITOR),
    ):
        result = run_k4run(**{**defaults, **kwargs})
    return result, popen


def test_job_runs_in_its_working_directory(tmp_path):
    _, popen = _run(tmp_path, _finished_process())
    assert popen.call_args.kwargs["cwd"] == tmp_path / "work"


def test_command_is_timed_k4run_with_the_auditor(tmp_path):
    _, popen = _run(tmp_path, _finished_process(), extra_args=["--cms", "91"])
    command = popen.call_args.args[0]
    assert "/usr/bin/time -v k4run" in command
    tokens = shlex.split(command.replace("\\\n", " "))
    assert tokens[tokens.index("k4run") + 1:] == [
        "Reco.py", str(AUDITOR), "--num-events=4", "--cms", "91",
    ]


def test_allocation_counter_is_preloaded_after_the_setup_script(tmp_path):
    # The script may set LD_PRELOAD itself, so the counter is checked after it.
    setup = tmp_path / "setup.sh"
    setup_env = MagicMock(return_value=True)
    _, popen = _run(tmp_path, _finished_process(), setup_env=setup_env, setup_script=setup)
    assert setup_env.call_args.kwargs["setup_script"] == setup
    command = popen.call_args.args[0]
    assert command.index(f"source {setup}") < command.index(ALLOC_COUNTER_PRELOAD)
    assert command.index(ALLOC_COUNTER_PRELOAD) < command.index(" -v k4run")


@pytest.mark.parametrize("auditor, setup_script", [(True, None), (False, Path("setup.sh"))])
def test_no_preload_in_the_command_without_a_setup_script_or_auditor(
    tmp_path, auditor, setup_script
):
    _, popen = _run(tmp_path, _finished_process(), auditor=auditor, setup_script=setup_script)
    assert ALLOC_COUNTER_PRELOAD not in popen.call_args.args[0]


def test_without_the_auditor_the_job_runs_unaudited(tmp_path):
    _, popen = _run(tmp_path, _finished_process(), auditor=False)
    assert str(AUDITOR) not in popen.call_args.args[0]


def test_previous_auditor_output_is_never_credited(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    stale = [
        log_dir / "baseline_events.json",
        log_dir / "baseline_components.json",
        log_dir / "baseline_joboptions.opts",
    ]
    for path in stale:
        path.write_text('{"peak_vmem_mb": 9999}')

    def finish():
        assert not any(path.exists() for path in stale)

    result, _ = _run(tmp_path, _finished_process(finish))
    assert result.peak_vmem_mb is None


def test_peak_vmem_is_read_from_the_auditor_event_file(tmp_path):
    def finish():
        (tmp_path / "logs" / "baseline_events.json").write_text(
            '{"schema_version": 1, "peak_vmem_mb": 3072.5}'
        )

    result, _ = _run(tmp_path, _finished_process(finish))
    assert result.peak_vmem_mb == 3072.5
    assert result.returncode == 0
    assert result.n_events == 4


def test_missing_auditor_output_is_warned_about(tmp_path, capsys):
    _run(tmp_path, _finished_process())
    assert "auditor output is missing or unusable" in capsys.readouterr().out


def test_output_size_counts_only_files_the_job_created(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "staged.xml").write_bytes(b"x" * 2 * 1024**2)

    def finish():
        (work_dir / "out").mkdir()
        (work_dir / "out" / "reco_REC.edm4hep.root").write_bytes(b"x" * 3 * 1024**2)
        (work_dir / "reco_aida.root").write_bytes(b"x" * 1024**2)

    result, _ = _run(tmp_path, _finished_process(finish))
    assert result.output_size_mb == pytest.approx(4.0)
