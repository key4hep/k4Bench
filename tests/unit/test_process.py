"""Unit tests for k4bench.runner.process.

These run real (trivial) shell commands under GNU time, so the command
building, log streaming and ``time -v`` parsing are exercised end to end.
"""

from __future__ import annotations

import os
import shutil

import pytest

from k4bench.runner.process import run_timed, timed_shell_command

pytestmark = pytest.mark.skipif(
    shutil.which("time") is None, reason="GNU time is not installed"
)


def test_command_times_the_program_with_one_quoted_argument_per_line():
    cmd = timed_shell_command(["prog", "--flag=a b", "plain"])
    assert f"{shutil.which('time')} -v prog \\\n" in cmd
    assert "    '--flag=a b' \\\n" in cmd
    assert cmd.endswith("    plain")


def test_setup_script_is_sourced_before_the_program(tmp_path):
    cmd = timed_shell_command(["prog"], tmp_path / "setup env.sh")
    assert cmd.startswith(f"source '{tmp_path / 'setup env.sh'}'\n")


def test_output_reaches_the_log_and_the_time_report_is_parsed(tmp_path):
    log = tmp_path / "logs" / "run.log"
    proc = run_timed(
        timed_shell_command(["echo", "hello from the run"]),
        env=dict(os.environ),
        log_path=log,
    )
    assert proc.returncode == 0
    assert "hello from the run" in log.read_text()
    assert proc.metrics["wall_time_s"] is not None
    assert proc.metrics["peak_rss_mb"] is not None


def test_program_runs_in_the_given_directory(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    log = tmp_path / "run.log"
    proc = run_timed(timed_shell_command(["pwd"]), env=dict(os.environ), log_path=log, cwd=work)
    assert proc.returncode == 0
    assert log.read_text().splitlines()[0] == str(work)


def test_failing_program_reports_its_exit_code(tmp_path):
    proc = run_timed(
        timed_shell_command(["false"]),
        env=dict(os.environ),
        log_path=tmp_path / "run.log",
    )
    assert proc.returncode != 0


def test_verbose_also_streams_to_stdout(tmp_path, capsys):
    run_timed(
        timed_shell_command(["echo", "streamed"]),
        env=dict(os.environ),
        log_path=tmp_path / "run.log",
        verbose=True,
    )
    assert "streamed" in capsys.readouterr().out


def test_unparsed_time_report_warns_with_the_log_path(tmp_path, capsys):
    log = tmp_path / "baseline.log"
    proc = run_timed("echo no time report", env=dict(os.environ), log_path=log)
    assert proc.metrics["wall_time_s"] is None
    out = capsys.readouterr().out
    assert "WARNING [baseline]" in out
    assert str(log) in out
