"""Unit test for the per-run timeout watchdog in dd4bench.runner.executor.

The built ddsim command is patched to a plain ``sleep`` so no ddsim / GNU time
is needed — only the watchdog wiring is exercised.
"""

from __future__ import annotations

from unittest.mock import patch

from dd4bench.runner.executor import run_ddsim


def test_timeout_kills_run_and_marks_failed(tmp_path):
    with patch("dd4bench.runner.executor._build_command", return_value="sleep 30"), \
         patch("dd4bench.runner.executor.setup_plugin_environment", return_value=False):
        result = run_ddsim(
            xml_path=tmp_path / "geo.xml",
            label="slow_run",
            n_events=1,
            output_file=tmp_path / "out.root",
            log_dir=tmp_path / "logs",
            timeout_s=0.5,
        )

    # Killed by signal -> non-zero (negative) return code -> recorded as failed.
    assert result.returncode != 0
    assert not result.succeeded
    # The timeout is annotated in the log for downstream viewers.
    log_text = (tmp_path / "logs" / "slow_run.log").read_text()
    assert "TIMEOUT" in log_text


def test_no_timeout_lets_quick_run_complete(tmp_path):
    with patch("dd4bench.runner.executor._build_command", return_value="true"), \
         patch("dd4bench.runner.executor.setup_plugin_environment", return_value=False):
        result = run_ddsim(
            xml_path=tmp_path / "geo.xml",
            label="quick_run",
            n_events=1,
            output_file=tmp_path / "out.root",
            log_dir=tmp_path / "logs",
            timeout_s=30,
        )

    assert result.returncode == 0
    assert result.succeeded
    assert "TIMEOUT" not in (tmp_path / "logs" / "quick_run.log").read_text()
