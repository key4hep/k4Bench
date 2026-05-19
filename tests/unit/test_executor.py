"""Unit tests for dd4bench.runner.executor.

_build_command is a pure function (no subprocess, no filesystem), so we
test it directly.  run_ddsim itself requires a live ddsim binary and
belongs in the integration suite, except for the subprocess-mocked
regression tests in TestVerboseReturncode.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from dd4bench.runner.executor import _build_command, run_ddsim

XML = Path("/geo/ALLEGRO_o1_v03.xml")
OUTPUT = Path("/tmp/out.edm4hep.root")
SETUP = Path("/opt/k4geo/bin/thisk4geo.sh")


class TestBuildCommandManagedArgs:
    """The three executor-owned flags are always present and correctly set."""

    def _cmd(self, **kwargs) -> str:
        defaults = dict(
            xml_path=XML, n_events=5, output_file=OUTPUT,
            setup_script=None, extra_args=[], plugin_available=False,
        )
        return _build_command(**{**defaults, **kwargs})

    def test_compact_file_present(self):
        assert f"--compactFile={XML}" in self._cmd()

    def test_number_of_events_present(self):
        assert "--numberOfEvents=5" in self._cmd()

    def test_output_file_present(self):
        assert f"--outputFile={OUTPUT}" in self._cmd()

    def test_time_wrapper_present(self):
        assert "/usr/bin/time -v ddsim" in self._cmd()

    def test_n_events_reflects_argument(self):
        assert "--numberOfEvents=99" in self._cmd(n_events=99)


class TestBuildCommandSetupScript:
    """setup_script is optional; when supplied it is sourced first."""

    def test_no_setup_script_omits_source_line(self):
        cmd = _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=None, extra_args=[], plugin_available=False,
        )
        assert "source" not in cmd

    def test_setup_script_sourced_before_ddsim(self):
        cmd = _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=SETUP, extra_args=[], plugin_available=False,
        )
        assert f"source {SETUP}" in cmd
        # source line must precede the ddsim invocation
        assert cmd.index("source") < cmd.index("ddsim")


class TestBuildCommandExtraArgs:
    """Caller-supplied extra_args are passed through and shell-quoted."""

    def _cmd(self, extra_args: list[str]) -> str:
        return _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=None, extra_args=extra_args, plugin_available=False,
        )

    def test_single_flag_present(self):
        cmd = self._cmd(["--runType=batch"])
        assert "--runType=batch" in cmd

    def test_multiple_flags_all_present(self):
        cmd = self._cmd(["--runType=batch", "--enableGun"])
        assert "--runType=batch" in cmd
        assert "--enableGun" in cmd

    def test_key_value_pair_present(self):
        cmd = self._cmd(["--gun.particle", "e-"])
        assert "--gun.particle" in cmd
        assert "e-" in cmd

    def test_value_with_spaces_is_quoted(self):
        # shlex.quote wraps values containing spaces in single quotes.
        cmd = self._cmd(["--someArg", "value with spaces"])
        assert "'value with spaces'" in cmd

    def test_empty_extra_args_does_not_add_noise(self):
        cmd_empty = self._cmd([])
        cmd_none = _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=None, extra_args=[], plugin_available=False,
        )
        assert cmd_empty == cmd_none

    def test_extra_args_come_after_managed_args(self):
        cmd = self._cmd(["--enableGun"])
        managed_end = max(
            cmd.index("--compactFile"),
            cmd.index("--numberOfEvents"),
            cmd.index("--outputFile"),
        )
        assert cmd.index("--enableGun") > managed_end


class TestBuildCommandPluginAvailability:
    """plugin_available controls whether the timing action is injected."""

    def _cmd(self, *, plugin_available: bool, extra_args: list[str] | None = None) -> str:
        return _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=None, extra_args=extra_args or [],
            plugin_available=plugin_available,
        )

    def test_timing_action_injected_when_plugin_available(self):
        cmd = self._cmd(plugin_available=True)
        assert "DD4benchTimingAction" in cmd
        assert "--action.event" in cmd

    def test_timing_action_absent_when_plugin_unavailable(self):
        cmd = self._cmd(plugin_available=False)
        assert "DD4benchTimingAction" not in cmd

    def test_timing_action_not_duplicated_when_already_in_extra_args(self):
        cmd = self._cmd(
            plugin_available=True,
            extra_args=["--action.event", "DD4benchTimingAction"],
        )
        assert cmd.count("DD4benchTimingAction") == 1

    def test_region_actions_injected_when_plugin_available(self):
        cmd = self._cmd(plugin_available=True)
        assert "--action.stepping" in cmd
        assert "DD4benchRegionTimingAction" in cmd
        assert "--action.tracking" in cmd
        assert "DD4benchRegionTrackingAction" in cmd
        assert "DD4benchRegionEventAction" in cmd

    def test_region_actions_absent_when_plugin_unavailable(self):
        cmd = self._cmd(plugin_available=False)
        assert "DD4benchRegion" not in cmd

    def test_region_actions_not_duplicated_when_already_in_extra_args(self):
        cmd = self._cmd(
            plugin_available=True,
            extra_args=["--action.stepping", "DD4benchRegionTimingAction"],
        )
        assert cmd.count("DD4benchRegionTimingAction") == 1


class TestVerboseReturncode:
    """Regression: returncode must not be None after either streaming path."""

    def _make_proc(self):
        """Simulate a Popen where returncode is None until wait() is called."""
        mock_proc = MagicMock()
        mock_proc.returncode = None  # real Popen starts here
        mock_proc.stdout = iter([])  # no output lines

        def set_returncode():
            mock_proc.returncode = 0

        mock_proc.wait.side_effect = set_returncode
        return mock_proc

    def _run(self, tmp_path: Path, verbose: bool):
        mock_proc = self._make_proc()
        with patch("dd4bench.runner.executor.subprocess.Popen", return_value=mock_proc):
            return run_ddsim(
                xml_path=XML,
                label="test",
                n_events=2,
                output_file=tmp_path / "out.root",
                log_dir=tmp_path / "logs",
                verbose=verbose,
            )

    def test_verbose_returncode_is_not_none(self, tmp_path):
        """Verbose path must call proc.wait() so returncode is populated."""
        result = self._run(tmp_path, verbose=True)
        assert result.returncode is not None

    def test_nonverbose_returncode_is_not_none(self, tmp_path):
        """Non-verbose path (communicate) must also populate returncode."""
        result = self._run(tmp_path, verbose=False)
        assert result.returncode is not None