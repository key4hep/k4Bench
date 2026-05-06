"""Unit tests for dd4bench.runner.executor.

_build_command is a pure function (no subprocess, no filesystem), so we
test it directly.  run_ddsim itself requires a live ddsim binary and
belongs in the integration suite.
"""

from __future__ import annotations

from pathlib import Path

from dd4bench.runner.executor import _build_command

XML = Path("/geo/ALLEGRO_o1_v03.xml")
OUTPUT = Path("/tmp/out.edm4hep.root")
SETUP = Path("/opt/k4geo/bin/thisk4geo.sh")


class TestBuildCommandManagedArgs:
    """The three executor-owned flags are always present and correctly set."""

    def _cmd(self, **kwargs) -> str:
        defaults = dict(
            xml_path=XML, n_events=5, output_file=OUTPUT,
            setup_script=None, extra_args=[],
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
            setup_script=None, extra_args=[],
        )
        assert "source" not in cmd

    def test_setup_script_sourced_before_ddsim(self):
        cmd = _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=SETUP, extra_args=[],
        )
        assert f"source {SETUP}" in cmd
        # source line must precede the ddsim invocation
        assert cmd.index("source") < cmd.index("ddsim")


class TestBuildCommandExtraArgs:
    """Caller-supplied extra_args are passed through and shell-quoted."""

    def _cmd(self, extra_args: list[str]) -> str:
        return _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=None, extra_args=extra_args,
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
            setup_script=None, extra_args=[],
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
