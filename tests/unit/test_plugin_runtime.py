"""Unit tests for dd4bench.plugin.runtime.

All tests are pure Python — no subprocess is executed and no filesystem
paths outside tmp_path are touched. Subprocess calls and the
plugin-search helpers are patched so the suite runs without a built plugin.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import dd4bench.plugin.runtime as plugin_runtime
from dd4bench.plugin.runtime import (
    _find_plugin_root,
    ensure_plugin_built,
    find_plugin_lib_dir,
    setup_plugin_environment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin_libs(base: Path, *parts: str, which: tuple[str, ...] = ("event", "region")) -> Path:
    """Create plugin .so files at <base>/<parts>/ and return the dir.

    By default creates both plugins, matching the "complete build" state
    that find_plugin_lib_dir requires. Pass `which=("event",)` or
    `which=("region",)` to simulate a partial build.
    """
    lib_dir = base.joinpath(*parts)
    lib_dir.mkdir(parents=True, exist_ok=True)
    if "event" in which:
        (lib_dir / "libDD4benchTimingAction.so").touch()
        (lib_dir / "libDD4benchTimingAction.components").touch()
    if "region" in which:
        (lib_dir / "libDD4benchRegionTimingAction.so").touch()
        (lib_dir / "libDD4benchRegionTimingAction.components").touch()
    return lib_dir


def _mock_run(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# _find_plugin_root  (unchanged behavior)
# ---------------------------------------------------------------------------


class TestFindPluginRoot:
    """_find_plugin_root locates the plugin source directory."""

    def test_found_in_cwd(self, tmp_path):
        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            assert _find_plugin_root() == plugin_dir

    def test_found_via_package_root_parent(self, tmp_path):
        repo_dir = tmp_path / "repo"
        plugin_dir = repo_dir / "plugin"
        plugin_dir.mkdir(parents=True)
        cwd_dir = tmp_path / "other"
        cwd_dir.mkdir()
        with (
            patch("pathlib.Path.cwd", return_value=cwd_dir),
            patch.object(plugin_runtime, "_PACKAGE_ROOT", repo_dir / "dd4bench"),
        ):
            assert _find_plugin_root() == plugin_dir

    def test_cwd_candidate_takes_precedence(self, tmp_path):
        cwd_plugin = tmp_path / "cwd" / "plugin"
        cwd_plugin.mkdir(parents=True)
        pkg_plugin = tmp_path / "pkg" / "plugin"
        pkg_plugin.mkdir(parents=True)
        with (
            patch("pathlib.Path.cwd", return_value=tmp_path / "cwd"),
            patch.object(plugin_runtime, "_PACKAGE_ROOT", tmp_path / "pkg" / "dd4bench"),
        ):
            assert _find_plugin_root() == cwd_plugin

    def test_raises_when_neither_candidate_exists(self, tmp_path):
        cwd_dir = tmp_path / "empty"
        cwd_dir.mkdir()
        with (
            patch("pathlib.Path.cwd", return_value=cwd_dir),
            patch.object(plugin_runtime, "_PACKAGE_ROOT", tmp_path / "pkg" / "dd4bench"),
        ):
            with pytest.raises(FileNotFoundError, match="plugin directory"):
                _find_plugin_root()


# ---------------------------------------------------------------------------
# find_plugin_lib_dir
# ---------------------------------------------------------------------------


class TestFindPluginLibDir:
    """find_plugin_lib_dir returns the directory that holds BOTH .so files."""

    def test_found_in_install_lib(self, tmp_path):
        lib_dir = _make_plugin_libs(tmp_path, "install", "lib")
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            assert find_plugin_lib_dir() == lib_dir

    def test_found_in_install_lib64(self, tmp_path):
        lib_dir = _make_plugin_libs(tmp_path, "install", "lib64")
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            assert find_plugin_lib_dir() == lib_dir

    def test_empty_lib_falls_through_to_lib64(self, tmp_path):
        (tmp_path / "install" / "lib").mkdir(parents=True)
        lib64_dir = _make_plugin_libs(tmp_path, "install", "lib64")
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            assert find_plugin_lib_dir() == lib64_dir

    def test_install_lib_takes_precedence_over_lib64(self, tmp_path):
        lib_dir = _make_plugin_libs(tmp_path, "install", "lib")
        _make_plugin_libs(tmp_path, "install", "lib64")
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            assert find_plugin_lib_dir() == lib_dir

    def test_falls_back_to_build_dir(self, tmp_path):
        build_dir = _make_plugin_libs(tmp_path, "build")
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            assert find_plugin_lib_dir() == build_dir

    def test_install_takes_precedence_over_build(self, tmp_path):
        lib_dir = _make_plugin_libs(tmp_path, "install", "lib")
        _make_plugin_libs(tmp_path, "build")
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            assert find_plugin_lib_dir() == lib_dir

    def test_raises_when_no_libraries_found(self, tmp_path):
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="plugin libraries"):
                find_plugin_lib_dir()

    def test_accepts_versioned_so_suffix(self, tmp_path):
        lib_dir = tmp_path / "install" / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "libDD4benchTimingAction.so.1").touch()
        (lib_dir / "libDD4benchTimingAction.components").touch()
        (lib_dir / "libDD4benchRegionTimingAction.so.1").touch()
        (lib_dir / "libDD4benchRegionTimingAction.components").touch()
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            assert find_plugin_lib_dir() == lib_dir

    # --- new tests for the "both libraries required" behavior --------------

    def test_partial_build_event_only_is_rejected(self, tmp_path):
        """Event plugin alone in install/lib doesn't count as a complete build."""
        _make_plugin_libs(tmp_path, "install", "lib", which=("event",))
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="plugin libraries"):
                find_plugin_lib_dir()

    def test_partial_build_region_only_is_rejected(self, tmp_path):
        """Region plugin alone in install/lib doesn't count as a complete build."""
        _make_plugin_libs(tmp_path, "install", "lib", which=("region",))
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="plugin libraries"):
                find_plugin_lib_dir()

    def test_split_libraries_across_dirs_is_rejected(self, tmp_path):
        """Event in install/lib and region in build/ should NOT match either."""
        _make_plugin_libs(tmp_path, "install", "lib", which=("event",))
        _make_plugin_libs(tmp_path, "build", which=("region",))
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="plugin libraries"):
                find_plugin_lib_dir()

    def test_so_without_components_is_rejected(self, tmp_path):
        """.so files present but no .components → DDG4 can't resolve bundled factories."""
        lib_dir = tmp_path / "install" / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "libDD4benchTimingAction.so").touch()
        (lib_dir / "libDD4benchRegionTimingAction.so").touch()
        # No .components files — simulates the lib vs lib64 split seen on
        # GNUInstallDirs systems where cmake installs .so into both but
        # .components only into lib64.
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="plugin libraries"):
                find_plugin_lib_dir()

    def test_falls_through_to_lib64_when_lib_lacks_components(self, tmp_path):
        """lib has .so only; lib64 has both .so and .components — should return lib64."""
        lib_dir = tmp_path / "install" / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "libDD4benchTimingAction.so").touch()
        (lib_dir / "libDD4benchRegionTimingAction.so").touch()
        lib64_dir = _make_plugin_libs(tmp_path, "install", "lib64")
        with patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path):
            assert find_plugin_lib_dir() == lib64_dir


# ---------------------------------------------------------------------------
# ensure_plugin_built  (unchanged behavior, message text updated)
# ---------------------------------------------------------------------------


class TestEnsurePluginBuilt:
    """ensure_plugin_built builds the plugins only when libraries are absent."""

    def test_already_built_skips_subprocess(self, tmp_path):
        with (
            patch.object(plugin_runtime, "find_plugin_lib_dir", return_value=tmp_path),
            patch("dd4bench.plugin.runtime.subprocess.run") as mock_run,
        ):
            ensure_plugin_built()
            mock_run.assert_not_called()

    def test_missing_build_script_raises(self, tmp_path):
        with (
            patch.object(plugin_runtime, "find_plugin_lib_dir", side_effect=FileNotFoundError),
            patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path),
        ):
            with pytest.raises(FileNotFoundError, match="build script"):
                ensure_plugin_built()

    def test_failed_build_raises_runtime_error(self, tmp_path):
        (tmp_path / "build.sh").touch()
        with (
            patch.object(plugin_runtime, "find_plugin_lib_dir", side_effect=FileNotFoundError),
            patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path),
            patch(
                "dd4bench.plugin.runtime.subprocess.run",
                return_value=_mock_run(returncode=1, stderr="cmake error"),
            ),
        ):
            with pytest.raises(RuntimeError, match="Failed to build"):
                ensure_plugin_built()

    def test_successful_build_invokes_build_sh(self, tmp_path):
        build_sh = tmp_path / "build.sh"
        build_sh.touch()
        with (
            patch.object(plugin_runtime, "find_plugin_lib_dir", side_effect=FileNotFoundError),
            patch.object(plugin_runtime, "_find_plugin_root", return_value=tmp_path),
            patch(
                "dd4bench.plugin.runtime.subprocess.run",
                return_value=_mock_run(returncode=0),
            ) as mock_run,
        ):
            ensure_plugin_built()
            mock_run.assert_called_once_with(
                ["bash", str(build_sh)],
                capture_output=True,
                text=True,
            )


# ---------------------------------------------------------------------------
# setup_plugin_environment
# ---------------------------------------------------------------------------


class TestSetupPluginEnvironment:
    """setup_plugin_environment configures env vars for the timing plugins."""

    def test_returns_true_when_plugin_available(self, tmp_path):
        env: dict[str, str] = {}
        with (
            patch.object(plugin_runtime, "ensure_plugin_built"),
            patch.object(plugin_runtime, "find_plugin_lib_dir", return_value=tmp_path / "lib"),
        ):
            result = setup_plugin_environment(env=env, event_json_path=tmp_path / "ev.json")
        assert result is True

    def test_sets_ld_library_path_when_absent(self, tmp_path):
        lib_dir = tmp_path / "lib"
        env: dict[str, str] = {}
        with (
            patch.object(plugin_runtime, "ensure_plugin_built"),
            patch.object(plugin_runtime, "find_plugin_lib_dir", return_value=lib_dir),
        ):
            setup_plugin_environment(env=env, event_json_path=tmp_path / "ev.json")
        assert env["LD_LIBRARY_PATH"] == str(lib_dir)

    def test_prepends_to_existing_ld_library_path(self, tmp_path):
        lib_dir = tmp_path / "lib"
        env: dict[str, str] = {"LD_LIBRARY_PATH": "/existing/lib"}
        with (
            patch.object(plugin_runtime, "ensure_plugin_built"),
            patch.object(plugin_runtime, "find_plugin_lib_dir", return_value=lib_dir),
        ):
            setup_plugin_environment(env=env, event_json_path=tmp_path / "ev.json")
        assert env["LD_LIBRARY_PATH"] == f"{lib_dir}:/existing/lib"

    def test_sets_event_json_env_var_to_resolved_path(self, tmp_path):
        event_json = tmp_path / "subdir" / "events.json"
        env: dict[str, str] = {}
        with (
            patch.object(plugin_runtime, "ensure_plugin_built"),
            patch.object(plugin_runtime, "find_plugin_lib_dir", return_value=tmp_path / "lib"),
        ):
            setup_plugin_environment(env=env, event_json_path=event_json)
        assert env["DD4BENCH_EVENT_JSON"] == str(event_json.resolve())

    def test_returns_false_on_file_not_found(self, tmp_path):
        env: dict[str, str] = {}
        with patch.object(
            plugin_runtime, "ensure_plugin_built", side_effect=FileNotFoundError("no plugin")
        ):
            result = setup_plugin_environment(env=env, event_json_path=tmp_path / "ev.json")
        assert result is False

    def test_returns_false_on_runtime_error(self, tmp_path):
        env: dict[str, str] = {}
        with patch.object(
            plugin_runtime, "ensure_plugin_built", side_effect=RuntimeError("build failed")
        ):
            result = setup_plugin_environment(env=env, event_json_path=tmp_path / "ev.json")
        assert result is False

    def test_env_not_modified_on_failure(self, tmp_path):
        env: dict[str, str] = {"EXISTING": "value"}
        with patch.object(
            plugin_runtime, "ensure_plugin_built", side_effect=FileNotFoundError
        ):
            setup_plugin_environment(env=env, event_json_path=tmp_path / "ev.json")
        assert "LD_LIBRARY_PATH" not in env
        assert "DD4BENCH_EVENT_JSON" not in env
        assert env == {"EXISTING": "value"}

    # --- new tests for the region_json_path parameter ----------------------

    def test_stale_region_env_var_cleared_when_path_is_none(self, tmp_path):
        env: dict[str, str] = {"DD4BENCH_REGION_JSON": "/old/path/regions.json"}
        with (
            patch.object(plugin_runtime, "ensure_plugin_built"),
            patch.object(plugin_runtime, "find_plugin_lib_dir", return_value=tmp_path / "lib"),
        ):
            setup_plugin_environment(env=env, event_json_path=tmp_path / "ev.json")
        assert "DD4BENCH_REGION_JSON" not in env

    def test_region_env_var_unset_when_path_not_provided(self, tmp_path):
        env: dict[str, str] = {}
        with (
            patch.object(plugin_runtime, "ensure_plugin_built"),
            patch.object(plugin_runtime, "find_plugin_lib_dir", return_value=tmp_path / "lib"),
        ):
            setup_plugin_environment(env=env, event_json_path=tmp_path / "ev.json")
        assert "DD4BENCH_REGION_JSON" not in env

    def test_region_env_var_set_when_path_provided(self, tmp_path):
        region_json = tmp_path / "subdir" / "regions.json"
        env: dict[str, str] = {}
        with (
            patch.object(plugin_runtime, "ensure_plugin_built"),
            patch.object(plugin_runtime, "find_plugin_lib_dir", return_value=tmp_path / "lib"),
        ):
            setup_plugin_environment(
                env=env,
                event_json_path=tmp_path / "ev.json",
                region_json_path=region_json,
            )
        assert env["DD4BENCH_REGION_JSON"] == str(region_json.resolve())

    def test_both_env_vars_set_independently(self, tmp_path):
        event_json = tmp_path / "events.json"
        region_json = tmp_path / "regions.json"
        env: dict[str, str] = {}
        with (
            patch.object(plugin_runtime, "ensure_plugin_built"),
            patch.object(plugin_runtime, "find_plugin_lib_dir", return_value=tmp_path / "lib"),
        ):
            setup_plugin_environment(
                env=env,
                event_json_path=event_json,
                region_json_path=region_json,
            )
        assert env["DD4BENCH_EVENT_JSON"] == str(event_json.resolve())
        assert env["DD4BENCH_REGION_JSON"] == str(region_json.resolve())

    def test_returns_false_when_region_requested_but_plugin_unavailable(self, tmp_path):
        """Failure path is shared — region request doesn't change it."""
        env: dict[str, str] = {}
        with patch.object(
            plugin_runtime, "ensure_plugin_built", side_effect=FileNotFoundError("no plugin")
        ):
            result = setup_plugin_environment(
                env=env,
                event_json_path=tmp_path / "ev.json",
                region_json_path=tmp_path / "rg.json",
            )
        assert result is False
        assert "DD4BENCH_REGION_JSON" not in env
