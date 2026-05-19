from __future__ import annotations

import subprocess
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


# Plugin library filenames (glob patterns). Each tuple is
# (env_var_name, library_pattern, friendly_name).
_PLUGINS = (
    ("DD4BENCH_EVENT_JSON",  "libDD4benchTimingAction.so*",       "event timing"),
    ("DD4BENCH_REGION_JSON", "libDD4benchRegionTimingAction.so*", "region timing"),
)


def _find_plugin_root() -> Path:
    """Locate the DD4bench plugin source directory."""
    candidates = [
        Path.cwd() / "plugin",
        _PACKAGE_ROOT.parent / "plugin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate DD4bench plugin directory.")


def find_plugin_lib_dir() -> Path:
    """Return directory containing the DD4bench plugin libraries.

    The returned directory must contain both the .so files AND the
    .components manifests that DDG4 uses to resolve factory names.
    Without the .components files, DDG4 can only load plugins whose
    library name matches the class name exactly (e.g. libFoo.so for
    class Foo), so bundled plugins like DD4benchRegionEventAction
    (which lives in libDD4benchRegionTimingAction.so) would be silently
    skipped.
    """
    plugin_root = _find_plugin_root()
    library_patterns = [pattern for _, pattern, _ in _PLUGINS]
    # Derive the .components filename from the .so glob pattern.
    components_patterns = [p.split(".so")[0] + ".components" for p in library_patterns]

    search_dirs = [
        plugin_root / "install" / "lib",
        plugin_root / "install" / "lib64",
        plugin_root / "build",
    ]

    for libdir in search_dirs:
        if not libdir.exists():
            continue
        if (all(any(libdir.glob(p)) for p in library_patterns)
                and all(any(libdir.glob(p)) for p in components_patterns)):
            return libdir

    raise FileNotFoundError(
        "Could not locate DD4bench plugin libraries "
        f"({', '.join(library_patterns)})."
    )


def ensure_plugin_built() -> None:
    """Build the DD4bench plugins if needed."""
    try:
        find_plugin_lib_dir()
        return
    except FileNotFoundError:
        pass

    plugin_root = _find_plugin_root()
    build_script = plugin_root / "build.sh"

    if not build_script.exists():
        raise FileNotFoundError(f"Missing plugin build script: {build_script}")

    result = subprocess.run(
        ["bash", str(build_script)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build DD4bench timing plugins:\n"
            f"{result.stdout}\n{result.stderr}"
        )


def setup_plugin_environment(
    *,
    env: dict[str, str],
    event_json_path: Path,
    region_json_path: Path | None = None,
) -> bool:
    """Prepare environment variables for the DD4bench timing plugins.

    Parameters
    ----------
    env
        Environment dictionary to mutate (typically a copy of os.environ).
    event_json_path
        Output path for the per-event timing JSON.
    region_json_path
        Output path for the per-region timing JSON. If None, the region
        plugin will still be loadable but will write to its default
        location (dd4bench_regions.json in CWD) only if it ends up being
        activated by the steering script.

    Returns
    -------
    bool
        True if plugins are available and enabled.
        False if ddsim should run without per-event timing.
    """
    try:
        ensure_plugin_built()
        lib_dir = str(find_plugin_lib_dir())

        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir

        env["DD4BENCH_EVENT_JSON"] = str(event_json_path.resolve())

        if region_json_path is not None:
            env["DD4BENCH_REGION_JSON"] = str(region_json_path.resolve())

        return True

    except (
        FileNotFoundError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            f"NOTE: DD4bench timing plugins unavailable "
            f"({exc}); continuing without per-event timing."
        )

        return False
