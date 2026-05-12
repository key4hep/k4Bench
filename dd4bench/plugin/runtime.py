from __future__ import annotations

import subprocess
from pathlib import Path


def find_plugin_lib_dir() -> Path:
    """Return directory containing the DD4bench timing plugin."""

    plugin_root = Path.cwd() / "plugin"

    library_patterns = [
        "libDD4benchTimingAction.so",
        "libDD4benchTimingAction.dylib",
    ]

    for libdir in ("lib", "lib64"):
        candidate = plugin_root / "install" / libdir

        if not candidate.exists():
            continue

        for pattern in library_patterns:
            if any(candidate.glob(pattern)):
                return candidate

    build_dir = plugin_root / "build"

    for pattern in library_patterns:
        if any(build_dir.glob(pattern)):
            return build_dir

    raise FileNotFoundError("Could not locate DD4benchTimingAction plugin library.")


def ensure_plugin_built() -> None:
    """Build the DD4bench timing plugin if needed."""

    plugin_root = Path.cwd() / "plugin"

    try:
        find_plugin_lib_dir()
        return
    except FileNotFoundError:
        pass

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
            "Failed to build DD4bench timing plugin:\n"
            f"{result.stdout}\n{result.stderr}"
        )


def setup_plugin_environment(
    *,
    env: dict[str, str],
    event_json_path: Path,
) -> bool:
    """Prepare environment variables for the DD4bench timing plugin.

    Returns
    -------
    bool
        True if the plugin is available and enabled.
        False if ddsim should run without per-event timing.
    """
    try:
        ensure_plugin_built()

        lib_dir = str(find_plugin_lib_dir())

        existing = env.get("LD_LIBRARY_PATH", "")

        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir

        env["DD4BENCH_EVENT_JSON"] = str(event_json_path.resolve())

        return True

    except (
        FileNotFoundError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            f"NOTE: DD4bench timing plugin unavailable "
            f"({exc}); continuing without per-event timing."
        )

        return False
