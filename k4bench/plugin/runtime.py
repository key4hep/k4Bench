from __future__ import annotations

import subprocess
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


# Plugin library filenames (glob patterns). Each tuple is
# (env_var_name, library_pattern, friendly_name).
_PLUGINS = (
    ("K4BENCH_EVENT_JSON",  "libk4BenchTimingAction.so*",       "event timing"),
    ("K4BENCH_REGION_JSON", "libk4BenchRegionTimingAction.so*", "region timing"),
)

# The Gaudi auditor for k4run jobs, installed with its .components manifest
# under lib/gaudi-plugins/ or, with older Gaudi, lib/, and the options file
# that enables it.
_AUDITOR_LIBRARY = "libk4BenchAuditor.so*"
_AUDITOR_OPTIONS = "k4BenchAuditorOptions.py"

# Preloaded into an audited job so the auditor can count heap allocations;
# installed next to the auditor.
_ALLOC_COUNTER_LIBRARY = "libk4BenchAllocCounter.so"

# Allocators a job may preload to replace glibc's. The counter forwards to
# glibc's allocator and reads glibc's block headers, so preloaded before one of
# these it would silently swap the allocator being benchmarked for glibc's.
_PRELOADED_ALLOCATORS = ("jemalloc", "tcmalloc", "mimalloc", "tbbmalloc", "hoard", "snmalloc")

# Sanitizer runtimes replace the allocator too, and must come first in the
# preload list: put after the counter, they abort the job.
_SANITIZER_RUNTIMES = ("asan", "tsan", "msan", "lsan")

_ALLOCATOR_NOTE = (
    "NOTE: the job preloads its own allocator ({}); continuing without allocation "
    "counts, which would replace it with glibc's."
)

#: Bash that preloads the allocation counter named by ``K4BENCH_ALLOC_COUNTER``
#: unless ``LD_PRELOAD`` already holds an allocator or sanitizer runtime. A job
#: with a setup script runs it after sourcing the script, which may set
#: ``LD_PRELOAD`` itself; see :func:`setup_auditor_environment`.
ALLOC_COUNTER_PRELOAD = f"""\
k4bench_allocator=
IFS=': ' read -ra k4bench_preloads <<< "${{LD_PRELOAD-}}"
for k4bench_library in "${{k4bench_preloads[@]}}"; do
    k4bench_name=${{k4bench_library##*/}}
    case ${{k4bench_name,,}} in
        {"|".join(f"*{name}*" for name in (*_PRELOADED_ALLOCATORS, *_SANITIZER_RUNTIMES))})
            k4bench_allocator=$k4bench_library
            break ;;
    esac
done
if [[ -n $k4bench_allocator ]]; then
    echo "{_ALLOCATOR_NOTE.format("$k4bench_allocator")}"
elif [[ -n ${{K4BENCH_ALLOC_COUNTER-}} ]]; then
    export LD_PRELOAD="$K4BENCH_ALLOC_COUNTER${{LD_PRELOAD:+:$LD_PRELOAD}}"
fi
"""


def _find_plugin_root() -> Path:
    """Locate the k4Bench plugin source directory."""
    candidates = [
        Path.cwd() / "plugin",
        _PACKAGE_ROOT.parent / "plugin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate k4Bench plugin directory.")


def find_plugin_lib_dir() -> Path:
    """Return directory containing the k4Bench plugin libraries.

    The returned directory must contain both the .so files AND the
    .components manifests that DDG4 uses to resolve factory names.
    Without the .components files, DDG4 can only load plugins whose
    library name matches the class name exactly (e.g. libFoo.so for
    class Foo), so bundled plugins like k4BenchRegionEventAction
    (which lives in libk4BenchRegionTimingAction.so) would be silently
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
        "Could not locate k4Bench plugin libraries "
        f"({', '.join(library_patterns)})."
    )


def find_auditor_dir() -> Path:
    """Return the directory holding the k4BenchAuditor Gaudi plugin.

    Gaudi's plugin service finds a component through the ``.components``
    manifest next to its library, so the directory must hold both. Gaudi
    installs plugins into ``lib/gaudi-plugins`` or, in older releases, ``lib``,
    where the DDG4 plugins' manifests also live, so the manifest must be the
    one registering the auditor.
    """
    plugin_root = _find_plugin_root()
    for libdir in (
        plugin_root / "install" / "lib" / "gaudi-plugins",
        plugin_root / "install" / "lib64" / "gaudi-plugins",
        plugin_root / "install" / "lib",
        plugin_root / "install" / "lib64",
    ):
        if any(libdir.glob(_AUDITOR_LIBRARY)) and any(
            "k4BenchAuditor" in manifest.read_text() for manifest in libdir.glob("*.components")
        ):
            return libdir
    raise FileNotFoundError(f"Could not locate the k4Bench auditor ({_AUDITOR_LIBRARY}).")


def auditor_options_file() -> Path:
    """Gaudi options file that enables the auditor, appended after a job's own."""
    return _find_plugin_root() / "auditor" / _AUDITOR_OPTIONS


def ensure_plugin_built() -> None:
    """Build the k4Bench plugins if needed."""
    try:
        find_plugin_lib_dir()
        return
    except FileNotFoundError:
        pass
    _run_build_script("ddg4")


def ensure_auditor_built() -> None:
    """Build the k4Bench auditor if it is missing or stale.

    Always runs build.sh, which rebuilds only when a library is older than its
    sources, so an installed auditor never outlives a change to them. Only the
    auditor is built, so a DDG4 plugin that fails to build does not disable it.
    """
    _run_build_script("auditor")


def _run_build_script(target: str) -> None:
    """Run build.sh for one *target* (``ddg4`` or ``auditor``) only."""
    plugin_root = _find_plugin_root()
    build_script = plugin_root / "build.sh"

    if not build_script.exists():
        raise FileNotFoundError(f"Missing plugin build script: {build_script}")

    result = subprocess.run(
        ["bash", str(build_script), target],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build k4Bench timing plugins:\n"
            f"{result.stdout}\n{result.stderr}"
        )


def setup_plugin_environment(
    *,
    env: dict[str, str],
    event_json_path: Path,
    region_json_path: Path | None = None,
) -> bool:
    """Prepare environment variables for the k4Bench timing plugins.

    Parameters
    ----------
    env
        Environment dictionary to mutate (typically a copy of os.environ).
    event_json_path
        Output path for the per-event timing JSON.
    region_json_path
        Output path for the per-region timing JSON. If None, the region
        plugin will still be loadable but will write to its default
        location (k4bench_regions.json in CWD) only if it ends up being
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

        env["K4BENCH_EVENT_JSON"] = str(event_json_path.resolve())

        if region_json_path is not None:
            env["K4BENCH_REGION_JSON"] = str(region_json_path.resolve())
        else:
            env.pop("K4BENCH_REGION_JSON", None)

        return True

    except (
        FileNotFoundError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            f"NOTE: k4Bench timing plugins unavailable ({exc}); continuing "
            f"without per-event timing and without the virtual-peak and "
            f"anonymous-RSS measurements the regression engine judges."
        )

        return False


def setup_auditor_environment(
    *,
    env: dict[str, str],
    components_json_path: Path,
    event_json_path: Path,
    joboptions_path: Path,
    setup_script: Path | None = None,
) -> bool:
    """Prepare environment variables for a k4run job audited by k4Bench.

    The job must also be given :func:`auditor_options_file` as its last options
    file; this only makes the auditor findable, names its outputs and preloads
    the allocation counter, without which the auditor records no allocations.

    A *setup_script* sourced before the job may set ``LD_PRELOAD`` itself, so
    with one the counter is only named in ``K4BENCH_ALLOC_COUNTER``, and the
    job's command must run :data:`ALLOC_COUNTER_PRELOAD` after the script.

    Parameters
    ----------
    env
        Environment dictionary to mutate (typically a copy of os.environ).
    components_json_path
        Output path for the per-component JSON.
    event_json_path
        Output path for the per-event JSON, in the ddsim event format.
    joboptions_path
        Output path for the resolved job options dump.
    setup_script
        Shell script the job's command sources before running it, if any.

    Returns
    -------
    bool
        True if the auditor is available and enabled. False if the job should
        run without per-component and per-event measurements.
    """
    try:
        ensure_auditor_built()
        plugin_dir = str(find_auditor_dir())
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError) as exc:
        print(
            f"NOTE: k4Bench auditor unavailable ({exc}); continuing without "
            f"per-component and per-event measurements."
        )
        return False

    # On Linux Gaudi finds plugins through LD_LIBRARY_PATH, and its manifests
    # name libraries the dynamic linker resolves there; newer releases also
    # search GAUDI_PLUGIN_PATH.
    for variable in ("GAUDI_PLUGIN_PATH", "LD_LIBRARY_PATH"):
        existing = env.get(variable, "")
        env[variable] = f"{plugin_dir}:{existing}" if existing else plugin_dir
    env["K4BENCH_COMPONENTS_JSON"] = str(components_json_path.resolve())
    env["K4BENCH_EVENT_JSON"] = str(event_json_path.resolve())
    env["K4BENCH_JOBOPTIONS"] = str(joboptions_path.resolve())

    counter = Path(plugin_dir) / _ALLOC_COUNTER_LIBRARY
    env.pop("K4BENCH_ALLOC_COUNTER", None)
    if setup_script is not None and counter.is_file():
        env["K4BENCH_ALLOC_COUNTER"] = str(counter)
        return True
    existing = env.get("LD_PRELOAD", "")
    allocator = next(
        (
            library
            for library in existing.replace(" ", ":").split(":")
            if any(
                name in Path(library).name.lower()
                for name in (*_PRELOADED_ALLOCATORS, *_SANITIZER_RUNTIMES)
            )
        ),
        None,
    )
    if allocator is not None:
        print(_ALLOCATOR_NOTE.format(allocator))
    elif counter.is_file():
        env["LD_PRELOAD"] = f"{counter}:{existing}" if existing else str(counter)
    else:
        print(f"NOTE: {counter} not found; continuing without allocation counts.")
    return True
