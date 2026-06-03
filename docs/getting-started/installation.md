# Installation

k4Bench is a thin Python orchestrator around `ddsim`. It does not bundle a
physics stack — it expects to run **inside a [Key4hep](https://key4hep.github.io/key4hep-doc/)
environment**, which supplies the simulation toolchain and the Python packages
k4Bench builds on.

## Prerequisites

| Requirement | Why it's needed | How you get it |
| --- | --- | --- |
| Key4hep stack | Provides `ddsim`, DD4hep/DDG4, ROOT, and the Python deps | `source /cvmfs/sw.hsf.org/key4hep/setup.sh` (CVMFS) |
| Python ≥ 3.13 | k4bench targets 3.13 (`requires-python` in `pyproject.toml`) | Comes with the Key4hep stack |
| `ddsim` on `PATH` | The thing being benchmarked | Part of Key4hep |
| GNU `time` (`/usr/bin/time`) | Wall-time, RSS, and OS metrics are scraped from `time -v` | `dnf install time` / `apt install time` |
| A DD4hep geometry | The benchmark input | e.g. `$K4GEO/FCCee/...` from Key4hep |
| C++ toolchain + CMake | Builds the per-event/region timing plugins | Part of Key4hep |

!!! warning "GNU time, not the shell built-in"
    The metrics parser reads the verbose output of the standalone
    `/usr/bin/time -v` binary, **not** the Bash `time` keyword. If it is
    missing, `k4bench` raises a clear error at run time.

## Option A — Install from source (recommended)

This is the recommended path. It builds the C++ timing plugins, so you get the
full set of metrics (run-level **and** per-event / per-detector), plus the test
suite and the dashboard.

```bash
git clone https://github.com/key4hep/k4Bench.git
cd k4Bench

# setup.sh sources Key4hep, makes a CVMFS-aware venv, installs deps,
# builds the timing plugins, and installs pre-commit hooks.
source setup.sh

# Install the k4bench command (editable)
pip install --no-build-isolation -e ".[test]"
```

What [`setup.sh`](../developer-guide/development-setup.md) does, step by step:

1. Exports `K4BENCH_REPO` and prepends the plugin build/install dirs to
   `LD_LIBRARY_PATH` so DDG4 can find the timing libraries.
2. Sources the Key4hep stack (release pinned by `KEY4HEP_VERSION`, default
   `2026-04-08`) unless one is already active.
3. Creates a [`cvmfs-venv`](https://github.com/jbeirer/cvmfs-venv) named
   `py-venv` and activates it.
4. Installs the dev tooling from `requirements.txt` (`codespell`, `pre-commit`).
5. Builds the timing plugins via `plugin/build.sh` (idempotent).
6. Installs the `pre-commit` hooks.
7. Captures the full environment into a `.env` file so Jupyter kernels can
   reproduce the Key4hep environment.

After this you have the `k4bench` command on your `PATH` with full plugin
support. Jump to the [Quickstart](quickstart.md).

## Option B — Install from PyPI (no timing plugins)

A bare PyPI install gives you the `k4bench` command and **run-level** metrics
(wall time, RSS, throughput), but **not** the C++ timing plugins — those live in
the source tree's `plugin/` directory and are built from a Git checkout. Without
them, k4bench still runs and simply prints `NOTE: ... continuing without
per-event timing`. Prefer [Option A](#option-a-install-from-source-recommended)
unless you only need run-level numbers.

```bash
# 1. Enter the Key4hep environment (pick a release date)
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08

# 2. Create a venv that inherits the Key4hep site-packages
python -m venv ~/.venvs/k4bench --system-site-packages

# 3. Activate it
source ~/.venvs/k4bench/bin/activate

# 4. Install k4bench WITHOUT its declared deps (they already come from Key4hep)
pip install k4bench --no-deps
```

!!! tip "Why `--no-deps`?"
    k4bench's dependencies are already provided inside Key4hep at versions
    pinned by the stack. `--system-site-packages` exposes them to the venv, and
    `--no-deps` stops `pip` from pulling incompatible copies from PyPI that would
    shadow the stack's builds and cause subtle ABI issues.

## Building the timing plugins manually

The plugins are built automatically by `setup.sh`, but you can (re)build them
directly. The build is idempotent — it recompiles only when a `.cpp` is newer
than its `.so`.

```bash
# Requires the Key4hep/DD4hep environment to be sourced first.
bash plugin/build.sh
```

This produces:

- `plugin/install/lib/libk4BenchTimingAction.so` — per-event timing/RSS
- `plugin/install/lib/libk4BenchRegionTimingAction.so` — per-detector timing

k4bench locates these automatically at run time; see
[`k4bench.plugin.runtime`](../reference/api/plugin/runtime.md) and
[Timing plugins](../user-guide/features/timing-plugins.md).

## Verifying the installation

```bash
# The command exists and shows help
k4bench --help

# A tiny real run (2 events is the default) against any geometry you have
k4bench --xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml \
        --ddsim-args="--enableGun --gun.particle e-"
```

A successful run prints a summary table and writes a `*_results.csv` under
`logs/<geometry-stem>/`. If you built the plugins, you will also see
`..._events.json` and (when a stepping action is active) `..._regions.json`.

## Next steps

- [Quickstart](quickstart.md) — your first real benchmark, explained.
- [First workflow](first-workflow.md) — a full sweep end-to-end with analysis.
