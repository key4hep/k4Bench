# API reference

k4Bench is a small, importable Python package. This page is the entry point to
the auto-generated reference — the pages under it are built from the live
docstrings and type hints by [mkdocstrings](https://mkdocstrings.github.io/),
so they never drift from the code.

!!! tip "Docstring quality is doc quality"
    Because these pages render the source docstrings directly, improving a
    docstring improves the docs.

## How to import

```python
# High-level orchestration
from k4bench.benchmark.ddsim import BenchmarkConfig, SweepMode, run_sweep

# Results
from k4bench.results.model import RunResult

# Geometry
from k4bench.geometry.scanner import get_detector_names, resolve_includes
from k4bench.geometry.index import GeometryIndex
from k4bench.geometry.patcher import (
    build_patch, patched, patched_geometry, patched_geometry_keep_only,
)

# Analysis (the most common public surface)
from k4bench.analysis import (
    load_results, load_event_timing, load_region_timing,
    plot_run_overview, plot_event_timing, plot_event_memory, plot_region_timing,
)
```

## The modules at a glance

| Module | Public surface | Page |
| --- | --- | --- |
| `k4bench.cli` | `main` | [cli](api/cli.md) |
| `k4bench.benchmark.ddsim` | `BenchmarkConfig`, `SweepMode`, `run_sweep` | [benchmark.ddsim](api/benchmark/ddsim.md) |
| `k4bench.benchmark.k4run` | `K4runConfig`, `run_k4run_benchmark`, `planned_k4run_labels` | [benchmark.k4run](api/benchmark/k4run.md) |
| `k4bench.geometry.scanner` | `get_detector_names`, `resolve_includes` | [geometry.scanner](api/geometry/scanner.md) |
| `k4bench.geometry.index` | `GeometryIndex`, `FilesystemRef` | [geometry.index](api/geometry/index.md) |
| `k4bench.geometry.patcher` | `build_patch`, `patched`, `PatchResult`, `patched_geometry`, `patched_geometry_keep_only` | [geometry.patcher](api/geometry/patcher.md) |
| `k4bench.geometry.errors` | geometry exception hierarchy | [geometry.errors](api/geometry/errors.md) |
| `k4bench.runner.ddsim` | `run_ddsim` | [runner.ddsim](api/runner/ddsim.md) |
| `k4bench.runner.k4run` | `run_k4run` | [runner.k4run](api/runner/k4run.md) |
| `k4bench.runner.result` | `run_result`, `read_peak_vmem_mb` | [runner.result](api/runner/result.md) |
| `k4bench.runner.parser` | `parse_time_output` | [runner.parser](api/runner/parser.md) |
| `k4bench.results.model` | `RunResult` | [results.model](api/results/model.md) |
| `k4bench.results.reporter` | `print_summary`, `save_csv`, `print_run_result` | [results.reporter](api/results/reporter.md) |
| `k4bench.plugin.runtime` | `setup_plugin_environment`, `setup_auditor_environment`, `find_plugin_lib_dir`, `ensure_plugin_built` | [plugin.runtime](api/plugin/runtime.md) |
| `k4bench.analysis` | the loaders + plot functions | [analysis](api/analysis/index.md) |

The complete, navigable tree is in the **API** section of the sidebar.

## Stability

The functions and dataclasses listed above are the intended public API.
Underscore-prefixed names are internal, may change without notice, and are
excluded from these generated pages. The project is young and evolving, so even
the public surface may shift between releases — pin a version if you depend on
it programmatically.

## See also

- [Architecture → component diagrams](../architecture/component-diagrams.md) —
  how these classes relate.
- [Overview → two ways to drive it](../user-guide/overview.md#two-ways-to-drive-it)
  — the library entry point in context.
