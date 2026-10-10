# Configuration reference

Tables for the CLI flags and the nightly benchmark YAML keys. For narrative
explanations see [Configuration](../user-guide/configuration.md) and
[Commands](../user-guide/commands.md); for the library API, the
[API reference](api-reference.md).

## CLI flags

| Flag | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--xml` | path | — | ✅ | Top-level DD4hep compact XML for the geometry under test. |
| `--list-detectors` | bool | `false` | — | Print the subdetector names found in `--xml`, one per line, and exit; no simulation is run. |
| `--sweep` | bool | `false` | — | Full sweep: baseline + one run per detector removed. Mutually exclusive with `--sweep-detectors` / `--include-only` / `--exclude-only`. |
| `--sweep-detectors` | str… | — | — | Partial sweep: baseline + one run per named detector removed (like `--sweep`, restricted). Mutually exclusive group. |
| `--include-only` | str… | — | — | Single run keeping only the named detectors. Mutually exclusive group. |
| `--exclude-only` | str… | — | — | Single run with the named detectors removed. Mutually exclusive group. |
| `--events` | int | `2` | — | Events per run → injected as `--numberOfEvents`; used for `events_per_sec`. |
| `--ddsim-args` | str | `""` | — | Args passed verbatim to `ddsim`, as one quoted string. Use the `=` form. |
| `--output-file` | path | `/tmp/k4bench_out.edm4hep.root` | — | Temporary EDM4hep ROOT output (`--outputFile`); reused/overwritten, only size recorded. |
| `--output-dir` | path | `logs/<xml-stem>/` | — | Directory for logs and results; created if absent. |
| `--pickle` | str | *(none)* | — | If set, also write `list[RunResult]` as a pickle inside `--output-dir`. |
| `--verbose`, `-v` | bool | `false` | — | Stream `ddsim` output live (always captured to the `.log` regardless). |

### Interactions & validation

- `--list-detectors` short-circuits in `main()` before config building: any
  other flags (`--sweep`, `--events`, `--ddsim-args`, …) are parsed but ignored.
- `--sweep` / `--sweep-detectors` / `--include-only` / `--exclude-only` are an
  `argparse` mutually exclusive group — at most one. None → baseline.
- `--sweep-detectors` maps to `SweepMode.FULL` with the removal set narrowed to
  the named detectors; unknown names are warned and skipped, and all-unknown →
  `ValueError` listing the available detectors.
- `--include-only` with no names is impossible from the CLI (it requires `nargs="+"`),
  and a programmatic empty list raises in `BenchmarkConfig.__post_init__`.
- `--exclude-only` with names that are all unknown → `ValueError`; an effectively
  empty exclude set falls back to baseline.
- Don't put `--compactFile` / `--numberOfEvents` / `--outputFile` in
  `--ddsim-args`; they're injected and would collide.

### `k4bench k4run`

`k4bench k4run OPTIONS [OPTIONS ...] [flags]` benchmarks a k4run job instead,
measured per component by the k4BenchAuditor. Every run starts in a fresh
working directory, removed afterwards, so give paths in `--k4run-args` as
absolute paths.

| Flag | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `OPTIONS` | path… | — | ✅ | Gaudi options files, loaded in order; relative to `--stage-dir` when given. The auditor's options file is appended after them. |
| `--stage-dir` | path | *(none)* | — | Directory copied (writable) into every run's working directory, for jobs that must run from their own configuration directory. |
| `--events` | int | `2` | — | Events per run → injected as `--num-events`; used for `events_per_sec`. |
| `--k4run-args` | str | `""` | — | Args passed verbatim to `k4run` after the options files, as one quoted string. Use the `=` form. |
| `--variant` | `NAME=ARGS` | — | — | One more run, labelled `variant_NAME`, with `ARGS` appended to `--k4run-args`. Repeatable. |
| `--output-dir` | path | `logs/<options-stem>/` | — | Directory for logs and results; created if absent. |
| `--pickle` | str | *(none)* | — | As for ddsim. |
| `--verbose`, `-v` | bool | `false` | — | Stream `k4run` output live. |

Don't put `--num-events` in `--k4run-args`; it is injected.

### Library use

The CLI builds a `BenchmarkConfig` from these flags. Driving k4Bench from Python
uses the same fields plus `setup_script` (a shell script sourced before each
`ddsim` run), which has no CLI flag. See the
[`benchmark.ddsim` API](api/benchmark/ddsim.md) for the current field list.

## Nightly benchmark YAML keys

Files in `.github/benchmarks/*.yml`. The filename stem is the detector config
name (must match `^[A-Za-z0-9_-]+$`). Validated by `list_benchmarks.py`.
`tool` selects the benchmark: `ddsim` (the default) or `k4run`. The keys below
are ddsim's except where marked; a key of the other tool is an error.

### Top-level keys (defaults for every sample in the file)

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `xml` | str | ✅ | Geometry, `$K4GEO`-relative or absolute. May contain `$VAR` refs (e.g. `$DD4hepINSTALL` for DD4hep's own example detectors), expanded in the runner. |
| `verbose` | bool | — | Stream ddsim output (default `false`). |
| `sweep` | bool | — | Baseline + one run per subdetector dropped. |
| `sweep_detectors` | list | — | Baseline + one run per **named** subdetector dropped (partial sweep). |
| `include_only` | list | — | Keep only these subdetectors (single run). |
| `exclude_only` | list | — | Drop these subdetectors (single run). |
| `ddsim_args` | str | — | ddsim flags applied to every sample (concatenated with sample-level). |
| `steering_file` | str | — | `ddsim --steeringFile` path; `$VAR` (e.g. `$FCCCONFIG`) expanded in the runner. Its containing directory is put on `PYTHONPATH`, so a steering file that itself does a relative `from sibling import *` (e.g. CLDConfig's `cld_arc_steer.py`) resolves. |
| `smoke_timeout` | int > 0 | — | Minutes the bump-PR smoke test allows per geometry/steering pair (default `5`); for geometries that are slow to build, e.g. IDEA_o2's DR tube. Ignored by the nightly. |
| `tool` | str | — | `ddsim` (default) or `k4run`. |
| `options` | list | k4run ✅ | Gaudi options file(s), relative to `stage_dir`. |
| `stage_dir` | str | — | k4run: directory each run starts from a writable copy of; `$VAR` (e.g. `$CLDCONFIG`) expanded in the runner. |
| `k4run_args` | str | — | k4run: job arguments (concatenated with sample-level). `$VAR`s are expanded in the runner, including `$DETECTOR_XML` (the resolved `xml`) and `$LOCAL_INPUT_FILES` (the local copies of `input_files`). |
| `variants` | list | — | k4run: `{name, k4run_args}` entries; each is one more run, labelled `variant_<name>`, with its `k4run_args` appended. |
| `samples` | list | ✅ | List of sample entries (below). |

### Per-sample keys (under `samples:`)

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | str | ✅ | Slug (`^[A-Za-z0-9_.+-]+$`); becomes the EOS sample dir + job label. |
| `n_events` | int > 0 | ✅ | Events to simulate. |
| `ddsim_args` | str | — | **Appended** to top-level `ddsim_args` (not replaced). |
| `input_files` | list | — | HepMC path(s); mutually exclusive with `--enableGun`. k4run: frozen input file(s), copied locally before the run from an HTTPS URL (curl), an XRootD `root://` URL (xrdcp) or a local path. |
| `k4run_args` | str | — | k4run: **appended** to top-level `k4run_args`. |
| `variants` | list | — | k4run: replaces the top-level variants for this sample. |
| `steering_file` | str | — | Overrides the top-level steering file for this sample. |

### YAML validation rules

- `n_events` must be a positive integer.
- `input_files` and `--enableGun` (in `ddsim_args`) are mutually exclusive.
- `sweep` / `sweep_detectors` / `include_only` / `exclude_only` are mutually exclusive (at most one).
- Lists are joined to space-separated strings so they round-trip through GitHub
  Actions env vars unchanged.
- `ddsim_args` and `k4run_args` are the **only** keys that concatenate
  (top + sample); all others override.
- `tool: k4run` requires `options` and `xml` (the geometry it reconstructs,
  which names the EOS detector directory), and refuses `ddsim_args`,
  `steering_file`, `sweep`, `sweep_detectors`, `include_only` and
  `exclude_only`; a ddsim benchmark refuses `options`, `stage_dir`,
  `k4run_args` and `variants`.
- Variant names follow the sample-name rule, must be unique, and each needs
  `k4run_args`.
- Two configs of one tool uploading the same sample under one compact-file
  basename are refused. k4run runs upload to their own `_k4run/` tree, so a
  k4run sample may share its name with a ddsim one.

Full schema with examples: [File formats → benchmark YAML](file-formats.md#benchmark-yaml).

## See also

- [Commands](../user-guide/commands.md) — the CLI walkthrough with examples.
- [File formats](file-formats.md) — output schemas.
