# Configuration reference

Tables for the CLI flags and the nightly benchmark YAML keys. For narrative
explanations see [Configuration](../user-guide/configuration.md) and
[Commands](../user-guide/commands.md); for the library API, the
[API reference](api-reference.md).

## CLI flags

| Flag | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--xml` | path | — | ✅ | Top-level DD4hep compact XML for the geometry under test. |
| `--sweep` | bool | `false` | — | Full sweep: baseline + one run per detector removed. Mutually exclusive with `--include-only` / `--exclude-only`. |
| `--include-only` | str… | — | — | Single run keeping only the named detectors. Mutually exclusive group. |
| `--exclude-only` | str… | — | — | Single run with the named detectors removed. Mutually exclusive group. |
| `--events` | int | `2` | — | Events per run → injected as `--numberOfEvents`; used for `events_per_sec`. |
| `--ddsim-args` | str | `""` | — | Args passed verbatim to `ddsim`, as one quoted string. Use the `=` form. |
| `--output-file` | path | `/tmp/k4bench_out.edm4hep.root` | — | Temporary EDM4hep ROOT output (`--outputFile`); reused/overwritten, only size recorded. |
| `--output-dir` | path | `logs/<xml-stem>/` | — | Directory for logs and results; created if absent. |
| `--pickle` | str | *(none)* | — | If set, also write `list[RunResult]` as a pickle inside `--output-dir`. |
| `--verbose`, `-v` | bool | `false` | — | Stream `ddsim` output live (always captured to the `.log` regardless). |

### Interactions & validation

- `--sweep` / `--include-only` / `--exclude-only` are an `argparse` mutually
  exclusive group — at most one. None → baseline.
- `--include-only` with no names is impossible from the CLI (it requires `nargs="+"`),
  and a programmatic empty list raises in `BenchmarkConfig.__post_init__`.
- `--exclude-only` with names that are all unknown → `ValueError`; an effectively
  empty exclude set falls back to baseline.
- Don't put `--compactFile` / `--numberOfEvents` / `--outputFile` in
  `--ddsim-args`; they're injected and would collide.

### Library use

The CLI builds a `BenchmarkConfig` from these flags. Driving k4Bench from Python
uses the same fields plus `setup_script` (a shell script sourced before each
`ddsim` run), which has no CLI flag. See the
[`benchmark.ddsim` API](api/benchmark/ddsim.md) for the current field list.

## Nightly benchmark YAML keys

Files in `.github/benchmarks/*.yml`. The filename stem is the detector config
name (must match `^[A-Za-z0-9_-]+$`). Validated by `list_benchmarks.py`.

### Top-level keys (defaults for every sample in the file)

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `xml` | str | ✅ | Geometry, `$K4GEO`-relative or absolute. May contain `$VAR` refs (e.g. `$DD4hepINSTALL` for DD4hep's own example detectors), expanded in the runner. |
| `verbose` | bool | — | Stream ddsim output (default `false`). |
| `sweep` | bool | — | Baseline + one run per subdetector dropped. |
| `include_only` | list | — | Keep only these subdetectors (single run). |
| `exclude_only` | list | — | Drop these subdetectors (single run). |
| `ddsim_args` | str | — | ddsim flags applied to every sample (concatenated with sample-level). |
| `steering_file` | str | — | `ddsim --steeringFile` path; `$VAR` (e.g. `$FCCCONFIG`) expanded in the runner. Its containing directory is put on `PYTHONPATH`, so a steering file that itself does a relative `from sibling import *` (e.g. CLDConfig's `cld_arc_steer.py`) resolves. |
| `samples` | list | ✅ | List of sample entries (below). |

### Per-sample keys (under `samples:`)

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | str | ✅ | Slug (`^[A-Za-z0-9_.+-]+$`); becomes the EOS sample dir + job label. |
| `n_events` | int > 0 | ✅ | Events to simulate. |
| `ddsim_args` | str | — | **Appended** to top-level `ddsim_args` (not replaced). |
| `input_files` | list | — | HepMC path(s); mutually exclusive with `--enableGun`. |
| `steering_file` | str | — | Overrides the top-level steering file for this sample. |

### YAML validation rules

- `n_events` must be a positive integer.
- `input_files` and `--enableGun` (in `ddsim_args`) are mutually exclusive.
- `sweep` / `include_only` / `exclude_only` are mutually exclusive (at most one).
- Lists are joined to space-separated strings so they round-trip through GitHub
  Actions env vars unchanged.
- `ddsim_args` is the **only** key that concatenates (top + sample); all others
  override.

Full schema with examples: [File formats → benchmark YAML](file-formats.md#benchmark-yaml).

## See also

- [Commands](../user-guide/commands.md) — the CLI walkthrough with examples.
- [File formats](file-formats.md) — output schemas.
