# File formats

The artifacts k4Bench reads and writes. These are the contracts between the
runner, the analysis layer, the CI pipeline, and the dashboard.

!!! note "The loaders are the source of truth"
    The JSON schemas below are described at a useful level of detail, but they
    evolve with the code. The [analysis loaders](../user-guide/features/analysis.md)
    (`k4bench.analysis.loader`) are what actually parse these files — when in
    doubt about a field, check the loader.

## Output directory layout

A single `k4bench` run populates `--output-dir` (default `logs/<xml-stem>/`):

```text
logs/<geometry>/
├── <label>.log              # full ddsim stdout/stderr incl. the time -v block
├── <label>_results.csv      # one row: run-level metrics
├── <label>_events.json      # per-event timing/RSS   (event plugin only)
└── <label>_regions.json     # per-detector timing    (region plugin only)
```

In the nightly CI each run directory also gets `run_info.json` and
`machine_info.json`, and the whole directory is uploaded to
[EOS](#eos-layout).

## results CSV

`<label>_results.csv` is one header row plus one data row, whose columns are
exactly the fields of [`RunResult`](api/results/model.md) — see that page for
the authoritative list. The headline columns:

| Column | Unit | Meaning |
| --- | --- | --- |
| `label`, `returncode`, `n_events` | — | identity + ddsim exit code |
| `wall_time_s` | s | elapsed wall clock |
| `user_cpu_s`, `sys_cpu_s` | s | CPU times |
| `peak_rss_mb` | MB | peak resident memory |
| `output_size_mb` | MB | size of the EDM4hep ROOT output |
| `events_per_sec` | ev/s | throughput |

Metric fields can be empty when `/usr/bin/time -v` output can't be parsed (e.g.
a crashed run). [`load_results`](api/analysis/loader.md) reads these into a
DataFrame with appropriate numeric types.

## events JSON

Written by the event timing plugin: parallel arrays, one entry per event —
event numbers, per-event wall time (seconds), and RSS (MB) sampled at the start
and end of each event. [`load_event_timing`](api/analysis/loader.md) returns a
DataFrame per run label and adds an RSS-delta column. Event 0 is a
[warmup outlier](../user-guide/features/analysis.md#warmup-events).

## regions JSON

Written by the region timing plugin (`schema_version: 1`). It attributes Geant4
stepping time to top-level DD4hep detectors under two views,
[`at_location` and `by_birth`](../user-guide/features/timing-plugins.md#per-region-timing),
plus per-event totals, step counts, and metadata (timer used, the list of
attributed detectors, measured timer overhead). [`load_region_timing`](api/analysis/loader.md)
parses it into per-run DataFrames keyed by detector. The special bucket
`unattributed` collects steps outside any detector (vacuum/world).

## run_info.json & machine_info.json

Written by the nightly CI and read by the dashboard.

- **`run_info.json`** — describes one run directory: date, platform, Key4hep
  release, detector, sample, the GitHub run link and commit, event count, and
  the list of run labels (`configs`).
- **`machine_info.json`** — the benchmark host and its state *around* the run:
  CPU model/cores, RAM/swap totals, and `_start`/`_end` snapshots of load,
  available memory, CPU frequency, and thermal throttling. The pairs let the
  dashboard show whether the machine was loaded or throttling — context for
  trusting a number.

## Benchmark YAML

`.github/benchmarks/<detector>.yml` configures the nightly matrix; expanded by
`list_benchmarks.py`. Keys are tabulated in the
[Configuration reference](configuration-reference.md#nightly-benchmark-yaml-keys).
Example:

```yaml
xml: FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml
steering_file: $FCCCONFIG/FullSim/IDEA/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py
sweep: false
samples:
  - name: single_e-_10GeV
    n_events: 100
    ddsim_args: >-
      --enableGun --gun.particle e- --gun.distribution uniform --gun.energy 10*GeV
  - name: p8_ee_Zbb_ecm91
    n_events: 100
    input_files: root://eospublic.cern.ch//eos/.../events_noVtxSmear.hepmc
```

## EOS layout

Nightly results live under `EOS_ROOT = /eos/user/j/jbeirer/k4bench`, encoding
every browse dimension in the path so discovery is just directory listing:

```text
{detector}/{platform}/key4hep-{release}/{sample}/{YYYY-MM-DD}/
    run_info.json  machine_info.json
    {config}_results.csv  {config}_events.json  {config}_regions.json  {config}.log
_reports/{YYYY-MM-DD}/
    report.json
```

This is the integration contract between CI and the dashboard
([data flow](../architecture/data-flow.md#nightly-eos-dashboard)).
Underscore-prefixed top-level directories are reserved for non-detector data:
`_reports/` holds the nightly regression report (written by the
`regression-report` CI job, rendered by the dashboard's Regressions tab) and is
skipped by detector discovery.

## See also

- [Analysis](../user-guide/features/analysis.md) — the loaders that parse these.
- [`RunResult`](api/results/model.md) — the CSV's source of truth.
- [Configuration reference](configuration-reference.md) — the YAML keys.
