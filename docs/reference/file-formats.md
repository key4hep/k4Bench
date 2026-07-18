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
  release, detector, sample, the GitHub run link and commit, event count, the
  list of run labels (`configs`), and the stack's git provenance
  (`k4h_stack_root`, `k4h_packages`; see below).
- **`machine_info.json`** — the benchmark host and its state *around* the run:
  CPU model/cores, RAM/swap totals, and `_start`/`_end` snapshots of load,
  available memory, CPU frequency, and thermal throttling. The pairs let the
  dashboard show whether the machine was loaded or throttling — context for
  trusting a number.

### Stack provenance (`k4h_packages`)

`run_info.json` records the upstream commit of every package the Key4hep stack
built from git (~63 per nightly, ~11 KB), read off CVMFS as the benchmark runs:

```json
"k4h_stack_root": "/cvmfs/sw-nightlies.hsf.org/key4hep/releases/2026-07-10/x86_64-almalinux9-gcc14.2.0-opt",
"k4h_packages": {
  "k4geo":      {"commit": "0f226a98…", "version": "develop", "repo_url": "https://github.com/key4hep/k4geo.git"},
  "fcc-config": {"commit": "21647280…", "version": "develop", "repo_url": "https://github.com/HEP-FCC/FCC-config"}
}
```

This is what lets a regression be traced to the commits that could have caused
it: diffing two nights' maps gives the exact set of upstream changes between
them. It is captured at run time because it cannot be recovered later — the
CVMFS nightlies area keeps only about a month of releases, so the stack behind
an older run no longer exists to be read.

`repo_url` is the Spack recipe's URL verbatim, so it is `None` for the rare
package whose recipe ships no `git` attribute, and non-GitHub for packages
hosted elsewhere. Every field is best-effort: `k4h_packages` is absent for runs
predating provenance capture, and an empty map means *unknown*, never
*unchanged*.

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
    blame.json   (only on nights with an attributable confirmed regression)
```

This is the integration contract between CI and the dashboard
([data flow](../architecture/data-flow.md#nightly-eos-dashboard)).
Underscore-prefixed top-level directories are reserved for non-detector data:
`_reports/` holds the nightly regression report (written by the
`regression-report` CI job, rendered by the dashboard's Regressions tab) and is
skipped by detector discovery.

### Blame sidecar (`blame.json`)

For each confirmed regression whose blame window spans two *different* releases,
`blame.json` records the repositories that moved across that window and the pull
requests that could have caused it, ranked by how well each matches the
regression. It is a **sidecar**, deliberately separate from `report.json`: blame
needs GitHub, and a GitHub outage, a rate limit, or a force-pushed `develop` must
never degrade or fail the nightly report and its email. Different failure domain,
different file — written best-effort by `blame_report.py` *after* `report.json`
is uploaded, and **absent entirely on most nights** (most nights have no
confirmed, attributable regression).

```json
{
  "generated_at": "2026-07-05T06:10:00+00:00",
  "report_night": "2026-07-05",
  "entries": [
    {
      "detector": "ALLEGRO_o1_v03", "platform": "…", "sample": "single_e-_10GeV",
      "label": "baseline", "metric": "wall_time_s", "sub_detector": null,
      "base_release": "2026-07-03", "onset_release": "2026-07-04",
      "n_unchanged": 60,
      "repos": [
        {
          "package": "k4geo", "repo": "key4hep/k4geo",
          "base_commit": "0f226a98…", "head_commit": "21647280…",
          "compare_url": "https://github.com/key4hep/k4geo/compare/0f226a98…...21647280…",
          "status": "changed", "commits_unavailable": false, "truncated": false,
          "candidates": [
            {
              "repo": "key4hep/k4geo", "number": 1234,
              "title": "Lower the tracker step limit", "author": "…",
              "url": "https://github.com/key4hep/k4geo/pull/1234",
              "merged_at": "2026-07-04T…", "files": ["FCCee/ALLEGRO/…"],
              "additions": 20, "deletions": 4,
              "score": 72, "description": "raises the tracker step count, plausibly slower"
            }
          ]
        }
      ]
    }
  ]
}
```

Each entry's first seven fields are a `report.json` verdict's identity; the
dashboard joins an entry back to the confirmed regression it explains by that
identity **and** the `base_release`/`onset_release` window, so a sidecar left
over from an earlier build of the same night (the CI job also deletes the
remote sidecar on a rerun that produces none) can never attach to a re-anchored
regression. The pipeline collects every PR in each changed repo's commit range;
a separate **ranking stage** then scores each candidate *for that group* —
`score` is a 0–100 likelihood it is the cause and `description` a one-line
reason, judged once per detector/platform/sample/window (every metric — and
every benchmark-config label, e.g. a removal sweep's `baseline` vs.
`without_<detector>` — sharing that group and window shares one ranking,
applied to all of them). A *different* detector or sample sharing the same
release dates never shares a ranking. `commits_unavailable` marks a repo whose
range could not be enumerated at all; `truncated` marks a candidate list known
to be incomplete (compare/PR caps, or a PR that failed to fetch) — a regression
touching either is left **unranked**, since "most likely" over a partial
candidate set would overclaim. The ranking is a *lead* for a human, never a
claim of cause. Readers
drop unknown keys, so the schema can gain fields without breaking an older
dashboard; structurally malformed sidecars are hidden, never fatal.

The ranking stage is a **language model** that reads the metric that moved and
each candidate PR's actual code diff, and is configured entirely by environment —
`K4BENCH_LLM_URL`, `K4BENCH_LLM_MODEL`, `K4BENCH_LLM_API_KEY` and optional
`K4BENCH_LLM_MAX_TOKENS` (any OpenAI-compatible `/chat/completions` endpoint;
the model is a config value, not pinned in code). Transient connection, timeout,
HTTP 429 and HTTP 5xx failures retry with bounded backoff; length-truncated
responses grow the output allowance up to a fixed ceiling. The complete blame
stage also has a CI wall-clock limit. None of these failures can fail or delay
the already-uploaded nightly report beyond that bound.

Ranking is **optional**: with endpoint/model unset, candidates are still written
but left unranked (`score` 0, `description` ""), the dashboard shows the package
diff without the candidate ledger, and the email omits the "most likely" line.
When ranking *is* configured, CI publishes `blame.json` only if every candidate
of every fully-discovered regression has an explanation (a score of zero
remains valid; a likelihood that is not a number rejects that row rather than
becoming a fake 0%); an empty or partial model response is logged and the
sidecar is skipped rather than silently publishing a ranking the dashboard
would hide. The diffs the model reads are transient input, never
stored here (they are re-fetchable from GitHub); the sidecar keeps only the file
*paths* plus the ranker's `score`/`description`. The model may only score the
candidates it is given — a PR number it did not receive is dropped, so
`blame.json` can never surface an invented PR.

## See also

- [Analysis](../user-guide/features/analysis.md) — the loaders that parse these.
- [`RunResult`](api/results/model.md) — the CSV's source of truth.
- [Configuration reference](configuration-reference.md) — the YAML keys.
