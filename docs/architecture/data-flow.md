# Data flow

How data moves through the system at runtime — first a single sweep, then the
nightly CI → EOS → dashboard pipeline.

## A sweep, end to end

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as cli
    participant B as benchmark
    participant G as geometry
    participant E as runner
    participant DD as ddsim (child proc)
    participant R as results

    User->>CLI: k4bench --xml … --sweep --ddsim-args="…"
    CLI->>B: run_sweep(config)
    B->>G: scan detectors
    B->>E: baseline run (original XML)
    loop per detector
        B->>G: patch geometry (temp XML, detector removed)
        B->>E: run (patched XML)
        E->>DD: /usr/bin/time -v ddsim …
        DD-->>E: stdout → log; optional plugin JSON
        E-->>B: RunResult
        B->>G: patch directory cleaned up
    end
    B-->>CLI: list[RunResult]
    CLI->>R: print summary + write CSV
    CLI-->>User: table + exit code
```

Key points along the path:

- The temporary EDM4hep output file is **reused** across runs — only its size is
  recorded.
- Run-level metrics are scraped from the `/usr/bin/time -v` block in the log;
  optional per-event / per-detector JSON is written directly by the
  [timing plugins](../user-guide/features/timing-plugins.md).
- Each patched geometry lives in one private temp directory only for the
  duration of its run.

The instrumentation/physics split that makes this possible is described in the
[architecture overview](overview.md#guiding-principle-separate-instrumentation-from-physics).

## Nightly CI → EOS → dashboard { #nightly-eos-dashboard }

```mermaid
sequenceDiagram
    autonumber
    participant Cron as nightly (cron)
    participant Gate as resolve-release job
    participant Job as benchmark job
    participant K as k4bench
    participant EOS as CERN EOS
    participant Reg as regression-report job
    participant Mail as CERN e-group
    participant Dash as dashboard

    Cron->>Job: expand .github/benchmarks/{suite}/{concept}/*.yml → matrix
    Cron->>Gate: wait for today's Key4hep stack on CVMFS
    Gate-->>Job: release every job of this night sources
    Job->>K: run (per config/sample, from benchmark_job.py)
    K-->>Job: logs/benchmark/ (CSV + JSON + log)
    Job->>Job: write run_info.json + machine_info.json
    Job->>EOS: upload to runs/{suite}/{concept}/{detector}/{platform}/key4hep-{release}/{sample}/{date}/
    Reg->>EOS: pull trailing sim run window per (detector, platform, sample)
    Reg->>Reg: k4bench.regression: reliability-filtered step detection
    Reg->>EOS: upload reports/sim/{date}/report.json
    Reg-->>Mail: email on confirmed regressions/failures only
    Dash->>EOS: list runs, download the files a view reads over HTTPS (cached on disk)
    Dash->>Dash: load via k4bench.analysis; render tabs
    Dash->>EOS: fetch reports/sim/{date}/report.json (Regressions tab)
```

The `resolve-release` job (`.github/scripts/resolve_release.sh`) names one
Key4hep release per night and the whole fan-out sources exactly that release, so
samples of the same night can never be filed under two releases. A view counts
only once its build has finished: LCG can publish a view before the packages it
links are built, so the gate also requires the slot's
`nightlies/…/isDone-<platform>` marker, written no later than the view. When no
complete stack is published for the day it falls back to the newest complete one
and says so in the job summary; the night still runs, because a repeat
measurement of a release is what `k4bench/regression/engine.py` uses to confirm a
WATCH.

The EOS directory layout is the integration contract between CI and the
dashboard — see [File formats → EOS layout](../reference/file-formats.md#eos-layout).
A run directory holds one attempt's files, and its `run_info.json` says they
are all there: the upload takes `run_info.json` down before it changes
anything, removes the files only an earlier attempt of the same job, release
and day wrote, and uploads `run_info.json` last (see
[File formats → Uploading a run](../reference/file-formats.md#uploading-a-run)).
A reader takes a run without it for one still being uploaded and reads it again
later (`k4bench.remote.upload_finished`), unless the run is two days past its
date, when no job can still be writing it.

A finished run is taken to be immutable, so the dashboard downloads each file of a
run at most once, and only when a view reads it (`k4bench.remote.ensure_run_artifacts`):
each file is published into its on-disk cache atomically, under a per-run lock
and manifest, so concurrent reruns never see a half-written file and two views
asking for different files of one run both get theirs. The nightly report and
the CLI download whole runs (`ensure_run_cached`), `run_info.json` last,
published with a single atomic rename of the run directory and marked
`.complete`. A same-day re-run that replaces a run already in a cache is not
seen there until that cache is cleared; the nightly report starts from an empty
one every night, and the dashboard's goes with its pod.

The `regression-report` job (`.github/scripts/regression_report.{py,sh}`) runs
after the benchmark fan-out, so a crashed detector job still surfaces in the
report — its missing upload *is* the failure signal. It judges the suite its
`suite` input names, reading that suite's run tree alone and writing that
suite's report; the nightly runs it once for each suite its fan-out
benchmarked. A fan-out where *every*
job failed, or that never started, uploads nothing at all and would otherwise
republish the previous night, so `nightly.yml` passes its own Actions run id:
a report naming none of that run's measurements is reported as tonight's
outage instead (see
[File formats → The night a report covers](../reference/file-formats.md#the-night-a-report-covers-reportjson)).
A job that failed before writing any result but uploaded its `run_info.json`
still names its CI run there, so its triple is reported as that fan-out's
failed configs, never as an upload that did not happen.

The report's EOS reads are strict: a listing or download WebEOS does not answer is
retried with backoff (`READ_BACKOFF_S`, within a per-report
`READ_BACKOFF_BUDGET_S`), all reads stop after `READ_DEADLINE_S` or
`READ_BREAKER_TRIPLES` failed triples in a row, and a triple whose runs still
cannot be read is reported as *not judged* rather than judged on part of its
history. A fan-out that unread triples might hide is not reported as an
outage: nothing is published (see
[File formats → Triples a report could not read](../reference/file-formats.md#triples-a-report-could-not-read-reportjson)).
It reuses the same pure building blocks as the dashboard (`k4bench.remote` for
discovery/download, `k4bench.analysis.trend` for aggregation,
`k4bench.results.reliability_evidence` for the per-run reliability verdict) and
adds the step detector in `k4bench/regression/engine.py`. The precomputed
`reports/sim/{date}/report.json` is what the dashboard's Regressions tab renders —
it never recomputes verdicts live.

## Deployment

The dashboard is containerised (`dashboard/Dockerfile`), pushed to `ghcr.io`,
and rolled out on CERN's OpenShift PaaS via the `openshift/` manifests by the
`deploy-dashboard.yml` workflow. It serves at
[k4bench-dashboard.app.cern.ch](https://k4bench-dashboard.app.cern.ch/).

## See also

- [Component diagrams](component-diagrams.md) — the static structure.
- [File formats](../reference/file-formats.md) — every artifact's schema.
