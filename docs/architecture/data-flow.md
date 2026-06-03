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
        B->>G: temp files cleaned up
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
- Patched geometries live in temp files only for the duration of their run.

The instrumentation/physics split that makes this possible is described in the
[architecture overview](overview.md#guiding-principle-separate-instrumentation-from-physics).

## Nightly CI → EOS → dashboard { #nightly-eos-dashboard }

```mermaid
sequenceDiagram
    autonumber
    participant Cron as nightly (cron)
    participant Job as benchmark job
    participant K as k4bench
    participant EOS as CERN EOS
    participant Dash as dashboard

    Cron->>Job: expand .github/benchmarks/*.yml → matrix
    Job->>K: run (per detector/sample)
    K-->>Job: logs/<detector>/ (CSV + JSON + log)
    Job->>Job: write run_info.json + machine_info.json
    Job->>EOS: upload to {detector}/{platform}/key4hep-{release}/{sample}/{date}/
    Dash->>EOS: list + download runs over HTTPS (cached on disk)
    Dash->>Dash: load via k4bench.analysis; render tabs
```

The EOS directory layout is the integration contract between CI and the
dashboard — see [File formats → EOS layout](../reference/file-formats.md#eos-layout).
Because historical runs are immutable, the dashboard downloads each at most once
and publishes it into its on-disk cache atomically, so concurrent reruns never
see a half-written run.

## Deployment

The dashboard is containerised (`dashboard/Dockerfile`), pushed to `ghcr.io`,
and rolled out on CERN's OpenShift PaaS via the `openshift/` manifests by the
`deploy-dashboard.yml` workflow. It serves at
[k4bench-dashboard.app.cern.ch](https://k4bench-dashboard.app.cern.ch/).

## See also

- [Component diagrams](component-diagrams.md) — the static structure.
- [File formats](../reference/file-formats.md) — every artifact's schema.
