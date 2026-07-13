# Dashboard

The k4Bench dashboard is a [Streamlit](https://streamlit.io/) app that browses
benchmark history collected by the nightly CI. It's deployed at
**[k4bench-dashboard.app.cern.ch](https://k4bench-dashboard.app.cern.ch/)** and
also runs locally against your own `logs/` directory.

Its internals are documented under
[Architecture → dashboard](../../architecture/data-flow.md#nightly-eos-dashboard);
this page is the *user* guide.

## Purpose

Looking at one run tells you today's cost. The dashboard answers the questions
that need *history* and *comparison*:

- How does each detector's cost compare in the latest run?
- Has a Key4hep release made simulation slower or hungrier?
- How does per-event time/memory drift across releases?

## Where the data comes from

Every night, CI runs a curated set of benchmarks (defined in
`.github/benchmarks/*.yml`), uploads each result to CERN EOS, and the dashboard
reads them over HTTPS from a WebEOS endpoint. The EOS layout is hierarchical:

```text
{detector}/{platform}/key4hep-{release}/{sample}/{YYYY-MM-DD}/
    run_info.json  machine_info.json
    {config}_results.csv  {config}_events.json  {config}_regions.json  {config}.log
```

The sidebar lets you drill down through that hierarchy: **detector → platform →
Key4hep release → physics sample → run date**.

!!! note "Example detectors"
    A couple of detectors (currently: `SiD`) come from a simulation toolkit's
    own reference/tutorial geometry rather than a maintained FCC/Key4hep
    experiment design. Picking one of these shows a note in the sidebar
    linking back to its source, so it doesn't read as "just another
    experiment" (see `EXAMPLE_DETECTORS` in
    [`dashboard/ui_chrome.py`](../../architecture/data-flow.md)).

## The tabs

| Tab | What it shows | Backed by |
| --- | --- | --- |
| **Overview / Impact** | run-level metrics for the selected run; per-detector impact vs baseline | `*_results.csv` |
| **Event timing** | per-event wall time distributions | `*_events.json` |
| **Event memory** | per-event RSS and growth | `*_events.json` |
| **Region timing** | per-subdetector stepping time, `at_location` vs `by_birth`, step counts, attribution analysis | `*_regions.json` |
| **Trends** | metrics over time across releases | many runs, windowed |
| **Regressions** | the nightly cross-detector regression report | `_reports/{date}/report.json` |
| **Overview** | cross-detector metric comparison, snapshot & history | `_reports/{date}/report.json` |
| **Machine info** | the host the benchmark ran on (CPU, RAM, governor, throttling) | `machine_info.json` |
| **Logs** | the raw `ddsim` log for the selected run | `*.log` |

### Region timing tab

The richest tab. It mirrors the [region plugin's](timing-plugins.md#per-region-timing)
two attribution views (`at_location` and `by_birth`) and adds:

- **Current run** — per-detector time for the selected run.
- **Attribution analysis** — the gap between the two views (intrinsic vs
  imported cost).
- **Step analysis** — timer-interval counts per detector.
- **Historical** — the same detector's time across runs.

### Trends tab

Plots a metric over time. The x-axis is anchored on the **Key4hep release date**
(falling back to the run date), so you see regressions aligned with releases.
A sidebar **look-back window** (`Last 7 days`, `30`, `90`, `6 months`, `All`, or
a custom range) controls how much history is downloaded and shown. The window is
anchored on the *latest available* run, not today, so it always shows data even
if the nightly hasn't run recently (see
[`trend_window.resolve_window`](../../reference/api/analysis/index.md) — pure
logic, unit-tested).

Nights the nightly regression detector **confirmed** a step are ringed in red
on the lines, and nights it flagged but hasn't confirmed are ringed as ⚠️ watch
points — both on by default, each behind its own toggle. It's the same
Confirmed/Watch toggle and marker language as the Overview tab, so a flag here
means exactly what it means there. Flags appear on the metrics the engine judges
(wall time, user CPU, peak RSS) plus throughput, which borrows the wall-time
verdict since throughput is exactly `n_events / wall_time_s` (the same
regression, inverted); CPU efficiency and context switches carry none. Runs that
failed the host-reliability check are excluded by default with the same
warning/toggle as every other historical view.

!!! tip "Warmup is excluded"
    Trend and summary statistics drop event 0 (warmup), matching the
    [analysis convention](analysis.md#warmup-events).

### Regressions tab

Where the Trends tab lets you *look for* regressions, this tab shows what the
nightly detector *found* — across **all** detectors at once, independent of the
sidebar selection. Every night, CI (the `regression-report` job in
`nightly.yml`) walks the full EOS history and judges every
`(detector, platform, sample, config, metric)` series with a conservative
step-change detector (`k4bench/regression/`):

- **Baseline.** Each night is compared against the median and spread (scaled
  MAD, not mean/stddev, so one bad night can't skew it) of the trailing **14
  reliable** runs before it. Nights flagged by the
  [machine-info reliability check](#the-tabs) are excluded outright and never
  judged themselves. With fewer than **7** reliable runs to compare against, the
  metric is left *unknown* rather than flagged.
- **What trips a flag.** The value has to clear *two* gates at once: a robust
  **z-score above 3.5** (a statistical outlier) **and** a **practical-effect
  floor** — at least 5 % for time and memory, 3 percentage points for CPU
  efficiency, and a wider floor for the noisier region metrics. Requiring both
  keeps a very steady metric (tiny MAD) from flagging on a change too small to
  care about.
- **Watch, then regression.** The first night to clear both gates is a
  **⚠️ Watch**. It only becomes a confirmed **🔴 Regression** once the *next*
  reliable night moves the same way again — a two-strike rule that is the main
  defence against false alarms.
- **Re-anchoring.** A confirmed regression is treated as a **change-point**: the
  baseline resets to the new level, so a deliberate step (say, a physics change)
  is flagged exactly once instead of every night until the window rolls over.
  A second change right afterwards is still caught.
- **Direction** (faster/slower, more/less memory) is shown but not treated as
  good or bad — a regression is simply any confirmed step beyond the baseline in
  either direction.
- **Failures.** A config exiting non-zero, or a whole run missing, is a
  **❌ Failure** and skips the confirmation step (it alerts immediately).
- **Region timing** (sub-detector) is *not* flagged — it is the noisiest series
  in the system, dominated by timer-granularity wobble at microsecond scale, so
  the regression report judges only top-level run and per-event metrics.

The tab shows one report per night (pick earlier nights from the selector): an
at-a-glance banner, one expander per detector (collapsed by default — the badge
tells you which need attention), a **change ledger** — a compact, sortable table
of tonight's flagged metrics (a 🔴/⚠️ severity badge, the config, the metric, an
↑/↓ direction, and a Δ-vs-baseline magnitude bar), worst first — and a **Show
trend** drill-down that plots the metric's recent history with the baseline band
it was judged against (the flagged night marked 🔴, the night it was first
watched marked ⚠️). Confirmed regressions and failures — and only those — are
also emailed to the team's e-group by the same CI job.

### Overview tab

Where every other metric tab compares configs *within* the selected detector,
this one compares the detectors *against each other* — always on their
**baseline** config, for the sidebar-selected platform and sample, over the
sidebar's trend window (the same scoping as Run Trends, minus the detector).
It reads the same nightly `_reports/{date}/report.json` as the Regressions
tab — whose verdicts carry the raw nightly value of every run and per-event
metric for all detectors — so the whole comparison loads from one small JSON
per night, with no per-detector run downloads.

Two figures, each with its own legend below it (one colour per detector,
consistent across both):

- **historical trends** — the two selected metrics side by side (CPU,
  Memory), one line per detector across every nightly tag in the sidebar's
  trend window (x-axis: the Key4hep nightly tag, like every other trend
  view; a nightly benchmarked twice collapses to one point, newest run wins),
  so cross-detector gaps can be tracked over time. Nights the
  regression detector **confirmed** a step are ringed in red on the lines, and
  nights it flagged but hasn't confirmed are ringed as ⚠️ watch points — both
  on by default, each behind its own toggle. A *relative* toggle
  rescales each line to its first night = 100 %, making drift comparable
  across detectors of very different absolute cost;
- **performance landscape** — the selected time metric against the selected
  memory metric, one point per detector — closer to the origin is faster
  *and* leaner. The metric selectors offer mean/median event time, wall time
  or user CPU for time; mean event RSS or peak RSS for memory (shown in GB),
  and the selection is shareable via `?tmetric=`/`?mmetric=`.

Colours follow the detector *family* (ALLEGRO, CLD, …), with versions of one
family distinguished by dash pattern and marker symbol, so experiment-level
comparisons read at a glance. Runs that failed the host-reliability check are
excluded by default with the same warning/toggle as every other historical
view (the nightly report carries each night's per-detector verdict); their raw
values are still recorded — as unjudged points, never flagged — so disabling
the toggle plots them like Run Trends does. Value
axes are logarithmic by default (a toggle switches to linear) — the detectors
span more than a decade in both time and memory, so a linear scale squashes
the small ones into an unreadable cluster.

Detectors are only compared like-for-like: anything not benchmarked with the
selected sample/platform is listed as excluded instead of silently plotted
against a different workload. The values shown are the raw nightly
measurements the regression engine judged — baselines and Δ-verdicts stay in
the Regressions tab.

## Running the dashboard locally

You can point the dashboard at either a local directory or the remote WebEOS
endpoint, via environment variables read by
[`dashboard/config.py`](../../architecture/data-flow.md):

| Variable | Default | Meaning |
| --- | --- | --- |
| `K4BENCH_DATA_DIR` | `logs` | local directory to read runs from |
| `K4BENCH_DATA_URL` | *(unset)* | WebEOS base URL; when set, overrides local reads |
| `K4BENCH_CACHE_DIR` | `$TMPDIR/k4bench_cache` | on-disk cache for downloaded runs |

=== "Against your own runs"

    ```bash
    pip install -r dashboard/requirements.txt
    cd dashboard
    K4BENCH_DATA_DIR=../logs/ALLEGRO_o1_v03 streamlit run app.py
    ```

=== "Against the hosted data"

    ```bash
    cd dashboard
    K4BENCH_DATA_URL=https://k4bench-data.web.cern.ch streamlit run app.py
    ```

The dashboard imports `k4bench.analysis` for its loaders, so a working k4bench
install is required.

## How downloads are cached

Historical runs are **immutable**, so the dashboard downloads each run at most
once into `K4BENCH_CACHE_DIR` and reuses it across reruns. Downloads are staged
in a temp dir and published with a single atomic `rename`, so a reader never
sees a half-written run and an interrupted download leaves no partial run
behind. Filenames from the directory listing are validated (decoded, then
rejected if they aren't a single plain path component) to prevent path-traversal.
Details: [Architecture → data flow](../../architecture/data-flow.md).

## Deployment

The dashboard is containerised (`dashboard/Dockerfile`), pushed to `ghcr.io`,
and deployed to CERN's OpenShift PaaS via `openshift/` manifests
(Deployment/Service/Route) by the `deploy-dashboard.yml` workflow. The full
deployment path is in
[Architecture → data flow](../../architecture/data-flow.md#deployment).
