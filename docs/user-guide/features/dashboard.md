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

The bar at the top switches between two pages, one per benchmark suite:
**Simulation** (ddsim, the default page at `/`) and **Reconstruction** (k4run,
at `/reconstruction`). Each page has its own sidebar, sections and selections,
and reads only its own suite's runs. Most of this guide describes the
Simulation page; the
[Reconstruction page](#reconstruction-page) section covers the other one.

## Where the data comes from

Every night, CI runs a curated set of benchmarks (defined in
`.github/benchmarks/{suite}/{concept}/{detector}.yml`), uploads each result to
CERN EOS, and the dashboard reads them over HTTPS from a WebEOS endpoint: the
`sim` suite's on the Simulation page, the `reco` suite's on the Reconstruction
page. The EOS layout is hierarchical (see
[File formats → EOS layout](../../reference/file-formats.md#eos-layout)):

```text
runs/sim/{concept}/{detector}/{platform}/key4hep-{release}/{sample}/{YYYY-MM-DD}/
    run_info.json  machine_info.json
    {config}_results.csv  {config}_events.json  {config}_regions.json  {config}.log
```

The sidebar lets you drill down through that hierarchy: **detector → platform →
physics sample → Key4hep release**. Single-run tabs use the newest uploaded
run for that selection; the trend window controls the multi-run views.

Each level is rebuilt from the one above it, so changing the detector can leave
the selected platform or sample with no match — in which case it re-defaults to
a valid one and says so right under the selector. That notice matters because
the Overview tab is scoped by platform and sample *across all detectors*:
without it, a sample that quietly moved would look like detectors vanishing
from its charts.

Not every tab honours the whole hierarchy, so a single muted line directly
under the section bar names the slice the tab in front of you actually uses —
for example `Showing CLD — AlmaLinux 9 · GCC 14.2.0 (optimized) — Single e⁻ ·
10GeV — 2026-07-10`. Each level is written the way the tabs and the nightly
mail write it, so a sample reads as its physics rather than as its EOS
directory. Where a tab deliberately ignores a level it says so
instead of quietly dropping it: Overview reads **all detectors**, Run Trends
reads **all releases in the trend window**, and Stack Changes says its package
diff is **platform-wide**, because a Key4hep release is the same stack whatever
benchmarked it. The line also follows the controls inside a tab that change its
scope: switching to a **Historical Trends** view replaces the release with *all
releases in the trend window*, and switching on Stack Changes' **Whole platform**
toggle replaces the detector and sample with *all detectors* and *all samples*.
The line never repeats a date range, report night or run date — each tab prints
its own, in its own words. In local mode there is no hierarchy to show, so it
names the run directory instead.

!!! note "Example detectors"
    A couple of detectors (currently: `SiD`) come from a simulation toolkit's
    own reference/tutorial geometry rather than a maintained FCC/Key4hep
    experiment design. Picking one of these shows a note in the sidebar
    linking back to its source, so it doesn't read as "just another
    experiment" (see `EXAMPLE_DETECTORS` in
    [`dashboard/ui_chrome.py`](../../architecture/data-flow.md)).

## The tabs

The Simulation page's sections:

| Tab | What it shows | Backed by |
| --- | --- | --- |
| **Config Impact** | baseline-relative configuration impact in the selected run | `*_results.csv` |
| **Event timing** | per-event wall time distributions | `*_events.json` |
| **Event memory** | per-event RSS and growth | `*_events.json` |
| **Correlations** | each event's time, and each detector's time and steps, against what the event holds: its generator particles and hits | `event_source.json.gz`, `*_event_size.json`, `*_events.json`, `*_regions.json` |
| **Event size** | what each event of the output file takes on disk after ROOT's compression, collection by collection | `*_event_size.json` |
| **Region timing** | per-subdetector stepping time, `at_location` vs `by_birth`, step counts, attribution analysis | `*_regions.json` |
| **Trends** | metrics over time across releases | many runs, windowed |
| **Regressions** | the nightly regression report for the selected detector/platform/sample | `reports/sim/{date}/report.json` |
| **Stack Changes** | which Key4hep packages moved between two nightly releases | `run_info.json` (`k4h_packages`) |
| **Overview** | cross-detector comparison: performance trends, landscape, regression status & the nightly mail | `reports/sim/{date}/report.json` |
| **Machine info** | the host the benchmark ran on (CPU, RAM, governor, throttling) | `machine_info.json` |
| **Logs** | the raw `ddsim` log for the selected run | `*.log` |

### Display options

Anything that changes *how* a figure is drawn — colour palette, opacity, style
cycling, smoothing, histogram bins, error bars, mean lines, how many detectors
a ranking keeps — lives in a **👁️ Display options** popover at the right-hand
end of the view's control row. It looks and behaves the same in every view that
has one (Event Timing, Event Memory, Region Timing's four views
and Run Trends), and each popover ends with **Reset to defaults**, which
restores every control inside it in one click.

What stays on the page is everything that changes *what* you are looking at:
baseline, configuration, attribution, metric, view, report night, the
reliability scope and the regression flag pills. Overview's Log/Linear/Relative
% scale stays out on the page for the same reason — it changes what the numbers
mean, not how they are painted. Whenever unreliable runs are present, the
shared **Runs · ⚠ N unreliable** selector occupies the right-hand end of that
view's existing controls line (beside Display options where present); it never
inserts a separate warning or controls row.

The controls that stay on the page are also in the page URL, so a copied link
reopens what was on screen: the section, its view, the configuration and the
other choices next to them. Display options are not in it, so a link draws the
figures the default way; neither is the trend window (see below), nor a row
selected in a table or a node clicked in a graph, nor Event Timing's **Timing
of** algorithms, whose names carry the run's times.

Tabs with multiple views use the same lightweight selector: horizontal radio
buttons under a bold **View** label. The segmented bar is reserved for the
dashboard's top-level section navigation and for compact value choices such as
Scale; it is not repeated as tab-local navigation.

Palette defaults follow the data: a figure picks the smallest Matplotlib tab-N
palette that colours every series without repeating a colour, and re-picks it
when the number of series crosses a boundary. An explicit choice sticks until
that happens.

### Config Impact tab

The five summary cards name the alternative configuration with the largest
estimated impact on wall time, peak virtual memory, peak RSS, user CPU and
output size. Pick a metric to open its horizontal ranking: the selected
baseline is the zero line, better-than-baseline changes extend right in blue,
and worse changes extend left in orange. Positive has the same meaning for
every metric — less
time/memory/CPU/output, or more throughput. The default **Full detector**
(`baseline`) comparison is the usual subdetector-removal study; the wording
remains baseline-relative if you select another reference. Throughput remains
available in the metric selector as the rate-oriented view of wall time. Hover a
bar for rounded raw measurements and their absolute change. The compact ranking
keeps the 12 largest absolute impacts, so it retains both major gains and major
regressions. When more than 12 configurations are comparable, switch on **All
configs** beside it to expand the chart.

With the full-detector baseline, the removal impacts are ablation estimates
rather than additive accounting. Removing material changes particle transport,
so the individual percentages need not add up to the cost of the full detector.
For a direct view of where simulation time is spent, **Region Timing** attributes
per-subdetector Geant4 stepping time both to where tracks run and where they were
created.

### Region timing tab

The richest tab. It mirrors the [region plugin's](timing-plugins.md#per-region-timing)
two attribution views (`at_location` and `by_birth`) and adds:

- **Current run** — per-detector time for the selected run; **Top N detectors**
  (in Display options) sets how many of the most expensive regions it ranks.
- **Attribution analysis** — the gap between the two views (intrinsic vs
  imported cost).
- **Step analysis** — timer-interval counts per detector.
- **Historical** — the same detector's time across runs.

### Correlations tab

What makes an event expensive. The baseline configuration's per-event costs
(or those of a benchmark's single configuration) are set against what each
event holds:

- its generator particles, from the run's `event_source.json.gz`: canned
  quantities such as the charged final-state particles, those born more than
  0.5 mm off the beam axis, the highest pT or the visible energy, and, for a
  gun sample of one particle per event, that particle's |cos θ|, θ, φ and
  energy;
- what each collection holds in the event (`n_<collection>` elements and, for
  calorimeter hits, `E_<collection>` energy): simulation's output, or
  reconstruction's input, and reconstruction's outputs as `out_n_…`,
  `out_E_…`.

A simulation benchmark records what its events hold for its first run only;
its other configurations take a detector out and are not offered. A run
recorded before this existed has nothing to show, unless its source was added
later; a note then says whether it came from the run's own input or was
inferred from another detector's simulation of the same sample.

Top to bottom:

- **Metric**: the event's time; in simulation a detector's time or steps; in
  reconstruction an algorithm's time, CPU time, instructions, allocations,
  allocated bytes or peak heap.
- A matrix of the costliest 15 costs of that kind (down) against the
  quantities most associated with them (across, by the names an expression
  uses). Each square is coloured by Spearman's rank correlation ρ over the
  events (red, the cost rises with the quantity; blue, it falls) and sized by
  its strength; hovering says what the quantity is and over how many events.
  Click a square to plot it. A cost that depends on a quantity symmetrically or
  periodically (on θ forward and backward alike, on φ) shows little however
  strongly it depends on it, which is what the plot is for, and among this many
  pairs a few strong ones turn up by chance: a square is a lead, not a
  significance.
- One row for the plot: **Of**, the cost; **Against**, an expression, always
  on show: a quantity's name (`n_VertexBarrelCollection`) or any per-event
  combination of the generator particles and the quantities, e.g.
  `count(final & charged & (pt > 1))`, filled at first with the quantity most
  associated with the cost; and **Log y**.
- **What can I write?** offers examples that work for the run on screen and
  lists every name an expression can use there: the run's quantities (what each
  is, its range in this run, how a canned one is written), the particle fields
  with their units (used inside `count(mask)`, `sum`, `mean`, `max` or
  `min(x, mask)`, which fold an event's particles into one number), and the
  reductions and operators. An expression is parsed and computed piece by
  piece, never run as Python, and is limited in length and size.
- The plot, with its robust trend line (Theil–Sen), and, folded under it, the
  events more than three robust standard deviations above the line (when most
  events lie exactly on it, as a count that rarely changes does, every event
  above it): their event number, the line's value and how far above it they
  lie. Above that
  line, which is not to say expensive for everything they hold.

The warm-up event is left out. The events are matched to what they hold by
their position in the job's files; a run whose profiling counted other events,
or whose files' EventHeaders name other events, says so rather than matching
them. A reconstruction run without its input recorded shows no correlations,
since nothing shows its outputs kept the order the job read its events in, and
an output without an EventHeader in every event, or whose input lacks one, is
left out, which the page notes. See
[File formats](../../reference/file-formats.md#event-source-json).

### Event Size tab

What each event of a configuration's output takes on disk after ROOT's
compression, for planning production storage: on the Simulation page ddsim's
output file, on the Reconstruction page each EDM4hep file the k4run job wrote
(an **Output file** selector appears when there are several, the largest
first; files without EDM4hep events, such as AIDA histograms, are listed under
the table with their size).

- **Current Run** shows the file's **event size** (its events tree's compressed
  bytes over its events), its **file size** on disk, its **compression** (the
  events' uncompressed over compressed bytes, with the ROOT setting the job
  wrote it with, e.g. ZLIB level 1) and its **largest collection**'s share.
  Two donuts split the event by collection and by EDM4hep type, the six
  largest each and the rest as *Other*. The table below gives every collection, largest first, with its type, compressed and
  uncompressed kB per event, its share of the event and its compression ratio;
  a collection's bytes are those of all its branches, its relations and vector
  members included. *Frame parameters* are podio's per-event parameters.
- **Historical Trends** (remote mode) stacks the event size night by night, the
  largest collections over the window each a series of their own (**Top N
  collections** in Display options) and the rest as *Other*, or draws a line per
  collection; **Show** switches between kB per event and each collection's share
  of the event. Sizes do not depend on how loaded the host was, so this view has
  no run-quality filter.

The numbers are the benchmark's own files: tens to a few thousand events,
written with the job's compression setting. A production file of many more
events fills ROOT's baskets further, which can compress a little differently.
See [File formats](../../reference/file-formats.md#event-size-json).

### Trends tab

Plots a metric over time. The x-axis is anchored on the **Key4hep release date**
(falling back to the run date), so you see regressions aligned with releases.
A sidebar **release slider** controls how much history is downloaded and shown:
every run of the Key4hep releases between its two handles, including a release
re-run on later nights. What it ranges over follows the open tab. On the
cross-detector **Overview** it is **every release benchmarked on the platform**,
by any detector, so the same range covers the same releases whichever detector
the sidebar has selected, and switching detector keeps it: a range over one
detector's releases would shrink to that detector's history whenever a recently
added one was picked, and detectors that had run for longer would silently lose
their history from its charts. On every other tab it is the releases the
**selected detector** was benchmarked with, so after a detector version bump the
slider holds only the new version's releases rather than a range it has no runs
in. The slider opens on the **10 newest releases**. On the Overview a release
is offered once a nightly report has judged it: the Overview draws the reports,
so a release tonight's batch is still uploading has nothing to show there
until its report lands. On the other tabs it is offered as soon as the selected
detector's run is uploaded. When a new release becomes available, or the tab or
detector changes what the slider ranges over, it reopens on the newest releases
rather than staying on a range picked for something else. The window is not
part of the page URL. (Its resolution lives in [`trend_window`](../../reference/api/analysis/index.md) — pure logic,
unit-tested.)

Nights the nightly regression detector **confirmed** a step are ringed in red
on the lines, and nights it flagged but hasn't confirmed are ringed as ⚠️ watch
points — both on by default, each behind its own toggle. It's the same
Confirmed/Watch toggle and marker language as the Overview tab, so a flag here
means exactly what it means there. Flags appear on the metrics the engine judges
(wall time, peak virtual memory) plus throughput, which borrows the wall-time
verdict since throughput is exactly `n_events / wall_time_s` (the same
regression, inverted); user CPU, peak RSS, CPU efficiency and context switches
are recorded only and carry none. Runs that
failed the host-reliability check are excluded by default with the same
**Runs · ⚠ N unreliable** selector as every other historical view: **Reliable
only** (the default) or **All runs**. The **Data range** caption above the
chart spans the nightly tags actually plotted; a tag whose every run was
excluded is counted apart (`9 nightly tags · 1 hidden as unreliable`), so the
caption and the chart never disagree. Palette, style cycling, opacity
and line smoothing sit in this tab's [Display options](#display-options)
popover, alongside the pills on the same row.

Below the figure, a collapsed **Run-to-run variability** table answers "how much
does this metric normally move from run to run?" for peak virtual memory and
peak RSS. The Event Memory tab's historical view has the same table for mean
anonymous, total and file-backed RSS and anonymous RSS growth (MB/event), and
plots that growth rate as its own panel. Each row gives two readings per config:

- **Same-release spread (stack changes excluded)** — the typical change between
  consecutive runs of the *same* Key4hep release, over the last 14 such pairs
  in the window, with the number of pairs used. Those runs share the Key4hep
  stack, but not necessarily the k4Bench checkout: consecutive runs of one
  release are routinely benchmarked by different k4Bench commits, so a harness,
  plugin or configuration change between them counts here alongside measurement
  noise. It is the closest the history gets to a noise estimate, not noise
  alone. It reads N/A when fewer than 3 same-release pairs exist, and never
  falls back to the other column.
- **Recent movement (all runs, incl. stack changes)** — the same statistic
  over the last 7 runs regardless of release, so it also contains Key4hep stack
  changes. It describes recent movement; it is not the exact noise floor the
  regression engine applied to the newest release, which is computed before that
  release's own runs are added.

"Last" follows the regression engine's order — release date, then run — not
wall-clock time: a later rerun of an old release is counted with that release.

Both are the scaled median absolute successive difference (robust, so one
isolated step barely moves it), shown in the metric's unit and, for levels and
peaks, as a percentage of the median. Runs failing the host-reliability check
are always left out, whichever runs the selector shows, and missing values are
gaps, never zeros. Neither reading is an uncertainty or confidence interval, and
neither is stored: both are recomputed from the window on every render.

!!! tip "Warmup is excluded"
    Trend and summary statistics drop event 0 (warmup), matching the
    [analysis convention](analysis.md#warmup-events).

### Regressions tab

Where the Trends tab lets you *look for* regressions, this tab shows what the
nightly detector *found* — for the sidebar-selected detector, platform and
sample, the same scoping as every trend view (the cross-detector picture lives
in the [Overview tab](#overview-tab)). Every night, CI (the `regression-report`
job in `nightly.yml`) walks the full EOS history and judges every
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
- **The workload is pinned and recorded.** Timing measures software only if both
  nights simulated the same events, so the benchmarks fix the ddsim seed and
  each run records the seed it used in `run_info.json`. Without one, ddsim draws
  a fresh seed per run and every night re-rolls the shower physics. A change of
  the recorded seed — the migration to a pinned seed, a later change of the
  value — does **not** interrupt judging: the history stays one continuous
  series, and a level the new sample lands on is confirmed or not by the same
  two-strike rule and release-median gate as everything else, with a bounded
  onset window an attribution can be hung on. What the change gets is a
  **note on the report**, naming the old and new seed, so a group-wide move is
  not mistaken for a code change.
- **Every configuration reports what it measured.** When a whole run group
  moves together — a host or stack effect rather than a detector one — each
  config's row states the move it recorded, so the number in the report is the
  number in the run data and nothing has to be reconstructed to check it. The
  repetition is real: a group-wide move genuinely did happen to every config,
  and reading it once per config is the price of every row being reproducible.
- **Trimmed alongside total.** Timing is judged on the mean, the median *and* an
  **upper-trimmed mean** that drops the slowest 5 % of events — one-sided, so
  only the slow tail goes. It reports the typical event; the untrimmed totals
  stay judged because the tail is exactly where a tail-confined regression would
  appear.
- **One measurement, one test.** `user_cpu_s` tracks `wall_time_s` closely
  enough that judging both would count the same event twice, so it is recorded
  and plottable but never flagged.
- **Watch, then regression.** The first night to clear both gates is a
  **⚠️ Watch**. It only becomes a confirmed **🔴 Regression** once the *next*
  reliable night moves the same way again — a two-strike rule that is the main
  defence against false alarms. The confirming night may re-benchmark the same
  release as the watch night: a second run of the same binary tripping the same
  way is exactly the independent evidence the rule wants.
- **Releases, not nights.** The unit of change is the **Key4hep release**;
  nights that re-benchmark one release are repeat measurements of the same
  binary. Every night of a release is judged against the same frozen baseline,
  and once a step is confirmed for a release, every later night of that release
  that trips the same way is confirmed too, carrying the same onset window — so
  all of a release's reports agree on its regressions, modulo each night's own
  measured value. A repeat confirmation is labelled as such in the report and
  email ("repeat — first confirmed {night}"), so consecutive alerts read as
  "still present", not fresh news.
- **Confirmations follow the balance of evidence.** A single quiet night inside
  a confirmed release cannot outvote the two nights that confirmed — it is
  reported OK with a note that the release's median still sits beyond the
  baseline ("tonight's value looks like noise"). But when quiet nights drag the
  **median of the release's nights** back inside the gates, the confirmation is
  revoked ("confirmation revised") — the confirming nights were more likely
  noise — and the baseline is never re-anchored onto a fluke level.
- **Re-anchoring.** A confirmed regression that still stands at the next
  *release boundary* is treated as a **change-point**: the baseline resets to
  the new level when the next release arrives, so a deliberate step (say, a
  physics change) is flagged
  on the release that introduced it — on every night that measured that
  release — and falls quiet from the next release on, instead of alerting every
  night until the window rolls over. A second change right afterwards is still
  caught.
- **Late confirmations keep the right window.** A step too small to clear the
  gates while the baseline still holds an older level can clear them days
  later, once that level has rolled out of the baseline, with nothing new in
  the measurement. Its watch night is then not where the step landed: when the
  night the window would start from is itself flagged by the baseline that
  confirmed the step, the window is moved back to the first night already at
  the new level. The baseline median line still shows the level the step was
  judged against.
- **Continued series across a platform migration.** A platform is a separate
  series, so a new one would have no baseline for its first week. A platform
  that **replaces** another instead continues its series: the old platform's
  nights come first in the history, as ordinary baseline points. A shift caused
  by the migration is reported once, watch then regression, with a window from
  the old platform's last night to the new platform's first, and the baseline
  re-anchors on the new level. Once the new platform has run, the old one's
  absence is silence rather than a *no run uploaded* failure, and it sorts to
  the end of the sidebar's platform list so a fresh visit lands on one still
  running. Platforms benchmarked side by side stay independent series.
- **Continued series across a detector version switch.** A geometry version
  bump works the same way: a detector config that **replaces** another
  (`ALLEGRO_o1_v03` → `ALLEGRO_o1_v04`, listed in
  `k4bench/regression/lineage.py`) continues its series on the same platform,
  so the geometry change is reported as a regression with a window from the
  old version's last night to the new one's first. The ranker is told the
  window spans the switch, and k4Bench's own pull request that made it is among
  the candidates. The replaced version sorts to the end of the sidebar's
  detector list. Other options of a
  concept (`ALLEGRO_o2_v01`) stay independent series.
- **Direction** (faster/slower, more/less memory) is shown but not treated as
  good or bad — a regression is simply any confirmed step beyond the baseline in
  either direction.
- **Failures.** A config exiting non-zero, or a whole run missing, is a
  **❌ Failure** and skips the confirmation step (it alerts immediately).
- **Region timing** (sub-detector) is *not* flagged — it is the noisiest series
  in the system, dominated by timer-granularity wobble at microsecond scale, so
  the regression report judges only top-level run and per-event metrics.

A report is written per *run*, so several nights routinely re-benchmark one
fixed release. Their reports mostly agree — every night of a release is judged
against the same baseline — but can still differ: the first strike is only a
watch, a marginal night can come out OK, and reports built before the
release-grouped engine confirmed on a single night. The sidebar's **release**
selects which of that release's report nights are on offer; the tab defaults to
the most **attention-worthy** one (a confirmed regression or failure outranks a
watch outranks a quiet night, newest breaking ties), so the most alarming night
is never hidden behind a quieter one. A **Report night** pill always appears
above the report — one option carrying that night's glance badge
(❌ / 🔴 / ⚠️ / ✅) even when the release was only benchmarked once, and one pill
per night to step through the rest when it was benchmarked several times. The
newest release (the sidebar default) still surfaces the latest report even
when it is newer than the release's last run, so a "no run uploaded" failure
stays visible. The banner, flagged metrics, and trend preview follow the
selected night's historical evidence. Package and pull-request attribution is
instead **release-level**: it is fixed from the first report night that
confirmed a change window for the release and reused on every rerun, because
no upstream package changes between repeat measurements of the same binary.

A benchmark job is dated when it *starts*, so a nightly batch beginning near
midnight splits across two dates and some detectors are stamped a day behind
the report night. Those are still tonight's measurements and are reported as
such, with a note saying so — they are recognised by the CI run they came from,
which every detector of one nightly shares, so a detector whose job crashed and
uploaded nothing is still reported as a **missing run** rather than passing as
a straddling batch — once the report night names a CI run, only a detector
naming that same run is kept, whatever the gap between the two dates. Nights
recorded before that CI run existed fall back to accepting a single night's
lag, and their note says that is what happened. The Overview's charts follow
each run by the date it actually ran, so a straddling job is still covered by
the unreliable-run selector — it is one of tonight's runs, and the
date it is stamped with should not decide whether the filter can see it.

A `?report=YYYY-MM-DD` query parameter pins one report night directly and is
authoritative when valid — this is the stable deep link the nightly email and
the Overview roster generate, so a link to a confirmed regression keeps
pointing at the exact report the email described.

The selected run group renders flat: an at-a-glance banner (regressed / watch
/ failures / within-baseline counts) and a **trend preview** that plots a
flagged metric's recent history with the baseline band it was judged against
(the flagged night marked 🔴, the night it was first watched marked ⚠️), a few
releases further out when the release has since moved on. The window contains
up to **14 distinct Key4hep releases** through the flagged release plus up to
**7 later releases** when available; repeat measurements of one release stay
visible without consuming another release slot. Its dropdown lists every
flagged metric worst first, each option carrying its own severity badge and
Δ-vs-baseline — so scanning the list alone shows the size of every flag,
without a separate ledger table. The preview opens on the most severe flag
automatically; switch it to any other flagged metric, or to "—" to hide it.
Confirmed regressions and failures — and only those — are also emailed to the
team's e-group by the same CI job, with per-group links landing directly on
this scoped view, pinned (via `stack=` and `report=`) to the exact report
night they describe.

For a confirmed regression the drill-down also shades the **release window the
step entered in**. Because confirmation is a two-strike rule, the reported night
is one reliable night *after* the step appeared, so the cause is upstream of the
⚠️ onset, not at the 🔴 report — the amber band spans from the last release seen
at the accepted level up to the onset release. The **upstream-changes card**
below the preview (see below) is where to follow that window into the Stack
Changes tab. When both ends of the window are the *same* release the band
gives way to a "nothing upstream changed" note on that card: the stack did not
move across the step, so the cause is the host, the sample, or noise rather
than an upstream commit.

Below the drill-down, the release gets an **upstream-changes card** naming the
packages that moved in the blame window established by its first confirmed
report, each linking to its commit range, plus an **Open in Stack Changes →**
link that seeds that tab with the exact release range. That same card is shown
on every report-night view of a re-benchmarked release: a later run may provide
the second strike for another metric, but it cannot create a new package-change
story for unchanged software. When the canonical report contains several
distinct windows, each retains its own card. When the blame
[sidecar](../../reference/file-formats.md#blame-sidecar-blamejson)
carries a **ranking**, the card also lists **suggested candidate pull requests**:
each PR in the window with a 0–100% **Likelihood** it is the cause and a one-line
*Why*. The ranking is produced offline by a **language model** that read the
metric that moved and each PR's actual code diff (configured in CI via
`K4BENCH_LLM_*`; see the [sidecar format](../../reference/file-formats.md#blame-sidecar-blamejson)) —
the dashboard only displays the stored result. Several PRs can land in one
package's range, so each is scored on its own. All ranked leads are shown in
one sortable table.
The group is judged once
per detector/platform/sample/window: when several metrics (or several
benchmark-config labels, e.g. a removal sweep's `baseline` and
`no_<detector>` runs) stepped across the same release boundary in the
same run group, they share one diff and one candidate set, so the card shows a
single verdict rather than a table per metric. A different detector or sample
sharing the same release dates never shares a verdict. This is a ranked
**lead for a human, not a verdict** — a suggestion, not proof of cause, in
keeping with the
detector's *no evidence ⇒ no verdict* rule; the nightly email surfaces the same
top candidates under each regression. Most nights carry no `blame.json` at all,
and a night whose candidates are not yet ranked says that no AI ranking is
stored rather than leaving an ambiguous blank below the package diff.

### Stack Changes tab

Answers "what came in last night?" — and, when a metric has stepped, "what
upstream change could that be?". Pick two nightly tags and it lists the Key4hep
packages whose commit differs between them, each linking to the range on GitHub.
When you open the tab, **To release** defaults to the stack selected in the
sidebar and **From release** to the release immediately before it, when one is
available. A compact line below the range reports how many tracked packages
changed and remained unchanged without turning those counts into a second
dashboard banner.

The package diff is cross-detector: a Key4hep release is one stack, sourced
identically by every detector benchmarked against it, so only the platform
scopes it.

Below the diff, **Regressions in this range** lists the confirmed metrics whose
onset falls inside the selected release interval. It is scoped to the sidebar's
detector and sample. A **Whole platform (+N metrics)** toggle appears only when
widening would reveal confirmed metrics in other detector/sample scopes; a
package change can affect any detector that sources it.

The worst metric opens in a compact selector rather than a ledger. Every option
shows its severity, metric (including its region), config, Δ-vs-baseline and
exact **blame window**; in the all-detectors view it also names the detector and
sample. If distinct steps otherwise have the same label, their onset run is
shown too. The selected item
uses the same trend component as the Regressions tab: up to **14 releases** of
history and **7 later releases**, the accepted-baseline band, first-watch and
confirmation markers, unreliable-run filtering, and the shaded blame window.
This makes a multi-release cumulative diff safe to inspect without implying
that every package above belongs to every metric below. When that selected
metric's blame window is narrower than the package comparison, **Focus package
diff on …** resets From/To to the exact interval in one click.
If package provenance is unavailable, or the selected endpoint stacks are
identical, the section says so instead of describing the metric changes as
effects of a package diff that was not observed. For a cumulative range with
identical endpoints it also warns that an intermediate release may still have
moved.

When the selected regression has a ranked blame sidecar, its **AI-generated PR
ranking** appears directly below the trend. The same complete ledger and
"suggestion, not proof" framing are reused from the Regressions tab;
Stack Changes does not recompute or reinterpret the model output. When no
ranking was stored, the tab says so explicitly instead of leaving an ambiguous
blank.

The selected regression is part of the URL (`reg_*` parameters), including its
detector/sample identity, config, metric, region, exact onset run and whether
**Whole platform** is enabled. Sharing, reopening, or navigating back to the URL
therefore restores the exact investigation rather than the range's default
metric. An explicitly linked metric remains selectable even when it falls below
the normal worst-30 display cap. Historical report JSONs
for a cold range are fetched as one parallel batch; the selected trend still
downloads run data only when it is shown.

**It compares releases, not run dates.** The nightly build does not publish
every day; a benchmark then re-uses the newest release available, so several
consecutive run dates routinely share one identical stack — most run dates on
EOS are not release dates at all, and only dates that have a release are
selectable.

Picking two releases that are far apart gives the **cumulative** change across
every release in between, which the header states explicitly — a month-wide diff
looks no different from one night's in the table.

An empty diff is a result, not a blank: if two releases sit at the same commit
for every tracked package, nothing upstream changed between them, so a metric
that moved did so for another reason — the host, the sample, or noise.

The data comes from `k4h_packages` in each run's `run_info.json`, recorded from
CVMFS as the benchmark runs (see
[file formats](../../reference/file-formats.md#stack-provenance-k4h_packages)).
Releases benchmarked before provenance capture — or whose stack had already aged
off CVMFS when the history was backfilled — cannot be compared, and the tab says
so rather than showing an empty diff.

### Overview tab

Where every other metric tab compares configs *within* the selected detector,
this one compares the detectors *against each other* — always on their
**baseline** config, for the sidebar-selected platform and sample, over the
sidebar's trend window (the same scoping as Run Trends, minus the detector —
and the window is anchored detector-independently, so switching the sidebar
detector never changes which detectors this tab shows).
It reads the same nightly `reports/sim/{date}/report.json` as the Regressions
tab — whose verdicts carry the raw nightly value of every run and per-event
metric for all detectors — so the whole comparison loads from one small JSON
per night, with no per-detector run downloads.

Four views, dispatched by the same View switcher as the other multi-view tabs
(one colour per detector, consistent across the figures). The selected view
rides in the URL as `?view=`, so a copied link reopens the view it came from
along with the parameters only that view reads:

Like every tab-local view selector in the dashboard, the Overview switcher uses
simple horizontal radio buttons under a bold **View** label. It is visually
separate from a single bordered comparison toolbar. **Time**, **Memory** and
**Scale** form its left-hand group; **Regressions** and the explicit
**Runs · ⚠ N unreliable** choice form its right-hand group on the same row.
When unreliable runs are present, that choice offers:
**Reliable only** (the default) or **All runs**; affected dates and the Machine
Info pointer live in the control's help instead of a separate banner. The rows
wrap naturally on narrow screens, with no CSS override. Each detector family
has its own vertical legend below the figure: the family is the heading and its
versions sit directly beneath it, while every version remains individually
toggleable. Legend columns are centre-anchored and move into additional rows
when there are many families or unusually long labels, keeping the plot usable
at narrower dashboard widths.

- **Performance Trends** — the two selected metrics side by side (CPU,
  Memory), one line per detector across every nightly tag in the sidebar's
  trend window (x-axis: the Key4hep nightly tag, like every other trend
  view; a nightly benchmarked twice collapses to one point, newest run wins),
  so cross-detector gaps can be tracked over time. Nights the
  regression detector **confirmed** a step are ringed in red on the lines, and
  nights it flagged but hasn't confirmed are ringed as ⚠️ watch points — both
  on by default, each behind its own toggle. The **Scale** switcher's
  *Relative %* option rescales each line to its first night = 100 %, making
  drift comparable across detectors of very different absolute cost; it sits on
  the page rather than in Display options because it changes what the numbers
  mean. Any scoped detector the
  chart has **no line for** is named with the reason — every run excluded as
  unreliable · no value for the selected metrics · ran but produced no
  comparable metrics (a hard-failed config) · no run in the window, with the
  night it last ran · not benchmarked with this sample/platform — so a missing
  detector doesn't read as a quiet one. The roster comes from the nightly
  reports' run *groups*, not from their metric values, which is what lets a
  detector whose config failed outright be named at all. It is stated whether or
  not there is a chart, since a window with nothing to draw is when it matters
  most. One gap remains: a detector retired past the engine's seven-day
  missing-run grace period is dropped from the reports entirely and cannot be
  named — widen the window to a range it still ran in to see it;
- **Performance Landscape** — the selected time metric against the selected
  memory metric, one point per detector at its **most recent run** — closer to
  the origin is faster *and* leaner. It follows the runs rather than the
  reports, so it is unaffected by the Regression Status night on screen, and a
  detector that missed the newest night keeps its last measured point (the
  caption and the hover name its nightly tag) instead of dropping off the
  chart; the same fallback applies to a run excluded by the reliability
  toggle; a detector that ran but produced no usable point — a hard-failed
  config, or one whose every run the toggle dropped — is named in the caption
  rather than left out. That last run is found even when it falls outside the trend window
  — a detector that has been missing for a few nights is shown at the night it
  really last ran, not at wherever the window happens to end. Both coordinates
  always come from one run, reruns of a nightly tag included. The metric selectors
  offer mean/median/trimmed-mean event time, wall time or user CPU for time;
  mean event anonymous RSS, peak virtual memory, mean event RSS, peak RSS or
  mean event file-backed RSS for memory (shown in GB), are shared with the
  trends view, and
  the selection is shareable via `?tmetric=`/`?mmetric=`;
- **Regression Status** — since the Regressions tab is scoped to one
  detector, the **cross-detector regression picture lives here**: a **report
  night picker** (every fetched night that covers the scope, newest first,
  each badged with its worst cross-detector state; defaults to the newest,
  pinnable via `?report=` — the same parameter the roster links and alert
  emails carry, and the trend window sets how far back it reaches; changing
  the sidebar platform or sample returns it to the newest night, since two
  scopes rarely flag their regression on the same one), a
  banner with that night's verdict counts across all scoped detectors
  (checked / 🔴 regressed / ⚠️ watch / ❌ failures), a per-detector status
  roster (badge, flag counts, worst flagged metric, and a link that opens the
  Regressions tab scoped to that detector and pinned to the selected night),
  and a **flagged-metric trend** that opens on the night's worst flag — the
  metric's history over the trend window (plus the selected night itself when
  it lies outside) with the baseline band its verdict was judged against and,
  for a confirmed step, the shaded blame window, all built from the cached
  reports with no run downloads. The view renders even
  on a night whose configs all hard-failed (when there are no values to plot),
  so a failure is never hidden behind an empty chart;
- **Nightly Report** — one night's regression report in the form the Key4hep
  e-group received it. Nothing is archived for it: the mail body is *rebuilt*
  from the already-fetched report (plus that night's blame sidecar) by the same
  renderer the nightly mail uses, so it is a reconstruction rather than a stored
  copy — an old night is redrawn by today's renderer, and the mail's *CI run*
  button is absent because that workflow-run URL is not stored in the report.
  Its night picker reaches as far back as the trend window and is deep-linkable
  via `?report=`, the same parameter the Regressions tab and the alert emails
  already speak. Unlike every other view it is **not** scoped to the sidebar's
  platform/sample — the mail covers every detector, platform and sample the
  night benchmarked — which is why it stays reachable even when the selected
  scope has no data. Links open in a new tab.

Colours follow the detector *family* (ALLEGRO, CLD, …), with versions of one
family distinguished by dash pattern and marker symbol, so experiment-level
comparisons read at a glance. Runs that failed the host-reliability check are
excluded by default with the same **Reliable only / All runs** selector as every historical
view (the nightly report carries each night's per-detector verdict); their raw
values are still recorded — as unjudged points, never flagged — so choosing
**All runs** plots them like Run Trends does. One selector covers all three views,
the Regression Status trend preview included, and it reaches
every report in view — including nights outside the trend window, which the
landscape and the night picker both read, and any the landscape reaches back to
for a detector that has not run in a while. Exclusion drops the *run*,
not the nightly tag: a tag benchmarked twice keeps the point measured by
whichever rerun passed, and a 🔴/⚠️ marker leaves the chart with the run that
earned it rather than ringing on a rerun that was never flagged.

A point on these charts is a Key4hep nightly **tag**, not a run — the unit the
regression engine judges on — so where a tag was benchmarked more than once it
carries the *newest* run's value and the *worst* verdict among its runs. A
release that regressed is therefore still marked even when a later rerun of the
same stack came out quiet. The per-run view is the Regressions tab's drill-down,
where every run gets its own point and a flag sits only on the run that earned
it; if that run was excluded or falls outside the window, the drill-down hides
the marker and says why rather than moving it.

The landscape is the one chart that is a *run*, not a tag: its point comes from
a single run of the detector's newest tag, so the two coordinates are always the
same measurement even when a rerun re-measured only one of them. Value
axes are logarithmic by default (the **Scale** switcher offers Linear as well)
— the detectors span more than a decade in both time and memory, so a linear
scale squashes the small ones into an unreadable cluster.

Detectors are only compared like-for-like: anything not benchmarked with the
selected sample/platform is listed as excluded instead of silently plotted
against a different workload. The values shown are the raw nightly
measurements the regression engine judged — baselines and Δ-verdicts stay in
the Regressions tab.

## Reconstruction page

The Reconstruction page reads `runs/reco/`: k4run reconstruction jobs run on a
frozen simulated input under the k4Bench Gaudi auditor, which records each
event's time and memory and, per component (algorithm or service), its time in
every phase of the job (`{config}_components.json`, see
[File formats](../../reference/file-formats.md#eos-layout)). Its sidebar works
as on the Simulation page but offers only the reco tree's detectors, platforms
and samples. A configuration is `baseline` or a `variant_<name>` of the job.

No nightly report judges the reco suite yet, so this page has no Regressions or
Stack Changes section and its views carry no verdicts: every value is the
measurement as uploaded.

| Section | What it shows | Backed by |
| --- | --- | --- |
| **Overview** | every reco detector's baseline time and memory side by side, over the trend window | `*_results.csv`, `*_events.json` of every detector's runs |
| **Breakdown** | where the job's time goes, phase by phase and component by component | `*_components.json` |
| **Efficiency** | which algorithms run inefficiently: time per event against instructions per cycle and heap allocations, and how time, instructions and instructions per cycle move night to night | `*_components.json` |
| **Memory** | where the job's heap goes component by component, from initialization through every event to finalization, and what grows or steps up without being released | `*_heap.json`, `*_components.json`, `*_events.json` |
| **Data Flow** | which algorithm feeds which, through the collections it writes and a later one reads | `*_dataflow.json`, `*_components.json` |
| **Event timing** | per-event wall time distributions, as on the Simulation page, and any algorithms' own time per event beside them | `*_events.json`, `*_components.json` |
| **Event memory** | per-event RSS and growth, as on the Simulation page | `*_events.json` |
| **Correlations** | each event's time, and each algorithm's time, instructions and allocations, against what the event holds, as on the Simulation page | `event_source.json.gz`, `*_event_size.json`, `*_components.json` |
| **Event size** | what each event of the EDM4hep files the job wrote takes on disk, as on the Simulation page | `*_event_size.json` |
| **Machine info** | the host the benchmark ran on | `machine_info.json` |
| **Logs** | the k4run log, read as a Gaudi log, and the job options | `*.log`, `*_joboptions.opts` |

**Overview** draws the Simulation Overview's Performance Trends and Performance
Landscape views from the runs themselves, with the same like-for-like rules: one
platform and sample, the baseline configuration. It is available only against
the remote store. If a detector's runs could not be listed, a warning names it
instead of it silently dropping out of the charts.

**Breakdown** has three views:

- **Current Run** shows each phase of the selected run that lasted 50 ms or
  more (in practice configure, initialize, event loop and finalize) as a card
  with its share of the job, then a donut each
  for initialization, the event loop and finalization: the five components
  that took longest, the rest as one *Other* slice, and in grey the part of the
  phase no component was charged for (*Unattributed*). The middle of each donut
  is the phase's measured duration. A component keeps one colour across the
  donuts and the chart below them, which stacks the event loop's components
  event by event, in event order or longest first; the first event, which warms
  the job up, is drawn paler.
- **Components** lists every algorithm and service of the selected run's
  configuration, most expensive first, with its time per phase, its class (for
  a wrapped Marlin processor, the processor's) and the stack package its
  library comes from, linking to that package's repository at the commit the
  run's stack built and to a search for the class's definition. A package
  built from a release is named with its version, and linked only where its
  repository is known (Gaudi's is).
- **Historical Trends** draws one phase at a time across the trend window: its
  top components (six by default, set in Display options), the rest as
  *Other*, and *Unattributed*, as stacked columns or as lines, in seconds or as
  a share of the phase. When a nightly tag was run more than once, only its
  latest run is drawn. Configure is timed only as a whole, so it is one line.

Component times are *self* times: what the auditor charged to that component
alone. They add up within a phase, which is why the donuts and columns can be
stacked, and the part of a phase's duration that no component accounts for
shows up as *Unattributed* rather than being spread over the others.

**Efficiency** has two views:

- **Current Run** draws one point per algorithm of a configuration's event
  loop, coloured by the stack package its code comes from: its self time per
  event across (log scale) and its instructions per cycle up, the warm-up event
  left out. **Point size** chooses what a point's area shows: heap allocations
  per event (the default), the allocation rate (allocations per second of the
  algorithm's own time, how hard it leans on the allocator however long it
  runs), allocated memory per event, or nothing. **Named algorithms** sets how many of
  the slowest carry their name, up to all of them. An algorithm runs
  efficiently at around 2 or more instructions per cycle; below 1, the shaded
  band, it mostly waits, usually on memory. Hovering gives the numbers. For a
  run whose hardware counters could not be read, the view says why instead.
- **Historical Trends** draws the slowest algorithms (Top N in Display
  options) across the trend window in three panels: time, instructions and
  instructions per cycle per event. Instructions follow the work an algorithm
  does, not the machine it ran on, so time that moved while the instructions
  held is the machine, and instructions that moved are the software, its
  configuration or its input.

**Memory** reads the run's memory pass (`{config}_heap.json`, see
[File formats](../../reference/file-formats.md#heap-json)): a second run of the
job after the timed one, which charges every heap block to the component that
allocated it and keeps it charged until it is freed. Runs from before the
pass existed (nightlies before 2026-10-19), and configurations whose pass
failed, have none, and the section says so. It has three views:

- **Current Run** gives the configuration's heap in four tiles: what it held
  before the first event (with its largest holder), in the event loop (the
  median at an event's end, every event in the sparkline), at its largest, and
  how much it grows per event (the median slope between every two events, so a
  single large event does not move it). A warning names each component that
  grew by 10 MB or more over the events, steadily (in both halves of them),
  and after finalize still holds 10 MB or more of it, at least half of what it
  gained. That is what a leak looks like; a cache that fills as new detector
  cells are seen looks the same, while a table filled once, in a single step,
  or a buffer released at the end does not. A run with fewer than 10 events
  after the warm-up is too short to tell, and the view says so instead. A
  second warning names each component whose heap stepped up: rose by 100 MB
  or more from one event's end to the next's, and at every event's end from
  then to the last still held at least half of what it rose by. It gives the
  largest of those steps with their
  events, and what the component held at the last event and after finalize.
  That is how a memory pool that grows to fit the largest event so far looks
  (GGTFTrackFinder, which runs ONNX Runtime, rises by +4.13, +8.25 and
  +16.5 GB on ALLEGRO_o1_v04 Zbb), and the leak warning cannot see it, since
  the heap then holds still. A button under each warning opens each component it names
  in **Components**, its row selected. Below, every component's heap from
  initialization through each event to finalization is stacked, the most
  stable at the bottom: every component either warning names and the largest components (six by
  default, set by **Largest shown**), the rest as *Other*, and what no component
  allocated (the Python configuration, ROOT's start-up) as *No component*,
  both in grey. A band wide enough carries its name at its right end. The
  default palette has a colour for each series, and a component keeps its
  colour when more or fewer are shown; a smaller palette chosen in Display
  options repeats colours rather than hide a suspect. The dashed line is the
  timed run's anonymous RSS, and a triangle marks each step, in its
  component's colour, with its size in the hover. **Group by** stacks the stack packages the
  components come from instead, and **Change since the first event** draws
  what each one holds beyond what it held at the first event's end, so growth
  rises and a one-off step jumps, with *Other* and *No component* as grey
  lines: together the lines make the whole heap's change, so growth in a
  component not named still shows.
- **Components** lists every component that held heap, the largest typical
  holder first: what it held before the first event (*Initialized*), at an
  event's end (*Typical*, the median) and after finalize (*Finalized*), its
  heap across the events as a sparkline, its *working set* (the most it held
  during an event beyond what it held as the event started: what the event
  needed of it: median, 95th percentile, and the most any event needed with
  that event's number), its growth per event, and the package links of
  Breakdown's Components. *⚠ grows* and *⚠ steps* mark what the two
  warnings name.
  Selecting a row draws that component event by event: a bar from what it
  held as each event started up to the most it held during it, and a mark at
  what it held at the event's end. With nothing selected, it draws the
  fastest-growing suspect, else the component that stepped up the most, else
  the one whose events need the most.
- **Historical Trends** draws one of a configuration's numbers across the
  trend window, chosen by **Metric**: what each component held before the
  first event, at a typical event's end (the median), its working set (the
  median and the 95th percentile, as two panels) or its growth per event. It
  names the largest components over the window (six by default, set by
  **Largest shown**; for the working set, the largest of each panel) and
  every component that grows or steps up and keeps it on any night of the
  window besides, a line each, in one colour whatever is shown.
  The first two add up to the job's heap, so the rest of it is one more line,
  *Other*, and what no component allocated another, *No component*, both in
  grey. When a nightly tag was run more than once, only its latest run is
  drawn.

Heap is memory allocated, not memory resident: the timed run's anonymous RSS
can sit below the heap when blocks were allocated and never touched, and above
it by what libraries map themselves. A block stays charged to the component
that created it, so the event data the input reader creates is the reader's
until the output writer frees it. The warm-up event is left out of every
statistic over events.

**Data Flow** draws one configuration's algorithms left to right in data-flow
order, joined by the collections one writes and a later one reads
(`{config}_dataflow.json`, see
[File formats](../../reference/file-formats.md#dataflow-json)). A dot's area is
the algorithm's event-loop time per event, and hovering an edge's middle lists
the collections it carries, with the name a converter gave one on the way
(`MCParticles → MCParticle`). Algorithms sharing a class are one square node
(`DDPlanarDigiProcessor ×6`): clicking it shows each instance, and **Expand
all**/**Collapse all** do so for every class. Clicking an algorithm lists what
it reads, with who wrote each collection, and what it writes, with who reads
it. What the Writer saves is drawn on request, as it reads nearly everything.
Algorithms that no recorded collection links to another, such as Marlin
processors that declare no collections, are not drawn; a caption under the
graph counts them, and the **Limitations** expander names them along with
everything else the run could not record: reads with no recorded writer, and
sources the run predates. Drag to pan, scroll to zoom, double-click to reset.

**Event timing** draws the per-event time as the Simulation page does, and its
**Timing of** selector, on one row with the Baseline and Configurations, sets
any algorithms beside the whole event: each one's own time in each event, what
it spent outside the algorithms it ran itself, the warm-up event left out. It
opens on the whole event beside the baseline configuration's slowest algorithm,
for the baseline alone. Algorithms are offered slowest first, with their mean
time per event. Every timing of every chosen configuration is one series in the
histogram, the ratio panels and the Statistics table, named by the
configuration while one timing is chosen, by the timing while one configuration
is, and by both otherwise (`VertexFinder · baseline`). The first timing chosen
sets which configurations are offered. Each timing is compared with the same
timing of the baseline configuration, never with another timing, so timings of
the baseline alone have no ratio panels. The historical trends draw one
**Configuration**'s whole event beside its **Slowest algorithms** (five at
first, ranked by their mean time per event over the runs drawn: those the
**Runs** choice keeps, each nightly tag's latest; up to all of them), on a log
y axis unless **Log y axis** is switched off. They draw the algorithms whenever
the window has their timing, even if the selected run has none.

**Logs** reads the log's Gaudi and Marlin severities rather than the word
"error": the **Errors** filter keeps ERROR and FATAL messages, ROOT errors,
Python tracebacks, crashes and the job's non-zero exit lines. A failed job is
explained by its first FATAL message, crash or k4run error, and the component
that printed it. Below the log, **Job options** shows the configuration's resolved
properties, filterable and downloadable.

## Running the dashboard locally

You can point the dashboard at either a local directory or the remote WebEOS
endpoint, via environment variables read by
[`dashboard/config.py`](../../architecture/data-flow.md):

| Variable | Default | Meaning |
| --- | --- | --- |
| `K4BENCH_DATA_DIR` | `logs` | local directory to read runs from |
| `K4BENCH_DATA_URL` | *(unset)* | WebEOS base URL; when set, overrides local reads |
| `K4BENCH_CACHE_DIR` | `$TMPDIR/k4bench_cache` | on-disk cache for downloaded runs |
| `K4BENCH_DASHBOARD_URL` | `https://k4bench-dashboard.app.cern.ch` | this deployment's own public URL; the Overview tab's **Nightly Report** view needs it because the embedded mail's deep links must be absolute |

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
