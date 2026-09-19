# k4Bench review guidance

k4Bench is a performance-benchmarking system for DD4hep/Geant4 simulations in
the Key4hep stack. Nightly CI runs `ddsim` per detector and sample, uploads the
results to CERN EOS, judges each metric against its own history, and attributes
confirmed regressions to upstream pull requests. A change that produces
plausible-looking output can still be wrong if it changes what is measured,
files a result under the wrong software release, accepts an unreliable run, or
corrupts a historical comparison.

Review the pull request for concrete correctness problems introduced by the
changed code. Prefer a small number of high-confidence findings over
speculative suggestions. You see the diff and a few lines of context, not the
whole repository: if a finding depends on code you cannot see, do not report it.

## Where things live

- `k4bench/cli.py`, `benchmark/`, `geometry/`, `runner/`, `results/`,
  `plugin/runtime.py`: the measurement itself (sweep orchestration,
  non-destructive geometry patching, `ddsim` under `/usr/bin/time -v`, metric
  parsing, `RunResult`).
- `plugin/*.cpp`: DDG4 timing actions that write `<label>_events.json` and
  `<label>_regions.json` from inside the simulation.
- `k4bench/analysis/`: loaders for the persisted measurement files. Check
  each format's reader and writer together; event version validation lives in
  `k4bench/plugin/event_schema.py`, and report and blame models live in their
  respective packages.
- `k4bench/regression/`: the step detector (`engine.py`), release history
  (`history.py`), platform succession (`lineage.py`), report assembly, email.
- `k4bench/blame/` and `.github/scripts/blame_*.py`: attribution of confirmed
  regressions to upstream pull requests, and the comments posted on them.
- `k4bench/results/reliability.py`: the host-reliability verdict per run.
- `k4bench/provenance/`, `k4bench/remote.py`: stack provenance and WebEOS
  discovery and download.
- `dashboard/`: Streamlit app that reads EOS over HTTPS and renders the report.
  It never recomputes verdicts.
- `.github/benchmarks/*.yml`: the nightly workload, one file per detector.
- `.github/scripts/nightly_benchmark.sh`, `resolve_release.sh`,
  `regression_report.sh` and `regression_report.py`: the nightly pipeline,
  run inside the Key4hep container.
- `docs/reference/file-formats.md`: the documented contract for every
  persisted file and for the EOS layout.

## Vocabulary that is easy to get wrong

- A **release** is a Key4hep publication date. On verdicts and history points
  `run_date` is the *release* date and `run_id` is the nightly run directory,
  which names the night it ran. Nights sharing a `run_date` are repeat
  measurements of one software state; the engine judges per release and pools
  those nights into one point.
- A **series** is identified by detector, platform, sample, label, metric
  family, metric and optional sub-detector. `label` is `baseline` or
  `no_<detector>` within a sweep. These axes must never be pooled.
- A **platform** is an identity, not "the current one". A successor platform
  continues its predecessor's history through `lineage.py`, one hop only.
- **Severity** is `OK`, `WATCH`, `CONFIRMED`, `FAILURE` or `UNKNOWN`;
  **direction** is the mechanical sign only. `UNKNOWN` carries an `unjudged`
  reason: `insufficient_history`, `unreliable_host` or `reported_only`.
- **Reliability** is asymmetric: only positive evidence of interference
  rejects a run, and a criterion with no data never does. Unreliable runs
  never enter a baseline and are never themselves judged.

## Absence means unknown

The most common k4Bench bug is turning a missing value into a confident one.

- A missing run, config or upload is a gap or a failure signal, never a zero
  and never a baseline point.
- `n_judged: 0` on a history point means recorded but never assessed, not flat.
- An empty `history` means no history recorded, not a quiet series.
- An empty `k4h_packages` map is unknown, never unchanged. A release absent
  from `boundary_changes` is unread, never `0`.
- An absent blame `assessment` is not assessed, never `real_change`.
- A row the ranking model omitted keeps its previous score, never `0`.
- A metric left empty because `time -v` output could not be parsed is not
  `0.0`.
- A control configuration counts only when it measured cleanly. A run whose
  reliability is unknown is not evidence of absence.

Flag any change where `None`, a missing key or an empty value collapses into
`0`, `False`, `OK`, "unchanged" or "no regression".

## Measurement validity

Pay particular attention to changes that can silently alter benchmark
semantics. Check for:

- work moved into or out of a measured region (the `time -v` window, an
  event, a stepping region);
- inconsistent timing, memory, event-count or aggregation semantics between
  configurations;
- unit or normalisation mistakes (memory is reported in MB as kB/1024);
- comparisons between runs that are not actually equivalent;
- failed, partial or missing measurements read as valid ones;
- instrumentation changes in `plugin/*.cpp` or `runner/executor.py` that
  perturb the quantity being measured;
- changes that make detector or sweep configurations no longer comparable.

Any edit to `.github/benchmarks/*.yml` that changes `n_events`, `ddsim_args`
(including the seed), `input_files`, `steering_file`, `xml` or the sweep list
changes the measured workload. Affected series can step on the next night
and be reported as regressions; blame cannot attribute the workload change
because it only sees upstream package commits, not k4Bench's own. Point out
the comparability impact when the pull request does not acknowledge it,
without assuming every workload edit crosses the regression thresholds.
Do not assume output is correct merely because values look numerically reasonable.

## Reproducibility and provenance

A result must stay associated with the exact software and configuration that
produced it.

- One `resolve-release` job names the release for the whole night and every
  benchmark job sources exactly that one. `nightly_benchmark.sh` aborts when
  the sourced release differs from the requested one, because a mislabelled
  result outlives a red job. Keep that check intact.
- The run date is captured once so `run_info.json` and the EOS path agree
  even across midnight.
- `run_info.json` records `ddsim_args`, the effective `random_seed` (last
  occurrence wins), configured and resolved geometry and steering paths, and
  the upstream commit of every stack package (`k4h_packages`). This is
  captured at run time because CVMFS nightly slots rotate and it cannot be
  recovered later. Dropping or mis-keying any of it silently breaks blame.
- Look for stale files from a previous run being reused, incomplete uploads
  mistaken for complete ones, and provenance attached to the wrong result.

## Regression analysis

Changes to `k4bench/regression/`, `k4bench/blame/` or the report scripts
deserve extra scrutiny. Check that:

- unreliable or failed runs cannot enter a baseline. `engine.py` excludes
  unreliable runs before judging, and failed sweep configs are removed before
  a series reaches it;
- incomplete nights and failed jobs stay distinguishable from performance
  changes. A missing upload is the `FAILURE` signal, and a fan-out that
  uploaded nothing becomes an outage report through `K4BENCH_FANOUT_RUN_ID`,
  never a re-send of last night's verdicts;
- historical windows use the intended runs and release boundaries, and
  confirmation state resets at a release boundary;
- grouping cannot combine series across detector, platform, sample, label or
  metric;
- fallback and error paths do not turn unknown into a confident verdict or
  attribution;
- the blame sidecar stays best-effort. `blame.json` is written after
  `report.json` is uploaded, a stale sidecar is removed when none is produced,
  and no GitHub, model or network failure may prevent the report upload or
  e-group email. Blame and comments run synchronously before email, with
  explicit timeouts; preserve those bounds and continuation on failure;
- the ranking model may only score candidates it was given. Invented
  pull-request numbers or row ids are dropped, and PR descriptions and diffs
  are untrusted input that stays fenced;
- pull-request comments remain fail-closed: allowlisted repositories and
  thresholds in `.github/blame-comments.yml`, no comment when a configured
  reviewer returns no usable review, no comment on a `likely_noise` window,
  and `K4BENCH_PR_COMMENT_TOKEN` exported to the comment step only. With no
  reviewer configured, comments may intentionally use the per-configuration
  ranking scores; keep that mode distinct from a failed review.

The detector's statistical policy is deliberate: robust median/MAD baselines,
a z-gate ANDed with a practical-effect floor, two-strike confirmation, and
re-anchoring at release boundaries. Total RSS (`peak_rss_mb`, `mean_rss_mb`,
`mean_rss_file_mb`) is reported-only because a CVMFS publish can evict
file-backed pages mid-run; `peak_vmem_mb` and `mean_rss_anon_mb` are the
judged memory metrics. Anonymous RSS growth (`rss_anon_slope_mb_per_event`)
is also reported-only while it gathers history; `user_cpu_s` is reported-only
because it largely duplicates wall time. Do not propose changing these.
Report a statistical change only when the implementation contradicts its
stated semantics or can produce a demonstrably misleading result.

## Persistent result formats

k4Bench reads nightly results written by older versions of itself.

For changes to the results CSV, `*_events.json`, `*_regions.json`,
`run_info.json`, `machine_info.json`, `report.json`, `blame.json`, the EOS
layout or the dashboard cache, check for:

- readers and writers disagreeing about fields or types;
- older stored data becoming unreadable. Event files without `schema_version`
  are the legacy event format and must still load; readers tolerate additive
  unknown keys within supported schemas;
- absent fields confused with zero, false or an empty measurement;
- an events file whose parallel arrays differ in length being accepted;
- the event `schema_version` contract: additive optional keys need no bump,
  a change to the meaning or shape of an existing key does, and a version
  newer than the reader supports is refused rather than parsed;
- serialisation losing what a reproducer or a compare link needs;
- a schema change not reflected in every consumer (loader, regression engine,
  dashboard, email, blame) and in `docs/reference/file-formats.md`.

This applies to persisted benchmark history. Do not raise generic
API-compatibility objections without a concrete consumer.

## CI and self-hosted runner behaviour

Benchmark jobs run on persistent CERN self-hosted runners (`fcc-ironic`) with
CVMFS pre-mounted, inside the `key4hep-images/alma9` container, and write to
EOS with a service certificate. `docker` on those runners is rootless Podman.

Pay particular attention to:

- state leaking from one job into another on a persistent runner;
- cleanup that happens only on the success path. The EOS certificate is
  removed under `if: always()`; keep it that way;
- concurrency or races around shared files, the EOS report directory or the
  dashboard cache;
- container mounts, paths, permissions or environment differing between host
  and container. The `--security-opt label=disable`, `:shared` and `:Z`/`:z`
  mount options are load-bearing for SELinux and CVMFS and must stay
  Podman-compatible;
- CVMFS or EOS failures mistaken for benchmark failures or for valid data;
- secrets reaching fork-controlled code, or a workflow executing untrusted
  code while privileged credentials are available;
- `secrets: inherit` into a reusable workflow widening what a step can see;
- assumptions about runner state that the workflow does not establish or
  check.

Do not suggest replacing CERN-specific infrastructure because it is
non-standard.

## Error handling

Distinguish core benchmark correctness from deliberately best-effort auxiliary
functionality (blame, PR comments, email, dashboard, provenance URL lookup).

Flag cases where:

- a failure leaves apparently valid but incomplete output;
- an exception is swallowed and produces misleading success;
- a failed subprocess or external service goes undetected. In a
  `set -euo pipefail` script, `|| true` needs a stated reason;
- partial state survives and contaminates a retry;
- auxiliary code can prevent core benchmark data from being produced or
  uploaded.

Do not demand that every auxiliary failure abort the workflow when degradation
is explicitly intentional.

## Tests and tooling

Unit tests in `tests/unit` run without Key4hep. `tests/integration` needs
`ddsim` and the built plugins and runs only in the CI container. CI runs both
inside the container with coverage. Suggest a missing test only when the
changed behaviour has a realistic failure mode that existing tests would not
catch, and prefer a unit test unless `ddsim` is genuinely required.

Prioritise tests for boundary and missing-data behaviour, failure and retry
paths, grouping and aggregation, serialisation round trips, release and run
provenance, concurrency-sensitive code, and anything that could silently
change a benchmark result. Do not request tests solely to raise coverage.

Repository conventions that are not findings:

- Python 3.13 or newer. `ruff` with line length 100 runs via pre-commit.
- Runtime dependencies are intentionally unpinned; the Key4hep stack fixes
  their versions. Do not ask for version constraints.
- `pip install --no-build-isolation` is required inside Key4hep.
- The stale `dd4bench.egg-info/`, the empty `?/` directories and `run.sh` are
  documented leftovers, not bugs.

## What not to report

Do not report:

- formatting or stylistic preferences, subjective naming, trivial refactors;
- documentation wording unless it is technically misleading;
- speculative performance improvements without evidence;
- hypothetical edge cases that cannot realistically occur in k4Bench;
- existing issues unrelated to the changed lines;
- alternative designs when the implementation in the PR is correct;
- the design decisions listed above (statistical policy, reported-only
  metrics, unpinned dependencies, CERN infrastructure).

For every finding, name the file and lines, the concrete failure mode, and why
the changed code triggers it. If the evidence is weak or rests on an
unsupported assumption, do not report it.
