# Make regression verdicts consistent across nights that re-benchmark the same Key4hep release

## 1. The problem, concretely

Compare these two reports (same triple, same release `key4hep-2026-06-27`):

- `?...&stack=key4hep-2026-06-27&report=2026-06-27` — many confirmed regressions.
- `?...&stack=key4hep-2026-06-27&report=2026-06-28` — the same metrics are quiet ("within
  baseline variation (re-anchoring after confirmed change …)").

Both nights ran the **identical software stack** (`key4hep-2026-06-27` — the nightly
build does not publish every day, so consecutive runs frequently re-measure one
release). A regression is a property of the *release*, not of the *night that happened
to measure it first*, so both reports should show the same regressions, modulo machine
noise. Today they don't, by design of the current engine — a design that predates
multi-night-per-release awareness.

### Why it happens — the exact mechanism

The step detector (`k4bench/regression/engine.py:evaluate_series`) walks the metric
history **one night at a time**, ordered by `(run_date, run_id)` where `run_date` is the
release date (`x_date = k4h_release_date.fillna(run_date)`, see
`k4bench/analysis/trend.py:_finalize_dates`) and `run_id` is the nightly run directory
name. Three state updates happen **per night**, treating every night as a new software
state:

1. Every judged night's value is appended to the trailing 14-point baseline window.
2. Two-strike rule: a night that trips both gates is `WATCH`; the *next* reliable night
   tripping in the same direction is `CONFIRMED`.
3. **Change-point re-anchoring**: immediately after a `CONFIRMED` night, the baseline is
   cleared and re-seeded at the post-change level, so the very next night is judged
   against the *new* level and comes out `OK`.

Trace of the observed case (regression introduced around release 2026-06-26):

| night (run_id) | release measured | judged against | verdict |
|---|---|---|---|
| 2026-06-26 | key4hep-2026-06-26 | pre-change baseline | ⚠️ WATCH (first strike) |
| 2026-06-27 | key4hep-2026-06-27 | pre-change baseline | 🔴 CONFIRMED → **baseline re-anchors to the new level** |
| 2026-06-28 | key4hep-2026-06-27 (same!) | **post-change baseline** | ✅ OK — the regression "disappears" |

Step 3 is the "reset" the user refers to. Its purpose ("an accepted deliberate change
alerts exactly once, not every night for two weeks") is sound **across releases**, but a
second night on the *same* release is a repeat measurement of the same binary, not a
new software state that "accepted" the change. Re-anchoring between two nights of one
release makes the second night's report contradict the first for identical software.

The dashboard currently *works around* this instead of fixing it: the Regressions tab
grew a night picker that defaults to "the most attention-worthy night"
(`dashboard/tabs/regressions.py:_pick_night`), Run Trends and the Overview keep the
"worst verdict across a tag's reruns" (`dashboard/tabs/trends.py:_severity_lookup`,
`dashboard/tabs/detectors_overview.py:history_frame`). Those keep the confirmation
night *reachable*, but every other night of the release still renders a false all-clear.

## 2. Target semantics — the spec

**The unit of change is the release. Nights are repeat measurements.** All engine state
transitions must happen at *release boundaries*; within a release, every night is judged
against the same frozen baseline. "Release" here means the normalized `run_date` value
of the history row (`x_date`); rows whose `k4h_release_date` is NaT fall back to their
run date and therefore form single-night groups — the current behaviour degrades to
exactly today's walk when every release has one night.

Precise rules (R1–R6). Group the sorted history rows by `run_date`; call each group a
release R with reliable nights n1..nk in `run_id` order:

- **R1 — frozen per-release baseline.** Compute the baseline snapshot `(median, MAD)`
  once when entering release R, from state accumulated from releases `< R` only. Judge
  *every* night of R against that snapshot. Values measured under R never enter the
  baseline used for R's own nights. (This is what makes reports for all nights of one
  release agree, modulo each night's own measured value.) The existing interim
  re-anchoring rule — while the post-change segment has `< MIN_BASELINE_RUNS` points,
  use the segment median with the pre-change MAD as spread proxy — is applied when
  computing the snapshot, unchanged.

- **R2 — two-strike confirmation, unchanged.** A tripping night is `WATCH` if no
  matching pending direction exists; `CONFIRMED` if the pending `WATCH` (set either by a
  previous release's night **or by an earlier night of the same release**) moved the same
  direction. A second night of the same release confirming the first is *good* evidence
  — same binary, independent run, rules out one-night machine flukes — and is already
  today's behaviour; keep it. A clean (non-tripping) night still clears an unconfirmed
  pending `WATCH`, as today.

- **R3 — a confirmation is sticky for the rest of its release.** Once a change is
  confirmed in direction D during release R (whether the pending came from R-1 or from
  within R), every *later* night of R that trips in direction D is also `CONFIRMED`, and
  carries **the same** `onset_*`/`last_accepted_*` window as the first confirmation —
  do not re-stamp a new onset per night. An OK night inside R (marginal value, noise)
  does not clear the release's confirmed state (contrast with R2's clearing of an
  *unconfirmed* WATCH); it just isn't flagged itself. A trip in the *opposite* direction
  follows the normal R2 logic (fresh WATCH). Identical windows across the release's
  nights keep the blame sidecar stable: `k4bench/blame/builder.py` dedupes and ranks per
  `(platform, base, onset)` window, and the dashboard's blame cards dedupe by window too.

- **R4 — re-anchoring is deferred to the release boundary.** When the walk leaves a
  release R in whose scope a confirmation happened (entering the first night of R' > R,
  or ending the series): clear the baseline and re-seed it with **all of R's reliable
  judged values** (better level estimate than today's two seed points), set
  `anchor_mad` to the pre-change snapshot MAD, `anchor_date` to R's release date, clear
  pending, and set `last_accepted` to R's last reliable night. If R saw no confirmation,
  append R's values to the baseline in night order (WATCH values included, as today —
  one outlier cannot move a 14-point median) and carry any still-pending WATCH into R'.

- **R5 — warm-up and reliability rules unchanged.** Fewer than `MIN_BASELINE_RUNS`
  baseline points ⇒ `UNKNOWN`, value appended immediately (no judging happens, so the
  freeze doesn't apply). `reliable is False` nights are skipped entirely — they neither
  judge, confirm, clear, nor narrow a blame window.

- **R6 — `last_accepted` keeps its current meaning** (newest night observed at the
  accepted level, updated on OK nights and at re-anchor). Note a nice consequence: if
  night n1 of R is OK and n2/n3 of R trip and confirm, the window becomes
  `(R, R]` — a *same-release* window, which the dashboard already renders as "no tracked
  Key4hep package changed; look at benchmark code/config, inputs, environment, or noise"
  (`tabs/_blame.WindowKind.SAME_STACK`). That is exactly the right attribution for a
  step between two runs of one release.

### What the observed case looks like afterwards

| night | release | judged against | verdict |
|---|---|---|---|
| 2026-06-26 | 2026-06-26 | pre-change snapshot | ⚠️ WATCH |
| 2026-06-27 | 2026-06-27 | pre-change snapshot | 🔴 CONFIRMED (onset = 06-26's night) |
| 2026-06-28 | 2026-06-27 | **same pre-change snapshot** | 🔴 CONFIRMED, same window |
| first night of next release | 2026-06-30 | re-anchored on release 06-27's nights | ✅ OK (change accepted) |

"Alert exactly once" becomes "alert exactly once **per release transition**": the alert
repeats on every night that re-measures the offending release (that's the point — the
regression is still present in that release), and falls quiet from the next release on.

### Consequences to accept (deliberate, not bugs)

- The nightly e-group email will list the same confirmed regressions on every night of
  the affected release. That is desired visibility. Optional polish (separate, small):
  have `k4bench/regression/notify.py` / `render.py` mention "first confirmed on
  {onset night}" so repeat emails read as repeats.
- The baseline window (`BASELINE_WINDOW_RUNS = 14`) counts *measurements*, so a release
  benchmarked on several nights occupies several window slots and the window spans fewer
  distinct releases. That is statistically fine (repeat measurements of accepted
  software are legitimate baseline samples) — document it, don't fight it.
- Report JSON schema does **not** change. Only which severities/windows appear on which
  nights changes.

## 3. Implementation checklist

### 3.1 Engine (`k4bench/regression/engine.py`)

Rework `evaluate_series` into the release-grouped walk (R1–R6). Practical shape: keep
the single chronological pass, but batch rows sharing a `run_date`; compute the snapshot
at group entry; hold per-group state (`release_confirmed`, its direction and stamped
window); apply baseline appends / re-anchor at group exit. Update the module docstring's
items 5 and 6 and the `evaluate_series` docstring to describe the release semantics —
per the repo's comment policy, describe only the *current* behaviour, never "this used
to re-anchor per night".

### 3.2 Tests (`tests/unit/test_regression_engine.py`)

The `_history` helper generates one release per night, so all existing tests exercise
the degenerate case and should pass unchanged (any that break indicate a semantics
regression — investigate, don't adjust them blindly). Add a helper that builds a history
with explicit `(run_id, run_date)` pairs, and cover at least:

1. Step introduced at release R, R benchmarked on 3 nights: expect
   `WATCH, CONFIRMED, CONFIRMED` on R's nights, identical windows on both CONFIRMED,
   then OK on the next release's first night (re-anchor happened at the boundary).
2. WATCH on release R-1's single night, R benchmarked twice: `CONFIRMED, CONFIRMED` on
   R's nights, window `(last night of R-2's level, R-1's night]`.
3. All nights of a re-benchmarked release judged against the same
   `baseline_median`/`baseline_mad` (assert equality across the release's verdicts).
4. Same-release onset (R6): n1 of R OK, n2 WATCH, n3 CONFIRMED with
   `last_accepted_run_date == onset_run_date` (same release).
5. An OK noise night between two CONFIRMED nights of one release does not clear the
   confirmed state or change the stamped window.
6. A second step in the *next* release right after a confirmed one is still caught
   (re-anchor seeded from the whole previous release, interim MAD rule intact).
7. An unreliable night inside a multi-night release is skipped and doesn't perturb the
   frozen snapshot or the window.

Run with `py-venv/bin/python -m pytest tests/unit/test_regression_engine.py` (system
python is 3.9 and too old), then the full suite including
`tests/integration`/`tests/unit/test_regression_report.py`.

### 3.3 Narrative text that states the old behaviour

These all currently assert "a confirmed regression appears on exactly one night / the
baseline re-anchors the following night" and must be rewritten to the release semantics
(grep for `re-anchor` and `exactly one` to be exhaustive):

- `dashboard/tabs/regressions.py` — module docstring (lines ~7–14),
  `_candidate_nights` docstring, `_select_night` `help=` text (~350), banner help texts
  are still correct (two-strike wording unchanged).
- `k4bench/regression/render.py:_group_links` docstring (~189) — the `?report=` pinning
  stays (still wanted for WATCH-vs-CONFIRMED nights and pre-backfill history), but the
  rationale text changes.
- `dashboard/app.py:476` comment.
- `dashboard/tabs/trends.py:_severity_lookup` docstring (~53) and comment at ~320 —
  **keep the worst-across-reruns mechanism** (nights of one release can still differ:
  WATCH → CONFIRMED progression, marginal OK nights, and pre-backfill reports), just fix
  the stated rationale.
- `dashboard/tabs/detectors_overview.py` — `history_frame` docstring (~259–264) and the
  comment at ~285; same: keep the mechanism, fix the rationale.
- `dashboard/tabs/_regression_flags.py:51` comment.
- `dashboard/tabs/stack_changes.py:279` comment.
- `k4bench/blame/models.py:265` comment.
- `docs/user-guide/features/dashboard.md` — the "Re-anchoring" bullet (~124) and the
  whole "appears on exactly **one** report night" paragraph (~135–146).
- `docs/reference/file-formats.md:195` — check and adjust the re-anchor mention.

Keep `_pick_night`'s attention-worthy default: it still breaks WATCH-vs-CONFIRMED ties
correctly and protects pre-backfill history; after backfill it usually resolves to the
newest night anyway.

### 3.4 No changes needed

`report_builder.py` (the walk is entirely inside `evaluate_series`; `_series_history`
already carries `run_date = x_date`), `models.py`, `render.py` JSON round-trip,
`blame/builder.py` (windows arrive already deduped per R3), notify gating.

## 4. Backfill of historical reports on EOS

Historical `_reports/{night}/report.json` were produced by the old engine, so e.g. the
2026-06-28 report permanently shows the false all-clear. The engine change also shifts
baseline composition and re-anchor timing *globally*, so regenerate **all** report
nights, not just multi-night releases. Everything needed is on EOS: reports are pure
functions of the run directories (`run_info.json`, results/event CSVs,
`machine_info.json`), which persist indefinitely. Make sure that the backfill script you
generate uses all threads on the machine to write to backfill as quickly as possible.

### 4.1 An `as_of` seam in the builder

`build_nightly_report`/`build_group_report` currently always judge the newest run. Add
an optional `as_of: str | None` (a `YYYY-MM-DD` night): filter the sorted
`(date, stack)` pairs to `date <= as_of` *before* taking the trailing
`FETCH_WINDOW_RUNS` window. Nothing else changes — `_finalize_report` then reproduces
that night's report semantics automatically (report night = newest run ≤ `as_of`;
triples whose newest run is older get the historical missing-run / retired handling).
Mirror the same cutoff in `build_nightly_report_local` so the integration test can cover
it.

### 4.2 `.github/scripts/backfill_reports.py`

New script, cloned from the discipline of `.github/scripts/backfill_blame.py` (read it
first — it is the house style for backfills):

- **Same IO-seam structure** (`BackfillIO`-style dataclass) so a smoke test drives it
  over a local tree offline.
- Nights = `k4bench.remote.list_report_dates(data_url)`, filtered by
  `--since/--until/--limit`.
- Per night N: `build_nightly_report(data_url, cache_dir, as_of=N)`, serialize with
  `k4bench.regression.render.to_json`, upload atomically (xrdcp to a hidden temp name
  beside the target, then `xrdfs mv` into place — copy `_xrootd_uploader`).
- **Dry-run by default**; `--apply` uploads. In dry-run print a per-night diff summary
  vs the existing report (counts of CONFIRMED/WATCH/FAILURE before → after) so the
  effect is reviewable before any write.
- **Archive before overwrite**: unlike blame, the old reports are *not* reproducible
  from current code. Before the first `--apply`, download every existing
  `_reports/{night}/report.json` into a local archive directory (`--archive-dir`,
  refuse to apply without it unless `--no-archive` is passed explicitly).
- Reuse one `--cache-dir` across nights — consecutive nights share almost their entire
  run window, so the cache makes the whole backfill cheap.
- Run it **off the benchmark runner** (dev box), like the blame backfill.

Note an accepted side effect: regenerated reports reflect *today's* engine (including
gates added since those nights, e.g. `ABS_DELTA_FLOOR`), and fresh `generated_at`
timestamps. That is the point — one consistent detection history.

### 4.3 Blame sidecars second

Report backfill changes which nights carry confirmed regressions (quiet rerun nights
gain them), so their `blame.json` must be (re)built **after** the reports:
`python .github/scripts/backfill_blame.py --overwrite --apply ...` over the same night
range, with `GITHUB_TOKEN` and the `K4BENCH_LLM_*` env set (see that script's
docstring; `--sleep` for free-tier pacing). Order matters — a sidecar must never be
joined to a report it did not examine.

### 4.4 Verification

1. Offline: point `regression_report.py --data-dir` at a fixture tree with a
   multi-night release and eyeball the walk.
2. Dry-run the backfill for `--since 2026-06-25 --until 2026-06-30` and check the diff
   summary: 2026-06-28 must gain the regressions 2026-06-27 shows.
3. After `--apply`, load both dashboard URLs from §1: the same confirmed regressions
   (slightly different measured values) must appear on both report nights, each with the
   same blame window and candidate PRs, and the night picker badges should agree
   (🔴 on both pills).
