"""Assemble a :class:`~k4bench.blame.models.BlameReport` from a nightly report.

For each confirmed regression whose blame window spans two *different* releases,
diff the two releases' package maps, resolve each changed GitHub repo's commit
range to its pull requests, and rank them for that regression. The result is the
sidecar the CLI uploads to ``_reports/{night}/blame.json``.

Its dependencies are injected rather than imported, which is what keeps this
package offline-testable and lets CI reuse work it has already done:
* ``packages_for_release(platform, release) -> dict | None`` — a release's
  ``k4h_packages`` map. In CI this reads the run cache the report build already
  populated; the integration test reads a local tree. ``None`` means the release
  predates provenance capture or has aged off CVMFS — its regressions get no
  blame rather than a wrong one.
* ``github`` — a :class:`~k4bench.blame.github.GitHubClient`, or ``None`` to skip
  PR resolution entirely (no token available): the diffs are still recorded, the
  repos just carry no candidates.
* ``k4bench_commit_for_run(detector, platform, release, sample, run_id) ->
  (sha, run_url) | None`` — the harness's own commit at one exact *run*.
  k4Bench is not in any release's package map (it measures the stack, it is
  not part of it), yet its changes move measurements too; this lookup is what
  lets the window's diff include it. ``None`` — not injected, or a run that
  cannot be read — means the harness's movement is unknown and no entry claims
  anything about it.
* ``provenance_for_run(detector, platform, release, sample, run_id) ->
  StackSnapshot | None`` — the stack one exact run recorded, including what it
  loaded. A changed package that neither end of a rank group's runs loaded is
  dropped before PR discovery; when any of those runs lacks a loaded set, every
  changed package stays. It also supplies the window's external changes.

Windows that cannot be attributed are dropped, not recorded empty: an open-ended
window (no settled baseline), a window whose provenance is missing, and a window
where nothing at all moved are handled live by the dashboard from ``report.json``
alone. A *same-release* window is attributed exactly when the harness's own
commits moved between its two runs — the upstream stack is identical by
construction, so the harness is the only thing a diff can name. ``blame.json``
exists only to carry the one thing the dashboard cannot compute itself — the
ranked PRs.

- :mod:`.night`   — one night's build, keyed by release: the stack diffs, the
  pull requests behind them, the historical retrieval and the entries.
- :mod:`.runs`    — what a window's two exact runs recorded: the harness's own
  commit range and the loaded stack.
- :mod:`.ranking` — the ranker's input for one rank group, and its answer folded
  back onto the candidates.
"""

from k4bench.blame.builder.night import PackagesForRelease, build_blame_report
from k4bench.blame.builder.runs import K4BenchCommitForRun

__all__ = ["K4BenchCommitForRun", "PackagesForRelease", "build_blame_report"]
