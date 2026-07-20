#!/usr/bin/env python3
"""
Re-run the LLM ranking stage over already-published ``blame.json`` sidecars on
EOS, without rediscovering candidate PRs from package diffs.

Use this after changing :mod:`k4bench.blame.rank` (prompt, model, context
budget, ...) to refresh the score/description of every already-known
candidate PR. Everything else already recorded in ``blame.json`` — which
packages changed, which PRs are candidates, their metadata (title, author,
files, commit range) — is left exactly as it was; only ``score`` and
``description`` are replaced. This is deliberately narrower than rebuilding
blame.json from scratch (``blame_report.py``'s job): it never touches GitHub
compare/discovery, so it cannot change *which* PRs are candidates, and it
never touches package provenance, so an old release that has since aged off
CVMFS does not cost this run any coverage.

For each night under ``{--reports-dir}/{night}/`` that has both a
``report.json`` and a ``blame.json`` with at least one candidate PR:

  1. Load ``report.json`` for the per-metric facts (direction, pct_change,
     ...) that ``blame.json`` itself does not carry.
  2. Load ``blame.json`` and drop any entry whose exact identity (detector,
     platform, sample, label, metric, sub_detector, window) no longer matches
     a CONFIRMED regression in ``report.json`` — the same staleness rule
     :meth:`~k4bench.blame.models.BlameReport.entry_for` applies, just run in
     reverse. A confirmation the engine has since revised must not get a
     freshly-ranked sidecar entry.
  3. Group the remaining entries the same way the builder does — one rank
     call per (detector, platform, sample, base_release, onset_release) — and
     refetch each already-known candidate PR's diff by number
     (:func:`k4bench.blame.github.fetch_pr`; a merged PR's diff is immutable,
     so this reproduces the original ranker input, just never persisted).
  4. Rank the group with the configured model and fold the fresh
     score/description onto each entry's matching candidates.

Ranking is skipped (existing scores/descriptions kept) for any group whose
stored discovery was already incomplete (rate-limited or truncated at
build time — a partial candidate set must not be re-judged as complete), or
whose window has no live verdict per (2) above.

Nights run concurrently (``--workers``; this is an I/O-bound workload —
GitHub + LLM HTTP calls). Each worker thread builds its own
``GitHubClient``/``Ranker``, never sharing a ``requests.Session`` across
threads — this codebase has already chased one hard-to-reproduce concurrency
bug from pooled-connection reuse (``k4bench/remote.py``, PR #92); a fresh
session per thread avoids the whole class of it.

Default is a dry run: prints what would change and writes nothing. ``--apply``
writes each updated ``blame.json`` back to its EOS path, atomically (a
same-directory temp file, then ``os.replace``); ``--archive-dir`` optionally
copies the previous ``blame.json`` there first.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    sys.path.remove(str(_REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPO_ROOT))

_log = logging.getLogger("backfill_blame_rank")

_DEFAULT_REPORTS_DIR = "/eos/user/j/jbeirer/k4bench/_reports"
#: Cap on the "one worker per night" default — this is I/O-bound (GitHub + LLM
#: HTTP), so more threads than nights buys nothing, and an uncapped default
#: would open one connection to each endpoint per night as the report history
#: grows. An explicit --workers overrides this outright.
_MAX_DEFAULT_WORKERS = 32


# ── Pure re-rank logic (offline-testable: report/blame are already-parsed
#    objects, github/ranker are injected) ───────────────────────────────────

@dataclass
class NightResult:
    night: str
    status: str  # "no-blame" | "no-candidates" | "reranked" | "error"
    groups_total: int = 0
    groups_reranked: int = 0
    groups_incomplete: int = 0
    groups_declined: int = 0
    entries_stale: int = 0
    candidates_reranked: int = 0
    blame: object = None  # BlameReport | None — the (possibly updated) sidecar
    diff_lines: list = field(default_factory=list)
    error: str | None = None

    @property
    def changed(self) -> bool:
        return self.status == "reranked" and self.groups_reranked > 0


def _live_verdict_keys(report) -> dict[tuple, list]:
    """``report``'s attributable CONFIRMED verdicts, grouped by the same
    (detector, platform, sample, base_release, onset_release) rank group the
    builder groups on — mirrors
    :func:`k4bench.blame.builder.build_blame_report`'s ``verdicts_by_rank_group``
    exactly, so a rerank ranks against the identical metric set the original
    build did."""
    out: dict[tuple, list] = {}
    for v in report.regressions:
        base, onset = v.last_accepted_run_date, v.onset_run_date
        if not (onset and base and base < onset):
            continue
        key = (v.detector, v.platform, v.sample, base, onset)
        out.setdefault(key, []).append(v)
    return out


def _entry_identity(e) -> tuple:
    return (e.detector, e.platform, e.sample, e.label, e.metric, e.sub_detector,
            e.base_release, e.onset_release)


def _live_entry_identities(report) -> set[tuple]:
    """Every current CONFIRMED verdict's identity+window, in a
    :class:`~k4bench.blame.models.BlameEntry`-shaped tuple — an entry whose
    identity is not in this set is stale (its verdict was revoked or its
    window moved since ``blame.json`` was written) and must not be reranked."""
    out = set()
    for v in report.regressions:
        base, onset = v.last_accepted_run_date, v.onset_run_date
        if not (onset and base and base < onset):
            continue
        out.add((v.detector, v.platform, v.sample, v.label, v.metric,
                  v.sub_detector, base, onset))
    return out


def _rank_group(e) -> tuple:
    return (e.detector, e.platform, e.sample, e.base_release, e.onset_release)


def _metric_steps(verdicts) -> tuple:
    from k4bench.blame.rank import MetricStep
    return tuple(
        MetricStep(
            metric=v.metric, metric_family=v.metric_family,
            direction=v.direction.value, pct_change=v.pct_change,
            label=v.label, sub_detector=v.sub_detector,
        )
        for v in verdicts
    )


def _candidates_for_group(group_entries, github) -> tuple:
    """Every distinct candidate PR across *group_entries*' repos, with a
    freshly-fetched diff sample — the population is exactly what the original
    build recorded, never rediscovered, so a rerank can only refresh scores,
    never add or drop a candidate."""
    from k4bench.blame.github import fetch_pr
    from k4bench.blame.rank import RankCandidate

    seen: dict[tuple[str, int], object] = {}
    for entry in group_entries:
        for repo in entry.repos:
            for pr in repo.candidates:
                key = (pr.repo, pr.number)
                if key in seen:
                    continue
                patch = ""
                if github is not None:
                    try:
                        fetched = fetch_pr(github, pr.repo, pr.number)
                    except Exception:
                        _log.exception("fetch_pr failed for %s#%s — ranking from metadata only",
                                        pr.repo, pr.number)
                        fetched = None
                    if fetched is not None:
                        _, patch = fetched
                seen[key] = RankCandidate(
                    repo=pr.repo, number=pr.number, title=pr.title,
                    files=pr.files, patch=patch,
                )
    return tuple(seen.values())


def _apply_rankings(repo, rankings: dict) -> object:
    candidates = tuple(
        replace(pr, score=ranking.score, description=ranking.description)
        if (ranking := rankings.get((pr.repo, pr.number))) is not None else pr
        for pr in repo.candidates
    )
    return replace(repo, candidates=candidates)


def rerank_report(night: str, report, blame, *, github, ranker) -> NightResult:
    """The offline core: given an already-parsed ``report`` (NightlyReport)
    and ``blame`` (BlameReport), return a :class:`NightResult` carrying the
    updated sidecar (or the original, unchanged, when nothing could be
    reranked)."""
    from k4bench.blame.rank import RankRequest

    if not any(c for e in blame.entries for r in e.repos for c in r.candidates):
        return NightResult(night, "no-candidates", blame=blame)

    live_identities = _live_entry_identities(report)
    verdicts_by_group = _live_verdict_keys(report)

    entries = list(blame.entries)
    groups: dict[tuple, list[int]] = {}
    entries_stale = 0
    for i, e in enumerate(entries):
        if _entry_identity(e) not in live_identities:
            entries_stale += 1
            continue
        groups.setdefault(_rank_group(e), []).append(i)

    result = NightResult(
        night, "reranked", groups_total=len(groups), entries_stale=entries_stale,
    )

    for group_key, idxs in groups.items():
        group_entries = [entries[i] for i in idxs]
        if any(e.discovery_incomplete for e in group_entries):
            result.groups_incomplete += 1
            continue
        verdicts = verdicts_by_group.get(group_key)
        if not verdicts:
            # Defensive only: every entry here passed the identity-liveness
            # check above, which already guarantees a same-window verdict.
            _log.warning("%s: group %s has live entries but no matching verdict "
                         "— skipping", night, group_key)
            continue
        candidates = _candidates_for_group(group_entries, github)
        if not candidates:
            continue
        detector, platform, sample, base, onset = group_key
        request = RankRequest(
            metrics=_metric_steps(verdicts), detector=detector, platform=platform,
            sample=sample, base_release=base, onset_release=onset, candidates=candidates,
        )
        try:
            rankings = ranker.rank(request)
        except Exception:
            _log.exception("%s: ranker raised for group %s", night, group_key)
            rankings = {}
        if not rankings:
            result.groups_declined += 1
            continue
        result.groups_reranked += 1
        result.candidates_reranked += len(rankings)

        old_by_key = {}
        for e in group_entries:
            for r in e.repos:
                for pr in r.candidates:
                    old_by_key.setdefault((pr.repo, pr.number), (pr.score, pr.description))
        for key, ranking in sorted(rankings.items()):
            old_score, old_desc = old_by_key.get(key, (None, ""))
            repo, number = key
            arrow = f"{old_score:.0f} -> {ranking.score:.0f}" if old_score is not None else f"(new) -> {ranking.score:.0f}"
            result.diff_lines.append(
                f"    {repo}#{number}  {arrow}  \"{ranking.description}\""
                + (f"  [was: \"{old_desc}\"]" if old_desc and old_desc != ranking.description else "")
            )

        for i in idxs:
            entries[i] = replace(entries[i], repos=tuple(
                _apply_rankings(r, rankings) for r in entries[i].repos
            ))

    from k4bench.blame.models import BlameReport
    from datetime import datetime, timezone
    result.blame = BlameReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        report_night=blame.report_night,
        entries=tuple(entries),
    )
    return result


# ── I/O: one night, given real filesystem paths and factories for the
#    per-thread GitHub client / ranker ──────────────────────────────────────

def rerank_night(
    night: str,
    reports_dir: Path,
    *,
    make_github,
    make_ranker,
) -> NightResult:
    from k4bench.blame.models import BlameReport, BlameSchemaError
    from k4bench.regression.render import from_json

    blame_path = reports_dir / night / "blame.json"
    report_path = reports_dir / night / "report.json"
    if not blame_path.is_file():
        return NightResult(night, "no-blame")
    try:
        blame = BlameReport.from_json(json.loads(blame_path.read_text()))
    except (BlameSchemaError, ValueError, OSError) as exc:
        return NightResult(night, "error", error=f"unreadable blame.json: {exc}")

    if not any(c for e in blame.entries for r in e.repos for c in r.candidates):
        return NightResult(night, "no-candidates")

    if not report_path.is_file():
        return NightResult(night, "error", error=f"no report.json at {report_path}")
    try:
        report = from_json(json.loads(report_path.read_text()))
    except (ValueError, OSError) as exc:
        return NightResult(night, "error", error=f"unreadable report.json: {exc}")

    return rerank_report(
        night, report, blame, github=make_github(), ranker=make_ranker()
    )


def _write_blame_atomic(path: Path, blame) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(blame.to_json(), indent=2) + "\n")
    os.replace(tmp, path)  # same-directory rename: atomic on the EOS mount


def _archive(path: Path, archive_dir: Path, night: str) -> None:
    if not path.is_file():
        return
    dest_dir = archive_dir / night
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest_dir / "blame.json")


def _summary_line(r: NightResult) -> str:
    if r.status == "no-blame":
        return f"{r.night}: no blame.json — skipped"
    if r.status == "no-candidates":
        return f"{r.night}: blame.json has no candidate PRs — skipped"
    if r.status == "error":
        return f"{r.night}: ERROR — {r.error}"
    bits = [f"{r.groups_reranked}/{r.groups_total} group(s) reranked",
            f"{r.candidates_reranked} candidate PR(s) refreshed"]
    if r.groups_incomplete:
        bits.append(f"{r.groups_incomplete} group(s) skipped (incomplete discovery)")
    if r.entries_stale:
        bits.append(f"{r.entries_stale} entry(ies) skipped (stale — revoked/moved since)")
    if r.groups_declined:
        bits.append(f"{r.groups_declined} skipped (ranker declined)")
    return f"{r.night}: " + ", ".join(bits)


def _ranker_from_args(args) -> object | None:
    from k4bench.blame.llm import ChatClient
    from k4bench.blame.rank import OpenAICompatRanker, ranker_from_env

    if not args.llm_url and not args.llm_model:
        return ranker_from_env()
    if not (args.llm_url and args.llm_model):
        _log.error("--llm-url and --llm-model must both be set to override the ranker")
        return None
    kwargs = {}
    if args.llm_max_tokens:
        kwargs["max_tokens"] = args.llm_max_tokens
    return OpenAICompatRanker(client=ChatClient(
        url=args.llm_url, model=args.llm_model, api_key=args.llm_api_key, **kwargs
    ))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reports-dir", default=_DEFAULT_REPORTS_DIR,
        help=f"EOS reports root, one subdir per night (default: {_DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--night", action="append", dest="nights",
        help="Only rerank this night (YYYY-MM-DD); repeatable. Default: every "
             "night under --reports-dir",
    )
    parser.add_argument(
        "--github-token", default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token for refetching PR diffs (default: $GITHUB_TOKEN); "
             "without one, ranking falls back to title/files only",
    )
    parser.add_argument("--llm-url", default=None, help="Override $K4BENCH_LLM_URL")
    parser.add_argument("--llm-model", default=None, help="Override $K4BENCH_LLM_MODEL")
    parser.add_argument("--llm-api-key", default=os.environ.get("K4BENCH_LLM_API_KEY"),
                         help="Override $K4BENCH_LLM_API_KEY")
    parser.add_argument("--llm-max-tokens", type=int, default=None,
                         help="Override $K4BENCH_LLM_MAX_TOKENS")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Nights reranked concurrently (each gets its own GitHub/LLM HTTP "
             f"session). Default: one worker per night, capped at {_MAX_DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write updated blame.json back to EOS. Without this, dry run only.",
    )
    parser.add_argument(
        "--archive-dir", type=Path, default=None,
        help="Copy each night's pre-rerank blame.json here (under {night}/blame.json) "
             "before overwriting; only used with --apply",
    )
    parser.add_argument(
        "--no-diff", action="store_true",
        help="Suppress the per-candidate old->new score/description lines — "
             "print only the per-night summary",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _resolve_nights(reports_dir: Path, requested: list[str] | None) -> list[str] | str:
    """The sorted night list to process, or an error message string."""
    if not requested:
        return sorted(p.name for p in reports_dir.iterdir() if p.is_dir())
    nights = sorted(requested)
    missing = [n for n in nights if not (reports_dir / n).is_dir()]
    if missing:
        return f"no such night(s) under {reports_dir}: {', '.join(missing)}"
    return nights


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ranker = _ranker_from_args(args)
    if ranker is None:
        print(
            "ERROR: no ranker configured — set K4BENCH_LLM_URL/K4BENCH_LLM_MODEL "
            "or pass --llm-url/--llm-model", file=sys.stderr,
        )
        return 1
    _log.info("ranking with model=%s url=%s", getattr(ranker, "model", "?"),
               getattr(ranker, "url", "?"))
    if not args.github_token:
        _log.warning("no GitHub token — PR diffs cannot be refetched; ranking "
                      "from title/files only")

    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_dir():
        print(f"ERROR: no such reports dir: {reports_dir}", file=sys.stderr)
        return 1

    nights = _resolve_nights(reports_dir, args.nights)
    if isinstance(nights, str):
        print(f"ERROR: {nights}", file=sys.stderr)
        return 1
    workers = args.workers or max(1, min(len(nights), _MAX_DEFAULT_WORKERS))

    def make_github():
        from k4bench.blame.github import GitHubClient
        return GitHubClient(token=args.github_token) if args.github_token else None

    def make_ranker():
        return _ranker_from_args(args)

    _log.info(
        "%s mode — %d night(s), %d worker(s). GitHub/LLM calls happen either way "
        "(a rerank result is only real if the LLM actually saw it); --apply only "
        "gates the EOS write.",
        "APPLY" if args.apply else "DRY RUN", len(nights), workers,
    )

    results: list[NightResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(rerank_night, night, reports_dir,
                        make_github=make_github, make_ranker=make_ranker): night
            for night in nights
        }
        for future in as_completed(futures):
            night = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                _log.exception("%s: worker raised", night)
                results.append(NightResult(night, "error", error=str(exc)))

    results.sort(key=lambda r: r.night)
    for r in results:
        print(_summary_line(r))
        if not args.no_diff:
            for line in r.diff_lines:
                print(line)

    changed = [r for r in results if r.changed]
    total_groups = sum(r.groups_reranked for r in results)
    total_candidates = sum(r.candidates_reranked for r in results)
    print(
        f"\n{len(changed)}/{len(results)} night(s) have fresh rankings — "
        f"{total_groups} group(s), {total_candidates} candidate PR(s) total"
    )

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to write to EOS.")
        return 0

    written = 0
    for r in changed:
        blame_path = reports_dir / r.night / "blame.json"
        if args.archive_dir:
            _archive(blame_path, args.archive_dir, r.night)
        _write_blame_atomic(blame_path, r.blame)
        written += 1
        _log.info("%s: wrote %s", r.night, blame_path)
    print(f"\nWrote {written} updated blame.json file(s) to {reports_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
