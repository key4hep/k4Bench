#!/usr/bin/env python3
"""
Build the blame sidecar for a nightly report and write it as ``blame.json``.

Thin CLI over :func:`k4bench.blame.builder.build_blame_report`: reads the
already-built ``report.json``, and for each confirmed regression with a real
``(baseline, onset]`` release window, diffs the two releases' package maps and
asks GitHub which pull requests landed in each changed repo — writing

    {output-dir}/blame.json   — the sidecar the dashboard/email read back

This is best-effort and runs *after* ``report.json`` is built and uploaded: a
GitHub outage, a rate limit, or a force-pushed ``develop`` must never degrade the
nightly report or its email. The caller wraps this in ``|| true``; it also exits
0 whenever it produced a file, even an empty one (most nights have no confirmed,
attributable regression at all).

Provenance is read from local run directories the report build already cached
(``--cache-dir``) or a local tree (``--data-dir``); ``--data-url`` is a remote
fallback for a release whose runs are not in the cache. ``GITHUB_TOKEN`` (5000
req/hr) enables PR resolution; without it the diffs are still written, just with
no candidate PRs.

Candidate ranking is configured entirely by environment (``K4BENCH_LLM_URL`` /
``K4BENCH_LLM_MODEL`` / ``K4BENCH_LLM_API_KEY``, read by
:func:`k4bench.blame.rank.ranker_from_env`): with them set, each regression's
candidate PRs are scored 0–100 and described by a model reading the real diffs;
unset, candidates are written unranked. The endpoint must be *off-box* — this
runs on a benchmark machine, so ranking is a network call, never local compute.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path


def _local_packages(roots: list[str], platform: str, release: str) -> dict | None:
    """A release's ``k4h_packages`` from a local run tree, or ``None``.

    Both the CI run cache and the integration test's data tree share the EOS
    layout ``{det}/{platform}/{stack}/{sample}/{date}/run_info.json`` with
    ``stack == key4hep-{release}``. Every run under one release recorded the same
    stack, so the first readable one answers the question."""
    for root in roots:
        pattern = f"{root}/*/{platform}/key4hep-{release}/*/*/run_info.json"
        for path in sorted(glob.glob(pattern)):
            try:
                packages = json.loads(Path(path).read_text()).get("k4h_packages")
            except (OSError, ValueError):
                continue
            if packages:
                return packages
    return None


def _make_packages_for_release(
    roots: list[str], data_url: str | None, detectors_by_platform: dict[str, list[str]]
):
    """A ``(platform, release) -> packages`` lookup: local cache first, then an
    optional remote WebEOS fallback for a release the cache doesn't hold."""
    def packages_for_release(platform: str, release: str) -> dict | None:
        local = _local_packages(roots, platform, release)
        if local:
            return local
        if data_url:
            from k4bench.remote import fetch_stack_packages
            for detector in detectors_by_platform.get(platform, ()):
                packages = fetch_stack_packages(
                    data_url, detector, platform, f"key4hep-{release}"
                )
                if packages:
                    return packages
        return None

    return packages_for_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", default="report/report.json",
        help="The already-built report.json to attribute (default: report/report.json)",
    )
    parser.add_argument(
        "--output-dir", default="report",
        help="Where blame.json is written (default: report)",
    )
    parser.add_argument(
        "--cache-dir", default=os.environ.get("K4BENCH_CACHE_DIR"),
        help="Run cache the report build populated, read for provenance "
             "(default: $K4BENCH_CACHE_DIR)",
    )
    parser.add_argument(
        "--data-dir",
        help="Local run tree read for provenance instead of/in addition to the cache",
    )
    parser.add_argument(
        "--data-url", default=os.environ.get("K4BENCH_DATA_URL"),
        help="WebEOS base URL, used only as a provenance fallback "
             "(default: $K4BENCH_DATA_URL)",
    )
    parser.add_argument(
        "--github-token", default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token for PR resolution (default: $GITHUB_TOKEN); without "
             "it, diffs are written but no candidate PRs",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from k4bench.blame.builder import build_blame_report
    from k4bench.blame.github import GitHubClient
    from k4bench.blame.rank import ranker_from_env
    from k4bench.regression.render import from_json

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"ERROR: no report at {report_path}", file=sys.stderr)
        return 1
    report = from_json(json.loads(report_path.read_text()))

    detectors_by_platform: dict[str, list[str]] = {}
    for g in report.groups:
        detectors_by_platform.setdefault(g.platform, [])
        if g.detector not in detectors_by_platform[g.platform]:
            detectors_by_platform[g.platform].append(g.detector)

    roots = [r for r in (args.data_dir, args.cache_dir) if r]
    packages_for_release = _make_packages_for_release(
        roots, args.data_url, detectors_by_platform
    )
    github = GitHubClient(token=args.github_token) if args.github_token else None
    if github is None:
        logging.getLogger(__name__).warning(
            "blame_report: no GITHUB_TOKEN — writing diffs without candidate PRs"
        )

    ranker = ranker_from_env()
    if ranker is None:
        logging.getLogger(__name__).info(
            "blame_report: no K4BENCH_LLM_* config — candidates written unranked"
        )

    blame = build_blame_report(
        report, packages_for_release=packages_for_release, github=github, ranker=ranker
    )

    if not blame.entries:
        # Most nights have no confirmed, attributable regression. Writing nothing
        # keeps the sidecar's *presence* meaningful — a blame.json exists only
        # when there is blame — so the dashboard/email treat its absence as the
        # normal case rather than "an empty result to parse".
        print(f"blame for {blame.report_night or 'no data'}: nothing to attribute")
        return 0

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "blame.json").write_text(json.dumps(blame.to_json(), indent=2) + "\n")

    n_candidates = sum(len(e.candidates) for e in blame.entries)
    print(
        f"blame for {blame.report_night or 'no data'}: "
        f"{len(blame.entries)} attributed regression(s), "
        f"{n_candidates} candidate PR(s) -> {out / 'blame.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
