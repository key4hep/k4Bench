#!/usr/bin/env python3
"""
Build the nightly regression report and write it to an output directory.

Thin CLI over :func:`k4bench.regression.report_builder.build_nightly_report`:
walks the EOS run history (or a local tree with the same layout), runs the
step detector over every (detector, platform, sample, config, metric) series,
and writes

    {output-dir}/report.json   — machine-readable (dashboard + email gating)
    {output-dir}/report.md     — human-readable summary (Actions artifact)

Exit code is 0 whenever the report was produced, regardless of its content —
alert delivery is gated on report.json's ``summary`` block by the workflow.

``--as-of`` rebuilds a *past* night instead of the newest one, truncating every
series to runs on or before it. That is how a report is regenerated after the
judging rules change: the same runs, walked by today's engine.

``--night`` reports a night *newer* than every run on EOS, as one missing run
per triple. That is a night whose benchmarking uploaded nothing at all: without
it the report would carry the previous night's date and verdicts again.

``--fanout-run-id`` is the nightly CI case of that: the report must hold a run
made by the named Actions run, or it is last night's report again and tonight
is reported as an outage instead (see :func:`outage_report`).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# Executing a file below ``.github/scripts`` otherwise puts that directory—not
# the checkout root—first on sys.path. Prefer the mounted checkout over a stale
# k4bench installation in long-lived CI/dev virtual environments: without this
# the report is built by whatever version happens to be installed, which reads
# as a code change silently not taking effect. The same preamble guards
# ``blame_report.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    sys.path.remove(str(_REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    from k4bench.regression.models import NightlyReport

_NIGHT = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _night(text: str) -> str:
    """argparse type for a night: nights are compared as strings throughout
    (``2026-9-8`` would sort after ``2026-09-07`` *and* name a new EOS
    directory), so only the canonical form is accepted."""
    if not _NIGHT.match(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a YYYY-MM-DD night")
    return text


def tonight() -> str:
    """The night a run started now would be filed under.

    The same clock and format as ``date +%Y-%m-%d`` in
    ``nightly_benchmark.sh`` — both run in the same container image — so an
    outage night is labelled by the convention the run directories follow.
    """
    return datetime.now().strftime("%Y-%m-%d")


def outage_report(
    build: Callable[..., NightlyReport], report: NightlyReport, run_id: str, night: str,
) -> NightlyReport:
    """Tonight's report when Actions run *run_id* uploaded nothing.

    A fan-out where *every* job failed uploads no run; the newest run on EOS
    is then still the previous night's, and publishing *report* would overwrite
    that night and mail the e-group its verdicts a second time. The report is
    rebuilt for *night* instead — one missing run per triple — so it says what
    happened. A partial failure never gets here: one surviving upload names the
    run, and the missing triples are already reported one by one.

    Refusing beats mislabelling, so this exits non-zero and publishes nothing
    when *night* is not newer than the newest run on EOS (that run's night is
    already reported, by the fan-out that made it) or when the rebuilt report
    would be empty (past ``MISSING_RUN_GRACE_DAYS`` every triple is retired,
    and an empty report is an all-clear, not an outage).
    """
    newest = report.report_night
    print(
        f"run {run_id} uploaded no run: reporting {night} as an outage instead "
        f"of republishing {newest or 'no data'}.", file=sys.stderr,
    )
    if not newest:
        raise SystemExit(
            "ERROR: no run on EOS at all, so no triple to report as missing "
            "tonight — publishing nothing"
        )
    if night <= newest:
        raise SystemExit(
            f"ERROR: tonight ({night}) is not newer than the newest run on EOS "
            f"({newest}), whose report is that run's fan-out's to publish — "
            "publishing nothing"
        )
    report = build(night=night)
    if report.report_night != night:
        raise SystemExit(
            f"ERROR: outage report is dated {report.report_night!r}, expected "
            f"{night} — publishing nothing"
        )
    if not report.groups:
        raise SystemExit(
            f"ERROR: outage report for {night} holds no run group at all "
            "(every triple retired?) — publishing nothing"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--data-url",
        default=os.environ.get("K4BENCH_DATA_URL"),
        help="WebEOS base URL of the benchmark data "
             "(default: $K4BENCH_DATA_URL)",
    )
    source.add_argument(
        "--data-dir",
        help="Local directory tree with the EOS layout instead of a URL "
             "(offline mode, used by the integration test)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("K4BENCH_CACHE_DIR"),
        help="Download cache for --data-url mode (default: $K4BENCH_CACHE_DIR)",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Where report.json/report.md are written",
    )
    night_source = parser.add_mutually_exclusive_group()
    night_source.add_argument(
        "--as-of", type=_night,
        help="Judge the history as it stood on this night (YYYY-MM-DD): every "
             "series is truncated to runs on or before it, and the report night "
             "becomes the newest run that survives. Rebuilds a past night's "
             "report exactly as that night would have built it; omit for the "
             "nightly CI case, which judges everything uploaded so far",
    )
    night_source.add_argument(
        "--night", type=_night,
        help="Report this night (YYYY-MM-DD) even though no run carries it: "
             "every triple becomes a missing run for it. For a night whose "
             "benchmarking uploaded nothing. Ignored when a run is dated on or "
             "after it — a report covers the runs it holds",
    )
    night_source.add_argument(
        "--fanout-run-id",
        default=os.environ.get("K4BENCH_FANOUT_RUN_ID") or None,
        help="Actions run id of the benchmark fan-out this report must cover "
             "(default: $K4BENCH_FANOUT_RUN_ID). A report holding none of its "
             "runs is last night's again: tonight is then reported as an "
             "outage, as with --night, or nothing is written and the exit "
             "code is non-zero. Omit for a manual rebuild of whatever EOS holds",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from k4bench.regression.email import to_markdown
    from k4bench.regression.render import to_json
    from k4bench.regression.report_builder import (
        build_nightly_report,
        build_nightly_report_local,
        report_covers_run,
    )

    if args.data_dir:
        build = partial(build_nightly_report_local, args.data_dir, as_of=args.as_of)
    elif args.data_url:
        build = partial(
            build_nightly_report, args.data_url, args.cache_dir, as_of=args.as_of,
        )
    else:
        parser.error("either --data-url (or $K4BENCH_DATA_URL) or --data-dir is required")

    report = build(night=args.night)
    if args.fanout_run_id and not report_covers_run(report, args.fanout_run_id):
        report = outage_report(build, report, args.fanout_run_id, tonight())

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = to_json(report)
    (out / "report.json").write_text(json.dumps(data, indent=2) + "\n")
    (out / "report.md").write_text(to_markdown(report))

    s = data["summary"]
    print(
        f"report for {s['report_night'] or 'no data'}: "
        f"{s['n_detectors']} detector(s), {s['n_regressions']} regression(s), "
        f"{s['n_failures']} failure(s), "
        f"{s['n_watches']} on watch -> {out / 'report.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
