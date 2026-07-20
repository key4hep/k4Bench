#!/usr/bin/env python3
"""
Comment on the pull requests tonight's blame sidecar holds responsible.

Thin CLI over :mod:`k4bench.blame.comment` (which decides what is said, and to
whom) and :mod:`k4bench.blame.publish` (which writes it): reads the already-built
``report.json`` and its ``blame.json``, and for every confirmed regression whose
change window is attributed to a merged pull request at or above the configured
likelihood, upserts one comment on that pull request.

Runs last in the nightly job and is best-effort throughout: it is the only step
that writes outside this repository, so it must never be able to affect the
report, the sidecar, or the e-group email. Most nights it does nothing at all —
most nights have no confirmed regression, let alone a confidently attributed one.

Which repositories may be commented in, and how confident the ranker must be,
come from ``.github/blame-comments.yml`` (``--config``); the allowlist ships
empty, so the bot is inert until a repository is added there by pull request.
The config is parsed here rather than in the package for the same reason
``.github/benchmarks/*.yml`` is: the ``k4bench`` package stays free of a YAML
dependency, and every knob crosses the boundary as plain values.

Writing needs a token carrying ``pull-requests: write`` on the allowlisted
repositories — ``K4BENCH_PR_COMMENT_TOKEN``, deliberately *not* the workflow's
built-in ``GITHUB_TOKEN``, which is read-only and scoped to k4Bench alone.
Without it (or with ``--dry-run``) the rendered comments are logged and nothing
is written, which is how a new repository is checked before it is enabled.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Executing a file below ``.github/scripts`` otherwise puts that directory—not
# the checkout root—first on sys.path. Prefer the mounted checkout over a stale
# k4bench installation in long-lived CI/dev virtual environments.
_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    sys.path.remove(str(_REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPO_ROOT))

_log = logging.getLogger(__name__)


def _load_policy(path: Path, overrides: dict):
    """The comment policy from *path*, with any CLI *overrides* applied.

    A missing file is an empty policy (the bot is off), matching the shipped
    state; a malformed one raises, because a config that cannot be read must
    never be guessed at — see :class:`k4bench.blame.comment.CommentConfigError`.
    """
    import yaml

    from k4bench.blame.comment import CommentPolicy

    data = {}
    if path.is_file():
        data = yaml.safe_load(path.read_text()) or {}
    else:
        _log.info("blame_comment: no config at %s — the bot is off", path)
    if isinstance(data, dict):
        data.update({k: v for k, v in overrides.items() if v is not None})
    return CommentPolicy.from_config(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", default="report/report.json",
        help="The already-built report.json (default: report/report.json)",
    )
    parser.add_argument(
        "--blame", default="report/blame.json",
        help="The blame sidecar for that report (default: report/blame.json)",
    )
    parser.add_argument(
        "--config", default=".github/blame-comments.yml",
        help="Repository allowlist and thresholds "
             "(default: .github/blame-comments.yml)",
    )
    parser.add_argument("--dashboard-url", default=os.environ.get("K4BENCH_DASHBOARD_URL"))
    parser.add_argument(
        "--token", default=os.environ.get("K4BENCH_PR_COMMENT_TOKEN"),
        help="GitHub token with pull-requests:write on the allowlisted repos "
             "(default: $K4BENCH_PR_COMMENT_TOKEN); without one the run is a dry run",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        default=bool(os.environ.get("K4BENCH_PR_COMMENT_DRY_RUN")),
        help="Log the exact comments instead of posting them "
             "(also set by a non-empty $K4BENCH_PR_COMMENT_DRY_RUN)",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="Override the config's likelihood threshold for one run",
    )
    parser.add_argument(
        "--max-comments", type=int, default=None,
        help="Override the config's per-night comment cap for one run",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    from k4bench.blame.comment import CommentConfigError, select
    from k4bench.blame.github import GitHubClient
    from k4bench.blame.models import BlameReport, BlameSchemaError
    from k4bench.blame.publish import publish
    from k4bench.regression.render import from_json

    report_path, blame_path = Path(args.report), Path(args.blame)
    if not report_path.is_file():
        print(f"ERROR: no report at {report_path}", file=sys.stderr)
        return 1
    if not blame_path.is_file():
        # The common case: no confirmed regression tonight, or nothing the
        # sidecar could attribute. Nothing to say, and that is not a failure.
        print(f"No blame sidecar at {blame_path}: nothing to comment on")
        return 0

    try:
        policy = _load_policy(Path(args.config), {
            "min_score": args.min_score, "max_comments": args.max_comments,
        })
    except CommentConfigError as exc:
        print(f"ERROR: bad comment config {args.config}: {exc}", file=sys.stderr)
        return 1
    if not policy.enabled:
        print(f"No repositories enabled in {args.config}: nothing to comment on")
        return 0

    report = from_json(json.loads(report_path.read_text()))
    try:
        blame = BlameReport.from_json(json.loads(blame_path.read_text()))
    except (BlameSchemaError, ValueError) as exc:
        print(f"ERROR: unreadable blame sidecar {blame_path}: {exc}", file=sys.stderr)
        return 1

    comments = select(report, blame, policy, dashboard_url=args.dashboard_url)
    if not comments:
        print(
            f"blame comments for {blame.report_night or 'no data'}: no candidate "
            f"reached {policy.min_score:g}% in an enabled repository"
        )
        return 0

    dry_run = args.dry_run or not args.token
    if dry_run and not args.dry_run:
        _log.warning(
            "blame_comment: no K4BENCH_PR_COMMENT_TOKEN — dry run, nothing posted"
        )
    result = publish(GitHubClient(token=args.token), comments, dry_run=dry_run)
    print(
        f"blame comments for {blame.report_night or 'no data'}: {result.summary}"
    )
    # A write that failed is worth a red step *inside this isolated block* — the
    # caller already contains it, and silence would hide a revoked token forever.
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
