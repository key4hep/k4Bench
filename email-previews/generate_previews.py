#!/usr/bin/env python3
"""Render the nightly k4Bench email as HTML from real published reports.

Manual/integration tool — **not** covered by the unit suite (those must never
touch the network). Fetches ``report.json`` / ``blame.json`` from the WebEOS
data host, runs the *production* renderer (:mod:`k4bench.regression.email`), and
writes one full-report HTML preview per requested night. It never sends email
and fails clearly on any HTTP error rather than emitting a partial preview.

Same-release reconfirmations reuse the first-confirmation night's sidecar, so a
night's needed historical sidecars are fetched exactly the way
:mod:`k4bench.regression.notify` does in production.

Usage::

    python email-previews/generate_previews.py 2026-06-27 2026-06-28
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Prefer the repo checkout over any installed copy, so the preview always uses
# the working tree's renderer (this file lives one level down from the root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from k4bench.blame.models import BlameReport, BlameSchemaError
from k4bench.regression.email import subject, to_html
from k4bench.regression.models import NightlyReport
from k4bench.regression.render import from_json

DATA_URL = "https://k4bench-data.web.cern.ch"
DASHBOARD_URL = "https://k4bench-dashboard.app.cern.ch"


def _fetch_json(url: str, *, required: bool) -> dict | None:
    """GET *url* as JSON. A missing optional file (404) returns ``None``; any
    other HTTP/parse error is fatal — a preview must never silently drop data."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and not required:
            return None
        raise SystemExit(f"HTTP {exc.code} fetching {url}") from exc
    except (urllib.error.URLError, ValueError) as exc:
        raise SystemExit(f"failed fetching {url}: {exc}") from exc


def _load_blame(night: str) -> BlameReport | None:
    raw = _fetch_json(f"{DATA_URL}/_reports/{night}/blame.json", required=False)
    if not raw:
        return None
    try:
        return BlameReport.from_json(raw)
    except BlameSchemaError as exc:
        print(f"  warning: malformed blame.json for {night} — {exc}", file=sys.stderr)
        return None


def _historical_blame(report: NightlyReport) -> dict[str, BlameReport]:
    """First-confirmation sidecars for tonight's reconfirmations, fetched once
    per unique night — mirrors :func:`k4bench.regression.notify._load_historical_blame`."""
    nights = {
        v.first_confirmed_run_id
        for v in report.reconfirmed_regressions
        if v.first_confirmed_run_id
    }
    out: dict[str, BlameReport] = {}
    for night in sorted(nights):
        blame = _load_blame(night)
        if blame is not None:
            out[night] = blame
    return out


def _standalone(body: str, night: str) -> str:
    """Wrap the production email body in a minimal ``charset``-declaring
    document so a browser opening the ``file://`` preview decodes the UTF-8
    correctly. In the real email this is unnecessary — the MIME part carries the
    charset — so the renderer emits the body alone and only this preview tool
    adds the wrapper."""
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>k4Bench nightly — {night}</title></head>"
        f"<body>{body}</body></html>\n"
    )


def _counts(report: NightlyReport) -> str:
    return (
        f"{len(report.new_regressions)} new · "
        f"{len(report.reconfirmed_regressions)} reconfirmed · "
        f"{len(report.watches)} watch · "
        f"{len(report.failures) + len(report.job_failures)} failures"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nights", nargs="+", help="Report nights, e.g. 2026-06-27")
    parser.add_argument("--out-dir", default=str(Path(__file__).parent))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for night in args.nights:
        print(f"[{night}] fetching report + blame …")
        report = from_json(_fetch_json(f"{DATA_URL}/_reports/{night}/report.json", required=True))
        blame = _load_blame(night)
        historical = _historical_blame(report)

        html = to_html(
            report, dashboard_url=DASHBOARD_URL, actions_url=None,
            blame=blame, historical_blame=historical,
        )
        path = out_dir / f"k4bench-nightly-{night}.html"
        path.write_text(_standalone(html, night), encoding="utf-8")
        size = path.stat().st_size
        print(f"  subject: {subject(report)}")
        print(f"  counts:  {_counts(report)}")
        print(f"  wrote:   {path.name} ({size:,} bytes)")
        if historical:
            print(f"  reused historical sidecars: {', '.join(sorted(historical))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
