"""E-group email delivery for the nightly regression report.

Stdlib-only (``smtplib``/``email``): the nightly job runs on CERN-network
runners, so mail goes directly through CERN's outbound relay without
authentication — no SMTP secrets to manage. If the relay ever starts
rejecting these runners (SPF/relay ACL), fall back to an authenticated
SMTP action with repo secrets.

Every night's report is emailed, regardless of content — regressions,
failures, or a clean run all send.

Runnable as ``python -m k4bench.regression.notify report.json --to …`` — the
module gates itself only on missing recipient config (exits quietly when
none is set), so the CI step can run unconditionally.
"""

from __future__ import annotations

import argparse
import json
import logging
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from pathlib import Path

from k4bench.blame.models import BlameReport, BlameSchemaError
from k4bench.regression.email import subject as email_subject
from k4bench.regression.email import to_html, to_markdown
from k4bench.regression.models import NightlyReport
from k4bench.regression.render import from_json
from k4bench.remote import fetch_blame

_log = logging.getLogger(__name__)

#: CERN outbound mail relay, reachable unauthenticated from CERN-network
#: runners. Confirm with CERN IT if delivery starts failing.
DEFAULT_SMTP_HOST = "cernmx.cern.ch"
DEFAULT_SMTP_PORT = 25


def send_report_email(
    report: NightlyReport,
    *,
    to_addr: str,
    from_addr: str,
    smtp_host: str = DEFAULT_SMTP_HOST,
    smtp_port: int = DEFAULT_SMTP_PORT,
    dashboard_url: str | None = None,
    actions_url: str | None = None,
    blame: BlameReport | None = None,
    historical_blame: dict[str, BlameReport] | None = None,
) -> bool:
    """Send the report to *to_addr*, every night, regardless of content.

    *blame*, when present, adds ranked candidate PRs under each confirmed
    regression it has attributed; *historical_blame* maps a
    ``first_confirmed_run_id`` to that night's sidecar, reused for same-release
    reconfirmations. Both are best-effort and never gate the email.

    Returns ``True`` once the mail has been handed to the relay.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = email_subject(report)
    msg["From"] = from_addr
    msg["To"] = to_addr
    # Standard automated-message headers so mail clients and the relay treat
    # this as machine-generated (no auto-replies, sortable date, unique id).
    msg["Date"] = formatdate(localtime=False)
    msg["Message-ID"] = make_msgid(domain="cern.ch")
    msg["Auto-Submitted"] = "auto-generated"
    # Plain-text fallback first, HTML preferred-last per MIME convention.
    msg.attach(MIMEText(
        to_markdown(
            report, dashboard_url=dashboard_url, actions_url=actions_url,
            blame=blame, historical_blame=historical_blame,
        ),
        "plain", "utf-8",
    ))
    msg.attach(MIMEText(
        to_html(
            report, dashboard_url=dashboard_url, actions_url=actions_url,
            blame=blame, historical_blame=historical_blame,
        ),
        "html", "utf-8",
    ))

    with smtplib.SMTP(smtp_host, smtp_port) as smtp:
        smtp.sendmail(from_addr, [to_addr], msg.as_string())
    _log.info("send_report_email: sent nightly report email to %s", to_addr)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Email the nightly regression report to the e-group "
                    "(no-ops when no recipient is configured)."
    )
    parser.add_argument("report", help="Path to report.json")
    parser.add_argument("--to", default="", help="Recipient (the CERN e-group address)")
    parser.add_argument("--from-addr", default="", help="Sender address")
    parser.add_argument("--smtp-host", default=DEFAULT_SMTP_HOST)
    parser.add_argument("--smtp-port", type=int, default=DEFAULT_SMTP_PORT)
    parser.add_argument("--dashboard-url", default=None)
    parser.add_argument("--actions-url", default=None)
    parser.add_argument(
        "--data-url", default=None,
        help="WebEOS base URL of the benchmark data. When set, first-confirmation "
             "blame sidecars are fetched to reuse attribution for same-release "
             "reconfirmations. Omitting it keeps offline/local rendering.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.to or not args.from_addr:
        # Repo variables not configured yet — never an error, so the nightly
        # job stays green until the e-group exists.
        _log.info("notify: recipient/sender not configured — skipping email")
        return 0

    with open(args.report) as fh:
        report = from_json(json.load(fh))
    blame = _load_blame(args.report)
    send_report_email(
        report,
        to_addr=args.to, from_addr=args.from_addr,
        smtp_host=args.smtp_host, smtp_port=args.smtp_port,
        dashboard_url=args.dashboard_url, actions_url=args.actions_url,
        blame=blame,
        historical_blame=_load_historical_blame(report, args.data_url),
    )
    return 0


def _load_blame(report_path: str) -> BlameReport | None:
    """The ``blame.json`` sitting beside *report_path*, or ``None``.

    Best-effort by contract: most nights write no sidecar (nothing to
    attribute), and a missing or malformed one must never block the email — the
    report's core is complete without it."""
    path = Path(report_path).with_name("blame.json")
    try:
        return BlameReport.from_json(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        _log.debug("notify: no blame sidecar at %s — %s", path, exc)
        return None


def _load_historical_blame(
    report: NightlyReport, data_url: str | None
) -> dict[str, BlameReport]:
    """First-confirmation sidecars needed to reuse attribution for tonight's
    same-release reconfirmations, keyed by ``first_confirmed_run_id``.

    Best-effort in every respect: with no *data_url* configured (offline/local
    rendering) this is empty and reconfirmed cards simply carry no reused
    ranking. Each distinct night is fetched at most once, and any missing,
    malformed, or timed-out sidecar silently degrades to no ranking for that
    reconfirmed group rather than blocking the email."""
    if not data_url:
        return {}
    nights = {
        v.first_confirmed_run_id
        for v in report.reconfirmed_regressions
        if v.first_confirmed_run_id
    }
    out: dict[str, BlameReport] = {}
    for night in sorted(nights):
        raw = fetch_blame(data_url, night)
        if not raw:
            continue
        try:
            out[night] = BlameReport.from_json(raw)
        except BlameSchemaError as exc:
            _log.debug("notify: malformed historical blame for %s — %s", night, exc)
    return out


if __name__ == "__main__":
    sys.exit(main())
