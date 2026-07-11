"""E-group email delivery for the nightly regression report.

Stdlib-only (``smtplib``/``email``): the nightly job runs on CERN-network
runners, so mail goes directly through CERN's outbound relay without
authentication — no SMTP secrets to manage. If the relay ever starts
rejecting these runners (SPF/relay ACL), fall back to an authenticated
SMTP action with repo secrets.

Send policy (see :func:`classify_report`):

- **any confirmed regression (either direction) or any hard failure** sends
  (alert). Direction carries no good/bad judgment — the report doesn't
  silently swallow half of what it confirms just because it moved the
  "nice" way, so both directions alert equally;
- everything else — ``WATCH``/``OK``/``UNKNOWN`` nights — never sends.

Runnable as ``python -m k4bench.regression.notify report.json --to …`` — the
module gates itself (exits quietly when there is nothing to send or no
recipient is configured), so the CI step can run unconditionally.
"""

from __future__ import annotations

import argparse
import json
import logging
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from k4bench.regression.models import NightlyReport
from k4bench.regression.render import from_json, to_html, to_markdown

_log = logging.getLogger(__name__)

#: CERN outbound mail relay, reachable unauthenticated from CERN-network
#: runners. Confirm with CERN IT if delivery starts failing.
DEFAULT_SMTP_HOST = "cernmx.cern.ch"
DEFAULT_SMTP_PORT = 25


def classify_report(report: NightlyReport) -> str | None:
    """``"alert"`` or ``None`` (nothing to send)."""
    return "alert" if report.has_alertable else None


def _subject(report: NightlyReport) -> str:
    night = report.report_night or "no data"
    n_fail = len(report.failures) + len(report.job_failures)
    parts = []
    if report.regressions:
        parts.append(f"{len(report.regressions)} regression(s)")
    if n_fail:
        parts.append(f"{n_fail} failure(s)")
    # A forced send on a clean night has nothing to enumerate — say so rather
    # than trailing an empty ": ".
    return f"[k4Bench] {night}: " + (", ".join(parts) or "no regressions")


def send_report_email(
    report: NightlyReport,
    *,
    to_addr: str,
    from_addr: str,
    smtp_host: str = DEFAULT_SMTP_HOST,
    smtp_port: int = DEFAULT_SMTP_PORT,
    dashboard_url: str | None = None,
    actions_url: str | None = None,
    force: bool = False,
) -> bool:
    """Send the report to *to_addr* if its content warrants it.

    Returns ``True`` when a mail was handed to the relay, ``False`` when the
    send policy said there was nothing to send. *force* bypasses the send
    policy so a clean night is delivered anyway — used by the manual
    ``workflow_dispatch`` trigger.
    """
    kind = classify_report(report) or ("forced" if force else None)
    if kind is None:
        _log.info("send_report_email: nothing alertable — not sending")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = _subject(report)
    msg["From"] = from_addr
    msg["To"] = to_addr
    # Plain-text fallback first, HTML preferred-last per MIME convention.
    msg.attach(MIMEText(to_markdown(report, dashboard_url=dashboard_url), "plain", "utf-8"))
    msg.attach(MIMEText(
        to_html(report, dashboard_url=dashboard_url, actions_url=actions_url),
        "html", "utf-8",
    ))

    with smtplib.SMTP(smtp_host, smtp_port) as smtp:
        smtp.sendmail(from_addr, [to_addr], msg.as_string())
    _log.info("send_report_email: sent %s email to %s", kind, to_addr)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Email the nightly regression report to the e-group "
                    "(no-ops when there is nothing to send or no recipient)."
    )
    parser.add_argument("report", help="Path to report.json")
    parser.add_argument("--to", default="", help="Recipient (the CERN e-group address)")
    parser.add_argument("--from-addr", default="", help="Sender address")
    parser.add_argument("--smtp-host", default=DEFAULT_SMTP_HOST)
    parser.add_argument("--smtp-port", type=int, default=DEFAULT_SMTP_PORT)
    parser.add_argument("--dashboard-url", default=None)
    parser.add_argument("--actions-url", default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Send even when the report is not alertable (manual dispatch)",
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
    send_report_email(
        report,
        to_addr=args.to, from_addr=args.from_addr,
        smtp_host=args.smtp_host, smtp_port=args.smtp_port,
        dashboard_url=args.dashboard_url, actions_url=args.actions_url,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
