"""Unit tests for the email gate (:mod:`k4bench.regression.notify`).

``smtplib`` is always stubbed — no test may hand mail to a real relay.
"""

from __future__ import annotations

import dataclasses
import email
import json

import pytest

from k4bench.blame.models import BlameEntry, BlameReport, CandidatePR, RepoBlame
from k4bench.regression import notify
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.regression.render import to_json


class _FakeSMTP:
    sent: list[tuple[str, list[str], str]] = []

    def __init__(self, host, port):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sendmail(self, from_addr, to_addrs, msg):
        _FakeSMTP.sent.append((from_addr, to_addrs, msg))


@pytest.fixture(autouse=True)
def fake_smtp(monkeypatch):
    _FakeSMTP.sent = []
    monkeypatch.setattr(notify.smtplib, "SMTP", _FakeSMTP)
    return _FakeSMTP


def _verdict(severity: Severity, direction: Direction) -> MetricVerdict:
    return MetricVerdict(
        detector="DET", platform="PLAT", sample="single_e", label="baseline",
        metric_family="time", metric="wall_time_s", sub_detector=None,
        run_id="2026-01-12", run_date="2026-01-12", value=120.0,
        baseline_median=100.0, baseline_mad=0.6, pct_change=0.2, z_score=33.0,
        severity=severity, direction=direction, reason="test",
    )


def _report(*verdicts: MetricVerdict, job_failures: list[str] | None = None) -> NightlyReport:
    group = RunGroupReport(
        detector="DET", platform="PLAT", sample="single_e",
        k4h_release="key4hep-2026-01-01", run_date="2026-01-12",
        run_id="2026-01-12", verdicts=list(verdicts),
        job_failures=job_failures or [],
    )
    return NightlyReport(generated_at="2026-01-12T06:00:00+00:00", groups=[group])


def _send(report: NightlyReport) -> bool:
    return notify.send_report_email(
        report, to_addr="egroup@cern.ch", from_addr="noreply@cern.ch"
    )


def test_confirmed_regression_sends_alert(fake_smtp):
    assert _send(_report(_verdict(Severity.CONFIRMED, Direction.UP)))
    ((frm, to, msg),) = fake_smtp.sent
    assert to == ["egroup@cern.ch"]
    assert "1 regression(s)" in msg


def test_job_failure_sends_alert(fake_smtp):
    assert _send(_report(job_failures=["no run uploaded for 2026-01-12"]))
    assert len(fake_smtp.sent) == 1


def test_confirmed_regression_sends_regardless_of_direction(fake_smtp):
    # Direction carries no good/bad judgment: a downward step alerts exactly
    # like an upward one — the report doesn't silently swallow half of what
    # it confirms just because it moved the "nice" way.
    assert _send(_report(_verdict(Severity.CONFIRMED, Direction.DOWN)))
    assert len(fake_smtp.sent) == 1


def test_clean_night_still_sends(fake_smtp):
    # Every night's report is emailed, regardless of content.
    assert _send(_report(
        _verdict(Severity.WATCH, Direction.UP),
        _verdict(Severity.OK, Direction.NONE),
    ))
    ((_frm, _to, msg),) = fake_smtp.sent
    assert "no regressions" in msg


def test_cli_sends_without_force_flag(fake_smtp, tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text('{"generated_at": "x", "groups": []}')
    assert notify.main([
        str(report_path), "--to", "egroup@cern.ch", "--from-addr", "noreply@cern.ch",
    ]) == 0
    assert len(fake_smtp.sent) == 1


def test_cli_skips_quietly_without_recipient(fake_smtp, tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text('{"generated_at": "x", "groups": []}')
    assert notify.main([str(report_path)]) == 0
    assert fake_smtp.sent == []


# ── Blame sidecar loading ─────────────────────────────────────────────────────

def _decoded_body(msg_str: str) -> str:
    """The message's text parts, base64/QP-decoded — the body substrings live
    here, not in the raw MIME string (only the ASCII Subject header does)."""
    message = email.message_from_string(msg_str)
    return "".join(
        part.get_payload(decode=True).decode("utf-8", "replace")
        for part in message.walk()
        if part.get_content_type() in ("text/plain", "text/html")
    )


def _blame_sidecar() -> BlameReport:
    return BlameReport(
        generated_at="2026-01-12T06:00:00", report_night="2026-01-12",
        entries=(BlameEntry(
            detector="DET", platform="PLAT", sample="single_e", label="baseline",
            metric="wall_time_s", sub_detector=None,
            base_release="2026-01-05", onset_release="2026-01-09",
            repos=(RepoBlame(
                package="k4geo", repo="key4hep/k4geo",
                base_commit="a" * 40, head_commit="c" * 40,
                compare_url="https://github.com/key4hep/k4geo/compare/a...c",
                status="changed",
                candidates=(CandidatePR(
                    repo="key4hep/k4geo", number=1234, title="Lower the step limit",
                    author="alice", url="https://github.com/key4hep/k4geo/pull/1234",
                    score=72.0, description="raises the step count",
                ),),
            ),),
        ),),
    )


def _confirmed_report_path(tmp_path):
    """A report whose confirmed verdict carries the window
    :func:`_blame_sidecar` attributes — the join needs identity *and* window."""
    verdict = dataclasses.replace(
        _verdict(Severity.CONFIRMED, Direction.UP),
        onset_run_id="2026-01-09", onset_run_date="2026-01-09",
        last_accepted_run_id="2026-01-05", last_accepted_run_date="2026-01-05",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(to_json(_report(verdict))))
    return report_path


def test_load_blame_reads_sidecar(tmp_path):
    report_path = _confirmed_report_path(tmp_path)
    (tmp_path / "blame.json").write_text(json.dumps(_blame_sidecar().to_json()))
    blame = notify._load_blame(str(report_path))
    assert blame is not None
    assert blame.entries[0].candidates[0].number == 1234


def test_load_blame_absent_returns_none(tmp_path):
    assert notify._load_blame(str(tmp_path / "report.json")) is None


def test_load_blame_malformed_returns_none(tmp_path):
    report_path = _confirmed_report_path(tmp_path)
    (tmp_path / "blame.json").write_text("{ not valid json")
    assert notify._load_blame(str(report_path)) is None


def test_load_blame_wrong_schema_returns_none(tmp_path):
    # Valid JSON that is not a blame report (entries missing required fields)
    # must degrade exactly like invalid JSON — never block the email.
    report_path = _confirmed_report_path(tmp_path)
    (tmp_path / "blame.json").write_text('{"entries": [{"detector": "DET"}]}')
    assert notify._load_blame(str(report_path)) is None


def test_cli_renders_the_blame_lead_when_present(fake_smtp, tmp_path):
    report_path = _confirmed_report_path(tmp_path)
    (tmp_path / "blame.json").write_text(json.dumps(_blame_sidecar().to_json()))
    assert notify.main([
        str(report_path), "--to", "egroup@cern.ch", "--from-addr", "noreply@cern.ch",
    ]) == 0
    ((_frm, _to, msg),) = fake_smtp.sent
    body = _decoded_body(msg)
    assert "key4hep/k4geo#1234" in body
    assert "72%" in body
    assert "raises the step count" in body


def test_cli_still_sends_when_blame_is_malformed(fake_smtp, tmp_path):
    report_path = _confirmed_report_path(tmp_path)
    (tmp_path / "blame.json").write_text("{ not valid json")
    assert notify.main([
        str(report_path), "--to", "egroup@cern.ch", "--from-addr", "noreply@cern.ch",
    ]) == 0
    assert len(fake_smtp.sent) == 1  # the email is never blocked by a bad sidecar
