"""Unit tests for the email gate (:mod:`k4bench.regression.notify`).

``smtplib`` is always stubbed — no test may hand mail to a real relay.
"""

from __future__ import annotations

import pytest

from k4bench.regression import notify
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)


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


def test_watch_or_ok_only_never_sends(fake_smtp):
    assert not _send(_report(
        _verdict(Severity.WATCH, Direction.UP),
        _verdict(Severity.OK, Direction.NONE),
    ))
    assert fake_smtp.sent == []


def test_force_sends_clean_night(fake_smtp):
    # A manual dispatch with --force delivers even a non-alertable report.
    report = _report(_verdict(Severity.OK, Direction.NONE))
    assert notify.send_report_email(
        report, to_addr="egroup@cern.ch", from_addr="noreply@cern.ch", force=True
    )
    ((_frm, _to, msg),) = fake_smtp.sent
    assert "no regressions" in msg


def test_cli_force_flag_sends(fake_smtp, tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text('{"generated_at": "x", "groups": []}')
    assert notify.main([
        str(report_path), "--to", "egroup@cern.ch",
        "--from-addr", "noreply@cern.ch", "--force",
    ]) == 0
    assert len(fake_smtp.sent) == 1


def test_cli_skips_quietly_without_recipient(fake_smtp, tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text('{"generated_at": "x", "groups": []}')
    assert notify.main([str(report_path)]) == 0
    assert fake_smtp.sent == []
