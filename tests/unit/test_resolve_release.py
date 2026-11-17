"""Tests for the nightly LCG-view gate."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/resolve_release.sh"
PLATFORM = "test-platform"
LATEST = "latest"


class Gate:
    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "views/test-view"
        self.nightlies = tmp_path / "nightlies/test-view"
        now = datetime.now(timezone.utc)
        self.today = now.strftime("%Y-%m-%d")
        self.yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        self.sleep_log = tmp_path / "sleeps"
        self._out = tmp_path / "github_output"
        self._summary = tmp_path / "github_summary"

        old = self.publish(self.yesterday)
        (self.root / LATEST).mkdir(parents=True)
        (self.root / LATEST / PLATFORM).symlink_to(old.parent, target_is_directory=True)

        self._bin = tmp_path / "bin"
        self._bin.mkdir()
        stub = self._bin / "sleep"
        stub.write_text(
            "#!/bin/bash\n"
            f'echo "$@" >> "{self.sleep_log}"\n'
            'if [[ -n "${SLEEP_FINISHES:-}" ]]; then\n'
            '  mkdir -p "$(dirname "${SLEEP_FINISHES}")"\n'
            '  touch "${SLEEP_FINISHES}"\n'
            "fi\n"
            'if [[ -n "${SLEEP_PUBLISHES:-}" ]]; then\n'
            '  mkdir -p "$(dirname "${SLEEP_PUBLISHES}")"\n'
            '  printf "# Generated: %s\\n" "${SLEEP_GENERATED}" > "${SLEEP_PUBLISHES}"\n'
            "fi\n"
        )
        stub.chmod(0o755)

    def marker(self, release: str) -> Path:
        slot = datetime.strptime(release, "%Y-%m-%d").strftime("%a")
        return self.nightlies / slot / f"isDone-{PLATFORM}"

    def publish(self, release: str, *, finished_before: float | None = 3600) -> Path:
        """Publish *release*'s view; its build finished *finished_before*
        seconds earlier (negative: after the view), or not at all (None)."""
        date = datetime.strptime(release, "%Y-%m-%d")
        setup = self.root / date.strftime("%a") / PLATFORM / "setup.sh"
        setup.parent.mkdir(parents=True, exist_ok=True)
        setup.write_text(f"# Generated: {date.strftime('%a %b %d 01:25:04 %Y')}\n")
        marker = self.marker(release)
        if finished_before is None:
            marker.unlink(missing_ok=True)
        else:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            view_time = setup.stat().st_mtime
            os.utime(marker, (view_time - finished_before,) * 2)
        return setup

    def run(self, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT)],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{self._bin}:{os.environ['PATH']}",
                "VIEWS_ROOT": str(self.root),
                "NIGHTLIES_ROOT": str(self.nightlies),
                "PLATFORM_DIR": PLATFORM,
                "LATEST_DIR": LATEST,
                "GITHUB_OUTPUT": str(self._out),
                "GITHUB_STEP_SUMMARY": str(self._summary),
                **env,
            },
        )

    @property
    def outputs(self) -> dict[str, str]:
        if not self._out.exists():
            return {}
        return dict(line.split("=", 1) for line in self._out.read_text().splitlines())

    @property
    def summary(self) -> str:
        return self._summary.read_text() if self._summary.exists() else ""

    @property
    def waits(self) -> int:
        return len(self.sleep_log.read_text().splitlines()) if self.sleep_log.exists() else 0


@pytest.fixture
def gate(tmp_path: Path) -> Gate:
    return Gate(tmp_path)


def _expected(gate: Gate, release: str, today: str) -> dict[str, str]:
    setup = gate.publish(release)
    return {"release": release, "is_today": today, "setup": str(setup.resolve())}


def test_todays_release_is_taken_without_waiting(gate: Gate):
    expected = _expected(gate, gate.today, "true")
    assert gate.run().returncode == 0
    assert gate.outputs == expected
    assert gate.waits == 0


def test_release_published_during_the_wait_is_picked_up(gate: Gate):
    date = datetime.strptime(gate.today, "%Y-%m-%d")
    setup = gate.root / date.strftime("%a") / PLATFORM / "setup.sh"
    proc = gate.run(
        SLEEP_FINISHES=str(gate.marker(gate.today)),
        SLEEP_PUBLISHES=str(setup),
        SLEEP_GENERATED=date.strftime("%a %b %d 01:25:04 %Y"),
    )
    assert proc.returncode == 0
    assert gate.outputs == {
        "release": gate.today,
        "is_today": "true",
        "setup": str(setup.resolve()),
    }
    assert gate.waits == 1


def test_retries_are_exhausted_before_giving_up(gate: Gate):
    assert gate.run(RETRIES="3").returncode == 0
    assert gate.waits == 3


def test_no_release_today_falls_back_to_latest_and_says_so(gate: Gate):
    proc = gate.run(RETRIES="1")
    assert proc.returncode == 0
    assert gate.outputs["release"] == gate.yesterday
    assert gate.outputs["is_today"] == "false"
    assert gate.outputs["setup"] == str(gate.publish(gate.yesterday).resolve())
    assert "::warning::" in proc.stdout
    assert "re-measurement" in gate.summary


def test_no_complete_view_fails_instead_of_publishing_empty_output(gate: Gate):
    gate.publish(gate.yesterday, finished_before=None)
    proc = gate.run(RETRIES="0")
    assert proc.returncode != 0
    assert "::error::" in proc.stdout
    assert gate.outputs == {}


@pytest.mark.parametrize(
    "finished_before",
    [None, -3600, 2 * 86400],
    ids=["build-unfinished", "build-finished-after-view", "stale-marker"],
)
def test_view_published_before_its_build_finished_is_not_taken(gate: Gate, finished_before):
    gate.publish(gate.today, finished_before=finished_before)
    proc = gate.run(RETRIES="2")
    assert proc.returncode == 0
    assert gate.waits == 2
    assert gate.outputs["release"] == gate.yesterday
    assert gate.outputs["is_today"] == "false"
    assert "build has not finished" in proc.stdout
    assert "before its build finished" in gate.summary


def test_view_regenerated_after_its_build_finishes_is_picked_up(gate: Gate):
    setup = gate.publish(gate.today, finished_before=None)
    date = datetime.strptime(gate.today, "%Y-%m-%d")
    proc = gate.run(
        SLEEP_FINISHES=str(gate.marker(gate.today)),
        SLEEP_PUBLISHES=str(setup),
        SLEEP_GENERATED=date.strftime("%a %b %d 01:25:04 %Y"),
    )
    assert proc.returncode == 0
    assert gate.outputs["release"] == gate.today
    assert gate.waits == 1


def test_fallback_takes_the_newest_complete_view_not_latest(gate: Gate):
    incomplete = gate.publish(gate.today, finished_before=None)
    (gate.root / LATEST / PLATFORM).unlink()
    (gate.root / LATEST / PLATFORM).symlink_to(incomplete.parent, target_is_directory=True)
    older = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
    stale = gate.publish(older)
    os.utime(stale, (stale.stat().st_mtime - 2 * 86400,) * 2)
    os.utime(gate.marker(older), (stale.stat().st_mtime - 3600,) * 2)
    assert gate.run(RETRIES="0").returncode == 0
    assert gate.outputs["release"] == gate.yesterday


def test_fallback_ranks_views_by_release_not_by_file_time(gate: Gate):
    gate.publish(gate.today, finished_before=None)
    older = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
    rewritten = gate.publish(older)
    later = gate.publish(gate.yesterday).stat().st_mtime + 600
    os.utime(rewritten, (later,) * 2)
    os.utime(gate.marker(older), (later - 3600,) * 2)
    assert gate.run(RETRIES="0").returncode == 0
    assert gate.outputs["release"] == gate.yesterday


def test_requested_release_whose_build_has_not_finished_fails(gate: Gate):
    gate.publish(gate.today, finished_before=None)
    proc = gate.run(RELEASE_REQUESTED=gate.today)
    assert proc.returncode != 0
    assert "before its build finished" in proc.stdout
    assert gate.outputs == {}


def test_requested_release_skips_the_wait(gate: Gate):
    proc = gate.run(RELEASE_REQUESTED=gate.yesterday, RETRIES="4")
    assert proc.returncode == 0
    assert gate.outputs["release"] == gate.yesterday
    assert gate.outputs["is_today"] == "false"
    assert gate.waits == 0
    assert "pinned" in gate.summary


def test_requested_release_matching_today_is_reported_as_today(gate: Gate):
    gate.publish(gate.today)
    assert gate.run(RELEASE_REQUESTED=gate.today).returncode == 0
    assert gate.outputs["is_today"] == "true"


@pytest.mark.parametrize("requested", ["2026-7-3", "latest", "2026-07-28-opt"])
def test_requested_release_must_be_a_date(gate: Gate, requested: str):
    proc = gate.run(RELEASE_REQUESTED=requested)
    assert proc.returncode != 0
    assert "::error::" in proc.stdout
    assert gate.outputs == {}


def test_requested_release_must_exist(gate: Gate):
    proc = gate.run(RELEASE_REQUESTED="2019-01-01")
    assert proc.returncode != 0
    assert "::error::" in proc.stdout
    assert gate.outputs == {}
