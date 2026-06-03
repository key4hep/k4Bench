"""Unit tests for the dashboard's remote WebEOS access layer.

``dashboard/remote.py`` is not part of the importable ``k4bench`` package and
its sibling modules (``data``/``config``) pull in Streamlit, so it is loaded here
in isolation by file path. ``remote`` itself only depends on stdlib + requests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REMOTE_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "remote.py"


def _load_remote():
    spec = importlib.util.spec_from_file_location("k4bench_dashboard_remote", _REMOTE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


remote = _load_remote()


# ---------------------------------------------------------------------------
# Fake WebEOS server
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str = "", content: bytes = b""):
        self.text = text
        self.content = content

    def raise_for_status(self) -> None:  # pragma: no cover - always ok here
        pass


def _apache_listing(names: list[str]) -> str:
    """Render an Apache-style directory listing the regex in remote.py parses."""
    rows = ['<a href="?C=N;O=D">Name</a>']  # a sort link, must be ignored
    for n in names:
        rows.append(f'<a href="{n}">{n}</a>')
    return "<html><body>" + "\n".join(rows) + "</body></html>"


class FakeWeb:
    """In-memory WebEOS tree. ``tree`` maps a URL path to either a list of child
    directory names (rendered as a listing) or to bytes (a file). Records every
    requested URL so tests can assert on HTTP volume."""

    def __init__(self, tree: dict[str, object]):
        self.tree = tree
        self.requested: list[str] = []

    def get(self, url: str, timeout: int | None = None) -> _FakeResponse:
        self.requested.append(url)
        key = url.rstrip("/")
        if key not in self.tree:
            raise AssertionError(f"unexpected URL requested: {url}")
        node = self.tree[key]
        if isinstance(node, bytes):
            return _FakeResponse(content=node)
        # directory → listing of (dir children with trailing slash, file children without)
        names = list(node)  # type: ignore[arg-type]
        return _FakeResponse(text=_apache_listing(names))


BASE = "https://eos.example/data"


def _build_tree() -> dict[str, object]:
    """Two stacks; sample present in both. Stack A has 2 dates, B has 1 (older)."""
    det, plat, sample = "DET", "PLAT", "single_e"
    stackA, stackB = "key4hep-2026-05-20", "key4hep-2026-05-10"
    runs = {
        stackA: ["2026-05-20", "2026-05-21"],
        stackB: ["2026-05-10"],
    }
    tree: dict[str, object] = {}
    # detector → platform → stacks
    tree[f"{BASE}/{det}/{plat}"] = [f"{stackA}/", f"{stackB}/"]
    for stack, dates in runs.items():
        tree[f"{BASE}/{det}/{plat}/{stack}/{sample}"] = [f"{d}/" for d in dates]
        for d in dates:
            run_url = f"{BASE}/{det}/{plat}/{stack}/{sample}/{d}"
            tree[run_url] = ["baseline_results.csv", "run_info.json"]
            tree[f"{run_url}/baseline_results.csv"] = b"label\nbaseline\n"
            tree[f"{run_url}/run_info.json"] = b'{"date": "%s"}' % d.encode()
    return tree


@pytest.fixture
def web(monkeypatch):
    fake = FakeWeb(_build_tree())
    monkeypatch.setattr(remote.requests, "get", fake.get)
    return fake


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_run_dates_all_stacks_lists_without_downloading_files(web):
    out = remote.list_run_dates_all_stacks(BASE, "DET", "PLAT", "single_e")
    assert out == {
        "key4hep-2026-05-20": ["2026-05-20", "2026-05-21"],
        "key4hep-2026-05-10": ["2026-05-10"],
    }
    # Discovery must only hit directory listings — never a *_results.csv / *.json file.
    assert not any(u.endswith((".csv", ".json")) for u in web.requested)


def test_list_run_dates_skips_stacks_missing_the_sample(web, monkeypatch):
    import requests as real_requests

    orig_get = web.get

    def get(url, timeout=None):
        if url.rstrip("/").endswith("key4hep-2026-05-10/single_e"):
            raise real_requests.RequestException("404")
        return orig_get(url, timeout=timeout)

    monkeypatch.setattr(remote.requests, "get", get)
    out = remote.list_run_dates_all_stacks(BASE, "DET", "PLAT", "single_e")
    assert list(out) == ["key4hep-2026-05-20"]


# ---------------------------------------------------------------------------
# Per-run cache
# ---------------------------------------------------------------------------


def test_ensure_run_cached_downloads_then_is_idempotent(web, tmp_path):
    run_dir = remote.ensure_run_cached(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21",
        cache_root=str(tmp_path),
    )
    run_dir = Path(run_dir)
    assert (run_dir / "baseline_results.csv").read_bytes() == b"label\nbaseline\n"
    assert (run_dir / "run_info.json").exists()
    assert (run_dir / ".complete").exists()
    # Stable, content-addressable-by-coordinates path.
    assert run_dir == tmp_path / "DET" / "PLAT" / "key4hep-2026-05-20" / "single_e" / "2026-05-21"

    n_after_first = len(web.requested)
    again = remote.ensure_run_cached(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21",
        cache_root=str(tmp_path),
    )
    assert Path(again) == run_dir
    # A completed run must be served from cache with zero further HTTP.
    assert len(web.requested) == n_after_first


def test_ensure_run_cached_refetches_when_incomplete(web, tmp_path):
    run_dir = tmp_path / "DET" / "PLAT" / "key4hep-2026-05-20" / "single_e" / "2026-05-21"
    run_dir.mkdir(parents=True)
    # Files present but no .complete sentinel → treated as a partial download.
    (run_dir / "stale.txt").write_text("partial")
    remote.ensure_run_cached(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21",
        cache_root=str(tmp_path),
    )
    assert (run_dir / ".complete").exists()
    assert (run_dir / "baseline_results.csv").exists()


@pytest.mark.parametrize(
    "payload",
    [
        "../evil.csv",          # raw traversal
        "%2e%2e%2fevil.csv",    # percent-encoded "../evil.csv" — decodes to a separator
        "%2e%2e",               # percent-encoded ".."
    ],
)
def test_ensure_run_cached_rejects_unsafe_filename(monkeypatch, tmp_path, payload):
    fake = FakeWeb({
        f"{BASE}/DET/PLAT/S/smp/2026-01-01": [payload],
    })
    monkeypatch.setattr(remote.requests, "get", fake.get)
    with pytest.raises(ValueError, match="Unsafe filename"):
        remote.ensure_run_cached(
            BASE, "DET", "PLAT", "S", "smp", "2026-01-01", cache_root=str(tmp_path)
        )


# ---------------------------------------------------------------------------
# Windowed + latest fetch
# ---------------------------------------------------------------------------


def test_fetch_runs_windowed_fetches_only_given_runs(web, tmp_path):
    windowed = {"key4hep-2026-05-20": ["2026-05-21"]}  # one run, exclude the rest
    runs = remote.fetch_runs_windowed(
        BASE, "DET", "PLAT", "single_e", windowed, cache_root=str(tmp_path)
    )
    assert len(runs) == 1
    assert runs[0]["stack"] == "key4hep-2026-05-20"
    assert runs[0]["date"] == "2026-05-21"
    assert Path(runs[0]["run_dir"]).joinpath("baseline_results.csv").exists()
    # The excluded *run dates* were never requested. Match on the run-date path
    # segment ("/single_e/<date>") so we don't accidentally match the stack name
    # "key4hep-2026-05-20", which embeds the same date string.
    assert not any("/single_e/2026-05-20" in u for u in web.requested)
    assert not any("/single_e/2026-05-10" in u for u in web.requested)


def test_fetch_runs_windowed_empty_returns_empty(web, tmp_path):
    assert remote.fetch_runs_windowed(
        BASE, "DET", "PLAT", "single_e", {}, cache_root=str(tmp_path)
    ) == []


def test_ensure_latest_run_cached_picks_newest(web, tmp_path):
    run_dir = remote.ensure_latest_run_cached(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", cache_root=str(tmp_path)
    )
    assert Path(run_dir).name == "2026-05-21"  # newest of the two dates
