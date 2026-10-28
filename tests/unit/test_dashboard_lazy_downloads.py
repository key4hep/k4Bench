"""What the dashboard downloads, view by view: the reco page driven through
``AppTest`` against a fake WebEOS that counts every request.

A run's files are downloaded when a view first reads them, and the trend
window's runs when a history view first asks (``sidebar``, ``data.RunData``,
``section_dispatch.PageContext``). These pin that down by the requests made:
a current-run view asks for none of another view's files and none of the
history; a view's files are downloaded once, whatever the session; and a file
that could not be downloaded is said so, never drawn as a run without it.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from k4bench import remote  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"
BASE = "https://eos.example/data"
DET, PLAT, SAMPLE = "CLD_o2_v09", "x86_64-el9-gcc16-opt", "p8_ee_Zbb_ecm91"
#: Two releases, each run on two nights: the newest run is the selected one.
RUNS = {"2026-10-13": ["2026-10-13", "2026-10-14"], "2026-10-17": ["2026-10-17", "2026-10-18"]}
NEWEST = "2026-10-18"
#: What the selected run comes with: its metadata, host and results.
BASE_FILES = {
    "run_info.json", "machine_info.json", "baseline_results.csv",
    "variant_truth_tracking_results.csv",
}
#: Where the stack installed a package, which names a component's package.
MANIFEST = "LCG_externals.txt.gz"
_INSTALL = "/cvmfs/sft-nightlies.cern.ch/lcg/nightlies/k4fwcore/HEAD/x86_64-el9-gcc16-opt"


def _store() -> dict[str, object]:
    """The fake store: every run a copy of the reco fixture plus a heap file."""
    files = {p.name: p.read_bytes() for p in RECO_RUN.iterdir() if p.name != "README.md"}
    files["baseline_heap.json"] = b"{}"
    files[MANIFEST] = gzip.compress(f"k4fwcore; 98b4e; HEAD; {_INSTALL}; Gaudi-242b8\n".encode())
    platform = f"{BASE}/runs/reco/CLD/{DET}/{PLAT}"
    tree: dict[str, object] = {
        f"{BASE}/runs/reco": ["CLD/"],
        f"{BASE}/runs/reco/CLD": [f"{DET}/"],
        f"{BASE}/runs/reco/CLD/{DET}": [f"{PLAT}/"],
        platform: [f"key4hep-{rel}/" for rel in RUNS],
        # A report a night, which the trend window is bounded by.
        f"{BASE}/reports/reco": [f"{night}/" for nights in RUNS.values() for night in nights],
    }
    for rel, nights in RUNS.items():
        tree[f"{platform}/key4hep-{rel}"] = [f"{SAMPLE}/"]
        tree[f"{platform}/key4hep-{rel}/{SAMPLE}"] = [f"{night}/" for night in nights]
        for night in nights:
            run = f"{platform}/key4hep-{rel}/{SAMPLE}/{night}"
            info = json.loads(files["run_info.json"])
            info.update(date=night, k4h_release=f"key4hep-{rel}", k4h_release_date=rel)
            run_files = {**files, "run_info.json": json.dumps(info).encode()}
            tree[run] = list(run_files)
            tree.update({f"{run}/{name}": content for name, content in run_files.items()})
    return tree


class _Response:
    def __init__(self, text: str = "", content: bytes = b""):
        self.text, self.content = text, content

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return json.loads(self.content or self.text)


class _Store:
    """The fake WebEOS: serves :func:`_store`, counting every request."""

    def __init__(self):
        self.tree = _store()
        self.requests: Counter[str] = Counter()
        self.stalled: set[str] = set()

    def get(self, url: str, timeout=None):
        key = url.rstrip("/")
        self.requests[key] += 1
        if key.rsplit("/", 1)[-1] in self.stalled:
            raise remote.requests.ReadTimeout(f"read timed out: {url}")
        node = self.tree.get(key)
        if node is None:
            response = remote.requests.Response()
            response.status_code = 404
            raise remote.requests.HTTPError(f"404 for {url}", response=response)
        if isinstance(node, bytes):
            return _Response(content=node)
        return _Response(text="".join(f'<a href="{name}">{name}</a>' for name in node))

    def downloads(self) -> list[tuple[str, str]]:
        """``(night, file)`` of every file request since the last call, then forgets them."""
        got = sorted(
            (url.rsplit("/", 2)[-2], url.rsplit("/", 1)[-1])
            for url, n in self.requests.items() for _ in range(n)
            if not isinstance(self.tree.get(url), list)
        )
        self.requests.clear()
        return got


class _Session:
    def __init__(self, get):
        self.get = get


@pytest.fixture
def store(monkeypatch, tmp_path):
    import remote_cache
    import sidebar

    # Other app tests in this process swap the sidebar's store reads for
    # stubs; these tests read the (fake) store through the real ones.
    for name in vars(sidebar).keys() & vars(remote_cache).keys():
        if name.startswith("cached_") or name in ("read_again", "forget", "list_platform_stacks"):
            monkeypatch.setattr(sidebar, name, getattr(remote_cache, name))
    fake = _Store()
    monkeypatch.setattr(remote, "_get_session", lambda: _Session(fake.get))
    monkeypatch.setenv("K4BENCH_DATA_URL", BASE)
    monkeypatch.setenv("K4BENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("K4BENCH_DATA_DIR", raising=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield fake
    st.cache_data.clear()
    st.cache_resource.clear()


def _page_app(dashboard_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from reco import page

    page.main()


def _open(tab: str) -> AppTest:
    at = AppTest.from_function(_page_app, args=(str(_DASHBOARD_DIR),), default_timeout=120)
    at.query_params["tab"] = tab
    at.run()
    assert not at.exception, at.exception
    return at


def _section(at: AppTest, name: str) -> None:
    at.button_group[0].set_value(name).run()
    assert not at.exception, at.exception


def _of(downloads, night=NEWEST) -> set[str]:
    return {name for n, name in downloads if n == night}


def test_a_current_run_view_downloads_its_own_files_of_the_selected_run(store):
    at = _open("Breakdown")
    got = store.downloads()
    # The selected run alone, nothing of the window's other runs.
    assert {night for night, _ in got} == {NEWEST}
    assert _of(got) == BASE_FILES | {
        "baseline_components.json", "variant_truth_tracking_components.json",
    }
    assert len(got) == len(_of(got))  # each once
    # A fraction of downloading every run of the window whole.
    files_per_run = sum(1 for name in store.tree if name.rsplit("/", 2)[-2] == NEWEST)
    whole = sum(len(nights) for nights in RUNS.values()) * files_per_run
    assert len(got) * 8 < whole
    # The components' table names each one's package, from the stack
    # manifest, beside its heap; and the one inspected its collections.
    _section(at, "Components")
    got = store.downloads()
    assert {night for night, _ in got} == {NEWEST}
    assert MANIFEST in _of(got) and "baseline_heap.json" in _of(got)
    assert not any(name.endswith(("_events.json", ".log")) for name in _of(got))


def test_each_view_downloads_only_what_it_adds(store):
    at = _open("Breakdown")
    store.downloads()
    at.run()
    assert store.downloads() == []
    _section(at, "Logs")
    # The configuration on view, not every log of the run.
    assert store.downloads() == [(NEWEST, "baseline.log"), (NEWEST, "baseline_joboptions.opts")]
    _section(at, "Memory")
    got = store.downloads()
    assert _of(got) == {"baseline_heap.json"} and len(got) == 1
    _section(at, "Time")
    at.button_group(key="reco_time_lens").set_value("Per event").run()
    assert store.downloads() == [
        (NEWEST, "baseline_events.json"), (NEWEST, "variant_truth_tracking_events.json"),
    ]
    # Back where it started: nothing new.
    at.button_group(key="reco_time_lens").set_value("Phases and components").run()
    _section(at, "Logs")
    assert store.downloads() == []


def test_the_history_is_downloaded_only_when_a_history_view_opens(store):
    at = _open("Breakdown")
    store.downloads()
    # Moving the window lists, and downloads nothing.
    at.select_slider[0].set_value(("2026-10-13", "2026-10-17")).run()
    at.select_slider[0].set_value(("2026-10-17", "2026-10-17")).run()
    assert store.downloads() == []
    at.select_slider[0].set_value(("2026-10-13", "2026-10-17")).run()
    at.radio(key="reco_breakdown_view").set_value("Historical Trends").run()
    assert not at.exception, at.exception
    got = store.downloads()
    # Every other run of the window: its results, host and component costs.
    others = {night for nights in RUNS.values() for night in nights} - {NEWEST}
    assert {night for night, _ in got} == others
    for night in others:
        assert _of(got, night) == BASE_FILES | {
            "baseline_components.json", "variant_truth_tracking_components.json",
        }
    assert not any(name.endswith((".log", "_events.json", "_heap.json")) for _, name in got)
    at.run()
    assert store.downloads() == []


def test_a_second_visit_downloads_nothing_again(store):
    at = _open("Breakdown")
    at.radio(key="reco_breakdown_view").set_value("Historical Trends").run()
    _section(at, "Logs")
    store.downloads()
    # Another viewer, after the process's caches were dropped: the files are on disk.
    st.cache_data.clear()
    st.cache_resource.clear()
    at = _open("Breakdown")
    at.radio(key="reco_breakdown_view").set_value("Historical Trends").run()
    _section(at, "Logs")
    assert store.downloads() == []


def test_a_file_that_could_not_be_downloaded_is_said_so_and_read_again(store):
    at = _open("Breakdown")
    store.stalled = {"baseline_heap.json"}
    _section(at, "Memory")
    assert any("Could not download the files this view reads" in w.value for w in at.warning)
    # Not mistaken for a run without a heap breakdown.
    assert not any("No heap breakdown" in i.value for i in at.info)
    # The other views are unaffected.
    _section(at, "Logs")
    assert not any("Could not download" in w.value for w in at.warning)
    store.stalled = set()
    store.downloads()
    _section(at, "Memory")
    assert not any("Could not download" in w.value for w in at.warning)
    assert _of(store.downloads()) == {"baseline_heap.json"}


def test_the_packages_are_read_again_once_the_stack_manifests_arrive(tmp_path):
    from data import cached_load_run_packages

    st.cache_data.clear()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_info.json").write_text(json.dumps({"k4h_packages": {
        "k4fwcore": {"commit": "767ff81", "version": "HEAD"},
    }}))
    assert "install_path" not in cached_load_run_packages(str(run_dir))["k4fwcore"]
    (run_dir / MANIFEST).write_bytes(gzip.compress(
        f"k4fwcore; 98b4e; HEAD; {_INSTALL}; Gaudi-242b8\n".encode()
    ))
    assert cached_load_run_packages(str(run_dir))["k4fwcore"]["install_path"] == _INSTALL
    st.cache_data.clear()


def test_a_load_is_read_again_once_its_files_arrive(tmp_path):
    from data import cached_load_event_timing

    st.cache_data.clear()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(RECO_RUN / "baseline_results.csv", run_dir)
    assert cached_load_event_timing(str(run_dir)) is None
    # Another view's files arriving changes nothing it reads.
    shutil.copy(RECO_RUN / "baseline.log", run_dir)
    assert cached_load_event_timing(str(run_dir)) is None
    shutil.copy(RECO_RUN / "baseline_events.json", run_dir)
    assert set(cached_load_event_timing(str(run_dir))) == {"baseline"}
    st.cache_data.clear()
