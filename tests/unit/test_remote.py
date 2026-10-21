"""Unit tests for the remote WebEOS access layer (:mod:`k4bench.remote`).

``k4bench.remote`` only depends on stdlib + requests; ``dashboard/remote.py``
is a thin re-export shim kept for the dashboard's flat imports.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from k4bench import remote
from k4bench.layout import SIM, run_path


# ---------------------------------------------------------------------------
# Fake WebEOS server
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str = "", content: bytes = b""):
        self.text = text
        self.content = content

    def raise_for_status(self) -> None:  # pragma: no cover - always ok here
        pass

    def json(self):
        """Parse the body, raising ``ValueError`` on malformed JSON.

        ``requests`` raises a ``JSONDecodeError`` that subclasses ``ValueError``
        for the same case, which is what callers here catch.
        """
        return json.loads(self.content or self.text)


class _StubSession:
    """A get-only stand-in for ``requests.Session``.

    Every remote call in :mod:`k4bench.remote` now goes through a session
    obtained via ``remote._get_session()`` (see that function's docstring —
    one session per thread, all mounting the same connection-pooled adapter,
    which is what avoids the concurrent-cold-TLS-handshake race that used to
    segfault the dashboard). Substituting the session-getter itself, via
    :func:`_use_session`, is the one test seam that reaches every call site
    regardless of whether it happens to run on the main thread or one of
    ``fetch_runs_windowed``'s workers.

    *get* is stored as a plain **instance** attribute rather than a method
    defined on this class: a class-level function attribute is a descriptor
    and gets auto-bound with ``self`` as its first argument on instance
    access, which would silently prepend a stray positional argument to every
    test's ``(url, timeout=None)``-shaped fake. An instance attribute has no
    such rebinding, so a plain function or an already-bound method (e.g.
    ``FakeWeb.get``) both work unchanged.
    """

    def __init__(self, get):
        self.get = get


def _use_session(monkeypatch, get) -> None:
    """Make every :mod:`k4bench.remote` call see *get* as its GET implementation."""
    monkeypatch.setattr(remote, "_get_session", lambda: _StubSession(get))


def _http_error(url: str, status: int) -> remote.requests.HTTPError:
    """The error ``raise_for_status`` raises for *status*, response attached —
    the only way :mod:`k4bench.remote` can tell a 404 from a stall."""
    response = remote.requests.Response()
    response.status_code = status
    response.url = url
    return remote.requests.HTTPError(f"{status} for {url}", response=response)


def _not_found(url, timeout=None):
    raise _http_error(url, 404)


def _timed_out(url, timeout=None):
    raise remote.requests.ReadTimeout(f"read timed out: {url}")


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
#: The sim suite's run tree below :data:`BASE`.
RUNS = f"{BASE}/runs/sim"


def _build_tree() -> dict[str, object]:
    """Two stacks; sample present in both. Stack A has 2 dates, B has 1 (older).

    Stack A's runs carry ``k4h_packages`` (stack provenance); stack B's do not,
    standing in for a run from before provenance was recorded.
    """
    det, plat, sample = "DET", "PLAT", "single_e"
    stackA, stackB = "key4hep-2026-05-20", "key4hep-2026-05-10"
    runs = {
        stackA: ["2026-05-20", "2026-05-21"],
        stackB: ["2026-05-10"],
    }
    provenance = '"k4h_packages": {"k4geo": {"commit": "%s"}}' % ("a" * 40)
    tree: dict[str, object] = {}
    # detector → platform → stacks
    tree[f"{RUNS}/{det}/{det}/{plat}"] = [f"{stackA}/", f"{stackB}/"]
    for stack, dates in runs.items():
        # stack → samples → dates
        tree[f"{RUNS}/{det}/{det}/{plat}/{stack}"] = [f"{sample}/"]
        tree[f"{RUNS}/{det}/{det}/{plat}/{stack}/{sample}"] = [f"{d}/" for d in dates]
        for d in dates:
            run_url = f"{RUNS}/{det}/{det}/{plat}/{stack}/{sample}/{d}"
            tree[run_url] = ["baseline_results.csv", "run_info.json"]
            tree[f"{run_url}/baseline_results.csv"] = b"label\nbaseline\n"
            extra = f", {provenance}" if stack == stackA else ""
            tree[f"{run_url}/run_info.json"] = (
                '{"date": "%s"%s}' % (d, extra)
            ).encode()
    return tree


@pytest.fixture
def web(monkeypatch):
    fake = FakeWeb(_build_tree())
    _use_session(monkeypatch, fake.get)
    return fake


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_detectors_walks_every_concept_of_the_suite(monkeypatch):
    fake = FakeWeb({
        RUNS: ["CLD/", "ALLEGRO/"],
        f"{RUNS}/ALLEGRO": ["ALLEGRO_o2_v01/", "ALLEGRO_o1_v04/"],
        # A detector filed under another concept is not the concept's.
        f"{RUNS}/CLD": ["CLD_o2_v09/", "IDEA_o1_v04/"],
    })
    _use_session(monkeypatch, fake.get)
    assert remote.list_detectors(BASE, SIM) == ["ALLEGRO_o1_v04", "ALLEGRO_o2_v01", "CLD_o2_v09"]


def test_list_detectors_of_a_suite_without_runs_is_empty(monkeypatch):
    _use_session(monkeypatch, _not_found)
    assert remote.list_detectors(BASE, SIM) == []


def test_list_detectors_raises_when_a_concept_does_not_answer(monkeypatch):
    # A detector missing only because its concept's listing stalled would read
    # as one no longer benchmarked.
    def get(url, timeout=None):
        if url.rstrip("/") == f"{RUNS}/CLD":
            _timed_out(url)
        return FakeWeb({RUNS: ["ALLEGRO/", "CLD/"], f"{RUNS}/ALLEGRO": ["ALLEGRO_o1_v04/"]}).get(url)

    _use_session(monkeypatch, get)
    with pytest.raises(remote.IncompleteFetch) as caught:
        remote.list_detectors(BASE, SIM)
    assert caught.value.partial == ["ALLEGRO_o1_v04"]


def test_list_report_dates_newest_first_and_empty_when_absent(monkeypatch):
    fake = FakeWeb({f"{BASE}/reports/sim": ["2026-05-20/", "2026-05-22/", "2026-05-21/"]})
    _use_session(monkeypatch, fake.get)
    assert remote.list_report_dates(BASE, SIM) == ["2026-05-22", "2026-05-21", "2026-05-20"]

    _use_session(monkeypatch, _not_found)
    assert remote.list_report_dates(BASE, SIM) == []  # no report tree yet: not an error


@pytest.mark.parametrize("failure", ["timeout", "gateway"])
def test_list_report_dates_raises_when_the_listing_does_not_answer(monkeypatch, failure):
    # A stalled listing says nothing about whether reports exist. Returning []
    # read as "no reports yet" on the dashboard, and was cached as that.
    def get(url, timeout=None):
        if failure == "timeout":
            _timed_out(url)
        raise _http_error(url, 504)

    _use_session(monkeypatch, get)
    with pytest.raises(remote.requests.RequestException):
        remote.list_report_dates(BASE, SIM)


def test_fetch_report_parses_json(monkeypatch):
    class _JsonResponse(_FakeResponse):
        def json(self):
            import json
            return json.loads(self.content)

    def get(url, timeout=None):
        assert url == f"{BASE}/reports/sim/2026-05-22/report.json"
        return _JsonResponse(content=b'{"generated_at": "x", "groups": []}')

    _use_session(monkeypatch, get)
    assert remote.fetch_report(BASE, SIM, "2026-05-22") == {"generated_at": "x", "groups": []}


@pytest.mark.parametrize("fetch", [remote.fetch_report, remote.fetch_blame])
def test_report_fetches_strict_raise_only_when_the_server_did_not_answer(monkeypatch, fetch):
    _use_session(monkeypatch, _not_found)
    assert fetch(BASE, SIM, "2026-05-22", strict=True) is None  # absent: an answer

    _use_session(monkeypatch, lambda url, timeout=None: _FakeResponse(content=b"{not json"))
    assert fetch(BASE, SIM, "2026-05-22", strict=True) is None  # malformed: no retry fixes it

    _use_session(monkeypatch, _timed_out)
    assert fetch(BASE, SIM, "2026-05-22") is None  # lenient: the CI email carries on
    with pytest.raises(remote.IncompleteFetch):
        fetch(BASE, SIM, "2026-05-22", strict=True)


def test_list_stacks_newest_first(web):
    assert remote.list_stacks(BASE, SIM, "DET", "PLAT") == [
        "key4hep-2026-05-20", "key4hep-2026-05-10",
    ]


def test_fetch_stack_packages_reads_one_run_info(web):
    packages = remote.fetch_stack_packages(BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20")

    assert packages == {"k4geo": {"commit": "a" * 40}}
    # Every detector on a release sourced the same stack, so one run answers it:
    # walk to the first run and read only its run_info.json, never a run dir.
    assert not any(r.endswith("_results.csv") for r in web.requested)


def test_fetch_stack_packages_none_when_the_run_predates_capture(web):
    # A run with no k4h_packages is not an empty stack — it is a run from
    # before provenance was recorded, which the caller must not diff.
    assert remote.fetch_stack_packages(BASE, SIM, "DET", "PLAT", "key4hep-2026-05-10") is None


def test_fetch_stack_packages_skips_a_run_it_cannot_parse(web):
    url = f"{RUNS}/DET/DET/PLAT/key4hep-2026-05-20/single_e/2026-05-21/run_info.json"
    web.tree[url] = b"{not json"
    # 2026-05-21 sorts first (newest); a corrupt run_info must not mask the
    # provenance an older run of the same release still carries.
    assert remote.fetch_stack_packages(BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20") == {
        "k4geo": {"commit": "a" * 40}
    }


def test_fetch_stack_packages_none_when_the_release_is_absent(monkeypatch):
    _use_session(monkeypatch, _not_found)
    assert remote.fetch_stack_packages(BASE, SIM, "DET", "PLAT", "key4hep-1999-01-01") is None
    # Absent is an answer, so a strict caller may keep it.
    assert remote.fetch_stack_packages(
        BASE, SIM, "DET", "PLAT", "key4hep-1999-01-01", strict=True
    ) is None


def test_fetch_stack_packages_strict_raises_when_a_stall_hid_the_answer(monkeypatch):
    _use_session(monkeypatch, _timed_out)
    assert remote.fetch_stack_packages(BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20") is None
    with pytest.raises(remote.IncompleteFetch) as caught:
        remote.fetch_stack_packages(BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20", strict=True)
    assert caught.value.partial is None


def test_fetch_stack_packages_strict_stops_at_the_first_stall(web, monkeypatch):
    stalled = f"{RUNS}/DET/DET/PLAT/key4hep-2026-05-20/single_e/2026-05-21/run_info.json"
    older = f"{RUNS}/DET/DET/PLAT/key4hep-2026-05-20/single_e/2026-05-20/run_info.json"

    def get(url, timeout=None):
        if url == stalled:
            _timed_out(url)
        return web.get(url, timeout=timeout)

    _use_session(monkeypatch, get)
    # Lenient walks on past the stall and finds an older run's answer…
    assert remote.fetch_stack_packages(
        BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20"
    ) == {"k4geo": {"commit": "a" * 40}}
    # …strict stops there: on a degraded server every further read could cost
    # another timeout, and the uncached call is simply asked again next time.
    web.requested.clear()
    with pytest.raises(remote.IncompleteFetch):
        remote.fetch_stack_packages(BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20", strict=True)
    assert older not in web.requested


def test_list_run_dates_all_stacks_lists_without_downloading_files(web):
    out = remote.list_run_dates_all_stacks(BASE, SIM, "DET", "PLAT", "single_e")
    assert out == {
        "key4hep-2026-05-20": ["2026-05-20", "2026-05-21"],
        "key4hep-2026-05-10": ["2026-05-10"],
    }
    # Discovery must only hit directory listings — never a *_results.csv / *.json file.
    assert not any(u.endswith((".csv", ".json")) for u in web.requested)


def test_list_run_dates_skips_stacks_missing_the_sample(web, monkeypatch):
    orig_get = web.get

    def get(url, timeout=None):
        if url.rstrip("/").endswith("key4hep-2026-05-10/single_e"):
            raise _http_error(url, 404)
        return orig_get(url, timeout=timeout)

    _use_session(monkeypatch, get)
    out = remote.list_run_dates_all_stacks(BASE, SIM, "DET", "PLAT", "single_e")
    assert list(out) == ["key4hep-2026-05-20"]
    # A missing sample is an answer, not a failure, even for a strict caller.
    assert remote.list_run_dates_all_stacks(
        BASE, SIM, "DET", "PLAT", "single_e", strict=True
    ) == out


def _stall_on(web, suffix):
    """*web*'s GET, except that URLs ending in *suffix* time out."""
    def get(url, timeout=None):
        if url.rstrip("/").endswith(suffix):
            _timed_out(url)
        return web.get(url, timeout=timeout)
    return get


def test_list_run_dates_strict_raises_on_a_stalled_stack(web, monkeypatch):
    _use_session(monkeypatch, _stall_on(web, "key4hep-2026-05-10/single_e"))
    # Lenient (the nightly report builder): skip the stack and carry on.
    lenient = remote.list_run_dates_all_stacks(BASE, SIM, "DET", "PLAT", "single_e")
    assert list(lenient) == ["key4hep-2026-05-20"]
    # Strict (the dashboard's cache): the same result, but flagged incomplete.
    with pytest.raises(remote.IncompleteFetch) as caught:
        remote.list_run_dates_all_stacks(BASE, SIM, "DET", "PLAT", "single_e", strict=True)
    # What was listed before the scan stopped: never more than the lenient
    # result, and never anything it would not have listed.
    assert caught.value.partial.items() <= lenient.items()
    assert len(caught.value.failures) == 1
    # A request failure, so callers that fall back on one fall back on this.
    assert isinstance(caught.value, remote.requests.RequestException)


def test_scan_stack_samples_strict_raises_on_a_stalled_stack(web, monkeypatch):
    _use_session(monkeypatch, _stall_on(web, "key4hep-2026-05-10"))
    lenient = remote.scan_stack_samples(BASE, SIM, "DET", "PLAT")
    assert lenient == {"key4hep-2026-05-20": ["single_e"]}
    with pytest.raises(remote.IncompleteFetch) as caught:
        remote.scan_stack_samples(BASE, SIM, "DET", "PLAT", strict=True)
    assert caught.value.partial.items() <= lenient.items()


def test_strict_listing_stops_at_the_first_stall(monkeypatch):
    stacks = [f"key4hep-2026-05-{day:02d}" for day in range(1, 31)]
    parent = f"{RUNS}/DET/DET/PLAT"
    tree: dict[str, object] = {parent: [f"{s}/" for s in stacks]}
    for stack in stacks:
        tree[f"{parent}/{stack}/single_e"] = ["2026-05-01/"]
    fake = FakeWeb(tree)
    stalled = f"{stacks[0]}/single_e"
    _use_session(monkeypatch, lambda url, timeout=None: (
        _timed_out(url) if url.rstrip("/").endswith(stalled) else fake.get(url)
    ))

    with pytest.raises(remote.IncompleteFetch):
        remote.list_run_dates_all_stacks(BASE, SIM, "DET", "PLAT", "single_e", strict=True)
    # Nothing after the stall: on a degraded server each further listing could
    # cost another minute. (The stalled listing never reaches the fake.)
    assert fake.requested == [f"{parent}/"]

    fake.requested.clear()
    remote.list_run_dates_all_stacks(BASE, SIM, "DET", "PLAT", "single_e")
    assert len(fake.requested) == 1 + (len(stacks) - 1)  # lenient lists the rest


def test_scan_stack_samples_strict_skips_an_absent_stack(web, monkeypatch):
    def get(url, timeout=None):
        if url.rstrip("/").endswith("key4hep-2026-05-10"):
            raise _http_error(url, 404)
        return web.get(url, timeout=timeout)

    _use_session(monkeypatch, get)
    assert remote.scan_stack_samples(BASE, SIM, "DET", "PLAT", strict=True) == {
        "key4hep-2026-05-20": ["single_e"],
    }


# ---------------------------------------------------------------------------
# Per-run cache
# ---------------------------------------------------------------------------


def test_ensure_run_cached_downloads_then_is_idempotent(web, tmp_path):
    run_dir = remote.ensure_run_cached(
        BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21",
        cache_root=str(tmp_path),
    )
    run_dir = Path(run_dir)
    assert (run_dir / "baseline_results.csv").read_bytes() == b"label\nbaseline\n"
    assert (run_dir / "run_info.json").exists()
    assert (run_dir / ".complete").exists()
    # Stable, content-addressable-by-coordinates path: the store layout, so the
    # cache is itself a store root.
    assert run_dir == tmp_path / run_path(
        SIM, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21"
    )

    n_after_first = len(web.requested)
    again = remote.ensure_run_cached(
        BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21",
        cache_root=str(tmp_path),
    )
    assert Path(again) == run_dir
    # A completed run must be served from cache with zero further HTTP.
    assert len(web.requested) == n_after_first


def test_ensure_run_cached_refetches_when_incomplete(web, tmp_path):
    run_dir = tmp_path / run_path(SIM, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21")
    run_dir.mkdir(parents=True)
    # Files present but no .complete sentinel → treated as a partial download.
    (run_dir / "stale.txt").write_text("partial")
    remote.ensure_run_cached(
        BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21",
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
        f"{RUNS}/DET/DET/PLAT/S/smp/2026-01-01": [payload],
    })
    _use_session(monkeypatch, fake.get)
    with pytest.raises(ValueError, match="Unsafe filename"):
        remote.ensure_run_cached(
            BASE, SIM, "DET", "PLAT", "S", "smp", "2026-01-01", cache_root=str(tmp_path)
        )


# ---------------------------------------------------------------------------
# Windowed + latest fetch
# ---------------------------------------------------------------------------


def test_fetch_runs_windowed_fetches_only_given_runs(web, tmp_path):
    windowed = {"key4hep-2026-05-20": ["2026-05-21"]}  # one run, exclude the rest
    runs = remote.fetch_runs_windowed(
        BASE, SIM, "DET", "PLAT", "single_e", windowed, cache_root=str(tmp_path)
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


def test_fetch_runs_windowed_strict_raises_after_the_rest_landed(web, monkeypatch, tmp_path):
    import time

    stalled = "key4hep-2026-05-20/single_e/2026-05-21/baseline_results.csv"

    def get(url, timeout=None):
        if url.endswith(stalled):
            time.sleep(0.2)  # the healthy run lands first
            _timed_out(url)
        return web.get(url, timeout=timeout)

    _use_session(monkeypatch, get)
    windowed = {"key4hep-2026-05-20": ["2026-05-20", "2026-05-21"]}
    with pytest.raises(remote.IncompleteFetch) as caught:
        remote.fetch_runs_windowed(
            BASE, SIM, "DET", "PLAT", "single_e", windowed,
            cache_root=str(tmp_path), strict=True,
        )
    # The run that landed is handed back and already sits in the on-disk
    # cache, so a retry downloads only the one that stalled.
    assert [r["date"] for r in caught.value.partial] == ["2026-05-20"]
    assert Path(caught.value.partial[0]["run_dir"], ".complete").exists()

    web.requested.clear()
    _use_session(monkeypatch, web.get)
    runs = remote.fetch_runs_windowed(
        BASE, SIM, "DET", "PLAT", "single_e", windowed, cache_root=str(tmp_path), strict=True,
    )
    assert sorted(r["date"] for r in runs) == ["2026-05-20", "2026-05-21"]
    assert not any("/single_e/2026-05-20" in u for u in web.requested)


def test_fetch_runs_windowed_strict_starts_nothing_after_the_first_failure(monkeypatch):
    # Only work still queued in the executor can be cancelled. With more
    # workers than pooled connections, the surplus would count as started while
    # blocked on a connection, and run into the failing server once one freed.
    import threading
    import time

    dates = [f"2026-05-{day:02d}" for day in range(1, 31)]
    started: list[str] = []
    lock = threading.Lock()

    def ensure_run_cached(base_url, suite, detector, platform, stack, sample, date, cache_root):
        with lock:
            started.append(date)
        if date == dates[0]:
            _timed_out(f"{stack}/{date}")
        time.sleep(0.2)  # the healthy downloads are still in flight
        return f"/cache/{stack}/{date}"

    monkeypatch.setattr(remote, "ensure_run_cached", ensure_run_cached)
    with pytest.raises(remote.IncompleteFetch) as caught:
        remote.fetch_runs_windowed(
            BASE, SIM, "DET", "PLAT", "single_e", {"S": dates}, strict=True,
        )
    # What was in flight, plus at most the one the failed worker picked up
    # before the rest was cancelled — not all 30.
    assert len(started) <= remote._POOL_MAXSIZE + 1
    assert len(caught.value.partial) == len(started) - 1


def test_fetch_runs_windowed_strict_raises_a_local_error_as_itself(web, monkeypatch, tmp_path):
    # A full or unwritable cache is not EOS's fault, and asking EOS again does
    # not fix it: it must not surface as an incomplete remote read.
    import errno

    real = remote.ensure_run_cached

    def ensure_run_cached(base_url, suite, detector, platform, stack, sample, date, cache_root):
        if date == "2026-05-21":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(base_url, suite, detector, platform, stack, sample, date, cache_root)

    monkeypatch.setattr(remote, "ensure_run_cached", ensure_run_cached)
    windowed = {"key4hep-2026-05-20": ["2026-05-20", "2026-05-21"]}
    with pytest.raises(OSError) as caught:
        remote.fetch_runs_windowed(
            BASE, SIM, "DET", "PLAT", "single_e", windowed, cache_root=str(tmp_path), strict=True,
        )
    assert not isinstance(caught.value, remote.IncompleteFetch)
    assert caught.value.errno == errno.ENOSPC
    # Lenient (the nightly report builder) still skips the run and carries on.
    runs = remote.fetch_runs_windowed(
        BASE, SIM, "DET", "PLAT", "single_e", windowed, cache_root=str(tmp_path),
    )
    assert [r["date"] for r in runs] == ["2026-05-20"]


def test_fetch_runs_windowed_strict_does_not_flag_an_unsafe_listing(web, monkeypatch, tmp_path):
    # An unsafe file name will be just as unsafe next time: skipped, but the
    # result is complete as far as the server can make it.
    run = f"{RUNS}/DET/DET/PLAT/key4hep-2026-05-20/single_e/2026-05-21"
    web.tree[run] = ["%2e%2e%2fevil.csv"]
    runs = remote.fetch_runs_windowed(
        BASE, SIM, "DET", "PLAT", "single_e", {"key4hep-2026-05-20": ["2026-05-21"]},
        cache_root=str(tmp_path), strict=True,
    )
    assert runs == []


def test_fetch_runs_windowed_empty_returns_empty(web, tmp_path):
    assert remote.fetch_runs_windowed(
        BASE, SIM, "DET", "PLAT", "single_e", {}, cache_root=str(tmp_path)
    ) == []


def test_list_runs_lists_one_releases_run_dates_newest_first(web):
    # Discovery only: the sidebar's Run selector lists the dates without
    # downloading a run.
    assert remote.list_runs(BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20", "single_e") == [
        "2026-05-21", "2026-05-20",
    ]


# ---------------------------------------------------------------------------
# Per-thread session, shared connection-pooled adapter
# ---------------------------------------------------------------------------

def test_get_session_reuses_the_same_instance_within_a_thread():
    a = remote._get_session()
    b = remote._get_session()
    assert a is b


def test_get_session_is_a_real_requests_session():
    assert isinstance(remote._get_session(), remote.requests.Session)


def test_get_session_gives_each_thread_its_own_instance():
    # A session carries a cookie jar and redirect/auth hooks that every
    # request touches implicitly — sharing one instance across threads risks
    # a whole class of subtle races beyond the request path itself. Each
    # thread must get its own.
    import threading

    other = {}
    t = threading.Thread(target=lambda: other.setdefault("session", remote._get_session()))
    t.start()
    t.join()
    assert other["session"] is not remote._get_session()


def test_get_session_instances_share_the_one_pooled_adapter():
    # The actual concurrency ceiling lives on the adapter, not the session —
    # every thread's session must mount the identical adapter object for
    # _POOL_MAXSIZE to be a real, process-wide cap rather than per-thread.
    import threading

    other = {}
    t = threading.Thread(target=lambda: other.setdefault("session", remote._get_session()))
    t.start()
    t.join()
    mine = remote._get_session()
    assert mine.get_adapter("https://x") is other["session"].get_adapter("https://x")
    assert mine.get_adapter("https://x") is remote._adapter


@pytest.fixture
def local_server(monkeypatch):
    """A real HTTP server on localhost, answering each GET with the next of the
    responses a test queues: ``"stall"``, an HTTP status, or ``"ok"``. Returns
    ``(base_url, queue, hits)``."""
    import http.server
    import threading
    import time

    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(remote, "_TIMEOUT", 0.3)
    queue: list[object] = []
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            answer = queue.pop(0) if queue else "ok"
            try:
                if answer == "stall":
                    time.sleep(1.0)
                    return
                if answer != "ok":
                    self.send_response(int(answer))
                    self.end_headers()
                    return
                body = _apache_listing(["2026-09-24/", "2026-09-23/"]).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # the client gave up on the stalled request

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.handle_error = lambda *args: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", queue, hits
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("first", ["stall", 504])
def test_shared_adapter_retries_once(local_server, first):
    # Through the real session: a stalled read or a gateway timeout is asked
    # again, and the second answer is used.
    base, queue, hits = local_server
    queue.append(first)
    assert remote.list_report_dates(base, SIM) == ["2026-09-24", "2026-09-23"]
    assert len(hits) == 2


def test_shared_adapter_gives_up_after_one_retry(local_server):
    # Bounded: a request that never answers costs two attempts, not more.
    base, queue, hits = local_server
    queue.extend([504, 504, 504])
    with pytest.raises(remote.requests.HTTPError):
        remote.list_report_dates(base, SIM)
    assert len(hits) == 2


def test_shared_adapter_blocks_rather_than_opening_extra_connections():
    # pool_block=True is what makes _POOL_MAXSIZE a real ceiling: requests'
    # own default (block=False) would silently open unpooled extra
    # connections past pool_maxsize instead of enforcing it.
    assert remote._adapter._pool_block is True
    assert remote._adapter._pool_maxsize == remote._POOL_MAXSIZE


# ---------------------------------------------------------------------------
# Run-keyed harness provenance
# ---------------------------------------------------------------------------


def test_fetch_run_info_reads_and_returns_the_exact_record(monkeypatch):
    det, plat, stack, sample = "DET", "PLAT", "key4hep-2026-05-20", "single_e"
    url = f"{RUNS}/{det}/{det}/{plat}/{stack}/{sample}/run-17/run_info.json"
    fake = FakeWeb({url: b'{"commit_sha":"abc","n_events":100}'})
    _use_session(monkeypatch, fake.get)
    assert remote.fetch_run_info(
        BASE, SIM, det, plat, stack, sample, "run-17"
    ) == {"commit_sha": "abc", "n_events": 100}
    assert fake.requested == [url]


def test_fetch_run_commit_reads_exactly_one_run_info(monkeypatch):
    det, plat, stack, sample = "DET", "PLAT", "key4hep-2026-05-20", "single_e"
    run_url = "https://github.com/key4hep/k4Bench/actions/runs/9"
    url = f"{RUNS}/{det}/{det}/{plat}/{stack}/{sample}/2026-05-21/run_info.json"
    fake = FakeWeb({
        url: (
            '{"commit_sha": "%s", "github_run_url": "%s"}' % ("b" * 40, run_url)
        ).encode(),
    })
    _use_session(monkeypatch, fake.get)
    got = remote.fetch_run_commit(BASE, SIM, det, plat, stack, sample, "2026-05-21")
    assert got == ("b" * 40, run_url)
    # The named run's own file, in one GET — no listing walked, no sibling
    # sample or stack consulted: a same-day re-run can leave siblings at a
    # different commit, so "any run of that night" is not a safe stand-in.
    assert fake.requested == [url]


def test_fetch_run_commit_none_when_the_run_predates_capture(web):
    # The shared tree's runs carry k4h_packages but no commit_sha.
    assert remote.fetch_run_commit(
        BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-20"
    ) is None


def test_fetch_run_commit_none_when_the_run_is_absent(monkeypatch):
    def _404(url, timeout=None):
        raise remote.requests.RequestException("404")

    _use_session(monkeypatch, _404)
    assert remote.fetch_run_commit(
        BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-22"
    ) is None


def test_measured_stack_costs_one_listing_and_four_gets_per_endpoint(monkeypatch):
    from tests.unit.test_stack_capture import FIXTURES
    root = f"{RUNS}/ILD/ILD/platform/key4hep-2026-10-03/Zbb"
    run = f"{root}/2026-10-03"
    info = {
        "k4h_stack_capture": 1,
        "k4h_packages": json.loads((FIXTURES / "HEAD-2026-10-03.json").read_text()),
        "k4h_loaded_packages": json.loads((FIXTURES / "loaded.json").read_text()),
        "g4_datasets": {},
    }
    web = FakeWeb({
        root: ["2026-10-03/"],
        f"{run}/run_info.json": json.dumps(info).encode(),
        f"{run}/machine_info.json": b'{"os": "AlmaLinux 9.8"}',
        f"{run}/LCG_externals.txt.gz": (FIXTURES / "LCG_externals.txt.gz").read_bytes(),
        f"{run}/LCG_contrib.txt.gz": (FIXTURES / "LCG_contrib.txt.gz").read_bytes(),
    })
    _use_session(monkeypatch, web.get)
    snapshot = remote.fetch_stack_snapshot(BASE, SIM, "ILD", "platform", "key4hep-2026-10-03", "Zbb", strict=True)
    assert snapshot.complete and "Geant4" in snapshot.loaded
    assert len(web.requested) == 5
    assert not any("LCG_builds" in path for path in web.requested)
    # Old runs have no manifest requests or machine fetch added to their cost.
    info.pop("k4h_stack_capture")
    info.pop("k4h_loaded_packages")
    web.tree[f"{run}/run_info.json"] = json.dumps(info).encode()
    web.requested.clear()
    legacy = remote.fetch_stack_snapshot(BASE, SIM, "ILD", "platform", "key4hep-2026-10-03", "Zbb", strict=True)
    assert legacy.loaded is None and not legacy.complete
    assert len(web.requested) == 2


def test_measured_stack_timeout_is_not_a_cacheable_absence(monkeypatch):
    _use_session(monkeypatch, _timed_out)
    with pytest.raises(remote.IncompleteFetch):
        remote.fetch_stack_snapshot(BASE, SIM, "ILD", "platform", "release", "Zbb", strict=True)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("missing", [
    "run_info.json", "LCG_externals.txt.gz", "LCG_contrib.txt.gz", "machine_info.json",
])
def test_measured_stack_skips_partial_newest_run(monkeypatch, strict, truncated, missing):
    from tests.unit.test_stack_capture import FIXTURES
    root = f"{RUNS}/ILD/ILD/platform/release/Zbb"
    newest, older = f"{root}/2026-10-04", f"{root}/2026-10-03"
    artifacts = {
        "run_info.json": json.dumps({
            "k4h_stack_capture": 1,
            "k4h_loaded_packages": json.loads((FIXTURES / "loaded.json").read_text()),
            "g4_datasets": {"G4LEDATA": "G4EMLOW8.8"},
        }).encode(),
        "machine_info.json": b'{"os": "AlmaLinux 9.8"}',
        **{name: (FIXTURES / name).read_bytes()
           for name in ("LCG_externals.txt.gz", "LCG_contrib.txt.gz")},
    }
    web = FakeWeb({
        root: ["2026-10-03/", "2026-10-04/"],
        **{f"{run}/{name}": content for run in (newest, older)
           for name, content in artifacts.items()},
    })

    if truncated:
        web.tree[f"{newest}/{missing}"] = artifacts[missing][: len(artifacts[missing]) // 2]

    def get(url, timeout=None):
        if url == f"{newest}/{missing}" and not truncated:
            web.requested.append(url)
            raise _http_error(url, 404)
        return web.get(url, timeout)

    _use_session(monkeypatch, get)
    snapshot = remote.fetch_stack_snapshot(BASE, SIM, "ILD", "platform", "release", "Zbb", strict=strict)
    assert snapshot is not None and snapshot.complete
    assert "Geant4" in snapshot.loaded
    assert snapshot.datasets == {"G4LEDATA": "G4EMLOW8.8"}
    assert snapshot.os == "AlmaLinux 9.8"
    assert web.requested[1] == f"{newest}/run_info.json"
    assert web.requested.count(f"{root}/") == 1
    assert all(f"{older}/{name}" in web.requested for name in artifacts)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("name", ["LCG_externals.txt.gz", "LCG_contrib.txt.gz"])
@pytest.mark.parametrize("content", [
    pytest.param(b"", id="empty-bytes"),
    pytest.param(gzip.compress(b""), id="empty-manifest"),
    pytest.param(b"not a gzip archive", id="bad-header"),
    pytest.param(gzip.compress(b"x" * 64)[:10] + b"\xff" * 30, id="corrupt-payload"),
])
def test_measured_stack_skips_a_corrupt_or_empty_newest_manifest(monkeypatch, strict, name, content):
    from tests.unit.test_stack_capture import FIXTURES
    root = f"{RUNS}/ILD/ILD/platform/release/Zbb"
    newest, older = f"{root}/2026-10-04", f"{root}/2026-10-03"
    artifacts = {
        "run_info.json": json.dumps({
            "k4h_stack_capture": 1,
            "k4h_loaded_packages": json.loads((FIXTURES / "loaded.json").read_text()),
        }).encode(),
        "machine_info.json": b'{"os": "AlmaLinux 9.8"}',
        **{manifest: (FIXTURES / manifest).read_bytes()
           for manifest in ("LCG_externals.txt.gz", "LCG_contrib.txt.gz")},
    }
    web = FakeWeb({
        root: ["2026-10-03/", "2026-10-04/"],
        **{f"{run}/{artifact}": data for run in (newest, older)
           for artifact, data in artifacts.items()},
    })
    web.tree[f"{newest}/{name}"] = content
    _use_session(monkeypatch, web.get)
    snapshot = remote.fetch_stack_snapshot(BASE, SIM, "ILD", "platform", "release", "Zbb", strict=strict)
    assert snapshot is not None and snapshot.complete
    assert snapshot.packages["Geant4"]["version"]
    assert f"{older}/{name}" in web.requested


@pytest.mark.parametrize("failure", ["timeout", "server_error"])
def test_measured_stack_does_not_skip_transiently_unreadable_newest_run(monkeypatch, failure):
    root = f"{RUNS}/ILD/ILD/platform/release/Zbb"
    newest = f"{root}/2026-10-04/run_info.json"
    web = FakeWeb({root: ["2026-10-03/", "2026-10-04/"]})

    def get(url, timeout=None):
        if url == newest:
            if failure == "timeout":
                raise remote.requests.ReadTimeout(url)
            raise _http_error(url, 503)
        return web.get(url, timeout)

    _use_session(monkeypatch, get)
    with pytest.raises(remote.IncompleteFetch):
        remote.fetch_stack_snapshot(BASE, SIM, "ILD", "platform", "release", "Zbb", strict=True)


def test_other_scope_supplies_versions_without_claiming_loaded_packages(monkeypatch):
    from tests.unit.test_stack_capture import FIXTURES
    root = f"{RUNS}/ILD/ILD/platform/release"
    run = f"{root}/Zbb/2026-10-03"
    web = FakeWeb({
        root: ["Zbb/"], f"{root}/Zbb": ["2026-10-03/"],
        f"{run}/run_info.json": json.dumps({
            "k4h_stack_capture": 1,
            "k4h_loaded_packages": {"Geant4": {"version": "11.4.3"}},
            "g4_datasets": {"G4LEDATA": "G4EMLOW8.8"},
        }).encode(),
        f"{run}/machine_info.json": b'{"os":"AlmaLinux 9.8"}',
        **{f"{run}/{name}": (FIXTURES / name).read_bytes()
           for name in ("LCG_externals.txt.gz", "LCG_contrib.txt.gz")},
    })
    _use_session(monkeypatch, web.get)
    snapshot = remote.fetch_stack_snapshot(BASE, SIM, "ILD", "platform", "release", None)
    assert snapshot.complete and snapshot.packages["Geant4"]["version"] == "11.4.3"
    assert snapshot.loaded is None and snapshot.datasets is None and snapshot.os is None


# ---------------------------------------------------------------------------
# Downloading part of a run
# ---------------------------------------------------------------------------

#: One run of every kind of file a dashboard view reads, the per-event and log
#: files the bulk of it.
_PART_RUN = f"{RUNS}/DET/DET/PLAT/key4hep-2026-05-20/single_e/2026-05-21"
_PART_FILES = {
    "run_info.json": b'{"date": "2026-05-21"}',
    "machine_info.json": b"{}",
    "baseline_results.csv": b"label\nbaseline\n",
    "variant_results.csv": b"label\nvariant\n",
    "baseline_events.json": b"e" * 4000,
    "variant_events.json": b"e" * 4000,
    "baseline_heap.json": b"h" * 8000,
    "baseline.log": b"l" * 16000,
}
_RUN = (BASE, SIM, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21")
_BASE = ("run_info.json", "machine_info.json", "*_results.csv")


class _CountingWeb(FakeWeb):
    """:class:`FakeWeb` also counting the bytes it served."""

    def __init__(self, tree):
        super().__init__(tree)
        self.served = 0

    def get(self, url, timeout=None):
        response = super().get(url, timeout)
        self.served += len(response.content) + len(response.text)
        return response

    def files(self) -> list[str]:
        """The names of the files requested, in order, the listings left out."""
        return [u.rsplit("/", 1)[-1] for u in self.requested if u.rstrip("/") != _PART_RUN]


@pytest.fixture
def part_web(monkeypatch):
    tree = {_PART_RUN: list(_PART_FILES)}
    tree.update({f"{_PART_RUN}/{name}": content for name, content in _PART_FILES.items()})
    fake = _CountingWeb(tree)
    _use_session(monkeypatch, fake.get)
    return fake


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / remote.ARTIFACT_MANIFEST).read_text())


def test_part_of_a_run_downloads_only_the_files_asked_for(part_web, tmp_path):
    run_dir = remote.ensure_run_artifacts(*_RUN, _BASE, cache_root=str(tmp_path))
    assert run_dir == tmp_path / remote.SUBSET_TREE / run_path(*_RUN[1:])
    assert sorted(part_web.files()) == [
        "baseline_results.csv", "machine_info.json", "run_info.json", "variant_results.csv",
    ]
    assert len(part_web.requested) == 5  # and the one listing
    assert part_web.served < sum(map(len, _PART_FILES.values())) / 10
    assert sorted(p.name for p in run_dir.iterdir() if not p.name.startswith(".")) == sorted(
        part_web.files()
    )
    # Part of a run never passes for all of it.
    assert not (run_dir / ".complete").exists()
    assert not (tmp_path / run_path(*_RUN[1:])).exists()
    files = _manifest(run_dir)["files"]
    assert {name: entry["state"] for name, entry in files.items()} == {
        "run_info.json": "fetched", "machine_info.json": "fetched",
        "baseline_results.csv": "fetched", "variant_results.csv": "fetched",
    }
    # The rest is listed, not requested.
    assert set(_manifest(run_dir)["listing"]) == set(_PART_FILES)


def test_asking_again_downloads_nothing_and_asking_for_more_only_the_rest(part_web, tmp_path):
    remote.ensure_run_artifacts(*_RUN, _BASE, cache_root=str(tmp_path))
    first = len(part_web.requested)
    remote.ensure_run_artifacts(*_RUN, _BASE, cache_root=str(tmp_path))
    remote.ensure_run_artifacts(*_RUN, ("run_info.json",), cache_root=str(tmp_path))
    assert len(part_web.requested) == first
    run_dir = remote.ensure_run_artifacts(
        *_RUN, ("run_info.json", "*_events.json"), cache_root=str(tmp_path),
    )
    # The listing is not read again, nor the run info.
    assert part_web.files()[first - 1:] == ["baseline_events.json", "variant_events.json"]
    assert len(part_web.requested) == first + 2
    assert (run_dir / "baseline_results.csv").exists()
    assert (run_dir / "baseline_events.json").read_bytes() == _PART_FILES["baseline_events.json"]


def test_a_whole_run_already_cached_is_reused(part_web, tmp_path):
    whole = remote.ensure_run_cached(*_RUN, cache_root=str(tmp_path))
    part_web.requested.clear()
    assert remote.ensure_run_artifacts(*_RUN, _BASE, cache_root=str(tmp_path)) == whole
    assert part_web.requested == []


def test_a_whole_run_downloaded_later_leaves_the_part_alone(part_web, tmp_path):
    part = remote.ensure_run_artifacts(*_RUN, _BASE, cache_root=str(tmp_path))
    whole = remote.ensure_run_cached(*_RUN, cache_root=str(tmp_path))
    assert whole != part and (whole / ".complete").exists()
    assert (part / "baseline_results.csv").exists()
    assert not (part / ".complete").exists()


def test_a_file_that_failed_is_recorded_and_fetched_again(part_web, monkeypatch, tmp_path):
    stalled = {"variant_events.json"}

    def get(url, timeout=None):
        if url.rsplit("/", 1)[-1] in stalled:
            part_web.requested.append(url)
            raise remote.requests.ReadTimeout(f"read timed out: {url}")
        return part_web.get(url, timeout)

    _use_session(monkeypatch, get)
    with pytest.raises(remote.requests.ReadTimeout):
        remote.ensure_run_artifacts(*_RUN, ("*_events.json",), cache_root=str(tmp_path))
    run_dir = tmp_path / remote.SUBSET_TREE / run_path(*_RUN[1:])
    states = {n: e["state"] for n, e in _manifest(run_dir)["files"].items()}
    assert states == {"baseline_events.json": "fetched", "variant_events.json": "failed"}
    assert "read timed out" in _manifest(run_dir)["files"]["variant_events.json"]["error"]
    assert not (run_dir / "variant_events.json").exists()
    # Another view's files of the same run are not held up by it.
    remote.ensure_run_artifacts(*_RUN, ("baseline_heap.json",), cache_root=str(tmp_path))
    assert (run_dir / "baseline_heap.json").exists()

    stalled.clear()
    part_web.requested.clear()
    remote.ensure_run_artifacts(*_RUN, ("*_events.json",), cache_root=str(tmp_path))
    assert part_web.files() == ["variant_events.json"]
    assert _manifest(run_dir)["files"]["variant_events.json"]["state"] == "fetched"


def test_a_listing_that_failed_records_nothing(part_web, monkeypatch, tmp_path):
    _use_session(monkeypatch, _timed_out)
    with pytest.raises(remote.requests.ReadTimeout):
        remote.ensure_run_artifacts(*_RUN, _BASE, cache_root=str(tmp_path))
    run_dir = tmp_path / remote.SUBSET_TREE / run_path(*_RUN[1:])
    assert not (run_dir / remote.ARTIFACT_MANIFEST).exists()
    _use_session(monkeypatch, part_web.get)
    remote.ensure_run_artifacts(*_RUN, _BASE, cache_root=str(tmp_path))
    assert (run_dir / "run_info.json").exists()


def test_absent_files_are_recorded_and_not_asked_for_again(part_web, monkeypatch, tmp_path):
    # Listed, then gone: a 404 is an answer.
    del part_web.tree[f"{_PART_RUN}/baseline_heap.json"]

    def get(url, timeout=None):
        if url.rstrip("/") not in part_web.tree:
            part_web.requested.append(url)
            raise _http_error(url, 404)
        return part_web.get(url, timeout)

    _use_session(monkeypatch, get)
    artifacts = ("*_heap.json", "event_source.json.gz")
    run_dir = remote.ensure_run_artifacts(*_RUN, artifacts, cache_root=str(tmp_path))
    states = {n: e["state"] for n, e in _manifest(run_dir)["files"].items()}
    # The one never listed is absent without a request.
    assert states == {"baseline_heap.json": "absent", "event_source.json.gz": "absent"}
    assert part_web.files() == ["baseline_heap.json"]
    part_web.requested.clear()
    remote.ensure_run_artifacts(*_RUN, artifacts, cache_root=str(tmp_path))
    assert part_web.requested == []


@pytest.mark.parametrize("payload", ["../evil.csv", "%2e%2e%2fevil.csv", "%2e%2e"])
def test_part_of_a_run_rejects_an_unsafe_listing(monkeypatch, tmp_path, payload):
    fake = FakeWeb({_PART_RUN: ["run_info.json", payload]})
    _use_session(monkeypatch, fake.get)
    with pytest.raises(ValueError, match="Unsafe filename"):
        remote.ensure_run_artifacts(*_RUN, ("run_info.json",), cache_root=str(tmp_path))
    run_dir = tmp_path / remote.SUBSET_TREE / run_path(*_RUN[1:])
    assert not (run_dir / remote.ARTIFACT_MANIFEST).exists()
    assert fake.requested == [_PART_RUN + "/"]
    assert not list(tmp_path.rglob("evil.csv"))


def test_concurrent_parts_of_one_run_both_land(part_web, monkeypatch, tmp_path):
    import threading
    import time

    def slow(url, timeout=None):
        # Long enough that the two downloads overlap.
        time.sleep(0.01)
        return part_web.get(url, timeout)

    _use_session(monkeypatch, slow)
    asks = [("*_events.json",), ("*_heap.json", "*.log"), _BASE, ("*_events.json",)]
    errors = []

    def ask(artifacts):
        try:
            remote.ensure_run_artifacts(*_RUN, artifacts, cache_root=str(tmp_path))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=ask, args=(a,)) for a in asks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    run_dir = tmp_path / remote.SUBSET_TREE / run_path(*_RUN[1:])
    assert {n: e["state"] for n, e in _manifest(run_dir)["files"].items()} == dict.fromkeys(
        _PART_FILES, "fetched",
    )
    for name, content in _PART_FILES.items():
        assert (run_dir / name).read_bytes() == content
    # Each file once, the listing once, whatever the interleaving.
    assert sorted(part_web.files()) == sorted(_PART_FILES)
    assert len(part_web.requested) == len(_PART_FILES) + 1
    assert not (run_dir / ".complete").exists()
    assert not (tmp_path / run_path(*_RUN[1:])).exists()


def test_a_manifest_of_another_version_is_read_again(part_web, tmp_path):
    run_dir = remote.ensure_run_artifacts(*_RUN, ("run_info.json",), cache_root=str(tmp_path))
    (run_dir / remote.ARTIFACT_MANIFEST).write_text('{"version": 0}')
    part_web.requested.clear()
    remote.ensure_run_artifacts(*_RUN, ("run_info.json",), cache_root=str(tmp_path))
    assert part_web.files() == ["run_info.json"]
    assert _manifest(run_dir)["version"] == 1


def test_windowed_runs_download_only_the_files_asked_for(web, tmp_path):
    windowed = {"key4hep-2026-05-20": ["2026-05-20", "2026-05-21"]}
    runs = remote.fetch_runs_windowed(
        BASE, SIM, "DET", "PLAT", "single_e", windowed, cache_root=str(tmp_path),
        artifacts=("run_info.json",),
    )
    assert sorted(r["date"] for r in runs) == ["2026-05-20", "2026-05-21"]
    assert not any(u.endswith("baseline_results.csv") for u in web.requested)
    for run in runs:
        assert Path(run["run_dir"]).is_relative_to(tmp_path / remote.SUBSET_TREE)
        assert not (Path(run["run_dir"]) / ".complete").exists()
    # Without artifacts, whole runs as ever.
    whole = remote.fetch_runs_windowed(
        BASE, SIM, "DET", "PLAT", "single_e", windowed, cache_root=str(tmp_path),
    )
    assert all((Path(r["run_dir"]) / ".complete").exists() for r in whole)
