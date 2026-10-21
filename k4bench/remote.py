"""Discover and download benchmark run data from a WebEOS HTTP endpoint.

The WebEOS site serves an Apache-style directory listing at K4BENCH_DATA_URL,
laid out as :mod:`k4bench.layout` defines::

    {base_url}/
      runs/{suite}/                                 e.g. runs/sim/
        {concept}/                                  e.g. ALLEGRO/
          {detector}/                               e.g. ALLEGRO_o1_v03/
            {platform}/                             e.g. x86_64-el9-gcc16-opt/
              {stack}/                              e.g. key4hep-2026-05-19/
                {sample}/                           e.g. single_e-_10GeV/
                  {YYYY-MM-DD}/                     e.g. 2026-05-23/
                    run_info.json
                    machine_info.json
                    {config}_results.csv
                    {config}_events.json
                    {config}_regions.json
                    {config}.log
      reports/{suite}/{YYYY-MM-DD}/report.json, blame.json

Every function reading runs takes the suite it reads right after *base_url*.

A run is downloaded into a local cache either whole (:func:`ensure_run_cached`)
or file by file, only the files a reader asks for (:func:`ensure_run_artifacts`).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote

import requests
from urllib3.util import Retry

from k4bench.artifacts import RUN_INFO, matching
from k4bench.layout import (
    concept_of, concept_path, report_path, run_path, run_url, store_url,
)

_log = logging.getLogger(__name__)

_DIR_LINK_RE = re.compile(r'href="([^"/][^"]*/?)"', re.IGNORECASE)

#: ``(connect, read)`` seconds. The read limit sits above the WebEOS gateway's
#: own 30 s: a directory listing sometimes takes 15–25 s to generate, for
#: several requests in a row, while files still arrive in a tenth of a second
#: — and a client that gives up first only retries into the same wait. Past
#: 30 s the gateway answers ``504`` itself, which :data:`_RETRY` retries.
_TIMEOUT = (10, 35)

#: Ceiling on concurrent connections to the WebEOS host. A deliberate
#: stability constant rather than a CPU- or task-count-derived guess (neither
#: reflects how many simultaneous TLS connections are sensible against one
#: remote server) — kept modest since these are small static files with
#: little to gain from more than a handful in flight. Only real with
#: ``pool_block=True`` below; ``requests``' own default (``block=False``)
#: would silently open extra unpooled connections past this instead of
#: enforcing it.
_POOL_MAXSIZE = 8

#: Retries for the WebEOS gateway's transient failures: a ``502``/``503``/
#: ``504`` (the gateway gives up on a slow listing after 30 s, and the same
#: listing is usually fast again soon after), a dropped connection, or a read
#: that stalls past :data:`_TIMEOUT`. One retry, reads only (every call here is
#: a GET), so a request that never answers costs about two gateway timeouts —
#: roughly a minute — before it fails. ``raise_on_status=False`` hands the last
#: error response back, so ``raise_for_status`` still reports it.
_RETRY = Retry(
    total=1,
    backoff_factor=0.5,
    status_forcelist=(502, 503, 504),
    allowed_methods=frozenset({"GET", "HEAD"}),
    raise_on_status=False,
)

#: One adapter, shared by every thread's session (see :func:`_get_session`),
#: so this is a real, process-wide cap rather than one per thread.
_adapter = requests.adapters.HTTPAdapter(
    pool_maxsize=_POOL_MAXSIZE, pool_block=True, max_retries=_RETRY,
)

_thread_local = threading.local()


def _get_session() -> requests.Session:
    """One ``requests.Session`` per thread, all mounting the shared,
    connection-pooled :data:`_adapter`.

    Replaces a fresh ad-hoc session per call (bare ``requests.get()`` opens
    and tears one down every time) — several threads each doing a cold TLS
    handshake at once was a reproducible way to crash the interpreter
    natively. A session per thread, rather than one session shared by all,
    rules out any race on a session's own mutable state (cookie jar,
    redirect/auth hooks) at negligible cost: the pooled connections all live
    on the shared adapter, so this allocates no new sockets, only a
    lightweight wrapper object per thread.
    """
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.mount("https://", _adapter)
        session.mount("http://", _adapter)
        _thread_local.session = session
    return session


class IncompleteFetch(requests.RequestException):
    """A read that could not be completed because the server did not answer.

    Raised only to callers passing ``strict=True``. By default those functions
    skip what they could not read (logged) and return the rest, or ``None``;
    :attr:`partial` is what had been read when the strict call stopped, so a
    caller that can use an incomplete result still may — deliberately. The
    dashboard caches every result, and one missing items only because WebEOS
    stalled must be read again, not remembered as the answer.

    A :class:`requests.RequestException`, so a caller that already falls back
    on a failed request falls back on this too, rather than mistaking a partial
    listing for a complete one.
    """

    def __init__(self, partial: object, failures: list[str]):
        shown = "; ".join(failures[:3])
        more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
        super().__init__(f"{len(failures)} read(s) failed: {shown}{more}")
        self.partial = partial
        self.failures = failures


def _is_absent(exc: requests.RequestException) -> bool:
    """True for a 404: the path does not exist, which is an answer. A timeout,
    a dropped connection or a 5xx says nothing about what is there."""
    response = getattr(exc, "response", None)
    return response is not None and response.status_code == 404


def _list_each(
    urls: dict[str, str], *, strict: bool, caller: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """List every directory in *urls* (``{key: url}``), one after another.

    Returns the listings that answered, keyed like *urls*, and a description
    of each that failed without an answer. A 404 is left out quietly. With
    *strict*, the walk stops at the first failure: the result is incomplete
    either way, and on a degraded server every further listing could cost
    another timeout. Sequential because a cold listing is generated no faster
    when several are asked for at once.
    """
    found: dict[str, list[str]] = {}
    failures: list[str] = []
    for key, url in urls.items():
        try:
            found[key] = _list_subdirs(url)
        except requests.RequestException as exc:
            if _is_absent(exc):
                _log.debug("%s: skipping %s — %s", caller, url, exc)
                continue
            _log.warning("%s: skipping %s — %s", caller, url, exc)
            failures.append(f"{key}: {exc}")
            if strict:
                break
    return found, failures


def _list_subdirs(url: str) -> list[str]:
    """Return directory names (without trailing slash) from an Apache listing."""
    url = url.rstrip("/") + "/"
    resp = _get_session().get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    return [
        m.group(1).rstrip("/")
        for m in _DIR_LINK_RE.finditer(resp.text)
        if m.group(1).endswith("/") and not m.group(1).startswith("?")
    ]


def _list_files(url: str) -> list[str]:
    """Return file names (no slash) from an Apache directory listing."""
    url = url.rstrip("/") + "/"
    resp = _get_session().get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    return [
        m.group(1)
        for m in _DIR_LINK_RE.finditer(resp.text)
        if not m.group(1).endswith("/") and not m.group(1).startswith("?")
    ]


# ── Discovery helpers (one per hierarchy level) ───────────────────────────────

def list_detectors(base_url: str, suite: str) -> list[str]:
    """Return the detector names benchmarked in *suite*, sorted.

    One listing for the suite's concepts, then one per concept. A suite with no
    run tree yet (a 404) has no detectors. A concept listing the server does
    not answer raises :class:`IncompleteFetch` carrying the detectors listed so
    far: a detector missing from the answer would read as one no longer run.
    """
    try:
        concepts = _list_subdirs(run_url(base_url, suite))
    except requests.RequestException as exc:
        if not _is_absent(exc):
            raise
        _log.debug("list_detectors: no %s run tree — %s", suite, exc)
        return []
    listed, failures = _list_each(
        {concept: store_url(base_url, concept_path(suite, concept)) for concept in concepts},
        strict=True, caller="list_detectors",
    )
    detectors = sorted(
        detector
        for concept, names in listed.items()
        for detector in names
        if concept_of(detector) == concept
    )
    if failures:
        raise IncompleteFetch(detectors, failures)
    return detectors


def list_platforms(base_url: str, suite: str, detector: str) -> list[str]:
    """Return available platforms for *detector*."""
    return _list_subdirs(run_url(base_url, suite, detector))


def list_samples(base_url: str, suite: str, detector: str, platform: str, stack: str) -> list[str]:
    """Return available physics samples for *(detector, platform, stack)*."""
    return sorted(_list_subdirs(run_url(base_url, suite, detector, platform, stack)))


def scan_stack_samples(
    base_url: str, suite: str, detector: str, platform: str, *, strict: bool = False,
) -> dict[str, list[str]]:
    """Return ``{stack: [samples]}`` for *(detector, platform)*, newest stack first.

    Single source of truth for both the cross-release sample union and the
    per-sample stack list, so the sidebar scans the release tree only once
    (one listing per stack) instead of twice. A sample may be added or dropped
    between Key4hep releases; callers derive the union and the per-sample stacks
    from this map. Stacks whose listing fails are skipped (logged); with
    *strict*, the first that fails without an answer stops the scan and raises
    :class:`IncompleteFetch` carrying what was listed.
    """
    stacks = sorted(_list_subdirs(run_url(base_url, suite, detector, platform)), reverse=True)
    listed, failures = _list_each(
        {stack: run_url(base_url, suite, detector, platform, stack) for stack in stacks},
        strict=strict, caller="scan_stack_samples",
    )
    out = {stack: sorted(samples) for stack, samples in listed.items()}
    if strict and failures:
        raise IncompleteFetch(out, failures)
    return out


def list_runs(
    base_url: str,
    suite: str,
    detector: str,
    platform: str,
    stack: str,
    sample: str,
) -> list[str]:
    """Return available run dates for a *(detector, platform, stack, sample)* combination, newest first."""
    runs = _list_subdirs(run_url(base_url, suite, detector, platform, stack, sample))
    return sorted(runs, reverse=True)


def list_run_dates_all_stacks(
    base_url: str,
    suite: str,
    detector: str,
    platform: str,
    sample: str,
    *,
    strict: bool = False,
) -> dict[str, list[str]]:
    """Return ``{stack: [run_dates]}`` for *(detector, platform, sample)*.

    Discovery only — issues directory listings and **no** file downloads. The
    run-date directory names are ``YYYY-MM-DD``, so the full set of available
    dates per stack is obtained cheaply; this populates the trend-window control
    and lets the caller download only the runs inside the selected window.
    Stacks that do not contain *sample* (a 404) or whose listing fails are
    skipped; with *strict*, the first that fails without an answer stops the
    scan and raises :class:`IncompleteFetch` carrying what was listed.
    """
    stacks = _list_subdirs(run_url(base_url, suite, detector, platform))
    listed, failures = _list_each(
        {stack: run_url(base_url, suite, detector, platform, stack, sample) for stack in stacks},
        strict=strict, caller="list_run_dates_all_stacks",
    )
    out = {stack: sorted(runs) for stack, runs in listed.items() if runs}
    if strict and failures:
        raise IncompleteFetch(out, failures)
    return out


# ── Download (persistent immutable cache) ──────────────────────────────────────

def _default_cache_root() -> Path:
    return Path(
        os.environ.get(
            "K4BENCH_CACHE_DIR", str(Path(tempfile.gettempdir()) / "k4bench_cache")
        )
    )


def ensure_run_cached(
    base_url: str,
    suite: str,
    detector: str,
    platform: str,
    stack: str,
    sample: str,
    date: str,
    cache_root: str | None = None,
) -> Path:
    """Download one run into a stable cache path and return it.

    The cache mirrors the store layout (:mod:`k4bench.layout`) below *cache_root*.
    Historical runs are immutable, so a run whose ``.complete`` sentinel exists is
    returned without any HTTP. To stay correct across concurrent reruns and
    processes, files are downloaded into a private temp dir and the finished run
    is published with a single atomic ``rename``: a reader therefore never sees a
    half-written ``run_dir``, and an interrupted download leaves no partial run
    behind (the temp dir is discarded) rather than a dir that looks cached.
    """
    root = Path(cache_root) if cache_root else _default_cache_root()
    run_dir = root / run_path(suite, detector, platform, stack, sample, date)
    sentinel = run_dir / ".complete"
    if sentinel.exists():
        return run_dir

    url = run_url(base_url, suite, detector, platform, stack, sample, date)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage in a sibling temp dir (same filesystem, so the publish rename is atomic).
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{date}.tmp-", dir=run_dir.parent))
    session = _get_session()
    try:
        for fname in _list_files(url):
            decoded = _safe_name(fname)
            resp = session.get(f"{url}/{fname}", timeout=_TIMEOUT)
            resp.raise_for_status()
            (tmp_dir / decoded).write_bytes(resp.content)
        (tmp_dir / ".complete").write_text("")
        _publish_run_dir(tmp_dir, run_dir, sentinel)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return run_dir


def _safe_name(listed: str) -> str:
    """The file name a run's listing spells as *listed*, decoded; raises
    ``ValueError`` unless it is a single, plain path component."""
    # Validate the *decoded* name: a percent-encoded separator (e.g.
    # "%2e%2e%2fevil.csv" → "../evil.csv") slips past a raw-name check but
    # escapes the run dir once written, so decode first, then reject any
    # name that is not a single, plain path component.
    decoded = unquote(listed)
    if (
        not decoded
        or decoded in (".", "..")
        or "/" in decoded
        or "\\" in decoded
        or Path(decoded).name != decoded
    ):
        raise ValueError(f"Unsafe filename in listing: {listed!r}")
    return decoded


def _publish_run_dir(tmp_dir: Path, run_dir: Path, sentinel: Path) -> None:
    """Atomically move a fully-staged *tmp_dir* into its final *run_dir*."""
    try:
        os.replace(tmp_dir, run_dir)
    except OSError:
        # ``run_dir`` already exists and is non-empty. Either another worker
        # published the same immutable run concurrently (sentinel present → trust
        # it, drop our copy) or a stale partial dir from an interrupted attempt is
        # in the way (clear it and retry, since rename can't replace a non-empty dir).
        if sentinel.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return
        shutil.rmtree(run_dir, ignore_errors=True)
        os.replace(tmp_dir, run_dir)


#: Below the cache root, the runs downloaded file by file
#: (:func:`ensure_run_artifacts`), apart from the whole runs
#: :func:`ensure_run_cached` publishes: neither ever replaces or clears a
#: directory of the other, and part of a run never passes for all of it.
SUBSET_TREE = "subsets"

#: A subset directory's record of its run: the run's listing, and what became
#: of each file asked for (see :func:`ensure_run_artifacts`).
ARTIFACT_MANIFEST = ".artifacts.json"
_MANIFEST_VERSION = 1

#: What became of a file asked for, in the manifest. A file listed but never
#: asked for has no entry: it was not requested.
FETCHED, ABSENT, FAILED = "fetched", "absent", "failed"


def ensure_run_artifacts(
    base_url: str,
    suite: str,
    detector: str,
    platform: str,
    stack: str,
    sample: str,
    date: str,
    artifacts: Iterable[str],
    cache_root: str | None = None,
) -> Path:
    """Download the files of one run that *artifacts* match, and return the
    directory holding them.

    For a reader that needs only part of a run. *artifacts* are file names or
    patterns over them (:func:`k4bench.artifacts.matching`), matched against
    the run's listing, so a label's file is fetched only where the run has it.
    A run :func:`ensure_run_cached` already holds whole is returned as it is.
    Otherwise the files land in the run's directory below :data:`SUBSET_TREE`,
    whose manifest (:data:`ARTIFACT_MANIFEST`) keeps the run's listing and,
    for each file asked for, whether it was fetched, is absent, or failed.
    A file fetched or known absent costs no request, so asking again for what
    is there is free, and asking for more fetches only the rest.

    Each file is published with one atomic rename, so a reader never sees
    half of one, and the manifest is written under a lock on the directory,
    so concurrent callers asking for different files of one run each end with
    theirs. Nothing here writes ``.complete``: a run's directory here never
    holds all of it by contract, whatever it happens to hold.

    A listing that does not answer raises and records nothing. A file that
    does not answer raises too, recorded as failed after those fetched before
    it; the next call asks for it again. A 404 is an answer: the file is
    absent. An unsafe name in the listing raises ``ValueError`` and records
    nothing, as in :func:`ensure_run_cached`.
    """
    root = Path(cache_root) if cache_root else _default_cache_root()
    rel = run_path(suite, detector, platform, stack, sample, date)
    whole = root / rel
    if (whole / ".complete").exists():
        return whole
    run_dir = root / SUBSET_TREE / rel
    artifacts = tuple(artifacts)
    if _pending(run_dir, _read_manifest(run_dir), artifacts) == []:
        return run_dir

    url = run_url(base_url, suite, detector, platform, stack, sample, date)
    run_dir.mkdir(parents=True, exist_ok=True)
    with _locked(run_dir):
        # Re-read under the lock: another caller may have fetched them meanwhile.
        manifest = _read_manifest(run_dir)
        if manifest is None:
            listing = {_safe_name(name): name for name in _list_files(url)}
            manifest = {"version": _MANIFEST_VERSION, "listing": listing, "files": {}}
        files = manifest["files"]
        for name in artifacts:
            if name not in manifest["listing"] and not _is_pattern(name):
                files[name] = {"state": ABSENT}
        session = _get_session()
        try:
            for name in _pending(run_dir, manifest, artifacts):
                try:
                    resp = session.get(f"{url}/{manifest['listing'][name]}", timeout=_TIMEOUT)
                    resp.raise_for_status()
                except requests.RequestException as exc:
                    if not _is_absent(exc):
                        files[name] = {"state": FAILED, "error": str(exc)}
                        raise
                    files[name] = {"state": ABSENT}
                    continue
                _publish_file(run_dir, name, resp.content)
                files[name] = {"state": FETCHED, "bytes": len(resp.content)}
        finally:
            _publish_file(run_dir, ARTIFACT_MANIFEST, json.dumps(manifest).encode())
    return run_dir


def _is_pattern(name: str) -> bool:
    """Whether *name* is a pattern rather than one file's name."""
    return any(char in name for char in "*?[")


def _pending(run_dir: Path, manifest: dict | None, artifacts: tuple[str, ...]) -> list[str] | None:
    """The listed files *artifacts* match that *run_dir* still lacks and that
    are not known absent; ``None`` while the listing is unknown."""
    if manifest is None:
        return None
    files = manifest["files"]
    pending = []
    for name in matching(manifest["listing"], artifacts):
        state = files.get(name, {}).get("state")
        if state == ABSENT or state == FETCHED and (run_dir / name).is_file():
            continue
        pending.append(name)
    return pending


def _read_manifest(run_dir: Path) -> dict | None:
    """*run_dir*'s manifest, or ``None`` when it has none this version reads."""
    try:
        manifest = json.loads((run_dir / ARTIFACT_MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get("version") != _MANIFEST_VERSION:
        return None
    return manifest


def _publish_file(run_dir: Path, name: str, content: bytes) -> None:
    """Write *content* as *run_dir*'s file *name* with one atomic rename: a
    reader finds all of it or none."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=run_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(tmp, run_dir / name)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def _locked(run_dir: Path) -> Iterator[None]:
    """Hold *run_dir*'s lock, which every thread and process downloading into
    it takes in turn."""
    with open(run_dir / ".lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def fetch_runs_windowed(
    base_url: str,
    suite: str,
    detector: str,
    platform: str,
    sample: str,
    stacks_dates: dict[str, list[str]],
    cache_root: str | None = None,
    *,
    strict: bool = False,
    artifacts: tuple[str, ...] | None = None,
) -> list[dict]:
    """Fetch every ``(stack, date)`` in *stacks_dates* in parallel, returning a
    list of ``{"stack", "date", "run_dir"}`` for the runs successfully cached.

    Each run is fetched at most once (see :func:`ensure_run_cached`); callers pass
    an already date-windowed *stacks_dates* so only in-window runs are downloaded.
    Given *artifacts*, only the files of each run they match are
    (:func:`ensure_run_artifacts`), and a run counts as cached once those are.
    Runs that fail to download are logged and skipped rather than aborting the load.

    With *strict*, no download starts after the first failure that makes the
    result incomplete, and the call raises once those in flight have finished —
    so every run that landed stays in the on-disk cache and a retry fetches
    only what is missing. A run the server did not answer for raises
    :class:`IncompleteFetch` carrying the runs that landed; a local error — the
    cache directory full or unwritable — raises that ``OSError`` itself, since
    it is neither EOS's fault nor fixed by asking EOS again. A run whose listing
    names an unsafe file is skipped either way: it will be just as unsafe next
    time.

    At most :data:`_POOL_MAXSIZE` downloads run at once — the connection
    pool's own ceiling (see :func:`_get_session`). More workers would add no
    throughput, only threads blocked waiting for a connection, and a blocked
    worker's download already counts as started: it could no longer be
    cancelled, and would run into the same failing server once a connection
    freed up. Work beyond the ceiling therefore waits in the executor's queue,
    where cancelling it stops it.
    """
    tasks = [(stack, date) for stack, dates in stacks_dates.items() for date in dates]
    if not tasks:
        return []

    results: list[dict] = []
    failures: list[str] = []
    local_error: OSError | None = None
    def fetch(stack: str, date: str) -> Path:
        if artifacts is None:
            return ensure_run_cached(
                base_url, suite, detector, platform, stack, sample, date, cache_root,
            )
        return ensure_run_artifacts(
            base_url, suite, detector, platform, stack, sample, date, artifacts, cache_root,
        )

    with ThreadPoolExecutor(max_workers=_POOL_MAXSIZE) as pool:
        futures = {pool.submit(fetch, stack, date): (stack, date) for stack, date in tasks}
        for fut in as_completed(futures):
            if fut.cancelled():
                continue
            stack, date = futures[fut]
            try:
                run_dir = fut.result()
            # A RequestException is an OSError too, so it must be caught first.
            except requests.RequestException as exc:
                _log.warning("fetch_runs_windowed: failed %s/%s — %s", stack, date, exc)
                if _is_absent(exc):
                    continue
                failures.append(f"{stack}/{date}: {exc}")
            except ValueError as exc:
                _log.warning("fetch_runs_windowed: failed %s/%s — %s", stack, date, exc)
                continue
            except OSError as exc:
                _log.warning("fetch_runs_windowed: failed %s/%s — %s", stack, date, exc)
                local_error = local_error or exc
            else:
                results.append({"stack": stack, "date": date, "run_dir": str(run_dir)})
                continue
            if strict:
                for other in futures:
                    other.cancel()
    if strict and local_error is not None:
        raise local_error
    if strict and failures:
        raise IncompleteFetch(results, failures)
    return results


def list_stacks(base_url: str, suite: str, detector: str, platform: str) -> list[str]:
    """Return the Key4hep releases benchmarked for *(detector, platform)*, newest first.

    Directory names as stored (``key4hep-{YYYY-MM-DD}``). Discovery only — one
    listing, no downloads.
    """
    return sorted(_list_subdirs(run_url(base_url, suite, detector, platform)), reverse=True)


def fetch_stack_packages(
    base_url: str, suite: str, detector: str, platform: str, stack: str, *, strict: bool = False,
) -> dict | None:
    """Return the ``k4h_packages`` map of a release, or ``None``.

    Every detector benchmarked against a given release sourced the *same*
    stack, so any one run under it answers the question — this walks to the
    first run it finds and reads only that ``run_info.json`` (two listings and
    one small GET), rather than downloading a run directory.

    ``None`` covers both "no run found" and "that run predates provenance
    capture": in either case the release's packages are unknown, which a caller
    must not confuse with an empty stack. With *strict*, the first read the
    server does not answer stops the walk and raises :class:`IncompleteFetch`
    instead: the ``None`` it would otherwise end in might not be the answer.
    """
    root = run_url(base_url, suite, detector, platform, stack)

    def _stop_if_strict(where: str, exc: requests.RequestException) -> None:
        if strict and not _is_absent(exc):
            raise IncompleteFetch(None, [f"{where}: {exc}"]) from exc

    try:
        samples = sorted(_list_subdirs(root))
    except requests.RequestException as exc:
        _log.debug("fetch_stack_packages: no samples under %s — %s", root, exc)
        _stop_if_strict(root, exc)
        return None

    for sample in samples:
        sample_url = run_url(base_url, suite, detector, platform, stack, sample)
        try:
            dates = sorted(_list_subdirs(sample_url), reverse=True)
        except requests.RequestException as exc:
            _stop_if_strict(sample_url, exc)
            continue
        for date in dates:
            url = f"{run_url(base_url, suite, detector, platform, stack, sample, date)}/{RUN_INFO}"
            try:
                resp = _get_session().get(url, timeout=_TIMEOUT)
                resp.raise_for_status()
                packages = resp.json().get("k4h_packages")
            except ValueError as exc:
                # Malformed JSON (requests' decode error is also a
                # RequestException, so this clause must come first).
                _log.debug("fetch_stack_packages: %s — %s", url, exc)
                continue
            except requests.RequestException as exc:
                _log.debug("fetch_stack_packages: %s — %s", url, exc)
                _stop_if_strict(url, exc)
                continue
            if packages:
                return packages
    return None


def fetch_stack_snapshot(
    base_url: str, suite: str, detector: str, platform: str, stack: str, sample: str | None,
    *, strict: bool = False,
):
    """Latest available run's stack for one scope, skipping partial uploads.

    Only captured runs need the two compressed manifests and machine snapshot.
    A complete newest run costs one listing and four GETs.
    Other detectors cannot stand in for this scope's measured loaded set.
    """
    import gzip
    import zlib
    from dataclasses import replace
    from k4bench.provenance.stack import STACK_MANIFESTS, parse_manifest, stack_snapshot

    def manifest(content: bytes) -> str:
        # A corrupt or empty archive is a partial upload, not an empty stack.
        try:
            source = gzip.decompress(content).decode()
        except (gzip.BadGzipFile, zlib.error) as exc:
            raise ValueError(f"unreadable manifest: {exc}") from exc
        if not parse_manifest(source):
            raise ValueError("manifest lists no packages")
        return source

    root = run_url(base_url, suite, detector, platform, stack, sample)
    try:
        if sample is None:
            for candidate in sorted(_list_subdirs(root)):
                snapshot = fetch_stack_snapshot(
                    base_url, suite, detector, platform, stack, candidate, strict=strict,
                )
                if snapshot is not None:
                    # Another scope can supply the platform's package versions,
                    # but cannot establish what this benchmark loaded or ran on.
                    return replace(snapshot, loaded=None, datasets=None, os=None)
            return None
        dates = sorted(_list_subdirs(root), reverse=True)
        for date in dates:
            url = f"{root}/{date}"

            def get(name):
                response = _get_session().get(f"{url}/{name}", timeout=_TIMEOUT)
                response.raise_for_status()
                return response

            try:
                info = get(RUN_INFO).json()
                if not info.get("k4h_packages") and not info.get("k4h_stack_capture"):
                    continue
                if not info.get("k4h_stack_capture"):
                    return stack_snapshot(info, {})
                manifests = [manifest(get(name).content) for name in STACK_MANIFESTS]
                return stack_snapshot(info, get("machine_info.json").json(), manifests)
            except (requests.RequestException, ValueError, EOFError) as exc:
                # A truncated file is a partial upload too; requests' JSON
                # decode error is a ValueError as well as a RequestException.
                if not isinstance(exc, (ValueError, EOFError)) and not _is_absent(exc):
                    raise
                _log.debug("fetch_stack_snapshot: incomplete run %s — %s", url, exc)
    except (requests.RequestException, ValueError, OSError, EOFError) as exc:
        if strict and not (isinstance(exc, requests.RequestException) and _is_absent(exc)):
            raise IncompleteFetch(None, [f"{root}: {exc}"]) from exc
        _log.debug("fetch_stack_snapshot: %s — %s", root, exc)
    return None


def fetch_run_provenance(
    base_url: str, suite: str, detector: str, platform: str, stack: str, sample: str, run_id: str,
):
    """Only the small, run-specific loaded facts needed by blame (two GETs)."""
    from k4bench.provenance.stack import stack_snapshot

    info = fetch_run_info(base_url, suite, detector, platform, stack, sample, run_id)
    if not info or not info.get("k4h_stack_capture"):
        return None
    url = f"{run_url(base_url, suite, detector, platform, stack, sample, run_id)}/machine_info.json"
    try:
        response = _get_session().get(url, timeout=_TIMEOUT)
        response.raise_for_status()
        machine = response.json()
    except (requests.RequestException, ValueError):
        machine = {}
    return stack_snapshot(info, machine)


def fetch_run_info(
    base_url: str, suite: str, detector: str, platform: str, stack: str, sample: str,
    run_id: str,
) -> dict | None:
    """Return one exact run's parsed ``run_info.json``, or ``None``.

    There is deliberately no directory walk or same-release shortcut: manual
    dispatches and partial re-runs can leave neighbouring run directories with
    different workloads and harness commits.
    """
    url = f"{run_url(base_url, suite, detector, platform, stack, sample, run_id)}/{RUN_INFO}"
    try:
        resp = _get_session().get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        info = resp.json()
    except (requests.RequestException, ValueError) as exc:
        _log.debug("fetch_run_info: %s — %s", url, exc)
        return None
    return info if isinstance(info, dict) else None


def fetch_run_commit(
    base_url: str, suite: str, detector: str, platform: str, stack: str, sample: str,
    run_id: str,
) -> tuple[str, str | None] | None:
    """Return k4Bench's own ``(commit_sha, github_run_url)`` at one exact run,
    or ``None``.

    The run-keyed sibling of :func:`fetch_stack_packages`, for the one
    provenance field that varies per run rather than per release: the commit of
    the harness that produced the run. Unlike the release-keyed read there is
    no walk and no "any run answers" shortcut — a same-day manual dispatch or
    partial re-run can leave sibling groups carrying different commits, so only
    the named run's own ``run_info.json`` is read (one small GET).

    ``None`` covers "no such run" and "that run predates the field" — the
    harness's position is unknown either way.
    """
    info = fetch_run_info(base_url, suite, detector, platform, stack, sample, run_id)
    if info is None:
        return None
    sha = info.get("commit_sha")
    if sha:
        return str(sha), info.get("github_run_url") or None
    return None


def list_report_dates(base_url: str, suite: str) -> list[str]:
    """Return *suite*'s nightly regression-report dates (newest first).

    Reports live at ``{base_url}/reports/{suite}/{YYYY-MM-DD}/report.json``,
    written by the nightly ``regression-report`` CI job. An absent report tree
    (a 404: no report generated yet) is not an error — it returns an empty
    list. Any other failure raises: a listing the server did not answer says
    nothing about whether reports exist, and must not read as "none yet".
    """
    try:
        return sorted(_list_subdirs(store_url(base_url, report_path(suite))), reverse=True)
    except requests.RequestException as exc:
        if not _is_absent(exc):
            raise
        _log.debug("list_report_dates: no %s report tree — %s", suite, exc)
        return []


def _fetch_json(url: str, *, strict: bool, missing_is_normal: bool) -> dict | None:
    """GET and parse *url*, or ``None`` when it is absent, malformed or — unless
    *strict* — unreachable. *missing_is_normal* keeps an expected 404 at debug
    level; every other failure is logged as a warning."""
    try:
        resp = _get_session().get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except ValueError as exc:
        # Malformed JSON (requests' decode error is also a RequestException,
        # so this clause must come first): the file will not parse next time
        # either.
        _log.warning("could not parse %s — %s", url, exc)
        return None
    except requests.RequestException as exc:
        if _is_absent(exc):
            log = _log.debug if missing_is_normal else _log.warning
            log("not found: %s — %s", url, exc)
            return None
        _log.warning("could not fetch %s — %s", url, exc)
        if strict:
            raise IncompleteFetch(None, [f"{url}: {exc}"]) from exc
        return None


def fetch_report(
    base_url: str, suite: str, date: str, *, strict: bool = False,
) -> dict | None:
    """Fetch and parse one of *suite*'s nightly regression reports, or ``None``
    on failure.

    With *strict*, a transient failure raises :class:`IncompleteFetch` instead
    of returning ``None``."""
    url = store_url(base_url, f"{report_path(suite, date)}/report.json")
    return _fetch_json(url, strict=strict, missing_is_normal=False)


def fetch_blame(
    base_url: str, suite: str, date: str, *, strict: bool = False,
) -> dict | None:
    """Fetch and parse one night's blame sidecar, or ``None`` when absent.

    Blame lives beside the report at ``reports/{suite}/{date}/blame.json``, written
    best-effort by the nightly ``regression-report`` job only on nights with a
    confirmed regression it could attribute — so **most nights have none**, and a
    404 here is the common case, not an error (logged at debug, returns ``None``).
    Uses the same shared, connection-pooled session as every other WebEOS read
    (see :func:`_get_session`) rather than a bare ``requests.get``, so it shares
    the one real concurrency ceiling instead of opening its own connections.
    With *strict*, a transient failure raises :class:`IncompleteFetch` instead
    of returning ``None``.
    """
    url = store_url(base_url, f"{report_path(suite, date)}/blame.json")
    return _fetch_json(url, strict=strict, missing_is_normal=True)
