"""Read an LCG Key4hep nightly's identity and git provenance off CVMFS.

LCG rotates weekday-named views, so their generated timestamp is the release
identity. Package revisions come from the manifest's install paths and the
``.buildinfo_*`` file installed with each HEAD build. Metadata failures return
an empty result rather than failing the benchmark they describe.
"""

from __future__ import annotations

import gzip
import logging
import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

_log = logging.getLogger(__name__)
_GENERATED_RE = re.compile(r"^#\s*Generated:\s*(.+)$", re.M)
_REVISION_RE = re.compile(r"\bREVISION:\s*'?([0-9a-f]{7,40})(?:\|\d+)?", re.I)
_GITHASH_RE = re.compile(r"\bGITHASH:\s*'([0-9a-f]{7,40})'", re.I)
_EXTERNAL_RE = re.compile(
    r"^\s*(?:LCG_external_package|LCG_AA_project)\(\s*(\S+)\s+HEAD\s+"
    r"GIT=([^\s)]+)",
    re.M,
)
_INCLUDE_RE = re.compile(r"^\s*include\((heptools-[A-Za-z0-9_-]+)\)", re.M)
_URL_RE = re.compile(r"^(?:https?://(?:www\.)?|git@)(?P<host>[^/:]+)[:/](?P<slug>.+?)(?:\.git)?/?$")
_FORGE_INFIX = {"github": "", "gitlab": "/-"}
CONTRIB_ROOT = Path("/cvmfs/sft.cern.ch/lcg/contrib")
STACK_MANIFESTS = ("LCG_externals.txt.gz", "LCG_contrib.txt.gz")


def loaded_paths_file(run_dir: Path, label: str) -> Path:
    return run_dir / f".{label}.loaded"


def capture_stack(run_dir: Path, info: dict, env: dict[str, str]) -> None:
    """Save the manifests and reduce all measured configs' private path lists.

    Missing instrumentation on any measured config makes the union unknown.
    Dataset directories come from the instrumented process's environment;
    the job environment also records them when instrumentation is unavailable.
    """
    info["g4_datasets"] = {
        key: Path(value).name for key, value in sorted(env.items())
        if key.startswith("G4") and key.endswith("DATA") and value
    }
    handoffs = [loaded_paths_file(run_dir, label) for label in info.get("configs", [])]
    try:
        manifest = _manifest(Path(env["K4H_STACK_SETUP"]).resolve())
        if manifest is None:
            return
        sources = [manifest, manifest.with_name(manifest.name.replace("externals", "contrib"))]
        packages = {}
        for source, filename in zip(sources, STACK_MANIFESTS):
            raw = source.read_bytes()
            (run_dir / filename).write_bytes(gzip.compress(raw, mtime=0))
            packages.update(parse_manifest(raw.decode()))
        info["k4h_stack_capture"] = 1
        paths, datasets = [], None
        for handoff in handoffs:
            rows = handoff.read_text().splitlines()
            if not rows or rows[0] != "# k4bench loaded paths v1":
                return
            paths.extend(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), row)
                         for row in rows[1:] if row.startswith("/"))
            config_data = {key: Path(value).name for row in rows[1:]
                           if row.startswith("G4") and "=" in row
                           for key, value in [row.split("=", 1)]}
            if datasets is not None and config_data != datasets:
                # The job environment would now name data some config did not use.
                _log.warning("Geant4 dataset environment differs across configs")
                info.pop("g4_datasets", None)
                return
            datasets = config_data
        if not handoffs or not paths:
            return
        # Gaudi executes these options before enabling the auditor. Staged
        # copies have local paths, so retain the source paths the job recorded.
        if info.get("tool") == "k4run":
            stage = Path(info.get("resolved_stage_dir") or ".")
            paths.extend(str(stage / option) for option in info.get("options", []))
        # ddsim executes its steering file without mapping it, and the file can
        # belong to a Python-only package such as CLDConfig.
        if steering := info.get("resolved_steering_file"):
            paths.append(steering)
        loaded, unknown = mapped_packages(paths, packages)
        unmatched = [path for path in unknown if path.startswith("/cvmfs/")]
        if unmatched:
            _log.warning("Mapped CVMFS paths absent from manifests: %s", unmatched)
        info["k4h_loaded_packages"] = {
            name: {"version": packages[name]["version"], **build_info(packages[name]["install_path"]),
                   **info.get("k4h_packages", {}).get(name, {})}
            for name in sorted(loaded)
        }
        info["g4_datasets"] = datasets
    except Exception as exc:
        _log.warning("Loaded stack provenance not recorded: %s", exc)
    finally:
        for path in handoffs:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                _log.warning("Could not remove private provenance handoff %s", path)


def parse_manifest(source: str, *, contrib_root: Path = CONTRIB_ROOT) -> dict[str, dict]:
    """Read package rows, ignoring the manifest's platform/compiler headers."""
    packages = {}
    for row in source.splitlines():
        fields = [field.strip() for field in row.split(";")]
        if len(fields) < 4 or not fields[0] or not fields[3]:
            continue
        name, build_hash, version, install = fields[:4]
        path = Path(install)
        packages[name] = {
            "version": version, "hash": build_hash,
            "install_path": str(path if path.is_absolute() else contrib_root / path),
        }
    return packages


def build_info(install: str) -> dict[str, str]:
    """Available upstream and toolchain revisions of one installed build."""
    try:
        source = next(Path(install).glob(".buildinfo_*.txt")).read_text()
    except (OSError, StopIteration):
        return {}
    return {
        key: match.group(1)
        for key, pattern in (("commit", _REVISION_RE), ("githash", _GITHASH_RE))
        if (match := pattern.search(source))
    }


def mapped_packages(paths: list[str], packages: dict[str, dict]) -> tuple[set[str], list[str]]:
    """Resolve view symlinks and match the longest manifest install prefix.

    Unmatched paths are returned to the caller, including OS and harness files.
    No CVMFS location is assumed: contrib and stable releases use the same match.
    """
    prefixes = sorted(
        ((str(Path(p["install_path"]).resolve()).rstrip("/") + "/", name)
         for name, p in packages.items()),
        key=lambda item: -len(item[0]),
    )
    loaded, unknown = set(), []
    for path in sorted(set(paths)):
        real = str(Path(path.removesuffix(" (deleted)")).resolve())
        name = next((name for prefix, name in prefixes if real.startswith(prefix)), None)
        if name is None:
            unknown.append(path)
        else:
            loaded.add(name)
    return loaded, unknown


@dataclass(frozen=True)
class StackSnapshot:
    """One recorded stack and the measured scope that supplied it."""

    packages: dict[str, dict]
    loaded: frozenset[str] | None = None
    datasets: dict[str, str] | None = None
    os: str | None = None
    complete: bool = False


def stack_snapshot(info: dict, machine: dict, manifests: list[str] | tuple[str, ...] = ()) -> StackSnapshot:
    """Combine immutable artifacts without resolving their now-rotating paths."""
    packages = {}
    for source in manifests:
        packages.update(parse_manifest(source))
    for name, package in info.get("k4h_packages", {}).items():
        packages.setdefault(name, {}).update(package)
    loaded = info.get("k4h_loaded_packages")
    for name, package in (loaded or {}).items():
        packages.setdefault(name, {}).update(package)
    return StackSnapshot(
        packages, frozenset(loaded) if isinstance(loaded, dict) else None,
        info.get("g4_datasets"), machine.get("os") or None, len(manifests) == 2,
    )


def stack_identity(stack_setup: str | Path) -> tuple[str, str]:
    """Return ``(publication_date, platform)`` from an LCG view setup."""
    path = Path(stack_setup).resolve()
    try:
        match = _GENERATED_RE.search(path.read_text())
        if match:
            return parsedate_to_datetime(match.group(1)).date().isoformat(), path.parent.name
    except (OSError, TypeError, ValueError):
        pass
    return "", "unknown"


@dataclass(frozen=True)
class RepoRef:
    """A package's upstream repository, parsed far enough to build links."""

    forge: str
    host: str
    slug: str

    @property
    def url(self) -> str:
        return f"https://{self.host}/{self.slug}"

    def compare_url(self, base: str, head: str) -> str:
        return f"{self.url}{_FORGE_INFIX[self.forge]}/compare/{base}...{head}"


def parse_repo(url: str | None) -> RepoRef | None:
    """Parse a GitHub/GitLab repository URL used by attribution links."""
    match = _URL_RE.match(url.strip()) if url else None
    if not match:
        return None
    host, slug = match.group("host"), match.group("slug")
    if host == "github.com":
        if slug.count("/") != 1:
            return None
        forge = "github"
    elif "gitlab" in host:
        forge = "gitlab"
    else:
        return None
    return RepoRef(forge, host, slug)


def _manifest(stack_setup: Path) -> Path | None:
    """Map ``views/{version}/{slot}/{platform}`` to its nightly manifest."""
    try:
        platform, slot, version, views = (
            stack_setup.parent.name,
            stack_setup.parents[1].name,
            stack_setup.parents[2].name,
            stack_setup.parents[3],
        )
    except IndexError:
        return None
    if views.name != "views":
        return None
    return views.parent / "nightlies" / version / slot / f"LCG_externals_{platform}.txt"


def _repository_urls(view: str, lcgcmake_commit: str) -> dict[str, str]:
    """Read HEAD URLs from the exact, recursively included LCGCMake toolchain."""
    if not lcgcmake_commit:
        return {}
    pending, seen, repos = [f"heptools-{view}"], set(), {}
    while pending:
        toolchain = pending.pop()
        if toolchain in seen:
            continue
        seen.add(toolchain)
        url = (
            "https://gitlab.cern.ch/sft/stacks/lcgcmake/-/raw/"
            f"{lcgcmake_commit}/cmake/toolchain/{quote(toolchain, safe='')}.cmake"
        )
        try:
            with urlopen(url, timeout=10) as response:  # noqa: S310 - fixed HTTPS host
                source = response.read().decode()
        except (OSError, UnicodeError) as exc:
            _log.warning("cannot read LCGCMake repository metadata (%s)", exc)
            continue
        for name, repo in _EXTERNAL_RE.findall(source):
            repos.setdefault(name.lower(), repo)  # the including view is the override
        pending.extend(_INCLUDE_RE.findall(source))
    return repos


def read_stack(stack_setup: str | Path) -> tuple[Path | None, dict[str, dict]]:
    """Return the LCG manifest and git-built HEAD packages for a view setup."""
    setup = Path(stack_setup).resolve()
    manifest = _manifest(setup)
    if manifest is None:
        return None, {}
    try:
        rows = manifest.read_text().splitlines()
    except OSError as exc:
        _log.warning("cannot read LCG manifest '%s' (%s)", manifest, exc)
        return None, {}

    packages: dict[str, dict] = {}
    build_revisions: dict[str, str] = {}
    for name, package in parse_manifest("\n".join(rows)).items():
        if package["version"] != "HEAD":
            continue
        metadata = build_info(package["install_path"])
        if metadata.get("commit"):
            build_revisions[name] = metadata.get("githash", "")
            packages[name] = {
                "commit": metadata["commit"], "version": "HEAD", "repo_url": None,
            }
    # Incremental builds can use different toolchains within one view. Resolve
    # each package against its own build, fetching each distinct revision once.
    repos_by_revision = {
        revision: _repository_urls(setup.parents[2].name, revision)
        for revision in dict.fromkeys(build_revisions.values()) if revision
    }
    for name, package in packages.items():
        package["repo_url"] = repos_by_revision.get(build_revisions[name], {}).get(name.lower())
    return manifest, packages
