"""Resolve a repo's commit range to the pull requests that landed in it.

The one network-touching module in :mod:`k4bench.blame`. It runs in CI, in the
nightly ``regression-report`` job, *after* the report is built and uploaded — so
its failures are contained to a missing ``blame.json``, never a missing report.

Everything here degrades rather than raises, with one deliberate exception: a
GitHub **rate limit** raises :class:`RateLimitError`, because past that point no
further repo can be resolved and the builder should stop and keep what it has
rather than hammer a throttled API. A 404 (``develop`` force-pushed, base commit
gone), the 250-commit compare cap, and a non-GitHub host are all normal outcomes
recorded on the result, not errors.

Auth is a ``GITHUB_TOKEN`` (5000 req/hr; a regression touches ~15 repos, so a
night is tens of calls). Without one the public limit is 60/hr and will throttle
almost immediately — the builder treats the token as effectively required and
simply produces diffs without candidates when it is missing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import requests

from k4bench.blame.models import CandidatePR

_log = logging.getLogger(__name__)

#: Squash-merge subject convention, ``"Fix step limit (#1234)"``. The last
#: ``(#N)`` on the first line is the merged PR: a title can mention an earlier
#: issue in prose, but GitHub appends the merged number at the end.
_PR_NUMBER_RE = re.compile(r"\(#(\d+)\)\s*$")

_TIMEOUT = 15
#: Bound per repo/PR so one sweeping range can't explode the call count or the
#: file lists stored in ``blame.json``.
_MAX_PRS_PER_REPO = 40
_MAX_FILES_PER_PR = 100
#: Fallback ``/commits/{sha}/pulls`` lookups per range. A non-squash repo whose
#: subjects carry no ``(#N)`` pays one API call per commit, and a compare can
#: hold up to 250 — bound that spend the same way the PR fetches are bounded.
_MAX_COMMIT_PR_LOOKUPS = 40

#: Patch budget handed to the ranker per PR. The assembled diff is *transient*
#: ranker input — never stored in ``blame.json``, always re-fetchable from GitHub
#: — so it is bounded per file and in total: free models have small context
#: windows, and one runaway PR must not crowd out the others. Overflow past
#: either cap is marked so the model knows it is reading a sample, not the whole
#: change.
_MAX_PATCH_CHARS_PER_FILE = 2000
_MAX_PATCH_CHARS_PER_PR = 6000
_PATCH_TRUNCATION_MARK = "\n… (truncated)"


class RateLimitError(RuntimeError):
    """GitHub returned a rate-limit (403/429) response — abort the night."""


@dataclass
class GitHubClient:
    """Thin authenticated GET wrapper over the GitHub REST API.

    ``session`` is injectable so tests substitute a fake with no network and so
    a real run reuses one pooled connection across the night's tens of calls.
    """

    token: str | None = None
    api_url: str = "https://api.github.com"
    session: requests.Session = field(default_factory=requests.Session)
    timeout: int = _TIMEOUT

    def get(self, path: str, params: dict | None = None) -> requests.Response:
        """GET ``{api_url}{path}``. Raises :class:`RateLimitError` on a throttled
        response; every other status is returned for the caller to interpret."""
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = self.session.get(
            f"{self.api_url}{path}", headers=headers, params=params, timeout=self.timeout
        )
        if resp.status_code in (403, 429) and _is_rate_limited(resp):
            raise RateLimitError(f"GitHub rate limit hit on {path}")
        return resp


def _is_rate_limited(resp: requests.Response) -> bool:
    """True when a 403/429 is a rate limit rather than a plain permission error.

    A primary limit sets ``X-RateLimit-Remaining: 0``; a secondary limit sends
    ``Retry-After`` or says so in the body. A 403 that is none of these (e.g. a
    private repo) is a permission problem for one repo, not a reason to abort the
    whole night."""
    if resp.headers.get("X-RateLimit-Remaining") == "0":
        return True
    if "Retry-After" in resp.headers:
        return True
    try:
        message = resp.json().get("message", "")
    except ValueError:
        return False
    return "rate limit" in message.lower()


@dataclass
class RepoResolution:
    """The GitHub half of a :class:`~k4bench.blame.models.RepoBlame`: the
    candidate PRs found in the range plus the two "couldn't see everything"
    flags the builder copies onto the blame.

    ``patches`` maps a PR number to its bounded unified-diff sample — *transient*
    ranker input the builder hands to the ranking stage, keyed alongside
    ``candidates`` but deliberately **not** part of the persisted
    :class:`CandidatePR`: the diff is re-fetchable from GitHub forever, so
    ``blame.json`` keeps only the file paths and the ranker's verdict."""

    candidates: list[CandidatePR] = field(default_factory=list)
    patches: dict[int, str] = field(default_factory=dict)
    commits_unavailable: bool = False
    truncated: bool = False


def parse_pr_number(subject: str) -> int | None:
    """The merged PR number from a commit subject, or ``None``.

    Reads only the first line: a squash-merge body can quote other commits whose
    own ``(#N)`` must not be mistaken for this merge."""
    first_line = subject.splitlines()[0] if subject else ""
    match = _PR_NUMBER_RE.search(first_line)
    return int(match.group(1)) if match else None


def resolve_repo_prs(
    client: GitHubClient, slug: str, base: str, head: str
) -> RepoResolution:
    """Pull requests merged in ``base...head`` of GitHub repo *slug*.

    One ``compare`` call yields every commit in the range; PR numbers come from
    the squash-merge subject convention, falling back to
    ``/commits/{sha}/pulls`` only for a commit whose subject carries none. Each
    distinct PR is then fetched for its title/author/churn, its changed paths,
    and a bounded diff sample (the transient ranker input, kept in
    ``result.patches``). A 404 or the 250-commit cap is recorded on the result,
    not raised.
    """
    result = RepoResolution()
    compare = client.get(f"/repos/{slug}/compare/{base}...{head}")
    if compare.status_code == 404:
        # develop was force-pushed / rewritten and base is gone; still show SHAs.
        result.commits_unavailable = True
        return result
    if compare.status_code != 200:
        _log.warning("resolve_repo_prs: compare %s %s..%s -> HTTP %s",
                     slug, base, head, compare.status_code)
        result.commits_unavailable = True
        return result

    data = compare.json()
    commits = data.get("commits", []) or []
    total = data.get("total_commits", len(commits))
    if total > len(commits):
        # The compare endpoint caps at 250 commits; a one-night window never
        # approaches it, but a wide backfill window could.
        result.truncated = True

    pr_numbers: list[int] = []
    seen: set[int] = set()
    lookups_left = _MAX_COMMIT_PR_LOOKUPS
    for commit in commits:
        if len(pr_numbers) >= _MAX_PRS_PER_REPO:
            break
        subject = (commit.get("commit") or {}).get("message", "")
        number = parse_pr_number(subject)
        if number is None:
            if lookups_left <= 0:
                continue
            lookups_left -= 1
            number = _pr_for_commit(client, slug, commit.get("sha", ""))
        if number is not None and number not in seen:
            seen.add(number)
            pr_numbers.append(number)

    for number in pr_numbers[:_MAX_PRS_PER_REPO]:
        fetched = _fetch_pr(client, slug, number)
        if fetched is not None:
            pr, patch = fetched
            result.candidates.append(pr)
            if patch:
                result.patches[number] = patch
    return result


def _pr_for_commit(client: GitHubClient, slug: str, sha: str) -> int | None:
    """The PR a commit belongs to, via the commits→pulls endpoint. Only used
    when the subject carries no ``(#N)`` — a merge commit or a non-squash repo."""
    if not sha:
        return None
    resp = client.get(f"/repos/{slug}/commits/{sha}/pulls")
    if resp.status_code != 200:
        return None
    try:
        pulls = resp.json()
    except ValueError:
        return None
    return pulls[0].get("number") if pulls else None


def _fetch_pr(
    client: GitHubClient, slug: str, number: int
) -> tuple[CandidatePR, str] | None:
    """One PR's metadata with its changed paths, and a bounded diff sample, or
    ``None`` if the PR itself cannot be read (deleted, or a transient error).

    Returns the persisted :class:`CandidatePR` alongside the *transient* patch
    text — the diff is ranker input, never stored on the candidate (see
    :class:`RepoResolution`)."""
    resp = client.get(f"/repos/{slug}/pulls/{number}")
    if resp.status_code != 200:
        _log.debug("_fetch_pr: %s#%s -> HTTP %s", slug, number, resp.status_code)
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    files, patch = _fetch_pr_files(client, slug, number)
    pr = CandidatePR(
        repo=slug,
        number=number,
        title=str(data.get("title", "")),
        author=str((data.get("user") or {}).get("login", "")),
        url=str(data.get("html_url", f"https://github.com/{slug}/pull/{number}")),
        merged_at=data.get("merged_at"),
        files=files,
        additions=int(data.get("additions") or 0),
        deletions=int(data.get("deletions") or 0),
    )
    return pr, patch


def _fetch_pr_files(
    client: GitHubClient, slug: str, number: int
) -> tuple[tuple[str, ...], str]:
    """A PR's changed paths and a bounded sample of its unified diff.

    One page (``per_page=100``) of ``/pulls/{n}/files`` carries both the paths
    the ranker keys on — persisted on the :class:`CandidatePR` — and each file's
    ``patch`` hunk, which is assembled into the transient diff sample. The paths
    are always kept (cheap, high-signal); the diff is capped per file and per PR
    (see the ``_MAX_PATCH_*`` bounds) with overflow marked ``… (truncated)``.
    Binary files and pure renames carry no ``patch``, so they contribute their
    path but no diff text."""
    resp = client.get(
        f"/repos/{slug}/pulls/{number}/files", params={"per_page": _MAX_FILES_PER_PR}
    )
    if resp.status_code != 200:
        return (), ""
    try:
        files = resp.json()
    except ValueError:
        return (), ""

    paths: list[str] = []
    chunks: list[str] = []
    used = 0
    truncated = False
    for entry in files:
        filename = entry.get("filename")
        if not filename:
            continue
        paths.append(filename)
        patch = entry.get("patch")
        if not patch:
            # Binary file or a pure rename: no hunk to show. The path already
            # rode onto ``paths`` above — it is signal even without a diff.
            continue
        if used >= _MAX_PATCH_CHARS_PER_PR:
            truncated = True
            continue
        clip = patch[:_MAX_PATCH_CHARS_PER_FILE]
        truncated = truncated or len(clip) < len(patch)
        clip = clip[: _MAX_PATCH_CHARS_PER_PR - used]
        truncated = truncated or len(clip) < len(patch)
        chunks.append(f"--- {filename} ---\n{clip}")
        used += len(clip)

    patch_text = "\n".join(chunks)
    if truncated and patch_text:
        patch_text += _PATCH_TRUNCATION_MARK
    return tuple(paths), patch_text
