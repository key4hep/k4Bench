"""Rank a regression's candidate pull requests with a language model.

The builder collects, for one confirmed regression, every pull request in the
commit range of every package that moved across the blame window — but *which*
of them caused the step is a judgement over the real diffs, not a path match.
This module makes that judgement with a model: it is handed the metric that
moved and each candidate's code change, and returns a 0–100 likelihood and a
one-line reason per PR.

Three properties are load-bearing, and shape the whole module:

* **Model-independence.** The one adapter, :class:`OpenAICompatRanker`, speaks
  the OpenAI *chat-completions* wire shape over :mod:`requests` — no vendor SDK,
  no pinned model. Provider, model and key are environment variables
  (:func:`ranker_from_env`), so switching from one free endpoint to another is a
  settings change, never a code change. :class:`Ranker` is a ``Protocol`` so a
  second adapter can drop in without the builder knowing.

* **Only-reorder.** The model may only score the candidates it was given.
  :func:`_parse_rankings` drops any ``(repo, pr)`` the request did not contain,
  so a hallucinated PR number is structurally impossible to surface — never
  merely unlikely. The builder drops unknown keys a second time (defence in
  depth).

* **Honest failure.** Every failure path — no config, HTTP error, timeout,
  malformed JSON — returns ``{}``. "No ranking" is a real state the rest of the
  pipeline already handles (the dashboard hides the ledger, the email omits the
  "most likely" line); a confident wrong culprit would be worse than none. The
  score is a *lead for a human*, in keeping with this repo's "no evidence ⇒ no
  verdict" culture — never a verdict.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Protocol

import requests

_log = logging.getLogger(__name__)


# ── The request/response contract ─────────────────────────────────────────────

@dataclass(frozen=True)
class RankCandidate:
    """One pull request offered to the ranker.

    ``repo``/``number`` are the key the builder matches the ranking back on;
    ``patch`` is the bounded diff sample (:mod:`k4bench.blame.github`) and is
    transient — it is model input, never persisted."""

    repo: str  # "owner/repo" slug
    number: int
    title: str
    files: tuple[str, ...] = ()
    patch: str = ""


@dataclass(frozen=True)
class RankRequest:
    """Everything the ranker sees for one regression: the metric that moved, and
    every candidate PR across every package that changed in the window."""

    metric: str
    metric_family: str
    direction: str
    pct_change: float | None
    detector: str
    platform: str
    sample: str
    sub_detector: str | None
    base_release: str | None
    onset_release: str
    candidates: tuple[RankCandidate, ...] = ()


@dataclass(frozen=True)
class Ranking:
    """The ranker's verdict on one PR: a 0–100 likelihood it is the cause, and a
    one-line reason grounded in its diff."""

    score: float
    description: str


class Ranker(Protocol):
    """The narrow seam the builder ranks through — model-agnostic by design."""

    def rank(self, request: RankRequest) -> dict[tuple[str, int], Ranking]:
        """Map ``(repo_slug, pr_number)`` to a :class:`Ranking`.

        Only PRs present in ``request.candidates`` may appear in the result;
        anything else the caller drops. Return ``{}`` to decline — the builder
        then leaves the candidates unranked, an honest state, not an error.
        """
        ...


# ── The OpenAI-compatible adapter ─────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You attribute a software performance regression to the pull request most "
    "likely responsible. You are given the metric that moved and, for each "
    "package that changed, the pull requests in its commit range with their "
    "code diffs. Score each PR independently 0-100 for how likely it caused the "
    "regression, and give a one-sentence reason grounded in the diff. Do not "
    "invent PRs. Output JSON only."
)

#: Total prompt-body budget (chars). Per-PR patches are already bounded in
#: :mod:`k4bench.blame.github`; this is the backstop that keeps a wide window
#: (many PRs) inside a small-context model by dropping the tail's *diffs* — the
#: file paths and titles always survive, so those PRs are still scored, just
#: from metadata (§13 of the workplan).
_MAX_PROMPT_CHARS = 45000
_MAX_FILES_LISTED = 12
_MAX_DESCRIPTION_CHARS = 200

_TIMEOUT = 60


@dataclass
class OpenAICompatRanker:
    """Ranker backed by any OpenAI *chat-completions* endpoint.

    ``url`` is the API base (e.g. ``https://openrouter.ai/api/v1``);
    ``/chat/completions`` is appended. ``session`` is injectable so tests
    substitute a fake with no network. On any HTTP/parse failure — after one
    retry — :meth:`rank` returns ``{}`` rather than raising: a late report beats
    a blocked one, and blame is a best-effort sidecar."""

    url: str
    model: str
    api_key: str | None = None
    session: requests.Session = field(default_factory=requests.Session)
    timeout: int = _TIMEOUT
    max_tokens: int = 1024

    def rank(self, request: RankRequest) -> dict[tuple[str, int], Ranking]:
        if not request.candidates:
            return {}
        try:
            content = self._complete(request)
        except Exception as exc:
            # Timeout, connection error, HTTP status, bad shape — all the same
            # outcome: no ranking. Never let the LLM path raise into the builder.
            _log.warning("rank: LLM call failed (%s) — candidates left unranked", exc)
            return {}
        return _parse_rankings(content, request)

    def _complete(self, request: RankRequest) -> str:
        """POST the prompt and return the assistant message text. One timeout +
        one retry, then give up (raise, for :meth:`rank` to swallow)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(request)},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            # JSON mode when the backend honours it; parsing never depends on it.
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.url.rstrip("/") + "/chat/completions"

        last_exc: Exception | None = None
        for attempt in range(2):  # one try, one retry
            try:
                resp = self.session.post(
                    endpoint, json=payload, headers=headers, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                return str(data["choices"][0]["message"]["content"] or "")
            except Exception as exc:
                last_exc = exc
                _log.debug("rank: attempt %d/%d failed: %s", attempt + 1, 2, exc)
        raise last_exc  # both attempts failed; rank() turns this into {}


def ranker_from_env() -> Ranker | None:
    """An :class:`OpenAICompatRanker` from ``K4BENCH_LLM_*``, or ``None``.

    Ranking is *off by default*: unset ``K4BENCH_LLM_URL`` or
    ``K4BENCH_LLM_MODEL`` returns ``None``, and the builder then collects
    candidates without scoring them. Only a configured environment (CI with the
    secrets, or a dev box for backfill) enables the model."""
    url = os.environ.get("K4BENCH_LLM_URL", "").strip()
    model = os.environ.get("K4BENCH_LLM_MODEL", "").strip()
    if not url or not model:
        return None
    api_key = os.environ.get("K4BENCH_LLM_API_KEY", "").strip() or None
    return OpenAICompatRanker(url=url, model=model, api_key=api_key)


# ── Prompt assembly ───────────────────────────────────────────────────────────

_RESPONSE_INSTRUCTION = (
    'Respond with JSON only, no prose: {"rankings": [{"repo": "<owner/repo>", '
    '"pr": <number>, "likelihood": <0-100>, "reason": "<one sentence grounded '
    'in the diff>"}]}. Score every candidate listed above and invent none.'
)


def _direction_phrase(request: RankRequest) -> str:
    """``"up +20.0%"`` / ``"down -5.0%"`` / ``"changed"`` — the mechanical sign
    of the step, never a good/bad judgement (the report's own convention)."""
    word = {"UP": "up", "DOWN": "down"}.get((request.direction or "").upper(), "changed")
    if request.pct_change is not None and math.isfinite(request.pct_change):
        return f"{word} {request.pct_change * 100:+.1f}%"
    return word


def _regression_line(request: RankRequest) -> str:
    subject = request.metric
    if request.sub_detector:
        subject += f" [{request.sub_detector}]"
    window = request.onset_release
    if request.base_release:
        window = f"{request.base_release} → {request.onset_release}"
    return (
        f"{request.detector} / {request.sample} — {subject} "
        f"{_direction_phrase(request)} between releases {window} "
        f"on {request.platform}."
    )


def _format_files(files: tuple[str, ...]) -> str:
    shown = list(files[:_MAX_FILES_LISTED])
    suffix = f", … (+{len(files) - len(shown)} more)" if len(files) > len(shown) else ""
    return ", ".join(shown) + suffix


def _render_candidate(candidate: RankCandidate, budget: int) -> tuple[str, int]:
    """One PR's prompt block and the diff budget left after it.

    The number, title and file paths are always included; the diff is appended
    only while *budget* remains, so a wide window degrades to titles+paths for
    its tail rather than overflowing a small-context model."""
    lines = [f"- #{candidate.number} — {candidate.title}"]
    if candidate.files:
        lines.append(f"  files: {_format_files(candidate.files)}")
    if candidate.patch and budget > 0:
        patch = candidate.patch
        if len(patch) > budget:
            patch = patch[:budget] + "\n… (truncated)"
        budget -= len(patch)
        lines.append("  diff:")
        lines.append(textwrap.indent(patch, "    "))
    return "\n".join(lines), budget


def _build_user_prompt(request: RankRequest) -> str:
    """The user message: the regression, then every candidate grouped by package.

    Candidates keep their input order (worst-first from the builder), so the
    diff budget, when it runs out, drops the *least* churny PRs' diffs first."""
    parts = [
        "Regression: " + _regression_line(request),
        "",
        "Candidate pull requests, grouped by package — score each on its own:",
    ]
    by_repo: dict[str, list[RankCandidate]] = {}
    for candidate in request.candidates:
        by_repo.setdefault(candidate.repo, []).append(candidate)

    budget = _MAX_PROMPT_CHARS
    for repo, candidates in by_repo.items():
        parts.append("")
        parts.append(f"## {repo}")
        for candidate in candidates:
            block, budget = _render_candidate(candidate, budget)
            parts.append(block)

    parts.append("")
    parts.append(_RESPONSE_INSTRUCTION)
    return "\n".join(parts)


# ── Defensive response parsing ────────────────────────────────────────────────

#: First ``{`` to last ``}`` — recovers the object from a reply wrapped in code
#: fences or trailing prose, the two ways a chatty model breaks "JSON only".
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def _extract_json(content: str) -> object | None:
    """Parse the model's reply as JSON, tolerating fences and stray prose."""
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\b", "", text, flags=re.I).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            return int(float(value))  # type: ignore[arg-type]  # "1234.0"
        except (TypeError, ValueError):
            return None


def _clamp_score(value: object) -> float:
    """A finite 0–100 float, or 0.0 for anything unparseable/non-finite."""
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(100.0, score))


def _one_line(value: object) -> str:
    """*value* as a single trimmed line, newlines collapsed, length capped."""
    if not value:
        return ""
    return " ".join(str(value).split())[:_MAX_DESCRIPTION_CHARS].strip()


def _parse_rankings(
    content: str, request: RankRequest
) -> dict[tuple[str, int], Ranking]:
    """Turn the model's reply into ``{(repo, number): Ranking}``.

    Enforces the only-reorder rule here as well as in the builder: a
    ``(repo, pr)`` the request did not contain is dropped, so no invented PR can
    reach the caller. Any shape drift — not an object, no ``rankings`` list, a
    row missing ``repo``/``pr`` — yields ``{}`` or skips that row rather than
    raising."""
    known = {(c.repo, c.number) for c in request.candidates}
    data = _extract_json(content)
    if not isinstance(data, dict):
        return {}
    rows = data.get("rankings")
    if not isinstance(rows, list):
        return {}

    out: dict[tuple[str, int], Ranking] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        repo = row.get("repo")
        number = _coerce_int(row.get("pr"))
        if repo is None or number is None:
            continue
        key = (str(repo), number)
        if key not in known:
            continue  # only-reorder: never surface a PR the input didn't hold
        out[key] = Ranking(
            score=_clamp_score(row.get("likelihood")),
            description=_one_line(row.get("reason")),
        )
    return out
