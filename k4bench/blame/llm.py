"""The one way this package talks to a language model.

Two stages ask a model questions — :mod:`k4bench.blame.rank` scores every
candidate pull request against one benchmark configuration's regressions, and
:mod:`k4bench.blame.attribute` reviews one pull request against the whole
window — and both need the same thing underneath: an HTTP request to an
OpenAI-compatible endpoint that survives a flaky provider, and a JSON reply
recovered from whatever the model actually wrote. That plumbing lives here once.

Three properties shape it, and they are the same ones the stages above rely on:

* **Model-independence.** :class:`ChatClient` speaks the OpenAI
  *chat-completions* wire shape over :mod:`requests` — no vendor SDK, no pinned
  model. Provider, model and key are environment variables
  (:func:`chat_client_from_env`), so moving between endpoints is a settings
  change, never a code change.

* **Bounded persistence.** Transient connection, timeout, 429 and 5xx failures
  retry with ``Retry-After``/exponential backoff; a reply cut off by the output
  limit retries with twice the allowance; a 400 rejecting JSON mode drops the
  field and asks again. Every one of those is bounded, and the bound is the same
  for both stages.

* **Honest failure.** :meth:`ChatClient.complete` raises once the attempts are
  spent; the *callers* turn that into "no ranking" / "no attribution", which the
  rest of the pipeline already handles. Nothing here invents a fallback answer.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

_log = logging.getLogger(__name__)


#: Initial output allowance. A ranking or attribution row is small, but some
#: OpenAI-compatible reasoning models charge hidden reasoning against
#: ``max_tokens`` before emitting the JSON, so a tight budget can truncate even a
#: handful of rows. ``K4BENCH_LLM_MAX_TOKENS`` raises or lowers this floor for a
#: particular provider/model without a code change; callers scale it further per
#: request, and :data:`MAX_OUTPUT_TOKENS` caps the whole thing.
DEFAULT_MAX_TOKENS = 4096
MAX_OUTPUT_TOKENS = 32768

_TIMEOUT = 60
_MAX_ATTEMPTS = 4
_MAX_RETRY_DELAY = 30.0


@dataclass
class ChatClient:
    """A configured OpenAI *chat-completions* endpoint.

    ``url`` is the API base (e.g. ``https://openrouter.ai/api/v1``);
    ``/chat/completions`` is appended. ``session`` is injectable so tests
    substitute a fake with no network.

    One piece of state is deliberate: ``json_mode`` starts on and latches off
    for the rest of the client's life the first time a provider rejects
    ``response_format`` with a 400. Compatibility varies across
    "OpenAI-compatible" providers, it is a property of the endpoint rather than
    of one request, and parsing never depended on it — so the discovery is made
    once per run instead of costing every call a failed attempt.
    """

    url: str
    model: str
    api_key: str | None = None
    session: requests.Session = field(default_factory=requests.Session)
    timeout: int = _TIMEOUT
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_attempts: int = _MAX_ATTEMPTS
    sleep_fn: Callable[[float], None] = field(default=time.sleep, repr=False)
    json_mode: bool = True

    def complete(
        self, system: str, user: str, *, max_output_tokens: int | None = None
    ) -> tuple[str, str]:
        """POST one exchange and return ``(assistant text, finish reason)``.

        *max_output_tokens* is the caller's estimate of what its reply needs; the
        client takes the larger of that and its configured floor, capped at
        :data:`MAX_OUTPUT_TOKENS`. Raises the last failure once the bounded
        attempts are spent — callers degrade that to "no answer".
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": min(
                MAX_OUTPUT_TOKENS, max(self.max_tokens, max_output_tokens or 0)
            ),
        }
        # JSON mode when the backend honours it; parsing never depends on it.
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.url.rstrip("/") + "/chat/completions"

        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            resp = None
            length_limited = False
            try:
                # A fresh mapping matters for diagnostics/tests as well as for
                # requests hooks that retain the submitted payload: increasing
                # the retry budget must not retroactively rewrite attempt one.
                attempt_payload = dict(payload)
                resp = self.session.post(
                    endpoint, json=attempt_payload, headers=headers, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                finish_reason = str(choice.get("finish_reason") or "unknown")
                if finish_reason == "length":
                    previous = attempt_payload["max_tokens"]
                    length_limited = True
                    payload["max_tokens"] = min(MAX_OUTPUT_TOKENS, previous * 2)
                    raise ValueError(
                        f"LLM response truncated at {previous} output tokens"
                    )
                return str(choice["message"]["content"] or ""), finish_reason
            except Exception as exc:
                last_exc = exc
                stripped_compat = False
                if (
                    isinstance(exc, requests.HTTPError)
                    and getattr(resp, "status_code", None) == 400
                    and "response_format" in payload
                ):
                    # This provider rejects JSON mode outright. Drop the field and
                    # ask once more before treating the 400 as fatal — and latch
                    # it off so the rest of the run never pays for it again.
                    payload.pop("response_format")
                    self.json_mode = False
                    stripped_compat = True
                can_retry = stripped_compat or _retryable_failure(
                    exc, resp, length_limited=length_limited,
                    can_grow=payload["max_tokens"] > attempt_payload["max_tokens"],
                )
                if not can_retry or attempt + 1 >= self.max_attempts:
                    break
                delay = (
                    0.0 if length_limited or stripped_compat
                    else _retry_delay(resp, attempt)
                )
                _log.warning(
                    "llm: attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1, self.max_attempts, exc, delay,
                )
                if delay:
                    self.sleep_fn(delay)
        raise last_exc  # bounded attempts failed; the caller degrades to no answer


def _retryable_failure(
    exc: Exception,
    response,
    *,
    length_limited: bool,
    can_grow: bool,
) -> bool:
    """Whether an endpoint failure is worth another request."""
    if length_limited:
        return can_grow
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    status = getattr(response, "status_code", None)
    if isinstance(exc, requests.HTTPError):
        return status == 429 or (status is not None and status >= 500)
    # A successful HTTP response with an invalid JSON/chat-completion shape can
    # be a transient provider proxy failure. Retry it within the same bound.
    return True


def _retry_delay(response, attempt: int) -> float:
    """Provider ``Retry-After`` seconds or bounded exponential backoff."""
    # ``requests.Response.__bool__`` is false for 4xx/5xx—the exact responses
    # whose headers matter here—so test identity, never truthiness.
    raw = (
        getattr(response, "headers", {}).get("Retry-After")
        if response is not None else None
    )
    if raw is not None:
        try:
            return min(_MAX_RETRY_DELAY, max(0.0, float(raw)))
        except (TypeError, ValueError):
            pass
    return min(_MAX_RETRY_DELAY, float(2 ** attempt))


def chat_client_from_env(*, model_env: str | None = None) -> ChatClient | None:
    """A :class:`ChatClient` from ``K4BENCH_LLM_*``, or ``None``.

    Model calls are *off by default*: unset ``K4BENCH_LLM_URL`` or
    ``K4BENCH_LLM_MODEL`` returns ``None``, and each stage then does its job
    without one. Only a configured environment (CI with the secrets, or a dev
    box for a backfill) enables them.

    *model_env* names an optional environment variable consulted before
    ``K4BENCH_LLM_MODEL`` — the seam that lets one stage run on a different
    model from another (see
    :func:`k4bench.blame.attribute.attributor_from_env`) without a second
    endpoint, key or code path.
    """
    url = os.environ.get("K4BENCH_LLM_URL", "").strip()
    model = ""
    if model_env:
        model = os.environ.get(model_env, "").strip()
    model = model or os.environ.get("K4BENCH_LLM_MODEL", "").strip()
    if not url or not model:
        return None
    api_key = os.environ.get("K4BENCH_LLM_API_KEY", "").strip() or None
    raw_max_tokens = os.environ.get("K4BENCH_LLM_MAX_TOKENS", "").strip()
    max_tokens = DEFAULT_MAX_TOKENS
    if raw_max_tokens:
        try:
            configured = int(raw_max_tokens)
            if configured <= 0:
                raise ValueError("must be positive")
            max_tokens = min(configured, MAX_OUTPUT_TOKENS)
        except ValueError:
            _log.warning(
                "llm: ignoring invalid K4BENCH_LLM_MAX_TOKENS=%r; using %d",
                raw_max_tokens, DEFAULT_MAX_TOKENS,
            )
    return ChatClient(url=url, model=model, api_key=api_key, max_tokens=max_tokens)


# ── Defensive response parsing ────────────────────────────────────────────────

#: First ``{`` to last ``}`` — recovers the object from a reply wrapped in code
#: fences or trailing prose, the two ways a chatty model breaks "JSON only".
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def extract_json(content: str) -> object | None:
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


def coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            return int(float(value))  # type: ignore[arg-type]  # "1234.0"
        except (TypeError, ValueError):
            return None


def parse_score(value: object) -> float | None:
    """A finite float clamped to 0–100, or ``None`` when *value* is not a
    number (missing, prose, NaN, infinity).

    ``None`` rejects the whole row rather than defaulting to 0.0: a zero is a
    *judgement* ("this PR is not the cause"), and publishing one the model never
    made — possibly inverting a "likelihood: high" it did make — would be a
    confident wrong answer, the exact thing this pipeline refuses to emit."""
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return max(0.0, min(100.0, score))


def one_line(value: object, limit: int) -> str:
    """*value* as a single trimmed line, newlines collapsed, length capped."""
    if not value:
        return ""
    return " ".join(str(value).split())[:limit].strip()
