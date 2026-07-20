"""Unit tests for :mod:`k4bench.blame.llm` — the one way this package talks to a
language model.

Both model stages (:mod:`k4bench.blame.rank`, :mod:`k4bench.blame.attribute`)
share this transport, so its contract is asserted once, here: survive a flaky
provider within a bounded number of attempts, grow the output budget when a
reply is cut off, tolerate a provider that rejects JSON mode, and recover the
JSON object from whatever the model actually wrote. Every test mocks the HTTP
layer — no live model call is ever made.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from k4bench.blame.llm import (
    DEFAULT_MAX_TOKENS,
    ChatClient,
    chat_client_from_env,
    coerce_int,
    extract_json,
    one_line,
    parse_score,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, body, status=200, headers=None):
        self._body = body
        self.status_code = status
        self.headers = headers or {}

    def __bool__(self):
        return self.status_code < 400  # mirror requests.Response truthiness

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class _FakeSession:
    """Serves one queued action per ``post`` — a :class:`_FakeResp` to return or
    an ``Exception`` to raise — so a test can script a retry."""

    def __init__(self, actions):
        self._actions = list(actions)
        self.calls: list[SimpleNamespace] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(SimpleNamespace(url=url, json=json, headers=headers))
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _completion(content: str, *, finish_reason: str = "stop") -> _FakeResp:
    """A chat-completions response whose assistant message is *content*."""
    return _FakeResp({
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    })


def _client(actions, **kwargs) -> ChatClient:
    kwargs.setdefault("sleep_fn", lambda _seconds: None)
    kwargs.setdefault("api_key", "secret")
    return ChatClient(
        url="https://llm.example/api/v1", model="some/model",
        session=_FakeSession(actions), **kwargs,
    )


def _complete(client: ChatClient, **kwargs) -> tuple[str, str]:
    return client.complete("system text", "user text", **kwargs)


# ── The request ───────────────────────────────────────────────────────────────

def test_sends_model_endpoint_auth_and_both_messages():
    client = _client([_completion("ok")])
    assert _complete(client) == ("ok", "stop")
    call = client.session.calls[0]
    assert call.url == "https://llm.example/api/v1/chat/completions"
    assert call.json["model"] == "some/model"
    assert call.json["response_format"] == {"type": "json_object"}
    assert call.json["temperature"] == 0
    assert call.headers["Authorization"] == "Bearer secret"
    assert [m["role"] for m in call.json["messages"]] == ["system", "user"]
    assert call.json["messages"][0]["content"] == "system text"
    assert call.json["messages"][1]["content"] == "user text"


def test_a_keyless_endpoint_sends_no_authorization():
    client = _client([_completion("ok")], api_key=None)
    _complete(client)
    assert "Authorization" not in client.session.calls[0].headers


def test_the_output_budget_is_the_larger_of_the_floor_and_the_request():
    client = _client([_completion("ok"), _completion("ok")])
    _complete(client, max_output_tokens=100)  # under the floor
    _complete(client, max_output_tokens=9000)  # over it
    assert client.session.calls[0].json["max_tokens"] == DEFAULT_MAX_TOKENS
    assert client.session.calls[1].json["max_tokens"] == 9000


# ── Bounded persistence ───────────────────────────────────────────────────────

def test_a_length_limited_reply_retries_with_twice_the_budget():
    client = _client([
        _completion('{"partial":', finish_reason="length"),
        _completion('{"whole": true}'),
    ])
    assert _complete(client)[0] == '{"whole": true}'
    first, second = (c.json["max_tokens"] for c in client.session.calls)
    assert second == 2 * first


def test_a_growing_budget_does_not_rewrite_the_earlier_attempt():
    # The recorded payload of attempt one must still show what was actually
    # sent — a shared mutable payload would make retries unreadable in a log.
    client = _client([
        _completion("", finish_reason="length"),
        _completion("done"),
    ])
    _complete(client)
    assert client.session.calls[0].json["max_tokens"] == DEFAULT_MAX_TOKENS


def test_a_rate_limit_honours_retry_after():
    delays: list[float] = []
    client = _client([
        _FakeResp({}, status=429, headers={"Retry-After": "7"}),
        _completion("ok"),
    ], sleep_fn=delays.append)
    assert _complete(client)[0] == "ok"
    assert delays == [7.0]


def test_an_unparsable_retry_after_falls_back_to_backoff():
    delays: list[float] = []
    client = _client([
        _FakeResp({}, status=503, headers={"Retry-After": "soon"}),
        _completion("ok"),
    ], sleep_fn=delays.append)
    _complete(client)
    assert delays == [1.0]


def test_a_transient_connection_error_is_retried():
    client = _client([requests.ConnectionError("blip"), _completion("ok")])
    assert _complete(client)[0] == "ok"
    assert len(client.session.calls) == 2


def test_a_persistent_server_error_raises_after_the_attempt_budget():
    client = _client([_FakeResp({}, status=500) for _ in range(4)])
    with pytest.raises(requests.HTTPError):
        _complete(client)
    assert len(client.session.calls) == 4


def test_a_timeout_raises_after_the_attempt_budget():
    client = _client([requests.Timeout("slow") for _ in range(4)])
    with pytest.raises(requests.Timeout):
        _complete(client)


# ── JSON-mode compatibility ───────────────────────────────────────────────────

def test_a_400_retries_once_without_response_format():
    client = _client([_FakeResp({}, status=400), _completion("ok")])
    assert _complete(client)[0] == "ok"
    assert "response_format" in client.session.calls[0].json
    assert "response_format" not in client.session.calls[1].json


def test_the_json_mode_strip_latches_for_the_rest_of_the_run():
    # Compatibility is a property of the endpoint, not of one request: once a
    # provider has rejected JSON mode, no later call may pay for it again.
    client = _client([_FakeResp({}, status=400), _completion("ok"), _completion("ok")])
    _complete(client)
    _complete(client)
    assert client.json_mode is False
    assert "response_format" not in client.session.calls[2].json


def test_a_400_that_persists_without_json_mode_is_fatal():
    client = _client([_FakeResp({}, status=400), _FakeResp({}, status=400)])
    with pytest.raises(requests.HTTPError):
        _complete(client)
    assert len(client.session.calls) == 2  # not retried a third time


# ── chat_client_from_env ──────────────────────────────────────────────────────

def _clear_llm_env(monkeypatch):
    for var in (
        "K4BENCH_LLM_URL", "K4BENCH_LLM_MODEL", "K4BENCH_LLM_API_KEY",
        "K4BENCH_LLM_MAX_TOKENS", "K4BENCH_LLM_SUMMARY_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_env_none_when_unset(monkeypatch):
    _clear_llm_env(monkeypatch)
    assert chat_client_from_env() is None


def test_env_none_when_only_the_url_is_set(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/api/v1")
    assert chat_client_from_env() is None


def test_env_builds_a_client_when_configured(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/api/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "some/model")
    monkeypatch.setenv("K4BENCH_LLM_API_KEY", "k")
    client = chat_client_from_env()
    assert (client.url, client.model, client.api_key) == (
        "https://llm.example/api/v1", "some/model", "k",
    )


def test_env_key_is_optional(monkeypatch):
    # A keyless endpoint (a local server) is valid — URL+model are the only gate.
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("K4BENCH_LLM_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "local")
    assert chat_client_from_env().api_key is None


def test_env_model_override_wins_but_only_when_set(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "cheap/model")
    assert chat_client_from_env(model_env="K4BENCH_LLM_SUMMARY_MODEL").model == (
        "cheap/model"
    )
    monkeypatch.setenv("K4BENCH_LLM_SUMMARY_MODEL", "strong/model")
    assert chat_client_from_env(model_env="K4BENCH_LLM_SUMMARY_MODEL").model == (
        "strong/model"
    )
    # The override is per-stage: the stage that does not ask for it is unaffected.
    assert chat_client_from_env().model == "cheap/model"


def test_env_accepts_a_max_tokens_override(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "reasoning-model")
    monkeypatch.setenv("K4BENCH_LLM_MAX_TOKENS", "16384")
    assert chat_client_from_env().max_tokens == 16384


@pytest.mark.parametrize("bad", ["many", "0", "-1"])
def test_env_ignores_an_invalid_max_tokens(monkeypatch, caplog, bad):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "some-model")
    monkeypatch.setenv("K4BENCH_LLM_MAX_TOKENS", bad)
    assert chat_client_from_env().max_tokens == DEFAULT_MAX_TOKENS
    assert "ignoring invalid K4BENCH_LLM_MAX_TOKENS" in caplog.text


# ── Defensive response parsing ────────────────────────────────────────────────

def test_extract_json_reads_plain_fenced_and_embedded_objects():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 1} Hope this helps.') == {"a": 1}


@pytest.mark.parametrize("content", ["", "I cannot help with that.", "{not json}"])
def test_extract_json_returns_none_on_anything_unusable(content):
    assert extract_json(content) is None


def test_parse_score_clamps_to_the_scale():
    assert parse_score(150) == 100.0
    assert parse_score(-5) == 0.0
    assert parse_score("42") == 42.0


@pytest.mark.parametrize("value", [None, "very high", float("nan"), float("inf")])
def test_parse_score_rejects_what_is_not_a_number(value):
    # None rejects the row; publishing 0.0 would invert "likelihood: high" into
    # a confident acquittal the model never made.
    assert parse_score(value) is None


def test_coerce_int_reads_numbers_written_as_text_or_floats():
    assert coerce_int("1234") == 1234
    assert coerce_int("1234.0") == 1234
    assert coerce_int(None) is None
    assert coerce_int("PR #12") is None


def test_one_line_flattens_and_clips():
    assert one_line("line one\nline two\t  with   spaces", 100) == (
        "line one line two with spaces"
    )
    assert one_line("x" * 50, 10) == "x" * 10
    assert one_line(None, 10) == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
