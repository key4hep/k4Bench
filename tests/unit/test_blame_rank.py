"""Unit tests for :mod:`k4bench.blame.rank` — the LLM candidate ranker.

Every test mocks the HTTP layer: no live model call is ever made. The contract
under test is the one the builder and the UI depend on — score each PR
independently, never surface a PR the request didn't contain, and turn *any*
failure (bad JSON, HTTP error, timeout) into ``{}`` so candidates stay unranked
rather than the pipeline breaking.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from k4bench.blame.rank import (
    OpenAICompatRanker,
    RankCandidate,
    RankRequest,
    Ranking,
    _build_user_prompt,
    ranker_from_env,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

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


def _completion(content: str) -> _FakeResp:
    """A chat-completions response whose assistant message is *content*."""
    return _FakeResp({"choices": [{"message": {"content": content}}]})


def _ranker(actions, **kwargs) -> OpenAICompatRanker:
    return OpenAICompatRanker(
        url="https://llm.example/api/v1", model="some/model",
        api_key="secret", session=_FakeSession(actions), **kwargs,
    )


def _request(candidates=None) -> RankRequest:
    if candidates is None:
        candidates = (
            RankCandidate(repo="key4hep/k4geo", number=10, title="Lower the step limit",
                          files=("FCCee/ALLEGRO/a.xml",), patch="@@\n+ more steps here"),
            RankCandidate(repo="AIDASoft/DD4hep", number=20, title="Refactor the field",
                          files=("core/field.cpp",), patch="@@\n- old code"),
        )
    return RankRequest(
        metric="wall_time_s", metric_family="time", direction="UP", pct_change=0.2,
        detector="IDEA_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_mu-", sub_detector=None,
        base_release="2026-07-03", onset_release="2026-07-04",
        candidates=candidates,
    )


def _rankings_json(*rows: dict) -> str:
    import json
    return json.dumps({"rankings": list(rows)})


# ── Prompt assembly ───────────────────────────────────────────────────────────

def test_prompt_carries_the_regression_and_every_candidate():
    prompt = _build_user_prompt(_request())
    # What moved, which way, and the window.
    assert "wall_time_s" in prompt
    assert "up +20.0%" in prompt
    assert "IDEA_o1_v03" in prompt
    assert "2026-07-03 → 2026-07-04" in prompt
    # Each package heading, each PR, and its actual diff — not just paths.
    assert "## key4hep/k4geo" in prompt and "## AIDASoft/DD4hep" in prompt
    assert "#10 — Lower the step limit" in prompt
    assert "FCCee/ALLEGRO/a.xml" in prompt
    assert "+ more steps here" in prompt
    # The response shape it must answer in.
    assert '"rankings"' in prompt


def test_prompt_direction_and_subdetector_render():
    req = _request(candidates=(
        RankCandidate(repo="key4hep/k4geo", number=1, title="t"),
    ))
    down = _build_user_prompt(RankRequest(**{**req.__dict__, "direction": "DOWN",
                                             "pct_change": -0.05, "sub_detector": "VertexBarrel"}))
    assert "down -5.0%" in down
    assert "wall_time_s [VertexBarrel]" in down


# ── Parsing a good response ───────────────────────────────────────────────────

def test_parses_a_good_response_scoring_each_pr_independently():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 80, "reason": "raises step count"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 15, "reason": "unrelated cleanup"},
    )
    result = _ranker([_completion(body)]).rank(_request())
    assert result[("key4hep/k4geo", 10)] == Ranking(80.0, "raises step count")
    assert result[("AIDASoft/DD4hep", 20)] == Ranking(15.0, "unrelated cleanup")


def test_sends_model_endpoint_and_auth():
    ranker = _ranker([_completion(_rankings_json())])
    ranker.rank(_request())
    call = ranker.session.calls[0]
    assert call.url == "https://llm.example/api/v1/chat/completions"
    assert call.json["model"] == "some/model"
    assert call.json["response_format"] == {"type": "json_object"}
    assert call.headers["Authorization"] == "Bearer secret"


def test_parses_json_wrapped_in_code_fences():
    body = "```json\n" + _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 50, "reason": "maybe"}
    ) + "\n```"
    result = _ranker([_completion(body)]).rank(_request())
    assert result[("key4hep/k4geo", 10)].score == 50.0


def test_parses_json_embedded_in_prose():
    body = "Sure! Here is my assessment: " + _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 42, "reason": "plausible"}
    ) + " Hope this helps."
    result = _ranker([_completion(body)]).rank(_request())
    assert result[("key4hep/k4geo", 10)].score == 42.0


# ── Guardrails ────────────────────────────────────────────────────────────────

def test_invented_pr_is_dropped():
    # The model returns a PR that was never in the request → it must not appear.
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": "real"},
        {"repo": "key4hep/ghost", "pr": 999, "likelihood": 99, "reason": "hallucinated"},
    )
    result = _ranker([_completion(body)]).rank(_request())
    assert set(result) == {("key4hep/k4geo", 10)}
    assert ("key4hep/ghost", 999) not in result


def test_scores_are_clamped_to_0_100():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 150, "reason": "over"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": -5, "reason": "under"},
    )
    result = _ranker([_completion(body)]).rank(_request())
    assert result[("key4hep/k4geo", 10)].score == 100.0
    assert result[("AIDASoft/DD4hep", 20)].score == 0.0


def test_description_is_collapsed_to_one_line():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 60,
         "reason": "line one\nline two\t  with   spaces"}
    )
    result = _ranker([_completion(body)]).rank(_request())
    assert result[("key4hep/k4geo", 10)].description == "line one line two with spaces"


def test_non_numeric_likelihood_becomes_zero_not_an_error():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": "very high", "reason": "x"}
    )
    result = _ranker([_completion(body)]).rank(_request())
    assert result[("key4hep/k4geo", 10)].score == 0.0


# ── Failure modes all collapse to {} ─────────────────────────────────────────

def test_malformed_json_yields_empty():
    result = _ranker([_completion("I cannot help with that.")]).rank(_request())
    assert result == {}


def test_missing_rankings_key_yields_empty():
    result = _ranker([_completion('{"something_else": 1}')]).rank(_request())
    assert result == {}


def test_http_error_yields_empty_after_retry():
    ranker = _ranker([_FakeResp({}, status=500), _FakeResp({}, status=500)])
    assert ranker.rank(_request()) == {}
    assert len(ranker.session.calls) == 2  # one try, one retry, then give up


def test_timeout_yields_empty():
    ranker = _ranker([requests.Timeout("slow"), requests.Timeout("slow")])
    assert ranker.rank(_request()) == {}


def test_transient_error_then_success_is_retried():
    body = _rankings_json({"repo": "key4hep/k4geo", "pr": 10, "likelihood": 55, "reason": "ok"})
    ranker = _ranker([requests.ConnectionError("blip"), _completion(body)])
    result = ranker.rank(_request())
    assert result[("key4hep/k4geo", 10)].score == 55.0
    assert len(ranker.session.calls) == 2


def test_no_candidates_short_circuits_without_calling_the_model():
    ranker = _ranker([])  # no queued action; a post() would IndexError
    assert ranker.rank(_request(candidates=())) == {}
    assert ranker.session.calls == []


# ── ranker_from_env ───────────────────────────────────────────────────────────

def test_ranker_from_env_none_when_unset(monkeypatch):
    for var in ("K4BENCH_LLM_URL", "K4BENCH_LLM_MODEL", "K4BENCH_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert ranker_from_env() is None


def test_ranker_from_env_none_when_only_url_set(monkeypatch):
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/api/v1")
    monkeypatch.delenv("K4BENCH_LLM_MODEL", raising=False)
    assert ranker_from_env() is None


def test_ranker_from_env_builds_ranker_when_configured(monkeypatch):
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/api/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "some/model")
    monkeypatch.setenv("K4BENCH_LLM_API_KEY", "k")
    ranker = ranker_from_env()
    assert isinstance(ranker, OpenAICompatRanker)
    assert ranker.url == "https://llm.example/api/v1"
    assert ranker.model == "some/model"
    assert ranker.api_key == "k"


def test_ranker_from_env_key_optional(monkeypatch):
    # A keyless endpoint (local server) is valid — URL+model are the only gate.
    monkeypatch.setenv("K4BENCH_LLM_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "local")
    monkeypatch.delenv("K4BENCH_LLM_API_KEY", raising=False)
    ranker = ranker_from_env()
    assert isinstance(ranker, OpenAICompatRanker)
    assert ranker.api_key is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
