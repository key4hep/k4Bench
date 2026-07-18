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

from k4bench.blame import rank as rank_mod
from k4bench.blame.rank import (
    MetricStep,
    OpenAICompatRanker,
    RankCandidate,
    RankRequest,
    Ranking,
    _build_user_prompt,
    ranker_from_env,
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


def _ranker(actions, **kwargs) -> OpenAICompatRanker:
    kwargs.setdefault("sleep_fn", lambda _seconds: None)
    return OpenAICompatRanker(
        url="https://llm.example/api/v1", model="some/model",
        api_key="secret", session=_FakeSession(actions), **kwargs,
    )


def _request(candidates=None, metrics=None) -> RankRequest:
    if candidates is None:
        candidates = (
            RankCandidate(repo="key4hep/k4geo", number=10, title="Lower the step limit",
                          files=("FCCee/ALLEGRO/a.xml",), patch="@@\n+ more steps here"),
            RankCandidate(repo="AIDASoft/DD4hep", number=20, title="Refactor the field",
                          files=("core/field.cpp",), patch="@@\n- old code"),
        )
    if metrics is None:
        metrics = (
            MetricStep(metric="wall_time_s", metric_family="time", direction="UP", pct_change=0.2),
        )
    return RankRequest(
        metrics=metrics,
        detector="IDEA_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_mu-",
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
    down = _build_user_prompt(_request(
        candidates=(RankCandidate(repo="key4hep/k4geo", number=1, title="t"),),
        metrics=(MetricStep(metric="wall_time_s", metric_family="time",
                             direction="DOWN", pct_change=-0.05,
                             sub_detector="VertexBarrel"),),
    ))
    assert "down -5.0%" in down
    assert "wall_time_s [VertexBarrel]" in down


def test_prompt_carries_every_metric_sharing_the_window():
    # Two metrics stepped across the same release boundary — the model must see
    # both, not just one arbitrary metric standing in for the window.
    prompt = _build_user_prompt(_request(metrics=(
        MetricStep(metric="wall_time_s", metric_family="time", direction="UP", pct_change=0.2),
        MetricStep(metric="peak_rss_mb", metric_family="memory", direction="UP", pct_change=0.15),
    )))
    assert "wall_time_s" in prompt and "up +20.0%" in prompt
    assert "peak_rss_mb" in prompt and "up +15.0%" in prompt


def test_diff_budget_is_shared_fairly_not_first_come_first_served(monkeypatch):
    # Under budget pressure a candidate's position must not decide whether its
    # diff survives: the small diff stays whole, the two large ones shrink
    # evenly — including the *first* one, which a sequential budget would have
    # let swallow everything.
    monkeypatch.setattr(rank_mod, "_MAX_PROMPT_CHARS", 100)
    candidates = (
        RankCandidate(repo="key4hep/k4geo", number=1, title="big1", patch="~" * 300),
        RankCandidate(repo="key4hep/k4geo", number=2, title="small", patch="=" * 30),
        RankCandidate(repo="key4hep/k4geo", number=3, title="big2", patch="^" * 300),
    )
    prompt = _build_user_prompt(_request(candidates=candidates))
    assert "=" * 30 in prompt                    # the small diff is whole
    assert prompt.count("~") == prompt.count("^")  # the big ones shrink evenly
    assert 0 < prompt.count("~") < 300


def test_allocate_diff_budget_waterfills():
    assert rank_mod._allocate_diff_budget([30, 300, 300], 100) == [30, 35, 35]
    assert rank_mod._allocate_diff_budget([10, 20], 100) == [10, 20]  # all fits
    assert rank_mod._allocate_diff_budget([], 100) == []
    assert rank_mod._allocate_diff_budget([50, 50], 0) == [0, 0]


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
    ranker = _ranker([_completion(_rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 1, "reason": "x"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 1, "reason": "y"},
    ))])
    ranker.rank(_request())
    call = ranker.session.calls[0]
    assert call.url == "https://llm.example/api/v1/chat/completions"
    assert call.json["model"] == "some/model"
    assert call.json["response_format"] == {"type": "json_object"}
    assert call.headers["Authorization"] == "Bearer secret"


def test_output_budget_scales_for_wide_candidate_windows():
    candidates = tuple(
        RankCandidate(repo="key4hep/k4geo", number=n, title=f"PR {n}")
        for n in range(1, 6)
    )
    ranker = _ranker([_completion(_rankings_json(*(
        {"repo": c.repo, "pr": c.number, "likelihood": 1, "reason": "x"}
        for c in candidates
    )))])
    ranker.rank(_request(candidates=candidates))
    assert ranker.session.calls[0].json["max_tokens"] == 4096


def test_length_limited_response_retries_with_twice_the_output_budget():
    candidates = tuple(
        RankCandidate(repo="key4hep/k4geo", number=n, title=f"PR {n}")
        for n in range(1, 6)
    )
    body = _rankings_json(*(
        {"repo": c.repo, "pr": c.number, "likelihood": 0, "reason": "unrelated"}
        for c in candidates
    ))
    ranker = _ranker([
        _completion('{"rankings":[', finish_reason="length"),
        _completion(body),
    ])
    result = ranker.rank(_request(candidates=candidates))
    assert len(result) == 5
    assert all(r.score == 0 and r.description == "unrelated" for r in result.values())
    assert ranker.session.calls[0].json["max_tokens"] == 4096
    assert ranker.session.calls[1].json["max_tokens"] == 8192


def test_parses_json_wrapped_in_code_fences():
    body = "```json\n" + _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 50, "reason": "maybe"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    ) + "\n```"
    result = _ranker([_completion(body)]).rank(_request())
    assert result[("key4hep/k4geo", 10)].score == 50.0


def test_parses_json_embedded_in_prose():
    body = "Sure! Here is my assessment: " + _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 42, "reason": "plausible"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    ) + " Hope this helps."
    result = _ranker([_completion(body)]).rank(_request())
    assert result[("key4hep/k4geo", 10)].score == 42.0


# ── Guardrails ────────────────────────────────────────────────────────────────

def test_invented_pr_is_dropped():
    # The model returns a PR that was never in the request → it must not appear.
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": "real"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
        {"repo": "key4hep/ghost", "pr": 999, "likelihood": 99, "reason": "hallucinated"},
    )
    result = _ranker([_completion(body)]).rank(_request())
    assert set(result) == {("key4hep/k4geo", 10), ("AIDASoft/DD4hep", 20)}
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
         "reason": "line one\nline two\t  with   spaces"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    )
    result = _ranker([_completion(body)]).rank(_request())
    assert result[("key4hep/k4geo", 10)].description == "line one line two with spaces"


def test_non_numeric_likelihood_rejects_the_row():
    # "very high" must not be published as 0% — that would invert the model's
    # meaning. The row is rejected; the candidate stays unranked (and coverage
    # then blocks publication rather than shipping a made-up zero).
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": "very high", "reason": "x"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    )
    result = _ranker([_completion(body), _completion(body)]).rank(_request())
    assert ("key4hep/k4geo", 10) not in result
    assert result[("AIDASoft/DD4hep", 20)].score == 5.0


def test_empty_reason_rejects_the_row_and_is_recovered_by_the_retry():
    # A bare score without a reason violates the contract and would die at the
    # coverage gate anyway — rejecting it here makes the candidate "missing",
    # so the follow-up attempt can recover the sidecar instead of losing it.
    first = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": ""},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    )
    second = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": "explains it"},
    )
    ranker = _ranker([_completion(first), _completion(second)])
    result = ranker.rank(_request())
    assert result[("key4hep/k4geo", 10)] == Ranking(70.0, "explains it")
    assert result[("AIDASoft/DD4hep", 20)].score == 5.0
    assert len(ranker.session.calls) == 2


def test_missing_and_non_finite_likelihoods_reject_the_row():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "reason": "no likelihood at all"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": float("nan"), "reason": "nan"},
    )
    result = _ranker([_completion(body), _completion(body)]).rank(_request())
    assert result == {}


# ── Failure modes all collapse to {} ─────────────────────────────────────────

def test_malformed_json_yields_empty_and_warns(caplog):
    result = _ranker([
        _completion("I cannot help with that."),
        _completion("I cannot help with that."),
    ]).rank(_request())
    assert result == {}
    assert "no usable ranking (0/2 candidates" in caplog.text
    assert "response prefix='I cannot help with that.'" in caplog.text


def test_missing_rankings_key_yields_empty():
    result = _ranker([
        _completion('{"something_else": 1}'),
        _completion('{"something_else": 1}'),
    ]).rank(_request())
    assert result == {}


def test_partial_response_is_completed_by_one_followup_call():
    first = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": "high"}
    )
    second = _rankings_json(
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"}
    )
    ranker = _ranker([_completion(first), _completion(second)])
    result = ranker.rank(_request())
    assert set(result) == {("key4hep/k4geo", 10), ("AIDASoft/DD4hep", 20)}
    assert len(ranker.session.calls) == 2


def test_http_error_yields_empty_after_retry():
    ranker = _ranker([_FakeResp({}, status=500) for _ in range(4)])
    assert ranker.rank(_request()) == {}
    assert len(ranker.session.calls) == 4


def test_timeout_yields_empty():
    ranker = _ranker([requests.Timeout("slow") for _ in range(4)])
    assert ranker.rank(_request()) == {}


def test_rate_limit_honours_retry_after():
    delays = []
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 55, "reason": "ok"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    )
    ranker = _ranker([
        _FakeResp({}, status=429, headers={"Retry-After": "7"}),
        _completion(body),
    ], sleep_fn=delays.append)
    assert len(ranker.rank(_request())) == 2
    assert delays == [7.0]


def test_400_retries_once_without_response_format_then_succeeds():
    # Not every "OpenAI-compatible" provider implements JSON mode; a 400 gets
    # one retry with the optional field stripped before counting as fatal.
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 55, "reason": "ok"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    )
    ranker = _ranker([_FakeResp({}, status=400), _completion(body)])
    assert len(ranker.rank(_request())) == 2
    assert "response_format" in ranker.session.calls[0].json
    assert "response_format" not in ranker.session.calls[1].json


def test_non_retryable_client_error_fails_after_the_compat_strip():
    # A 400 that persists without response_format is a real request error.
    ranker = _ranker([_FakeResp({}, status=400), _FakeResp({}, status=400)])
    assert ranker.rank(_request()) == {}
    assert len(ranker.session.calls) == 2


def test_transient_error_then_success_is_retried():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 55, "reason": "ok"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    )
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


def test_ranker_from_env_accepts_max_tokens_override(monkeypatch):
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "reasoning-model")
    monkeypatch.setenv("K4BENCH_LLM_MAX_TOKENS", "16384")
    ranker = ranker_from_env()
    assert isinstance(ranker, OpenAICompatRanker)
    assert ranker.max_tokens == 16384


def test_ranker_from_env_ignores_invalid_max_tokens(monkeypatch, caplog):
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "some-model")
    monkeypatch.setenv("K4BENCH_LLM_MAX_TOKENS", "many")
    ranker = ranker_from_env()
    assert isinstance(ranker, OpenAICompatRanker)
    assert ranker.max_tokens == 4096
    assert "ignoring invalid K4BENCH_LLM_MAX_TOKENS" in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
