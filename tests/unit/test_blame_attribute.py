"""Unit tests for :mod:`k4bench.blame.attribute` — the cross-configuration review.

This is the pass that decides what a pull-request comment claims, so the tests
here are about the two things that make such a claim defensible: the model is
*shown* the evidence that distinguishes a detector-specific cause from a shared
one (which configurations moved, and which measured the same window and did
not), and nothing it says can put a regression in front of a reader that k4Bench
did not measure.

Every test mocks the HTTP layer — no live model call is ever made.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests

from k4bench.blame import attribute as attr_mod
from k4bench.blame.attribute import (
    AttributionRequest,
    CompetingPR,
    OpenAICompatAttributor,
    PackageChangeFact,
    RegressionFact,
    ScopeOutcome,
    attributor_from_env,
    build_user_prompt,
)
from k4bench.blame.llm import ChatClient


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, body, status=200, headers=None):
        self._body = body
        self.status_code = status
        self.headers = headers or {}

    def __bool__(self):
        return self.status_code < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class _FakeSession:
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
    return _FakeResp({
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    })


def _attributor(actions, **kwargs) -> OpenAICompatAttributor:
    kwargs.setdefault("sleep_fn", lambda _seconds: None)
    return OpenAICompatAttributor(client=ChatClient(
        url="https://llm.example/api/v1", model="some/model",
        api_key="secret", session=_FakeSession(actions), **kwargs,
    ))


def _reply(summary: str = "ALLEGRO moved and IDEA did not.", **likelihoods) -> str:
    return json.dumps({
        "summary": summary,
        "attributions": [
            {"id": row_id, "likelihood": value}
            for row_id, value in likelihoods.items()
        ],
    })


# ── Fixtures for the request ──────────────────────────────────────────────────

_PLATFORM = "x86_64-almalinux9-gcc14.2.0-opt"


def _fact(row_id="r1", detector="ALLEGRO_o1_v03", metric="wall_time_s",
          label="baseline", **kw) -> RegressionFact:
    return RegressionFact(
        id=row_id, detector=detector, platform=kw.pop("platform", _PLATFORM),
        sample=kw.pop("sample", "p8_ee_Zbb_ecm91"), label=label, metric=metric,
        metric_family=kw.pop("metric_family", "time"),
        sub_detector=kw.pop("sub_detector", None),
        direction=kw.pop("direction", "UP"), pct_change=kw.pop("pct_change", 0.18),
        value=kw.pop("value", 0.412), baseline_median=kw.pop("baseline_median", 0.348),
        z_score=kw.pop("z_score", 8.1),
        scope_score=kw.pop("scope_score", 91.0),
        scope_reason=kw.pop("scope_reason", "raises the step count"),
        scope_state=kw.pop("scope_state", "ranked"),
    )


def _request(**kw) -> AttributionRequest:
    kw.setdefault("regressions", (_fact(),))
    return AttributionRequest(
        repo=kw.pop("repo", "key4hep/k4geo"),
        number=kw.pop("number", 1234),
        title=kw.pop("title", "Lower the step limit"),
        base_release=kw.pop("base_release", "2026-06-27"),
        onset_release=kw.pop("onset_release", "2026-07-04"),
        files=kw.pop("files", ("FCCee/ALLEGRO/compact/a.xml",)),
        patch=kw.pop("patch", "@@\n+ more steps here"),
        additions=kw.pop("additions", 12),
        deletions=kw.pop("deletions", 3),
        **kw,
    )


# ── The prompt carries the evidence ───────────────────────────────────────────

def test_prompt_carries_every_regression_with_its_id_and_measurement():
    prompt = build_user_prompt(_request(regressions=(
        _fact("r1", metric="wall_time_s"),
        _fact("r2", metric="sim_mem_mb", metric_family="memory", pct_change=0.09),
    )))
    assert "[r1] wall_time_s (baseline)" in prompt
    assert "[r2] sim_mem_mb (baseline)" in prompt
    assert "up +18.0%" in prompt and "up +9.0%" in prompt
    # A percentage alone under-reads: the absolute size and the distance from
    # the noise are what separate a marginal step from an unmistakable one.
    assert "0.412 vs 0.348 baseline" in prompt
    assert "z=8.1" in prompt


def test_prompt_states_the_window_and_the_run_context():
    prompt = build_user_prompt(_request())
    assert "Change window: 2026-06-27 → 2026-07-04" in prompt
    assert "### ALLEGRO_o1_v03" in prompt
    assert "p8_ee_Zbb_ecm91" in prompt
    assert "Pythia8: e⁺e⁻ → Z → bb (91 GeV)" in prompt
    for part in ("x86_64", "AlmaLinux 9", "GCC 14.2.0", "optimized"):
        assert part in prompt


def test_prompt_groups_rows_by_configuration_so_the_pattern_is_readable():
    # The comparison across detectors is the whole task; the model should read
    # it off the shape of the prompt, not reconstruct it from a flat list.
    prompt = build_user_prompt(_request(regressions=(
        _fact("r1", detector="ALLEGRO_o1_v03"),
        _fact("r2", detector="ALLEGRO_o1_v03", label="without_HCal"),
        _fact("r3", detector="ALLEGRO_o2_v01"),
    )))
    assert prompt.count("### ALLEGRO_o1_v03") == 1
    assert prompt.count("### ALLEGRO_o2_v01") == 1
    assert "(without_HCal)" in prompt


def test_prompt_carries_the_first_passs_prior_for_this_pull_request():
    prompt = build_user_prompt(_request(regressions=(
        _fact(scope_score=91.0, scope_reason="raises the step count"),
    )))
    assert "Earlier per-configuration review of this pull request here: 91/100" in prompt
    assert "raises the step count" in prompt


def test_prompt_states_what_measured_the_window_and_did_not_confirm():
    # The negative evidence, and the reason this stage exists.
    prompt = build_user_prompt(_request(outcomes=(
        ScopeOutcome(detector="IDEA_o1_v03", platform=_PLATFORM,
                     sample="p8_ee_Zbb_ecm91", label="baseline", status="clean"),
        ScopeOutcome(detector="IDEA_o2_v01", platform=_PLATFORM,
                     sample="p8_ee_Zbb_ecm91", label="without_HCAL",
                     status="watch", watched=("wall_time_s",)),
    )))
    assert "did NOT confirm" in prompt
    assert "IDEA_o1_v03" in prompt and "no metric stepped" in prompt
    assert "IDEA_o2_v01" in prompt
    assert "under the confirmation threshold (wall_time_s)" in prompt
    # The configuration label is part of the identity: without it the prompt's
    # "baseline vs without_<X>" reasoning has nothing to attach to.
    assert "· baseline:" in prompt and "· without_HCAL:" in prompt


def test_prompt_sizes_the_release_diff_by_what_did_not_change():
    prompt = build_user_prompt(_request(
        packages_by_platform={_PLATFORM: (
            PackageChangeFact(package="k4geo", status="CHANGED"),
            PackageChangeFact(package="edm4hep", status="ADDED"),
        )},
        unchanged_by_platform={_PLATFORM: 18},
    ))
    assert "2 of 20 tracked" in prompt
    # One platform: the sentence stays unqualified, because there is nothing to
    # tell apart.
    assert "Packages that changed across the release window (" in prompt
    assert "- k4geo" in prompt
    assert "- edm4hep [ADDED]" in prompt


def test_prompt_carries_the_pull_request_under_review_with_its_diff():
    prompt = build_user_prompt(_request())
    assert "key4hep/k4geo#1234: Lower the step limit (+12/-3)" in prompt
    assert "FCCee/ALLEGRO/compact/a.xml" in prompt
    assert "+ more steps here" in prompt


def test_prompt_carries_every_competitor_with_its_prior_and_diff():
    prompt = build_user_prompt(_request(competitors=(
        CompetingPR(repo="AIDASoft/DD4hep", number=20, url="https://gh/dd4hep/20",
                    title="Refactor the field", files=("core/field.cpp",),
                    additions=4, deletions=40, scope_score=61.0,
                    scope_reason="touches shared stepping", patch="- old code"),
    )))
    assert "AIDASoft/DD4hep#20 — Refactor the field (+4/-40)" in prompt
    assert "https://gh/dd4hep/20" in prompt
    assert "core/field.cpp" in prompt
    assert "earlier per-configuration review: 61/100 — touches shared stepping" in prompt
    assert "- old code" in prompt


def test_prompt_says_so_when_this_was_the_only_candidate():
    prompt = build_user_prompt(_request(competitors=()))
    assert "this is the only candidate" in prompt


def test_an_unrecognized_sample_and_platform_degrade_to_the_raw_names():
    prompt = build_user_prompt(_request(regressions=(
        _fact(sample="brand_new_sample", platform="riscv64-unknown"),
    )))
    assert "brand_new_sample" in prompt
    assert "riscv64-unknown" in prompt


def test_the_reviewed_diff_keeps_its_floor_against_a_crowded_window():
    # Thirty competing pull requests must not be able to price the diff under
    # review out of its own prompt.
    subject = "S" * 20000
    competitors = tuple(
        CompetingPR(repo="key4hep/k4geo", number=n, url=f"https://gh/{n}",
                    title=f"PR {n}", patch="C" * 5000, scope_score=50.0)
        for n in range(1, 31)
    )
    prompt = build_user_prompt(_request(patch=subject, competitors=competitors))
    assert prompt.count("S") >= attr_mod._SUBJECT_DIFF_FLOOR
    assert "… (truncated)" in prompt


def test_competitors_are_capped_by_strength_not_by_walk_order():
    competitors = tuple(
        CompetingPR(repo="key4hep/k4geo", number=n, url=f"https://gh/{n}",
                    title=f"PR {n}", scope_score=float(n))
        for n in range(1, attr_mod.MAX_COMPETITORS + 6)
    )
    prompt = build_user_prompt(_request(competitors=competitors))
    strongest = competitors[-1]
    weakest = competitors[0]
    assert f"#{strongest.number} — PR {strongest.number}" in prompt
    assert f"#{weakest.number} — PR {weakest.number}" not in prompt


def test_a_very_wide_window_keeps_the_largest_movements():
    rows = tuple(
        _fact(f"r{n}", metric=f"m{n}", pct_change=n / 1000)
        for n in range(1, attr_mod._MAX_ATTRIBUTED_ROWS + 11)
    )
    prompt = build_user_prompt(_request(regressions=rows))
    assert f"[{rows[-1].id}]" in prompt   # the biggest step is scored
    assert f"[{rows[0].id}]" not in prompt  # the smallest is dropped, not zeroed
    assert prompt.count("] m") == attr_mod._MAX_ATTRIBUTED_ROWS


def test_the_system_prompt_names_the_cross_configuration_rules():
    system = attr_mod._SYSTEM_PROMPT
    assert "Reason across configurations" in system
    assert "without_" in system          # the detector-removal sweep's meaning
    assert "owner/repo#number" in system  # how an alternative may be named
    assert "Never write a URL" in system


# ── The answer ────────────────────────────────────────────────────────────────

def test_a_good_reply_scores_every_row_and_carries_the_narrative():
    request = _request(regressions=(_fact("r1"), _fact("r2", metric="sim_mem_mb")))
    attribution = _attributor([
        _completion(_reply("ALLEGRO moved, IDEA did not.", r1=92, r2=61))
    ]).attribute(request)
    assert attribution.likelihoods == {"r1": 92.0, "r2": 61.0}
    assert attribution.summary == "ALLEGRO moved, IDEA did not."
    assert attribution.top_score == 92.0


def test_an_invented_regression_is_dropped():
    request = _request(regressions=(_fact("r1"),))
    attribution = _attributor([
        _completion(_reply(r1=90, r9="80"))
    ]).attribute(request)
    assert set(attribution.likelihoods) == {"r1"}


def test_a_row_past_the_prompt_cap_cannot_be_scored_either():
    # Only-echo is enforced against what the prompt *offered*, not against every
    # regression in the window: a row the model was never shown can only have
    # been guessed at, and a guessed judgement of an unreviewed row is exactly
    # what must never reach someone else's pull request.
    rows = tuple(
        _fact(f"r{n}", metric=f"m{n}", pct_change=n / 1000)
        for n in range(1, attr_mod._MAX_ATTRIBUTED_ROWS + 3)
    )
    request = _request(regressions=rows)
    dropped = rows[0].id   # smallest movement — cut from the prompt
    offered = rows[-1].id
    attribution = _attributor([
        _completion(_reply(**{dropped: 95, offered: 40}))
    ]).attribute(request)
    assert set(attribution.likelihoods) == {offered}


def test_a_row_the_model_skipped_is_simply_absent_not_zero():
    # The caller falls back to that row's per-configuration score; publishing a
    # zero the model never committed to would invert its meaning.
    request = _request(regressions=(_fact("r1"), _fact("r2")))
    attribution = _attributor([_completion(_reply(r1=88))]).attribute(request)
    assert attribution.likelihoods == {"r1": 88.0}


def test_scores_are_clamped_and_unreadable_ones_reject_their_row():
    request = _request(regressions=(_fact("r1"), _fact("r2"), _fact("r3")))
    attribution = _attributor([
        _completion(_reply(r1=150, r2=-5, r3="very likely"))
    ]).attribute(request)
    assert attribution.likelihoods == {"r1": 100.0, "r2": 0.0}


def test_the_summary_is_flattened_and_capped():
    long_summary = "a" * (attr_mod._MAX_SUMMARY_CHARS + 500)
    attribution = _attributor([
        _completion(_reply("line one\nline  two", r1=90))
    ]).attribute(_request())
    assert attribution.summary == "line one line two"
    attribution = _attributor([
        _completion(_reply(long_summary, r1=90))
    ]).attribute(_request())
    assert len(attribution.summary) == attr_mod._MAX_SUMMARY_CHARS


# ── Every failure is the same decline ─────────────────────────────────────────

def test_scores_without_a_narrative_are_declined(caplog):
    # A table of numbers with no stated reasoning, posted into someone else's
    # repository, is worse than the per-configuration verdict it falls back to.
    assert _attributor([_completion(_reply("", r1=95))]).attribute(_request()) is None
    assert "gave no summary" in caplog.text


@pytest.mark.parametrize("content", [
    "I cannot help with that.",
    '{"summary": "words", "attributions": "not a list"}',
    '{"summary": "words"}',
    '{"summary": "words", "attributions": []}',
])
def test_an_unusable_reply_declines(content):
    assert _attributor([_completion(content)]).attribute(_request()) is None


def test_an_http_failure_declines_rather_than_raising(caplog):
    attributor = _attributor([_FakeResp({}, status=500) for _ in range(4)])
    assert attributor.attribute(_request()) is None
    assert "falling back to the per-configuration scores" in caplog.text


def test_a_timeout_declines():
    attributor = _attributor([requests.Timeout("slow") for _ in range(4)])
    assert attributor.attribute(_request()) is None


def test_no_regressions_short_circuits_without_calling_the_model():
    attributor = _attributor([])  # a post() would IndexError
    assert attributor.attribute(_request(regressions=())) is None
    assert attributor.client.session.calls == []


def test_the_output_budget_scales_with_the_number_of_rows():
    rows = tuple(_fact(f"r{n}") for n in range(1, 21))
    attributor = _attributor([_completion(_reply(r1=50))])
    attributor.attribute(_request(regressions=rows))
    budget = attributor.client.session.calls[0].json["max_tokens"]
    assert budget >= attr_mod._OUTPUT_TOKENS_BASE + attr_mod._OUTPUT_TOKENS_PER_ROW * 20


# ── attributor_from_env ───────────────────────────────────────────────────────

def test_reviewing_is_off_until_an_endpoint_and_model_are_configured(monkeypatch):
    for var in ("K4BENCH_LLM_URL", "K4BENCH_LLM_MODEL", "K4BENCH_LLM_SUMMARY_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert attributor_from_env() is None
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/v1")
    assert attributor_from_env() is None


def test_the_summary_model_overrides_the_ranker_model_for_this_pass(monkeypatch):
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "cheap/model")
    assert attributor_from_env().client.model == "cheap/model"
    monkeypatch.setenv("K4BENCH_LLM_SUMMARY_MODEL", "strong/model")
    assert attributor_from_env().client.model == "strong/model"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
