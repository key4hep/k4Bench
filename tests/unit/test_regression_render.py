"""Unit tests for report rendering (:mod:`k4bench.regression.render`)."""

from __future__ import annotations

import json
import math

from k4bench.blame.models import BlameEntry, BlameReport, CandidatePR, RepoBlame
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.regression.render import from_json, to_html, to_json, to_markdown


def _verdict(**overrides) -> MetricVerdict:
    base = dict(
        detector="DET", platform="PLAT", sample="single_e", label="baseline",
        metric_family="time", metric="wall_time_s", sub_detector=None,
        run_id="2026-01-12", run_date="2026-01-12", value=120.0,
        baseline_median=100.0, baseline_mad=0.6, pct_change=0.20, z_score=33.0,
        severity=Severity.CONFIRMED, direction=Direction.UP,
        reason="+20.0% vs baseline median 100 (robust z=33.0)",
    )
    base.update(overrides)
    return MetricVerdict(**base)


def _full_report() -> NightlyReport:
    group = RunGroupReport(
        detector="DET", platform="PLAT", sample="single_e",
        k4h_release="key4hep-2026-01-01", run_date="2026-01-12", run_id="2026-01-12",
        verdicts=[
            _verdict(),
            _verdict(metric="mean_time_s", direction=Direction.DOWN, pct_change=-0.10),
            _verdict(metric="median_time_s", severity=Severity.WATCH),
            _verdict(metric="returncode", metric_family="status", value=1.0,
                     severity=Severity.FAILURE, direction=Direction.NONE,
                     reason="config exited with returncode 1"),
            _verdict(metric="peak_rss_mb", severity=Severity.OK,
                     direction=Direction.NONE, z_score=math.inf),
        ],
        job_failures=["config 'variant' produced no results tonight"],
        notes=["tonight's run failed the host reliability check"],
    )
    return NightlyReport(generated_at="2026-01-12T06:00:00+00:00", groups=[group])


def test_render_empty_report_does_not_crash():
    report = NightlyReport(generated_at="2026-01-12T06:00:00+00:00")
    md, html = to_markdown(report), to_html(report)
    assert "no data" in md and "no data" in html
    assert to_json(report)["summary"]["has_alertable"] is False


def test_markdown_orders_and_summarises():
    md = to_markdown(_full_report())
    # Heading is prefixed with the detector's overall status badge — this
    # group has a job failure, which outranks everything else.
    assert "## ❌ DET" in md
    assert "| wall_time_s | baseline | 120 | 100 | +20.0% | 🔴 Regression |" in md
    # Both directions get the same badge — no good/bad split by direction.
    assert md.count("🔴 Regression") == 2
    assert "❌ Failure" in md
    # WATCH/OK are summarised, never table rows.
    assert "⚠️ Watch" not in md.split("|---|")[-1].split("_")[0]
    assert "1 on watch" in md
    assert "config 'variant' produced no results tonight" in md


def test_html_is_self_contained_and_links():
    html = to_html(
        _full_report(),
        dashboard_url="https://dash.example/",
        actions_url="https://github.com/x/y/actions/runs/1",
    )
    assert "<style" not in html and "<script" not in html  # inline styles only
    # Top banner deep-links to the Overview tab — the cross-detector summary.
    assert "https://dash.example/?tab=Overview" in html
    # Per-group links carry the full triple — the Regressions tab is scoped by
    # the sidebar's (detector, platform, sample), the same way Run Trends is —
    # plus the release (stack) and the exact report night, so an emailed link
    # keeps pointing at its confirmation night after later reruns.
    assert (
        "tab=Regressions&detector=DET&platform=PLAT&sample=single_e"
        "&stack=key4hep-2026-01-01&report=2026-01-12" in html
    )
    assert html.count("🔴 Regression") == 2


def test_dashboard_links_merge_into_an_existing_query_string():
    # A dashboard_url that already carries its own query (e.g. behind a proxy
    # path) gets "tab"/"detector" merged in, not appended raw.
    html = to_html(_full_report(), dashboard_url="https://dash.example/app?env=prod")
    assert "env=prod" in html and "tab=Regressions" in html
    md = to_markdown(_full_report(), dashboard_url="https://dash.example/app?env=prod")
    assert "env=prod" in md and "tab=Regressions" in md


def test_markdown_links_to_dashboard_only_when_url_given():
    # Both links are scoped to the group's full (detector, platform, sample)
    # triple — the Regressions tab reads that scope from the sidebar too.
    md_with = to_markdown(_full_report(), dashboard_url="https://dash.example/")
    # The Regressions link is pinned to the release and its report night; Run
    # Trends stays scoped to the triple only (it has no report night).
    assert (
        "[↗ Regressions](https://dash.example/?tab=Regressions"
        "&detector=DET&platform=PLAT&sample=single_e"
        "&stack=key4hep-2026-01-01&report=2026-01-12)" in md_with
    )
    assert (
        "[↗ Run Trends](https://dash.example/?tab=Run+Trends"
        "&detector=DET&platform=PLAT&sample=single_e)" in md_with
    )
    md_without = to_markdown(_full_report())
    assert "↗ Regressions" not in md_without
    assert "↗ Run Trends" not in md_without


def test_regressions_link_omits_stack_for_a_release_less_group():
    # A stale/missing-run group carries no k4h_release — the link must still
    # point at the report night, just without a stack= to seed the sidebar.
    group = RunGroupReport(
        detector="DET", platform="PLAT", sample="single_e",
        k4h_release="", run_date="2026-01-12", run_id="2026-01-12",
        job_failures=["no run uploaded for 2026-01-12"],
    )
    report = NightlyReport(generated_at="2026-01-12T06:00:00+00:00", groups=[group])
    md = to_markdown(report, dashboard_url="https://dash.example/")
    assert (
        "[↗ Regressions](https://dash.example/?tab=Regressions"
        "&detector=DET&platform=PLAT&sample=single_e&report=2026-01-12)" in md
    )
    assert "stack=" not in md


def test_footer_present_in_both_formats():
    report = _full_report()
    md, html = to_markdown(report), to_html(report)
    for text in ("© 2026 CERN", "FCC project", "jbeirer@cern.ch"):
        assert text in md
        assert text in html


def test_group_title_prettifies_known_sample_and_platform_layouts():
    from k4bench.regression.render import _group_title

    group = RunGroupReport(
        detector="IDEA_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="p8_ee_Zbb_ecm91", k4h_release="key4hep-2026-01-01",
        run_date="2026-01-12", run_id="2026-01-12",
    )
    assert _group_title(group) == (
        "Pythia8: e⁺e⁻ → Z → bb (91 GeV) · AlmaLinux 9 · GCC 14.2.0 (optimized)"
    )

    group2 = RunGroupReport(
        detector="IDEA_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e-_10GeV", k4h_release="key4hep-2026-01-01",
        run_date="2026-01-12", run_id="2026-01-12",
    )
    assert _group_title(group2) == "Single e⁻ · 10GeV · AlmaLinux 9 · GCC 14.2.0 (optimized)"


def test_group_title_falls_back_to_raw_strings_for_unknown_layouts():
    from k4bench.regression.render import _group_title

    group = RunGroupReport(
        detector="DET", platform="some-weird-platform-string",
        sample="a_totally_unknown_sample_name", k4h_release="key4hep-2026-01-01",
        run_date="2026-01-12", run_id="2026-01-12",
    )
    assert _group_title(group) == "a_totally_unknown_sample_name · some-weird-platform-string"


def test_json_roundtrip_and_sanitization():
    report = _full_report()
    data = to_json(report)
    # Strict JSON: the infinite z-score must be serialized as null.
    text = json.dumps(data)  # would raise on raw inf with allow_nan=False semantics
    ok_verdict = [v for v in data["groups"][0]["verdicts"] if v["severity"] == "OK"]
    assert ok_verdict[0]["z_score"] is None
    assert data["summary"] == {
        "report_night": "2026-01-12",
        "n_detectors": 1,
        "n_regressions": 2,  # both directions confirmed — no good/bad split
        "n_watches": 1,
        "n_failures": 2,  # one config FAILURE + one job failure
        "has_alertable": True,
    }
    rebuilt = from_json(json.loads(text))
    assert rebuilt.report_night == report.report_night
    assert len(rebuilt.regressions) == 2
    assert all(v.severity is Severity.CONFIRMED for v in rebuilt.regressions)
    assert rebuilt.groups[0].job_failures == report.groups[0].job_failures
    assert rebuilt.has_alertable


def test_blame_window_survives_the_json_roundtrip():
    report = _full_report()
    report.groups[0].verdicts = [_verdict(
        onset_run_id="2026-01-11", onset_run_date="2026-01-09",
        last_accepted_run_id="2026-01-10", last_accepted_run_date="2026-01-05",
    )]
    rebuilt = from_json(json.loads(json.dumps(to_json(report))))
    v = rebuilt.regressions[0]
    assert (v.onset_run_id, v.onset_run_date) == ("2026-01-11", "2026-01-09")
    assert (v.last_accepted_run_id, v.last_accepted_run_date) == ("2026-01-10", "2026-01-05")


def test_from_json_reads_reports_written_before_the_window_existed():
    report = _full_report()
    data = to_json(report)
    for v in data["groups"][0]["verdicts"]:
        for key in ("onset_run_id", "onset_run_date",
                    "last_accepted_run_id", "last_accepted_run_date"):
            del v[key]
    v = from_json(data).regressions[0]
    assert (v.onset_run_id, v.last_accepted_run_id) == (None, None)


def test_from_json_ignores_fields_it_does_not_know():
    # The deployed dashboard is not necessarily built from the commit that
    # wrote the report, so a report gaining a field must not break it.
    data = to_json(_full_report())
    for v in data["groups"][0]["verdicts"]:
        v["some_field_from_a_later_release"] = "surprise"
    assert len(from_json(data).regressions) == 2


#: The blame window fields added to every verdict.
_WINDOW_FIELDS = {
    "onset_run_id", "onset_run_date", "last_accepted_run_id", "last_accepted_run_date",
}
#: The verdict schema a reader deployed before this feature knew about. The
#: compatibility contract is that the new fields are *purely additive* to this
#: set — anything else (a renamed or dropped field) breaks an old reader in a
#: way the new reader's unknown-key filter cannot rescue.
_PRE_WINDOW_FIELDS = {
    "detector", "platform", "sample", "label", "metric_family", "metric",
    "sub_detector", "run_id", "run_date", "value", "baseline_median",
    "baseline_mad", "pct_change", "z_score", "severity", "direction", "reason",
}


# ── Blame in the email (the "most likely cause" lead) ─────────────────────────

def _windowed_report() -> NightlyReport:
    """:func:`_full_report` whose confirmed wall_time_s verdict carries the
    blame window :func:`_blame_report` attributes — the join requires identity
    *and* window to match, so a stale sidecar can never attach."""
    report = _full_report()
    group = report.groups[0]
    group.verdicts[0] = _verdict(
        onset_run_id="2026-01-09", onset_run_date="2026-01-09",
        last_accepted_run_id="2026-01-05", last_accepted_run_date="2026-01-05",
    )
    return report


def _blame_report(*, reason="raises the tracker step count", score=72.0) -> BlameReport:
    """Blame joining the wall_time_s regression of :func:`_windowed_report`."""
    entry = BlameEntry(
        detector="DET", platform="PLAT", sample="single_e", label="baseline",
        metric="wall_time_s", sub_detector=None,
        base_release="2026-01-05", onset_release="2026-01-09",
        repos=(RepoBlame(
            package="k4geo", repo="key4hep/k4geo",
            base_commit="a" * 40, head_commit="c" * 40,
            compare_url="https://github.com/key4hep/k4geo/compare/aaa...ccc",
            status="changed",
            candidates=(
                CandidatePR(
                    repo="key4hep/k4geo", number=1234, title="Lower the step limit",
                    author="alice", url="https://github.com/key4hep/k4geo/pull/1234",
                    score=score, description=reason,
                ),
            ),
        ),),
    )
    return BlameReport(
        generated_at="2026-01-12T06:00:00", report_night="2026-01-12", entries=(entry,)
    )


def test_markdown_renders_the_suggested_cause_with_blame():
    md = to_markdown(_windowed_report(), blame=_blame_report())
    assert "Suggested causes" in md
    assert "not evidence" in md  # framed as a lead, never a verdict
    assert "`key4hep/k4geo#1234`" in md
    assert "(72%)" in md
    assert "raises the tracker step count" in md
    assert "https://github.com/key4hep/k4geo/pull/1234" in md


def test_html_renders_the_suggested_cause_and_escapes_the_reason():
    html = to_html(_windowed_report(), blame=_blame_report(reason="drops <b>steps</b> & inlines"))
    assert "Suggested causes" in html
    assert "key4hep/k4geo#1234" in html
    assert "(72%)" in html
    # LLM-derived text is escaped — it must not inject markup into the email.
    assert "drops &lt;b&gt;steps&lt;/b&gt; &amp; inlines" in html
    assert "<b>steps</b>" not in html


def test_email_unchanged_without_blame():
    # No sidecar → not a word about candidates, exactly as before the feature.
    assert "Suggested causes" not in to_markdown(_full_report())
    assert "Suggested causes" not in to_html(_full_report())


def test_unranked_blame_entry_renders_no_lead():
    # Candidates collected but not yet ranked (score 0, no description) → silence,
    # mirroring the dashboard's has_ranking gate.
    blame = _blame_report(reason="", score=0.0)
    assert "Suggested causes" not in to_markdown(_windowed_report(), blame=blame)
    assert "Suggested causes" not in to_html(_windowed_report(), blame=blame)


def test_to_json_stays_free_of_blame():
    # Blame is a separate sidecar; the report JSON the dashboard reads back must
    # not gain blame fields (to_json / from_json are untouched by this feature).
    text = json.dumps(to_json(_full_report())).lower()
    assert "likelihood" not in text
    assert "candidate" not in text


def test_new_report_is_additive_over_the_pre_window_schema():
    # The load-bearing compatibility direction: a report the *current* writer
    # emits must stay readable by a reader deployed before these fields existed
    # (once that reader also drops unknowns — the deployed reader must ship
    # first). That holds iff the window fields are the *only* additions, so a
    # verdict stripped of them reconstructs exactly the old schema.
    data = to_json(_full_report())
    for g in data["groups"]:
        for v in g["verdicts"]:
            assert v.keys() == _PRE_WINDOW_FIELDS | _WINDOW_FIELDS
            old_view = {k: val for k, val in v.items() if k in _PRE_WINDOW_FIELDS}
            MetricVerdict(**{
                **old_view,
                "severity": Severity(old_view["severity"]),
                "direction": Direction(old_view["direction"]),
            })
