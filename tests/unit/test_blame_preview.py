"""``.github/scripts/blame_preview.py`` — the guards around a write-capable tool.

The rendering path is production code tested elsewhere (``test_blame_comment``,
``test_blame_publish``); what belongs here is everything this script adds on top
of it, all of which is about not doing something surprising: it writes only when
asked twice, it refuses to preview a comment production would not have produced,
and it redirects a body without rewriting it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"


def _load_preview():
    """Import the CLI by path, with its own directory importable — the script
    imports its sibling ``blame_comment`` by module name, which only resolves
    when ``.github/scripts`` is on ``sys.path`` (running it directly puts it
    there; pytest does not)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "blame_preview", _SCRIPTS / "blame_preview.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preview():
    return _load_preview()


_PLAN_NIGHT = "2026-06-27"


@pytest.fixture
def wired(monkeypatch, preview):
    """The script with every I/O boundary replaced: EOS reads, the selection,
    the renderer and the upsert. Returns the recorded publish calls."""
    from k4bench.blame import comment as comment_mod
    from k4bench.blame import github as github_mod
    from k4bench.blame import publish as publish_mod
    from k4bench.blame.comment import PRComment
    from k4bench.blame.models import BlameReport
    from k4bench import remote as remote_mod

    blame = BlameReport(generated_at="x", report_night=_PLAN_NIGHT, entries=())
    monkeypatch.setattr(
        remote_mod, "fetch_report", lambda *_a: {"generated_at": "x", "groups": []}
    )
    monkeypatch.setattr(
        remote_mod, "fetch_blame", lambda *_a: json.loads(json.dumps(blame.to_json()))
    )

    class _Plan:
        repo, number = "key4hep/k4geo", 607

    monkeypatch.setattr(comment_mod, "select", lambda *_a, **_k: [_Plan()])
    rendered = PRComment(
        repo="key4hep/k4geo", number=607, marker="<!-- m -->",
        body="the rendered body naming key4hep/k4geo#607", score=91.0,
    )
    monkeypatch.setattr(
        comment_mod, "build_comments", lambda *_a, **_k: [rendered]
    )
    monkeypatch.setattr(github_mod, "GitHubClient", lambda **_k: object())

    calls: list[dict] = []

    def _publish(_client, comments, *, dry_run=False):
        calls.append({"comments": comments, "dry_run": dry_run})
        return publish_mod.PublishResult(planned=[c.target for c in comments])

    monkeypatch.setattr(publish_mod, "publish", _publish)
    return calls


@pytest.fixture
def llm(monkeypatch):
    """A configured reviewer, so the fail-closed guard is not what is under test."""
    from k4bench.blame import attribute as attribute_mod

    monkeypatch.setattr(attribute_mod, "attributor_from_env", lambda: object())


# ── Writing is opt-in twice ───────────────────────────────────────────────────

def test_a_token_in_the_environment_is_not_consent_to_post(
    preview, wired, llm, monkeypatch
):
    # The whole point of the flag: an exported write token must not turn a
    # preview run into an edit on someone else's pull request.
    monkeypatch.setenv("K4BENCH_PR_COMMENT_TOKEN", "ghp_real_write_token")
    assert preview.main(["--night", _PLAN_NIGHT]) == 0
    assert [call["dry_run"] for call in wired] == [True]


def test_posting_needs_an_explicit_target(preview):
    with pytest.raises(SystemExit) as exc:
        preview.main(["--post", "--token", "t"])
    assert exc.value.code == 2


def test_posting_needs_a_write_token(preview, monkeypatch):
    monkeypatch.delenv("K4BENCH_PR_COMMENT_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        preview.main(["--post", "--post-to", "key4hep/k4Bench#106"])
    assert exc.value.code == 2


def test_post_writes_and_redirects_the_body_untouched(preview, wired, llm):
    # The redirect moves only the write target: the body still names --only, so
    # the reviewer reads exactly what would have landed there.
    assert preview.main([
        "--post", "--post-to", "key4hep/k4Bench#106", "--token", "t",
    ]) == 0
    (call,) = wired
    assert call["dry_run"] is False
    (comment,) = call["comments"]
    assert (comment.repo, comment.number) == ("key4hep/k4Bench", 106)
    assert comment.body == "the rendered body naming key4hep/k4geo#607"
    # The marker is the window's, not the target's, so a re-run edits this same
    # comment rather than stacking a second one.
    assert comment.marker == "<!-- m -->"


# ── A preview is a preview of *production* ────────────────────────────────────

def test_no_llm_configured_is_an_error_not_a_different_comment(
    preview, wired, monkeypatch, capsys
):
    from k4bench.blame import attribute as attribute_mod

    monkeypatch.setattr(attribute_mod, "attributor_from_env", lambda: None)
    assert preview.main([]) == 1
    assert not wired
    assert "--ranker-only" in capsys.readouterr().err


def test_ranker_only_asks_for_the_fallback_on_purpose(preview, wired, monkeypatch):
    from k4bench.blame import attribute as attribute_mod

    monkeypatch.setattr(
        attribute_mod, "attributor_from_env",
        lambda: pytest.fail("--ranker-only must not consult the environment"),
    )
    assert preview.main(["--ranker-only"]) == 0
    assert [call["dry_run"] for call in wired] == [True]


# ── References the tool must refuse rather than guess through ─────────────────

@pytest.mark.parametrize(
    "ref",
    ["k4geo#607", "key4hep/k4geo", "key4hep/k4geo#seven", "/k4geo#607", "a/b/c#1"],
)
def test_a_malformed_pull_request_reference_is_refused(preview, ref):
    with pytest.raises(SystemExit) as exc:
        preview.main(["--only", ref])
    assert exc.value.code == 2


def test_a_repo_outside_the_allowlist_renders_nothing(preview, wired, llm, capsys):
    assert preview.main(["--only", "someone/elses-repo#1"]) == 1
    assert not wired
    assert "allowlist" in capsys.readouterr().err
