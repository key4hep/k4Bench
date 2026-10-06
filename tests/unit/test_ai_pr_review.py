"""Review automation admits detector bumps without admitting bot reply loops."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".github/scripts/ai_pr_review.py"
spec = importlib.util.spec_from_file_location("ai_pr_review", SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
REPOSITORY = "key4hep/k4Bench"


@pytest.fixture
def event():
    return {
        "sender": {"type": "Bot", "login": "github-actions[bot]"},
        "pull_request": {
            "head": {"ref": "bump/IDEA_o1", "repo": {"full_name": REPOSITORY}},
            "draft": False,
        },
    }


def test_detector_bump_requests_the_normal_reviews(event):
    assert review.requested_commands(event, REPOSITORY) == ["review", "describe", "improve"]


@pytest.mark.parametrize("case", ["fork", "other_branch", "other_bot", "draft", "bot_comment"])
def test_unrelated_or_ineligible_bot_events_do_not_start_reviews(event, case):
    pr = event["pull_request"]
    if case == "fork":
        pr["head"]["repo"]["full_name"] = "someone/k4Bench"
    elif case == "other_branch":
        pr["head"]["ref"] = "dependabot/pip/update"
    elif case == "other_bot":
        event["sender"]["login"] = "unrelated[bot]"
    elif case == "draft":
        pr["draft"] = True
    else:
        event["comment"] = {"body": "/review", "author_association": "MEMBER"}
    assert review.requested_commands(event, REPOSITORY) == []


def test_human_review_requests_still_work(event):
    event["sender"] = {"type": "User", "login": "maintainer"}
    event["pull_request"]["head"]["ref"] = "feature"
    assert review.requested_commands(event, REPOSITORY) == ["review", "describe", "improve"]
    event["comment"] = {"body": "/review", "author_association": "MEMBER"}
    assert review.requested_commands(event, REPOSITORY) == ["comment"]
    event["comment"]["author_association"] = "NONE"
    assert review.requested_commands(event, REPOSITORY) == []


def test_bot_issue_comments_cannot_trigger_a_review(event):
    del event["pull_request"]
    event["issue"] = {
        "pull_request": {"url": "https://api.github.com/repos/key4hep/k4Bench/pulls/1"}
    }
    event["comment"] = {"body": "/review", "author_association": "MEMBER"}
    assert review.requested_commands(event, REPOSITORY) == []
