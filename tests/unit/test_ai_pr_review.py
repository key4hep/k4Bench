"""Review automation runs for maintainers, never for bot events."""

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
        "sender": {"type": "User", "login": "maintainer"},
        "pull_request": {
            "head": {"ref": "bump/IDEA_o1", "repo": {"full_name": REPOSITORY}},
            "draft": False,
        },
    }


def test_ready_detector_bump_requests_the_normal_reviews(event):
    assert review.requested_commands(event, REPOSITORY) == ["review", "describe", "improve"]


@pytest.mark.parametrize("case", ["bot", "bot_comment", "fork", "draft"])
def test_bot_or_ineligible_events_do_not_start_reviews(event, case):
    pr = event["pull_request"]
    if case == "bot":
        event["sender"] = {"type": "Bot", "login": "github-actions[bot]"}
    elif case == "bot_comment":
        event["sender"] = {"type": "Bot", "login": "github-actions[bot]"}
        event["comment"] = {"body": "/review", "author_association": "MEMBER"}
    elif case == "fork":
        pr["head"]["repo"]["full_name"] = "someone/k4Bench"
    else:
        pr["draft"] = True
    assert review.requested_commands(event, REPOSITORY) == []


def test_human_review_requests_still_work(event):
    event["pull_request"]["head"]["ref"] = "feature"
    assert review.requested_commands(event, REPOSITORY) == ["review", "describe", "improve"]
    event["comment"] = {"body": "/review", "author_association": "MEMBER"}
    assert review.requested_commands(event, REPOSITORY) == ["comment"]
    event["comment"]["author_association"] = "NONE"
    assert review.requested_commands(event, REPOSITORY) == []
