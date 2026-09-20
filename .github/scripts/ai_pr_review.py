"""Check review requests on the runner; execute them inside the PR-Agent container."""

import asyncio
import json
import os
from pathlib import Path
import sys

CONTRIBUTORS = {"OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"}
AUTOMATIC_COMMANDS = ["review", "describe", "improve"]


def is_review_request(event, repository):
    if event.get("sender", {}).get("type") == "Bot":
        return False

    pr = event.get("pull_request") or event.get("issue", {}).get("pull_request")
    if not pr:
        return False

    comment = event.get("comment")
    if comment:
        is_command = comment.get("body", "").startswith("/")
        is_contributor = comment.get("author_association") in CONTRIBUTORS
        return is_command and is_contributor

    head_repository = pr.get("head", {}).get("repo") or {}
    is_same_repository = head_repository.get("full_name") == repository
    return is_same_repository and not pr.get("draft", False)


def requested_commands(event, repository):
    if not is_review_request(event, repository):
        return []
    # Keep the comment body in the event file, not in matrix values or container names.
    if event.get("comment"):
        return ["comment"]
    return AUTOMATIC_COMMANDS


def run_review(event):
    # These dependencies are provided by the pinned container image.
    from pr_agent.agent.pr_agent import PRAgent
    from pr_agent.config_loader import get_settings
    from pr_agent.servers.github_app import handle_line_comments

    comment = event.get("comment")
    command = comment["body"] if comment else f"/{os.environ['PR_AGENT_COMMAND']}"
    if os.environ["GITHUB_EVENT_NAME"] == "pull_request_review_comment" and "/ask" in command:
        command = handle_line_comments(event, command)
    pr = event.get("pull_request") or event["issue"]["pull_request"]
    get_settings().set("github.user_token", os.environ["GITHUB_TOKEN"])
    get_settings().set("config.is_auto_command", not comment)
    # The pinned action runner ignores False for push/comment commands.
    return int(not asyncio.run(PRAgent().handle_request(pr["url"], command)))


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if "--check" in sys.argv:
        commands = requested_commands(event, os.environ["GITHUB_REPOSITORY"])
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"commands={json.dumps(commands)}\n")
        return 0
    return run_review(event)


if __name__ == "__main__":
    sys.exit(main())
