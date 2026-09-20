#!/usr/bin/env bash
# Run one model; the workflow selects the model, key, and fallback order.
set -euo pipefail

trap 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Bound streaming and retries, leaving time for the fallback model.
timeout --foreground --kill-after=10s 25m \
  docker run --rm --init --name "$CONTAINER_NAME" \
  --security-opt label=disable \
  --volume "$GITHUB_WORKSPACE/.github/scripts/ai_pr_review.py":/tmp/ai_pr_review.py:ro \
  --volume "$GITHUB_EVENT_PATH":/tmp/github-event.json:ro \
  --entrypoint python \
  -e GITHUB_EVENT_PATH=/tmp/github-event.json \
  -e GITHUB_EVENT_NAME -e GITHUB_TOKEN -e OPENAI__KEY -e PR_AGENT_COMMAND \
  -e GITHUB_ACTIONS=true -e CI=true -e LOG_SANE=1 \
  -e OPENAI__API_BASE=https://aigw.cern.ch/v1 \
  -e config.model="$MODEL" -e config.propagate_tool_errors=true \
  "$CONTAINER_IMAGE" /tmp/ai_pr_review.py
