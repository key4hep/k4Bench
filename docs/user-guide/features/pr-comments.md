# Pull-request comments

When the nightly benchmarks confirm a regression, the blame ranker already
answers "which pull request most likely caused this". That answer reaches the
e-group mail and the dashboard's Regressions tab — but not the person who wrote
the change. This feature closes that gap: k4Bench posts one comment on the pull
request it holds responsible, in the repository that PR lives in.

It is the **only** part of k4Bench that writes outside this repository, so
everything below is built around not abusing that.

Implemented by [`k4bench.blame.comment`](../../reference/api/blame/comment.md)
(what is said, and to whom — pure, no network) and
[`k4bench.blame.publish`](../../reference/api/blame/publish.md) (the write), driven
by `.github/scripts/blame_comment.py` in step 5c of the nightly
`regression-report` job.

## When a comment is posted

Every one of these gates must pass. They are deliberately narrow — a comment in
someone else's repository, on the strength of a model's judgement, is worth
being wrong about far less often than it is worth being silent.

| Gate | Rule |
|:---|:---|
| Repository | The candidate's repo is listed in `.github/blame-comments.yml`. An empty list makes the bot inert. |
| Likelihood | The ranker's score is at or above `min_score` (default 80). |
| Merged | The PR is merged — an open PR cannot have shipped in a release. |
| Complete discovery | The blame entry's candidate search was complete. Naming one PR out of a knowingly partial set is the overclaim the ranker itself refuses to make. |
| Confirmed tonight | Selection is driven from the *report*'s confirmed regressions, so a comment can only describe a regression that is confirmed in tonight's report. |
| Not a storm | More than `max_comments` (default 10) comments in one night suppresses **all** of them: a night that loud is a bug, not a night. |

Most nights nothing is posted at all — most nights have no confirmed
regression, let alone a confidently attributed one.

## Configuration

`.github/blame-comments.yml`, reviewed by pull request rather than flipped in a
CI setting:

```yaml
min_score: 80
max_comments: 10
repos:
  - key4hep/k4geo
```

The file is strict: an unknown key, a wrong type, or an out-of-range value stops
the step instead of defaulting. Every field here decides whether — and where —
the bot writes, so a typo must never silently widen or narrow its reach.

Writing also needs `K4BENCH_PR_COMMENT_TOKEN`, a token carrying
`pull-requests: write` on the allowlisted repositories. It is **not** the
workflow's built-in `GITHUB_TOKEN`, which is read-only and scoped to k4Bench
alone. Without a token — or with `K4BENCH_PR_COMMENT_DRY_RUN` set, or
`--dry-run` — the exact comment bodies are logged and nothing is written. That is
how a new repository is checked before it is added to the allowlist.

## What the comment says

One comment covers one `(pull request, change window)` pair, because the
reader's question — "did my change do this?" — is asked once per window.

The ranker scores a candidate once per benchmark scope
(`detector, platform, sample`), so one comment can carry several independent
judgements. The strongest scope leads in full: its likelihood, the ranker's
one-line reason, what moved, the other candidates scored for that scope, and
dashboard links to check the claim. Every further scope of the same window
becomes one summary row keeping its own likelihood — and a **Ranking** column
saying whether this PR was top-ranked there or behind a stronger candidate,
since a comment fires on any score above the threshold.

Two rules run through the rendering:

- **The comment never claims more than the ranker did.** It says "the most
  likely cause" only where this PR outranks every other candidate *in that
  scope*, and says in words when the preference was thin — or ran against this
  PR.
- **Nobody is notified who did not need to be.** Competing candidates are named
  as inert `owner/repo#123` text, never as references GitHub would turn into a
  cross-reference on their timeline. Externally-authored prose (PR titles, the
  model's reason) is defanged first, so a title cannot smuggle a mention, an
  issue reference, an image, or a link into a comment k4Bench signs its own name
  to.

## Lifecycle of a comment

Each comment carries a hidden marker keyed to its change window. The publisher
upserts on that marker:

- **Same window, next night** — the existing comment is edited in place, so a
  regression standing for a week is one comment, not seven.
- **Nothing changed** — no request at all. An edit re-surfaces the comment for
  everyone watching the PR, so it must mean something changed. The body
  therefore contains nothing nightly-varying (no run URL, no report-night
  parameter).
- **A genuinely different window** — a separate comment, with its own marker.
- **The regression resolves, or the score drops below `min_score`** — the
  comment is **left exactly as it is**. It is not edited, retracted, or deleted.
  It records what the benchmarks saw at the time, which stays true even once the
  metric recovers; and silently rewriting a comment people have already replied
  to is worse than leaving a dated one standing. Follow-ups belong in the thread.

Safety rules on the write path:

- **Never post blind.** If the existing comments cannot be read in full, the PR
  is skipped — a duplicate comment is worse than a missing one.
- **Edit only our own comment.** A comment is the bot's own only when its first
  line is the marker *and* its author is the token's login. If that login cannot
  be established, the run fails closed and posts nothing.
- **One failure is one PR's failure.** A repo the token cannot write to does not
  silence the others. Only a rate limit stops the run.

The whole step is best-effort and isolated: it runs after `report.json` is
already uploaded, under a wall-clock timeout, inside a block that degrades any
failure to a log line. Commenting must never be able to affect the report, the
blame sidecar, or the e-group email.

## Replying to a comment

Every comment says an AI ranker made the call, and ends with an invitation to
reply if the attribution looks wrong. That reply is the feedback channel — the
bot does not read it, but the people running k4Bench do.
