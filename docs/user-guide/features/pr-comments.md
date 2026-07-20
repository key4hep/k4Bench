# Pull-request comments

When the nightly benchmarks confirm a regression, the blame ranker already
answers "which pull request most likely caused this". That answer reaches the
e-group mail and the dashboard's Regressions tab — but not the person who wrote
the change. This feature closes that gap: k4Bench posts one comment on the pull
request it holds responsible, in the repository that PR lives in.

It is the **only** part of k4Bench that writes outside this repository, so
everything below is built around not abusing that.

Implemented by [`k4bench.blame.comment`](../../reference/api/blame/comment.md)
(who is commented on — pure, no network — and what the comment says),
[`k4bench.blame.attribute`](../../reference/api/blame/attribute.md) (the
cross-configuration review behind the claim) and
[`k4bench.blame.publish`](../../reference/api/blame/publish.md) (the write), driven
by `.github/scripts/blame_comment.py` in step 5c of the nightly
`regression-report` job.

## Two passes

The nightly ranker ([`k4bench.blame.rank`](../../reference/api/blame/rank.md))
asks *"which of these pull requests caused **this configuration's**
regressions?"* — once per `(detector, platform, sample)` run group. That is the
right question for the dashboard and the sidecar, where every regression row
wants a likelihood scoped to the run it was measured on.

It is the wrong question for a comment. The strongest evidence for or against a
claim about one change is *cross-configuration*: the same step hitting ALLEGRO
and not IDEA, under the same sample and the same platform, says something no
per-configuration call can see — because no per-configuration call is ever shown
the other configurations.

So each selected pull request gets a **second pass** that asks the transposed
question: *"which of this window's regressions did **this** pull request
cause?"*, once per `(pull request, change window)`. It is shown everything —
every confirmed regression across every detector, sample, platform and benchmark
configuration; the configurations that measured the same window and did *not*
confirm; the release's package diff; and every other pull request that landed in
the window, with its diff and the first pass's judgement of it. It returns a
likelihood per regression and the narrative the comment quotes.

Three rules bound it:

- **Only-echo.** A regression id the request did not contain is dropped, so a
  regression the model invented cannot reach a comment. A row it simply omitted
  keeps its per-configuration score — an unanswered row is not a zero.
- **Honest failure.** No model configured, an HTTP error, an unusable reply: the
  comment renders from the per-configuration scores, exactly as it did before
  this pass existed.
- **Narrowing only.** This pass never *causes* a comment. Selection happens on
  the first pass's scores, and a review that finds every regression unlikely can
  only withdraw the comment. The second opinion may acquit, not accuse.

Both passes use the same `K4BENCH_LLM_*` configuration and are off by default;
`K4BENCH_LLM_SUMMARY_MODEL` optionally points this pass alone at a stronger
model, since it runs at most `max_comments` times a night and feeds the only
outward-facing artifact.

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
| Not withdrawn | When the cross-configuration review ran and scored *every* regression in the window below `min_score`, the comment is dropped. See [Two passes](#two-passes). |

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
alone — that one is used here too, but only to read the candidates' diffs for the
review, so a write token is never spent on an ordinary public-repo read. Without
a write token — or with `K4BENCH_PR_COMMENT_DRY_RUN` set, or
`--dry-run` — the exact comment bodies are logged and nothing is written. That is
how a new repository is checked before it is added to the allowlist.

## What the comment says

One comment covers one `(pull request, change window)` pair, because the
reader's question — "did my change do this?" — is asked once per window.

It opens with the change window and the reviewer's short account of the pattern
it found, then **one table** of every regression in that window — metric,
detector, sample, benchmark configuration, how far it moved, and the attribution
likelihood — ordered by likelihood, the first five visible and the rest folded
into a disclosure. Which configurations moved (and, by their absence, which did
not) is the substance of the claim, so it is one list a reader can scan rather
than one configuration in full and the others in a footnote. A **Platform**
column appears only when the window spans more than one. Each detector cell links
to that regression's own dashboard view; below the table sit the other candidates
in the window with their likelihoods, and the package diff for the release.

Two rules run through the rendering:

- **The comment never claims more than the models did.** Without a
  cross-configuration review it quotes the per-configuration ranker and says "the
  most likely cause" only where this PR outranks every other candidate, saying in
  words when the preference was thin — or ran against this PR.
- **Nobody is notified who did not need to be.** Competing candidates are named
  as inert `owner/repo#123` text, never as references GitHub would turn into a
  cross-reference on their timeline. Externally-authored prose (PR titles, both
  models' text) is defanged first, so neither a title nor a narrative can smuggle
  a mention, an issue reference, an image, or a link into a comment k4Bench signs
  its own name to — including the `owner/repo#number` the reviewer is explicitly
  asked to name an alternative with.

## Lifecycle of a comment

Each comment carries a hidden marker keyed to its change window. The publisher
upserts on that marker:

- **Same window, next night** — the existing comment is edited in place, so a
  regression standing for a week is one comment, not seven.
- **Nothing changed** — no request at all. An edit re-surfaces the comment for
  everyone watching the PR, so it must mean something changed. The body contains
  nothing nightly-varying (no run URL, no report-night parameter), and "changed"
  is judged on a second hidden line: a digest of the *benchmark facts* — the
  window, the regressions and how far they moved, the field of candidates. The
  narrative and the likelihoods are model output that drifts between nights
  without anything having happened, so they are deliberately left out of it.
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

Every comment says an AI made the call, and ends with an invitation to
reply if the attribution looks wrong. That reply is the feedback channel — the
bot does not read it, but the people running k4Bench do.
