# OpenRouter Code Review Action (Review + Act)

Run an OpenRouter-routed model to review pull requests and, on demand, make
autonomous edits driven by `/codex` comments. Review path uses direct
OpenRouter chat completions with schema-enforced JSON output; Act path shells
out to [aider](https://aider.chat) so models and providers can be swapped
quickly when pricing or quality changes.

- **Review**: posts precise inline review comments and a PR-level summary. When
  there are no findings, only the summary is posted.
- **Act**: applies focused edits when trusted users comment `/codex`; commits
  and pushes to the PR branch via aider.

Ship as `dotcms/openrouter-code-review-action@v1`. Models are declared in an
in-repo config file (`.openrouter-review.yml`) so swapping providers when
pricing or quality changes is a one-line edit — no code or workflow change.

**Contents**

- [Quick Start (Review)](#quick-start-review)
- [Act on `/codex` Comments](#act-on-codex-comments)
- [Swapping the Model](#swapping-the-model-single-edit-config-file)
- [Inputs](#inputs) · [Outputs](#outputs) · [Required Permissions](#required-permissions)
- [Review Continuation](#review-continuation) · [Deduplication](#deduplication-on-repeated-runs)
- [Security & Permissions](#security--permissions) · [Troubleshooting](#troubleshooting)
- Migrating from `dotcms/codex-code-review-action`? See [MIGRATION.md](./MIGRATION.md).
- Contributing or swapping the review/act model? See
  [CONTRIBUTING.md](./CONTRIBUTING.md) and the parity gate documented in
  [eval/README.md](./eval/README.md#parity-gate-sub-ac-41--42--43).

## Quick Start (Review)

```yaml
name: PR Review
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
permissions:
  contents: read
  pull-requests: write
  issues: write
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: OpenRouter autonomous review
        uses: dotcms/openrouter-code-review-action@v1
        with:
          mode: review
          openrouter_api_key: ${{ secrets.OPENROUTER_API_KEY }}
```

## Act on `/codex` Comments

When a trusted user comments `/codex <instructions>` on a PR, the action checks
out the branch, runs aider against the configured `act.model`, and pushes the
result. Give aider a working environment so it can build/test before pushing.

```yaml
name: PR Act
on:
  issue_comment: { types: [created] }
  pull_request_review_comment: { types: [created] }
permissions:
  contents: write
  pull-requests: write
  issues: write
concurrency:
  group: openrouter-act-${{ github.event.issue.number || github.event.pull_request.number || github.ref }}
  cancel-in-progress: false
jobs:
  act:
    name: Act on /codex comments
    if: >-
      (
        (
          github.event_name == 'issue_comment' &&
          startsWith(github.event.comment.body, '/codex') &&
          github.event.issue.pull_request
        ) || (
          github.event_name == 'pull_request_review_comment' &&
          startsWith(github.event.comment.body, '/codex')
        )
      ) &&
      github.actor != 'dependabot[bot]'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          ref: ${{ github.event.pull_request.head.sha || format('refs/pull/{0}/head', github.event.issue.number) }}
          token: ${{ secrets.REPO_ACCESS_TOKEN }}

      # Give aider a working environment so it can build/test.
      # Replace with your own setup (install deps, run migrations, etc.).
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci

      - name: OpenRouter autonomous edits
        uses: dotcms/openrouter-code-review-action@v1
        with:
          mode: act
          openrouter_api_key: ${{ secrets.OPENROUTER_API_KEY }}
          allowed_commenter_associations: MEMBER,OWNER,COLLABORATOR
```

### `/codex` Commands

- **`/codex <instructions>`** — apply minimal diffs matching the instructions.
- Bare **`/codex`** is ignored; include explicit instructions after the command.
- **`/codex address comments`** (or natural variants like "please fix the
  review comments") — address unresolved review threads. Only unresolved
  threads are considered; resolved threads are ignored.

## Swapping the Model (Single-Edit Config File)

Model selection lives in an in-repo file — `.openrouter-review.yml` at the
consuming repo's root by default — so flipping review or act to a different
OpenRouter slug is a single edit. No Python change, no workflow change, no
action-input change required.

```yaml
# .openrouter-review.yml
review:
  model: anthropic/claude-opus-4.7    # change this line to swap review model
  reasoning_effort: medium            # optional — minimal | low | medium | high
  web_search_mode: live               # optional — disabled | cached | live

act:
  model: anthropic/claude-opus-4.7    # change this line to swap act model
  reasoning_effort: medium
```

The defaults are `anthropic/claude-opus-4.7` for both modes. To pin a different
provider — e.g. `openai/gpt-5.4`, `google/gemini-2.5-pro`,
`anthropic/claude-sonnet-4.5` — change the slug on the matching `model:` line
and commit. The next run uses the new model.

Override the file path with the `config_path` action input or
`OPENROUTER_REVIEW_CONFIG` env var (e.g. `ci/openrouter-models.yml`). Per-call
action inputs (`model:`, `reasoning_effort:`) still win over the file when
present, so a single workflow can opt out for a one-off run.

`web_search_mode: live` appends OpenRouter's `:online` variant to the slug at
call time; `cached` and `disabled` skip the live web fetch.

## Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `openrouter_api_key` | OpenRouter API key for routed model calls | *required* |
| `config_path` | Path to the in-repo model config file | `.openrouter-review.yml` |
| `mode` | `review` or `act` | `review` |
| **Model** | | |
| `model` | Per-call override; the in-repo config file is the normal swap point | `anthropic/claude-opus-4.7` |
| `reasoning_effort` | `minimal` / `low` / `medium` / `high` | `medium` |
| `web_search_mode` | `disabled` / `cached` / `live` | `live` |
| **Review-only** | | |
| `additional_prompt` | Extra reviewer instructions (verbatim) | |
| **Act-only** | | |
| `act_instructions` | Extra guidance appended to the edit prompt | |
| `allowed_commenter_associations` | Comma-separated GitHub `author_association` values allowed to trigger Act mode | `MEMBER,OWNER,COLLABORATOR` |
| `dry_run` | `0` or `1` — skip push | `0` |
| **Debug** | | |
| `debug_level` | `0` (off) / `1` (basic) / `2` (trace) | `1` |
| `stream_agent_messages` | `0` or `1` — stream model output to logs | `1` |
| **Advanced** | | |
| `extra_pip_args` | Additional pip flags (e.g., `--index-url`) | |

> `openai_api_key` is still accepted for legacy parity but the OpenRouter path
> is the supported route for all model calls. Migration drops direct vendor
> SDKs entirely — see `seed.yaml` for the constraint set.

## Outputs

This action exposes no formal `outputs:` block. Side effects are the unit of
work:

| Side effect | Where | Mode |
|-------------|-------|------|
| Inline review comments | PR diff | `review` |
| PR-level summary | PR issue comments (refreshed each run) | `review` |
| Review continuation cache | `actions/cache` keyed by repo, PR, model, SHA | `review` |
| Streamed model tokens | GitHub Actions step log (`stream_agent_messages: 1`) | `review`, `act` |
| Commits + push to PR branch | PR head branch | `act` |
| Reply comment on the triggering `/codex` thread | PR issue comments | `act` |
| Non-zero exit on hard failure | Job status | `review`, `act` |

If a downstream job needs the review status, gate on the job's `result`
(`needs.review.result == 'success'`).

## Required Permissions

The minimum permissions needed depend on mode:

| Mode | `contents` | `pull-requests` | `issues` |
|------|------------|-----------------|----------|
| `review` | `read` | `write` | `write` |
| `act` | `write` | `write` | `write` |

`contents: write` is required only by Act mode — aider needs to push commits
back to the PR branch. Review mode never modifies the repo. `issues: write`
covers both the PR-level summary (review) and the `/codex` reply (act);
GitHub treats PR comments as issue comments.

For PRs from forks, the default `GITHUB_TOKEN` cannot push to the fork's
branch. Run Act only on branches in the main repo, or supply a PAT
(`secrets.REPO_ACCESS_TOKEN` in the example above) with fork write access.

## What It Posts

- **Inline comments** anchored to exact diff lines. If a line isn't in the
  current diff, the finding is filtered or remapped before submission so
  GitHub does not return 422.
- **PR-level summary** as an issue comment on each run (refreshed on re-runs;
  prior summaries are deleted).
- **Multi-line suggestions** only when contiguous and short; otherwise a
  single-line comment.

## Review Continuation

On repeated `pull_request` review runs, the action continues the prior review
instead of restarting from scratch.

1. The PR summary stores the previously reviewed head SHA in hidden metadata.
2. Review mode caches isolated review state keyed by repository, PR number,
   model, and reviewed SHA.
3. On the next push, the action restores that cache, resumes the latest stored
   review thread, and scopes the prompt to the SHA-delta since the previously
   reviewed SHA.
4. If the prior SHA is no longer an ancestor (force-push, rebase), the cache
   is missing, or no thread can be restored, the action falls back to a fresh
   full review.

## Deduplication on Repeated Runs

When a prior review exists on the PR, reruns reuse only **unresolved
action-authored review threads** as context.

1. **Inline semantic dedup** — prior unresolved comments are passed to the
   model's structured-output turn so it can avoid reposting the same issue as
   a new finding.
2. **Re-adjudicated carry-forward** — the model separately marks which of
   those prior unresolved comments are still relevant now. Only those count
   toward the PR summary.
3. **Separated counts** — the summary reports new findings and still-relevant
   prior findings separately.

## Security & Permissions

- Act mode enforces a built-in `author_association` allowlist. Keep the
  workflow-level `if:` guard as defense in depth if you want early job
  skipping.
- Invalid `allowed_commenter_associations` values fail fast at startup so auth
  policy drift is visible immediately.
- For forks, the default `GITHUB_TOKEN` generally cannot push — run Act only
  on branches in the main repo, or use a PAT with fork access.
- Grant only what's needed: `contents: write` (push), `pull-requests: write`
  (reviews), `issues: write` (summary comments and Act replies).

## Troubleshooting

- **422 Unprocessable Entity**: target line not in PR head diff. The action's
  filter/remap pipeline should catch this before submission; if you still see
  422s, set `debug_level: 2` to log anchors.
- **Model errors**: ensure your OpenRouter key has access to the selected slug
  and that the slug exists in the OpenRouter catalog.
- Review uses built-in prompts (see `prompts/review.md`). Customize with
  `additional_prompt`.

## Local Development

```bash
uv sync                # install deps
make lint              # format, lint, type-check
GITHUB_TOKEN=… OPENROUTER_API_KEY=… PYTHONPATH=. python -m cli.main \
  --repo owner/repo --pr 123 --mode review --dry-run
```

## Release & Versioning

This repo uses [Release Please](https://github.com/googleapis/release-please)
in no-PR mode. Tags and GitHub Releases are created automatically on push to
`main`. After publish, the `v1` tag is updated to point to the latest release.

To force a specific version: Actions > "Release Please" > Run workflow >
provide `release_as` (e.g., `1.3.0`).

Pin major (`@v1`) for floating updates inside the major, or pin a tag
(`@v1.3.0`) for reproducible builds.
