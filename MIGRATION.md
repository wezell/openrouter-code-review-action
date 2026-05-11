# Migration Guide — `dotcms/codex-code-review-action` → `dotcms/openrouter-code-review-action@v1`

> This is a **hard fork**, not a drop-in upgrade. Action coordinates, inputs,
> outputs, model wire protocol, and the act-mode agent have all changed.
> There is **no backwards-compatible mode** — pinning the new action without
> updating workflow YAML will fail validation.

For a release-style summary see [CHANGELOG.md](./CHANGELOG.md). This document is
the operator-facing migration reference: what to change in workflow YAML, what
each Codex-era input/output maps to (or doesn't), and the order of operations
to flip a repository over safely.

---

## TL;DR

1. Repoint `uses:` from `dotcms/codex-code-review-action@<sha|vX>` to
   `dotcms/openrouter-code-review-action@v1`.
2. Add `OPENROUTER_API_KEY` to repo / org secrets and pass it as
   `openrouter_api_key`.
3. Drop `openai_api_key` (or leave it; it is accepted-but-ignored and will be
   removed in a future major).
4. Commit `.openrouter-review.yml` at the repo root declaring `review.model`
   and `act.model`.
5. Re-audit job `permissions:` against the matrix below.
6. Discard any cached Codex artifacts — review continuation now keys on
   `(repo, PR, model, head_sha)` and will repopulate on the next run.

---

## Action coordinates

| Aspect            | Codex action                                  | OpenRouter action                                  |
| ----------------- | --------------------------------------------- | -------------------------------------------------- |
| `uses:`           | `dotcms/codex-code-review-action@<sha|vX>`    | `dotcms/openrouter-code-review-action@v1`          |
| Floating tag      | `@v1` on the Codex repo                       | `@v1` on the OpenRouter repo (managed by Release Please) |
| Required secret   | `OPENAI_API_KEY` (Codex SDK)                  | `OPENROUTER_API_KEY`                               |
| Wire protocol     | Codex SDK                                     | OpenRouter chat completions API (HTTP/SSE)         |
| Default model     | Codex-managed default                         | `anthropic/claude-opus-4.7`                        |
| Act-mode agent    | Custom Codex agentic loop                     | `aider` subprocess                                 |

The Codex action remains at its previous coordinates for unmigrated consumers.
Nothing in this repo is wire-compatible with it.

---

## Input mapping

The supported input set is **enumerated** in [`action.yml`](./action.yml).
Inputs from the Codex action that are not listed below are **not honored** —
they will be silently ignored by `actions/checkout`-style input passing, but
the behavior they previously controlled is either gone, renamed, or moved
into `.openrouter-review.yml`.

| Codex action input                | OpenRouter action equivalent                         | Notes |
| --------------------------------- | ---------------------------------------------------- | ----- |
| `openai_api_key`                  | **Deprecated.** Set `openrouter_api_key` instead.    | Still accepted as an input for parity but ignored by the OpenRouter path. Will be dropped in a future major. |
| _(implicit Codex SDK auth)_       | `openrouter_api_key` (**required**)                  | Sourced from `secrets.OPENROUTER_API_KEY`. The action errors out at startup if missing. |
| `mode`                            | `mode` (`review` \| `act`)                           | Same name, same values. |
| `model`                           | `model` (full OpenRouter slug)                       | Codex-style names (e.g. `gpt-5`) **do not** work; use vendor-prefixed slugs like `anthropic/claude-opus-4.7`, `openai/gpt-5.4`, `google/gemini-2.5-pro`. Normally leave this unset and edit `.openrouter-review.yml`. |
| `reasoning_effort`                | `reasoning_effort` (`minimal` \| `low` \| `medium` \| `high`) | Same name. Mapped to OpenRouter's reasoning controls. Default `medium`. |
| `web_search` / Codex web toggle   | `web_search_mode` (`disabled` \| `cached` \| `live`) | Renamed and re-shaped. `live` is the default and maps to OpenRouter's `:online` slug variant. |
| `dry_run`                         | `dry_run` (`0` \| `1`)                               | Same. |
| `debug_level`                     | `debug_level` (`0` \| `1` \| `2`)                    | Same. |
| `stream_agent_messages`           | `stream_agent_messages` (`0` \| `1`)                 | Same name; now controls OpenRouter SSE forwarding to the step log. |
| `act_instructions`                | `act_instructions`                                   | Same. Passed through to aider. |
| `allowed_commenter_associations`  | `allowed_commenter_associations`                     | Same. Default `MEMBER,OWNER,COLLABORATOR`. Now enforced **in-process**, not just in the workflow `if:` guard. Invalid values fail fast at startup. |
| `additional_prompt`               | `additional_prompt`                                  | Same. Appended verbatim to the review prompt. |
| `extra_pip_args`                  | `extra_pip_args`                                     | Same. |
| Codex model fallback chain inputs | **Removed.**                                         | One model per call. Edit `.openrouter-review.yml` to swap. |
| Codex-era cache-key inputs        | **Removed.**                                         | Continuation key is now `(repo, PR, model, head_sha)` and is internal. |
| _(new)_                           | `config_path`                                        | Path to the in-repo model config. Defaults to `.openrouter-review.yml`. Override via input or `OPENROUTER_REVIEW_CONFIG` env var. |

> **Rule of thumb:** if a Codex-era input touched _which model runs_ or _how
> the agent loops_, it is gone — those decisions now live in
> `.openrouter-review.yml` and aider, respectively.

---

## Output / side-effect mapping

The action is composite and does not declare typed `outputs:` in `action.yml`.
Its observable side-effects map as follows:

| Codex action behavior                           | OpenRouter action behavior                                              |
| ----------------------------------------------- | ----------------------------------------------------------------------- |
| Inline review comments via Codex JSON           | Inline review comments via **schema-enforced JSON**, anchored to validated diff lines, batched into one `POST /pulls/{n}/reviews`. Falls back to per-comment posting if GitHub returns 422. |
| PR summary comment with Codex metadata          | PR summary comment embedding hidden `<!-- openrouter-review:meta -->` block with the previously reviewed head SHA and (repo, PR, model) tuple. Used to drive review continuation. |
| Codex-managed conversation cache                | `actions/cache` entry keyed on `(repo, PR, model, head_sha)`. Force-pushes that break ancestry, missing cache, or unrestorable threads → fresh full review (no silent partial state). |
| Token / usage counters in Codex shape           | Token / usage counters in OpenRouter shape, surfaced in the step summary. |
| Custom agentic edit/test/commit cycle (act)     | aider drives the edit/test/commit cycle; commits are pushed to the **PR head branch only**. |
| Exit code                                       | Same shape: `0` on success, non-zero on hard failures (auth, schema, GitHub API). Soft failures (e.g. 422 on a review batch) degrade to per-comment fallback before failing. |

Anything downstream that grepped Codex-shaped log lines or parsed the prior
summary block needs to be updated to the new metadata block name.

---

## Required permissions

Re-audit the consuming workflow's `permissions:` against the mode you run.

| Mode    | Required `permissions:`                                        | Fork PR caveat |
| ------- | -------------------------------------------------------------- | -------------- |
| review  | `pull-requests: write`, `contents: read`                       | Default `GITHUB_TOKEN` works for same-repo PRs. Fork PRs need explicit secrets per GitHub policy. |
| act     | `pull-requests: write`, `contents: write`                      | Default `GITHUB_TOKEN` **cannot push to fork branches.** Pass an explicit PAT (e.g. `secrets.REPO_ACCESS_TOKEN`) and run only on same-repo branches. |

The Codex action's permission table is a strict subset of this — anything that
worked there continues to satisfy review mode here.

---

## Step-by-step upgrade

The following is the recommended order; each step is independently revertable
until step 4.

### 1. Add the new secret

In repo (or org) settings, add `OPENROUTER_API_KEY`. You can keep
`OPENAI_API_KEY` set during the cutover; it is harmless.

### 2. Add `.openrouter-review.yml` at the repo root

```yaml
# .openrouter-review.yml
review:
  model: anthropic/claude-opus-4.7
  reasoning_effort: medium
  web_search_mode: live

act:
  model: anthropic/claude-opus-4.7
  reasoning_effort: medium
```

Swapping providers later (e.g. `openai/gpt-5.4`, `google/gemini-2.5-pro`) is
a one-line edit to this file. No code, workflow, or action input change is
required.

### 3. Update workflow YAML

Before:

```yaml
- uses: dotcms/codex-code-review-action@v1
  with:
    openai_api_key: ${{ secrets.OPENAI_API_KEY }}
    mode: review
    reasoning_effort: medium
```

After:

```yaml
- uses: dotcms/openrouter-code-review-action@v1
  with:
    openrouter_api_key: ${{ secrets.OPENROUTER_API_KEY }}
    mode: review
    reasoning_effort: medium
    # model / web_search_mode normally come from .openrouter-review.yml
```

For act mode, additionally set `permissions: contents: write` on the job and
gate on `author_association` in the workflow `if:` (the action also enforces
this in-process).

### 4. Cut a test PR

Open a small PR against a branch you control. Confirm:

- The "OpenRouter review" run posts inline comments and a summary that
  contains the `<!-- openrouter-review:meta -->` block.
- A second push to the same PR triggers a continuation run that references
  the prior head SHA.
- A trusted-author comment of `/codex address comments` (or `/codex
  <instructions>`) triggers an act-mode run that pushes a commit on the PR
  head branch.

### 5. Remove `openai_api_key`

Once steps 1–4 are green for at least one full review/act cycle, drop the
`openai_api_key` input from your workflow YAML. The deprecation warning will
go away and you'll be ready for the future major that removes it entirely.

### 6. Discard Codex-era caches

There is nothing to actively delete — Codex-era `actions/cache` entries will
expire naturally and the new action does not read them. Just don't carry
forward any `restore-keys:` patterns that referenced Codex artifacts.

---

## Things that intentionally do **not** carry over

These are removed by design. If you depended on any of them, the migration
requires a workflow- or repo-level change, not an action-level one.

- **Direct vendor SDK fallbacks.** Calls do not fall through to OpenAI,
  Anthropic, Google, or any other vendor SDK if OpenRouter is unreachable.
  The run fails loudly instead of silently switching providers.
- **Mid-run model fallback chain.** A single model handles a given call. If
  the configured model is unhealthy, edit `.openrouter-review.yml` and rerun.
- **Codex-managed conversation state.** Continuation is driven by the hidden
  metadata block on the PR summary plus an `actions/cache` entry; there is no
  Codex-side state.
- **Codex-shaped tool transcripts.** Inline comments and summary text are now
  produced from a schema-enforced JSON turn; any consumer parsing the prior
  Codex transcript shape must be retargeted at the new schema or the rendered
  summary.

---

## Rollback

If you need to roll back to the Codex action mid-cutover:

1. Revert the workflow YAML change (`uses:` line, input renames).
2. Re-add the `openai_api_key` input if you removed it.
3. Leave `.openrouter-review.yml` in the repo — the Codex action ignores it.
4. The new action's `actions/cache` entries will expire; no cleanup required.

There is no shared state between the two actions, so rollback is a pure YAML
revert. Just don't run them concurrently against the same PR — they will
fight over the summary comment.
