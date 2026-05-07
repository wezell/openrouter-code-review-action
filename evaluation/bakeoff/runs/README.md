# Bake-off run artifacts

This directory holds the per-PR raw-findings artifacts produced by both
sides of the Codex → OpenRouter parity bake-off, one subdirectory per
sample PR (`pr-<number>/`):

```
evaluation/bakeoff/runs/
├── pr-35567/
│   ├── codex.json         # Codex baseline run — populated by Sub-AC 2.1 path
│   └── openrouter.json    # OpenRouter run — populated by this Sub-AC (2.2)
├── pr-35509/
│   ├── codex.json
│   └── openrouter.json
└── ...
```

The artifact schema, sample PR registry, labeling rubric, FP rules, and
overlap math all live in [`eval/baseline-methodology.md`](../../../eval/baseline-methodology.md)
§4. This README documents only how the OpenRouter side gets *captured*.

## What this Sub-AC delivers

[`evaluation/bakeoff/run_openrouter.py`](../run_openrouter.py) is the
runner. It:

1. Reads [`evaluation/bakeoff/sample-prs.yml`](../sample-prs.yml) for
   the curated PR sample (Sub-AC 1.1).
2. Resolves the review model + reasoning effort + web-search mode by
   loading [`.openrouter-review.yml`](../../../.openrouter-review.yml)
   through `cli.core.model_config.load_model_config` — this is the same
   resolver the action itself uses, so a model swap is honored without
   touching the runner.
3. For each PR:
   - Fetches the unified diff at the pinned `head_sha` via the `gh`
     CLI (falling back to the GitHub REST API when `gh` is absent).
   - POSTs to OpenRouter `/api/v1/chat/completions` with the strict
     `response_format` from
     `cli.core.models.openrouter_review_response_format()` so the
     provider rejects any output that isn't a valid `ReviewRunResult`.
   - Validates the returned JSON locally through
     `validate_review_payload` and persists the result.
4. Writes one `pr-<number>/openrouter.json` artifact in the §4.2 schema.

The runner is intentionally self-contained: it talks to OpenRouter
directly (via `urllib`) rather than going through `cli/workflows/
review_workflow.py`. The workflow is mid-port from Codex to OpenRouter
(AC 1's body), and we don't want bake-off-evidence collection to be
gated on that port landing.

## Run modes

| Mode | What it does | When to use |
|------|--------------|-------------|
| `scaffold` *(default)* | Writes `status=unfilled` placeholders for every sample PR with the model + reasoning_effort + web_search_mode that the live run *would* use. Designed to be checked in. | Whenever the sample, model config, or runner contract changes. |
| `dry` | Fetches the diff via `gh pr diff` *and* builds the full OpenRouter request payload (model, messages, response_format, reasoning), then persists `status=skipped_dry_run` with the resolved payload-shape telemetry (diff_chars, payload_messages, response_format_kind, online-suffix model_resolved, reasoning_effort_resolved). The POST is *not* issued. Captures end-to-end evidence that the runner ran against the curated sample without spending tokens. | Demonstrating Sub-AC 1.2 end-to-end against the curated sample without a key. Verifying that the resolved payload matches what `--mode live` would send. |
| `replay` | Loads a previously-captured raw `/api/v1/chat/completions` response from `evaluation/bakeoff/replay/<pr_id>.json`, runs it through the same `_extract_review_payload` + `validate_review_payload` path as `--mode live`, and persists `status=replayed` with the validated `review_run_result`. Slots without a recorded response stay `status=unfilled`. | Reproducing the bake-off offline once an operator has captured live responses. Catching parser/schema regressions against real recorded data. Cheap re-runs after a parser change without re-billing. |
| `live` | Fetches the diff, calls OpenRouter, validates the schema, persists `status=completed` (or `status=error` per PR if a single call fails — one bad PR doesn't sink the rest of the run). | Producing real bake-off evidence. |

## Required environment for `--mode live`

| Variable / tool | Why |
|-----------------|-----|
| `OPENROUTER_API_KEY` | Auth for `/api/v1/chat/completions`. |
| `gh` CLI authenticated **or** `GITHUB_TOKEN` set | Used to fetch the unified diff at the pinned `head_sha`. The runner prefers `gh` because it transparently follows the diff redirect; the REST fallback uses `Accept: application/vnd.github.v3.diff`. |

## Usage

```bash
# Default: refresh all 10 placeholders (no network, no tokens).
.venv/bin/python evaluation/bakeoff/run_openrouter.py

# Dry-run the whole sample: fetch diffs via gh, build payloads, no POST.
# Captures Sub-AC 1.2 evidence that the harness ran end-to-end against
# the curated sample without spending tokens.
.venv/bin/python evaluation/bakeoff/run_openrouter.py --mode dry

# Dry-run a single PR (no OpenRouter call, but does fetch the diff).
.venv/bin/python evaluation/bakeoff/run_openrouter.py --mode dry --only pr-35567

# Replay-run the sample from committed live captures.
# Slots without a recorded response stay status=unfilled.
.venv/bin/python evaluation/bakeoff/run_openrouter.py --mode replay

# Live-run the whole sample.
OPENROUTER_API_KEY=sk-or-... \
  .venv/bin/python evaluation/bakeoff/run_openrouter.py --mode live

# Live-run a single PR (useful for re-trying a single failed slot).
OPENROUTER_API_KEY=sk-or-... \
  .venv/bin/python evaluation/bakeoff/run_openrouter.py --mode live --only pr-35458

# Point at a different in-repo model config (e.g. an experiment file).
.venv/bin/python evaluation/bakeoff/run_openrouter.py \
  --mode live --config-path .openrouter-review.experiment.yml
```

### Replay capture-once-then-replay-anywhere

`--mode replay` is the offline-friendly counterpart to `--mode live`. The
operator who has an `OPENROUTER_API_KEY` runs `--mode live` once,
captures the raw `/api/v1/chat/completions` response, and commits it to:

```
evaluation/bakeoff/replay/<pr_id>.json
```

Anyone re-running the bake-off (CI, a reviewer on a fresh laptop) can
then run `--mode replay` to re-derive findings deterministically without
re-billing. The replay path runs the recorded response through the same
`_extract_review_payload` + `validate_review_payload` machinery as
`--mode live`, so a parser regression caught by replay is a parser
regression that would have failed against real captured data — not a
synthesized fixture. See [`../replay/README.md`](../replay/README.md)
for the capture format.

## Per-PR artifact schema

```jsonc
{
  "pr_id": "pr-35567",
  "run": "openrouter",
  "status": "completed | replayed | skipped_dry_run | unfilled | error",
  "head_sha": "<merge_commit SHA — the exact diff that was reviewed>",
  "repo": "dotCMS/core",
  "pr_number": 35567,
  "model": "anthropic/claude-opus-4.7",
  "reasoning_effort": "minimal | low | medium | high",
  "web_search_mode": "disabled | cached | live",
  "prior_review_state": null,            // null on the initial fresh-review pass
  "review_run_result": { ... } | null,   // ReviewRunResult.as_dict(); null when not run
  "posted": null,                        // bake-off runs do not post to GitHub
  "labels": [],                          // filled by the labeling sub-AC
  "solo_labeled": null,                  // filled by the labeling sub-AC
  "notes": "<runner-supplied free-form note: skip reason, error message, etc.>",
  "telemetry": {
    "diff_chars": 1773,
    "usage": { "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ... },
    "model_resolved": "anthropic/claude-opus-4.7"   // includes :online suffix when web_search_mode=live
  },
  "captured_at": "2026-05-07T01:59:41Z",
  "runner": "evaluation/bakeoff/run_openrouter.py",
  "runner_version": "1.0.0"
}
```

`labels[]` and `solo_labeled` are intentionally left empty here — they
are filled during the labeling step (methodology §2), not during
capture.

## Downstream: comparison reference set

The **labeling and scoring sub-ACs do not read this directory
directly.** They read the per-PR comparison records under
[`../comparison/`](../comparison/README.md), which the Sub-AC 1.1.3
organizer (`evaluation/bakeoff/build_comparison_set.py`) builds by
joining each `runs/<pr_id>/openrouter.json` with its paired Codex
artifact (`eval/runs/<eval_id>/codex.json`) and normalizing the
findings into a flat schema.

Whenever this directory's contents change — a new live capture, a
re-run after a model bump — re-run:

```bash
.venv/bin/python -m evaluation.bakeoff.build_comparison_set
```

to refresh the comparison set. CI runs the same module with
`--check` to flag drift.

## Status of this Sub-AC's deliverable

The 10 sample-PR slots have been swept end-to-end with `--mode dry`,
which fetches each PR's diff via `gh pr diff` at the pinned
`head_sha`, builds the OpenRouter request payload (model, messages,
response_format, reasoning), and persists the resolved payload-shape
telemetry without spending tokens. Each artifact records the diff
size, the resolved model (`anthropic/claude-opus-4.7:online` because
`web_search_mode=live`), the response-format kind (`json_schema`),
and the reasoning effort actually wired into the request.

| PR slot | Status | Notes |
|---------|--------|-------|
| pr-35567 | `skipped_dry_run` | Dry sweep — diff fetched, payload built, no POST. |
| pr-35509 | `skipped_dry_run` | Dry sweep. |
| pr-35498 | `skipped_dry_run` | Dry sweep. |
| pr-35458 | `skipped_dry_run` | Dry sweep. |
| pr-35491 | `skipped_dry_run` | Dry sweep. |
| pr-35469 | `skipped_dry_run` | Dry sweep. |
| pr-35518 | `skipped_dry_run` | Dry sweep. |
| pr-35449 | `skipped_dry_run` | Dry sweep. |
| pr-35465 | `skipped_dry_run` | Dry sweep. |
| pr-35522 | `skipped_dry_run` | Dry sweep. |

To populate `review_run_result` with real findings, re-run the script
with `--mode live` in an environment that has `OPENROUTER_API_KEY` set,
or capture once-and-commit a `evaluation/bakeoff/replay/<pr_id>.json`
file and re-run with `--mode replay`.

The reason `skipped_dry_run` (or `unfilled` scaffold) — not synthesized
findings — is committed here is that the two parity ACs require
**paired** OpenRouter and Codex captures on the same `head_sha`, and
the methodology (`eval/baseline-methodology.md` §4.1, §5) explicitly
forbids scoring against synthesized findings. Committing dry-sweep
evidence keeps the slots discoverable and the harness re-runnable
without binding the parity score to ad-hoc data; populated artifacts
arrive when `--mode live` (or `--mode replay` against a recorded
capture) runs.

## Why not reuse `cli/workflows/review_workflow.py`?

The workflow currently routes through `CodexClient` and is being
ported to OpenRouter as part of AC 1's body. The bake-off runner needs
to capture OpenRouter findings *now* so the port can be evaluated as
soon as it lands. We make this trade-off deliberately:

* **Pro:** decouples evidence collection from the in-flight workflow
  rewrite; the same runner survives the rewrite without changes.
* **Con:** the runner duplicates the bare minimum of the OpenRouter
  call shape (URL, headers, response_format, message construction).
  This is acceptable because (a) it's small (~50 LOC of glue) and
  (b) it goes through the *same* `openrouter_review_response_format`
  + `validate_review_payload` contract that the workflow port will
  use, so a parity drift between the runner and the workflow is
  caught by the schema validator, not by silent semantic drift.

When the workflow port lands and the runner is no longer the only
OpenRouter caller, this README and `run_openrouter.py` should be
revisited — at minimum, the system + user prompt construction should
be factored out of the runner and shared with the workflow.
