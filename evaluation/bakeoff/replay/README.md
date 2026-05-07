# Bake-off replay captures

This directory holds **raw OpenRouter `/api/v1/chat/completions`
responses** captured from `--mode live` runs of
[`evaluation/bakeoff/run_openrouter.py`](../run_openrouter.py). Each
file is the full JSON object OpenRouter returned, one per sample PR:

```
evaluation/bakeoff/replay/
├── pr-35567.json          # raw response captured against head_sha 9229960...
├── pr-35509.json
└── ...
```

The runner's `--mode replay` reads from this directory, runs each
recorded response through the same `_extract_review_payload` +
`validate_review_payload` machinery as the live path, and persists the
result as `status=replayed` in `runs/<pr_id>/openrouter.json`.

## Why replay

Live OpenRouter calls cost tokens and require an `OPENROUTER_API_KEY`.
Replay decouples evidence collection from credentials:

- **One operator captures live**, commits the raw responses here.
- **Anyone re-runs the bake-off offline** — CI, a reviewer on a fresh
  laptop, the parity-scoring sub-AC. The findings are real (no
  synthesis); only the network call is skipped.
- **Parser regressions surface against real captured data** rather than
  hand-crafted fixtures that drift from production schemas.

Per `eval/baseline-methodology.md` §4.1 and §5, the bake-off is scored
on real captured data — never on synthesized findings. Replay honors
that rule while making the captures cheap to re-derive.

## Capture format

A replay file is the full OpenRouter response object. The minimum
shape the validator needs is:

```jsonc
{
  "id": "<openrouter request id>",
  "model": "anthropic/claude-opus-4.7:online",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        // Strict-schema response_format means content is a JSON-encoded
        // ReviewRunResult (the same shape REVIEW_OUTPUT_SCHEMA enforces).
        "content": "{\"overall_correctness\": \"...\", \"findings\": [ ... ] }"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  }
}
```

The runner only reads `choices[0].message.content` (the structured
`ReviewRunResult` JSON) and `usage` (for cost telemetry). Everything
else is preserved verbatim so the replay file is also a faithful
provenance record of *what OpenRouter actually returned*.

## How to capture

```bash
# 1) Run live once with credentials:
OPENROUTER_API_KEY=sk-or-... \
  .venv/bin/python evaluation/bakeoff/run_openrouter.py --mode live --only pr-35567

# 2) Inspect runs/pr-35567/openrouter.json — the validated review_run_result is in there.
#
# 3) To enable offline replay, also commit the raw API response. The
#    simplest way is to wrap the live call with a small recorder. The
#    runner doesn't auto-record because the Sub-AC 1.2 capture step is
#    deliberately separate from offline-replay tooling — but a recorder
#    is a one-line change in evaluation/bakeoff/run_openrouter.py:
#    `_post_openrouter` returns the raw dict, persist it before the
#    review extraction step.
#
# 4) Move/copy the captured response into:
#       evaluation/bakeoff/replay/pr-35567.json
#
# 5) Verify replay reproduces the same review_run_result:
.venv/bin/python evaluation/bakeoff/run_openrouter.py --mode replay --only pr-35567
```

## Slots without recorded responses

`--mode replay` is **pending-tolerant**. A slot without a
`replay/<pr_id>.json` file is persisted as `status=unfilled` (with a
note pointing the operator at the missing file path). The other slots
in the sweep are unaffected. This mirrors the methodology's
"pending-tolerant" comparison-set design: a partial capture sweep
doesn't break downstream scoring.

## Why this directory may be empty

This is brownfield evidence-collection: the `--mode live` capture step
is gated on operator credentials. Until the first capture sweep runs,
this directory is intentionally empty (other than this README). The
`runs/README.md` Status table records which slots have been captured.
