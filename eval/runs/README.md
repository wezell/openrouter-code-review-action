# Bake-off run artifacts

Per-PR raw findings output for the Codex → OpenRouter parity bake-off.
The directory layout, schema, and intended consumers are pinned by
[`eval/baseline-methodology.md`](../baseline-methodology.md) §4.1 and §4.2.

## Layout

```
eval/runs/
  <pr_id>/
    codex.json        # raw Codex baseline ReviewRunResult + posting outcome
    codex_scored.json # per-finding match against eval/labeled_dataset/        (Sub-AC 2.3.2)
    openrouter.json   # raw OpenRouter run ReviewRunResult + posting outcome   (Sub-AC 1.3)
    labels.yaml       # per-finding TP/FP/DUP/CARRY labels                     (Sub-AC 1.4)
    meta.yaml         # head_sha, model, reasoning_effort, web_search_mode,
                      # capture_status, capture_runner_version, captured_at
  _codex_scoring_summary.json   # sample-level rollup of codex_scored.json    (Sub-AC 2.3.2)
```

`<pr_id>` matches the `id` field of an entry in `eval/sample-prs.yaml`
(e.g. `dotcms-core-35567`). One subdirectory per PR in the sample.

## Capture status

Each `codex.json` (and later `openrouter.json`) carries a top-level
`capture` block:

```json
{
  "capture": {
    "status": "pending" | "captured" | "failed" | "skipped",
    "runner_version": "1",
    "captured_at": null | "2026-05-06T18:00:00Z",
    "command": "python -m eval.run_codex_baseline --pr-id <id>",
    "notes": "free-form — what blocked or limited the run"
  },
  "review_run_result": { ... },
  "posted": { ... }
}
```

- **`pending`** — slot exists but no live capture has run yet. The
  `review_run_result` is `null`. Downstream Sub-ACs MUST treat
  pending PRs as "not yet measurable" (do not score, do not paper
  over with synthesized data — see methodology §2.3 rule 5: *no
  model self-grading*; the same precision applies to baseline
  capture).
- **`captured`** — `review_run_result` holds the raw model output
  parsed via `cli.core.models.validate_review_payload`.
- **`failed`** — capture attempted but errored; `notes` records why.
- **`skipped`** — explicitly skipped (e.g. PR removed from sample
  after triage). `notes` records the rationale; `eval/sample-prs.yaml`
  must list the PR in the `removed:` section.

## How to populate (live capture)

The Codex baseline is captured by `eval/run_codex_baseline.py`. The
runner reads `eval/sample-prs.yaml`, iterates over each `prs[]` entry,
and writes the captured payload into `eval/runs/<pr_id>/codex.json`.
Run it from a host that has:

1. The legacy Codex action installed (`pip install codex-python==1.122.0`).
2. A working git checkout of every sample PR's repo with the pinned
   `head_sha` reachable.
3. `OPENAI_API_KEY` set to a key that can call the recorded Codex model.
4. `GITHUB_TOKEN` with read access to the source repo (PR diff /
   file content / inline comment thread).

Typical invocation:

```bash
export OPENAI_API_KEY=...
export GITHUB_TOKEN=...
python -m eval.run_codex_baseline --pr-id dotcms-core-35567
# or, populate every pending PR:
python -m eval.run_codex_baseline --all
```

Each run:

1. Loads the pinned `head_sha` from `eval/sample-prs.yaml`.
2. Checks out `head_sha` in a clean working tree under
   `${RUNNER_TEMP:-/tmp}/codex-baseline/<pr_id>/`.
3. Invokes the existing Codex CLI (`cli.main`) in `--dry-run` mode so
   the run captures the structured `ReviewRunResult` without posting
   live comments to GitHub. Posting telemetry (`PostingOutcome`) is
   captured separately by replaying inline comments through the
   GitHub-pulls preview endpoint.
4. Writes `codex.json` and `meta.yaml` atomically (tmp file + rename)
   so partial captures cannot corrupt prior artifacts.

## Per-PR scoring artifacts (`codex_scored.json`)

`eval/score_codex_baseline.py` (Sub-AC 2.3.2) is the consumer of
`codex.json`. It walks each PR's captured findings, matches them
against the ground-truth entries in
[`eval/labeled_dataset/dataset.yaml`](../labeled_dataset/dataset.yaml)
per methodology §4.5, and writes one
`eval/runs/<pr_id>/codex_scored.json` per PR plus a sample-level
`_codex_scoring_summary.json` rollup.

Each entry in `findings_scored[]` records, for one Codex finding:

- `matched_dataset_id` — the best-matching ground-truth entry id
  (or `null` for unmatched findings).
- `assigned_label` — `TP` / `FP` / `UNMATCHED` based on the matched
  entry's label.
- `match_quality` — `exact_line` / `line_proximity` / `keyword_only`
  / `none`. Stronger qualities win when multiple entries qualify.
- `match_distance` — closest-line distance between the finding and
  the dataset entry (0 = overlap, > 0 = within ± tolerance).
- `dataset_entry` — the matched ground-truth entry inlined for
  cheap downstream consumption.

Per-PR `aggregate.{matched_tp, matched_fp, unmatched, by_quality,
by_severity}` and `dataset_coverage.{dataset_total, dataset_matched,
by_label_total, by_label_matched, dataset_unmatched_ids}` give the
parity score (Sub-AC 1.5) the per-PR axes it needs without
re-walking the raw artifacts.

Pending tolerance: the scorer treats every non-`captured` capture
state (pending / failed / unknown / missing) as **not yet
measurable** and writes `status = "skipped"` with a free-form
`notes` rationale instead of synthesising scores. This honours
methodology §2.3 rule 5 — never grade against synthesised data.

Usage:

```bash
# Score every sample PR (the default).
.venv/bin/python -m eval.score_codex_baseline

# Score a specific PR after re-capture.
.venv/bin/python -m eval.score_codex_baseline --pr-id dotcms-core-35567

# CI-style drift check (rebuild in memory; non-zero on disk drift).
.venv/bin/python -m eval.score_codex_baseline --check
```

Re-run the scorer whenever a `codex.json` artifact lands or the
labeled dataset changes; otherwise the on-disk
`codex_scored.json` files drift from their inputs and `--check`
flags it.

## Why these stay checked in

- The bake-off must be reproducible across model swaps. The fixed
  Codex baseline is the *anchor* every future model swap is scored
  against (per methodology §1.1 — "fixed once, snapshotted").
- Future model bumps will append `eval/runs/<pr_id>/openrouter-<model>.json`
  alongside the existing baseline; trend tracking depends on the
  historical Codex `codex.json` staying in-tree.
- Schema-conforming placeholders make it visible at-a-glance which
  PRs still need a live baseline capture, so the bake-off can't be
  silently scored on a partial sample.
