# Bake-off Evaluation

This directory contains the artifacts that govern the Codex → OpenRouter
bake-off described in `seed.yaml` acceptance criteria 1 and 2:

- **AC 1** — On a curated 5-10 PR sample, ≥80% finding overlap with Codex
  baseline.
- **AC 2** — False-positive rate does not exceed Codex baseline on the
  sample.

## Contents

| File | Purpose |
|------|---------|
| `baseline-methodology.md` | The authoritative methodology: sample-PR selection criteria, labeling rubric, false-positive definition, and the Codex baseline measurement procedure. **Read this first.** |
| `findings-schema.md` | **Sub-AC 2.1.** Normalized JSON findings schema (fields, types, severities, file/line anchors) documented side-by-side against the Codex baseline schema for diffability. Single source of truth for the wire shape every `eval/runs/<pr_id>/{codex,openrouter}.json` and `tests/fixtures/baseline/<pr-id>/codex-findings.json` agrees on. |
| `sample-prs.yaml` | The sample-PR registry template. Downstream Sub-ACs populate the `prs:` list with concrete PR identifiers, head SHAs, and review-mode flags. |
| `run_codex_baseline.py` | Per-PR Codex baseline runner (Sub-AC 1.2 / 2.1). Writes `eval/runs/<pr_id>/codex.json`. |
| `score_codex_baseline.py` | **Per-PR Codex scorer** (Sub-AC 2.3.2). Walks each PR's captured findings, matches them against the labeled-dataset ground-truth per methodology §4.5, and writes `eval/runs/<pr_id>/codex_scored.json` plus `eval/runs/_codex_scoring_summary.json`. Pending captures are surfaced as `status = "skipped"` rather than synthesised. Run `python -m eval.score_codex_baseline` (or `--check` for CI drift). |
| `runs/` | Per-PR Codex baseline run artifacts (raw findings JSON + meta) and per-PR scored projections. Pending placeholders are committed; live captures overwrite them in place. See `runs/README.md` for the layout. |
| `labeled_dataset/` | **Ground-truth labeled benchmark dataset** (Sub-AC 2.3.1): checked-in catalog of historical-finding entries with `TP` / `FP` labels per `baseline-methodology.md` §2-§3, plus a typed loader and validator. This is the long-lived benchmark every model swap is scored against. See `labeled_dataset/README.md` for the schema. |
| `../evaluation/bakeoff/comparison/` | **Structured comparison reference set** (Sub-AC 1.1.3): one normalized JSON per sample PR (joined Codex + OpenRouter view, flat finding fields, pre-computed match candidates) plus a sample-level `_index.json` rollup. This is what the labeling and scoring sub-ACs consume — not the raw `runs/` artifacts. See `../evaluation/bakeoff/comparison/README.md` for the schema. |

## Status

- Methodology and registry template: **defined** (Sub-AC 2.1).
- Sample PR selection: **defined** — 10 PRs selected by Sub-AC 1.1.1
  (rationale in `../evaluation/bakeoff/sample-prs.{md,yml}`).
- Codex baseline runs: **scaffolded** — `runs/<pr_id>/codex.json`
  placeholders committed; live captures pending OPENAI/GITHUB tokens.
- Comparison reference set: **scaffolded** — Sub-AC 1.1.3 organizer
  emits normalized per-PR JSON to
  `../evaluation/bakeoff/comparison/`. Refresh with
  `python -m evaluation.bakeoff.build_comparison_set`.
- Ground-truth labeled dataset: **defined** (Sub-AC 2.3.1) —
  `labeled_dataset/dataset.yaml` plus `labeled_dataset/schema.py` for
  validation. Every PR in the registry has at least one TP entry and
  at least one PR has FP traps; expand entries (and pin
  `line_range_status: confirmed`) as labelers inspect each diff.
- Codex baseline scoring: **defined** (Sub-AC 2.3.2) —
  `score_codex_baseline.py` matches captured Codex findings against
  the labeled dataset and writes `runs/<pr_id>/codex_scored.json`
  plus `runs/_codex_scoring_summary.json`. All 10 PRs currently
  scored as `status = "skipped"` because Codex baseline captures are
  still pending; re-running the scorer after live captures land
  flips them to `status = "scored"` with real per-finding labels.
- OpenRouter runs, per-run labeling, and parity scoring: **pending**
  (downstream Sub-ACs 2.2 / 1.4 / 1.5).

## Why this lives in the repo

The methodology is checked in so that:

1. The bake-off is reproducible — anyone can re-run it with identical
   selection rules and labeling rubric.
2. Downstream model swaps (per the Seed's swap-speed principle) can
   re-validate against the same fixed sample.
3. Evaluation principles (`finding_overlap`, `false_positive_discipline`)
   are pinned to a written contract, not folklore.

## Parity Gate (Sub-AC 4.1 / 4.2 / 4.3)

The parity gate is the CI quality bar that a candidate model must clear
before it can replace the Codex baseline in `.openrouter-review.yml`.
It is enforced by the `eval-overlap` job in `.github/workflows/ci.yml`
and re-runnable locally via the `make eval-overlap` /
`make eval-parity-report` targets.

### What it gates

| Metric         | Definition                                                                                                                  | Default threshold | Source artifact                |
|----------------|------------------------------------------------------------------------------------------------------------------------------|-------------------|--------------------------------|
| `micro.recall` | Pooled recall across the whole sample: `Σ TP / Σ (TP + FN)`. Bias-toward-PRs-with-more-findings is intentional — high-finding PRs dominate the bake-off signal. | **0.80**          | `eval/overlap_aggregate.json`  |
| `macro.recall` | Mean of per-PR recall. Available as alternate `--metric`; treats every PR as one vote regardless of finding count.          | (alternate)       | `eval/overlap_aggregate.json`  |
| `precision`    | Reported but **not gated** — false-positive discipline is policed by `eval/labeled_dataset/` scoring (Sub-AC 2.3.2), not the overlap gate. | n/a               | `eval/runs/<pr_id>/overlap.json` |

### Threshold rationale

- **0.80 (80%) micro-recall** comes directly from the Seed's
  `finding_overlap` evaluation principle — _"Review findings overlap ≥80%
  with Codex baseline on sample PRs."_ It is the single number a model
  swap is allowed to land on without explicit human override.
- **Why micro, not macro:** `seed.yaml` describes overlap on the
  **sample**, not on each PR independently. Pooled (micro) recall is the
  population-weighted answer; macro recall would let a single tiny PR
  with one missed finding tank the whole gate. Both are emitted to
  `overlap_aggregate.json` so reviewers can sanity-check both views.
- **Why recall, not F1:** the Codex baseline is the ground truth in this
  comparison. Missing a Codex finding is the failure mode the gate is
  built to catch; surfacing _additional_ findings on the candidate side
  is fine (and is separately measured against the labeled dataset).
- **Soft-pass during bring-up (`ALLOW_UNMEASURABLE=1`):** until both
  Codex and OpenRouter captures land on every sample PR, the aggregate
  has `ready_count = 0` and metrics are `null` ("unmeasurable"). The
  CI job currently runs with `ALLOW_UNMEASURABLE=1` (in
  `.github/workflows/ci.yml` and `Makefile`) so the gate emits a warning
  instead of failing. Flip to `0` once the sample is fully populated to
  enforce strictly. Soft-pass **never** treats a measured-but-low score
  as a pass — it only covers the unmeasurable case.

### Reading the parity report

`eval/parity_report.md` is the single human-readable artifact CI uploads
on every run (artifact name: `parity-report`, retention 30 days). It is
mechanically generated from three lower-tier artifacts and never edited
by hand:

1. `eval/runs/<pr_id>/overlap.json` — per-PR matches + precision/recall/F1.
2. `eval/overlap_aggregate.json` — sample-wide micro/macro rollup +
   per-severity breakdown.
3. `tests/fixtures/parity_report/` — Codex-vs-OpenRouter match deltas
   (matched / codex-only / openrouter-only finding lists).

Sections, in order:

- **Threshold gate** banner — `PASS` / `FAIL` / `PASS — unmeasurable`.
  Tells you immediately whether this candidate run cleared
  `OVERLAP_THRESHOLD` on `OVERLAP_METRIC`.
- **Sample Rollup** — sample size, ready-PR count, status breakdown
  (`ready` / `pending` / `unmeasurable`), and the micro+macro
  precision/recall/F1 table. `n/a` means at least one prerequisite
  capture is missing for that metric.
- **Per-severity Breakdown** — Codex/OpenRouter totals, matches, and
  per-side recall/precision split by P0–P3 + `unknown`. Use this to
  spot regressions concentrated in a single severity (e.g. matching well
  on P3 noise but missing P0 findings).
- **Per-PR Details** — one row per sample PR. `Δtotal` is
  `openrouter_total - codex_total`; large positive deltas with low
  recall are the classic "candidate is finding lots of unrelated stuff
  while missing what Codex flagged" failure mode.
- **Source Artifacts** — pointers back to the JSON tiers above for
  programmatic consumers.

### Regenerating the report

The full pipeline is wired through Make targets so contributors and CI
run the same commands:

```bash
# 1. Capture (or refresh) the underlying findings:
#      - python -m eval.run_codex_baseline
#      - python -m eval.run_openrouter_baseline
#    (Both write into eval/runs/<pr_id>/. See baseline-methodology.md.)

# 2. Score the overlap pipeline end-to-end + run the threshold gate.
#    Regenerates per-PR overlap.json and overlap_aggregate.json, then
#    fails if micro.recall < 0.80 (unless ALLOW_UNMEASURABLE=1).
make eval-overlap

# 3. Build the consolidated parity report (Markdown + JSON).
make eval-parity-report

# 4. Drift-check (CI uses this — read-only, no regeneration):
make eval-overlap-check
make eval-parity-report-check
```

Knobs (settable as Make vars or env vars):

| Var                  | Default       | Effect                                                                 |
|----------------------|---------------|------------------------------------------------------------------------|
| `OVERLAP_THRESHOLD`  | `0.80`        | Numeric pass bar applied to `OVERLAP_METRIC`.                          |
| `OVERLAP_METRIC`     | `micro.recall`| Which aggregate field the gate checks. `macro.recall` is the alternate. |
| `ALLOW_UNMEASURABLE` | `1`           | `1` → soft-pass when prerequisites missing; `0` → strict, fail on any unmeasurable metric. |

### Interpreting failures

| Symptom in `eval/parity_report.md`                                                | Likely cause                                                                                  | First thing to check                                                                                |
|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Banner says `FAIL — micro.recall = X < 0.80`                                       | Candidate model is missing Codex findings on enough of the sample to drop pooled recall.       | Per-severity breakdown — if P0/P1 recall tanked, prompt or model regressed on critical findings.    |
| Banner says `PASS — unmeasurable` with many `pending` PRs                         | Bake-off captures haven't landed yet for those PRs (no `head_sha`-aligned pair).               | `eval/runs/<pr_id>/codex.json` and `openrouter.json` exist + share `head_sha`.                      |
| `Δtotal` is large and positive on most rows but recall stays low                  | Candidate is verbose but not aligned with Codex — likely prompt drift or off-topic findings.    | Inline-anchor validity in the OpenRouter capture, plus prompt diffs vs. last green run.             |
| Row shows `status = unmeasurable` after a force-push                              | Reviewed-SHA ancestry was broken; fresh full review hasn't been re-captured.                   | Re-run baseline + candidate capture for that PR; per Seed, force-push falls back to fresh review.   |
| Drift-check (`make eval-parity-report-check`) fails on a PR with no code change   | Committed `parity_report.md/json` is stale relative to underlying `eval/runs/` artifacts.       | Re-run `make eval-parity-report` and commit the regenerated report.                                 |

### Where the gate lives in CI

- Job: `eval-overlap` in `.github/workflows/ci.yml`.
- Step `Run overlap pipeline + threshold gate` shells `make eval-overlap-check`.
- Steps `Build consolidated parity report` and `Drift-check parity report`
  shell `make eval-parity-report` / `make eval-parity-report-check`.
- Step `Upload parity report artifact` runs unconditionally
  (`if: always()`) so reviewers can download `parity_report.md` even
  when an earlier step failed — this is the primary triage artifact.
