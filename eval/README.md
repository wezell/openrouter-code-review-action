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
