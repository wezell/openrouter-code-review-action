# OpenRouter ↔ Codex parity report (Sub-AC 2.4)

> **Sub-AC 2.4 deliverable.** Add a comparison/report step that diffs
> candidate findings against the Codex baseline (counts, overlap,
> deltas) and emits a summary artifact for review.

This tree holds the side-by-side parity report consumed by reviewers
when deciding whether the OpenRouter-routed action is at parity with
the Codex baseline. Inputs:

| Side       | Source fixture                                                  | Producer (Sub-AC) |
|------------|-----------------------------------------------------------------|-------------------|
| Codex      | `tests/fixtures/baseline/<pr-id>/codex-findings.json`           | Sub-AC 2.2 (`eval.mirror_codex_baseline`) |
| OpenRouter | `tests/fixtures/candidate/<pr-id>/openrouter-findings.json`     | Sub-AC 2.3 (`eval.normalize_openrouter_findings`) |

Output (this directory):

```
tests/fixtures/parity_report/
├── README.md          # this file
├── _summary.json      # sample-level rollup (status counts, weighted overlap)
├── pr-35449.json      # one normalized parity record per sample PR
├── pr-35458.json
└── ...
```

## Generating / refreshing

```bash
# Rebuild every report from the current fixture trees.
.venv/bin/python -m eval.compare_findings

# Restrict to one PR (e.g. after a fresh OpenRouter capture lands).
.venv/bin/python -m eval.compare_findings --pr-id pr-35449

# CI-friendly drift check: rebuild in memory, exit non-zero if the
# on-disk report differs from the rebuild.
.venv/bin/python -m eval.compare_findings --check
```

The script is idempotent: re-running it overwrites every record. The
`generated_at` timestamp is excluded from `--check` comparisons so a
fresh-clone rebuild does not register as drift.

## Status semantics

Each per-PR record carries a single `status` that tells a reviewer
whether the numbers are meaningful:

| Status              | Meaning                                                                  |
|---------------------|--------------------------------------------------------------------------|
| `pending`           | Both sides are placeholders (no live capture). Counts are all 0.         |
| `codex_only`        | Codex captured, OpenRouter still pending — overlap not yet measurable.   |
| `openrouter_only`   | OpenRouter captured, Codex still pending.                                |
| `ready`             | Both sides captured at the same `head_sha`. Overlap numbers are valid.   |
| `head_sha_drift`    | Both captured but at different SHAs. Counts are emitted; overlap is `null`. |

`overlap_ratio_codex_baseline` and `overlap_ratio_openrouter_candidate`
are **only** populated when `status == "ready"`. Anything else surfaces
`null` so the reader never confuses "not measurable" with "0% parity".

## Per-PR record schema

```jsonc
{
  "schema_version": "1",
  "pr_id": "pr-35449",
  "status": "ready",
  "tolerance_lines": 3,
  "head_sha_baseline":  "<sha from codex-findings.json>",
  "head_sha_candidate": "<sha from openrouter-findings.json>",

  "codex": {
    "capture_status": "captured",
    "model": "gpt-5.4",
    "findings_count": 7,
    "by_severity": { "P0": 0, "P1": 2, "P2": 4, "P3": 1, "unknown": 0 },
    "notes": "..."
  },
  "openrouter": {
    "capture_status": "completed",
    "model": "anthropic/claude-opus-4.7",
    "findings_count": 6,
    "by_severity": { "P0": 0, "P1": 2, "P2": 3, "P3": 1, "unknown": 0 },
    "notes": "..."
  },

  "counts": {
    "codex_total": 7,
    "openrouter_total": 6,
    "matched_pairs": 5,
    "codex_only": 2,
    "openrouter_only": 1
  },
  "overlap": {
    "overlap_ratio_codex_baseline":      0.7143,    // matched / codex_total
    "overlap_ratio_openrouter_candidate": 0.8333,   // matched / openrouter_total
    "severity_match_count": 5
  },
  "deltas": {
    "delta_total": -1,                              // openrouter - codex
    "delta_by_severity": { "P0": 0, "P1": 0, "P2": -1, "P3": 0 }
  },

  "matches": [
    {
      "codex_finding_id":       "codex.f0001",
      "openrouter_finding_id":  "openrouter.f0002",
      "path": "core-web/.../UniqueFieldsValidator.java",
      "line_distance": 0,
      "codex_line_range":       [42, 44],
      "openrouter_line_range":  [43, 45],
      "codex_severity":         "P1",
      "openrouter_severity":    "P1",
      "severity_match":         true
    }
  ],
  "codex_only_findings":      [ /* {finding_id, path, line_start, line_end, severity, title, ...} */ ],
  "openrouter_only_findings": [ /* same */ ],

  "generated_at": "2026-05-07T15:54:33Z",
  "generator":    "eval/compare_findings.py"
}
```

## Sample-level rollup (`_summary.json`)

```jsonc
{
  "schema_version": "1",
  "generated_at":   "...",
  "generator":      "eval/compare_findings.py",
  "tolerance_lines": 3,
  "sample_size": 10,
  "status_counts": { "pending": 10 },         // status -> count across the sample

  "totals": {
    "codex_findings_total":    0,
    "openrouter_findings_total": 0,
    "matched_pairs_total":     0,
    "codex_only_total":        0,
    "openrouter_only_total":   0,
    "severity_match_total":    0
  },
  "ready_overlap": {
    "codex_findings":          0,
    "openrouter_findings":     0,
    "matched_pairs":           0,
    "overlap_ratio_codex_baseline":      null, // weighted across "ready" PRs only
    "overlap_ratio_openrouter_candidate": null
  },

  "prs": [
    {
      "pr_id": "pr-35449",
      "status": "pending",
      "codex_findings": 0,
      "openrouter_findings": 0,
      "matched_pairs": 0,
      "codex_only": 0,
      "openrouter_only": 0,
      "delta_total": 0,
      "head_sha_baseline":  "<sha or null>",
      "head_sha_candidate": "<sha or null>"
    }
    // ...one row per sample PR
  ]
}
```

The rollup is the **operator dashboard**: `status_counts` shows how
many PRs have ready overlap numbers and how many are still waiting on
captures; `ready_overlap.overlap_ratio_codex_baseline` is the
sample-weighted parity number used to gate the AC-1 finding-overlap
evaluation principle (≥ 80%).

## Why the report is checked into git

* **Diff-able state changes.** A capture lands → `_summary.json`'s
  `status_counts` shifts in a code-reviewable diff. Silent regression
  (e.g. an OpenRouter recapture that drops findings) shows up as a
  red flag in PR review.
* **Discoverable slots.** Each `pr-*.json` is a `pending` placeholder
  on a fresh clone, so reviewers see every sample slot rather than
  having to discover gaps PR-by-PR.
* **Reproducible parity.** Future model swaps re-run the same sample
  and rebuild the same report shape; historical records stay in-repo
  so trend tracking is a `git log` away.

## Relationship to the other parity artifacts

* `evaluation/bakeoff/comparison/` (Sub-AC 1.1.3) — pre-labeling view
  with interleaved findings + match candidates for the human labeling
  pass.
* `eval/runs/<pr_id>/codex_scored.json` (Sub-AC 2.3.2) — Codex side
  scored against the labeled ground-truth dataset.
* **This directory** (Sub-AC 2.4) — pure A/B comparison: Codex vs
  OpenRouter, no ground-truth dependency, runs against the committed
  fixtures so parity can be reported even before the labeled dataset
  is finalized.
