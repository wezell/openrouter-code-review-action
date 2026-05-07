# Ground-truth labeled dataset (Sub-AC 2.3.1)

> **Sub-AC 2.3.1.** Define a labeled dataset of historical PR review
> findings with ground-truth true-positive / false-positive labels for
> benchmarking.

This directory holds the **benchmark dataset** for the Codex →
OpenRouter parity bake-off. It is a checked-in catalog of candidate
review findings, each anchored to a real PR in the sample registry and
each carrying a retrospective `TP` (true positive) or `FP` (false
positive) ground-truth label.

## What's here

| File | Role |
|------|------|
| `dataset.yaml` | The dataset itself — labeled findings + taxonomies. |
| `schema.py`    | Typed loader, validator, allowed-value contract. |
| `__init__.py`  | Public API re-exports. |

## How this slots into the bake-off

The methodology in [`eval/baseline-methodology.md`](../baseline-methodology.md)
defines two distinct labeling jobs:

1. **Per-run-finding labels.** When a model run produces findings
   (Codex baseline or OpenRouter candidate), every finding gets a
   `TP / FP / DUP / CARRY` label so the run can be scored. Those labels
   live inside the run artifact (`labels[]` block in
   `eval/runs/<pr_id>/codex.json` and
   `evaluation/bakeoff/runs/pr-<n>/openrouter.json`).
2. **Ground-truth findings.** Independent of any model output, what
   *real* bugs are in each sample PR's diff at `head_sha`? And which
   common spurious complaints would a model on this kind of diff
   typically file? **That's this dataset.**

The two jobs feed each other: a labeler scoring run output cross-checks
against `dataset.yaml` so labels are anchored in the diff itself
(methodology §2.3 rule 5: *no model self-grading*). The dataset is also
the long-lived benchmark — when a model is later swapped in via
`.openrouter-review.yml`, the new run is scored against the same
`dataset.yaml`, so model swaps are comparable across time.

## How matching at score time works

Scoring is not a string match. A model finding **matches** a
`dataset.yaml` entry when **all three** of methodology §4.5 hold:

1. Same `path` (the dataset entry's `path_hint`).
2. Line ranges overlap by ≥1 line **or** are within ±3 lines of each
   other. Entries with `line_range_status: unverified` skip the line-
   range check; the labeler uses `match_keywords` as the anchor instead.
3. The labeler judges that the two findings describe the same
   underlying issue — not just adjacent code.

`match_keywords` is provided to make the labeler's judgement step
faster, not to replace it. It's an OR-set of distinctive substrings.

## Schema (per entry)

```yaml
- id: gt-<pr-number>-<seq>     # stable benchmark id, must start with "gt-"
  pr_id: dotcms-core-35509     # cross-ref to eval/sample-prs.yaml prs[].id
  head_sha: 89f68df...         # must match registry entry
  label: TP | FP               # ground-truth label
  severity: P0 | P1 | P2 | P3  # per prompts/review.md
  class: bug_correctness | security | performance |
         refactor_regression | missing_test | hallucination |
         pre_existing | style_nit | speculation | out_of_scope
  path_hint: <path or null>    # full path from repo root
  line_range:                  # null when status is unverified
    start: <int ≥ 1>
    end:   <int ≥ start>
  line_range_status: confirmed | unverified
  title: "P{N}: <short title>"
  rationale: <≤ 2 sentences — why this label, grounded in the diff>
  match_keywords: [<distinctive strings>]
  source: retrospective_diff_review | merged_pr_metadata |
          reviewer_comment_thread | bake_off_diagnostic
  confidence: high | medium | low
```

The allowed values are pinned in [`schema.py`](./schema.py); the
taxonomy blocks at the top of `dataset.yaml` are documentation for
human labelers. The validator rejects any entry that uses values not
in those allow-lists, so a typo can't silently widen the schema.

### `confirmed` vs `unverified` line ranges

Anchoring an entry to exact line numbers requires inspecting the PR's
diff at `head_sha`. The dataset accepts both states so it can be
useful immediately and refined incrementally:

* **`confirmed`** — `line_range` is non-null, pinned by inspection of
  the actual diff. The §4.5 matcher uses the line range directly.
* **`unverified`** — `line_range` is `null`. The matcher falls back
  to `path_hint` + `match_keywords` + reviewer judgement.

The validator enforces consistency: a `confirmed` entry must have a
non-null `line_range`; an `unverified` entry must have `line_range:
null`. At least one of `path_hint` or `line_range` must always be set
so the matcher has *some* anchor.

## Loading the dataset

```python
from eval.labeled_dataset import load_dataset

dataset = load_dataset()
# Per-PR slice
pr_findings = dataset.by_pr("dotcms-core-35509")
# Subset by label
tps = dataset.true_positives()
fps = dataset.false_positives()
```

Validation is automatic on load. Schema or cross-reference failures
raise `DatasetValidationError` with a precise message pointing at the
offending entry.

## Adding new entries

1. Pick a sample PR from `eval/sample-prs.yaml` and inspect its diff
   at the pinned `head_sha`.
2. Add either a real-bug entry (`label: TP`) or a common-spurious-
   complaint entry (`label: FP`). Both kinds belong here — the
   bake-off scores both finding-overlap (TPs) and FP-discipline
   (FPs).
3. Set `line_range_status: confirmed` once the line numbers are
   pinned; otherwise leave it `unverified` and provide `path_hint` +
   `match_keywords`.
4. Run `pytest tests/test_labeled_dataset.py` — the validator and
   cross-reference checks fire on every test run.

## Why this dataset is not Codex output

Methodology §2.3 rule 5 forbids model self-grading. A dataset built
out of Codex findings would build the bias-toward-Codex into the
baseline by construction, defeating the parity measurement. Entries
here are sourced from the **diff and merged-PR metadata** (what bug
is actually being fixed; what intentional change is being made) plus
**bake-off-diagnostic FP traps** (well-known spurious-complaint
classes for each PR archetype). When a labeler later inspects a real
diff, they may upgrade `confidence` and pin `line_range` from the
diff — but the label itself stays grounded in the diff, not in any
model's output.
