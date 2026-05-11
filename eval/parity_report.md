# Codex-vs-OpenRouter Parity Report

_Generated at: `2026-05-08T02:25:47Z`_

**Threshold gate:** PASS — micro.recall unmeasurable (ready_count=0); allow-unmeasurable set

## Sample Rollup

- Sample size: 10
- Ready PRs (both sides captured at same head_sha): 0
- Status breakdown: pending=10

| Metric | Micro (pooled) | Macro (per-PR mean) |
| --- | --- | --- |
| Precision | n/a | n/a |
| Recall    | n/a | n/a |
| F1        | n/a | n/a |

## Per-severity Breakdown

| Severity | Codex total | OpenRouter total | Matched (codex) | Matched (or) | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 0 | 0 | 0 | 0 | n/a | n/a |
| P1 | 0 | 0 | 0 | 0 | n/a | n/a |
| P2 | 0 | 0 | 0 | 0 | n/a | n/a |
| P3 | 0 | 0 | 0 | 0 | n/a | n/a |
| unknown | 0 | 0 | 0 | 0 | n/a | n/a |

## Per-PR Details

| PR | Status | Codex | OpenRouter | Matched | Codex-only | OR-only | Δtotal | Recall | Precision | F1 | Sev-match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dotcms-core-35449` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |
| `dotcms-core-35458` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |
| `dotcms-core-35465` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |
| `dotcms-core-35469` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |
| `dotcms-core-35491` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |
| `dotcms-core-35498` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |
| `dotcms-core-35509` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |
| `dotcms-core-35518` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |
| `dotcms-core-35522` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |
| `dotcms-core-35567` | pending | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | 0 |

## Source Artifacts

- Per-PR overlap: `eval/runs/<pr-id>/overlap.json`
- Aggregate rollup: `eval/overlap_aggregate.json`
- Codex-vs-OpenRouter delta: `tests/fixtures/parity_report/`

_Generator: `eval/parity_report.py` (schema v1)._
