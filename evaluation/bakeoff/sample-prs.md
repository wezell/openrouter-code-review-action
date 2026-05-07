# Parity bake-off: curated PR sample

> **Sub-AC 1.1.1.** Identify and select 5–10 representative historical PRs
> covering varied change types (bug fixes, features, refactors,
> security-relevant changes) and document the selection rationale.

This document defines the sample of merged dotCMS/core PRs the bake-off will
re-review with both the legacy Codex action (baseline) and the new
OpenRouter-routed action (candidate). The structured machine-readable index
lives next to it in [`sample-prs.yml`](./sample-prs.yml); downstream Sub-ACs
load that file and treat this `.md` as the human-facing rationale.

## Why a curated sample (and not a random one)

The seed locks the parity bar at:

- ≥80% finding overlap with the Codex baseline
- false-positive rate no worse than Codex
- inline comments anchored to valid PR-diff lines

A uniformly-random sample of recent merged PRs would skew toward whatever
class of change happens to dominate the past two weeks of activity (right
now, small backend bug fixes). That would let a candidate model pass the
80% bar by being good at exactly one finding class, while silently
regressing on the others. We want the bake-off to surface *class-specific*
regressions, not aggregate them away.

So the sample is hand-picked to span four axes simultaneously:

1. **Change type.** bug_fix · feature · refactor · security · performance · infra
2. **Surface.** backend (Java) · frontend (Angular/TypeScript) · infra (YAML)
3. **Diff size.** From a 3-line surgical fix (#35567) up to a 71-file
   mechanical refactor (#35449), so we can see whether either reviewer
   degrades at the tails.
4. **Security signal.** A flag for changes where false-positive discipline
   matters most, so we can compute FP rate on the security-sensitive subset
   independently of the overall sample.

## Sampling method

All PRs were drawn from the most-recent ~2 weeks of merged PRs on the primary
consumer repo, dotCMS/core, via:

```
gh pr list --repo dotCMS/core --state merged --limit 50 \
  --json number,title,labels,mergedAt,additions,deletions,changedFiles
```

From that list, ten PRs were selected by hand to satisfy the four axes above.
Each PR is reviewable against its **merge-commit SHA** (recorded in
`sample-prs.yml`), so both the Codex baseline run and the OpenRouter
candidate run see the identical diff regardless of subsequent main activity.

## The sample

| #     | Type        | Surface  | Sec? | Files | +/−        | One-liner |
|-------|-------------|----------|------|-------|------------|-----------|
| 35567 | bug_fix     | backend  |      | 1     | +0/−3      | Remove `@CloseDBIfOpened` to restore atomicity with contentlet save |
| 35509 | security    | backend  | ✓    | 3     | +64/−2     | Enforce anonymous-access restrictions on content-references endpoints |
| 35498 | security    | backend  | ✓    | 3     | +20/−11    | Disable Jersey WADL descriptor endpoint |
| 35458 | security    | backend  | ✓    | 4     | +300/−3    | Propagate role revocation through short-term permission cache |
| 35491 | performance | backend  |      | 2     | +184/−25   | Replace unbounded query loop with `LIMIT 1` DB call in `validateRelationships()` |
| 35469 | bug_fix     | backend  |      | 3     | +204/−31   | Guard StoryBlock JSON parsing against non-object scalars |
| 35518 | refactor    | frontend |      | 2     | +1/−78     | Remove duplicate copy-URL button from UVE editor toolbar |
| 35449 | refactor    | frontend |      | 71    | +251/−211  | Migrate from Material Icons to Material Symbols Outlined |
| 35465 | feature     | frontend |      | 6     | +267/−7    | Chip-style filter component for content-drive toolbar |
| 35522 | infra       | infra    | ✓    | 2     | +133/−58   | Extract rollback-safety check to a dedicated workflow |

### Coverage summary

- **Change type:** 2 bug_fix · 1 feature · 2 refactor · 3 security · 1 performance · 1 infra
- **Surface:** 6 backend · 3 frontend · 1 infra
- **Security-signal subset:** 4 PRs (35509, 35498, 35458, 35522)
- **Diff-size spread:** smallest 3 lines · median ~120 lines · largest 71 files / 462 lines

### Per-PR rationale

#### #35567 — `fix(unique-fields)` (bug_fix · backend · 3-line)
Three deleted lines removing a `@CloseDBIfOpened` annotation. The reviewer
must reason about transactional semantics from an *absence* — a class of
finding LLM reviewers commonly miss. Smallest possible diff in the sample;
serves as the floor case.

#### #35509 — `fix(content): enforce anonymous access` (security · backend)
AuthN/AuthZ fix on a REST endpoint with paired OpenAPI + integration-test
changes. Forces the reviewer to cross-check spec, implementation, and test
coverage in one diff. Direct exercise of the security-relevant finding class.

#### #35498 — `fix(rest-api): disable WADL` (security · backend)
Hardening change disabling an information-disclosure surface (Jersey WADL).
Tiny diff, high signal — the reviewer should affirm the hardening intent
without inventing spurious "removed feature" complaints.

#### #35458 — `fix(permissions): role revocation cache` (security · backend)
Cache-invalidation bug in the permission system — classically the hardest
class of bug for LLM reviewers (state synchronization across cache + factory +
integration test). Selected to stress cross-file reasoning without being so
large that diff context overflows.

#### #35491 — `fix(content): unbounded query loop` (performance · backend)
Replaces an unbounded loop with a `LIMIT 1` query. The reviewer should
affirm the perf rationale and verify the new query preserves correctness;
a strong test of "praise legitimate fixes without inventing new objections"
(false-positive discipline).

#### #35469 — `fix(block-editor): guard JSON parsing` (bug_fix · backend)
Defensive parsing against malformed input — a representative
"input validation" finding class. Includes paired transformer + integration
test, so the reviewer should confirm coverage maps to the new guard.

#### #35518 — `refactor(uve): remove duplicate button` (refactor · frontend)
Pure deletion refactor on the frontend (Angular component HTML + TS). Tests
whether the reviewer can recognize a clean "remove dead UI" change without
inventing missing-test or behavior-regression complaints — the canonical
FP trap for refactor PRs.

#### #35449 — `refactor: Material Symbols migration` (refactor · frontend)
Mechanical multi-file refactor (71 files). Stresses large-diff handling,
summarization discipline, and the reviewer's ability to skip per-file noise
on a uniform rename. Caps the upper end of the diff-size distribution
without overflowing context.

#### #35465 — `feat(content-drive): chip filter` (feature · frontend)
Net-new Angular feature with a paired component spec. Covers the "new code
path with tests" feature class on the frontend stack, where dotCMS reviewers
expect framework-aware findings (signals, change detection, accessibility).

#### #35522 — `ci: rollback-safety workflow` (infra · CI)
Pure GitHub Actions YAML change — exercises the reviewer on infra DSL where
false positives about "missing tests" or "unhandled errors" are common.
Carries `security_signal: true` because it touches the AI-orchestrator
rollback gate.

## How downstream Sub-ACs consume this

Subsequent bake-off Sub-ACs (capturing the Codex baseline; running the
OpenRouter candidate; computing finding-overlap and FP-rate) load
`sample-prs.yml` as their input. The `merge_commit` field is the canonical
review target — both runs check out that SHA so they review identical
diffs. The `change_type`, `surface`, and `security_signal` fields let the
overlap/FP metrics be sliced per class so a class-specific regression can't
be averaged away by the headline 80% number.

## Replacement policy

If a PR turns out to be unreviewable (e.g. the merge commit is force-pushed
out of the repo, or the diff ends up exceeding the model's context window),
replace it with another PR from the same `change_type` and `surface`
combination drawn from the same sampling window, and record the swap as a
note in `sample-prs.yml`. Do not silently shrink the sample below 5 PRs.
