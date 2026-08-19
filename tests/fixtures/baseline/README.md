# Baseline PR fixtures — selection rationale

> **Sub-AC 1.1.** Select 5–10 representative PRs spanning diff shapes
> (added-only, mixed, multi-file, renames, large) and document the
> selection rationale.
>
> **Sub-AC 1.2.** Capture each selected PR's diff and metadata
> (PR number, base/head SHAs, changed files) as fixture inputs under
> `tests/fixtures/baseline/<pr-id>/`.

This directory is the test-suite-facing pointer to the curated PR sample
that anchors the Codex-vs-OpenRouter parity bake-off (AC 1). The
machine-readable index, the merge-commit SHAs, and the per-PR change-type
notes live with the bake-off harness so a single source of truth feeds
both `pytest` fixtures and the bake-off runner:

- **Canonical machine index:** [`evaluation/bakeoff/sample-prs.yml`](../../../evaluation/bakeoff/sample-prs.yml)
- **Long-form per-PR rationale:** [`evaluation/bakeoff/sample-prs.md`](../../../evaluation/bakeoff/sample-prs.md)
- **Per-PR comparison records:** [`evaluation/bakeoff/comparison/`](../../../evaluation/bakeoff/comparison/)
- **Findings JSON schema (Sub-AC 2.1):** [`eval/findings-schema.md`](../../../eval/findings-schema.md) — the wire shape every `codex-findings.json` and `openrouter-findings.json` in this tree must conform to, with a field-by-field diff vs. the Codex baseline.

## Per-PR fixture inputs (Sub-AC 1.2)

Every PR in the sample has a checked-in fixture directory captured
directly from `gh pr diff` / `gh pr view --json` so tests can replay the
exact diff shape without a live GitHub round-trip:

```
tests/fixtures/baseline/_index.json                  # machine-readable rollup
tests/fixtures/baseline/<pr-id>/diff.patch           # unified diff at head SHA
tests/fixtures/baseline/<pr-id>/metadata.json        # gh pr view --json output
tests/fixtures/baseline/<pr-id>/codex-findings.json  # Sub-AC 1.3 Codex baseline
```

## Sub-AC 1.3 — Codex baseline findings (`codex-findings.json`)

Each fixture PR carries a `codex-findings.json` mirror of the raw
output produced by running the existing Codex-based action against
that PR. The upstream artifact lives at
[`eval/runs/dotcms-core-<NUMBER>/codex.json`](../../../eval/runs/) and
is captured by [`eval/run_codex_baseline.py`](../../../eval/run_codex_baseline.py)
on a host that has `OPENAI_API_KEY`, `GITHUB_TOKEN`, and the
`codex-python` package installed (the legacy runner — kept around only
to capture the parity baseline; it is not invoked by the new
OpenRouter-routed action).

The mirror into the fixture tree is produced by
[`eval/mirror_codex_baseline.py`](../../../eval/mirror_codex_baseline.py)
which translates the eval-tree id (`dotcms-core-35567`) into the
fixture-tree id (`pr-35567`) and copies the JSON shape verbatim:

```sh
python -m eval.mirror_codex_baseline           # write mirror
python -m eval.mirror_codex_baseline --check   # CI guard against drift
```

`tests/test_baseline_fixtures.py` pins the contract: every sample PR
must have a `codex-findings.json` carrying the structured fields
(`capture`, `review_run_result`, `posted`, `source`) and the mirror
must match the upstream `eval/runs/<id>/codex.json` byte-for-byte after
id translation.

PRs whose Codex baseline has not been live-captured yet (because the
host running the bake-off harness lacks `OPENAI_API_KEY` /
`GITHUB_TOKEN` / the `codex-python` package) carry
`capture.status = "pending"` with a `notes` field explaining why; once
a credentialed run produces `status = "captured"` with a populated
`review_run_result`, the mirror script propagates it forward.

`_index.json` carries one record per PR with `pr_id`, `number`,
`base_ref`, `base_sha`, `head_ref`, `head_sha`, `merge_commit_sha`,
`changed_files`, plus the diff's `sha256` and byte size, which lets a
test assert that the fixtures on disk match what the rest of the suite
expects without re-parsing every patch. `metadata.json` is the raw
`gh pr view --json` payload (number, base/head ref + oid, mergeCommit,
changed files with per-file additions/deletions, mergedAt, title,
author) — the canonical source for the per-PR fields the anchor engine
and review-posting fixtures key off.

Fixtures are regenerated with:

```sh
gh pr diff <number> --repo dotCMS/core > tests/fixtures/baseline/pr-<number>/diff.patch
gh pr view <number> --repo dotCMS/core --json \
  number,title,author,baseRefName,baseRefOid,headRefName,headRefOid,\
mergeCommit,mergedAt,additions,deletions,changedFiles,state,url,files \
  > tests/fixtures/baseline/pr-<number>/metadata.json
```

The sample-prs.yml `merge_commit` SHA must match
`metadata.json` `.mergeCommit.oid` for every PR; `_index.json` is the
single point of cross-check.

This README documents the selection through the lens Sub-AC 1.1 cares
about: the **diff shape**. The bake-off harness slices by `change_type`
and `surface`; the anchor engine and review-posting tests slice by diff
shape (added-only, mixed, multi-file, renames, large). Both views are
satisfied by the same 10 PRs, mapped below.

## Sample (10 PRs from dotCMS/core)

All PRs are reviewable against their **merge-commit SHA**, so the same
diff is replayable indefinitely regardless of subsequent `main` activity.

| #     | Files | +/−        | Diff shape                | Change type | Surface  | Sec? |
|-------|-------|------------|---------------------------|-------------|----------|------|
| 35567 | 1     | +0/−3      | deletion-only (floor)     | bug_fix     | backend  |      |
| 35509 | 3     | +64/−2     | added-only                | security    | backend  | ✓    |
| 35498 | 3     | +20/−11    | mixed                     | security    | backend  | ✓    |
| 35458 | 4     | +300/−3    | added-only, multi-file    | security    | backend  | ✓    |
| 35491 | 2     | +184/−25   | mixed                     | performance | backend  |      |
| 35469 | 3     | +204/−31   | mixed                     | bug_fix     | backend  |      |
| 35518 | 2     | +1/−78     | deletion-only             | refactor    | frontend |      |
| 35449 | 71    | +251/−211  | large, multi-file, rename-shape | refactor | frontend |  |
| 35465 | 6     | +267/−7    | added-only, multi-file    | feature     | frontend |      |
| 35522 | 2     | +133/−58   | mixed (YAML)              | infra       | infra    | ✓    |

### Diff-shape coverage

The Sub-AC 1.1 axes map onto the sample as follows:

- **added-only** (no deletions inside the changed hunks worth speaking of):
  PR 35509, PR 35458, PR 35465. Exercises the anchor engine's `RIGHT`-side
  line resolution and the "new file or near-new code" path through the
  review prompt.
- **mixed** (additions and deletions interleaved): PR 35498, PR 35491,
  PR 35469, PR 35522. Exercises hunk-walking, multi-line range anchors,
  and the per-comment fallback ladder (`SingleAnchor` ↔ `RangeAnchor`).
- **deletion-only** (the inverse mixed case the anchor engine must not
  drop on the floor): PR 35567 (3-line deletion), PR 35518 (78-line
  delete-dead-UI). Forces the reviewer to anchor commentary on the
  *absence* of code rather than newly added lines.
- **multi-file** (≥4 changed files): PR 35458, PR 35465, PR 35449.
  Exercises rename-map merging, per-file batching, and the review-posting
  workflow's batched `POST /pulls/{n}/reviews` call.
- **renames / mechanical multi-file rename-shape:** PR 35449 (71-file
  Material Icons → Material Symbols migration). The dotCMS sampling
  window did not produce a git-rename-detected PR in the requested size
  band; PR 35449 stands in for the same anchor-engine stress (uniform
  textual replacement across many files). If a true `status: renamed`
  PR is needed for the rename-map regression test, the fallback is
  documented in the *Replacement policy* section of
  [`sample-prs.md`](../../../evaluation/bakeoff/sample-prs.md).
- **large** (>50 changed files or >400 changed lines): PR 35449. Caps
  the upper end of the diff-size distribution so the reviewer's
  large-diff handling, summarization discipline, and per-file noise
  suppression are exercised without overflowing the model context window.

### Why these axes (and why not just random-sampled)

A random sample of recent merged PRs would skew toward whatever class of
change happens to dominate the past two weeks of activity (currently:
small backend bug fixes). That would let a candidate model pass the
seed's 80% finding-overlap bar by being good at exactly one diff shape
while silently regressing on the others.

The 10 PRs above span the full diff-shape distribution the production
reviewer will hit in practice. The bake-off can therefore detect a
*shape-specific* regression (e.g. "the new model anchors fine on
added-only diffs but mis-anchors on deletion-only ones") instead of
averaging the regression away in a single headline number.

## How tests consume this directory

The fixtures README is intentionally a thin pointer: it stays git-tracked
even if the comparison records under `evaluation/bakeoff/runs/` are
regenerated, and it gives `pytest` collectors a single discoverable
location for the diff-shape mapping.

A test that needs to assert "we have ≥1 PR per diff shape and the merge
commits resolve" should read [`sample-prs.yml`](../../../evaluation/bakeoff/sample-prs.yml)
and cross-check against the table above. The merge-commit SHAs in that
YAML are the canonical review target.

## Replacement policy

If a PR turns out to be unreviewable (force-pushed merge commit, diff
exceeds the model's context window, etc.), replace it with another PR
from the same diff-shape *and* `change_type` combination drawn from the
same sampling window, and record the swap in `sample-prs.yml`. Do not
silently shrink the sample below 5 PRs.
