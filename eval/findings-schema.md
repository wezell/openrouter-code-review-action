# Normalized findings JSON schema (OpenRouter) — diff vs. Codex baseline

> **Sub-AC 2.1.** Define the normalized JSON findings schema (fields,
> types, severities, file/line anchors) and document it alongside the
> Codex baseline schema for diffability.
>
> Pair file: [`baseline-methodology.md`](baseline-methodology.md)
> §4.2 (per-run artifact wrapper). Implementation: `cli/core/models.py`
> (`REVIEW_OUTPUT_SCHEMA`, `REVIEW_FINDING_SCHEMA`,
> `CARRIED_FORWARD_COMMENT_SCHEMA`). Reviewer prompt that produces this
> shape: [`prompts/review.md`](../prompts/review.md).

This document is the **single source of truth** for the JSON shape that
every review-run artifact under `eval/runs/<pr_id>/{codex,openrouter}.json`
and every test fixture under `tests/fixtures/baseline/<pr-id>/codex-findings.json`
agrees on. It is intentionally written so a reviewer can `diff` the
OpenRouter contract against the Codex baseline contract field-by-field
and spot drift before a model swap regresses the bake-off.

The downstream parity ACs (AC 1 finding-overlap, AC 2 false-positive
discipline, AC 3 inline-anchor validity) all assume the two paths emit
the same shape. If the shapes diverge, that drift must be recorded here
first so the bake-off scoring rubric stays valid.

---

## 1. Top-level review-run envelope

Both runs serialize a `ReviewRunResult` produced by the model into the
following JSON object. Field names, types, and ordering match between
the Codex path (legacy) and the OpenRouter path (new); the **only**
permitted divergence is documented in §6.

### 1.1 Schema (normative)

```jsonc
{
  "overall_correctness": "string",                // "patch is correct" | "patch is incorrect"
  "overall_explanation":  "string",               // free-form, model-authored
  "overall_confidence_score": 0.0,                // number in [0.0, 1.0] | null
  "findings": [ /* see §2 */ ],                   // array, possibly empty
  "carried_forward": [ /* see §3 */ ]             // array, possibly empty
}
```

### 1.2 Field contract

| Field                       | Type                               | Required | Nullable | Notes |
|-----------------------------|------------------------------------|----------|----------|-------|
| `overall_correctness`       | `string`                           | yes      | no       | One of `"patch is correct"` / `"patch is incorrect"`. Free string at the JSON-schema level; semantic enum is enforced by the reviewer prompt (`prompts/review.md` §"Final verdict"). |
| `overall_explanation`       | `string`                           | yes      | no       | One-paragraph rationale. Empty string is allowed but discouraged. |
| `overall_confidence_score`  | `number \| null`                   | yes      | yes      | `null` means "model declined to score". Numeric values outside `[0, 1]` are clipped by the reviewer prompt — no programmatic clamp on read. |
| `findings`                  | `array<ReviewFinding>`             | yes      | no       | Empty array is a valid run with zero findings. |
| `carried_forward`           | `array<CarriedForwardComment>`     | yes      | no       | Re-adjudicated prior comments. Empty on a fresh-full review. |

`additionalProperties: false` is enforced at every object level by
OpenRouter `response_format: json_schema, strict: true`. The Codex path
relied on Codex's `output_schema` parameter for the same guarantee.

---

## 2. `findings[]` — single inline review comment

Each element of `findings[]` is one inline GitHub review comment the
poster will create. The shape is identical between Codex and OpenRouter.

### 2.1 Schema (normative)

```jsonc
{
  "title": "🔴 [P1] path/to/file.py:42 short description",   // see §2.3
  "body":  "Markdown body posted as the comment.",
  "confidence_score": 0.85,                                   // number in [0.0, 1.0] | null
  "priority": 1,                                              // integer 0..3 | null
  "side": "RIGHT",                                            // "RIGHT" | "LEFT" | null
  "code_location": {
    "absolute_file_path": "path/to/file.py",
    "line_range": { "start": 42, "end": 44 }
  }
}
```

### 2.2 Field contract

| Field                                | Type                | Required | Nullable | Notes |
|--------------------------------------|---------------------|----------|----------|-------|
| `title`                              | `string`            | yes      | no       | Severity-emoji + `[P0..P3]` prefix per §2.3. Used as the comment's first line. |
| `body`                               | `string`            | yes      | no       | Markdown. May contain a single ` ```suggestion ` block for §2.4 inline suggestions. |
| `confidence_score`                   | `number \| null`    | yes      | yes      | Per-finding confidence. `null` = "no score". |
| `priority`                           | `integer \| null`   | yes      | yes      | `0`/`1`/`2`/`3` mapping to `P0`/`P1`/`P2`/`P3` (§2.3). `null` = unknown. |
| `side`                               | `string \| null`    | yes      | yes      | `"RIGHT"` for additions/context (default), `"LEFT"` only when anchoring on a deleted line. `null` = poster falls back to `"RIGHT"`. Enum-enforced on the wire. |
| `code_location.absolute_file_path`   | `string`            | yes      | no       | Repo-relative path (the field name retains "absolute" for legacy parity, but the poster treats it as repo-relative — see §6.1). |
| `code_location.line_range.start`     | `integer`           | yes      | no       | 1-based line number on the chosen `side`. |
| `code_location.line_range.end`       | `integer`           | yes      | no       | 1-based, `>= start`. Equal to `start` for single-line anchors; multi-line anchors set `end > start` (see §2.4). |

The local Python parser (`cli/core/models.py::ReviewFinding.from_mapping`)
adds two tolerances on top of the strict wire schema:

- `side` may be omitted entirely when reading historical fixtures
  written before the field existed; the parser substitutes `null` and
  the poster defaults to `"RIGHT"`.
- `line_range.end` may be inferred from `line_range.start` when missing
  (`end := start`), again only when reading legacy data. Wire output
  always carries both fields.

### 2.3 Severity model

Severity travels in **two redundant places** so a posted comment renders
correctly even when the structured JSON is stripped:

1. **Title prefix.** A severity emoji + `[P0..P3]` tag at the start of
   `title`. Required by `prompts/review.md` §"Format".

   | Tag    | Emoji | Meaning                                                                |
   |--------|-------|------------------------------------------------------------------------|
   | `[P0]` | `🔴`  | Drop-everything blocker. Universal issue, no input assumptions needed. |
   | `[P1]` | `🔴`  | Urgent. Should ship in the next cycle.                                 |
   | `[P2]` | `🟡`  | Normal. Fix eventually.                                                |
   | `[P3]` | `⚪`  | Nit / nice-to-have.                                                    |

2. **Numeric `priority` field.** The same severity as an integer:
   `0`/`1`/`2`/`3`. `null` is reserved for "model declined to assign".

The bake-off scoring rubric (`baseline-methodology.md` §4.4) excludes
`priority == 3` (P3) from the FP-rate denominator and the overlap
numerator so nit-discipline regressions are visible separately. Any
schema change that removes `priority` would invalidate that scoring
gate.

### 2.4 File / line anchor model

Anchors target the **post-image** of the diff by default
(`side == "RIGHT"`):

- **Single-line anchor** — `start == end`. The most common shape.
- **Multi-line range anchor** — `end > start`. Required when the body
  carries a ` ```suggestion ` replacement block; the range covers the
  span of lines the suggestion replaces. Single-line range anchors are
  collapsed to `start == end` even if the model emits the same value
  for both.
- **Deletion anchor** — `side == "LEFT"`, `start`/`end` index into the
  base-image (the pre-diff file). The poster requires this for
  comments about removed code; emitting `"RIGHT"` on a deletion-only
  hunk produces an invalid anchor and is rejected by AC 3 enforcement.

The anchor engine (`cli/review/anchor_engine.py`) is the single point
that validates a finding's anchor against the live diff hunks; this
schema only defines what the model is allowed to emit.

### 2.5 Required fields under strict mode

OpenRouter strict mode requires every property to appear in `required`
and every object to set `additionalProperties: false`. That is why the
schema in `cli/core/models.py` lists `side` and `code_location` as
`required` even though both are nullable / structurally fixed: the
strictness check is on **presence**, not on non-null content. Drop a
property from `required` and the upstream provider will reject the
response — silently regressing the bake-off until the next live run.

---

## 3. `carried_forward[]` — re-adjudicated prior comments

When a review run resumes from a prior state (continuation case in
`baseline-methodology.md` §1.3), the model reuses still-applicable
prior inline comments instead of re-emitting them as new findings.
Carried-forward comments do not count toward new-finding totals
(`baseline-methodology.md` §2.1, label `CARRY`).

### 3.1 Schema (normative)

```jsonc
{
  "comment_id":       "string",   // GitHub review-comment id of the prior comment being kept
  "current_evidence": "string"    // ≤ 1 sentence: why it still applies at head_sha
}
```

### 3.2 Field contract

| Field              | Type     | Required | Nullable | Notes |
|--------------------|----------|----------|----------|-------|
| `comment_id`       | `string` | yes      | no       | The opaque GitHub id (e.g. `"PRRC_kwDO..."`). Stringly typed because GitHub mixes integer and node ids. |
| `current_evidence` | `string` | yes      | no       | Short rationale that the bug still exists at the new `head_sha`. Empty string is invalid. |

A `carried_forward[]` entry is **not** an inline comment; the poster
does not re-create it. It only signals to the dedupe and review-state
layers that the prior comment id should be kept open. The fact that it
sits in the same envelope as `findings[]` is what lets a single
structured-output call cover both new findings and continuation logic.

---

## 4. Per-run artifact wrapper

The bake-off harness wraps the raw `ReviewRunResult` from §1 in a
runner-side envelope that records *how* the run was produced. The
wrapper shape is identical between `codex.json` and `openrouter.json`;
only the `run` discriminator changes.

```jsonc
{
  "run":                "codex" | "openrouter",   // discriminator
  "pr_id":              "dotcms-core-35509",      // matches eval/sample-prs.yaml id
  "head_sha":           "89f68df8...",            // commit reviewed
  "model":              "anthropic/claude-opus-4.7",  // OpenRouter slug, or Codex model id
  "reasoning_effort":   "medium",                 // "minimal" | "low" | "medium" | "high"
  "web_search_mode":    "disabled",               // "disabled" | "cached" | "live"
  "prior_review_state": null | { /* opaque */ },  // continuation input, if any
  "review_run_result":  { /* §1 envelope */ } | null,
  "posted": {
    "summary_id":       "IC_kwDO..." | null,
    "inline_ids":       [ "PRRC_kwDO...", ... ],
    "posting_outcome": {
      "batch_submitted":      0,
      "per_comment_fallback": 0,
      "skipped_after_422":    0
    }
  },
  "capture": {
    "captured_at":    "2026-05-07T14:22:00Z" | null,
    "command":        "python -m eval.run_codex_baseline --pr-id ...",
    "notes":          "free-form explanation, e.g. 'Live capture skipped: GITHUB_TOKEN not set'",
    "runner_version": "1",
    "status":         "captured" | "pending"
  },
  "source": {                                     // mirror-only block; see §5
    "eval_artifact":  "eval/runs/dotcms-core-35509/codex.json",
    "mirrored_by":    "eval.mirror_codex_baseline",
    "produced_by":    "eval.run_codex_baseline"
  }
}
```

`review_run_result` is `null` only when `capture.status == "pending"`
(no live run yet, schema-conforming placeholder). Once
`capture.status == "captured"`, `review_run_result` MUST be a fully
valid §1 envelope.

---

## 5. Codex-baseline mirror under `tests/fixtures/baseline/`

`tests/fixtures/baseline/<pr-id>/codex-findings.json` is a **byte-for-byte
mirror** of the corresponding `eval/runs/<eval-pr-id>/codex.json` after
id translation (`dotcms-core-35509` → `pr-35509`). The mirror is
written by `eval/mirror_codex_baseline.py` and pinned by
`tests/test_baseline_fixtures.py` (`--check` mode). The wrapper shape
in §4 is what tests assert against; the fixture is what makes the
parity bake-off replayable from a fresh git clone.

The mirror adds the `source` block so a reader of the fixture can find
the upstream artifact without grepping. That block does **not** appear
in `eval/runs/<pr_id>/codex.json` itself.

---

## 6. Codex baseline ↔ OpenRouter diff

The two paths emit the **same** §1 envelope and the **same** §2/§3
finding shapes. The differences below are the only documented
divergences; any new divergence must be recorded here as a row before
it is merged.

### 6.1 Field-by-field diff

| Field path                                       | Codex baseline                                                        | OpenRouter (new)                                                       | Diffable? |
|--------------------------------------------------|-----------------------------------------------------------------------|------------------------------------------------------------------------|-----------|
| `findings[]`                                     | array of `ReviewFinding`                                              | array of `ReviewFinding`                                               | ✓ identical |
| `findings[].title`                               | `string`, severity-prefixed                                           | `string`, severity-prefixed                                            | ✓ identical |
| `findings[].body`                                | `string`, markdown                                                    | `string`, markdown                                                     | ✓ identical |
| `findings[].confidence_score`                    | `number \| null`                                                      | `number \| null`                                                       | ✓ identical |
| `findings[].priority`                            | `integer \| null`                                                     | `integer \| null`                                                      | ✓ identical |
| `findings[].side`                                | not present in pre-Sub-AC-3.1 captures; lenient parser substitutes `null` | `"RIGHT" \| "LEFT" \| null`, **required** in strict schema           | ⚠ parser tolerates absence on read; OpenRouter writes always include it |
| `findings[].code_location.absolute_file_path`    | repo-relative path (despite the name)                                 | repo-relative path                                                     | ✓ identical (name preserved for diffability; renaming would be a breaking schema change) |
| `findings[].code_location.line_range.start`/`end`| 1-based integers                                                      | 1-based integers                                                       | ✓ identical |
| `carried_forward[]`                              | array of `{comment_id, current_evidence}`                             | array of `{comment_id, current_evidence}`                              | ✓ identical |
| `overall_correctness` / `overall_explanation` / `overall_confidence_score` | identical                                            | identical                                                              | ✓ identical |
| Wrapper `run`                                    | `"codex"`                                                             | `"openrouter"`                                                         | discriminator only |
| Wrapper `model`                                  | Codex model id, e.g. `"gpt-5.4"`                                      | OpenRouter slug, e.g. `"anthropic/claude-opus-4.7"`                    | identifier-space differs; type stays `string` |
| Wrapper `reasoning_effort`                       | `"minimal" \| "low" \| "medium" \| "high"`                            | `"minimal" \| "low" \| "medium" \| "high"`                             | ✓ identical |
| Wrapper `web_search_mode`                        | `"disabled" \| "cached" \| "live"`                                    | `"disabled" \| "cached" \| "live"`                                     | ✓ identical |
| Wrapper `posted.posting_outcome`                 | `{batch_submitted, per_comment_fallback, skipped_after_422}`          | `{batch_submitted, per_comment_fallback, skipped_after_422}`           | ✓ identical |
| Wrapper `prior_review_state`                     | opaque continuation blob or `null`                                    | opaque continuation blob or `null`                                     | ✓ identical |
| Wrapper `capture`                                | `{captured_at, command, notes, runner_version, status}`               | same shape; `command` references `run_openrouter` instead of `run_codex_baseline` | ✓ shape identical, command string differs |
| Wrapper `source` (fixture mirror only)           | `{eval_artifact, mirrored_by, produced_by}` referencing `eval.run_codex_baseline` | same shape; `produced_by` references the OpenRouter runner | ✓ shape identical |

Net: at the **finding level** (the only level scored by AC 1 / AC 2),
the two schemas are identical. Wrapper-level differences are limited
to identifier strings (`run`, `model`, `command`, `produced_by`) so a
flat field-by-field `diff` of two wrappers shows divergence only on
those four lines plus the actual content.

### 6.2 Why the field names match exactly

Field renaming would force the bake-off labeling rubric
(`baseline-methodology.md` §2.2) to maintain a translation table.
Because the rubric keys off `path`, `line_start`, `line_end`,
`severity`, the JSON-schema field names are kept stable across the
swap even when a more descriptive name (e.g. `repo_relative_path`)
would be preferable. Renames are deferred to a future major-version
schema bump that re-baselines Codex too.

### 6.3 Drift policy

A schema drift is allowed only when:

1. The new field/value space is **additive** (existing fixtures still
   parse), or
2. The drift is recorded as a new row in §6.1 **and** the bake-off
   harness emits both shapes for at least one transitional release
   so historical artifacts remain comparable.

A non-additive drift that is not recorded here invalidates every
prior bake-off score and forces a full re-label of the sample.

---

## 7. Where each rule lives in code

| Rule                                              | File                                          |
|---------------------------------------------------|-----------------------------------------------|
| Wire schema (strict, OpenRouter `response_format`) | `cli/core/models.py::REVIEW_OUTPUT_SCHEMA`     |
| Single-finding wire schema                         | `cli/core/models.py::REVIEW_FINDING_SCHEMA`    |
| Carried-forward wire schema                        | `cli/core/models.py::CARRIED_FORWARD_COMMENT_SCHEMA` |
| Local lenient parser (legacy fixture tolerance)    | `cli/core/models.py::ReviewFinding.from_mapping`, `ReviewRunResult.from_payload` |
| Strict-payload validator                           | `cli/core/models.py::validate_review_payload`  |
| Anchor validity (post-parse)                       | `cli/review/anchor_engine.py`                  |
| Inline-comment posting (anchor → GitHub)           | `cli/review/posting.py`                        |
| Codex baseline capture                             | `eval/run_codex_baseline.py`                   |
| Codex baseline mirror to fixtures                  | `eval/mirror_codex_baseline.py`                |
| Per-PR artifact wrapper layout                     | `eval/baseline-methodology.md` §4.2 (this doc supersedes for field types) |

If any of those files change a wire-visible field, this document is the
first thing to update — before the test fixtures, before the scoring
rubric, before the action.yml input doc.
