# Overlap Matching Algorithm Specification

> **Sub-AC 3.1 / 3.2.** Normative spec for the overlap-matching step
> that diffs Codex baseline findings against OpenRouter candidate
> findings. Implementation: `eval/overlap_score.py` (core matching
> engine, Sub-AC 3.2). Consumers: `eval/compare_findings.py` (parity
> report) and `eval/score_codex_baseline.py` (AC-gate scorer).
> Methodology context: `eval/baseline-methodology.md` §4.5.
>
> This document is the **single source of truth** for what counts as a
> match. When the implementation and this document disagree, this
> document wins and the implementation must be patched.

---

## 1. Scope

The overlap matcher consumes two normalized finding lists for a single
PR:

* **Baseline** — Codex `ReviewRunResult.findings[]`, mirrored under
  `tests/fixtures/baseline/<pr-id>/codex-findings.json` (Sub-AC 2.2).
* **Candidate** — OpenRouter `ReviewRunResult.findings[]`, mirrored
  under `tests/fixtures/candidate/<pr-id>/openrouter-findings.json`
  (Sub-AC 2.3).

It produces, deterministically:

* A **match set** — a list of `(codex_finding, openrouter_finding,
  line_distance)` triples.
* The **codex-only** complement — Codex findings that did not match.
* The **openrouter-only** complement — OpenRouter findings that did
  not match.
* Per-match **severity-equality** flag.

The matcher does **not** decide "is this finding TP/FP". That stays
human-judged per `baseline-methodology.md` §2 and §3. The matcher's
only job is "is this the *same anchor* on the *same path* in both
runs". That is a necessary precondition for the human reviewer's
`match_to_other_run` field but is not sufficient for it.

---

## 2. Inputs (per finding)

A normalized finding carries five fields the matcher reads:

| Field           | Source                                          | Required |
|-----------------|-------------------------------------------------|----------|
| `path`          | `code_location.absolute_file_path`              | yes      |
| `line_start`    | `code_location.line_range.start`                | yes      |
| `line_end`      | `code_location.line_range.end` (defaults to `line_start` when absent) | yes |
| `severity`      | Extracted from `priority` (preferred) or the `[P0..P3]` title prefix (fallback) — see `findings-schema.md` §2.3 | yes (may be `null`) |
| `finding_id`    | `<side>.f<NNNN>` (assigned by the normalizer; stable per run) | yes |

A finding **without** a usable `path` or **without** both line
endpoints is **non-matchable**: it never participates in a match and
is automatically routed to the side-only bucket. This is consistent
with the schema's anchor-validity rule (a finding the poster cannot
anchor cannot be compared either).

---

## 3. Match predicate (when do two findings match)

For a Codex finding `c` and an OpenRouter finding `o`, the predicate
`MATCHES(c, o)` is true **iff all three** rules below hold. The rules
are evaluated in order; the first failing rule is sufficient to
declare a non-match.

### 3.1 Same path (exact string equality)

```
c.path == o.path
```

* Comparison is **byte-exact**. No normalization (no case-folding, no
  symlink resolution, no `./` collapse, no Windows-vs-POSIX
  rewriting). The fixtures store repo-relative POSIX paths per
  `findings-schema.md` §2.2; any drift is a fixture bug, not a
  matcher bug.
* `null`/empty path on either side disqualifies the pair (§2,
  non-matchable).

### 3.2 Line-range proximity (file/line tolerance)

Let `d = line_distance(c.line_range, o.line_range)` be defined as:

* `0` if the two closed integer intervals `[c.line_start, c.line_end]`
  and `[o.line_start, o.line_end]` **overlap by ≥ 1 line**, i.e.
  `max(c.line_start, o.line_start) ≤ min(c.line_end, o.line_end)`.
* Otherwise the **gap** between the intervals — the number of lines
  separating the closer endpoints (`b_start - a_end` when `a_end <
  b_start`, symmetric on the other side). Always non-negative.
* `None` (undefined) if either range is missing endpoints.

The predicate passes when:

```
d != None  AND  d <= TOLERANCE
```

`TOLERANCE` is **3 lines** by default
(`compare_findings.LINE_PROXIMITY_TOLERANCE`), matching
`baseline-methodology.md` §4.5 rule 2 ("line ranges overlap or are
within ± 3 lines"). The CLI exposes `--tolerance N` for sensitivity
analysis but a non-default tolerance must be called out explicitly in
any bake-off report it produces.

Rationale: the model picks anchor lines off a few different mental
heuristics ("first line of the bug" vs "site of the symptom" vs
"start of the suggestion replacement"). ±3 absorbs that without
collapsing legitimately distinct nearby bugs (the same function can
contain two real bugs four or more lines apart).

### 3.3 Same finding (the human gate)

The matcher itself **does not enforce** `baseline-methodology.md`
§4.5 rule 3 ("the reviewer judges them to describe the same
underlying bug"). That is a labeling decision recorded in
`match_to_other_run` during the §2 labeling pass. The matcher's
output is therefore an **upper bound** on the true match set: every
real cross-run match is in the matcher's output, but the human
labeler may demote a matcher-asserted pair to "different bugs at
nearby lines" (recorded as two separate findings, not a match).

The parity report (`compare_findings.py`) is the **machine-only**
view of this upper bound; the labeled scorer
(`score_codex_baseline.py`) consumes the human-judged `labels[]` and
is what the AC gates score against.

---

## 4. Category equivalence rules

The matcher does **not** key on a `category` field — none exists in
the schema (§2 of `findings-schema.md`). The only attribute used for
*grouping* findings besides `(path, line_range)` is **severity**, and
severity is treated as a comparison annotation, not a match
prerequisite.

Specifically:

| Rule                                  | Behavior                                                              |
|---------------------------------------|-----------------------------------------------------------------------|
| Different severities still match      | A `[P0]` Codex finding and a `[P2]` OpenRouter finding on the same path/lines **do** match. The match record carries `severity_match: false`. |
| `null`/unknown severity still matches | A finding with `severity == null` matches a finding with any severity (or `null`). The match record carries `severity_match: false`. |
| Severity is an annotation, not a key  | Severity influences §6 reporting (severity-equality count, per-bucket deltas) and §5 weighting on tie-breaking, but never disqualifies a candidate pair from matching. |

This is intentional. Severity miscalibration ("Codex calls it P1,
OpenRouter calls it P2") is a real signal we want surfaced *as a
matched pair with `severity_match: false`*, not erased by being
forced into the side-only buckets. `baseline-methodology.md` §3 is
explicit: "Severity disagreement … is still TP if the underlying bug
is real."

### 4.1 Future category extensions

If a future schema revision introduces a finding `category` field
(e.g. `security`, `correctness`, `style`), the matcher will adopt the
following rule and this section will be updated:

* Findings with **identical** category strings match unmodified.
* Findings with **mismatched** category strings still match if §3
  passes, but get `category_match: false` in the report (parallel to
  `severity_match`).
* Findings with `null` category match anything (parallel to severity).

Until that schema field exists, this section's rules ignoring
category are normative.

---

## 5. Severity weighting

The matcher uses a **geometric-only** primary key (line distance) and
does **not** weight severity in pair selection. Severity participates
in the bake-off two ways downstream of matching:

1. **Report annotation.** Each match record carries a
   `severity_match: bool` flag (true iff both sides emit the same
   `P*` tag) and the parity report's
   `overlap.severity_match_count` counts how many matched pairs
   agree on severity.
2. **AC-gate weighting.** `score_codex_baseline.py` excludes
   `priority == 3` (P3) findings from the FP-rate denominator and the
   overlap numerator per `baseline-methodology.md` §4.4. That
   exclusion happens **after** matching, on the labeled scorer's
   view; the matcher itself keeps P3 pairs in `matches` so the
   parity report can surface "OpenRouter emits 3× more P3 nits"
   as a delta even though the gate ignores them.

Severity is **deliberately not** part of the tie-break key (§6.3).
The rationale: tie-breaking on severity introduces a coupling between
the matcher and the severity-extraction heuristic
(`_extract_severity` in `compare_findings.py`), so any drift in how
the title prefix or the `priority` field is parsed would silently
re-order matches. Keeping the tie-break purely on `finding_id`
ordering — which is fixed at normalize time — makes the matcher
output a function of `(path, line_range, finding_id)` only, and that
is what `tests/test_compare_findings.py` pins.

If a future revision wants to bias high-severity findings to win
ties, it must (a) update §6.3 below, (b) update
`_match_findings` to include severity in the sort tuple, and (c)
extend the test suite to pin the new behavior — in that order. Until
all three land, severity is a report annotation, not a sort key.

---

## 6. Algorithm: greedy 1:1 proximity match

Given two finding lists `Codex = [c_1..c_m]` and `OpenRouter =
[o_1..o_n]`, the matcher returns `(matches, codex_only,
openrouter_only)` such that every finding appears in **at most one**
output element.

### 6.1 Build candidate set

For every pair `(c, o)` with `c ∈ Codex`, `o ∈ OpenRouter`:

1. Skip if `c.path` or `o.path` is missing (§2).
2. Skip if `c.path != o.path` (§3.1).
3. Compute `d = line_distance(c, o)` (§3.2). Skip if `d == None` or
   `d > TOLERANCE`.
4. Emit candidate `(d, c.finding_id, o.finding_id, c, o)`.

The candidate set is the full Cartesian product of *path-equal,
in-tolerance* pairs. It is `O(m × n)` in the worst case but in practice
PR finding counts are small (single-digit per side), so this is
trivially fast.

### 6.2 Greedy selection

Sort candidates ascending by the tuple

```
(line_distance, codex_finding_id, openrouter_finding_id)
```

then walk in order. For each candidate `(d, cid, oid, c, o)`:

* If neither `cid` nor `oid` has been used in a prior selection,
  accept the pair: append to `matches`, mark both used.
* Otherwise, drop the candidate and continue.

After the walk, every Codex finding **not** used is in `codex_only`;
every OpenRouter finding **not** used is in `openrouter_only`.

This is a **stable** greedy match (each finding pairs with at most
one partner) and is **deterministic** — given the same two inputs and
tolerance, the output is bit-identical.

### 6.3 Tie-breaking semantics (normative)

When two or more candidate pairs are tied on `line_distance`, the
remaining sort keys break the tie in this strict order:

1. **Codex `finding_id` (lexicographic ascending).** Codex findings
   are emitted in the order the model produced them; pairing the
   *earliest* Codex finding first makes the report stable across
   reorderings of the OpenRouter list.
2. **OpenRouter `finding_id` (lexicographic ascending).** Final
   tiebreaker. With the synthesizer-assigned ids
   (`<side>.f<NNNN>`, fixed-width zero-padded), this is equivalent to
   "earliest OpenRouter finding wins".

The tie-breaker chain is **total** — there is no remaining ambiguity
after step 2 because `(codex_finding_id, openrouter_finding_id)` is
unique per candidate by construction (each Codex finding has one id;
each OpenRouter finding has one id; the candidate is the pair).

Severity is intentionally **excluded** from the tie-break (§5). A
Codex P0 paired against an equidistant OpenRouter P0 vs OpenRouter
P3 will resolve solely by `finding_id` ordering, which means whichever
OpenRouter finding the model emitted *first* in its array wins. This
trades "pair high-severity together" for "matcher output is a pure
function of normalized inputs", and the latter is what makes the
parity report bit-stable across hosts and reruns.

If a future change introduces a non-unique id scheme, this section
must be patched **before** the matcher is run, not after; otherwise
parity reports will diff non-deterministically across hosts.

### 6.4 Worked example

Inputs (single PR, `path` omitted for brevity, all on `src/foo.py`):

```
Codex:
  c.f0001  lines  10-12  P1
  c.f0002  lines  18-20  P2
  c.f0003  lines 100-100 P0

OpenRouter:
  or.f0001 lines  11-11  P1
  or.f0002 lines  19-19  P2
  or.f0003 lines  21-22  P3   # 1 line off c.f0002 (gap of 1)
  or.f0004 lines 200-200 P1
```

Candidate set (after §6.1, tolerance = 3):

| d | codex     | openrouter | note                       |
|---|-----------|------------|----------------------------|
| 0 | c.f0001   | or.f0001   | overlap (10-12 vs 11-11)   |
| 1 | c.f0002   | or.f0003   | gap of 1 (18-20 vs 21-22)  |
| 0 | c.f0002   | or.f0002   | overlap (18-20 vs 19-19)   |

`c.f0003` (line 100) finds no OpenRouter finding within ±3.
`or.f0004` (line 200) finds no Codex finding within ±3.

Sort by `(d, codex_id, openrouter_id)`:

1. `(0, c.f0001, or.f0001)` — accept.
2. `(0, c.f0002, or.f0002)` — accept.
3. `(1, c.f0002, or.f0003)` — drop (`c.f0002` already used).

Result:

* matches: `[(c.f0001, or.f0001, d=0), (c.f0002, or.f0002, d=0)]`
* codex_only: `[c.f0003]`
* openrouter_only: `[or.f0003, or.f0004]`

Note that `or.f0003` is *not* matched even though it is within
tolerance of `c.f0002`, because `c.f0002` was claimed first by the
closer (`d=0`) `or.f0002`. Severity does not enter this decision —
even if `or.f0003` and `c.f0002` had identical severity and
`or.f0002` did not, the closer pair still wins.

---

## 7. Output guarantees

The matcher's contract:

1. **One-to-one.** Every finding appears in exactly one of `matches`,
   `codex_only`, `openrouter_only`. No finding is in two output
   buckets.
2. **Order-independent within tolerance.** Reordering the input
   lists yields the same output (the ordering is recovered by the
   tie-break sort in §6.3).
3. **Idempotent.** Running the matcher twice on the same input
   produces byte-identical output.
4. **Lossless.** Every input finding is accounted for in the output;
   the matcher never silently drops a finding (non-matchable findings
   land in the side-only bucket, see §2).
5. **Path-disjoint.** Findings on different paths can never match,
   regardless of line distance (§3.1).

These guarantees are pinned by `tests/test_compare_findings.py` and
are the contract that `eval/compare_findings.py::_match_findings`
implements.

---

## 8. Relationship to other documents

| Concern                                  | Source of truth                                   |
|------------------------------------------|---------------------------------------------------|
| What `path`, `line_range`, `priority`, `title` mean | `eval/findings-schema.md` §2 |
| When a finding is TP / FP / DUP / CARRY  | `eval/baseline-methodology.md` §2, §3            |
| Aggregate AC gates (≥0.80 overlap, FP-rate parity) | `eval/baseline-methodology.md` §4.4    |
| The matcher's algorithm and tie-break    | **this document** (§3, §6)                        |
| Implementation                            | `eval/overlap_score.py::match_findings` (Sub-AC 3.2) |
| Wrapper for the parity report            | `eval/compare_findings.py::_match_findings`       |
| Tests                                     | `tests/test_overlap_score.py`, `tests/test_compare_findings.py` |
| Parity-report consumer artifacts          | `tests/fixtures/parity_report/*.json`             |

When in doubt, this document is the spec; code and tests follow it,
not the other way around.
