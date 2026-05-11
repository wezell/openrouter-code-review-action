# Contributing

Welcome. This repo is a GitHub Action that runs PR review through
OpenRouter-routed models, with an `act` mode built on aider. The action
is designed so that the **review or act model can be swapped with a
single edit to `.openrouter-review.yml`** — no Python changes — provided
the candidate clears the parity gate documented below.

## Local setup

```bash
uv sync --dev
make hooks       # installs pre-commit
make qa          # fmt, lint, type-check
uv run pytest -q # tests
```

See `AGENTS.md` for module layout, naming, and commit conventions.

## CI overview

CI is defined in `.github/workflows/ci.yml`. Two jobs run on every push
and pull request:

1. **`lint-type-test`** — ruff format/check, mypy, and the full pytest
   suite via `uv run pytest -q`.
2. **`eval-overlap`** — the **parity gate**. Runs the bake-off overlap
   pipeline against the curated sample PRs and uploads
   `eval/parity_report.md` + `eval/parity_report.json` as the
   `parity-report` artifact (30-day retention, uploaded
   `if: always()`).

The two jobs are sequential (`eval-overlap` `needs: lint-type-test`).
A failing parity gate blocks merge the same way a failing test does.

## The parity gate

The parity gate enforces the Seed's `finding_overlap` evaluation
principle: a candidate model run must overlap **≥80%** with the Codex
baseline on the curated 5–10 PR sample before that model can land in
`.openrouter-review.yml`.

| Knob (env / Make var) | Default        | What it does                                                                 |
|-----------------------|----------------|------------------------------------------------------------------------------|
| `OVERLAP_THRESHOLD`   | `0.80`         | Pass bar applied to `OVERLAP_METRIC`.                                        |
| `OVERLAP_METRIC`      | `micro.recall` | Which `eval/overlap_aggregate.json` field the gate checks. Use `macro.recall` to weight every PR equally instead of pooling. |
| `ALLOW_UNMEASURABLE`  | `1`            | While the bake-off sample is still being captured, soft-pass with a warning when metrics are `null`. Flip to `0` to enforce strictly. Soft-pass **never** turns a measured-but-low score into a pass. |

**Why micro-recall and not F1 or macro-recall:** the Codex baseline is
the ground truth for this comparison, so missed Codex findings are the
failure mode. Pooled (micro) recall weights PRs by finding count, which
matches how the Seed phrases the sample-wide overlap target. Macro
recall is emitted alongside but treats one-finding PRs and ten-finding
PRs equally, which over-penalizes small-PR misses. F1 would dilute the
gate with precision noise that is policed separately by the labeled
dataset (`eval/labeled_dataset/`, Sub-AC 2.3.2).

The full rationale, severity-aware metrics table, and step-by-step
report walkthrough live in [`eval/README.md`](./eval/README.md#parity-gate-sub-ac-41--42--43).

### Regenerating the parity report locally

CI and contributors use the same Make targets:

```bash
# Refresh per-PR + aggregate overlap and run the threshold gate.
make eval-overlap

# Build the consolidated Markdown + JSON parity report.
make eval-parity-report

# Read-only equivalents (what CI runs) — drift-check committed artifacts.
make eval-overlap-check
make eval-parity-report-check
```

Outputs:

- `eval/runs/<pr_id>/overlap.json` — per-PR matches with
  precision/recall/F1.
- `eval/overlap_aggregate.json` — sample-wide micro/macro rollup +
  per-severity breakdown.
- `eval/parity_report.md` + `eval/parity_report.json` — consolidated
  human-readable + machine-readable artifacts. **Commit these** when
  you regenerate them; `make eval-parity-report-check` fails CI when
  the committed copy drifts from a fresh rebuild.

### Reading `eval/parity_report.md`

Top-down sections:

1. **Threshold gate banner** — `PASS` / `FAIL` / `PASS — unmeasurable`.
2. **Sample Rollup** — sample size, ready/pending count, micro+macro
   precision/recall/F1.
3. **Per-severity Breakdown** — Codex vs OpenRouter totals, matches,
   recall/precision per P0–P3 + `unknown`. The fastest way to spot a
   regression that's concentrated on critical findings.
4. **Per-PR Details** — one row per sample PR. `Δtotal =
   openrouter_total - codex_total`. Large `Δtotal` with low recall is
   the canonical "verbose but off-topic" candidate-model failure.
5. **Source Artifacts** — pointers back to the per-PR and aggregate
   JSON tiers for programmatic consumers (badges, dashboards).

### When the gate fails

1. Download the `parity-report` artifact from the failing CI run.
2. Open `parity_report.md` and read the banner — it states which metric
   and value tripped the gate.
3. Check the per-severity table. A drop concentrated on P0/P1 almost
   always points at a prompt or model regression on critical findings.
4. Drill into `eval/runs/<pr_id>/overlap.json` for the worst-recall PR
   and inspect `tests/fixtures/parity_report/<pr_id>/` for the
   matched / codex-only / openrouter-only finding lists.
5. If the gate failed because of stale committed artifacts (the
   `Drift-check` step), regenerate locally with `make eval-parity-report`
   and commit.
6. If a force-push broke reviewed-SHA ancestry on a sample PR, the PR
   row will go `unmeasurable`. The action's policy is to fall back to a
   fresh full review; recapture that PR's baselines and re-run the
   pipeline.

## Swapping the review or act model

1. Edit `.openrouter-review.yml` — change the `model` (and optional
   `reasoning_effort`, `web_search_mode`) for the relevant mode.
2. Re-run the bake-off locally with `make eval-overlap` followed by
   `make eval-parity-report`.
3. If the gate passes (`micro.recall ≥ 0.80`), open a PR. The CI
   `eval-overlap` job re-runs the gate on the bake-off sample as the
   merge gate.
4. No Python changes are required for a model swap. If you find
   yourself editing `cli/` to make a swap work, that's a sign the
   abstraction has leaked — please raise it in review.

## Pre-submit checklist

- [ ] `make qa` passes (ruff format, ruff check, mypy).
- [ ] `uv run pytest -q` is green.
- [ ] If you touched anything under `eval/`, `cli/core/model_config.py`,
      `prompts/`, or `.openrouter-review.yml`, run `make eval-overlap`
      and `make eval-parity-report` and commit the regenerated artifacts.
- [ ] If you changed Action inputs or CLI args, update `README.md` and
      `action.yml` examples.
- [ ] Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`,
      `chore:`).
