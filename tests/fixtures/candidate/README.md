# OpenRouter candidate findings fixtures (Sub-AC 2.3)

This tree mirrors `eval/runs/dotcms-core-<NUMBER>/openrouter.json` into
`tests/fixtures/candidate/pr-<NUMBER>/openrouter-findings.json`,
normalised against the schema documented in
[`eval/findings-schema.md`](../../../eval/findings-schema.md).

The mirror is produced by:

```
python -m eval.normalize_openrouter_findings
```

Re-run with `--check` in CI to fail loudly on drift.

The pair fixture under `tests/fixtures/baseline/<pr-id>/codex-findings.json`
holds the Codex baseline; the parity-evidence comparison in Sub-AC 2.4
diffs the two trees PR-by-PR.

Files written here MUST conform to the §1 envelope (validated through
`cli.core.models.validate_review_payload`) so a schema regression in the
action surfaces at fixture-regen time, not at scoring time.
