# Changelog

All notable changes to `dotcms/openrouter-code-review-action` are documented in
this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Tags and GitHub Releases are produced by [Release Please](https://github.com/googleapis/release-please)
in no-PR mode; the floating `v1` tag is moved to the latest release inside the
major.

## [1.0.0] — 2026-05-07

Initial release of the OpenRouter-routed code review and act action. This is a
**hard fork** of dotCMS's prior Codex-based PR review action: the wire protocol,
agentic loop, and all action inputs have been replaced. There is **no
backwards-compatible migration path** — consumers must update their workflow
YAML when moving from the Codex action to `dotcms/openrouter-code-review-action@v1`.

### Added

- **OpenRouter-routed review path.** All LLM calls for review are issued
  directly against OpenRouter's chat completions API with schema-enforced JSON
  output. Findings, anchors, and the carry-forward adjudication round-trip
  through a single model contract.
- **Aider-based act mode.** When a trusted commenter posts `/codex
  <instructions>`, the action shells out to [aider](https://aider.chat) with the
  configured `act.model`, lets aider apply minimal diffs, and pushes the result
  to the PR branch. The custom agentic loop from the Codex action is gone.
- **In-repo model config (`.openrouter-review.yml`).** Review and act models,
  reasoning effort, and web-search mode are declared in a single file at the
  consuming repo's root. Swapping providers (e.g.
  `anthropic/claude-opus-4.7` → `openai/gpt-5.4` →
  `google/gemini-2.5-pro`) is a one-line edit. No code, workflow, or action
  input change required. The path is overridable via the `config_path` input
  or `OPENROUTER_REVIEW_CONFIG` env var.
- **Default model: `anthropic/claude-opus-4.7`** for both review and act. One
  model per call — no mid-run swaps and no fallback chain.
- **Token streaming to GitHub Actions logs.** OpenRouter SSE deltas reach the
  step log line-by-line via `PYTHONUNBUFFERED=1` and unbuffered Python.
  Toggle with `stream_agent_messages` (`0` / `1`).
- **Web-search modes (`disabled` / `cached` / `live`)** mapped onto OpenRouter's
  `:online` slug variant at call time. `live` is the default.
- **Schema-enforced inline comment parsing.** Inline review payloads are
  produced from a single JSON schema turn and anchored to validated diff lines
  before submission, so GitHub doesn't return 422.
- **Review continuation across runs.** The PR summary embeds the previously
  reviewed head SHA in hidden metadata; review state is cached per (repo, PR,
  model, SHA) and the next run resumes the prior thread, scoped to the SHA
  delta. Force-pushes that break ancestry, missing caches, or unrestorable
  threads fall back to a fresh full review.
- **Carry-forward dedup.** Unresolved action-authored review threads are passed
  to the structured-output turn to avoid reposting; the model separately
  re-adjudicates which prior findings are still relevant. The summary reports
  new and still-relevant counts separately.
- **`/codex` command surface** — `/codex <instructions>` for arbitrary edits,
  `/codex address comments` (and natural-language variants) for unresolved
  review threads. Bare `/codex` is intentionally ignored.
- **`author_association` allowlist enforced in-process.** Act mode validates
  `allowed_commenter_associations` (default `MEMBER,OWNER,COLLABORATOR`) at
  startup and exits early on invalid values, so auth-policy drift surfaces
  immediately rather than after a partial run.
- **uv-managed dependencies.** `uv sync` for local development; `requirements.txt`
  is the action's runtime install surface (`pip install -r`).
- **Release Please (no-PR mode)** for automated tagging, GitHub Release
  creation, and `v1` floating tag updates. Manual release-as override available
  via the workflow dispatch input.

### Changed

- **Wire protocol: Codex SDK → OpenRouter HTTP.** All model calls now route
  through OpenRouter. Direct vendor SDKs (`openai`, `anthropic`, etc.) and
  alternative routers are not used.
- **Agentic act loop: custom Codex loop → aider subprocess.** The action no
  longer drives its own agentic edit loop; aider owns the edit/test/commit
  cycle inside the action's checkout.
- **Action coordinates.** Published as
  `dotcms/openrouter-code-review-action@v1`. The prior Codex action remains at
  its original path for repos that have not migrated.
- **Default model.** Now `anthropic/claude-opus-4.7` for both review and act,
  routed through OpenRouter.
- **Required permissions matrix** is documented per mode (review needs
  `pull-requests: write`; act additionally needs `contents: write`). Fork PRs
  require an explicit PAT for act-mode pushes.

### Removed (Breaking)

- **Codex SDK runtime.** No Codex client, no Codex-managed conversation state,
  no Codex-shaped tool transcripts. Workflows that pinned the Codex action's
  SHA must repoint at `dotcms/openrouter-code-review-action@v1`.
- **Direct vendor SDK fallbacks.** Calls do not fall through to OpenAI,
  Anthropic, Google, or any other vendor SDK if OpenRouter is unreachable —
  the run fails loudly instead of silently switching providers.
- **Mid-run model fallback chain.** A single model handles a given call. If
  the configured model is unhealthy, edit `.openrouter-review.yml` and rerun.
- **Backwards-compatible action inputs.** Inputs from the prior Codex action
  are not honored. The supported input set is enumerated in `action.yml` and
  the README's Inputs table.

### Deprecated

- **`openai_api_key` input.** Accepted for parity with prior installations but
  ignored by the OpenRouter review/act path. Will be dropped in a future major.
  Set `openrouter_api_key` instead.

### Security

- **Act-mode allowlist enforced server-side.** The `author_association` check
  runs inside the action process, not just in the workflow `if:` guard, so a
  misconfigured workflow can't accidentally let untrusted commenters trigger
  edits.
- **Fork-PR push posture documented.** Default `GITHUB_TOKEN` cannot push to
  fork branches; act mode is expected to run only on same-repo branches or with
  an explicit PAT (`secrets.REPO_ACCESS_TOKEN`).
- **Aider edits stay on the PR branch.** Act mode commits and pushes to the PR
  head branch; cross-branch edits are out of scope for v1.
- **API keys are passed via env (`OPENROUTER_API_KEY`)**, never written to disk
  or echoed into logs.

### Migration from the prior Codex action

See [`MIGRATION.md`](./MIGRATION.md) for the full input/output mapping and
step-by-step upgrade guide. Quick version:

1. Replace the `uses:` reference with `dotcms/openrouter-code-review-action@v1`.
2. Replace `openai_api_key` / Codex-era inputs with `openrouter_api_key`
   (sourced from `secrets.OPENROUTER_API_KEY`).
3. Add `.openrouter-review.yml` at the repo root (or set `config_path`); the
   defaults in the README work as a starting point.
4. Audit job permissions against the table above (`pull-requests: write` for
   review; `contents: write` additionally for act).
5. Re-pin any cache keys / consumer scripts that referenced Codex-era artifacts
   — review continuation now keys on (repo, PR, model, SHA).

[1.0.0]: https://github.com/dotcms/openrouter-code-review-action/releases/tag/v1.0.0
