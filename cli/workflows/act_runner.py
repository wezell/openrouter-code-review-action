"""Act-mode entrypoint that drives the aider wrapper from action inputs.

Sub-AC 7.2.3 — this module is the bridge between the GitHub Action's
configuration surface (the OpenRouter model config + ``act_instructions``
+ secret) and the ``aider`` CLI wrapper in :mod:`cli.clients.aider_client`.

Responsibilities
----------------
* Pull the act-mode model slug, OpenRouter API key, repo path and
  streaming preference straight from a :class:`ReviewConfig` so callers
  do not have to re-plumb action inputs.
* Invoke :class:`AiderClient` once against the checked-out PR branch.
* Surface stdout, stderr, exit code, and an applied-edit summary in a
  single :class:`ActAgentResult` so the workflow layer can build a
  reply, log a structured audit trail, and decide whether to commit.

Non-goals
---------
* The runner does **not** stage, commit, or push. ``EditWorkflow`` owns
  git state. Aider is launched with ``--no-auto-commits`` so its edits
  land on disk and stop there.
* The runner does **not** parse aider stdout for semantic content. The
  applied-edit summary is computed from ``aider``'s own
  ``Applied edit to <path>`` lines (cheap, deterministic) — this is a
  human-readable summary only; authoritative diffs come from the git
  layer.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..clients.aider_client import (
    AiderClient,
    AiderExecutionError,
    AiderResult,
    DEFAULT_OPENROUTER_API_BASE,
)
from ..core.config import ReviewConfig
from ..core.exceptions import ConfigurationError

# Aider emits one ``Applied edit to <path>`` line per file it actually
# wrote. We extract those for the human-readable summary surfaced in
# the PR reply. The regex is anchored to the start of a line so prose
# mentions of "applied edit" inside a model message do not contaminate
# the summary.
_APPLIED_EDIT_RE = re.compile(
    r"(?im)^[ \t]*applied edit to[ \t]+(?P<path>[^\r\n]+?)[ \t]*$",
)

# Hard cap on the captured stream tail we surface back to the workflow
# layer. Aider runs can emit verbose chains of thought; the GitHub
# inline-comment limit is 65k chars and we already host the prompt and
# git diff alongside the agent output, so the tail keeps the budget.
_OUTPUT_TAIL_CHAR_LIMIT = 4_000


@dataclass(frozen=True)
class ActAgentResult:
    """Outcome of one aider act-mode invocation.

    The fields here are intentionally close to ``AiderResult`` plus a
    pre-computed ``applied_files`` summary so the workflow layer does
    not have to re-parse the stream.
    """

    returncode: int
    stdout: str
    stderr: str
    command: tuple[str, ...]
    cwd: Path
    applied_files: tuple[str, ...] = field(default_factory=tuple)
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and self.error_message is None

    def render_summary(self, *, max_output_chars: int = _OUTPUT_TAIL_CHAR_LIMIT) -> str:
        """Return a human-readable single-string summary of this run.

        The summary is intended for inclusion in a PR reply: it
        opens with a status line (``aider exited with status N``), the
        applied-edit list, then the tail of stdout/stderr trimmed to
        ``max_output_chars``. Trimming is mid-tail-aware: stderr is kept
        in full when present so model-side errors are not silently
        swallowed by stdout volume.
        """

        status = (
            f"aider exited with status {self.returncode}"
            if self.error_message is None
            else f"aider failed: {self.error_message}"
        )
        if self.applied_files:
            file_lines = "\n".join(f"- {path}" for path in self.applied_files)
            applied_block = "Applied edits:\n" + file_lines
        else:
            applied_block = "Applied edits: (none)"

        body_parts: list[str] = [status, "", applied_block]
        tail = _truncate_streams(
            stdout=self.stdout,
            stderr=self.stderr,
            limit=max_output_chars,
        )
        if tail:
            body_parts.extend(["", tail])
        return "\n".join(body_parts).rstrip() + "\n"


class ActModeRunner:
    """Run the aider wrapper for act mode, driven by :class:`ReviewConfig`.

    The runner is stateless beyond the constructor-bound ``ReviewConfig``
    + ``AiderClient`` pair. Tests inject a fake ``AiderClient`` to avoid
    spawning real subprocesses.
    """

    def __init__(
        self,
        config: ReviewConfig,
        *,
        aider_client: AiderClient | None = None,
    ) -> None:
        self.config = config
        self._aider_client = aider_client or AiderClient(
            api_key=config.openrouter_api_key or None,
            api_base=DEFAULT_OPENROUTER_API_BASE,
        )

    @property
    def model(self) -> str:
        """Return the OpenRouter act-mode model slug from config."""
        return self.config.act_model

    def execute(
        self,
        prompt: str,
        *,
        files: Sequence[str | Path] | None = None,
        repo_path: str | Path | None = None,
        extra_args: Iterable[str] | None = None,
        timeout: float | None = None,
    ) -> ActAgentResult:
        """Invoke aider once and return a structured ``ActAgentResult``.

        Parameters
        ----------
        prompt:
            Natural-language instructions to hand to aider as
            ``--message``. The caller (typically ``EditWorkflow``)
            already composes this from review-comment context + the
            ``act_instructions`` action input.
        files:
            Repo-relative paths aider should add to its chat context.
            Pass ``None`` (the default) to let aider auto-discover.
        repo_path:
            Override the repo root. Defaults to
            ``config.resolved_repo_root`` so callers can rely on the
            action's ``GITHUB_WORKSPACE``-derived setting.
        extra_args:
            Forwarded verbatim to ``aider`` argv. Use sparingly — the
            wrapper already covers the common knobs.
        timeout:
            Subprocess timeout in seconds. ``None`` waits indefinitely
            (the GitHub Actions runner enforces its own job-level
            timeout, so leaving this open is safe by default).
        """

        self._require_act_mode()
        message = (prompt or "").strip()
        if not message:
            raise ConfigurationError("Act-mode prompt is empty; refusing to invoke aider")

        cwd = Path(repo_path).expanduser().resolve() if repo_path else self.config.resolved_repo_root

        try:
            aider_result = self._aider_client.run(
                model=self.model,
                message=message,
                repo_path=cwd,
                files=files,
                # Aider's auto-commit logic conflicts with EditWorkflow's
                # snapshot-then-commit flow. We disable it so edits land
                # on disk and the host workflow handles git.
                auto_commit=False,
                stream=self.config.stream_output,
                extra_args=extra_args,
                # Forward stdout/stderr to the parent process so the
                # GitHub Actions runner sees aider's progress live, but
                # *also* capture them by leaving stdout/stderr=None so
                # subprocess.run captures them. We pick capture so we
                # can post the tail back to the PR reply; the streaming
                # path lives inside aider via --stream/--no-stream.
                check=False,
                timeout=timeout,
            )
        except AiderExecutionError as exc:
            applied = tuple(_extract_applied_files(exc.stdout))
            self._tee_streams(exc.stdout, exc.stderr)
            return ActAgentResult(
                returncode=exc.returncode if exc.returncode is not None else 1,
                stdout=exc.stdout,
                stderr=exc.stderr,
                command=tuple(),
                cwd=cwd,
                applied_files=applied,
                error_message=str(exc),
            )

        self._tee_streams(aider_result.stdout, aider_result.stderr)
        return _result_from_aider(aider_result)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_act_mode(self) -> None:
        if self.config.mode != "act":
            raise ConfigurationError(
                "ActModeRunner invoked while config.mode is "
                f"{self.config.mode!r}; expected 'act'."
            )
        if not self.model.strip():
            raise ConfigurationError(
                "Act model is empty. Set act.model in the in-repo "
                "OpenRouter config file (e.g. .openrouter-review.yml)."
            )
        if not self.config.openrouter_api_key.strip():
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required for act mode but is empty."
            )

    def _tee_streams(self, stdout: str, stderr: str) -> None:
        """Forward captured aider streams to the host process.

        Aider's own ``--stream`` flag forwards token deltas to its
        stdout in real time; once :class:`AiderClient` finishes we still
        re-emit the captured strings here so non-streaming runs (and
        any operator who scrapes the GitHub Actions log after the fact)
        see the full output. ``shutil.copyfileobj``-style chunking is
        unnecessary at this size — the captures are already strings.
        """
        if stdout:
            sys.stdout.write(stdout if stdout.endswith("\n") else stdout + "\n")
        if stderr:
            sys.stderr.write(stderr if stderr.endswith("\n") else stderr + "\n")
        # Flush so the GitHub Actions log sees the data immediately even
        # if the process exits seconds later.
        sys.stdout.flush()
        sys.stderr.flush()


def _result_from_aider(aider_result: AiderResult) -> ActAgentResult:
    return ActAgentResult(
        returncode=aider_result.returncode,
        stdout=aider_result.stdout,
        stderr=aider_result.stderr,
        command=aider_result.command,
        cwd=aider_result.cwd,
        applied_files=tuple(_extract_applied_files(aider_result.stdout)),
        error_message=None,
    )


def _extract_applied_files(stdout: str) -> list[str]:
    """Pull the deduplicated list of files aider says it applied edits to."""
    if not stdout:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _APPLIED_EDIT_RE.finditer(stdout):
        path = match.group("path").strip().strip('"').strip("'")
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _truncate_streams(*, stdout: str, stderr: str, limit: int) -> str:
    """Return a combined-stream tail capped at ``limit`` characters.

    stderr wins ties: when both streams are non-empty we always keep
    stderr in full (to surface aider/litellm errors) and trim stdout to
    the residual budget. The combined block is rendered as labelled
    fenced sections so a downstream Markdown comment renders cleanly.
    """

    if not stdout and not stderr:
        return ""

    sections: list[str] = []
    residual = max(limit, 0)
    err_clean = (stderr or "").strip()
    if err_clean:
        section = "stderr:\n" + _tail(err_clean, residual)
        sections.append(section)
        residual = max(residual - len(section), 0)
    out_clean = (stdout or "").strip()
    if out_clean and residual > 0:
        section = "stdout:\n" + _tail(out_clean, residual)
        sections.append(section)
    elif out_clean and not sections:
        # ``residual`` is zero only when ``limit`` itself was zero or
        # negative. Surface a tiny stub so callers can still see *some*
        # signal rather than an empty block.
        sections.append("stdout:\n" + _tail(out_clean, 256))
    return "\n\n".join(sections)


def _tail(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return "… (truncated)\n" + text[-limit:]


