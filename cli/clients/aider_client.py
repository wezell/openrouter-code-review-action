"""Aider invocation wrapper for the act-mode code-edit pipeline.

This module owns Sub-AC 7.2.1: build the ``aider`` CLI command (model, API
base, repo path, target files) and execute it as a subprocess against the
checked-out PR branch.

Design notes
------------
* The wrapper is intentionally narrow. It does **not** drive the agentic
  loop — that is aider's job — and it does **not** stage / commit / push
  on its own. The workflow layer (``edit_workflow.py``) is responsible
  for git operations after aider returns. This keeps the contract small:
  *build args, run subprocess in repo_path, surface returncode/output*.
* Aider routes through OpenRouter when the model slug carries the
  ``openrouter/`` prefix. We accept either an OpenRouter slug (e.g.
  ``anthropic/claude-opus-4.7``) or a pre-prefixed slug
  (``openrouter/anthropic/claude-opus-4.7``); the wrapper normalises to
  the prefixed form before passing ``--model`` to aider.
* The OpenRouter API key flows in via ``--api-key openrouter=<key>``
  rather than environment so the secret never has to be exported into
  the subprocess shell — aider's CLI accepts the per-provider override
  directly.
* The wrapper is subprocess-runner-agnostic: callers can swap
  ``subprocess.run`` for a fake in tests by passing ``runner=``.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ..core.exceptions import CodexReviewError

# Aider expects OpenRouter-routed slugs to carry an ``openrouter/`` prefix.
# See https://aider.chat/docs/llms/openrouter.html — the prefix tells
# aider's litellm layer which provider adapter to use.
_OPENROUTER_PREFIX = "openrouter/"

# Default base URL aider should use when reaching OpenRouter. Aider reads
# this from ``OPENROUTER_API_BASE`` (or the ``--openai-api-base`` flag);
# we keep it overridable so a self-hosted gateway can be substituted.
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Aider's binary name on $PATH. Overridable via the ``AIDER_BINARY`` env
# var so a uv-managed virtualenv can pin a specific shim.
DEFAULT_AIDER_BINARY = "aider"


class AiderExecutionError(CodexReviewError):
    """Raised when the aider subprocess fails (non-zero exit, missing binary)."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class AiderResult:
    """Outcome of a single aider subprocess invocation."""

    returncode: int
    stdout: str
    stderr: str
    command: tuple[str, ...]
    cwd: Path

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


# Subprocess runner contract: we accept anything that quacks like
# ``subprocess.run`` so tests can inject a fake. The runtime default is
# the real ``subprocess.run`` from the stdlib.
_SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


def _normalize_model_slug(model: str) -> str:
    """Return the model slug with the ``openrouter/`` prefix attached.

    OpenRouter slugs are typically ``<vendor>/<model>`` (e.g.
    ``anthropic/claude-opus-4.7``). Aider needs ``openrouter/<vendor>/<model>``
    to dispatch through the OpenRouter adapter. If the caller already
    pre-prefixed the slug we leave it alone — idempotent.
    """

    cleaned = (model or "").strip()
    if not cleaned:
        raise ValueError("Aider model slug must be a non-empty string")
    if cleaned.startswith(_OPENROUTER_PREFIX):
        return cleaned
    return f"{_OPENROUTER_PREFIX}{cleaned}"


def build_aider_command(
    *,
    model: str,
    message: str,
    files: Sequence[str | Path] | None = None,
    api_base: str | None = DEFAULT_OPENROUTER_API_BASE,
    api_key: str | None = None,
    auto_commit: bool = True,
    stream: bool = True,
    extra_args: Iterable[str] | None = None,
    aider_binary: str | None = None,
) -> list[str]:
    """Build the ``aider`` CLI argv for a single non-interactive turn.

    Parameters
    ----------
    model:
        OpenRouter slug. The ``openrouter/`` prefix is added if missing.
    message:
        The prompt aider will run as a one-shot. Equivalent to typing
        the message in the interactive REPL and exiting.
    files:
        Files aider should add to its chat context. Paths are passed
        through verbatim — the caller is responsible for choosing
        files that exist relative to ``repo_path``.
    api_base:
        OpenRouter base URL forwarded as ``--openai-api-base``. Pass
        ``None`` to omit (aider will then use its env-derived default).
    api_key:
        OpenRouter API key. When supplied, forwarded as
        ``--api-key openrouter=<key>`` so the value never has to be
        exported into the subprocess environment.
    auto_commit:
        When ``True`` (the default) aider commits each successful edit.
        When ``False`` we pass ``--no-auto-commits`` so the workflow
        layer can stage and commit on aider's behalf.
    stream:
        Toggle aider's token streaming (``--stream`` / ``--no-stream``).
    extra_args:
        Additional argv to append verbatim — useful for opt-in flags
        the wrapper does not yet model. Tests rely on this hook to
        exercise unknown-flag pass-through without rebuilding the
        wrapper.
    aider_binary:
        Override the executable name. Falls back to the ``AIDER_BINARY``
        env var, then to ``aider``.
    """

    binary = (
        aider_binary
        or os.environ.get("AIDER_BINARY", "").strip()
        or DEFAULT_AIDER_BINARY
    )

    argv: list[str] = [
        binary,
        "--model",
        _normalize_model_slug(model),
        "--message",
        message,
        # Non-interactive defaults: never prompt, never repaint.
        "--yes-always",
        "--no-pretty",
        "--no-show-model-warnings",
    ]

    if api_base:
        argv.extend(["--openai-api-base", api_base])

    if api_key:
        # Aider's per-provider key syntax avoids exporting the secret to
        # the subprocess shell. The token never leaves argv.
        argv.extend(["--api-key", f"openrouter={api_key}"])

    argv.append("--stream" if stream else "--no-stream")
    argv.append("--auto-commits" if auto_commit else "--no-auto-commits")

    if extra_args:
        argv.extend(str(arg) for arg in extra_args)

    if files:
        argv.extend(str(path) for path in files)

    return argv


def _build_subprocess_env(
    *,
    base_env: dict[str, str] | None,
    api_key: str | None,
    api_base: str | None,
) -> dict[str, str]:
    """Compose the env var bag for the aider subprocess.

    We deliberately do **not** export ``OPENROUTER_API_KEY`` when the
    caller supplied an explicit ``api_key`` — that path uses argv-level
    ``--api-key`` instead. We *do* set ``OPENROUTER_API_BASE`` because
    aider's litellm-backed router reads it during model validation
    *before* parsing ``--openai-api-base``.
    """

    env = dict(base_env if base_env is not None else os.environ)
    if api_base:
        env.setdefault("OPENROUTER_API_BASE", api_base)
    if api_key and not env.get("OPENROUTER_API_KEY"):
        # Belt-and-braces: aider's pre-flight may probe the env var even
        # when ``--api-key`` is supplied. Safe to set since the subprocess
        # is short-lived and inherits no descendants.
        env["OPENROUTER_API_KEY"] = api_key
    return env


class AiderClient:
    """Subprocess wrapper around the ``aider`` CLI.

    The client is stateless: every ``run()`` call assembles a fresh
    argv and shells out. Streaming output is forwarded directly to the
    caller's stdout/stderr so the GitHub Actions runner sees aider's
    progress in real time (parity with AC-6 streaming for review mode).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = DEFAULT_OPENROUTER_API_BASE,
        aider_binary: str | None = None,
        runner: _SubprocessRunner | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._api_base = (api_base or "").strip() or None
        self._aider_binary = (aider_binary or "").strip() or None
        self._runner: _SubprocessRunner = runner or subprocess.run  # type: ignore[assignment]

    @property
    def api_base(self) -> str | None:
        return self._api_base

    def build_command(
        self,
        *,
        model: str,
        message: str,
        files: Sequence[str | Path] | None = None,
        auto_commit: bool = True,
        stream: bool = True,
        extra_args: Iterable[str] | None = None,
    ) -> list[str]:
        """Return the argv aider would be invoked with — useful for dry-runs."""
        return build_aider_command(
            model=model,
            message=message,
            files=files,
            api_base=self._api_base,
            api_key=self._api_key,
            auto_commit=auto_commit,
            stream=stream,
            extra_args=extra_args,
            aider_binary=self._aider_binary,
        )

    def run(
        self,
        *,
        model: str,
        message: str,
        repo_path: str | Path,
        files: Sequence[str | Path] | None = None,
        auto_commit: bool = True,
        stream: bool = True,
        extra_args: Iterable[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        check: bool = True,
        stdout: IO[Any] | int | None = None,
        stderr: IO[Any] | int | None = None,
    ) -> AiderResult:
        """Execute aider as a subprocess against ``repo_path``.

        ``repo_path`` MUST be the working directory of the checked-out
        PR branch. We resolve and validate it before spawning so we
        fail fast rather than letting aider operate on an unrelated
        cwd. ``check=True`` raises :class:`AiderExecutionError` on a
        non-zero exit; ``check=False`` returns the result regardless.
        """

        cwd = Path(repo_path).expanduser().resolve()
        if not cwd.exists() or not cwd.is_dir():
            raise AiderExecutionError(
                f"Aider repo_path does not exist or is not a directory: {cwd}"
            )

        argv = self.build_command(
            model=model,
            message=message,
            files=files,
            auto_commit=auto_commit,
            stream=stream,
            extra_args=extra_args,
        )

        # Resolve the binary explicitly so a missing tool produces a
        # clear AiderExecutionError instead of a stack trace from
        # subprocess. ``shutil.which`` honours ``PATH`` from the
        # composed env so a uv-installed aider in ``.venv/bin`` is
        # picked up correctly.
        process_env = _build_subprocess_env(
            base_env=env,
            api_key=self._api_key,
            api_base=self._api_base,
        )

        resolved_binary = shutil.which(argv[0], path=process_env.get("PATH"))
        if resolved_binary is None:
            raise AiderExecutionError(
                f"Aider binary '{argv[0]}' not found on PATH. "
                "Install aider-chat (e.g. `uv pip install aider-chat`) before running act mode."
            )
        argv[0] = resolved_binary

        try:
            completed = self._runner(
                argv,
                cwd=str(cwd),
                env=process_env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                capture_output=stdout is None and stderr is None,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:  # pragma: no cover - guarded above
            raise AiderExecutionError(f"Aider binary not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AiderExecutionError(
                f"Aider subprocess timed out after {timeout}s",
            ) from exc

        result = AiderResult(
            returncode=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            command=tuple(argv),
            cwd=cwd,
        )

        if check and not result.succeeded:
            raise AiderExecutionError(
                f"Aider exited with status {result.returncode}",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        return result
