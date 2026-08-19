"""On-disk persistence for OpenRouter agent conversations.

The Codex SDK stored threads in ``CODEX_HOME`` and the review-resume
flow listed them back through the SDK's app server. The OpenRouter
client speaks plain HTTP, so it owns the equivalent persistence
itself: one JSON file per thread under a single directory.

* Default directory: ``$OPENROUTER_THREAD_DIR`` if set, else
  ``$RUNNER_TEMP/openrouter-review-threads`` (what the composite
  action should cache), else ``~/.openrouter-review/threads``.
* File name: ``openrouter-thread-v1-<repo>-pr-<n>-<model>-<id>.json``
  so ``latest_thread_id`` can scope "resume this PR/model" with a glob
  and consumers never deserialize unrelated threads.

Threads are plain wire-format message lists: they can be replayed to
OpenRouter verbatim on the next run to continue a review conversation.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

THREAD_FILE_PREFIX = "openrouter-thread-v1"
THREAD_SCHEMA_VERSION = 1


class OpenRouterThreadStore:
    """Filesystem-backed store of OpenRouter conversation threads."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @classmethod
    def default_directory(cls) -> Path:
        """Resolve the default thread directory from the environment."""
        override = os.environ.get("OPENROUTER_THREAD_DIR", "").strip()
        if override:
            return Path(override)
        runner_temp = os.environ.get("RUNNER_TEMP", "").strip()
        if runner_temp:
            return Path(runner_temp) / "openrouter-review-threads"
        return Path.home() / ".openrouter-review" / "threads"

    def save(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        *,
        repository: str,
        pr_number: int | None,
        model: str,
    ) -> Path:
        """Persist (or refresh) a thread's message history."""
        if not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "openrouter-thread-v1",
            "schema_version": THREAD_SCHEMA_VERSION,
            "thread_id": thread_id,
            "repository": repository,
            "pr_number": pr_number,
            "model": model,
            "updated_at": datetime.now(UTC).isoformat(),
            "messages": list(messages),
        }
        target = self._thread_path(
            thread_id,
            repository=repository,
            pr_number=pr_number,
            model=model,
        )
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    def load(self, thread_id: str) -> list[dict[str, Any]] | None:
        """Return the stored messages for a thread, or None if absent/corrupt."""
        for path in self._iter_thread_files():
            payload = _read_thread_payload(path)
            if payload is not None and payload.get("thread_id") == thread_id:
                messages = payload.get("messages")
                if isinstance(messages, list):
                    return [message for message in messages if isinstance(message, dict)]
                return None
        return None

    def latest_thread_id(
        self,
        *,
        repository: str,
        pr_number: int | None,
        model: str,
    ) -> str | None:
        """Return the most recently updated thread id for a repo/PR/model."""
        prefix = self._thread_prefix(repository=repository, pr_number=pr_number, model=model)
        candidates: list[tuple[float, str]] = []
        for path in self._iter_thread_files():
            if not path.name.startswith(prefix):
                continue
            payload = _read_thread_payload(path)
            if payload is None:
                continue
            thread_id = payload.get("thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                continue
            candidates.append((path.stat().st_mtime, thread_id))
        if not candidates:
            return None
        return max(candidates)[1]

    def _iter_thread_files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(
            path
            for path in self.directory.iterdir()
            if path.is_file() and path.name.startswith(f"{THREAD_FILE_PREFIX}-")
        )

    def _thread_path(
        self,
        thread_id: str,
        *,
        repository: str,
        pr_number: int | None,
        model: str,
    ) -> Path:
        name = (
            f"{THREAD_FILE_PREFIX}-{_sanitize(repository)}-pr-{pr_number or 0}-"
            f"{_sanitize(model)}-{_sanitize(thread_id)}.json"
        )
        return self.directory / name

    def _thread_prefix(
        self,
        *,
        repository: str,
        pr_number: int | None,
        model: str,
    ) -> str:
        return (
            f"{THREAD_FILE_PREFIX}-{_sanitize(repository)}-pr-{pr_number or 0}-{_sanitize(model)}-"
        )


def _read_thread_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != "openrouter-thread-v1":
        return None
    return payload


def _sanitize(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return sanitized.strip("-") or "unknown"
