"""Review state schema and PR identifier keying convention.

This module is the **dedicated types/schema layer** for prior-review state in
the OpenRouter code-review action. It replaces the implicit
``CODEX_HOME``-cached-thread model with an explicit, JSON-serialisable
contract that names the PR identifier, the per-finding record, and the
top-level envelope persisted between review runs.

Concept map
-----------

* ``PriorReviewKey`` — the *PR identifier keying convention*. A
  ``(repository, pr_number, review_model, reviewed_head_sha)`` tuple that
  derives a stable cache key string of the form
  ``openrouter-review-v1-<repo>-pr-<n>-<model>-<sha>``. Two runs with the
  same key reference the same persisted state; the SHA component is what
  lets a force-push (which breaks ancestry) miss the cache cleanly and
  fall back to a fresh review.
* ``PersistedReviewFinding`` — one row in the *findings* table. Captures
  the diff anchor (path / line / end-line), the rendered finding content
  (title / body / severity / suggestion), and the **GitHub comment IDs
  GitHub gave us when we posted the finding** so the next run can
  correlate "still applicable" carry-forward decisions back to live
  comment threads.
* ``PriorReviewState`` — the persisted envelope. Versioned schema header
  (``schema_version``), the identifying key, the reviewed HEAD SHA, the
  list of findings, the set of carried-forward thread IDs, and an optional
  summary issue-comment ID. Round-trips to/from JSON without loss.

Why a fresh module instead of extending ``cli/core/models.py``?
``models.py`` describes the **model output contract** that lives for a
single LLM call (``ReviewFinding``, ``ReviewRunResult``). This module
describes the **persistence contract** that lives across runs. They share
some shapes but evolve at different rates: bumping the persisted schema
must not silently break a model-output validator, and adding a new
model-output field must not invalidate cached state on disk. Keeping them
in separate files makes the boundary explicit.

Why ``schema_version`` is a string instead of an int? Because future
migrations may want named tags (``"v1.1"``, ``"v2-pre"``) and pinning to a
string up front avoids a downstream rewrite.

Why ``openrouter-review-v1`` and not the legacy ``dotbot-review-v1``?
Cache keys are also the GitHub Actions cache-restore keys; flipping the
prefix gives the new action a clean cache namespace and guarantees a
post-migration run cannot accidentally pick up a stale review thread cache.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .exceptions import ReviewContractError

# ---------------------------------------------------------------------------
# Versioning + naming conventions
# ---------------------------------------------------------------------------

#: Persisted-state schema version. Bump when the JSON shape changes in a way
#: that older readers can't safely interpret. Readers should refuse unknown
#: versions rather than silently mis-parsing.
REVIEW_STATE_SCHEMA_VERSION = "v1"

#: Cache-key prefix for the new OpenRouter-routed action. Distinct from the
#: legacy resume-state prefix so a post-migration run cannot read
#: stale thread state.
OPENROUTER_REVIEW_CACHE_PREFIX = "openrouter-review-v1"

#: Marker prepended to the summary issue comment so future runs can find
#: it via a substring scan. Replaces ``"dotbot code review:"``.
OPENROUTER_REVIEW_SUMMARY_MARKER = "OpenRouter Code Review:"

#: HTML-comment metadata block embedded inside the summary comment body. The
#: payload carries the reviewed HEAD SHA so the next run can compute the
#: SHA-delta without re-walking the entire repo.
_SUMMARY_METADATA_RE = re.compile(r"<!--\s*openrouter-review-meta\s+({.*?})\s*-->")


def render_review_summary_metadata(
    reviewed_head_sha: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Render the HTML-comment metadata block embedded in the summary comment.

    The format is intentionally minimal — a single JSON object carrying the
    reviewed HEAD SHA plus (optionally) the model and reasoning effort that
    produced the review, so the renderer and parser can never disagree on
    field naming. ``parse_reviewed_head_sha`` is the inverse operation; older
    summaries without ``model``/``reasoning_effort`` still parse.
    """
    payload_data: dict[str, Any] = {"reviewed_head_sha": reviewed_head_sha}
    if model is not None:
        payload_data["model"] = model
    if reasoning_effort is not None:
        payload_data["reasoning_effort"] = reasoning_effort
    payload = json.dumps(
        payload_data,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"<!-- openrouter-review-meta {payload} -->"


def parse_reviewed_head_sha(summary_body: str) -> str | None:
    """Extract a previously embedded ``reviewed_head_sha`` from a comment body.

    Returns ``None`` (rather than raising) for missing or malformed
    metadata so summary-comment scans are tolerant of operator edits.
    """
    match = _SUMMARY_METADATA_RE.search(summary_body)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    reviewed_head_sha = payload.get("reviewed_head_sha")
    if not isinstance(reviewed_head_sha, str):
        return None
    normalized = reviewed_head_sha.strip()
    return normalized or None


# ---------------------------------------------------------------------------
# PR identifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorReviewKey:
    """Identifies a prior review record by its (repo, PR, model, SHA) tuple.

    Two runs with the same :class:`PriorReviewKey` describe the same
    persisted state. ``reviewed_head_sha`` is part of the identity on
    purpose — when the PR HEAD moves to a *new* SHA, the key changes, the
    cache misses, and the workflow either reuses prior state under a
    secondary restore key or falls back to a fresh full review.
    """

    repository: str
    pr_number: int
    review_model: str
    reviewed_head_sha: str

    def __post_init__(self) -> None:
        if not self.repository or "/" not in self.repository:
            raise ReviewContractError("PriorReviewKey.repository must be in 'owner/repo' form")
        if not isinstance(self.pr_number, int) or self.pr_number <= 0:
            raise ReviewContractError("PriorReviewKey.pr_number must be a positive integer")
        if not isinstance(self.review_model, str) or not self.review_model.strip():
            raise ReviewContractError("PriorReviewKey.review_model must be a non-empty string")
        if not isinstance(self.reviewed_head_sha, str) or not self.reviewed_head_sha.strip():
            raise ReviewContractError("PriorReviewKey.reviewed_head_sha must be a non-empty string")

    def cache_key(self) -> str:
        """Render the deterministic, sanitized cache key string.

        Output shape::

            openrouter-review-v1-<repo>-pr-<n>-<model>-<sha>

        Components that contain characters outside ``[A-Za-z0-9._-]`` are
        collapsed to dashes so the key is safe to use as a GitHub Actions
        cache key, a filesystem directory name, etc.
        """
        sanitized_repository = _sanitize_cache_component(self.repository)
        sanitized_model = _sanitize_cache_component(self.review_model)
        sanitized_sha = _sanitize_cache_component(self.reviewed_head_sha)
        return (
            f"{OPENROUTER_REVIEW_CACHE_PREFIX}-"
            f"{sanitized_repository}-pr-{self.pr_number}-"
            f"{sanitized_model}-{sanitized_sha}"
        )

    def with_sha(self, new_head_sha: str) -> PriorReviewKey:
        """Return a copy of this key with a different ``reviewed_head_sha``.

        Used when computing the *current* run's key from a prior key —
        repository, pr_number, and model stay the same; only the SHA
        component flips so the cache key changes when the PR HEAD moves.
        """
        return PriorReviewKey(
            repository=self.repository,
            pr_number=self.pr_number,
            review_model=self.review_model,
            reviewed_head_sha=new_head_sha,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "pr_number": self.pr_number,
            "review_model": self.review_model,
            "reviewed_head_sha": self.reviewed_head_sha,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PriorReviewKey:
        if not isinstance(payload, Mapping):
            raise ReviewContractError("PriorReviewKey payload must be a mapping")
        try:
            return cls(
                repository=_require_str(payload, "repository"),
                pr_number=_require_int(payload, "pr_number"),
                review_model=_require_str(payload, "review_model"),
                reviewed_head_sha=_require_str(payload, "reviewed_head_sha"),
            )
        except ReviewContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewContractError(f"Invalid PriorReviewKey payload: {exc}") from exc


# ---------------------------------------------------------------------------
# Finding record
# ---------------------------------------------------------------------------


# Severity is intentionally an open string so the action can adopt new
# tiers without a code change, but we declare the canonical four so the
# review-posting layer has something stable to map model-emitted
# ``priority`` ints to.
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"
KNOWN_SEVERITIES: frozenset[str] = frozenset(
    {SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL}
)


@dataclass(frozen=True)
class PersistedReviewFinding:
    """One finding row persisted between review runs.

    Captures three independent pieces of state:

    1. **Diff anchor** — ``path`` / ``start_line`` / ``end_line`` /
       ``side``. This is how the next run correlates a finding back to a
       diff line on the new HEAD; if the line no longer exists on the
       current HEAD, the finding is treated as "carried forward" rather
       than re-emitted.
    2. **Content** — ``title`` / ``body`` / ``severity`` / ``suggestion``.
       What we showed the human reviewer. ``severity`` is the
       human-readable analog of the model's numeric ``priority`` (kept
       open as a string so we can adopt new tiers without a schema bump).
    3. **GitHub comment IDs** — ``inline_comment_id`` for inline review
       comments (POST /pulls/{n}/comments), ``general_comment_id`` for
       remapped findings posted as issue comments (POST
       /issues/{n}/comments), ``review_thread_id`` for the GraphQL node
       ID of the thread the inline comment belongs to. At least one of
       these is set when the finding was actually published.

    Frozen on purpose — once persisted, a finding is immutable. Updates
    happen by writing a new ``PriorReviewState`` envelope.
    """

    path: str
    start_line: int
    end_line: int
    title: str
    body: str
    severity: str = SEVERITY_MEDIUM
    suggestion: str | None = None
    side: str = "RIGHT"
    priority: int | None = None
    confidence_score: float | None = None
    inline_comment_id: int | None = None
    general_comment_id: int | None = None
    review_thread_id: str | None = None
    finding_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ReviewContractError("PersistedReviewFinding.path must be a non-empty string")
        if not isinstance(self.start_line, int) or self.start_line <= 0:
            raise ReviewContractError(
                "PersistedReviewFinding.start_line must be a positive integer"
            )
        if not isinstance(self.end_line, int) or self.end_line < self.start_line:
            raise ReviewContractError("PersistedReviewFinding.end_line must be >= start_line")
        if not isinstance(self.title, str):
            raise ReviewContractError("PersistedReviewFinding.title must be a string")
        if not isinstance(self.body, str):
            raise ReviewContractError("PersistedReviewFinding.body must be a string")
        if not isinstance(self.severity, str) or not self.severity.strip():
            raise ReviewContractError("PersistedReviewFinding.severity must be a non-empty string")
        if self.side not in ("LEFT", "RIGHT"):
            raise ReviewContractError("PersistedReviewFinding.side must be 'LEFT' or 'RIGHT'")
        if self.suggestion is not None and not isinstance(self.suggestion, str):
            raise ReviewContractError("PersistedReviewFinding.suggestion must be a string or null")
        if self.priority is not None and not isinstance(self.priority, int):
            raise ReviewContractError("PersistedReviewFinding.priority must be an integer or null")
        if self.confidence_score is not None and not isinstance(
            self.confidence_score, (int, float)
        ):
            raise ReviewContractError(
                "PersistedReviewFinding.confidence_score must be a number or null"
            )
        if self.inline_comment_id is not None and not isinstance(self.inline_comment_id, int):
            raise ReviewContractError(
                "PersistedReviewFinding.inline_comment_id must be an integer or null"
            )
        if self.general_comment_id is not None and not isinstance(self.general_comment_id, int):
            raise ReviewContractError(
                "PersistedReviewFinding.general_comment_id must be an integer or null"
            )
        if self.review_thread_id is not None and not isinstance(self.review_thread_id, str):
            raise ReviewContractError(
                "PersistedReviewFinding.review_thread_id must be a string or null"
            )

    @property
    def is_published(self) -> bool:
        """Whether this finding was actually posted to GitHub on the prior run."""
        return (
            self.inline_comment_id is not None
            or self.general_comment_id is not None
            or self.review_thread_id is not None
        )

    @property
    def comment_id(self) -> int | None:
        """Convenience: the inline comment ID if present, else the general one."""
        return (
            self.inline_comment_id
            if self.inline_comment_id is not None
            else (self.general_comment_id)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "title": self.title,
            "body": self.body,
            "severity": self.severity,
            "suggestion": self.suggestion,
            "side": self.side,
            "priority": self.priority,
            "confidence_score": self.confidence_score,
            "inline_comment_id": self.inline_comment_id,
            "general_comment_id": self.general_comment_id,
            "review_thread_id": self.review_thread_id,
            "finding_fingerprint": self.finding_fingerprint,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PersistedReviewFinding:
        if not isinstance(payload, Mapping):
            raise ReviewContractError("PersistedReviewFinding payload must be a mapping")
        try:
            start_line = _require_int(payload, "start_line")
            end_line_raw = payload.get("end_line")
            end_line = _require_int(payload, "end_line") if end_line_raw is not None else start_line
            return cls(
                path=_require_str(payload, "path"),
                start_line=start_line,
                end_line=end_line,
                title=_optional_str(payload, "title", default=""),
                body=_optional_str(payload, "body", default=""),
                severity=_optional_str(payload, "severity", default=SEVERITY_MEDIUM)
                or SEVERITY_MEDIUM,
                suggestion=_optional_nullable_str(payload, "suggestion"),
                side=_optional_str(payload, "side", default="RIGHT") or "RIGHT",
                priority=_optional_nullable_int(payload, "priority"),
                confidence_score=_optional_nullable_number(payload, "confidence_score"),
                inline_comment_id=_optional_nullable_int(payload, "inline_comment_id"),
                general_comment_id=_optional_nullable_int(payload, "general_comment_id"),
                review_thread_id=_optional_nullable_str(payload, "review_thread_id"),
                finding_fingerprint=_optional_nullable_str(payload, "finding_fingerprint"),
            )
        except ReviewContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewContractError(f"Invalid PersistedReviewFinding payload: {exc}") from exc


# ---------------------------------------------------------------------------
# Top-level persisted state envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorReviewState:
    """Persisted review-state envelope keyed by :class:`PriorReviewKey`.

    This is the JSON document the action writes to the runner-temp cache
    after a review run and reads back at the start of the next run. It
    intentionally carries everything the next run needs to make its
    re-review-only-the-SHA-delta decision *without* having to call the
    GitHub API again:

    * ``schema_version`` — version pin so future readers can detect
      incompatible payloads.
    * ``key`` — the :class:`PriorReviewKey` that produced this state.
    * ``findings`` — the per-finding records (anchor + content +
      published comment IDs).
    * ``carried_forward_comment_ids`` — string IDs of prior comments the
      current run re-adjudicated as still applicable. Stored as strings
      because GitHub returns string node IDs from GraphQL even though the
      REST IDs are ints.
    * ``summary_comment_id`` — int ID of the run-summary issue comment so
      the next run can update-in-place rather than spamming a new comment.
    * ``generated_at`` — opaque timestamp string (ISO-8601 if present, but
      the parser does not enforce a format).
    """

    schema_version: str
    key: PriorReviewKey
    findings: tuple[PersistedReviewFinding, ...] = ()
    carried_forward_comment_ids: tuple[str, ...] = ()
    summary_comment_id: int | None = None
    generated_at: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_STATE_SCHEMA_VERSION:
            raise ReviewContractError(
                "Unsupported review-state schema version: "
                f"{self.schema_version!r} (expected {REVIEW_STATE_SCHEMA_VERSION!r})"
            )
        if self.summary_comment_id is not None and not isinstance(self.summary_comment_id, int):
            raise ReviewContractError(
                "PriorReviewState.summary_comment_id must be an integer or null"
            )
        if self.generated_at is not None and not isinstance(self.generated_at, str):
            raise ReviewContractError("PriorReviewState.generated_at must be a string or null")

    @property
    def reviewed_head_sha(self) -> str:
        """Convenience: the SHA the prior review ran against."""
        return self.key.reviewed_head_sha

    @property
    def cache_key(self) -> str:
        """Convenience: the deterministic cache key for this state."""
        return self.key.cache_key()

    @property
    def published_inline_comment_ids(self) -> tuple[int, ...]:
        """All inline comment IDs from prior posts (skipping unpublished findings)."""
        return tuple(
            finding.inline_comment_id
            for finding in self.findings
            if finding.inline_comment_id is not None
        )

    @property
    def published_general_comment_ids(self) -> tuple[int, ...]:
        """All issue-comment IDs from prior remap fallbacks."""
        return tuple(
            finding.general_comment_id
            for finding in self.findings
            if finding.general_comment_id is not None
        )

    @property
    def published_review_thread_ids(self) -> tuple[str, ...]:
        """All review-thread node IDs from prior inline comments."""
        return tuple(
            finding.review_thread_id
            for finding in self.findings
            if finding.review_thread_id is not None
        )

    @classmethod
    def empty(
        cls,
        key: PriorReviewKey,
        *,
        generated_at: str | None = None,
    ) -> PriorReviewState:
        """Build an empty state envelope for a key (zero findings)."""
        return cls(
            schema_version=REVIEW_STATE_SCHEMA_VERSION,
            key=key,
            findings=(),
            carried_forward_comment_ids=(),
            summary_comment_id=None,
            generated_at=generated_at,
            notes=(),
        )

    def with_findings(
        self,
        findings: Iterable[PersistedReviewFinding],
    ) -> PriorReviewState:
        """Return a copy with a different finding list (everything else same)."""
        return PriorReviewState(
            schema_version=self.schema_version,
            key=self.key,
            findings=tuple(findings),
            carried_forward_comment_ids=self.carried_forward_comment_ids,
            summary_comment_id=self.summary_comment_id,
            generated_at=self.generated_at,
            notes=self.notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key": self.key.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
            "carried_forward_comment_ids": list(self.carried_forward_comment_ids),
            "summary_comment_id": self.summary_comment_id,
            "generated_at": self.generated_at,
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Render the state as a JSON string suitable for atomic disk writes."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PriorReviewState:
        if not isinstance(payload, Mapping):
            raise ReviewContractError("PriorReviewState payload must be a mapping")

        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise ReviewContractError("PriorReviewState.schema_version must be a non-empty string")
        if schema_version != REVIEW_STATE_SCHEMA_VERSION:
            # Refuse rather than silently best-effort-parse: a future
            # writer may have added required fields the current reader
            # doesn't understand, and silently dropping them would cause
            # subtle review-continuation bugs.
            raise ReviewContractError(
                "Unsupported review-state schema version: "
                f"{schema_version!r} (expected {REVIEW_STATE_SCHEMA_VERSION!r})"
            )

        key_raw = payload.get("key")
        if not isinstance(key_raw, Mapping):
            raise ReviewContractError("PriorReviewState.key must be a mapping")
        key = PriorReviewKey.from_mapping(key_raw)

        findings_raw = payload.get("findings", [])
        if not isinstance(findings_raw, Sequence) or isinstance(findings_raw, (str, bytes)):
            raise ReviewContractError("PriorReviewState.findings must be an array")
        findings: list[PersistedReviewFinding] = []
        for index, item in enumerate(findings_raw):
            if not isinstance(item, Mapping):
                raise ReviewContractError(f"PriorReviewState.findings[{index}] must be an object")
            findings.append(PersistedReviewFinding.from_mapping(item))

        carried_raw = payload.get("carried_forward_comment_ids", [])
        if not isinstance(carried_raw, Sequence) or isinstance(carried_raw, (str, bytes)):
            raise ReviewContractError(
                "PriorReviewState.carried_forward_comment_ids must be an array"
            )
        carried_forward: list[str] = []
        for index, item in enumerate(carried_raw):
            if not isinstance(item, str):
                raise ReviewContractError(
                    f"PriorReviewState.carried_forward_comment_ids[{index}] must be a string"
                )
            carried_forward.append(item)

        summary_id_raw = payload.get("summary_comment_id")
        if summary_id_raw is not None and not isinstance(summary_id_raw, int):
            raise ReviewContractError(
                "PriorReviewState.summary_comment_id must be an integer or null"
            )

        generated_at_raw = payload.get("generated_at")
        if generated_at_raw is not None and not isinstance(generated_at_raw, str):
            raise ReviewContractError("PriorReviewState.generated_at must be a string or null")

        notes_raw = payload.get("notes", [])
        if not isinstance(notes_raw, Sequence) or isinstance(notes_raw, (str, bytes)):
            raise ReviewContractError("PriorReviewState.notes must be an array")
        notes: list[str] = []
        for index, item in enumerate(notes_raw):
            if not isinstance(item, str):
                raise ReviewContractError(f"PriorReviewState.notes[{index}] must be a string")
            notes.append(item)

        return cls(
            schema_version=schema_version,
            key=key,
            findings=tuple(findings),
            carried_forward_comment_ids=tuple(carried_forward),
            summary_comment_id=summary_id_raw,
            generated_at=generated_at_raw,
            notes=tuple(notes),
        )

    @classmethod
    def from_json(cls, raw: str) -> PriorReviewState:
        """Parse a JSON string previously produced by :meth:`to_json`."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReviewContractError(f"PriorReviewState JSON decode failed: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ReviewContractError("PriorReviewState JSON root must be an object")
        return cls.from_mapping(payload)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_cache_component(value: str) -> str:
    """Collapse non-alphanumeric runs to single dashes for cache-safe keys."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "unknown"


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        raise ReviewContractError(f"Missing required field {key!r}")
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ReviewContractError(f"Field {key!r} must be a non-empty string")
    return value


def _require_int(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise ReviewContractError(f"Missing required field {key!r}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReviewContractError(f"Field {key!r} must be an integer")
    return value


def _optional_str(payload: Mapping[str, Any], key: str, *, default: str) -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ReviewContractError(f"Field {key!r} must be a string")
    return value


def _optional_nullable_str(payload: Mapping[str, Any], key: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ReviewContractError(f"Field {key!r} must be a string or null")
    return value


def _optional_nullable_int(payload: Mapping[str, Any], key: str) -> int | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReviewContractError(f"Field {key!r} must be an integer or null")
    return value


def _optional_nullable_number(payload: Mapping[str, Any], key: str) -> float | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewContractError(f"Field {key!r} must be a number or null")
    return float(value)


__all__ = [
    "OPENROUTER_REVIEW_CACHE_PREFIX",
    "OPENROUTER_REVIEW_SUMMARY_MARKER",
    "REVIEW_STATE_SCHEMA_VERSION",
    "KNOWN_SEVERITIES",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "SEVERITY_HIGH",
    "SEVERITY_CRITICAL",
    "PersistedReviewFinding",
    "PriorReviewKey",
    "PriorReviewState",
    "parse_reviewed_head_sha",
    "render_review_summary_metadata",
]
