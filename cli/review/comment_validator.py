"""Parse and validate review-comment objects against PR diff hunks.

This module owns Sub-AC 3.2 of the inline-anchor-validity acceptance
criterion: take the schema-validated JSON output produced by the
OpenRouter review turn, turn each finding into a typed *comment
candidate*, and verify before submission that the candidate's file path
and line number actually anchor onto the PR diff hunks. Findings that
do not anchor are either remapped onto a usable surface or dropped with
a categorical reason — never posted blindly to GitHub, where the
inline-comment API would otherwise return ``422 Unprocessable Entity``.

Sub-AC 3.3.2 layers a remap/fallback strategy on top of the bare anchor
check. Three fallback rungs are applied in order, and every transition
between rungs is logged so operators can trace why a model finding
ended up where it did:

1. **Snap-to-nearest-valid-line** — handled inside the anchor engine
   (:func:`cli.review.anchor_engine.resolve_range` already collapses
   slightly-off line ranges to the nearest non-blank head-line within
   the file's hunks). The validator simply consumes the result.
2. **Convert to a general PR issue comment** — when the file is not in
   the diff at all (``missing_file_map``) or the patch has no
   anchorable HEAD line (``missing_anchor``), the finding is wrapped in
   a :class:`RemappedComment` carrying the original location text so
   the reviewer's signal is preserved as an issue comment instead of
   silently disappearing.
3. **Drop with logging** — used when even the general-comment fallback
   has nothing useful to post (e.g. caller asked for the strict-drop
   policy, or the finding body is empty so a remapped comment would be
   noise).

Drop / remap categories (mirrored on :class:`CommentValidationResult`
so the caller can surface them in the GH summary comment) are:

``missing_file_map``
    The finding's relative path is not one of the changed files in the
    PR. GitHub will reject any inline comment posted against an
    unchanged file with HTTP 422 ("path is not part of the diff").

``missing_anchor``
    The finding's path matches a changed file, but the requested line
    range does not resolve to any valid head-line in that file's diff
    hunks. This catches model hallucinations that point at lines that
    were never part of the patch.

``empty_content``
    Drop-only category — the finding has no title and no body, so even
    a fallback issue comment would be empty and is suppressed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..core.exceptions import ReviewContractError
from ..core.models import (
    FindingLocation,
    ReviewFinding,
    ReviewRunResult,
    validate_review_payload,
)
from .anchor_engine import Anchor, RangeAnchor, SingleAnchor, resolve_range
from .patch_parser import ParsedPatch, to_relative_path

# ---------------------------------------------------------------------------
# Drop / remap reasons & fallback policy
# ---------------------------------------------------------------------------

# Reasons used for both drops and remaps. ``empty_content`` is drop-only —
# it never makes sense to remap a finding with no body to a general issue
# comment (the resulting comment would be empty noise).
DropReason = Literal["missing_file_map", "missing_anchor", "empty_content"]

# Fallback strategy applied when a finding fails the inline-anchor check.
#
# ``remap_to_general`` (default)
#     Convert non-anchorable findings into general PR issue comments
#     instead of dropping them, so the reviewer's signal survives even
#     when the model points at the wrong file or a deletion-only diff.
#     The poster will surface them as standalone issue comments.
#
# ``drop``
#     Strict mode — drop non-anchorable findings, mirroring the original
#     pre-3.3.2 behaviour. Used by callers that explicitly want zero
#     non-inline output (e.g. tests that exercise the bare anchor path).
FallbackPolicy = Literal["remap_to_general", "drop"]


# Logger callback shape — matches :func:`cli.core.config.make_debug` so
# the workflow can pass its existing debug printer in unchanged.
DebugLogger = Callable[[int, str], None]


def _noop_debug(level: int, message: str) -> None:
    """Default no-op debug sink used when callers don't supply one."""
    return None


@dataclass(frozen=True)
class ValidatedComment:
    """A review finding that has been parsed and anchored to a real diff line.

    The payload-builder consumes these to produce the GitHub inline
    review comment requests. By the time a ``ValidatedComment`` exists
    we have already proven:

    * ``relative_path`` corresponds to a file present in the PR diff
      (so GitHub will accept ``path``);
    * ``anchor`` resolves onto valid head-lines inside one of that
      file's diff hunks (so GitHub will accept ``line`` /
      ``start_line``).
    """

    finding: ReviewFinding
    relative_path: str
    location: FindingLocation
    anchor: Anchor
    has_suggestion: bool

    @property
    def is_range_anchor(self) -> bool:
        return isinstance(self.anchor, RangeAnchor)


@dataclass(frozen=True)
class DroppedComment:
    """A finding that failed pre-submission validation against the PR diff."""

    finding: ReviewFinding
    relative_path: str
    location: FindingLocation
    reason: DropReason


# Reasons used for remap (subset of :data:`DropReason` — only the
# inline-anchor failures, never ``empty_content`` since an empty finding
# can't be remapped to a meaningful issue comment).
RemapReason = Literal["missing_file_map", "missing_anchor"]


@dataclass(frozen=True)
class RemappedComment:
    """A finding remapped from inline → general PR issue comment.

    When a finding cannot anchor onto a real diff line (file not in PR
    diff, or no anchorable HEAD line in the patch), we still want to
    surface the reviewer's signal somewhere visible. ``RemappedComment``
    captures the original finding plus the resolved relative path /
    location text so the poster can format a self-contained general PR
    issue comment that calls out *where* the finding was originally
    aimed.

    The validator never produces a ``RemappedComment`` for an empty
    finding (no title, no body) — those are dropped with
    ``empty_content`` instead so we don't post hollow noise.
    """

    finding: ReviewFinding
    relative_path: str
    location: FindingLocation
    reason: RemapReason


@dataclass(frozen=True)
class CommentValidationResult:
    """Aggregate outcome of validating findings against the PR diff.

    Findings flow through three buckets after the Sub-AC 3.3.2 fallback
    ladder:

    * ``valid`` — anchored cleanly on a real diff line (after
      snap-to-nearest if applicable). Safe to post inline.
    * ``remapped`` — could not anchor inline but has usable content, so
      it is posted as a general PR issue comment instead of being lost.
    * ``dropped`` — neither inline nor general posting will help (empty
      content, or strict-drop policy). Recorded for the summary so the
      operator sees *what* was suppressed.

    All three are preserved in input order so callers can correlate the
    final report back to the original review run.
    """

    valid: list[ValidatedComment] = field(default_factory=list)
    dropped: list[DroppedComment] = field(default_factory=list)
    remapped: list[RemappedComment] = field(default_factory=list)

    # ---- counters -------------------------------------------------------

    @property
    def dropped_missing_file_map(self) -> int:
        return sum(1 for d in self.dropped if d.reason == "missing_file_map")

    @property
    def dropped_missing_anchor(self) -> int:
        return sum(1 for d in self.dropped if d.reason == "missing_anchor")

    @property
    def dropped_empty_content(self) -> int:
        return sum(1 for d in self.dropped if d.reason == "empty_content")

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    @property
    def remapped_missing_file_map(self) -> int:
        return sum(1 for r in self.remapped if r.reason == "missing_file_map")

    @property
    def remapped_missing_anchor(self) -> int:
        return sum(1 for r in self.remapped if r.reason == "missing_anchor")

    @property
    def remapped_count(self) -> int:
        return len(self.remapped)

    @property
    def total_count(self) -> int:
        return len(self.valid) + len(self.dropped) + len(self.remapped)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def parse_review_payload(payload: Mapping[str, Any]) -> ReviewRunResult:
    """Parse the schema-validated JSON payload into a typed review result.

    Thin wrapper around :func:`validate_review_payload` so callers that
    only want the JSON-to-objects step (Sub-AC 3.2 step 1) do not have
    to reach into ``cli.core.models``. Contract violations raise
    :class:`ReviewContractError`.
    """
    return validate_review_payload(payload)


def _has_meaningful_content(finding: ReviewFinding) -> bool:
    """Return True if the finding has any non-blank title or body content.

    Used to gate the remap fallback: an empty finding cannot be turned
    into a useful general PR issue comment, so we drop it with the
    ``empty_content`` reason instead of posting a hollow comment.
    """
    return bool(finding.title.strip() or finding.body.strip())


def _classify_anchor_failure(
    *,
    finding: ReviewFinding,
    relative_path: str,
    location: FindingLocation,
    reason: RemapReason,
    fallback_policy: FallbackPolicy,
    debug: DebugLogger,
) -> RemappedComment | DroppedComment:
    """Apply the configured fallback policy to a non-anchorable finding.

    The validator centralises the "what do we do when the anchor check
    fails" decision here so the per-finding path is straightforward:
    snap-to-nearest happens inside :func:`resolve_range` before this is
    called; this routine only sees findings that even the snap couldn't
    save. It then either remaps to a general PR comment or drops with
    a logged reason.
    """
    if fallback_policy == "drop":
        debug(
            1,
            f"comment_validator: dropping finding ({reason}) for "
            f"{relative_path}:{location.start_line}-{location.end_line} — strict-drop policy",
        )
        return DroppedComment(
            finding=finding,
            relative_path=relative_path,
            location=location,
            reason=reason,
        )

    if not _has_meaningful_content(finding):
        debug(
            1,
            f"comment_validator: dropping finding (empty_content) for "
            f"{relative_path}:{location.start_line}-{location.end_line} — "
            f"original failure was {reason}, no body to remap",
        )
        return DroppedComment(
            finding=finding,
            relative_path=relative_path,
            location=location,
            reason="empty_content",
        )

    debug(
        1,
        f"comment_validator: remapping finding ({reason}) for "
        f"{relative_path}:{location.start_line}-{location.end_line} → general PR issue comment",
    )
    return RemappedComment(
        finding=finding,
        relative_path=relative_path,
        location=location,
        reason=reason,
    )


def validate_finding_against_diff(
    finding: ReviewFinding,
    file_maps: Mapping[str, ParsedPatch],
    rename_map: Mapping[str, str],
    repo_root: Path,
    *,
    max_suggestion_span: int = 5,
    fallback_policy: FallbackPolicy = "remap_to_general",
    debug: DebugLogger | None = None,
) -> ValidatedComment | RemappedComment | DroppedComment:
    """Validate a single finding's path and line against PR diff hunks.

    The function applies the Sub-AC 3.3.2 fallback ladder so we never
    issue requests we know will 422 *and* don't silently drop signal:

    1. Resolve the finding's absolute path to a repo-relative path,
       applying any rename mapping (``previous_filename`` → new). On a
       miss, fall through to step 3 with ``missing_file_map``.
    2. Ask the anchor engine to map the requested line range onto a
       real anchor inside one of the file's diff hunks. The anchor
       engine already snaps slightly-off line numbers to the nearest
       non-blank head-line, so this only fails when the patch has *no*
       anchorable HEAD line (e.g. deletion-only). On a miss, fall
       through to step 3 with ``missing_anchor``.
    3. Apply ``fallback_policy``:

       * ``remap_to_general`` (default) — return a
         :class:`RemappedComment` so the poster can publish the finding
         as a general PR issue comment, preserving the reviewer's
         signal. Empty-content findings are dropped instead.
       * ``drop`` — return a :class:`DroppedComment` mirroring the
         pre-3.3.2 strict behaviour.

    On success we return a :class:`ValidatedComment` carrying enough
    information for the payload builder to fall straight through to
    the GitHub request without re-parsing anything.
    """
    debug_logger = debug if debug is not None else _noop_debug
    location = FindingLocation.from_review_finding(finding)
    relative_path = to_relative_path(location.absolute_file_path, repo_root)
    relative_path = rename_map.get(relative_path, relative_path)

    file_map = file_maps.get(relative_path)
    if file_map is None:
        return _classify_anchor_failure(
            finding=finding,
            relative_path=relative_path,
            location=location,
            reason="missing_file_map",
            fallback_policy=fallback_policy,
            debug=debug_logger,
        )

    has_suggestion = "```suggestion" in finding.body
    anchor = resolve_range(
        location.start_line,
        location.end_line,
        has_suggestion,
        file_map,
        max_suggestion_span=max_suggestion_span,
    )
    if anchor is None:
        return _classify_anchor_failure(
            finding=finding,
            relative_path=relative_path,
            location=location,
            reason="missing_anchor",
            fallback_policy=fallback_policy,
            debug=debug_logger,
        )

    return ValidatedComment(
        finding=finding,
        relative_path=relative_path,
        location=location,
        anchor=anchor,
        has_suggestion=has_suggestion,
    )


def validate_findings_against_diff(
    findings: Sequence[ReviewFinding],
    file_maps: Mapping[str, ParsedPatch],
    rename_map: Mapping[str, str],
    repo_root: Path,
    *,
    max_suggestion_span: int = 5,
    fallback_policy: FallbackPolicy = "remap_to_general",
    debug: DebugLogger | None = None,
) -> CommentValidationResult:
    """Validate every finding in ``findings`` against the PR diff hunks.

    Order is preserved across the ``valid``, ``remapped`` and
    ``dropped`` buckets so callers can correlate validation output with
    the original review run.
    """
    result = CommentValidationResult()
    for finding in findings:
        outcome = validate_finding_against_diff(
            finding,
            file_maps,
            rename_map,
            repo_root,
            max_suggestion_span=max_suggestion_span,
            fallback_policy=fallback_policy,
            debug=debug,
        )
        if isinstance(outcome, ValidatedComment):
            result.valid.append(outcome)
        elif isinstance(outcome, RemappedComment):
            result.remapped.append(outcome)
        else:
            result.dropped.append(outcome)
    return result


def parse_and_validate_findings(
    payload: Mapping[str, Any],
    file_maps: Mapping[str, ParsedPatch],
    rename_map: Mapping[str, str],
    repo_root: Path,
    *,
    max_suggestion_span: int = 5,
    fallback_policy: FallbackPolicy = "remap_to_general",
    debug: DebugLogger | None = None,
) -> tuple[ReviewRunResult, CommentValidationResult]:
    """End-to-end: schema-validated JSON → typed result + validated comments.

    Convenience helper for callers (e.g. the review workflow) that want
    the full Sub-AC 3.2 pipeline in one call. Returns both the typed
    review run (so the caller still has access to the overall summary
    fields, carried-forward entries, etc.) and the comment-validation
    outcome.

    Raises :class:`ReviewContractError` if the JSON payload itself
    fails schema validation; finding-level diff-anchor failures are
    surfaced via :class:`CommentValidationResult.dropped` /
    ``CommentValidationResult.remapped``, not raised.
    """
    review = parse_review_payload(payload)
    validation = validate_findings_against_diff(
        review.findings,
        file_maps,
        rename_map,
        repo_root,
        max_suggestion_span=max_suggestion_span,
        fallback_policy=fallback_policy,
        debug=debug,
    )
    return review, validation


# ---------------------------------------------------------------------------
# Helpers re-exported for the payload builder
# ---------------------------------------------------------------------------

__all__ = [
    "CommentValidationResult",
    "DroppedComment",
    "DropReason",
    "FallbackPolicy",
    "RemappedComment",
    "RemapReason",
    "ValidatedComment",
    "parse_and_validate_findings",
    "parse_review_payload",
    "validate_finding_against_diff",
    "validate_findings_against_diff",
    # Re-exports so consumers don't need to know which module owns the
    # anchor types. Keeps Sub-AC 3.2 callers within a single import.
    "Anchor",
    "RangeAnchor",
    "SingleAnchor",
    "ReviewContractError",
]
