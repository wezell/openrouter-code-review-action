from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..clients.github_client import GitHubClientProtocol
from ..core.exceptions import GitHubAPIError
from ..core.filesystem import write_text_atomic
from ..core.github_types import PullRequestLikeProtocol
from ..core.models import InlineCommentPayload, ReviewFinding
from .anchor_engine import RangeAnchor
from .artifacts import ReviewArtifacts
from .comment_validator import (
    CommentValidationResult,
    FallbackPolicy,
    RemappedComment,
    ValidatedComment,
    validate_findings_against_diff,
)
from .patch_parser import ParsedPatch

# Marker prefix used to identify remapped finding comments so future
# review runs can distinguish them from operator-authored comments and
# clean them up if needed. Mirrors the role of ``SUMMARY_MARKER`` for
# the run-level summary issue comment.
REMAPPED_FINDING_MARKER = "<!-- openrouter-review:remapped-finding -->"


@dataclass(frozen=True)
class GeneralPRCommentPayload:
    """Payload for a general PR issue comment used as an anchor fallback.

    Sub-AC 3.3.2 routes findings that cannot be anchored on the diff
    (file not in the PR, deletion-only patch, etc.) into general PR
    issue comments instead of dropping them. These payloads are posted
    via ``GitHubClient.post_issue_comment`` rather than the inline
    review-comment endpoint, so the request shape is just a body
    string.
    """

    body: str
    # Tracking metadata for diagnostics / summary text. Not sent to GitHub.
    relative_path: str
    start_line: int
    end_line: int
    reason: str


@dataclass(frozen=True)
class InlineCommentBuildResult:
    payloads: list[InlineCommentPayload]
    dropped_missing_location: int = 0
    dropped_missing_file_map: int = 0
    dropped_missing_anchor: int = 0
    dropped_empty_content: int = 0
    remapped_payloads: list[GeneralPRCommentPayload] = field(default_factory=list)
    remapped_missing_file_map: int = 0
    remapped_missing_anchor: int = 0

    @property
    def dropped_count(self) -> int:
        return (
            self.dropped_missing_location
            + self.dropped_missing_file_map
            + self.dropped_missing_anchor
            + self.dropped_empty_content
        )

    @property
    def remapped_count(self) -> int:
        return len(self.remapped_payloads)

    def describe_drops(self) -> str:
        parts: list[str] = []
        if self.dropped_missing_location:
            parts.append(f"missing location={self.dropped_missing_location}")
        if self.dropped_missing_file_map:
            parts.append(f"missing file map={self.dropped_missing_file_map}")
        if self.dropped_missing_anchor:
            parts.append(f"missing anchor={self.dropped_missing_anchor}")
        if self.dropped_empty_content:
            parts.append(f"empty content={self.dropped_empty_content}")
        return ", ".join(parts)

    def describe_remaps(self) -> str:
        parts: list[str] = []
        if self.remapped_missing_file_map:
            parts.append(f"missing file map={self.remapped_missing_file_map}")
        if self.remapped_missing_anchor:
            parts.append(f"missing anchor={self.remapped_missing_anchor}")
        return ", ".join(parts)


@dataclass(frozen=True)
class InlineCommentPostResult:
    """Outcome of submitting inline review comments to GitHub.

    Sub-AC 3.3.3 added batched submission with a per-comment 422 fallback
    on top of the original per-comment shape, so this carries enough
    diagnostic state for the workflow summary log to explain *which* path
    actually published the comments:

    * ``batch_submitted`` — set when the single ``POST /pulls/{n}/reviews``
      batch succeeded outright (the happy path).
    * ``per_comment_fallback`` — set when batched submission was rejected
      with HTTP 422 and we fell back to the per-comment endpoint to
      isolate the offending payload(s) instead of dropping the whole
      review.
    * ``skipped_after_422`` — count of comments that were dropped during
      the per-comment fallback because GitHub specifically rejected them
      with 422 (e.g. anchor edge case the diff parser missed). Counted
      separately from ``posted_count`` so the operator can see signal
      loss vs. successful publication.
    """

    attempted_count: int
    posted_count: int
    dry_run: bool = False
    batch_submitted: bool = False
    per_comment_fallback: bool = False
    skipped_after_422: int = 0


@dataclass(frozen=True)
class RemappedCommentPostResult:
    """Outcome of posting general PR comments for remapped findings."""

    attempted_count: int
    posted_count: int
    dry_run: bool = False


@dataclass(frozen=True)
class ReviewPostingOutcome:
    total_findings: int
    prefiltered_count: int
    build_result: InlineCommentBuildResult
    post_result: InlineCommentPostResult
    remapped_post_result: RemappedCommentPostResult = field(
        default_factory=lambda: RemappedCommentPostResult(
            attempted_count=0,
            posted_count=0,
            dry_run=False,
        )
    )

    @classmethod
    def empty(cls, total_findings: int, *, dry_run: bool = False) -> ReviewPostingOutcome:
        return cls(
            total_findings=total_findings,
            prefiltered_count=0,
            build_result=InlineCommentBuildResult(payloads=[]),
            post_result=InlineCommentPostResult(attempted_count=0, posted_count=0, dry_run=dry_run),
            remapped_post_result=RemappedCommentPostResult(
                attempted_count=0,
                posted_count=0,
                dry_run=dry_run,
            ),
        )

    @property
    def publishable_count(self) -> int:
        return len(self.build_result.payloads)

    @property
    def published_count(self) -> int:
        return self.post_result.posted_count

    @property
    def remapped_count(self) -> int:
        return self.build_result.remapped_count

    @property
    def remapped_published_count(self) -> int:
        return self.remapped_post_result.posted_count

    @property
    def dropped_count(self) -> int:
        return self.prefiltered_count + self.build_result.dropped_count

    @property
    def batch_submitted(self) -> bool:
        """True when the inline comments shipped as a single batched PR review."""
        return self.post_result.batch_submitted

    @property
    def per_comment_fallback(self) -> bool:
        """True when batched submission was rejected with 422 and fallback ran."""
        return self.post_result.per_comment_fallback

    @property
    def skipped_after_422(self) -> int:
        """Count of inline comments dropped during the per-comment 422 fallback."""
        return self.post_result.skipped_after_422

    def describe_drops(self) -> str:
        parts: list[str] = []
        if self.prefiltered_count:
            parts.append(f"existing comments={self.prefiltered_count}")
        build_drops = self.build_result.describe_drops()
        if build_drops:
            parts.append(build_drops)
        if self.skipped_after_422:
            parts.append(f"422 retry rejection={self.skipped_after_422}")
        return ", ".join(parts)

    def describe_remaps(self) -> str:
        return self.build_result.describe_remaps()

    def as_dict(self) -> dict[str, object]:
        return {
            "total_findings": self.total_findings,
            "prefiltered_count": self.prefiltered_count,
            "publishable_count": self.publishable_count,
            "published_count": self.published_count,
            "remapped_count": self.remapped_count,
            "remapped_published_count": self.remapped_published_count,
            "dropped_count": self.dropped_count,
            "dry_run": self.post_result.dry_run,
            "drop_reasons": self.describe_drops(),
            "remap_reasons": self.describe_remaps(),
            "batch_submitted": self.batch_submitted,
            "per_comment_fallback": self.per_comment_fallback,
            "skipped_after_422": self.skipped_after_422,
        }


def persist_anchor_maps(
    file_maps: Mapping[str, ParsedPatch],
    artifacts: ReviewArtifacts,
) -> None:
    payload = {
        path: {
            "valid_head_lines": sorted(list(parsed.valid_head_lines)),
            "added_head_lines": sorted(list(parsed.added_head_lines)),
            "positions_by_head_line": {
                str(line): position for line, position in parsed.positions_by_head_line.items()
            },
            "hunks": parsed.hunks,
        }
        for path, parsed in file_maps.items()
    }
    artifacts.base_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        artifacts.anchor_maps_path,
        json.dumps(payload, indent=2),
    )


def _build_payload_from_validated(comment: ValidatedComment) -> InlineCommentPayload:
    """Render a :class:`ValidatedComment` into a GitHub inline-comment payload."""
    title = comment.finding.title.strip() or "Issue"
    body = comment.finding.body.strip()

    final_body = body
    if comment.has_suggestion and not isinstance(comment.anchor, RangeAnchor):
        # ```suggestion blocks only render correctly on multi-line
        # range anchors; if we collapsed to a single line, fall back to
        # a plain diff fence so GitHub does not silently drop the body.
        final_body = body.replace("```suggestion", "```diff")

    comment_body = f"{title}\n\n{final_body}" if final_body else title

    # Honour the schema-enforced ``side`` value when the model supplied
    # one (LEFT for deletion-anchored findings, RIGHT for additions /
    # context). When the model omitted it (legacy fixtures, or the
    # default RIGHT-side case), fall back to RIGHT so the post still
    # succeeds.
    side = comment.finding.side or "RIGHT"

    anchor = comment.anchor
    if isinstance(anchor, RangeAnchor):
        return InlineCommentPayload(
            body=comment_body,
            path=comment.relative_path,
            side=side,
            line=anchor.end_line,
            start_line=anchor.start_line,
            start_side=side,
        )

    return InlineCommentPayload(
        body=comment_body,
        path=comment.relative_path,
        side=side,
        line=anchor.line,
    )


def _format_remap_reason(reason: str) -> str:
    """Render the remap reason in a way that's useful in a PR comment.

    The validator's reason codes are machine-readable; this prefix is
    for human reviewers reading the resulting issue comment.
    """
    if reason == "missing_file_map":
        return "the referenced file is not part of this PR's diff"
    if reason == "missing_anchor":
        return "the referenced line is not part of this PR's diff hunks"
    return reason


def _build_payload_from_remapped(
    comment: RemappedComment,
) -> GeneralPRCommentPayload:
    """Render a :class:`RemappedComment` into a general PR issue-comment payload.

    The body is prefixed with a marker line so future review runs can
    detect comments authored by this fallback path, and includes a
    one-line "where the model pointed" header so the reviewer doesn't
    lose the context that an inline anchor would normally provide.
    """
    title = comment.finding.title.strip() or "Issue"
    body = comment.finding.body.strip()
    location_label = (
        f"{comment.relative_path}:{comment.location.start_line}-{comment.location.end_line}"
        if comment.location.start_line != comment.location.end_line
        else f"{comment.relative_path}:{comment.location.start_line}"
    )
    header = (
        f"_Posted as a general PR comment because {_format_remap_reason(comment.reason)}._\n"
        f"_Original target: `{location_label}`._"
    )
    parts = [REMAPPED_FINDING_MARKER, f"### {title}", header]
    if body:
        parts.append(body)
    text = "\n\n".join(parts)
    return GeneralPRCommentPayload(
        body=text,
        relative_path=comment.relative_path,
        start_line=comment.location.start_line,
        end_line=comment.location.end_line,
        reason=comment.reason,
    )


def build_inline_comment_payloads(
    findings: Sequence[ReviewFinding],
    file_maps: Mapping[str, ParsedPatch],
    rename_map: Mapping[str, str],
    repo_root: Path,
    *,
    dry_run: bool,
    debug: Callable[[int, str], None],
    validation: CommentValidationResult | None = None,
    fallback_policy: FallbackPolicy = "remap_to_general",
) -> InlineCommentBuildResult:
    """Validate findings against the PR diff and render GitHub payloads.

    Sub-AC 3.2 + 3.3.2 split cleanly into three steps:

    1. *Parse + validate* — :func:`validate_findings_against_diff`
       takes the schema-validated review findings, classifies each as
       a :class:`ValidatedComment` (anchored on a real diff line), a
       :class:`RemappedComment` (cannot anchor inline but content is
       worth posting as a general PR comment), or a
       :class:`DroppedComment` with a categorical reason. We do this
       here unless the caller already ran the validator and threaded
       the result in via ``validation``.
    2. *Render inline payloads* — for every valid candidate, build the
       GitHub inline review-comment payload. No validation happens in
       this step, so any payload returned has already been proven safe
       to post.
    3. *Render remap payloads* — for every remapped candidate, build a
       :class:`GeneralPRCommentPayload` carrying the formatted
       issue-comment body. These get posted via the issue-comment
       endpoint, not the inline-comment endpoint.

    The ``validation`` parameter exists so callers that need both the
    typed validation outcome (for summary/audit) and the request
    payloads can compute it once and reuse it. ``fallback_policy``
    is forwarded to the validator only when ``validation`` is not
    pre-supplied.
    """
    if validation is None:
        validation = validate_findings_against_diff(
            findings,
            file_maps,
            rename_map,
            repo_root,
            fallback_policy=fallback_policy,
            debug=debug,
        )

    if dry_run:
        for dropped in validation.dropped:
            debug(
                1,
                (
                    f"DRY_RUN: would skip ({dropped.reason}) for "
                    f"{dropped.relative_path}:"
                    f"{dropped.location.start_line}-{dropped.location.end_line}"
                ),
            )
        for remapped in validation.remapped:
            debug(
                1,
                (
                    f"DRY_RUN: would post general PR comment ({remapped.reason}) for "
                    f"{remapped.relative_path}:"
                    f"{remapped.location.start_line}-{remapped.location.end_line}"
                ),
            )

    payloads = [_build_payload_from_validated(comment) for comment in validation.valid]
    remapped_payloads = [_build_payload_from_remapped(comment) for comment in validation.remapped]

    return InlineCommentBuildResult(
        payloads=payloads,
        dropped_missing_location=0,
        dropped_missing_file_map=validation.dropped_missing_file_map,
        dropped_missing_anchor=validation.dropped_missing_anchor,
        dropped_empty_content=validation.dropped_empty_content,
        remapped_payloads=remapped_payloads,
        remapped_missing_file_map=validation.remapped_missing_file_map,
        remapped_missing_anchor=validation.remapped_missing_anchor,
    )


def post_inline_comments(
    github_client: GitHubClientProtocol,
    pr: PullRequestLikeProtocol,
    head_sha: str,
    payloads: Sequence[InlineCommentPayload],
    *,
    dry_run: bool,
    debug: Callable[[int, str], None],
) -> InlineCommentPostResult:
    """Publish inline review comments using a batched-then-fallback ladder.

    Sub-AC 3.3.3 wires the validated/remapped payloads from the
    filter/remap pipeline into GitHub. The validator already pre-empts
    most 422-causing payloads (file not in diff, line not in hunk, …),
    so the happy path is:

    1. **Batched submission** — issue a single
       ``POST /pulls/{n}/reviews`` carrying every payload as an embedded
       inline comment. One round-trip, one review wrapper, regardless of
       how many findings the model produced.

    2. **Per-comment 422 fallback** — if the batch submission is
       rejected with HTTP 422 ("Unprocessable Entity"), the GitHub API
       has refused the whole review because of one or more bad anchors.
       Rather than dropping every comment in the batch, we replay each
       payload through the per-comment endpoint
       (``POST /pulls/{n}/comments``). Per-comment 422s are isolated and
       counted on the result; everything else is posted normally.

    Non-422 errors propagate so the operator can see them — we don't
    swallow auth failures, network blips, or rate limits as signal loss.
    """
    if not payloads:
        if dry_run:
            debug(1, "DRY_RUN: no inline findings to post")
        return InlineCommentPostResult(attempted_count=0, posted_count=0, dry_run=dry_run)

    if dry_run:
        for payload in payloads:
            debug(
                1,
                (f"DRY_RUN: would POST /comments for {payload.path}:{payload.line}"),
            )
        debug(
            1,
            (
                f"DRY_RUN: would batch {len(payloads)} inline comment(s) into a single "
                "POST /pulls/{n}/reviews submission"
            ),
        )
        return InlineCommentPostResult(
            attempted_count=len(payloads),
            posted_count=0,
            dry_run=True,
        )

    # --- Step 1: batched submission --------------------------------------
    debug(
        2,
        f"Submitting {len(payloads)} inline comment(s) as a single batched PR review",
    )
    try:
        github_client.post_pull_request_review(pr, payloads, head_sha=head_sha)
        return InlineCommentPostResult(
            attempted_count=len(payloads),
            posted_count=len(payloads),
            dry_run=False,
            batch_submitted=True,
            per_comment_fallback=False,
            skipped_after_422=0,
        )
    except GitHubAPIError as exc:
        if exc.status_code != 422:
            # Anything that isn't a 422 (auth, rate-limit, transient
            # network error) is the operator's problem to look at — don't
            # silently degrade those into per-comment retries.
            raise
        debug(
            1,
            (
                f"Batched PR review rejected with 422 ({exc}); "
                f"falling back to per-comment submission for {len(payloads)} payload(s)"
            ),
        )

    # --- Step 2: per-comment fallback after 422 --------------------------
    posted_count = 0
    skipped_after_422 = 0
    for payload in payloads:
        try:
            github_client.post_inline_comment(pr, payload, head_sha=head_sha)
            posted_count += 1
        except GitHubAPIError as exc:
            if exc.status_code != 422:
                # Surface non-422 retry failures: the batch already
                # failed once and we don't want a flaky API to silently
                # tank the whole post-fallback round.
                raise
            skipped_after_422 += 1
            debug(
                1,
                (
                    "Skipping inline comment for "
                    f"{payload.path}:{payload.line} after per-comment 422 "
                    f"retry rejection ({exc})"
                ),
            )

    return InlineCommentPostResult(
        attempted_count=len(payloads),
        posted_count=posted_count,
        dry_run=False,
        batch_submitted=False,
        per_comment_fallback=True,
        skipped_after_422=skipped_after_422,
    )


def post_remapped_comments(
    github_client: GitHubClientProtocol,
    pr: PullRequestLikeProtocol,
    payloads: Sequence[GeneralPRCommentPayload],
    *,
    dry_run: bool,
    debug: Callable[[int, str], None],
) -> RemappedCommentPostResult:
    """Post remapped (non-anchorable) findings as general PR issue comments.

    The Sub-AC 3.3.2 fallback rung: when the inline-anchor check fails
    for a finding, the validator wraps it in a :class:`RemappedComment`
    so the build step can produce a :class:`GeneralPRCommentPayload`,
    and *this* function publishes it via the GitHub issue-comment
    endpoint (one comment per remapped finding). Each post is a single
    ``POST /issues/{n}/comments`` call so a partial failure on
    comment N+1 doesn't roll back comments 1..N.
    """
    if not payloads:
        if dry_run:
            debug(1, "DRY_RUN: no remapped findings to post as general PR comments")
        return RemappedCommentPostResult(attempted_count=0, posted_count=0, dry_run=dry_run)

    posted_count = 0
    for payload in payloads:
        location_label = (
            f"{payload.relative_path}:{payload.start_line}-{payload.end_line}"
            if payload.start_line != payload.end_line
            else f"{payload.relative_path}:{payload.start_line}"
        )
        if dry_run:
            debug(
                1,
                (
                    "DRY_RUN: would POST /issues/{n}/comments for remapped finding "
                    f"({payload.reason}) originally targeting {location_label}"
                ),
            )
            continue

        debug(
            2,
            (
                "Posting remapped finding as general PR comment "
                f"({payload.reason}) originally targeting {location_label}"
            ),
        )
        github_client.post_issue_comment(pr, payload.body)
        posted_count += 1

    return RemappedCommentPostResult(
        attempted_count=len(payloads),
        posted_count=(0 if dry_run else posted_count),
        dry_run=dry_run,
    )
