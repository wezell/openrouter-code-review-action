from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .exceptions import ReviewContractError

if TYPE_CHECKING:
    from .github_types import IssueCommentLikeProtocol, ReviewCommentLikeProtocol


@dataclass(frozen=True)
class CommentContext:
    """Context for comment-triggered edit commands."""

    id: int
    event_name: str
    author: str = ""
    body: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> CommentContext | None:
        if payload is None:
            return None
        try:
            comment_id = int(payload.get("id") or 0)
        except (TypeError, ValueError):
            return None

        event_name = str(payload.get("event_name") or "")
        if comment_id <= 0 or not event_name:
            return None
        author = str(payload.get("author") or "")
        body = str(payload.get("body") or "")
        return cls(id=comment_id, event_name=event_name, author=author, body=body)


@dataclass(frozen=True)
class FindingLocation:
    """Normalized finding location values parsed from model output."""

    absolute_file_path: str
    start_line: int
    end_line: int

    @classmethod
    def from_finding(cls, finding: Mapping[str, Any]) -> FindingLocation | None:
        loc = finding.get("code_location")
        if not isinstance(loc, Mapping):
            return None

        abs_path_raw = loc.get("absolute_file_path")
        abs_path = abs_path_raw.strip() if isinstance(abs_path_raw, str) else ""
        rng = loc.get("line_range")
        if not isinstance(rng, Mapping):
            return None

        start = _as_int(rng.get("start"), 0)
        end = _as_int(rng.get("end"), start)
        if end <= 0 and start > 0:
            end = start
        if not abs_path or start <= 0:
            return None
        return cls(abs_path, start, end)

    @classmethod
    def from_review_finding(cls, finding: ReviewFinding) -> FindingLocation:
        return cls(
            absolute_file_path=finding.code_location.absolute_file_path,
            start_line=finding.code_location.start_line,
            end_line=finding.code_location.end_line,
        )


@dataclass(frozen=True)
class ReviewFindingLocation:
    absolute_file_path: str
    start_line: int
    end_line: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ReviewFindingLocation | None:
        base = FindingLocation.from_finding({"code_location": payload})
        if base is None:
            return None
        return cls(
            absolute_file_path=base.absolute_file_path,
            start_line=base.start_line,
            end_line=base.end_line,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "absolute_file_path": self.absolute_file_path,
            "line_range": {
                "start": self.start_line,
                "end": self.end_line,
            },
        }


# Allowed values for an inline review comment's diff side. ``RIGHT`` (the
# new file content / additions) is the default for almost every finding;
# ``LEFT`` is only used when the comment anchors on a deletion. The
# OpenRouter strict schema enforces these literals on the wire; local
# Python validation tolerates ``None`` so persisted fixtures (and
# historical review-state JSON) without a ``side`` field still load.
REVIEW_COMMENT_SIDES: tuple[str, ...] = ("LEFT", "RIGHT")


@dataclass(frozen=True)
class ReviewFinding:
    title: str
    body: str
    confidence_score: float | None
    priority: int | None
    code_location: ReviewFindingLocation
    # ``side`` is the GitHub inline-comment "side" field. ``None`` means
    # "let the poster default to RIGHT" — kept nullable so fixtures and
    # state files written before Sub-AC 3.1 added the field still parse.
    side: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ReviewFinding:
        required_fields = {"title", "body", "confidence_score", "priority", "code_location"}
        missing_fields = sorted(required_fields - set(payload.keys()))
        if missing_fields:
            raise ReviewContractError(
                "Review finding missing required fields: " + ", ".join(missing_fields)
            )

        title_raw = payload.get("title")
        if not isinstance(title_raw, str):
            raise ReviewContractError("Review finding field 'title' must be a string")

        body_raw = payload.get("body")
        if not isinstance(body_raw, str):
            raise ReviewContractError("Review finding field 'body' must be a string")

        confidence_raw = payload.get("confidence_score")
        if confidence_raw is not None and not isinstance(confidence_raw, (int, float)):
            raise ReviewContractError(
                "Review finding field 'confidence_score' must be a number or null"
            )
        confidence_score = (
            float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
        )

        priority_raw = payload.get("priority")
        if priority_raw is not None and not isinstance(priority_raw, int):
            raise ReviewContractError("Review finding field 'priority' must be an integer or null")
        priority = int(priority_raw) if isinstance(priority_raw, int) else None

        code_location_raw = payload.get("code_location")
        if not isinstance(code_location_raw, Mapping):
            raise ReviewContractError("Review finding field 'code_location' must be an object")
        code_location = ReviewFindingLocation.from_mapping(code_location_raw)
        if code_location is None:
            raise ReviewContractError("Review finding field 'code_location' is invalid")

        # ``side`` is optional in the lenient parser path so legacy
        # fixtures keep loading, but if present it must be one of the
        # GitHub-accepted literals (or null).
        side_raw = payload.get("side")
        if side_raw is None:
            side: str | None = None
        elif isinstance(side_raw, str):
            normalised = side_raw.strip().upper()
            if normalised not in REVIEW_COMMENT_SIDES:
                raise ReviewContractError(
                    "Review finding field 'side' must be one of "
                    f"{list(REVIEW_COMMENT_SIDES)} or null"
                )
            side = normalised
        else:
            raise ReviewContractError("Review finding field 'side' must be a string or null")

        return cls(
            title=title_raw,
            body=body_raw,
            confidence_score=confidence_score,
            priority=priority,
            code_location=code_location,
            side=side,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "confidence_score": self.confidence_score,
            "priority": self.priority,
            "code_location": self.code_location.as_dict(),
            "side": self.side,
        }


@dataclass(frozen=True)
class PriorCodexReviewComment:
    """Unresolved Codex-authored review thread comment reused on reruns."""

    id: str
    thread_id: str
    path: str
    line: int
    body: str
    current_code: str
    is_currently_applicable: bool


@dataclass(frozen=True)
class CarriedForwardReviewComment:
    """Prior Codex review comment re-adjudicated as still applicable."""

    comment_id: str
    current_evidence: str


@dataclass(frozen=True)
class IssueCommentSnapshot:
    body: str
    created_at: str
    author: str = ""

    @classmethod
    def from_issue_comment(cls, comment: IssueCommentLikeProtocol) -> IssueCommentSnapshot:
        return cls(
            body=comment.body if isinstance(comment.body, str) else "",
            created_at=str(comment.created_at),
            author=comment.user.login if comment.user is not None else "",
        )


@dataclass(frozen=True)
class ReviewCommentSnapshot:
    body: str
    path: str
    line: int | None
    original_line: int | None
    author: str = ""
    created_at: str = ""
    diff_hunk: str = ""
    commit_id: str = ""
    in_reply_to_id: int | None = None

    @property
    def prompt_line(self) -> int | None:
        return self.line if self.line is not None else self.original_line

    @classmethod
    def from_review_comment(cls, comment: ReviewCommentLikeProtocol) -> ReviewCommentSnapshot:
        author_value = comment.user.login if comment.user is not None else None
        return cls(
            body=comment.body.strip() if isinstance(comment.body, str) else "",
            path=comment.path if isinstance(comment.path, str) else "",
            line=comment.line if isinstance(comment.line, int) else None,
            original_line=comment.original_line if isinstance(comment.original_line, int) else None,
            author=author_value if isinstance(author_value, str) else "",
            created_at=str(comment.created_at) if comment.created_at is not None else "",
            diff_hunk=comment.diff_hunk if isinstance(comment.diff_hunk, str) else "",
            commit_id=comment.commit_id if isinstance(comment.commit_id, str) else "",
            in_reply_to_id=comment.in_reply_to_id
            if isinstance(comment.in_reply_to_id, int)
            else None,
        )


@dataclass(frozen=True)
class ReviewThreadComment:
    """Normalized review-thread comment snapshot from GraphQL."""

    id: str
    body: str
    path: str
    line: int | None
    original_line: int | None
    author: str = ""

    @property
    def prompt_line(self) -> int | None:
        return self.line if self.line is not None else self.original_line


@dataclass(frozen=True)
class ReviewThreadSnapshot:
    """Normalized review thread snapshot with resolution state."""

    id: str
    is_resolved: bool
    comments: list[ReviewThreadComment]


@dataclass(frozen=True)
class UnresolvedReviewComment:
    """Normalized review-comment context for unresolved thread prompts."""

    id: str
    body: str
    path: str
    line: int | None
    original_line: int | None
    author: str = ""

    @property
    def prompt_line(self) -> int | None:
        return self.line if self.line is not None else self.original_line


@dataclass(frozen=True)
class UnresolvedReviewThread:
    """Normalized unresolved review thread used by edit mode."""

    id: str
    comments: list[UnresolvedReviewComment]


@dataclass(frozen=True)
class InlineCommentPayload:
    """Payload for posting a GitHub inline review comment."""

    body: str
    path: str
    side: str = "RIGHT"
    line: int = 0
    start_line: int | None = None
    start_side: str = "RIGHT"

    def to_request_payload(self, head_sha: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "body": self.body,
            "path": self.path,
            "side": self.side,
            "commit_id": head_sha,
            "line": int(self.line),
        }
        if self.start_line is not None:
            payload["start_line"] = int(self.start_line)
            payload["start_side"] = self.start_side
        return payload

    def to_review_comment_input(self) -> dict[str, Any]:
        """Render this payload for embedding inside a ``POST /pulls/{n}/reviews`` body.

        The Reviews API takes the ``commit_id`` once at the review level and
        accepts each inline comment as a smaller per-comment object: ``path``,
        ``body``, ``line``, ``side``, plus optional ``start_line`` /
        ``start_side`` for multi-line ranges. This helper produces exactly that
        shape so the github client can send the whole batch in a single POST
        instead of one POST per comment.
        """
        comment: dict[str, Any] = {
            "body": self.body,
            "path": self.path,
            "side": self.side,
            "line": int(self.line),
        }
        if self.start_line is not None:
            comment["start_line"] = int(self.start_line)
            comment["start_side"] = self.start_side
        return comment


@dataclass(frozen=True)
class ReviewRunResult:
    """Typed view of model output for a review run."""

    overall_correctness: str
    overall_explanation: str
    overall_confidence_score: float | None
    findings: list[ReviewFinding]
    carried_forward: list[CarriedForwardReviewComment] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReviewRunResult:
        required_fields = {
            "findings",
            "carried_forward",
            "overall_correctness",
            "overall_explanation",
            "overall_confidence_score",
        }
        missing_fields = sorted(required_fields - set(payload.keys()))
        if missing_fields:
            raise ReviewContractError(
                "Review output missing required fields: " + ", ".join(missing_fields)
            )

        findings_raw = payload.get("findings")
        if not isinstance(findings_raw, list):
            raise ReviewContractError("Review output field 'findings' must be an array")
        findings: list[ReviewFinding] = []
        for index, item in enumerate(findings_raw):
            if not isinstance(item, Mapping):
                raise ReviewContractError(
                    f"Review output finding at index {index} must be an object"
                )
            findings.append(ReviewFinding.from_mapping(item))

        overall_correctness_raw = payload.get("overall_correctness")
        if not isinstance(overall_correctness_raw, str):
            raise ReviewContractError("Review output field 'overall_correctness' must be a string")
        overall_correctness = overall_correctness_raw

        overall_explanation_raw = payload.get("overall_explanation")
        if not isinstance(overall_explanation_raw, str):
            raise ReviewContractError("Review output field 'overall_explanation' must be a string")
        overall_explanation = overall_explanation_raw

        confidence_raw = payload.get("overall_confidence_score")
        if confidence_raw is not None and not isinstance(confidence_raw, (int, float)):
            raise ReviewContractError(
                "Review output field 'overall_confidence_score' must be a number or null"
            )
        overall_confidence_score = (
            float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
        )
        carried_forward_raw = payload.get("carried_forward")
        if not isinstance(carried_forward_raw, list):
            raise ReviewContractError("Review output field 'carried_forward' must be an array")
        carried_forward: list[CarriedForwardReviewComment] = []
        for index, item in enumerate(carried_forward_raw):
            if not isinstance(item, Mapping):
                raise ReviewContractError(
                    f"Review output field 'carried_forward' item at index {index} must be an object"
                )
            comment_id = item.get("comment_id")
            if not isinstance(comment_id, str):
                raise ReviewContractError(
                    "Review output field 'carried_forward' "
                    f"item at index {index} must include string field 'comment_id'"
                )
            current_evidence = item.get("current_evidence")
            if not isinstance(current_evidence, str):
                raise ReviewContractError(
                    "Review output field 'carried_forward' "
                    f"item at index {index} must include string field 'current_evidence'"
                )
            carried_forward.append(
                CarriedForwardReviewComment(
                    comment_id=comment_id,
                    current_evidence=current_evidence,
                )
            )
        return cls(
            overall_correctness=overall_correctness,
            overall_explanation=overall_explanation,
            overall_confidence_score=overall_confidence_score,
            findings=findings,
            carried_forward=carried_forward,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_correctness": self.overall_correctness,
            "overall_explanation": self.overall_explanation,
            "overall_confidence_score": self.overall_confidence_score,
            "findings": [finding.as_dict() for finding in self.findings],
            "carried_forward": [
                {
                    "comment_id": item.comment_id,
                    "current_evidence": item.current_evidence,
                }
                for item in self.carried_forward
            ],
        }

    @property
    def carried_forward_comment_ids(self) -> list[str]:
        return [item.comment_id for item in self.carried_forward]


# --- JSON schema definitions for OpenRouter structured output --------------
#
# These schemas are designed to be compatible with OpenRouter's
# `response_format` parameter using `type: "json_schema"` with
# `strict: true`. Strict-mode requirements (inherited from the OpenAI
# Structured Outputs spec that OpenRouter forwards):
#   - Every object must declare every property in `required`.
#   - Every object must set `additionalProperties: false`.
#   - Nullable values use the `["<type>", "null"]` array form.
#
# `REVIEW_FINDING_SCHEMA` describes a single inline review comment as
# emitted by the model. `REVIEW_OUTPUT_SCHEMA` describes the full review
# run envelope (findings + carried_forward + overall summary).

REVIEW_FINDING_LOCATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "absolute_file_path": {"type": "string"},
        "line_range": {
            "type": "object",
            "properties": {
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            "required": ["start", "end"],
            "additionalProperties": False,
        },
    },
    "required": ["absolute_file_path", "line_range"],
    "additionalProperties": False,
}


REVIEW_FINDING_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body": {"type": "string"},
        "confidence_score": {"type": ["number", "null"]},
        "priority": {"type": ["integer", "null"]},
        # ``side`` mirrors the GitHub inline-comment "side" field — RIGHT
        # for additions/context (default), LEFT only when the comment
        # anchors on a deleted line. Nullable so the model can omit it
        # for the common RIGHT-side case; the poster falls back to RIGHT
        # whenever the value is null.
        "side": {
            "type": ["string", "null"],
            "enum": [*REVIEW_COMMENT_SIDES, None],
        },
        "code_location": REVIEW_FINDING_LOCATION_SCHEMA,
    },
    "required": [
        "title",
        "body",
        "confidence_score",
        "priority",
        "side",
        "code_location",
    ],
    "additionalProperties": False,
}


CARRIED_FORWARD_COMMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "comment_id": {"type": "string"},
        "current_evidence": {"type": "string"},
    },
    "required": ["comment_id", "current_evidence"],
    "additionalProperties": False,
}


REVIEW_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": REVIEW_FINDING_SCHEMA,
        },
        "carried_forward": {
            "type": "array",
            "items": CARRIED_FORWARD_COMMENT_SCHEMA,
        },
        "overall_correctness": {"type": "string"},
        "overall_explanation": {"type": "string"},
        "overall_confidence_score": {"type": ["number", "null"]},
    },
    "required": [
        "findings",
        "carried_forward",
        "overall_correctness",
        "overall_explanation",
        "overall_confidence_score",
    ],
    "additionalProperties": False,
}


OPENROUTER_REVIEW_SCHEMA_NAME = "review_output"


def build_openrouter_response_format(
    *,
    schema: Mapping[str, Any],
    name: str = OPENROUTER_REVIEW_SCHEMA_NAME,
    strict: bool = True,
) -> dict[str, Any]:
    """Build the OpenRouter chat-completions ``response_format`` payload.

    OpenRouter forwards OpenAI-compatible Structured Output requests when the
    payload follows::

        {
          "type": "json_schema",
          "json_schema": {
            "name": "<identifier>",
            "strict": true,
            "schema": { ... JSON Schema ... }
          }
        }

    Strict mode causes the upstream provider to reject any output that does
    not conform to the schema before it ever reaches us.
    """
    if not isinstance(name, str) or not name.strip():
        raise ReviewContractError("response_format schema name must be a non-empty string")
    if not isinstance(schema, Mapping):
        raise ReviewContractError("response_format schema must be a mapping")

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name.strip(),
            "strict": bool(strict),
            "schema": dict(schema),
        },
    }


def openrouter_review_response_format() -> dict[str, Any]:
    """Return the strict ``response_format`` payload for review runs."""
    return build_openrouter_response_format(
        schema=REVIEW_OUTPUT_SCHEMA,
        name=OPENROUTER_REVIEW_SCHEMA_NAME,
        strict=True,
    )


def validate_review_payload(payload: Mapping[str, Any]) -> ReviewRunResult:
    """Validate and parse a JSON payload against ``REVIEW_OUTPUT_SCHEMA``.

    OpenRouter's strict ``response_format`` already enforces the schema on
    the provider side; this function is the local belt-and-suspenders pass
    that turns a raw decoded JSON object into a typed ``ReviewRunResult``.
    Contract violations raise :class:`ReviewContractError`.
    """
    if not isinstance(payload, Mapping):
        raise ReviewContractError("Review payload must be a JSON object")
    return ReviewRunResult.from_payload(payload)


def validate_review_finding(payload: Mapping[str, Any]) -> ReviewFinding:
    """Validate a single inline review comment against ``REVIEW_FINDING_SCHEMA``."""
    if not isinstance(payload, Mapping):
        raise ReviewContractError("Review finding payload must be a JSON object")
    return ReviewFinding.from_mapping(payload)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
