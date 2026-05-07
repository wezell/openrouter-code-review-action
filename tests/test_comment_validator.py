"""Tests for Sub-AC 3.2: parse schema-validated JSON into comment objects and
validate each comment's file path and line number against the PR diff hunks
before submission.

These cover the standalone ``cli.review.comment_validator`` surface so the
"validate before posting" step is testable in isolation from the GitHub
client and the payload builder. The intent is that any finding we mark
``valid`` here would not produce a 422 from the GitHub inline-comment API,
and any finding we mark ``dropped`` would have produced a 422 if posted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cli.core.exceptions import ReviewContractError
from cli.core.models import ReviewFinding, ReviewFindingLocation, ReviewRunResult
from cli.review.anchor_engine import RangeAnchor, SingleAnchor, build_anchor_maps
from cli.review.comment_validator import (
    CommentValidationResult,
    DroppedComment,
    RemappedComment,
    ValidatedComment,
    parse_and_validate_findings,
    parse_review_payload,
    validate_finding_against_diff,
    validate_findings_against_diff,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeChangedFile:
    """Minimal ``ChangedFileProtocol`` shape used by ``build_anchor_maps``."""

    filename: str | None
    status: str
    patch: str | None
    previous_filename: str | None

    def __init__(
        self,
        filename: str,
        patch: str,
        *,
        status: str = "modified",
        previous_filename: str | None = None,
    ) -> None:
        self.filename = filename
        self.patch = patch
        self.status = status
        self.previous_filename = previous_filename


def _make_finding(
    *,
    abs_path: str,
    start_line: int,
    end_line: int | None = None,
    title: str = "Finding",
    body: str = "issue body",
) -> ReviewFinding:
    end = start_line if end_line is None else end_line
    return ReviewFinding(
        title=title,
        body=body,
        confidence_score=None,
        priority=None,
        code_location=ReviewFindingLocation(
            absolute_file_path=abs_path,
            start_line=start_line,
            end_line=end,
        ),
    )


def _conformant_payload(*, abs_path: str) -> dict[str, Any]:
    return {
        "overall_correctness": "patch is incorrect",
        "overall_explanation": "x",
        "overall_confidence_score": None,
        "carried_forward": [],
        "findings": [
            {
                "title": "Possible bug",
                "body": "explain it",
                "confidence_score": 0.7,
                "priority": 1,
                "code_location": {
                    "absolute_file_path": abs_path,
                    "line_range": {"start": 2, "end": 2},
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# parse_review_payload — JSON payload to typed comment objects
# ---------------------------------------------------------------------------


def test_parse_review_payload_returns_typed_review_run_result(tmp_path: Path) -> None:
    payload = _conformant_payload(abs_path=str((tmp_path / "sample.py").resolve()))
    result = parse_review_payload(payload)

    assert isinstance(result, ReviewRunResult)
    assert len(result.findings) == 1
    finding = result.findings[0]
    # The schema-validated JSON must yield a typed ReviewFinding object.
    assert isinstance(finding, ReviewFinding)
    assert finding.title == "Possible bug"
    assert finding.priority == 1
    assert finding.code_location.start_line == 2


def test_parse_review_payload_rejects_invalid_payload() -> None:
    with pytest.raises(ReviewContractError):
        parse_review_payload({"findings": "not-a-list"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_finding_against_diff — single-finding diff-hunk validation
# ---------------------------------------------------------------------------


def test_validate_finding_returns_valid_for_added_line(tmp_path: Path) -> None:
    repo_root = tmp_path
    filename = "added.py"
    abs_path = str((repo_root / filename).resolve())
    # Patch adds three lines starting at HEAD line 1.
    patch = "@@ -0,0 +1,3 @@\n+foo\n+bar\n+baz\n"
    file_maps = build_anchor_maps([_FakeChangedFile(filename, patch)])

    finding = _make_finding(abs_path=abs_path, start_line=2)

    outcome = validate_finding_against_diff(finding, file_maps, {}, repo_root)

    assert isinstance(outcome, ValidatedComment)
    assert outcome.relative_path == filename
    assert isinstance(outcome.anchor, SingleAnchor)
    assert outcome.anchor.line == 2
    assert outcome.has_suggestion is False


def test_validate_finding_drops_unchanged_file_under_strict_policy(tmp_path: Path) -> None:
    """Path is not in the PR diff → comment would 422 on GitHub.

    Under the strict ``drop`` policy this still results in a
    :class:`DroppedComment` so callers that explicitly opt out of the
    remap fallback get the original behaviour back.
    """
    repo_root = tmp_path
    abs_path = str((repo_root / "not-in-diff.py").resolve())
    file_maps = build_anchor_maps(
        [
            _FakeChangedFile("changed.py", "@@ -0,0 +1,1 @@\n+only\n"),
        ]
    )

    finding = _make_finding(abs_path=abs_path, start_line=1)

    outcome = validate_finding_against_diff(
        finding, file_maps, {}, repo_root, fallback_policy="drop"
    )

    assert isinstance(outcome, DroppedComment)
    assert outcome.reason == "missing_file_map"


def test_validate_finding_drops_when_diff_has_no_addable_lines_under_strict_policy(
    tmp_path: Path,
) -> None:
    """Path is in the diff but the patch has no addable HEAD lines.

    A deletion-only diff has no ``+`` or context (`` ``) rows on the
    HEAD side, so there is no line on which a GitHub inline comment
    could anchor. Under the ``drop`` policy the validator must drop
    with ``missing_anchor`` (the default ``remap_to_general`` policy
    is exercised in dedicated tests below).
    """
    repo_root = tmp_path
    filename = "deleted.py"
    abs_path = str((repo_root / filename).resolve())
    # Pure deletion: HEAD side is empty, so there is no anchorable line.
    patch = "@@ -1,2 +1,0 @@\n-foo\n-bar\n"
    file_maps = build_anchor_maps([_FakeChangedFile(filename, patch)])

    finding = _make_finding(abs_path=abs_path, start_line=1)

    outcome = validate_finding_against_diff(
        finding, file_maps, {}, repo_root, fallback_policy="drop"
    )

    assert isinstance(outcome, DroppedComment)
    assert outcome.reason == "missing_anchor"


# ---------------------------------------------------------------------------
# Sub-AC 3.3.2 — remap/fallback strategy for non-anchorable comments
# ---------------------------------------------------------------------------


def test_validate_finding_remaps_unchanged_file_to_general_pr_comment(tmp_path: Path) -> None:
    """Path not in PR diff → default policy remaps to a general PR comment.

    The reviewer's signal must not be silently lost just because the
    model picked the wrong file. The default ``remap_to_general``
    policy turns the finding into a :class:`RemappedComment` so the
    poster can publish it as an issue comment.
    """
    repo_root = tmp_path
    abs_path = str((repo_root / "not-in-diff.py").resolve())
    file_maps = build_anchor_maps(
        [
            _FakeChangedFile("changed.py", "@@ -0,0 +1,1 @@\n+only\n"),
        ]
    )

    finding = _make_finding(
        abs_path=abs_path,
        start_line=1,
        title="Worth keeping",
        body="here is the actual finding text",
    )

    outcome = validate_finding_against_diff(finding, file_maps, {}, repo_root)

    assert isinstance(outcome, RemappedComment)
    assert outcome.reason == "missing_file_map"
    assert outcome.relative_path == "not-in-diff.py"
    assert outcome.location.start_line == 1


def test_validate_finding_remaps_deletion_only_diff_to_general_pr_comment(tmp_path: Path) -> None:
    """No anchorable HEAD line → default policy remaps to a general PR comment."""
    repo_root = tmp_path
    filename = "deleted.py"
    abs_path = str((repo_root / filename).resolve())
    patch = "@@ -1,2 +1,0 @@\n-foo\n-bar\n"
    file_maps = build_anchor_maps([_FakeChangedFile(filename, patch)])

    finding = _make_finding(
        abs_path=abs_path,
        start_line=1,
        title="Concern about removal",
        body="explain the concern",
    )

    outcome = validate_finding_against_diff(finding, file_maps, {}, repo_root)

    assert isinstance(outcome, RemappedComment)
    assert outcome.reason == "missing_anchor"
    assert outcome.relative_path == filename


def test_validate_finding_drops_empty_finding_even_under_remap_policy(tmp_path: Path) -> None:
    """An empty finding (blank title + body) cannot be remapped meaningfully.

    Even with ``remap_to_general`` selected, a finding with no usable
    content has nothing to surface in a general PR comment, so it is
    dropped with ``empty_content`` instead of producing a hollow
    issue-comment post.
    """
    repo_root = tmp_path
    abs_path = str((repo_root / "ghost.py").resolve())
    file_maps: dict = {}

    finding = _make_finding(
        abs_path=abs_path,
        start_line=1,
        title="",
        body="   ",
    )

    outcome = validate_finding_against_diff(finding, file_maps, {}, repo_root)

    assert isinstance(outcome, DroppedComment)
    assert outcome.reason == "empty_content"


def test_validate_finding_records_remap_in_debug_logger(tmp_path: Path) -> None:
    """The remap path must produce a log line operators can grep for."""
    repo_root = tmp_path
    abs_path = str((repo_root / "ghost.py").resolve())
    file_maps = build_anchor_maps(
        [
            _FakeChangedFile("changed.py", "@@ -0,0 +1,1 @@\n+only\n"),
        ]
    )

    finding = _make_finding(
        abs_path=abs_path,
        start_line=1,
        title="Worth keeping",
        body="body",
    )

    log_lines: list[tuple[int, str]] = []

    def _capture(level: int, message: str) -> None:
        log_lines.append((level, message))

    outcome = validate_finding_against_diff(
        finding,
        file_maps,
        {},
        repo_root,
        debug=_capture,
    )

    assert isinstance(outcome, RemappedComment)
    assert any(
        "remapping finding (missing_file_map)" in message and "general PR issue comment" in message
        for _level, message in log_lines
    )


def test_validate_finding_applies_rename_map(tmp_path: Path) -> None:
    """Renamed files map old → new path before looking up the diff hunks."""
    repo_root = tmp_path
    filename_new = "new.py"
    filename_old = "old.py"
    abs_path = str((repo_root / filename_old).resolve())

    # The diff is keyed on the *new* filename; the model still references
    # the old path. Rename map must reroute it.
    patch = "@@ -0,0 +1,2 @@\n+a\n+b\n"
    file_maps = build_anchor_maps([_FakeChangedFile(filename_new, patch)])

    finding = _make_finding(abs_path=abs_path, start_line=1)

    outcome = validate_finding_against_diff(
        finding, file_maps, {filename_old: filename_new}, repo_root
    )

    assert isinstance(outcome, ValidatedComment)
    assert outcome.relative_path == filename_new


def test_validate_finding_promotes_range_for_suggestion_block(tmp_path: Path) -> None:
    """A multi-line ```suggestion body should produce a RangeAnchor."""
    repo_root = tmp_path
    filename = "range.py"
    abs_path = str((repo_root / filename).resolve())
    patch = "@@ -0,0 +1,5 @@\n+aaa\n+bbb\n+ccc\n+ddd\n+eee\n"
    file_maps = build_anchor_maps([_FakeChangedFile(filename, patch)])

    finding = _make_finding(
        abs_path=abs_path,
        start_line=2,
        end_line=4,
        body="please rewrite\n```suggestion\nXXX\nYYY\nZZZ\n```",
    )

    outcome = validate_finding_against_diff(finding, file_maps, {}, repo_root)

    assert isinstance(outcome, ValidatedComment)
    assert outcome.has_suggestion is True
    assert isinstance(outcome.anchor, RangeAnchor)
    assert outcome.anchor.start_line == 2
    assert outcome.anchor.end_line == 4
    assert outcome.is_range_anchor is True


# ---------------------------------------------------------------------------
# validate_findings_against_diff — bulk validation result shape
# ---------------------------------------------------------------------------


def test_validate_findings_aggregates_buckets_and_counts_under_strict_policy(
    tmp_path: Path,
) -> None:
    """Under ``drop`` policy, non-anchorable findings still go to ``dropped``."""
    repo_root = tmp_path
    filename = "f.py"
    deletion_only = "gone.py"
    abs_in_diff = str((repo_root / filename).resolve())
    abs_out_of_diff = str((repo_root / "ghost.py").resolve())
    abs_deletion = str((repo_root / deletion_only).resolve())

    file_maps = build_anchor_maps(
        [
            _FakeChangedFile(filename, "@@ -0,0 +1,2 @@\n+a\n+b\n"),
            # Deletion-only diff: filename is in the PR but no anchorable
            # HEAD line exists, so any finding against it must drop with
            # missing_anchor (cannot post a 422-free inline comment).
            _FakeChangedFile(deletion_only, "@@ -1,2 +1,0 @@\n-x\n-y\n"),
        ]
    )

    findings = [
        _make_finding(abs_path=abs_in_diff, start_line=1, title="A"),  # valid
        _make_finding(abs_path=abs_out_of_diff, start_line=1, title="B"),  # missing_file_map
        _make_finding(abs_path=abs_deletion, start_line=1, title="C"),  # missing_anchor
    ]

    result = validate_findings_against_diff(
        findings, file_maps, {}, repo_root, fallback_policy="drop"
    )

    assert isinstance(result, CommentValidationResult)
    assert result.total_count == 3
    assert len(result.valid) == 1
    assert result.valid[0].finding.title == "A"

    assert result.dropped_count == 2
    assert result.dropped_missing_file_map == 1
    assert result.dropped_missing_anchor == 1
    assert result.remapped_count == 0

    # Order is preserved within each bucket so callers can correlate
    # the report back to the original review run.
    assert [d.finding.title for d in result.dropped] == ["B", "C"]
    assert [d.reason for d in result.dropped] == [
        "missing_file_map",
        "missing_anchor",
    ]


def test_validate_findings_aggregates_remapped_bucket_under_default_policy(
    tmp_path: Path,
) -> None:
    """Default policy splits non-anchorable findings into the ``remapped`` bucket."""
    repo_root = tmp_path
    filename = "f.py"
    deletion_only = "gone.py"
    abs_in_diff = str((repo_root / filename).resolve())
    abs_out_of_diff = str((repo_root / "ghost.py").resolve())
    abs_deletion = str((repo_root / deletion_only).resolve())

    file_maps = build_anchor_maps(
        [
            _FakeChangedFile(filename, "@@ -0,0 +1,2 @@\n+a\n+b\n"),
            _FakeChangedFile(deletion_only, "@@ -1,2 +1,0 @@\n-x\n-y\n"),
        ]
    )

    findings = [
        _make_finding(abs_path=abs_in_diff, start_line=1, title="A"),  # valid
        _make_finding(
            abs_path=abs_out_of_diff, start_line=1, title="B"
        ),  # remap (missing_file_map)
        _make_finding(abs_path=abs_deletion, start_line=1, title="C"),  # remap (missing_anchor)
    ]

    result = validate_findings_against_diff(findings, file_maps, {}, repo_root)

    assert result.total_count == 3
    assert len(result.valid) == 1
    assert result.dropped_count == 0
    assert result.remapped_count == 2
    assert result.remapped_missing_file_map == 1
    assert result.remapped_missing_anchor == 1
    assert [r.finding.title for r in result.remapped] == ["B", "C"]
    assert [r.reason for r in result.remapped] == [
        "missing_file_map",
        "missing_anchor",
    ]


def test_validate_findings_returns_empty_result_when_no_findings(tmp_path: Path) -> None:
    result = validate_findings_against_diff([], {}, {}, tmp_path)
    assert result.total_count == 0
    assert result.dropped_count == 0
    assert result.valid == []
    assert result.dropped == []


# ---------------------------------------------------------------------------
# parse_and_validate_findings — combined end-to-end pipeline
# ---------------------------------------------------------------------------


def test_parse_and_validate_findings_returns_review_and_validation(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    filename = "sample.py"
    abs_path = str((repo_root / filename).resolve())

    patch = "@@ -0,0 +1,3 @@\n+foo\n+bar\n+baz\n"
    file_maps = build_anchor_maps([_FakeChangedFile(filename, patch)])

    payload = _conformant_payload(abs_path=abs_path)

    review, validation = parse_and_validate_findings(
        payload,
        file_maps,
        {},
        repo_root,
    )

    assert isinstance(review, ReviewRunResult)
    assert len(review.findings) == 1
    # The single finding anchors on line 2 in the diff.
    assert len(validation.valid) == 1
    assert isinstance(validation.valid[0].anchor, SingleAnchor)
    assert validation.valid[0].anchor.line == 2
    assert validation.dropped_count == 0


def test_parse_and_validate_findings_propagates_schema_errors(tmp_path: Path) -> None:
    bad_payload: dict[str, Any] = {
        "findings": [],
        # Missing carried_forward, etc. — schema must reject before we
        # ever try to validate against the diff.
    }
    with pytest.raises(ReviewContractError):
        parse_and_validate_findings(bad_payload, {}, {}, tmp_path)


def test_parse_and_validate_findings_drops_findings_outside_diff_under_strict_policy(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    abs_path = str((repo_root / "ghost.py").resolve())

    payload = _conformant_payload(abs_path=abs_path)

    # No changed files in the diff at all → strict policy drops with
    # missing_file_map, not raise.
    review, validation = parse_and_validate_findings(
        payload, {}, {}, repo_root, fallback_policy="drop"
    )

    assert len(review.findings) == 1  # parsed successfully
    assert len(validation.valid) == 0
    assert validation.dropped_missing_file_map == 1
    assert validation.remapped_count == 0


def test_parse_and_validate_findings_remaps_findings_outside_diff_by_default(
    tmp_path: Path,
) -> None:
    """Default fallback policy must surface non-anchorable findings as remaps."""
    repo_root = tmp_path
    abs_path = str((repo_root / "ghost.py").resolve())

    payload = _conformant_payload(abs_path=abs_path)

    review, validation = parse_and_validate_findings(payload, {}, {}, repo_root)

    assert len(review.findings) == 1
    assert len(validation.valid) == 0
    assert validation.dropped_count == 0
    assert validation.remapped_count == 1
    assert validation.remapped[0].reason == "missing_file_map"
    assert validation.remapped[0].relative_path == "ghost.py"
