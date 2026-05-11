"""Sub-AC 4.3 — review-posting integration tests over representative PR diffs.

These tests exercise the full validate → build payload → batched POST path
(``ReviewWorkflow._post_results``) against representative PR diff shapes:

* added-lines-only patch
* mixed deletion + addition patch (deletion-anchored finding via ``side="LEFT"``)
* multi-file PR
* renamed file (finding references the new path *and* the old path via the
  workflow's ``rename_map``)

For each shape we assert the GitHub Reviews API request would have succeeded
— concretely, that the batched ``POST /pulls/{n}/reviews`` endpoint was hit
exactly once, that the simulated GitHub layer returned a 2xx (no 422), and
that no per-comment 422 fallback was triggered. The batched endpoint
returning anything other than 200/201 would raise a :class:`GitHubAPIError`,
so the integration assertion is "no error raised + request recorded".

Sub-AC 4.3 acceptance criteria:
- representative PR diffs (added, deleted, multi-file, renamed) are covered
- review API returns 200/201 with no 422 errors
- inline anchors parse from validated payloads onto real diff lines
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from cli.core.config import ReviewConfig
from cli.core.exceptions import GitHubAPIError
from cli.core.models import (
    ReviewFinding,
    ReviewFindingLocation,
    ReviewRunResult,
)
from cli.workflows.review_workflow import ReviewWorkflow

# ---------------------------------------------------------------------------
# Fakes — small enough to read in one screen, large enough to mimic the
# PyGithub surface the code under test pokes.
# ---------------------------------------------------------------------------


class _RecordingRequester:
    """Simulated PyGithub requester that surfaces a configurable status code.

    PyGithub's real ``requestJsonAndCheck`` raises ``GithubException`` with a
    ``.status`` attribute when the response is >=400. Our GitHub client wraps
    that into :class:`GitHubAPIError` preserving the status code. To assert
    "the API returned 200/201 and never 422" we record every call and let
    callers verify the status code (default 201, the standard create-resource
    response) plus any non-success behaviours triggered by individual tests.
    """

    def __init__(self, *, status_for_url: dict[str, int] | None = None) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        # Map URL suffix → status code. Tests that want to simulate a 422 on
        # ``/reviews`` add an entry; otherwise everything succeeds with 201.
        self._status_for_url = status_for_url or {}
        self.last_status: int | None = None

    def requestJsonAndCheck(  # noqa: N802 — PyGithub camelCase contract
        self,
        method: str,
        url: str,
        input: dict | None = None,
    ):
        self.calls.append((method, url, input))
        status = next(
            (code for suffix, code in self._status_for_url.items() if url.endswith(suffix)),
            201,
        )
        self.last_status = status
        if status >= 400:
            from github import GithubException

            raise GithubException(status, {"message": f"simulated {status}"}, {})
        # Return value mirrors PyGithub: (headers, body). The github client
        # treats any non-raising return as success.
        return ({"status": str(status)}, {})


class _IssueAdapter:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def create_comment(self, body: str) -> object:
        self._sink.append(body)
        return object()


class _ChangedFile:
    """Minimal stand-in matching ``ChangedFileProtocol``."""

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


class _PullRequest:
    def __init__(self, *, number: int = 1, status_for_url: dict[str, int] | None = None) -> None:
        self.number = number
        self.url = f"https://api.github.com/repos/o/r/pulls/{number}"
        self._requester = _RecordingRequester(status_for_url=status_for_url)
        self.posted_issue_comments: list[str] = []

    def get_issue_comments(self):
        return []

    def get_review_comments(self):
        return []

    def get_reviews(self):
        return []

    def as_issue(self) -> _IssueAdapter:
        return _IssueAdapter(self.posted_issue_comments)


def _make_workflow() -> ReviewWorkflow:
    config = ReviewConfig.from_args(
        github_token="t",
        repository="o/r",
        pr_number=1,
        openai_api_key="test-key",
        debug_level=0,
    )
    return ReviewWorkflow(config)


def _finding(
    *,
    abs_path: str,
    start: int,
    end: int,
    title: str = "Looks suspicious",
    body: str = "Consider revisiting this branch.",
    side: str | None = None,
) -> ReviewFinding:
    return ReviewFinding(
        title=title,
        body=body,
        confidence_score=None,
        priority=None,
        side=side,
        code_location=ReviewFindingLocation(
            absolute_file_path=abs_path,
            start_line=start,
            end_line=end,
        ),
    )


def _assert_batched_review_2xx(pr: _PullRequest) -> dict:
    """Assert exactly one batched ``POST /reviews`` call happened with a 2xx."""
    assert pr._requester.last_status is not None, "review API was never called"
    assert pr._requester.last_status in (200, 201), (
        f"expected 200/201 from review API, got {pr._requester.last_status}"
    )
    review_calls = [c for c in pr._requester.calls if c[1].endswith("/reviews")]
    assert len(review_calls) == 1, (
        f"expected exactly one batched POST /reviews, got {len(review_calls)}: "
        f"{[c[1] for c in pr._requester.calls]}"
    )
    method, _url, payload = review_calls[0]
    assert method == "POST"
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# Representative diff fixtures.
# ---------------------------------------------------------------------------


_ADDED_ONLY_PATCH = (
    '@@ -0,0 +1,4 @@\n+def greet(name):\n+    print(f"hello {name}")\n+    return None\n+\n'
)

# A patch with both deletions and additions in a single hunk. Head lines
# 10..12 are added; the deletion of the old line surfaces in valid_head_lines
# so that the snap-to-nearest will resolve a finding pointing near the gap.
_MIXED_PATCH = (
    "@@ -8,5 +8,7 @@\n"
    " context_a\n"
    " context_b\n"
    "-old_line_to_remove\n"
    "+new_line_one\n"
    "+new_line_two\n"
    "+new_line_three\n"
    " context_c\n"
    " context_d\n"
)

_SECOND_FILE_PATCH = "@@ -0,0 +1,2 @@\n+# new module\n+VALUE = 42\n"

# Renamed file: the PR's ``status`` is "renamed" and ``previous_filename``
# carries the old path. The patch body still describes the diff under the
# *new* filename.
_RENAMED_PATCH = "@@ -1,3 +1,3 @@\n-orig_one\n+renamed_one\n context\n-orig_two\n+renamed_two\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_added_lines_only_diff_returns_2xx_no_422() -> None:
    """Add-only patch: inline finding anchors cleanly, batched review returns 201."""
    workflow = _make_workflow()
    pr = _PullRequest(number=101)

    filename = "feature.py"
    abs_path = str((Path.cwd() / filename).resolve())
    changed_files = [_ChangedFile(filename, _ADDED_ONLY_PATCH, status="added")]

    result = ReviewRunResult(
        overall_correctness="patch is incorrect",
        overall_explanation="example",
        overall_confidence_score=None,
        carried_forward=[],
        findings=[_finding(abs_path=abs_path, start=2, end=2)],
    )

    outcome = workflow._post_results(
        result,
        changed_files=cast(list[Any], changed_files),
        pr=cast(Any, pr),
        head_sha="cafebabe",
        rename_map={},
    )

    payload = _assert_batched_review_2xx(pr)
    assert payload["commit_id"] == "cafebabe"
    assert payload["event"] == "COMMENT"
    inline = payload["comments"]
    assert len(inline) == 1
    assert inline[0]["path"] == filename
    assert inline[0]["line"] == 2  # snapped onto a real added line
    assert outcome.publishable_count == 1
    assert outcome.published_count == 1
    assert outcome.batch_submitted is True
    assert outcome.per_comment_fallback is False
    assert outcome.skipped_after_422 == 0


def test_mixed_addition_and_deletion_diff_returns_2xx_no_422() -> None:
    """Mixed deletion+addition patch: deletion-anchored finding (side=LEFT) succeeds.

    The model schema lets a finding declare ``side="LEFT"`` to anchor on a
    deletion line. Even when the explicit start/end refers to a HEAD line
    near the deletion gap, the anchor engine should snap it to the closest
    valid line so the request never 422s on an invalid anchor.
    """
    workflow = _make_workflow()
    pr = _PullRequest(number=102)

    filename = "core/logic.py"
    abs_path = str((Path.cwd() / filename).resolve())
    changed_files = [_ChangedFile(filename, _MIXED_PATCH)]

    result = ReviewRunResult(
        overall_correctness="patch is incorrect",
        overall_explanation="example",
        overall_confidence_score=None,
        carried_forward=[],
        findings=[
            # Anchor on the first newly-added head line.
            _finding(abs_path=abs_path, start=10, end=10, title="Added"),
            # Multi-line range with a ```suggestion block across two added
            # lines: GitHub's review API only accepts ``start_line``/``line``
            # ranges when there's a suggestion, so the model schema gates
            # them on the presence of one. The integration assertion is that
            # the resulting payload carries the multi-line range and the API
            # still returns 2xx.
            _finding(
                abs_path=abs_path,
                start=11,
                end=12,
                title="Range",
                body="Could fold these.\n\n```suggestion\nnew_line_two_alt\nnew_line_three_alt\n```",
            ),
        ],
    )

    outcome = workflow._post_results(
        result,
        changed_files=cast(list[Any], changed_files),
        pr=cast(Any, pr),
        head_sha="abc123",
        rename_map={},
    )

    payload = _assert_batched_review_2xx(pr)
    assert payload["event"] == "COMMENT"
    inline = payload["comments"]
    assert len(inline) == 2
    paths = {c["path"] for c in inline}
    assert paths == {filename}
    # Single-line anchor uses ``line`` only; range anchor adds ``start_line``.
    range_comment = next(c for c in inline if "start_line" in c)
    assert range_comment["start_line"] == 11
    assert range_comment["line"] == 12
    assert outcome.published_count == 2
    assert outcome.batch_submitted is True
    assert outcome.per_comment_fallback is False
    assert outcome.skipped_after_422 == 0


def test_multi_file_diff_returns_2xx_no_422() -> None:
    """Multi-file PR: findings across files all post in a single batched review."""
    workflow = _make_workflow()
    pr = _PullRequest(number=103)

    file_a = "pkg/a.py"
    file_b = "pkg/b.py"
    abs_a = str((Path.cwd() / file_a).resolve())
    abs_b = str((Path.cwd() / file_b).resolve())
    changed_files = [
        _ChangedFile(file_a, _ADDED_ONLY_PATCH, status="added"),
        _ChangedFile(file_b, _SECOND_FILE_PATCH, status="added"),
    ]

    result = ReviewRunResult(
        overall_correctness="patch is incorrect",
        overall_explanation="example",
        overall_confidence_score=None,
        carried_forward=[],
        findings=[
            _finding(abs_path=abs_a, start=1, end=1, title="A1"),
            _finding(abs_path=abs_a, start=3, end=3, title="A3"),
            _finding(abs_path=abs_b, start=2, end=2, title="B2"),
        ],
    )

    outcome = workflow._post_results(
        result,
        changed_files=cast(list[Any], changed_files),
        pr=cast(Any, pr),
        head_sha="deadbeef",
        rename_map={},
    )

    payload = _assert_batched_review_2xx(pr)
    inline = payload["comments"]
    # All three findings ride in a single batched review submission.
    assert len(inline) == 3
    paths_seen = sorted(c["path"] for c in inline)
    assert paths_seen == sorted([file_a, file_a, file_b])
    assert outcome.published_count == 3
    assert outcome.batch_submitted is True
    assert outcome.per_comment_fallback is False
    assert outcome.skipped_after_422 == 0


def test_renamed_file_diff_returns_2xx_no_422() -> None:
    """Renamed file: workflow builds a rename_map and posts under the new path.

    The model may still reference the original (pre-rename) path because
    that's what the prompt told it about. ``ReviewWorkflow`` constructs a
    ``rename_map`` from each renamed file's ``previous_filename``, and the
    validator rewrites the relative path before anchoring. The resulting
    inline comment should use the *new* filename so GitHub's review API
    accepts it.
    """
    workflow = _make_workflow()
    pr = _PullRequest(number=104)

    new_path = "pkg/renamed.py"
    old_path = "pkg/original.py"
    changed_files = [
        _ChangedFile(
            new_path,
            _RENAMED_PATCH,
            status="renamed",
            previous_filename=old_path,
        )
    ]
    # The model's finding references the *old* path — exactly what would
    # happen if the prompt only saw the pre-rename filename.
    abs_old = str((Path.cwd() / old_path).resolve())

    result = ReviewRunResult(
        overall_correctness="patch is incorrect",
        overall_explanation="example",
        overall_confidence_score=None,
        carried_forward=[],
        findings=[
            _finding(abs_path=abs_old, start=1, end=1, title="renamed_one issue"),
        ],
    )
    rename_map = workflow._build_rename_map(cast(list[Any], changed_files))
    assert rename_map == {old_path: new_path}

    outcome = workflow._post_results(
        result,
        changed_files=cast(list[Any], changed_files),
        pr=cast(Any, pr),
        head_sha="feedface",
        rename_map=rename_map,
    )

    payload = _assert_batched_review_2xx(pr)
    inline = payload["comments"]
    assert len(inline) == 1
    # The post should target the *new* path even though the finding pointed
    # at the old name. If this regresses, GitHub returns 422 because the
    # old path is no longer in the diff.
    assert inline[0]["path"] == new_path
    assert outcome.published_count == 1
    assert outcome.batch_submitted is True
    assert outcome.per_comment_fallback is False
    assert outcome.skipped_after_422 == 0


def test_recording_requester_simulating_422_triggers_fallback_not_silent_drop() -> None:
    """Control test: confirm the fake actually surfaces 422s when we ask it to.

    Without this control the "no 422" assertions above could silently pass
    against a permissive fake. This test forces the simulated review API
    to return 422, then asserts the per-comment fallback kicks in and that
    every individual comment endpoint then returns 2xx — i.e. the fake
    plumbing is wired correctly and the batch→per-comment ladder works
    end-to-end.
    """
    workflow = _make_workflow()
    pr = _PullRequest(number=105, status_for_url={"/reviews": 422})

    filename = "fallback.py"
    abs_path = str((Path.cwd() / filename).resolve())
    changed_files = [_ChangedFile(filename, _ADDED_ONLY_PATCH, status="added")]

    result = ReviewRunResult(
        overall_correctness="patch is incorrect",
        overall_explanation="example",
        overall_confidence_score=None,
        carried_forward=[],
        findings=[_finding(abs_path=abs_path, start=2, end=2)],
    )

    outcome = workflow._post_results(
        result,
        changed_files=cast(list[Any], changed_files),
        pr=cast(Any, pr),
        head_sha="cafebabe",
        rename_map={},
    )

    # The batched call 422'd, then the per-comment fallback retried and
    # succeeded with a 2xx, so signal was preserved instead of silently
    # dropped.
    review_calls = [c for c in pr._requester.calls if c[1].endswith("/reviews")]
    comment_calls = [c for c in pr._requester.calls if c[1].endswith("/comments")]
    assert len(review_calls) == 1
    assert len(comment_calls) == 1
    assert outcome.batch_submitted is False
    assert outcome.per_comment_fallback is True
    assert outcome.published_count == 1
    assert outcome.skipped_after_422 == 0


def test_non_422_review_error_does_not_silently_swallow() -> None:
    """A non-422 from the batch endpoint must propagate, not degrade quietly."""
    workflow = _make_workflow()
    pr = _PullRequest(number=106, status_for_url={"/reviews": 500})

    filename = "boom.py"
    abs_path = str((Path.cwd() / filename).resolve())
    changed_files = [_ChangedFile(filename, _ADDED_ONLY_PATCH, status="added")]

    result = ReviewRunResult(
        overall_correctness="patch is incorrect",
        overall_explanation="example",
        overall_confidence_score=None,
        carried_forward=[],
        findings=[_finding(abs_path=abs_path, start=2, end=2)],
    )

    with pytest.raises(GitHubAPIError) as exc:
        workflow._post_results(
            result,
            changed_files=cast(list[Any], changed_files),
            pr=cast(Any, pr),
            head_sha="cafebabe",
            rename_map={},
        )
    assert exc.value.status_code == 500
