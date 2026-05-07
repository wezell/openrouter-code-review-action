from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from cli.clients.github_client import GitHubClientProtocol
from cli.core.config import ReviewConfig
from cli.core.exceptions import GitHubAPIError
from cli.core.models import (
    InlineCommentPayload,
    ReviewFinding,
    ReviewFindingLocation,
    ReviewRunResult,
)
from cli.review.posting import post_inline_comments
from cli.workflows.review_workflow import ReviewWorkflow


class FakeRequester:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    def requestJsonAndCheck(self, method: str, url: str, input: dict | None = None):  # noqa: N802
        self.calls.append((method, url, input))
        return {}


class FakeIssueComment:
    def __init__(self, body: str, *, comment_id: int = 1) -> None:
        self.body = body
        self.id = comment_id
        self.deleted = False
        self.created_at = "now"
        self.user = None

    def delete(self) -> None:
        self.deleted = True


class FakeChangedFile:
    def __init__(self, filename: str, patch: str) -> None:
        self.filename = filename
        self.patch = patch


class FakeIssueAdapter:
    """Stand-in for the ``Issue`` returned by ``PullRequest.as_issue()``.

    The production ``GitHubClient.post_issue_comment`` path calls
    ``pr.as_issue().create_comment(text)``; the fake just records the
    posted bodies so the remap path can be asserted in tests.
    """

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def create_comment(self, body: str) -> object:
        self._sink.append(body)
        return object()


class FakePR:
    def __init__(self, url: str, issue_comments: list[FakeIssueComment] | None = None) -> None:
        self.url = url
        self._requester = FakeRequester()
        self._issue_comments = issue_comments or []
        self.posted_issue_comments: list[str] = []
        self.number = 1

    # Methods used by the processor
    def get_issue_comments(self):
        return list(self._issue_comments)

    def get_review_comments(self):  # not used by these tests
        return []

    def get_reviews(
        self,
    ):  # used by _has_prior_codex_review; keep empty to avoid semantic dedup
        return []

    def as_issue(self) -> FakeIssueAdapter:
        return FakeIssueAdapter(self.posted_issue_comments)


def make_config() -> ReviewConfig:
    return ReviewConfig.from_args(
        github_token="t",
        repository="o/r",
        pr_number=1,
        openai_api_key="test-key",
        debug_level=0,
    )


def test_skips_summary_only_review_when_no_inline_comments(tmp_path: Path):
    config = make_config()
    rp = ReviewWorkflow(config)

    # No prior markers to avoid semantic dedup/model calls
    pr = FakePR(url="https://api.github.com/repos/o/r/pulls/1")

    result = ReviewRunResult.from_payload(
        {
            "findings": [],
            "carried_forward": [],
            "overall_correctness": "patch is correct",
            "overall_explanation": "",
            "overall_confidence_score": None,
        }
    )

    # changed_files empty -> file_maps empty
    outcome = rp._post_results(
        result,
        changed_files=[],
        pr=cast(Any, pr),
        head_sha="deadbeef",
        rename_map={},
    )

    # Ensure we did not POST a review (no summary-only reviews allowed)
    assert len(pr._requester.calls) == 0
    assert outcome.as_dict() == {
        "total_findings": 0,
        "prefiltered_count": 0,
        "publishable_count": 0,
        "published_count": 0,
        "remapped_count": 0,
        "remapped_published_count": 0,
        "dropped_count": 0,
        "dry_run": False,
        "drop_reasons": "",
        "remap_reasons": "",
        "batch_submitted": False,
        "per_comment_fallback": False,
        "skipped_after_422": 0,
    }


def test_creates_batched_review_with_inline_comment(tmp_path: Path):
    """Sub-AC 3.3.3: inline comments ship as a single batched PR review.

    The validated payload should be embedded inside a single
    ``POST /pulls/{n}/reviews`` request rather than going through the
    per-comment ``POST /pulls/{n}/comments`` endpoint. The per-comment
    endpoint is reserved for the 422-fallback ladder exercised by a
    separate test.
    """
    config = make_config()
    rp = ReviewWorkflow(config)

    pr = FakePR(url="https://api.github.com/repos/o/r/pulls/2")

    # Prepare a simple patch for sample.py with 3 added lines
    filename = "sample.py"
    patch = "@@ -0,0 +1,3 @@\n+foo\n+bar\n+baz\n"
    changed_files = [FakeChangedFile(filename, patch)]

    # Absolute path under current working directory
    abs_path = str((Path.cwd() / filename).resolve())

    result = ReviewRunResult.from_payload(
        {
            "overall_correctness": "patch is incorrect",
            "overall_explanation": "example",
            "overall_confidence_score": None,
            "carried_forward": [],
            "findings": [
                {
                    "title": "Example finding",
                    "body": "Please adjust this line.",
                    "confidence_score": None,
                    "priority": None,
                    "code_location": {
                        "absolute_file_path": abs_path,
                        "line_range": {"start": 2, "end": 2},
                    },
                }
            ],
        }
    )

    outcome = rp._post_results(
        result,
        changed_files=cast(list[Any], changed_files),
        pr=cast(Any, pr),
        head_sha="cafebabe",
        rename_map={},
    )

    # Exactly one POST: a single batched review with the inline comment
    # embedded inside the ``comments`` array.
    calls = pr._requester.calls
    assert len(calls) == 1
    method, url, payload = calls[0]
    assert method == "POST"
    assert url.endswith("/reviews")
    assert isinstance(payload, dict)
    assert payload.get("commit_id") == "cafebabe"
    assert payload.get("event") == "COMMENT"
    inline_comments = payload.get("comments")
    assert isinstance(inline_comments, list) and len(inline_comments) == 1
    assert inline_comments[0]["path"] == filename
    assert inline_comments[0]["line"] == 2
    assert outcome.publishable_count == 1
    assert outcome.published_count == 1
    assert outcome.batch_submitted is True
    assert outcome.per_comment_fallback is False
    assert outcome.skipped_after_422 == 0


def test_post_results_reports_remapped_findings(tmp_path: Path, capsys) -> None:
    """Default fallback policy: non-anchorable findings remap to general PR comments.

    The two findings below have non-empty titles but reference files
    that aren't in the diff. Under Sub-AC 3.3.2 the validator should
    remap them to general PR comments rather than drop them, and the
    workflow should post one issue comment per remapped finding plus a
    line in the operator log explaining the count.
    """
    config = make_config()
    rp = ReviewWorkflow(config)

    pr = FakePR(url="https://api.github.com/repos/o/r/pulls/3")

    outcome = rp._post_results(
        ReviewRunResult(
            overall_correctness="patch is incorrect",
            overall_explanation="example",
            overall_confidence_score=None,
            carried_forward=[],
            findings=[
                ReviewFinding(
                    title="Unknown file A",
                    body="something useful",
                    confidence_score=None,
                    priority=None,
                    code_location=ReviewFindingLocation(
                        absolute_file_path=str((tmp_path / "unknown-a.py").resolve()),
                        start_line=1,
                        end_line=1,
                    ),
                ),
                ReviewFinding.from_mapping(
                    {
                        "title": "Unknown file B",
                        "body": "another useful body",
                        "confidence_score": None,
                        "priority": None,
                        "code_location": {
                            "absolute_file_path": str((tmp_path / "unknown-b.py").resolve()),
                            "line_range": {"start": 1, "end": 1},
                        },
                    }
                ),
            ],
        ),
        changed_files=[],
        pr=cast(Any, pr),
        head_sha="cafebabe",
        rename_map={},
    )

    out = capsys.readouterr().out
    assert "Posting remapped 2/2 findings as general PR comments" in out
    assert "missing file map=2" in out
    assert outcome.dropped_count == 0
    assert outcome.remapped_count == 2
    assert outcome.remapped_published_count == 2
    assert outcome.describe_remaps() == "missing file map=2"

    # Each remapped finding should be posted as exactly one issue
    # comment via the as_issue().create_comment() path.
    assert len(pr.posted_issue_comments) == 2
    for body in pr.posted_issue_comments:
        # Marker proves the body went through the remap formatter, not
        # the inline-comment formatter.
        assert "<!-- openrouter-review:remapped-finding -->" in body
        # The "Original target" header preserves the location for
        # human reviewers despite the loss of the inline anchor.
        assert "Original target:" in body


def test_post_results_drops_empty_finding_with_logging(tmp_path: Path, capsys) -> None:
    """A finding with no title and no body should be dropped, not remapped.

    Even if the file isn't in the diff, an empty finding cannot be
    turned into a useful general PR comment, so the validator drops it
    with ``empty_content``. The log line still surfaces the drop so
    the operator can see what happened.
    """
    config = make_config()
    rp = ReviewWorkflow(config)

    pr = FakePR(url="https://api.github.com/repos/o/r/pulls/4")

    outcome = rp._post_results(
        ReviewRunResult(
            overall_correctness="patch is incorrect",
            overall_explanation="example",
            overall_confidence_score=None,
            carried_forward=[],
            findings=[
                ReviewFinding(
                    title="",
                    body="",
                    confidence_score=None,
                    priority=None,
                    code_location=ReviewFindingLocation(
                        absolute_file_path=str((tmp_path / "ghost.py").resolve()),
                        start_line=1,
                        end_line=1,
                    ),
                ),
            ],
        ),
        changed_files=[],
        pr=cast(Any, pr),
        head_sha="cafebabe",
        rename_map={},
    )

    out = capsys.readouterr().out
    assert "Posting dropped 1/1 findings before GitHub comment creation" in out
    assert "empty content=1" in out
    assert outcome.dropped_count == 1
    assert outcome.remapped_count == 0
    # No inline POST and no issue-comment POST should have been made.
    assert pr._requester.calls == []
    assert pr.posted_issue_comments == []


# ---------------------------------------------------------------------------
# Sub-AC 3.3.3 — batched submission + 422 retry fallback
# ---------------------------------------------------------------------------


class _FakeBatchPR:
    """Pull request stand-in for direct ``post_inline_comments`` testing.

    Tests below drive ``post_inline_comments`` directly so we can mock
    different GitHub-client behaviours (batch success, batch 422, etc.)
    without going through the whole review workflow. The PR object only
    needs to expose the attributes that the real github client and
    error formatter touch.
    """

    def __init__(self) -> None:
        self.number = 99
        self.url = "https://api.github.com/repos/o/r/pulls/99"


class _BatchFakeClient:
    """Fake GitHub client that records both batched and per-comment calls.

    Behaviour is configurable: ``batch_status`` = 422 to simulate the
    review API rejecting the bundle (triggering the per-comment
    fallback). ``per_comment_422_for`` lists ``(path, line)`` pairs that
    should 422 individually during the fallback retry. Anything else is
    accepted, mirroring the real API.
    """

    def __init__(
        self,
        *,
        batch_status: int | None = None,
        per_comment_422_for: tuple[tuple[str, int], ...] = (),
    ) -> None:
        self.batch_status = batch_status
        self.per_comment_422_for = set(per_comment_422_for)
        self.batch_calls: list[
            tuple[Any, tuple[InlineCommentPayload, ...], str]
        ] = []
        self.inline_calls: list[tuple[Any, InlineCommentPayload, str]] = []

    def post_pull_request_review(
        self,
        pr: Any,
        payloads: Any,
        *,
        head_sha: str,
    ) -> None:
        materialised = tuple(payloads)
        self.batch_calls.append((pr, materialised, head_sha))
        if self.batch_status is not None:
            raise GitHubAPIError(
                "Validation Failed: Pull request review is not anchored",
                status_code=self.batch_status,
            )

    def post_inline_comment(
        self,
        pr: Any,
        payload: InlineCommentPayload,
        *,
        head_sha: str,
    ) -> None:
        self.inline_calls.append((pr, payload, head_sha))
        if (payload.path, payload.line) in self.per_comment_422_for:
            raise GitHubAPIError(
                f"422 unprocessable: {payload.path}:{payload.line}",
                status_code=422,
            )


def _payload(path: str, line: int) -> InlineCommentPayload:
    return InlineCommentPayload(body=f"finding @ {path}:{line}", path=path, line=line)


def test_post_inline_comments_batches_into_single_review_submission() -> None:
    """Happy path: every inline payload ships in a single batched review."""
    client = _BatchFakeClient()
    pr = _FakeBatchPR()
    payloads = [_payload("a.py", 1), _payload("b.py", 5)]

    debug_messages: list[str] = []
    result = post_inline_comments(
        cast(GitHubClientProtocol, client),
        cast(Any, pr),
        "deadbeef",
        payloads,
        dry_run=False,
        debug=lambda level, message: debug_messages.append(f"{level}:{message}"),
    )

    # Exactly one batched call, no per-comment fallback triggered.
    assert len(client.batch_calls) == 1
    assert client.batch_calls[0][1] == tuple(payloads)
    assert client.batch_calls[0][2] == "deadbeef"
    assert client.inline_calls == []
    assert result.attempted_count == 2
    assert result.posted_count == 2
    assert result.batch_submitted is True
    assert result.per_comment_fallback is False
    assert result.skipped_after_422 == 0


def test_post_inline_comments_falls_back_to_per_comment_on_422() -> None:
    """422 from the batch should trigger the per-comment retry fallback."""
    client = _BatchFakeClient(batch_status=422)
    pr = _FakeBatchPR()
    payloads = [_payload("a.py", 1), _payload("b.py", 5), _payload("c.py", 9)]

    debug_messages: list[str] = []
    result = post_inline_comments(
        cast(GitHubClientProtocol, client),
        cast(Any, pr),
        "deadbeef",
        payloads,
        dry_run=False,
        debug=lambda level, message: debug_messages.append(f"{level}:{message}"),
    )

    assert len(client.batch_calls) == 1
    # Every payload was retried via the per-comment endpoint.
    assert [call[1] for call in client.inline_calls] == payloads
    assert result.attempted_count == 3
    assert result.posted_count == 3
    assert result.batch_submitted is False
    assert result.per_comment_fallback is True
    assert result.skipped_after_422 == 0
    # The fallback transition should be visible in the debug log so an
    # operator can trace why the per-comment endpoint started firing.
    assert any(
        "Batched PR review rejected with 422" in message for message in debug_messages
    )


def test_post_inline_comments_drops_individual_422_during_fallback() -> None:
    """Per-comment 422s during the fallback are isolated and counted."""
    client = _BatchFakeClient(
        batch_status=422,
        per_comment_422_for=(("b.py", 5),),
    )
    pr = _FakeBatchPR()
    payloads = [_payload("a.py", 1), _payload("b.py", 5), _payload("c.py", 9)]

    debug_messages: list[str] = []
    result = post_inline_comments(
        cast(GitHubClientProtocol, client),
        cast(Any, pr),
        "deadbeef",
        payloads,
        dry_run=False,
        debug=lambda level, message: debug_messages.append(f"{level}:{message}"),
    )

    # The bad comment was skipped; the other two succeeded.
    assert len(client.inline_calls) == 3
    assert result.attempted_count == 3
    assert result.posted_count == 2
    assert result.skipped_after_422 == 1
    assert result.per_comment_fallback is True
    assert any("Skipping inline comment for b.py:5" in m for m in debug_messages)


def test_post_inline_comments_propagates_non_422_batch_errors() -> None:
    """Non-422 batch errors must NOT silently degrade to per-comment retries."""
    client = _BatchFakeClient(batch_status=500)
    pr = _FakeBatchPR()
    payloads = [_payload("a.py", 1)]

    with pytest.raises(GitHubAPIError) as exc_info:
        post_inline_comments(
            cast(GitHubClientProtocol, client),
            cast(Any, pr),
            "deadbeef",
            payloads,
            dry_run=False,
            debug=lambda level, message: None,
        )
    assert exc_info.value.status_code == 500
    # Per-comment fallback should NOT have run for non-422 failures.
    assert client.inline_calls == []


def test_post_inline_comments_dry_run_skips_network() -> None:
    """Dry run logs intended posts and never touches either endpoint."""
    client = _BatchFakeClient()
    pr = _FakeBatchPR()
    payloads = [_payload("a.py", 1), _payload("b.py", 5)]

    debug_messages: list[str] = []
    result = post_inline_comments(
        cast(GitHubClientProtocol, client),
        cast(Any, pr),
        "deadbeef",
        payloads,
        dry_run=True,
        debug=lambda level, message: debug_messages.append(f"{level}:{message}"),
    )

    assert client.batch_calls == []
    assert client.inline_calls == []
    assert result.dry_run is True
    assert result.attempted_count == 2
    assert result.posted_count == 0
    assert any("DRY_RUN: would batch 2 inline comment(s)" in m for m in debug_messages)
