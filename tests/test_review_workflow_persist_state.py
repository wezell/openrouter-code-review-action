"""Sub-AC 4.4 — workflow persistence of prior review state after posting.

These tests cover the contract added by Sub-AC 4.4: after a review run
finishes posting and refreshing the run-summary comment, the workflow
must lay down a new prior-review-state envelope under the current head
SHA that **merges** prior findings/IDs into the new record so the next
run sees a single consistent snapshot.

Concept map
-----------

* ``_FakeIssueWithIds`` / ``_FakePRWithIds`` mirror the ``_FakePR``
  shape from ``test_review_workflow.py`` but expose a real
  ``IssueComment`` ``.id`` so the workflow can capture the new summary
  comment ID for persistence.
* ``_make_codex_response`` is a small builder for a structured review
  payload so each test can vary findings without re-rolling the JSON.
* ``_load_persisted`` re-reads the on-disk envelope through the
  manager so we exercise the *file format* — not just the in-memory
  return value.

Each test exercises one slice of the merge contract:

1. First-run creates an envelope keyed by the current head SHA with
   the new findings populated and a stamped run note.
2. A second run on a new head SHA inherits prior findings as
   carried-forward IDs while replacing the findings list with the new
   run's output.
3. dry_run skips persistence entirely.
4. A previously-known summary comment ID survives even when the
   current run cannot capture a new one (test fake returns no id).
5. Findings whose anchor is outside the repo root are skipped instead
   of poisoning the persisted record.
6. Persistence failure does not unwind the rest of the run — the
   review still returns its result and the summary comment still
   posts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from cli.core.config import ReviewConfig
from cli.core.models import ReviewThreadComment, ReviewThreadSnapshot
from cli.core.review_state import (
    PersistedReviewFinding,
    PriorReviewKey,
    PriorReviewState,
    SEVERITY_CRITICAL,
)
from cli.core.review_state_manager import ReviewStateManager
from cli.workflows.review_workflow import ReviewWorkflow

# ---------------------------------------------------------------------------
# Fakes (slimmer copies of the shapes used in test_review_workflow.py, with
# create_comment returning a real id-bearing object).
# ---------------------------------------------------------------------------


@dataclass
class _FakeUser:
    login: str = "octocat"


@dataclass
class _FakeHead:
    sha: str = "head-sha-2"
    ref: str = "feature"
    label: str = "octocat:feature"


@dataclass
class _FakeRepoOwner:
    login: str = "owner"


@dataclass
class _FakeRepo:
    owner: _FakeRepoOwner = field(default_factory=_FakeRepoOwner)
    name: str = "repo"


@dataclass
class _FakeBase:
    ref: str = "main"
    sha: str = "base-sha"
    label: str = "owner:main"
    repo: _FakeRepo = field(default_factory=_FakeRepo)


@dataclass
class _FakeChangedFile:
    filename: str
    status: str = "modified"
    patch: str | None = "@@ -1 +1 @@\n-old\n+new\n"
    previous_filename: str | None = None


class _FakeReviewComment:
    def __init__(self, body: str, *, path: str = "src.py", line: int = 3) -> None:
        self.id = 1
        self.body = body
        self.path = path
        self.line = line
        self.original_line = line
        self.in_reply_to_id = None
        self.diff_hunk = "@@ -1 +1 @@"
        self.commit_id = "head-sha-2"
        self.user = _FakeUser("reviewer")
        self.created_at = "now"


class _FakeIssueComment:
    def __init__(
        self,
        body: str,
        *,
        comment_id: int = 1,
        login: str = "commenter",
    ) -> None:
        self.body = body
        self.id = comment_id
        self.deleted = False
        self.user = _FakeUser(login)
        self.created_at = "now"

    def delete(self) -> None:
        self.deleted = True


@dataclass
class _CreatedIssueComment:
    id: int
    body: str


class _FakeIssueWithIds:
    """``as_issue()`` analog whose ``create_comment`` returns an id."""

    def __init__(self, *, next_id: int = 9999, return_obj: bool = True) -> None:
        self._next_id = next_id
        self.return_obj = return_obj
        self.created_comments: list[_CreatedIssueComment] = []

    def create_comment(self, text: str) -> _CreatedIssueComment | None:
        comment = _CreatedIssueComment(id=self._next_id, body=text)
        self.created_comments.append(comment)
        self._next_id += 1
        if not self.return_obj:
            return None
        return comment


class _FakePRWithIds:
    def __init__(
        self,
        *,
        head_sha: str = "head-sha-2",
        changed_files: list[_FakeChangedFile] | None = None,
        review_comments: list[_FakeReviewComment] | None = None,
        issue_comments: list[_FakeIssueComment] | None = None,
        next_summary_id: int = 9999,
        summary_returns_obj: bool = True,
    ) -> None:
        self.number = 7
        self.title = "Improve review flow"
        self.body = "PR body"
        self.html_url = "https://example.test/pr/7"
        self.state = "open"
        self.url = "https://api.example.test/repos/owner/repo/pulls/7"
        self.user = _FakeUser()
        self.head: _FakeHead | None = _FakeHead(sha=head_sha)
        self.base = _FakeBase()
        self._issue_comments = issue_comments or []
        self._review_comments = review_comments or []
        self._review_threads = [
            ReviewThreadSnapshot(
                id=f"thread-{index}",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id=f"comment-{index}",
                        body=comment.body,
                        path=comment.path,
                        line=comment.line,
                        original_line=comment.original_line,
                        author=comment.user.login,
                    )
                ],
            )
            for index, comment in enumerate(self._review_comments, start=1)
        ]
        self._issue = _FakeIssueWithIds(
            next_id=next_summary_id, return_obj=summary_returns_obj
        )
        self._changed_files = changed_files or [_FakeChangedFile("src.py")]

    def get_files(self) -> list[_FakeChangedFile]:
        return list(self._changed_files)

    def get_issue_comments(self) -> list[_FakeIssueComment]:
        return list(self._issue_comments)

    def get_review_comments(self) -> list[_FakeReviewComment]:
        return list(self._review_comments)

    def get_reviews(self) -> list[Any]:
        return []

    def as_issue(self) -> _FakeIssueWithIds:
        return self._issue


class _FakeGitHubClient:
    def __init__(self, pr: _FakePRWithIds) -> None:
        self.pr = pr

    def get_pr(self, pr_number: int) -> _FakePRWithIds:
        return self.pr

    def get_review_threads(self, pr: _FakePRWithIds) -> list[ReviewThreadSnapshot]:
        return list(self.pr._review_threads)

    def post_inline_comment(self, pr: _FakePRWithIds, payload: Any, *, head_sha: str) -> None:
        pass

    def post_pull_request_review(
        self, pr: _FakePRWithIds, payloads: Any, *, head_sha: str
    ) -> None:
        pass

    def post_issue_comment(self, pr: _FakePRWithIds, text: str) -> None:
        pr.as_issue().create_comment(text)


class _FakeCodexClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def execute_structured(
        self,
        prompt: str,
        *,
        output_schema: dict[str, object],
        schema_prompt: str,
        sandbox_mode: str,
        resume_thread_id: str | None = None,
    ) -> str:
        return self.response


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, *, dry_run: bool = False) -> ReviewConfig:
    return ReviewConfig(
        github_token="token",
        repository="owner/repo",
        pr_number=7,
        mode="review",
        dry_run=dry_run,
        repo_root=tmp_path,
        review_model="anthropic/claude-opus-4.7",
    )


def _codex_response(
    *,
    findings: list[dict[str, Any]],
    overall: str = "patch is incorrect",
    explanation: str = "Needs one fix.",
    carried_forward: list[dict[str, str]] | None = None,
) -> str:
    return json.dumps(
        {
            "overall_correctness": overall,
            "overall_explanation": explanation,
            "overall_confidence_score": 0.9,
            "carried_forward": carried_forward or [],
            "findings": findings,
        }
    )


def _finding_payload(
    *,
    sample_file: Path,
    title: str = "Example finding",
    body: str = "Details",
    priority: int = 1,
    line: int = 1,
) -> dict[str, Any]:
    return {
        "title": title,
        "body": body,
        "confidence_score": 0.9,
        "priority": priority,
        "code_location": {
            "absolute_file_path": str(sample_file.resolve()),
            "line_range": {"start": line, "end": line},
        },
    }


def _load_persisted(
    runner_temp: Path, key: PriorReviewKey
) -> PriorReviewState | None:
    manager = ReviewStateManager.from_runner_temp(str(runner_temp))
    return manager.store.load(key)


def _key(*, head_sha: str, model: str = "anthropic/claude-opus-4.7") -> PriorReviewKey:
    return PriorReviewKey(
        repository="owner/repo",
        pr_number=7,
        review_model=model,
        reviewed_head_sha=head_sha,
    )


# ---------------------------------------------------------------------------
# 1. First-run creates an envelope on disk
# ---------------------------------------------------------------------------


def test_first_run_persists_state_under_current_head_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    sample_file = tmp_path / "src.py"
    sample_file.write_text("old\nnew\n", encoding="utf-8")

    pr = _FakePRWithIds(
        head_sha="sha-A",
        changed_files=[
            _FakeChangedFile(
                "src.py",
                patch="@@ -1,1 +1,2 @@\n old\n+new\n",
            )
        ],
        next_summary_id=4242,
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        codex_client=cast(
            Any,
            _FakeCodexClient(
                _codex_response(
                    findings=[
                        _finding_payload(
                            sample_file=sample_file,
                            title="Bug A",
                            body="Body A",
                            priority=1,
                            line=2,
                        ),
                    ]
                )
            ),
        ),
    )

    result = workflow.process_review(7)
    assert result.review.findings[0].title == "Bug A"

    persisted = _load_persisted(runner_temp, _key(head_sha="sha-A"))
    assert persisted is not None
    assert persisted.key.reviewed_head_sha == "sha-A"
    assert persisted.summary_comment_id == 4242
    assert len(persisted.findings) == 1
    finding = persisted.findings[0]
    assert finding.path == "src.py"
    assert finding.start_line == 2
    assert finding.end_line == 2
    assert finding.title == "Bug A"
    assert finding.severity == SEVERITY_CRITICAL  # priority 1 → critical
    assert persisted.carried_forward_comment_ids == ()
    assert any(note.startswith("openrouter-review:run") for note in persisted.notes)


# ---------------------------------------------------------------------------
# 2. Second run merges prior findings into carried-forward IDs
# ---------------------------------------------------------------------------


def test_second_run_merges_prior_findings_into_carried_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    # Skip the legacy CODEX_HOME resume path so the test focuses on
    # the new state-manager persistence flow.
    monkeypatch.setenv("CODEX_REVIEW_CACHE_HIT", "false")

    # Seed an existing envelope under sha-A with one published finding
    # so the second run has something to carry forward.
    seeded_key = _key(head_sha="sha-A")
    seeded_manager = ReviewStateManager.from_runner_temp(str(runner_temp))
    seeded_manager.record_review_run(
        seeded_key,
        findings=(
            PersistedReviewFinding(
                path="src.py",
                start_line=2,
                end_line=2,
                title="Bug A",
                body="Body A",
                severity=SEVERITY_CRITICAL,
                side="RIGHT",
                priority=1,
                inline_comment_id=111,
                review_thread_id="PRRT_111",
            ),
        ),
        carried_forward_comment_ids=("legacy-id",),
        summary_comment_id=4242,
    )

    sample_file = tmp_path / "src.py"
    sample_file.write_text("old\nnew\nnewer\n", encoding="utf-8")

    pr = _FakePRWithIds(
        head_sha="sha-B",
        changed_files=[
            _FakeChangedFile(
                "src.py",
                patch="@@ -1,2 +1,3 @@\n old\n new\n+newer\n",
            )
        ],
        # Surface a prior summary comment so the workflow can resolve
        # the previous SHA hint via the in-memory issue-comment list.
        issue_comments=[
            _FakeIssueComment(
                body=(
                    "Codex Autonomous Review:\n"
                    "<!-- codex-review-meta "
                    '{"reviewed_head_sha":"sha-A"} -->\n'
                    "summary"
                ),
                comment_id=4242,
                login="bot",
            )
        ],
        next_summary_id=5555,
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        codex_client=cast(
            Any,
            _FakeCodexClient(
                _codex_response(
                    findings=[
                        _finding_payload(
                            sample_file=sample_file,
                            title="Bug B",
                            body="Body B",
                            priority=2,
                            line=3,
                        ),
                    ]
                )
            ),
        ),
    )

    workflow.process_review(7)

    persisted = _load_persisted(runner_temp, _key(head_sha="sha-B"))
    assert persisted is not None
    assert persisted.summary_comment_id == 5555
    titles = tuple(f.title for f in persisted.findings)
    assert titles == ("Bug B",)  # findings are *replaced* by the new run
    # carried-forward merges the seeded "legacy-id" with the prior
    # published inline + thread IDs.
    assert "legacy-id" in persisted.carried_forward_comment_ids
    assert "111" in persisted.carried_forward_comment_ids
    assert "PRRT_111" in persisted.carried_forward_comment_ids
    # Ordering is stable (first-seen wins) — legacy-id from prior
    # carried-forward list comes before the published-id rollups.
    legacy_idx = persisted.carried_forward_comment_ids.index("legacy-id")
    inline_idx = persisted.carried_forward_comment_ids.index("111")
    assert legacy_idx < inline_idx


# ---------------------------------------------------------------------------
# 3. dry_run does not write any state
# ---------------------------------------------------------------------------


def test_dry_run_skips_state_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    sample_file = tmp_path / "src.py"
    sample_file.write_text("old\nnew\n", encoding="utf-8")

    pr = _FakePRWithIds(
        head_sha="sha-DR",
        changed_files=[
            _FakeChangedFile(
                "src.py",
                patch="@@ -1,1 +1,2 @@\n old\n+new\n",
            )
        ],
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path, dry_run=True),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        codex_client=cast(
            Any,
            _FakeCodexClient(
                _codex_response(
                    findings=[
                        _finding_payload(sample_file=sample_file, line=2),
                    ]
                )
            ),
        ),
    )

    workflow.process_review(7)

    assert _load_persisted(runner_temp, _key(head_sha="sha-DR")) is None


# ---------------------------------------------------------------------------
# 4. Prior summary id survives when the current run can't capture one
# ---------------------------------------------------------------------------


def test_prior_summary_id_survives_when_current_create_returns_no_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    monkeypatch.setenv("CODEX_REVIEW_CACHE_HIT", "false")

    seeded_key = _key(head_sha="sha-A")
    ReviewStateManager.from_runner_temp(str(runner_temp)).record_review_run(
        seeded_key,
        findings=(),
        summary_comment_id=4242,
    )

    sample_file = tmp_path / "src.py"
    sample_file.write_text("old\nnew\n", encoding="utf-8")

    pr = _FakePRWithIds(
        head_sha="sha-B",
        changed_files=[
            _FakeChangedFile(
                "src.py",
                patch="@@ -1,1 +1,2 @@\n old\n+new\n",
            )
        ],
        issue_comments=[
            _FakeIssueComment(
                body=(
                    "Codex Autonomous Review:\n"
                    "<!-- codex-review-meta "
                    '{"reviewed_head_sha":"sha-A"} -->\n'
                    "summary"
                ),
                comment_id=4242,
                login="bot",
            )
        ],
        # The fake returns None from create_comment so the workflow
        # cannot capture a new id.
        summary_returns_obj=False,
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        codex_client=cast(
            Any,
            _FakeCodexClient(
                _codex_response(
                    findings=[_finding_payload(sample_file=sample_file, line=2)]
                )
            ),
        ),
    )

    workflow.process_review(7)

    persisted = _load_persisted(runner_temp, _key(head_sha="sha-B"))
    assert persisted is not None
    # Prior summary id (4242) is preserved because we couldn't capture
    # a new one this run.
    assert persisted.summary_comment_id == 4242


# ---------------------------------------------------------------------------
# 5. Findings outside the repo root are skipped
# ---------------------------------------------------------------------------


def test_finding_outside_repo_root_is_skipped_for_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sample_file = repo_root / "src.py"
    sample_file.write_text("old\nnew\n", encoding="utf-8")

    # Outside-of-repo file the model hallucinated; persistence must
    # drop it rather than write a poisoned anchor.
    foreign_file = tmp_path / "foreign.py"
    foreign_file.write_text("foreign\n", encoding="utf-8")

    pr = _FakePRWithIds(
        head_sha="sha-X",
        changed_files=[
            _FakeChangedFile(
                "src.py",
                patch="@@ -1,1 +1,2 @@\n old\n+new\n",
            )
        ],
    )
    workflow = ReviewWorkflow(
        ReviewConfig(
            github_token="token",
            repository="owner/repo",
            pr_number=7,
            mode="review",
            dry_run=False,
            repo_root=repo_root,
            review_model="anthropic/claude-opus-4.7",
        ),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        codex_client=cast(
            Any,
            _FakeCodexClient(
                _codex_response(
                    findings=[
                        # Valid: inside repo root.
                        _finding_payload(
                            sample_file=sample_file,
                            title="Inside",
                            line=2,
                        ),
                        # Invalid: outside repo root.
                        _finding_payload(
                            sample_file=foreign_file,
                            title="Outside",
                            line=1,
                        ),
                    ]
                )
            ),
        ),
    )

    workflow.process_review(7)

    persisted = _load_persisted(runner_temp, _key(head_sha="sha-X"))
    assert persisted is not None
    titles = tuple(f.title for f in persisted.findings)
    assert titles == ("Inside",)


# ---------------------------------------------------------------------------
# 6. Persistence failures don't unwind the rest of the run
# ---------------------------------------------------------------------------


def test_persistence_failure_does_not_unwind_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    sample_file = tmp_path / "src.py"
    sample_file.write_text("old\nnew\n", encoding="utf-8")

    pr = _FakePRWithIds(
        head_sha="sha-FAIL",
        changed_files=[
            _FakeChangedFile(
                "src.py",
                patch="@@ -1,1 +1,2 @@\n old\n+new\n",
            )
        ],
        next_summary_id=4242,
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        codex_client=cast(
            Any,
            _FakeCodexClient(
                _codex_response(
                    findings=[_finding_payload(sample_file=sample_file, line=2)]
                )
            ),
        ),
    )

    # Force the persistence path to raise mid-write — the workflow must
    # swallow the exception and still return a usable result.
    def _explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "cli.workflows.review_workflow.ReviewStateManager.record_review_run",
        _explode,
    )

    result = workflow.process_review(7)
    # Run still produced a result and the summary still posted.
    assert result.review.findings[0].title == "Example finding"
    assert pr.as_issue().created_comments
    # No file written because record_review_run blew up.
    assert _load_persisted(runner_temp, _key(head_sha="sha-FAIL")) is None
