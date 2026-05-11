"""Sub-AC 4.3 — review pipeline restricted to SHA-delta files/hunks.

These tests exercise the workflow-level integration of
:class:`cli.core.sha_delta.ShaDeltaResolver`. The decision layer itself is
covered by ``tests/test_sha_delta.py``; here we verify that the
review workflow:

* Extracts post-image paths from a unified diff.
* Computes SHA-delta paths from either the inline diff or the
  ``git diff --name-only`` fall-back, gracefully degrading when the
  probe raises.
* Renders an ``<sha_delta_scope>`` prompt block only when the
  resolver's decision is ``incremental`` and emits the full file set
  in every other case.
* Filters the PR's changed-file list down to the delta when prior
  review state is found and the SHA range is reachable.
* Drops back to the full PR file set on first run, on force-push
  ancestry break, when the delta intersects nothing in the PR file
  list, and when the resolver cannot be constructed.
* Wires the scoped file list into ``compose_prompt`` so the model
  sees only the changed files inside the delta range.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from cli.core.config import ReviewConfig
from cli.core.review_state import PriorReviewKey, PriorReviewState
from cli.core.review_state_store import ReviewStateStore
from cli.core.sha_delta import (
    SHA_DELTA_DECISION_FRESH_REVIEW,
    SHA_DELTA_DECISION_INCREMENTAL,
    ShaDeltaResult,
)
from cli.review.posting import ReviewPostingOutcome
from cli.review.resume_state import render_review_summary_metadata
from cli.workflows.review_workflow import (
    SUMMARY_MARKER,
    ReviewWorkflow,
    _extract_paths_from_unified_diff,
    _ShaDeltaScope,
)


# ---------------------------------------------------------------------------
# Shared fakes (smaller, focused subset of test_review_workflow.py's stubs)
# ---------------------------------------------------------------------------


@dataclass
class _FakeUser:
    login: str = "octocat"


@dataclass
class _FakeHead:
    sha: str = "head-sha"
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


@dataclass
class _FakeIssue:
    created_comments: list[str] = field(default_factory=list)

    def create_comment(self, text: str) -> None:
        self.created_comments.append(text)


class _FakePR:
    def __init__(
        self,
        *,
        changed_files: Sequence[_FakeChangedFile],
        issue_comments: Sequence[Any] = (),
        review_comments: Sequence[Any] = (),
    ) -> None:
        self.number = 7
        self.title = "Improve review flow"
        self.body = "PR body"
        self.html_url = "https://example.test/pr/7"
        self.state = "open"
        self.url = "https://api.example.test/repos/owner/repo/pulls/7"
        self.user = _FakeUser()
        self.head: _FakeHead | None = _FakeHead()
        self.base = _FakeBase()
        self._changed_files = list(changed_files)
        self._issue_comments = list(issue_comments)
        self._review_comments = list(review_comments)
        self._issue = _FakeIssue()

    def get_files(self) -> list[_FakeChangedFile]:
        return list(self._changed_files)

    def get_issue_comments(self) -> list[Any]:
        return list(self._issue_comments)

    def get_review_comments(self) -> list[Any]:
        return list(self._review_comments)

    def as_issue(self) -> _FakeIssue:
        return self._issue


class _FakeIssueComment:
    def __init__(self, body: str, *, login: str = "reviewer", comment_id: int = 1) -> None:
        self.body = body
        self.id = comment_id
        self.user = _FakeUser(login)
        self.created_at = "now"

    def delete(self) -> None:  # pragma: no cover — not exercised in this file
        return None


class _FakeGitHubClient:
    def __init__(self, pr: _FakePR) -> None:
        self.pr = pr
        self.calls: list[int] = []

    def get_pr(self, pr_number: int) -> _FakePR:
        self.calls.append(pr_number)
        return self.pr

    def get_review_threads(self, pr: _FakePR) -> list[Any]:
        return []

    def post_inline_comment(self, pr: _FakePR, payload: Any, *, head_sha: str) -> None:
        return None

    def post_pull_request_review(
        self,
        pr: _FakePR,
        payloads: Any,
        *,
        head_sha: str,
    ) -> None:
        return None


class _FakeCodexClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def execute_structured(
        self,
        prompt: str,
        *,
        output_schema: dict[str, object],
        schema_prompt: str,
        sandbox_mode: str,
        resume_thread_id: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "output_schema": output_schema,
                "schema_prompt": schema_prompt,
                "sandbox_mode": sandbox_mode,
                "resume_thread_id": resume_thread_id,
            }
        )
        return self.response


def _make_config(tmp_path: Path) -> ReviewConfig:
    return ReviewConfig(
        github_token="token",
        repository="owner/repo",
        pr_number=7,
        mode="review",
        repo_root=tmp_path,
    )


def _seed_prior_state(
    runner_temp: Path,
    *,
    config: ReviewConfig,
    reviewed_head_sha: str,
) -> PriorReviewKey:
    """Drop an empty :class:`PriorReviewState` envelope on disk.

    Mirrors the on-disk layout the real workflow would have written
    after a previous review run, so :class:`ShaDeltaResolver` finds it
    via ``$RUNNER_TEMP/openrouter-review-state``.
    """
    store = ReviewStateStore.from_runner_temp(str(runner_temp))
    key = PriorReviewKey(
        repository=config.repository,
        pr_number=config.pr_number or 0,
        review_model=(config.review_model or "anthropic/claude-opus-4.7").strip(),
        reviewed_head_sha=reviewed_head_sha,
    )
    store.save(PriorReviewState.empty(key, generated_at="2026-05-07T00:00:00Z"))
    return key


# ---------------------------------------------------------------------------
# _extract_paths_from_unified_diff
# ---------------------------------------------------------------------------


def test_extract_paths_from_unified_diff_returns_post_image_paths() -> None:
    diff = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/src/bar.py b/src/baz.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    paths = _extract_paths_from_unified_diff(diff)

    # Post-image (b/...) is what survives — bar.py was renamed to baz.py.
    assert paths == frozenset({"src/foo.py", "src/baz.py"})


def test_extract_paths_from_unified_diff_handles_quoted_post_image() -> None:
    """git quotes paths with non-printables on the b-side; we must unwrap.

    The pre-image (``a/...``) is permitted to be unquoted with
    non-space characters; the post-image (``b/...``) is the side that
    drives anchoring, so the regex specifically tolerates quoted
    post-image paths and we must decode the C-style escapes git emits.
    """
    diff = (
        'diff --git a/src/foo.py b/"src/renamed\\t.py"\n'
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    paths = _extract_paths_from_unified_diff(diff)
    assert paths == frozenset({"src/renamed\t.py"})


def test_extract_paths_from_unified_diff_empty_diff() -> None:
    assert _extract_paths_from_unified_diff("") == frozenset()


# ---------------------------------------------------------------------------
# _build_sha_delta_block
# ---------------------------------------------------------------------------


def _make_scope(
    *,
    decision: str,
    previous_sha: str | None,
    incremental_diff: str | None,
    commit_shas: tuple[str, ...] = (),
    scoped_paths: tuple[str, ...] = (),
    scoped_files: Sequence[_FakeChangedFile] = (),
) -> _ShaDeltaScope:
    result = ShaDeltaResult(
        decision=decision,
        current_head_sha="head-sha",
        prior_state=PriorReviewState.empty(
            PriorReviewKey(
                repository="owner/repo",
                pr_number=7,
                review_model="anthropic/claude-opus-4.7",
                reviewed_head_sha="head-sha",
            )
        ),
        previous_reviewed_sha=previous_sha,
        incremental_diff=incremental_diff,
        commit_shas=commit_shas,
        reason="test",
        fallback_key=None,
    )
    return _ShaDeltaScope(
        result=result,
        scoped_changed_files=list(scoped_files),
        scoped_paths=scoped_paths,
    )


def test_build_sha_delta_block_returns_empty_for_non_incremental(
    tmp_path: Path,
) -> None:
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    fresh_scope = _make_scope(
        decision=SHA_DELTA_DECISION_FRESH_REVIEW,
        previous_sha=None,
        incremental_diff=None,
    )
    assert workflow._build_sha_delta_block(None) == ""
    assert workflow._build_sha_delta_block(fresh_scope) == ""


def test_build_sha_delta_block_renders_inline_diff(tmp_path: Path) -> None:
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    inline_diff = "diff --git a/src/foo.py b/src/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    scope = _make_scope(
        decision=SHA_DELTA_DECISION_INCREMENTAL,
        previous_sha="prev-sha",
        incremental_diff=inline_diff,
        scoped_paths=("src/foo.py",),
    )
    block = workflow._build_sha_delta_block(scope)
    assert "<sha_delta_scope>" in block
    assert "<previous_reviewed_head_sha>prev-sha</previous_reviewed_head_sha>" in block
    assert "<current_head_sha>head-sha</current_head_sha>" in block
    assert "<delta_files>\n- src/foo.py\n</delta_files>" in block
    assert "<incremental_diff>" in block
    assert inline_diff in block
    assert "<incremental_commits>" not in block


def test_build_sha_delta_block_falls_back_to_commit_list_when_diff_capped(
    tmp_path: Path,
) -> None:
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    scope = _make_scope(
        decision=SHA_DELTA_DECISION_INCREMENTAL,
        previous_sha="prev-sha",
        incremental_diff=None,
        commit_shas=("commit-1", "commit-2"),
        scoped_paths=("src/foo.py",),
    )
    block = workflow._build_sha_delta_block(scope)
    assert "<incremental_diff>" not in block
    assert "<incremental_commits>" in block
    assert "commit-1" in block
    assert "commit-2" in block
    assert "git diff prev-sha..head-sha" in block


# ---------------------------------------------------------------------------
# _compute_sha_delta_paths
# ---------------------------------------------------------------------------


def test_compute_sha_delta_paths_uses_inline_diff(tmp_path: Path) -> None:
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    inline = (
        "diff --git a/src/foo.py b/src/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/src/bar.py b/src/bar.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    scope_result = ShaDeltaResult(
        decision=SHA_DELTA_DECISION_INCREMENTAL,
        current_head_sha="head-sha",
        prior_state=PriorReviewState.empty(
            PriorReviewKey(
                repository="owner/repo",
                pr_number=7,
                review_model="anthropic/claude-opus-4.7",
                reviewed_head_sha="head-sha",
            )
        ),
        previous_reviewed_sha="prev-sha",
        incremental_diff=inline,
        commit_shas=("commit-1",),
        reason="test",
    )
    paths = workflow._compute_sha_delta_paths(scope_result)
    assert paths == frozenset({"src/foo.py", "src/bar.py"})


def test_compute_sha_delta_paths_falls_back_to_git_name_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    captured: list[str] = []

    def _fake_name_only(revision_range: str) -> list[str]:
        captured.append(revision_range)
        return ["src/foo.py", "src/bar.py"]

    monkeypatch.setattr(
        "cli.workflows.review_workflow.git_diff_name_only",
        _fake_name_only,
    )
    scope_result = ShaDeltaResult(
        decision=SHA_DELTA_DECISION_INCREMENTAL,
        current_head_sha="head-sha",
        prior_state=PriorReviewState.empty(
            PriorReviewKey(
                repository="owner/repo",
                pr_number=7,
                review_model="anthropic/claude-opus-4.7",
                reviewed_head_sha="head-sha",
            )
        ),
        previous_reviewed_sha="prev-sha",
        incremental_diff=None,
        commit_shas=("commit-1",),
        reason="test",
    )
    paths = workflow._compute_sha_delta_paths(scope_result)
    assert captured == ["prev-sha..head-sha"]
    assert paths == frozenset({"src/foo.py", "src/bar.py"})


def test_compute_sha_delta_paths_returns_empty_on_git_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )

    def _boom(revision_range: str) -> list[str]:
        raise subprocess.CalledProcessError(returncode=128, cmd=["git", "diff"])

    monkeypatch.setattr(
        "cli.workflows.review_workflow.git_diff_name_only",
        _boom,
    )
    scope_result = ShaDeltaResult(
        decision=SHA_DELTA_DECISION_INCREMENTAL,
        current_head_sha="head-sha",
        prior_state=PriorReviewState.empty(
            PriorReviewKey(
                repository="owner/repo",
                pr_number=7,
                review_model="anthropic/claude-opus-4.7",
                reviewed_head_sha="head-sha",
            )
        ),
        previous_reviewed_sha="prev-sha",
        incremental_diff=None,
        commit_shas=("commit-1",),
        reason="test",
    )
    assert workflow._compute_sha_delta_paths(scope_result) == frozenset()


# ---------------------------------------------------------------------------
# _resolve_sha_delta_scope
# ---------------------------------------------------------------------------


def test_resolve_sha_delta_scope_returns_none_when_pr_number_missing(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    config.pr_number = None
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    assert (
        workflow._resolve_sha_delta_scope(
            [_FakeChangedFile("src/foo.py")],
            head_sha="head-sha",
            previous_head_sha=None,
        )
        is None
    )


def test_resolve_sha_delta_scope_fresh_review_keeps_full_file_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    config = _make_config(tmp_path)
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    files = [_FakeChangedFile("src/foo.py"), _FakeChangedFile("src/bar.py")]

    scope = workflow._resolve_sha_delta_scope(
        cast(Any, files),
        head_sha="head-sha",
        previous_head_sha=None,
    )
    assert scope is not None
    assert scope.is_fresh_review
    # No prior state on disk → workflow must keep the full PR file list
    # so first-time reviews see everything.
    assert [f.filename for f in scope.scoped_changed_files] == [
        "src/foo.py",
        "src/bar.py",
    ]


def test_resolve_sha_delta_scope_filters_to_delta_paths_when_incremental(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    config = _make_config(tmp_path)
    _seed_prior_state(runner_temp, config=config, reviewed_head_sha="prev-sha")

    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.is_ancestor",
        lambda self, older, newer: True,
    )
    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.diff_text",
        lambda self, revision_range: (
            "diff --git a/src/bar.py b/src/bar.py\n@@ -1 +1 @@\n-old\n+new\n"
        ),
    )
    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.commit_shas",
        lambda self, revision_range: ["commit-1"],
    )

    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    files = [
        _FakeChangedFile("src/foo.py"),
        _FakeChangedFile("src/bar.py"),
        _FakeChangedFile("src/baz.py"),
    ]
    scope = workflow._resolve_sha_delta_scope(
        cast(Any, files),
        head_sha="head-sha",
        previous_head_sha="prev-sha",
    )
    assert scope is not None
    assert scope.is_incremental
    # Only the file mentioned in the inline diff survives the scoping
    # filter — unchanged regions / files outside the delta are skipped.
    assert [f.filename for f in scope.scoped_changed_files] == ["src/bar.py"]
    assert scope.scoped_paths == ("src/bar.py",)


def test_resolve_sha_delta_scope_force_push_drops_back_to_full_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    config = _make_config(tmp_path)
    _seed_prior_state(runner_temp, config=config, reviewed_head_sha="prev-sha")

    # Force-push: prior SHA is not an ancestor of HEAD → resolver returns
    # fresh_review per the seed contract.
    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.is_ancestor",
        lambda self, older, newer: False,
    )

    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    files = [_FakeChangedFile("src/foo.py"), _FakeChangedFile("src/bar.py")]
    scope = workflow._resolve_sha_delta_scope(
        cast(Any, files),
        head_sha="head-sha",
        previous_head_sha="prev-sha",
    )
    assert scope is not None
    assert scope.is_fresh_review
    assert [f.filename for f in scope.scoped_changed_files] == [
        "src/foo.py",
        "src/bar.py",
    ]


def test_resolve_sha_delta_scope_keeps_full_set_when_intersection_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Delta paths exist but reference a file outside the PR file list.

    Common case: a base-branch merge or rebase that touched files not
    in the PR. We must not silently hide the rest of the PR from the
    model — fall back to the full file set.
    """
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    config = _make_config(tmp_path)
    _seed_prior_state(runner_temp, config=config, reviewed_head_sha="prev-sha")

    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.is_ancestor",
        lambda self, older, newer: True,
    )
    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.diff_text",
        lambda self, revision_range: (
            "diff --git a/extra/unrelated.py b/extra/unrelated.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
    )
    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.commit_shas",
        lambda self, revision_range: ["commit-1"],
    )

    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, _FakeGitHubClient(_FakePR(changed_files=[]))),
        codex_client=cast(Any, _FakeCodexClient("{}")),
    )
    files = [_FakeChangedFile("src/foo.py"), _FakeChangedFile("src/bar.py")]
    scope = workflow._resolve_sha_delta_scope(
        cast(Any, files),
        head_sha="head-sha",
        previous_head_sha="prev-sha",
    )
    assert scope is not None
    assert scope.is_incremental
    assert [f.filename for f in scope.scoped_changed_files] == [
        "src/foo.py",
        "src/bar.py",
    ]


# ---------------------------------------------------------------------------
# process_review wires scoped file list into the prompt
# ---------------------------------------------------------------------------


def test_process_review_passes_scoped_files_to_compose_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("CODEX_REVIEW_PREVIOUS_HEAD_SHA", raising=False)
    # CACHE_HIT defaults to "true" when unset; explicitly fail the
    # legacy resume-cache check so the workflow takes the new SHA-delta
    # scope path instead of the CODEX_HOME-driven resume block.
    monkeypatch.setenv("CODEX_REVIEW_CACHE_HIT", "false")

    config = _make_config(tmp_path)
    _seed_prior_state(runner_temp, config=config, reviewed_head_sha="prev-sha")

    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.is_ancestor",
        lambda self, older, newer: True,
    )
    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.diff_text",
        lambda self, revision_range: (
            "diff --git a/src/bar.py b/src/bar.py\n@@ -1 +1 @@\n-old\n+new\n"
        ),
    )
    monkeypatch.setattr(
        "cli.core.sha_delta.DefaultGitDeltaProbe.commit_shas",
        lambda self, revision_range: ["commit-1"],
    )

    pr = _FakePR(
        changed_files=[
            _FakeChangedFile("src/foo.py"),
            _FakeChangedFile("src/bar.py"),
            _FakeChangedFile("src/baz.py"),
        ],
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\n{render_review_summary_metadata('prev-sha')}\nold",
                comment_id=10,
            )
        ],
    )
    github_client = _FakeGitHubClient(pr)
    codex_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": None,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        codex_client=cast(Any, codex_client),
    )

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: ReviewPostingOutcome.empty(0),
    )

    captured: dict[str, list[str]] = {"filenames": []}

    def _fake_compose(config: ReviewConfig, changed_files: Any, pr: Any, artifacts: Any) -> str:
        captured["filenames"] = [f.filename for f in changed_files]
        return "FAKE PROMPT"

    monkeypatch.setattr("cli.workflows.review_workflow.compose_prompt", _fake_compose)

    workflow.process_review(7)

    # Sub-AC 4.3 — only files inside the SHA-delta range reach the
    # prompt composer; src/foo.py and src/baz.py are skipped.
    assert captured["filenames"] == ["src/bar.py"]
    # The model run should have observed the <sha_delta_scope> block.
    assert "<sha_delta_scope>" in codex_client.calls[0]["prompt"]
    assert "<previous_reviewed_head_sha>prev-sha</previous_reviewed_head_sha>" in (
        codex_client.calls[0]["prompt"]
    )
