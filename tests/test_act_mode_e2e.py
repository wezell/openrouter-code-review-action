"""End-to-end integration tests for the act-mode trigger pipeline.

Sub-AC 7.4 — exercise the *full* slice from a GitHub Actions comment
event through ``cli.main:main`` to a push of aider-applied edits, with
the network/process boundaries (GitHub API, ``aider`` subprocess, git
push) mocked at their dependency seams. The aim is to lock in the
contract that:

* The trust gate in :func:`cli.main._is_commenter_allowed` is the *only*
  thing standing between an untrusted comment author and an
  ``ActModeRunner`` invocation. Untrusted authors must produce zero
  aider calls and zero pushes.
* For trusted authors, the ``/codex`` command text reaches
  ``ActModeRunner.execute`` verbatim (post extraction), the configured
  OpenRouter act-model is honoured, and a commit + push to the PR's head
  branch are issued.
* Both ``issue_comment`` and ``pull_request_review_comment`` event
  shapes drive the same workflow.
* Aider failure is surfaced as a non-zero exit and does *not* push.

These tests deliberately invoke ``main.main()`` rather than poking
``EditWorkflow`` directly — that's what makes them integration tests
for the trigger flow rather than unit tests for the workflow.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from cli import main as main_module
from cli.clients.git_ops import GitWorktreeSnapshot
from cli.workflows import edit_workflow as edit_workflow_module
from cli.workflows.act_runner import ActAgentResult

# ---------------------------------------------------------------------------
# Shared fakes — wired in via monkeypatch at the dependency boundaries the
# real production code uses (GitHubClient, ActModeRunner, git_ops module
# attributes). Each test gets a fresh instance.
# ---------------------------------------------------------------------------


@dataclass
class _ActCall:
    prompt: str
    kwargs: dict[str, Any]


@dataclass
class _FakeActRunner:
    """Stand-in for ``ActModeRunner`` capturing every ``execute`` call."""

    config: Any
    result: ActAgentResult | None = None
    raise_exc: Exception | None = None
    calls: list[_ActCall] = field(default_factory=list)

    def execute(self, prompt: str, **kwargs: Any) -> ActAgentResult:
        self.calls.append(_ActCall(prompt=prompt, kwargs=dict(kwargs)))
        if self.raise_exc is not None:
            raise self.raise_exc
        assert self.result is not None, "test forgot to wire an ActAgentResult"
        return self.result


class _FakePR:
    """Minimal PR stand-in — head/base refs + url. EditWorkflow only
    pulls ``head.ref`` for the push target and ``url`` for logging."""

    url = "https://api.example/repos/owner/repo/pulls/17"

    class _Head:
        ref = "feature-branch"

    class _Base:
        ref = "main"

    head = _Head()
    base = _Base()

    def get_review_comment(self, comment_id: int) -> Any:  # pragma: no cover
        raise AssertionError("review-comment lookup should not be needed for these tests")


@dataclass
class _FakeGitHubClient:
    """Captures the reply target so we can assert which API path was used."""

    replies: list[tuple[str, int | None, str]] = field(default_factory=list)

    def get_pr(self, pr_number: int) -> _FakePR:  # noqa: ARG002
        return _FakePR()

    def get_unresolved_threads(self, pr: _FakePR) -> list[Any]:  # noqa: ARG002
        return []

    def reply_to_review_comment(
        self,
        pr: _FakePR,
        comment_id: int,
        text: str,  # noqa: ARG002
    ) -> None:
        self.replies.append(("review_comment", comment_id, text))

    def post_issue_comment(self, pr: _FakePR, text: str) -> None:  # noqa: ARG002
        self.replies.append(("issue_comment", None, text))


def _ok_act_result(applied: tuple[str, ...] = ("a.py",)) -> ActAgentResult:
    rendered_paths = "".join(f"Applied edit to {p}\n" for p in applied)
    return ActAgentResult(
        returncode=0,
        stdout=f"Loading model …\n{rendered_paths}Done.\n",
        stderr="",
        command=("aider", "--message", "x"),
        cwd=Path("."),
        applied_files=applied,
        error_message=None,
    )


def _failing_act_result() -> ActAgentResult:
    return ActAgentResult(
        returncode=2,
        stdout="partial output\n",
        stderr="openrouter: rate-limited",
        command=("aider", "--message", "x"),
        cwd=Path("."),
        applied_files=(),
        error_message="aider failed",
    )


# ---------------------------------------------------------------------------
# Test fixture: writes the GitHub Actions event payload, sets the env vars
# main.py reads, and patches every external boundary the workflow touches.
# Returns the sticky state (act runner, github client, push log) so tests
# can make assertions on what happened.
# ---------------------------------------------------------------------------


class _E2EHarness:
    """Captures every cross-boundary effect for assertions after ``main.main()``.

    ``act_runner`` is resolved lazily because :class:`EditWorkflow` does
    not construct an :class:`ActModeRunner` until the trust gate passes —
    rejection paths must be allowed to expose ``calls == []`` without
    raising ``KeyError``.
    """

    def __init__(
        self,
        runner_holder: dict[str, _FakeActRunner],
        github_client: _FakeGitHubClient,
        pushed_branches: list[str],
        committed: list[tuple[str, list[str]]],
    ) -> None:
        self._runner_holder = runner_holder
        self.github_client = github_client
        self.pushed_branches = pushed_branches
        self.committed = committed

    @property
    def act_runner(self) -> _FakeActRunner:
        return self._runner_holder["runner"]


def _install_e2e_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    event_name: str,
    event_payload: dict[str, Any],
    act_result: ActAgentResult | None = None,
    act_raise: Exception | None = None,
    has_changes: bool = True,
    agent_changed_paths: tuple[str, ...] = ("a.py",),
    extra_env: dict[str, str] | None = None,
) -> _E2EHarness:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event_payload), encoding="utf-8")

    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CODEX_MODE", "act")
    if extra_env:
        for k, v in extra_env.items():
            monkeypatch.setenv(k, v)

    # ActModeRunner replacement — config is the only ctor arg the runner
    # accepts in its real signature, mirroring main → EditWorkflow's
    # construction.
    runner_holder: dict[str, _FakeActRunner] = {}

    def _runner_factory(config: Any) -> _FakeActRunner:
        runner = _FakeActRunner(
            config=config,
            result=act_result,
            raise_exc=act_raise,
        )
        runner_holder["runner"] = runner
        return runner

    monkeypatch.setattr(edit_workflow_module, "ActModeRunner", _runner_factory)

    # GitHubClient replacement at the EditWorkflow boundary (lazy
    # construction inside EditWorkflow.__init__).
    fake_gh = _FakeGitHubClient()
    monkeypatch.setattr(
        edit_workflow_module,
        "GitHubClient",
        lambda config: fake_gh,  # noqa: ARG005
    )

    # Pre/post snapshots — unique snapshots so the diff records the
    # agent-touched paths.
    before = GitWorktreeSnapshot(changed_paths=frozenset(), path_states={})
    after_states = {p: (True, f"hash-{p}") for p in agent_changed_paths}
    after = GitWorktreeSnapshot(
        changed_paths=frozenset(agent_changed_paths),
        path_states=after_states,
    )
    snapshots = iter([before, after])
    head_shas = iter(["sha-before", "sha-after"])

    monkeypatch.setattr(edit_workflow_module, "git_worktree_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(edit_workflow_module, "git_current_head_sha", lambda: next(head_shas))
    monkeypatch.setattr(
        edit_workflow_module,
        "git_remote_head_sha",
        lambda branch: "remote-head",  # noqa: ARG005
    )
    monkeypatch.setattr(edit_workflow_module, "git_rebase_in_progress", lambda: False)
    monkeypatch.setattr(edit_workflow_module, "git_has_changes", lambda: has_changes)
    monkeypatch.setattr(
        edit_workflow_module,
        "git_head_is_ahead",
        lambda branch: False,  # noqa: ARG005
    )
    monkeypatch.setattr(
        edit_workflow_module,
        "git_is_ancestor",
        lambda older, newer: True,  # noqa: ARG005
    )
    monkeypatch.setattr(edit_workflow_module, "git_setup_identity", lambda: None)

    committed: list[tuple[str, list[str]]] = []
    pushed_branches: list[str] = []

    def _capture_commit(message: str, paths: Any) -> bool:
        committed.append((message, list(paths)))
        return True

    monkeypatch.setattr(edit_workflow_module, "git_commit_paths", _capture_commit)
    monkeypatch.setattr(
        edit_workflow_module,
        "git_push_head_to_branch",
        lambda branch, debug: pushed_branches.append(branch),  # noqa: ARG005
    )
    monkeypatch.setattr(edit_workflow_module, "git_push", lambda: pushed_branches.append("HEAD"))
    monkeypatch.setattr(
        edit_workflow_module,
        "git_push_force_with_lease",
        lambda branch, expected: pushed_branches.append(  # noqa: ARG005
            f"force-with-lease:{branch}"
        ),
    )

    monkeypatch.setattr(sys, "argv", ["openrouter-review"])

    # main.main() instantiates EditWorkflow, which constructs the runner
    # lazily. We seed runner_holder with a pending value so tests can
    # always inspect ``harness.act_runner`` even if no runner was made
    # (rejection paths). Use a sentinel runner that fails loudly on use.
    sentinel = _FakeActRunner(
        config=None,
        raise_exc=AssertionError(
            "ActModeRunner.execute called for a path that should have rejected first"
        ),
    )
    runner_holder.setdefault("runner", sentinel)

    return _E2EHarness(
        runner_holder=runner_holder,
        github_client=fake_gh,
        pushed_branches=pushed_branches,
        committed=committed,
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_e2e_issue_comment_trusted_author_drives_aider_and_pushes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """MEMBER on issue_comment → aider runs, commit + push happen, reply posted."""

    payload = {
        "issue": {
            "number": 17,
            "pull_request": {"url": "https://example.test/pr/17"},
        },
        "comment": {
            "id": 4242,
            "body": "/codex fix the lint",
            "user": {"login": "octocat"},
            "author_association": "MEMBER",
        },
    }
    harness = _install_e2e_harness(
        monkeypatch,
        tmp_path,
        event_name="issue_comment",
        event_payload=payload,
        act_result=_ok_act_result(applied=("a.py",)),
    )

    rc = main_module.main()

    assert rc == 0, f"main exited non-zero; stderr={capsys.readouterr().err!r}"
    # ActModeRunner was constructed and called exactly once with the
    # extracted command (no /codex prefix) embedded in the prompt.
    assert isinstance(harness.act_runner, _FakeActRunner)
    assert len(harness.act_runner.calls) == 1
    prompt = harness.act_runner.calls[0].prompt
    assert "fix the lint" in prompt
    # Push went to the PR head branch — never to main.
    assert harness.pushed_branches == ["feature-branch"]
    # Exactly one commit, scoped to the agent-touched paths.
    assert len(harness.committed) == 1
    commit_msg, commit_paths = harness.committed[0]
    assert commit_paths == ["a.py"]
    assert commit_msg.splitlines()[0].startswith("act(aider):")
    # Reply went via the issue-comment endpoint (not review-comment).
    assert harness.github_client.replies
    kind, _, _text = harness.github_client.replies[0]
    assert kind == "issue_comment"
    # Audit log records the allowed decision.
    err = capsys.readouterr().err
    audit_lines = [line for line in err.splitlines() if line.startswith("[audit:act-mode-gate] ")]
    assert len(audit_lines) == 1
    payload_audit = json.loads(audit_lines[0][len("[audit:act-mode-gate] ") :])
    assert payload_audit["outcome"] == "allowed"
    assert payload_audit["author_association"] == "MEMBER"


def test_e2e_pull_request_review_comment_trusted_author_drives_aider_and_pushes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """COLLABORATOR on pull_request_review_comment also drives the flow."""

    payload = {
        # pull_request_review_comment events nest the PR under
        # ``pull_request`` with the PR number directly available.
        "pull_request": {
            "number": 17,
            "html_url": "https://example.test/pr/17",
        },
        "comment": {
            "id": 999,
            "body": "/codex: tighten error handling",
            "user": {"login": "alice"},
            "author_association": "COLLABORATOR",
            "pull_request_review_id": 100,
        },
    }
    harness = _install_e2e_harness(
        monkeypatch,
        tmp_path,
        event_name="pull_request_review_comment",
        event_payload=payload,
        act_result=_ok_act_result(applied=("src/x.py", "src/y.py")),
        agent_changed_paths=("src/x.py", "src/y.py"),
    )

    rc = main_module.main()

    assert rc == 0
    assert len(harness.act_runner.calls) == 1
    assert "tighten error handling" in harness.act_runner.calls[0].prompt
    assert harness.pushed_branches == ["feature-branch"]
    assert len(harness.committed) == 1
    _, commit_paths = harness.committed[0]
    assert commit_paths == ["src/x.py", "src/y.py"]
    # Reply must use the review-comment thread endpoint to keep the
    # response stitched to the original review thread.
    assert harness.github_client.replies
    kind, comment_id, _text = harness.github_client.replies[0]
    assert kind == "review_comment"
    assert comment_id == 999


# ---------------------------------------------------------------------------
# Trust-gate rejection paths — must never reach aider or git push
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "association",
    ["CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "NONE", "MANNEQUIN"],
)
def test_e2e_untrusted_author_associations_are_rejected_before_aider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    association: str,
) -> None:
    """Untrusted associations short-circuit before any aider invocation."""

    payload = {
        "issue": {
            "number": 17,
            "pull_request": {"url": "https://example.test/pr/17"},
        },
        "comment": {
            "id": 7,
            "body": "/codex rm -rf /",
            "user": {"login": "stranger"},
            "author_association": association,
        },
    }
    harness = _install_e2e_harness(
        monkeypatch,
        tmp_path,
        event_name="issue_comment",
        event_payload=payload,
        act_result=_ok_act_result(),
    )

    rc = main_module.main()

    # Rejection is a *graceful* no-op (rc=0), not an error — the Action
    # job should not be marked failed when an unauthorized user types
    # /codex.
    assert rc == 0
    # Critically: zero aider invocations and zero pushes.
    assert harness.act_runner.calls == []
    assert harness.pushed_branches == []
    assert harness.committed == []
    assert harness.github_client.replies == []
    # Operator-visible message + structured audit-log entry.
    captured = capsys.readouterr()
    assert "unauthorized commenter association" in captured.out
    audit_lines = [
        line for line in captured.err.splitlines() if line.startswith("[audit:act-mode-gate] ")
    ]
    assert len(audit_lines) == 1
    audit = json.loads(audit_lines[0][len("[audit:act-mode-gate] ") :])
    assert audit["outcome"] == "rejected"
    assert audit["author_association"] == association
    assert audit["author"] == "stranger"
    assert audit["pr_number"] == 17


def test_e2e_missing_author_association_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing author_association (forged event payload) → rejection."""

    payload = {
        "issue": {
            "number": 17,
            "pull_request": {"url": "https://example.test/pr/17"},
        },
        "comment": {
            "id": 8,
            "body": "/codex fix things",
            "user": {"login": "ghost"},
            # no author_association
        },
    }
    harness = _install_e2e_harness(
        monkeypatch,
        tmp_path,
        event_name="issue_comment",
        event_payload=payload,
        act_result=_ok_act_result(),
    )

    rc = main_module.main()

    assert rc == 0
    assert harness.act_runner.calls == []
    assert harness.pushed_branches == []
    audit_lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("[audit:act-mode-gate] ")
    ]
    assert len(audit_lines) == 1
    audit = json.loads(audit_lines[0][len("[audit:act-mode-gate] ") :])
    assert audit["outcome"] == "rejected"
    assert audit["author_association"] is None


def test_e2e_review_mode_ignores_codex_command_even_for_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A trusted OWNER /codex comment in review mode must not run aider."""

    payload = {
        "issue": {
            "number": 17,
            "pull_request": {"url": "https://example.test/pr/17"},
        },
        "comment": {
            "id": 11,
            "body": "/codex go wild",
            "user": {"login": "boss"},
            "author_association": "OWNER",
        },
    }
    harness = _install_e2e_harness(
        monkeypatch,
        tmp_path,
        event_name="issue_comment",
        event_payload=payload,
        act_result=_ok_act_result(),
        # Override CODEX_MODE → review so the dispatcher refuses.
        extra_env={"CODEX_MODE": "review"},
    )

    rc = main_module.main()

    assert rc == 0
    assert harness.act_runner.calls == []
    assert harness.pushed_branches == []
    out = capsys.readouterr().out
    assert "set CODEX_MODE=act" in out


def test_e2e_bare_codex_comment_does_not_trigger_aider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``/codex`` with no payload extracts to None → no aider call."""

    payload = {
        "issue": {
            "number": 17,
            "pull_request": {"url": "https://example.test/pr/17"},
        },
        "comment": {
            "id": 12,
            "body": "/codex",
            "user": {"login": "octocat"},
            "author_association": "MEMBER",
        },
    }
    harness = _install_e2e_harness(
        monkeypatch,
        tmp_path,
        event_name="issue_comment",
        event_payload=payload,
        act_result=_ok_act_result(),
    )

    rc = main_module.main()

    assert rc == 0
    assert harness.act_runner.calls == []
    assert harness.pushed_branches == []


def test_e2e_non_codex_comment_does_not_trigger_aider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Plain conversation must never reach the aider boundary."""

    payload = {
        "issue": {
            "number": 17,
            "pull_request": {"url": "https://example.test/pr/17"},
        },
        "comment": {
            "id": 13,
            "body": "looks great, ship it!",
            "user": {"login": "octocat"},
            "author_association": "MEMBER",
        },
    }
    harness = _install_e2e_harness(
        monkeypatch,
        tmp_path,
        event_name="issue_comment",
        event_payload=payload,
        act_result=_ok_act_result(),
    )

    rc = main_module.main()

    assert rc == 0
    assert harness.act_runner.calls == []
    assert harness.pushed_branches == []


# ---------------------------------------------------------------------------
# Aider failure path — trusted author, but aider exits non-zero
# ---------------------------------------------------------------------------


def test_e2e_aider_failure_does_not_push_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When aider reports failure, the workflow must abort *before* push."""

    payload = {
        "issue": {
            "number": 17,
            "pull_request": {"url": "https://example.test/pr/17"},
        },
        "comment": {
            "id": 21,
            "body": "/codex break things",
            "user": {"login": "octocat"},
            "author_association": "MEMBER",
        },
    }
    harness = _install_e2e_harness(
        monkeypatch,
        tmp_path,
        event_name="issue_comment",
        event_payload=payload,
        act_result=_failing_act_result(),
    )

    rc = main_module.main()

    # Trust gate passed (aider was invoked) …
    assert len(harness.act_runner.calls) == 1
    # … but the failure must not be silently committed/pushed.
    assert harness.pushed_branches == []
    assert harness.committed == []
    assert rc != 0
    # And the PR comment carries the failure summary so a human knows
    # what happened without cracking open the runner logs.
    assert harness.github_client.replies
    _, _, body = harness.github_client.replies[0]
    assert "aider failed" in body or "rate-limited" in body
    # Stderr contains the rendered summary too.
    err = capsys.readouterr().err
    assert "aider" in err.lower()
