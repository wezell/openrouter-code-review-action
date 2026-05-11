from __future__ import annotations

import re
import subprocess  # nosec B404
import sys
from collections.abc import Callable
from dataclasses import dataclass

from ..clients.codex_client import CodexClient
from ..clients.git_ops import (
    GitWorktreeSnapshot,
    git_changed_paths_since_snapshot,
    git_commit_paths,
    git_current_head_sha,
    git_format_called_process_error,
    git_has_changes,
    git_head_is_ahead,
    git_is_ancestor,
    git_push,
    git_push_force_with_lease,
    git_push_head_to_branch,
    git_rebase_in_progress,
    git_remote_head_sha,
    git_setup_identity,
    git_status_pretty,
    git_worktree_snapshot,
)
from ..clients.github_client import GitHubClient, GitHubClientProtocol
from ..core.config import ReviewConfig, make_debug
from ..core.exceptions import GitHubAPIError
from ..core.github_types import PullRequestLikeProtocol
from ..core.models import CommentContext, ReviewCommentSnapshot
from .act_runner import ActAgentResult, ActModeRunner
from .edit_prompt import (
    build_comment_context_block,
    build_edit_prompt,
    format_unresolved_threads_from_list,
)


@dataclass(frozen=True)
class _EditPreflightState:
    head_branch: str | None
    before_head_sha: str | None
    remote_head_sha: str | None
    before_snapshot: GitWorktreeSnapshot


@dataclass(frozen=True)
class _EditPromptState:
    prompt: str
    comment_context_warning: str | None


@dataclass(frozen=True)
class _ReviewCommentContextState:
    comment_snapshot: ReviewCommentSnapshot | None = None
    parent_snapshot: ReviewCommentSnapshot | None = None
    warning: str | None = None


@dataclass(frozen=True)
class _EditPostAgentState:
    changed: bool
    agent_touched_paths: tuple[str, ...]
    ahead: bool


@dataclass(frozen=True)
class _ActCommitMetadata:
    """Subset of act-mode information used to author the commit message.

    The runner-produced :class:`ActAgentResult` carries the model slug and
    applied-file list we want to record in the trailer of the commit
    message so reviewers can see (a) which OpenRouter model authored the
    edit and (b) which files aider claimed to touch — independent of the
    git diff itself, which only reflects what actually changed on disk.
    """

    model: str
    applied_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class _EditEarlyExit:
    message: str
    exit_code: int


_DebugFn = Callable[[int, str], None]
_REBASE_IN_PROGRESS_MESSAGE = (
    "Git operation failed: repository is in an active rebase state. "
    "Resolve or abort the rebase before rerunning /codex."
)


class EditWorkflow:
    """Workflow for edit commands against PR branches."""

    def __init__(
        self,
        config: ReviewConfig,
        *,
        codex_client: CodexClient | None = None,
        github_client: GitHubClientProtocol | None = None,
        act_runner: ActModeRunner | None = None,
    ) -> None:
        self.config = config
        # Codex-backed agent path is preserved for non-act callers (e.g.
        # legacy review-comment workflows). Act mode now routes through
        # the aider wrapper via ``ActModeRunner`` so the OpenRouter model
        # config from action inputs flows directly into ``aider``.
        self._codex_client = codex_client
        self.github_client: GitHubClientProtocol = github_client or GitHubClient(config)
        self._act_runner = act_runner
        self._debug = make_debug(config)

    @property
    def codex_client(self) -> CodexClient:
        """Lazy CodexClient — only constructed if a non-act path needs it."""
        if self._codex_client is None:
            self._codex_client = CodexClient(self.config)
        return self._codex_client

    @property
    def act_runner(self) -> ActModeRunner:
        """Lazy ActModeRunner bound to ``self.config``.

        Constructed on first use so non-act callers (review mode + tests
        that monkeypatch the runner) do not pay the import cost.
        """
        if self._act_runner is None:
            self._act_runner = ActModeRunner(self.config)
        return self._act_runner

    def process_edit_command(
        self,
        command_text: str,
        pr_number: int,
        comment_ctx: CommentContext | None = None,
    ) -> int:
        """Run a coding-agent edit command against the PR's branch."""
        self._debug(1, f"Edit command on PR #{pr_number}: {command_text[:120]}")

        pr = self.github_client.get_pr(pr_number)
        preflight_state = self._collect_preflight_state_or_reply(pr, comment_ctx)
        if preflight_state is None:
            return 2

        prompt_state = _prepare_edit_prompt(
            config=self.config,
            github_client=self.github_client,
            debug=self._debug,
            command_text=command_text,
            pr=pr,
            comment_ctx=comment_ctx,
        )
        if isinstance(prompt_state, _EditEarlyExit):
            return self._report_and_reply(
                pr=pr,
                comment_ctx=comment_ctx,
                message=prompt_state.message,
                exit_code=prompt_state.exit_code,
                stderr=prompt_state.exit_code != 0,
            )
        if prompt_state.comment_context_warning:
            print(prompt_state.comment_context_warning, file=sys.stderr)
            self._debug(1, prompt_state.comment_context_warning)

        act_metadata: _ActCommitMetadata | None = None
        if self.config.mode == "act":
            agent_output, act_metadata = self._execute_aider_turn_or_reply_with_metadata(
                pr,
                comment_ctx,
                prompt_state.prompt,
            )
        else:
            agent_output = self._execute_agent_turn_or_reply(
                pr,
                comment_ctx,
                prompt_state.prompt,
            )
        if agent_output is None:
            return 1

        post_agent_state = self._collect_post_agent_state_or_reply(pr, comment_ctx, preflight_state)
        if post_agent_state is None:
            return 2
        if isinstance(post_agent_state, _EditEarlyExit):
            return self._report_and_reply(
                pr=pr,
                comment_ctx=comment_ctx,
                message=post_agent_state.message,
                exit_code=post_agent_state.exit_code,
                stderr=True,
            )

        if not post_agent_state.agent_touched_paths and not post_agent_state.ahead:
            print("No agent-scoped changes to commit.")
            return _return_after_reply(
                github_client=self.github_client,
                debug=self._debug,
                pr=pr,
                comment_ctx=comment_ctx,
                text=_format_edit_reply(
                    agent_output or "(no output)",
                    pushed=False,
                    dry_run=self.config.dry_run,
                    changed=False,
                    extra_summary=prompt_state.comment_context_warning,
                ),
                exit_code=0,
            )

        if self.config.dry_run:
            print("DRY_RUN: would commit and push changes.")
            git_status_pretty()
            return _return_after_reply(
                github_client=self.github_client,
                debug=self._debug,
                pr=pr,
                comment_ctx=comment_ctx,
                text=_format_edit_reply(
                    agent_output or "(no output)",
                    pushed=False,
                    dry_run=True,
                    changed=True,
                    extra_summary=prompt_state.comment_context_warning,
                ),
                exit_code=0,
            )

        try:
            _finalize_git_edit(
                debug=self._debug,
                command_text=command_text,
                preflight_state=preflight_state,
                post_agent_state=post_agent_state,
                act_metadata=act_metadata,
            )
        except subprocess.CalledProcessError as exc:
            details = git_format_called_process_error(exc)
            return self._report_and_reply(
                pr=pr,
                comment_ctx=comment_ctx,
                message=f"Git operation failed:\n{details}",
                exit_code=2,
                stderr=True,
            )

        return _return_after_reply(
            github_client=self.github_client,
            debug=self._debug,
            pr=pr,
            comment_ctx=comment_ctx,
            text=_format_edit_reply(
                agent_output or "(no output)",
                pushed=True,
                dry_run=False,
                changed=True,
                extra_summary=prompt_state.comment_context_warning,
            ),
            exit_code=0,
        )

    def _report_and_reply(
        self,
        *,
        pr: PullRequestLikeProtocol,
        comment_ctx: CommentContext | None,
        message: str,
        exit_code: int,
        stderr: bool,
    ) -> int:
        if stderr:
            print(message, file=sys.stderr)
        else:
            print(message)
        return _return_after_reply(
            github_client=self.github_client,
            debug=self._debug,
            pr=pr,
            comment_ctx=comment_ctx,
            text=message,
            exit_code=exit_code,
        )

    def _collect_preflight_state_or_reply(
        self,
        pr: PullRequestLikeProtocol,
        comment_ctx: CommentContext | None,
    ) -> _EditPreflightState | None:
        try:
            preflight_state = _collect_preflight_state(pr)
            rebase_in_progress = git_rebase_in_progress()
        except subprocess.CalledProcessError as exc:
            details = git_format_called_process_error(exc)
            self._report_and_reply(
                pr=pr,
                comment_ctx=comment_ctx,
                message=f"Git state probe failed:\n{details}",
                exit_code=2,
                stderr=True,
            )
            return None
        if rebase_in_progress:
            self._report_and_reply(
                pr=pr,
                comment_ctx=comment_ctx,
                message=_REBASE_IN_PROGRESS_MESSAGE,
                exit_code=2,
                stderr=True,
            )
            return None
        return preflight_state

    def _execute_agent_turn_or_reply(
        self,
        pr: PullRequestLikeProtocol,
        comment_ctx: CommentContext | None,
        prompt: str,
    ) -> str | None:
        if self.config.mode == "act":
            return self._execute_aider_turn_or_reply(pr, comment_ctx, prompt)
        # Non-act callers (review-comment driven Codex flow) stay on the
        # Codex path until that workflow is also migrated.
        try:
            return self.codex_client.execute_text(
                prompt,
                sandbox_mode="danger-full-access",
            )
        except Exception as exc:
            self._report_and_reply(
                pr=pr,
                comment_ctx=comment_ctx,
                message=f"Edit failed: {exc}",
                exit_code=1,
                stderr=True,
            )
            return None

    def _execute_aider_turn_or_reply(
        self,
        pr: PullRequestLikeProtocol,
        comment_ctx: CommentContext | None,
        prompt: str,
    ) -> str | None:
        """Backwards-compatible wrapper that drops the metadata."""
        output, _ = self._execute_aider_turn_or_reply_with_metadata(pr, comment_ctx, prompt)
        return output

    def _execute_aider_turn_or_reply_with_metadata(
        self,
        pr: PullRequestLikeProtocol,
        comment_ctx: CommentContext | None,
        prompt: str,
    ) -> tuple[str | None, _ActCommitMetadata | None]:
        """Run the aider wrapper for act mode and surface its result.

        The structured :class:`ActAgentResult` produced by
        :class:`ActModeRunner` carries stdout, stderr, exit code, and a
        deduplicated ``applied_files`` list parsed from aider's
        ``Applied edit to <path>`` lines. We log the rendered summary
        for the GitHub Actions log, capture the model + applied-files
        for the eventual commit-message trailer, and return the same
        summary string as ``agent_output`` for the PR reply.
        """
        try:
            act_result: ActAgentResult = self.act_runner.execute(prompt)
        except Exception as exc:
            self._report_and_reply(
                pr=pr,
                comment_ctx=comment_ctx,
                message=f"Edit failed: {exc}",
                exit_code=1,
                stderr=True,
            )
            return None, None

        summary = act_result.render_summary()
        # Echo the run summary to the runner log so operators scrolling
        # the GitHub Actions output see exit code + applied edits even
        # when the PR reply path fails downstream.
        if act_result.succeeded:
            print(summary)
        else:
            print(summary, file=sys.stderr)

        if not act_result.succeeded:
            self._report_and_reply(
                pr=pr,
                comment_ctx=comment_ctx,
                message=summary,
                exit_code=act_result.returncode if act_result.returncode != 0 else 1,
                stderr=True,
            )
            return None, None

        metadata = _ActCommitMetadata(
            model=self.config.act_model,
            applied_files=tuple(act_result.applied_files),
        )
        return summary, metadata

    def _collect_post_agent_state_or_reply(
        self,
        pr: PullRequestLikeProtocol,
        comment_ctx: CommentContext | None,
        preflight_state: _EditPreflightState,
    ) -> _EditPostAgentState | _EditEarlyExit | None:
        try:
            return _collect_post_agent_state(preflight_state)
        except subprocess.CalledProcessError as exc:
            details = git_format_called_process_error(exc)
            self._report_and_reply(
                pr=pr,
                comment_ctx=comment_ctx,
                message=f"Git state probe failed:\n{details}",
                exit_code=2,
                stderr=True,
            )
            return None


def _return_after_reply(
    *,
    github_client: GitHubClientProtocol,
    debug: _DebugFn,
    pr: PullRequestLikeProtocol,
    comment_ctx: CommentContext | None,
    text: str,
    exit_code: int,
) -> int:
    if comment_ctx is None:
        return exit_code

    try:
        if comment_ctx.event_name.lower() == "pull_request_review_comment":
            github_client.reply_to_review_comment(pr, comment_ctx.id, text)
        else:
            github_client.post_issue_comment(pr, text)
        reply_ok = True
    except GitHubAPIError as exc:
        warning = f"Failed to reply to comment {comment_ctx.id}: {exc}"
        print(warning, file=sys.stderr)
        debug(1, warning)
        reply_ok = False
    if not reply_ok and exit_code == 0:
        print(
            "GitHub reply delivery failed after a locally successful edit workflow result.",
            file=sys.stderr,
        )
    return exit_code


def _collect_preflight_state(pr: PullRequestLikeProtocol) -> _EditPreflightState:
    head_branch = pr.head.ref if pr.head else None
    return _EditPreflightState(
        head_branch=head_branch,
        before_head_sha=git_current_head_sha(),
        remote_head_sha=git_remote_head_sha(head_branch),
        before_snapshot=git_worktree_snapshot(),
    )


def _prepare_edit_prompt(
    *,
    config: ReviewConfig,
    github_client: GitHubClientProtocol,
    debug: _DebugFn,
    command_text: str,
    pr: PullRequestLikeProtocol,
    comment_ctx: CommentContext | None,
) -> _EditPromptState | _EditEarlyExit:
    unresolved_block = ""
    if _wants_fix_unresolved(command_text):
        try:
            unresolved_threads = github_client.get_unresolved_threads(pr)
        except GitHubAPIError as exc:
            warning = (
                "Failed to retrieve review threads; refusing to continue without "
                f"unresolved-thread context: {exc}"
            )
            return _EditEarlyExit(message=warning, exit_code=2)

        debug(1, f"Unresolved threads found: {len(unresolved_threads)}")
        if not unresolved_threads:
            return _EditEarlyExit(
                message="No unresolved review threads detected; nothing to address.",
                exit_code=0,
            )

        unresolved_block = format_unresolved_threads_from_list(unresolved_threads)

    review_comment_context = _load_review_comment_context(pr, comment_ctx)
    comment_context_result = build_comment_context_block(
        config,
        comment_ctx,
        review_comment_snapshot=review_comment_context.comment_snapshot,
        parent_review_comment_snapshot=review_comment_context.parent_snapshot,
        lookup_warning=review_comment_context.warning,
    )
    prompt = build_edit_prompt(
        config,
        command_text,
        pr,
        comment_context_result.block,
        unresolved_block,
    )
    debug(1, f"Edit prompt ready ({len(prompt)} chars)")
    debug(
        2,
        "Edit prompt context "
        f"unresolved_threads={'yes' if unresolved_block else 'no'} "
        f"comment_context={'yes' if comment_context_result.block else 'no'} "
        f"command_chars={len(command_text.strip())}",
    )
    return _EditPromptState(
        prompt=prompt,
        comment_context_warning=comment_context_result.warning,
    )


def _load_review_comment_context(
    pr: PullRequestLikeProtocol,
    comment_ctx: CommentContext | None,
) -> _ReviewCommentContextState:
    if comment_ctx is None or comment_ctx.event_name.lower() != "pull_request_review_comment":
        return _ReviewCommentContextState()

    comment_id = comment_ctx.id
    try:
        snapshot = ReviewCommentSnapshot.from_review_comment(pr.get_review_comment(comment_id))
    except Exception as exc:
        return _ReviewCommentContextState(
            warning=f"Comment context lookup failed for review comment {comment_id}: {exc}"
        )

    parent_snapshot: ReviewCommentSnapshot | None = None
    warning: str | None = None
    needs_parent_context = (not snapshot.path) or (
        snapshot.line is None and snapshot.original_line is None
    )
    if needs_parent_context and snapshot.in_reply_to_id is not None:
        parent_id = snapshot.in_reply_to_id
        try:
            parent_snapshot = ReviewCommentSnapshot.from_review_comment(
                pr.get_review_comment(parent_id)
            )
        except Exception as exc:
            warning = f"Failed to load parent review comment {parent_id}: {exc}"

    return _ReviewCommentContextState(
        comment_snapshot=snapshot,
        parent_snapshot=parent_snapshot,
        warning=warning,
    )


def _collect_post_agent_state(
    preflight_state: _EditPreflightState,
) -> _EditPostAgentState | _EditEarlyExit:
    if git_rebase_in_progress():
        return _EditEarlyExit(
            message=_REBASE_IN_PROGRESS_MESSAGE,
            exit_code=2,
        )

    changed = git_has_changes()
    after_snapshot = git_worktree_snapshot()
    return _EditPostAgentState(
        changed=changed,
        agent_touched_paths=tuple(
            git_changed_paths_since_snapshot(
                preflight_state.before_snapshot,
                after_snapshot,
            )
        ),
        ahead=git_head_is_ahead(preflight_state.head_branch),
    )


def _finalize_git_edit(
    *,
    debug: _DebugFn,
    command_text: str,
    preflight_state: _EditPreflightState,
    post_agent_state: _EditPostAgentState,
    act_metadata: _ActCommitMetadata | None = None,
) -> None:
    if post_agent_state.changed and post_agent_state.agent_touched_paths:
        git_setup_identity()
        message = _build_commit_message(
            command_text=command_text,
            agent_touched_paths=post_agent_state.agent_touched_paths,
            act_metadata=act_metadata,
        )
        git_commit_paths(message, post_agent_state.agent_touched_paths)

    after_head_sha = git_current_head_sha()
    history_rewritten = (
        preflight_state.before_head_sha is not None
        and after_head_sha is not None
        and preflight_state.before_head_sha != after_head_sha
        and not git_is_ancestor(preflight_state.before_head_sha, after_head_sha)
    )

    if preflight_state.head_branch:
        if history_rewritten:
            debug(
                1,
                f"Detected rewritten history for {preflight_state.head_branch}; "
                "using force-with-lease push.",
            )
            git_push_force_with_lease(
                preflight_state.head_branch,
                preflight_state.remote_head_sha,
            )
        else:
            git_push_head_to_branch(preflight_state.head_branch, debug)
    else:
        git_push()

    print("Pushed edits successfully.")


_COMMIT_SUBJECT_LIMIT = 72


def _build_commit_message(
    *,
    command_text: str,
    agent_touched_paths: tuple[str, ...],
    act_metadata: _ActCommitMetadata | None,
) -> str:
    """Render a descriptive commit message for an aider-applied edit.

    Subject line follows Conventional-Commit-ish form: ``act(aider): <summary>``
    when act-mode metadata is present, otherwise falls back to the
    legacy ``Codex edit:`` form for non-act callers. The body includes
    the originating command, OpenRouter model slug, and the actual list
    of touched paths so the commit is self-describing in ``git log``
    without needing to dig into the PR.
    """

    raw_summary = command_text.splitlines()[0] if command_text.splitlines() else command_text
    summary = (raw_summary.strip() or "apply edits")[:_COMMIT_SUBJECT_LIMIT]

    if act_metadata is None:
        # Non-act callers (legacy Codex flow) keep their existing subject
        # so commit history reads consistently for that path.
        return f"Codex edit: {summary}"

    subject = f"act(aider): {summary}"
    body_lines: list[str] = [
        "",
        f"Applied via aider + OpenRouter model {act_metadata.model or '(unset)'}.",
    ]
    body_lines.append("")
    body_lines.append(f"Command: {command_text.strip() or '(empty)'}")

    file_list = list(act_metadata.applied_files) or list(agent_touched_paths)
    if file_list:
        body_lines.append("")
        body_lines.append("Files:")
        for path in file_list:
            body_lines.append(f"- {path}")
    body = "\n".join(body_lines).rstrip()
    return f"{subject}\n{body}" if body else subject


def _wants_fix_unresolved(text: str) -> bool:
    """Detect intent to address review comments with a minimal heuristic."""
    if not text:
        return False

    normalized = " ".join(text.lower().split())
    if re.search(r"\b(do\s+not|don't|dont)\s+(address|fix|resolve)\b", normalized):
        return False

    has_verb = bool(re.search(r"\b(address|fix|resolve)\b", normalized))
    has_noun = bool(
        re.search(
            r"\b((review\s+)?comments?|((review\s+)?threads?)|feedback|reviews?)\b",
            normalized,
        )
    )
    return has_verb and has_noun


def _format_edit_reply(
    agent_output: str,
    *,
    pushed: bool,
    dry_run: bool,
    changed: bool,
    extra_summary: str | None = None,
) -> str:
    if dry_run:
        status = "dry-run (no push)"
    elif pushed:
        status = "pushed changes"
    elif not changed:
        status = "no changes"
    else:
        status = "not pushed"
    header = f"Codex edit result ({status}):"
    body = agent_output.strip()
    if len(body) > 3500:
        body = body[:3500] + "\n\n… (truncated)"
    if extra_summary:
        return f"{header}\n\n{body}\n\n{extra_summary}"
    return f"{header}\n\n{body}"
