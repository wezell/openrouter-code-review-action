"""SHA-delta detection between the last reviewed SHA and the current HEAD.

Sub-AC 4.2 — this module is the **incremental-review decision layer**.
Given a freshly resolved :class:`PriorReviewKey` (current HEAD SHA) and an
optional hint at the previously reviewed SHA, it answers the operational
question the workflow has at the top of every run:

    *Can we re-use prior review state and diff only the SHA-delta, or do
    we need to throw it away and do a fresh full review?*

The seed contract pins three behaviours that this layer enforces:

* **Force-push that breaks reviewed-SHA ancestry falls back to fresh full
  review.** If the prior reviewed SHA is no longer reachable as an
  ancestor of the current HEAD, no incremental delta is computable and
  the workflow MUST start over.
* **Single source of truth for prior state.** Lookups go through
  :class:`~cli.core.review_state_manager.ReviewStateManager` (Sub-AC
  4.1.3) so the schema/store layers (Sub-AC 4.1.1 + 4.1.2) own the
  on-disk contract; this module is *purely* the decision wrapper.
* **No mid-run swap.** The model identity is part of the
  :class:`PriorReviewKey`, so a model bump deliberately misses the
  cache and produces a fresh review under the new key — no fallback
  chain.

Concept map
-----------

* :class:`ShaDeltaDecision` — string-typed enum of the three decision
  branches: ``same_sha``, ``incremental``, ``fresh_review``. Open string
  (not :class:`enum.Enum`) so the field round-trips through JSON
  telemetry without a custom encoder, mirroring the pattern in
  :mod:`cli.core.review_state_manager`.
* :class:`ShaDeltaResult` — frozen result envelope. Carries the chosen
  decision, the resolved prior state (always non-``None`` because the
  manager initialises an empty envelope on miss), the SHA pair, the
  computed diff text + commit list (only populated for the
  ``incremental`` branch), and a human-readable ``reason`` string for
  operator logs.
* :class:`GitDeltaProbe` — :class:`Protocol` for the three git
  operations this module needs (``is_ancestor``, ``diff_text``,
  ``commit_shas``). Default implementation simply delegates to
  :mod:`cli.clients.git_ops`; tests can pass a stub that records calls
  without spinning up a real repo.
* :class:`ShaDeltaResolver` — the high-level decision class. Holds the
  state manager and git probe; ``resolve(...)`` is the single entry
  point the workflow calls.

Why a fresh module instead of growing :class:`ReviewStateManager`?
The manager owns *state lookup and persistence*. This module owns
*decision making* — does the prior state, given the current SHA pair,
unlock incremental review? Mixing the two would force the manager to
import :mod:`cli.clients.git_ops` (and therefore subprocess) into the
state-management layer, which we deliberately kept I/O-pure.

Why a :class:`Protocol` for the git probe instead of monkey-patching the
git_ops module in tests? Because the unit-test surface for this layer is
"given these probe answers, did we make the right decision?" — keeping
the probe injectable means tests don't need a tmp git repo, don't need
``monkeypatch`` for module-global functions, and read top-down without
indirection. The default implementation is a thin pass-through so
production code is unchanged.

Why ``incremental_diff`` is capped by an in-module constant rather than
plumbed in from the caller? Because the cap is a function of "what does
the LLM context window comfortably accept", not a workflow-level policy
knob. Setting it here keeps the workflow code simple (decision in →
prompt block out) and lets a future model bump change the cap in one
place.
"""

from __future__ import annotations

import subprocess  # nosec B404 — used for typing CalledProcessError
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from .review_state import PriorReviewKey, PriorReviewState
from .review_state_manager import (
    LOOKUP_SOURCE_CURRENT_SHA,
    LOOKUP_SOURCE_PREVIOUS_SHA,
    ReviewStateLookupResult,
    ReviewStateManager,
)

# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------

#: A re-run on exactly the same head SHA the prior run reviewed. The
#: workflow can treat this either as a no-op or as a refresh of the
#: published comment set; the diff between current and previous is empty
#: by definition.
SHA_DELTA_DECISION_SAME_SHA: Final[str] = "same_sha"

#: Prior state was found under an ancestor SHA of the current HEAD; the
#: workflow can run an *incremental* review keyed against the diff
#: between the two SHAs. ``incremental_diff`` and ``commit_shas`` are
#: populated.
SHA_DELTA_DECISION_INCREMENTAL: Final[str] = "incremental"

#: No reusable prior state — either the current SHA had no record, no
#: previous-SHA hint resolved a neighbour, the previous-SHA neighbour is
#: not an ancestor of HEAD (force-push), or the diff probe itself
#: failed. The workflow MUST do a fresh full review under the current
#: head SHA.
SHA_DELTA_DECISION_FRESH_REVIEW: Final[str] = "fresh_review"

#: Maximum unified-diff line count we will inline into the review
#: prompt. Above this we surface only the commit list and let the model
#: re-fetch the diff with ``git diff prev..head``. Mirrors the
#: ``MAX_INLINE_INCREMENTAL_DIFF_LINES`` constant in the legacy resume
#: state module so behaviour is preserved.
MAX_INLINE_INCREMENTAL_DIFF_LINES: Final[int] = 500


# ---------------------------------------------------------------------------
# Git probe protocol + default implementation
# ---------------------------------------------------------------------------


class GitDeltaProbe(Protocol):
    """Minimal git surface :class:`ShaDeltaResolver` depends on.

    Splitting this out as a :class:`Protocol` lets unit tests stub each
    method in isolation without monkey-patching module globals or
    spinning up a real repo. The production implementation
    (:class:`DefaultGitDeltaProbe`) is a thin pass-through to
    :mod:`cli.clients.git_ops`.
    """

    def is_ancestor(self, older_sha: str, newer_sha: str) -> bool:
        """Return whether ``older_sha`` is an ancestor of ``newer_sha``.

        Mirrors :func:`cli.clients.git_ops.git_is_ancestor`. May raise
        :class:`subprocess.CalledProcessError` for git probe failures
        the resolver treats as "decision: fresh_review with reason".
        """
        ...

    def diff_text(self, revision_range: str) -> str:
        """Return ``git diff --unified=3 --no-color <revision_range>``.

        Mirrors :func:`cli.clients.git_ops.git_diff_text`.
        """
        ...

    def commit_shas(self, revision_range: str) -> list[str]:
        """Return commits in ``revision_range`` from oldest to newest.

        Mirrors :func:`cli.clients.git_ops.git_commit_shas`.
        """
        ...


@dataclass(frozen=True)
class DefaultGitDeltaProbe:
    """Production :class:`GitDeltaProbe` backed by :mod:`cli.clients.git_ops`.

    A thin adapter so production code is unchanged. The ``frozen``
    dataclass is empty on purpose — it just gives us a hashable,
    instance-able default that implements the protocol.
    """

    def is_ancestor(self, older_sha: str, newer_sha: str) -> bool:
        from ..clients.git_ops import git_is_ancestor

        return git_is_ancestor(older_sha, newer_sha)

    def diff_text(self, revision_range: str) -> str:
        from ..clients.git_ops import git_diff_text

        return git_diff_text(revision_range)

    def commit_shas(self, revision_range: str) -> list[str]:
        from ..clients.git_ops import git_commit_shas

        return git_commit_shas(revision_range)


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShaDeltaResult:
    """Outcome of a :meth:`ShaDeltaResolver.resolve` call.

    Carries enough information for the workflow to either:

    * Compose an incremental-review prompt block (``incremental_diff``,
      ``commit_shas``, ``previous_reviewed_sha``).
    * Skip the LLM call entirely on a same-SHA re-run.
    * Build a fresh-review prompt under the current head SHA.

    All branches always carry a usable :attr:`prior_state` because the
    manager initialises an empty envelope on miss, so workflow code
    never has to ``if state is None`` — it just inspects
    :attr:`decision` instead.
    """

    decision: str
    current_head_sha: str
    prior_state: PriorReviewState
    previous_reviewed_sha: str | None
    incremental_diff: str | None
    commit_shas: tuple[str, ...]
    reason: str
    fallback_key: PriorReviewKey | None = None

    @property
    def is_incremental(self) -> bool:
        """Convenience: whether the workflow should compose a delta prompt."""
        return self.decision == SHA_DELTA_DECISION_INCREMENTAL

    @property
    def is_fresh_review(self) -> bool:
        """Convenience: whether a fresh full review is required."""
        return self.decision == SHA_DELTA_DECISION_FRESH_REVIEW

    @property
    def is_same_sha(self) -> bool:
        """Convenience: whether the current HEAD matches the prior review."""
        return self.decision == SHA_DELTA_DECISION_SAME_SHA

    @property
    def diff_line_count(self) -> int:
        """Number of lines in :attr:`incremental_diff`, or 0 when absent."""
        if self.incremental_diff is None:
            return 0
        return len(self.incremental_diff.splitlines())


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class ShaDeltaResolver:
    """Decide between ``same_sha`` / ``incremental`` / ``fresh_review``.

    The resolver wires the on-disk state manager (Sub-AC 4.1.3) to the
    git probe so the workflow has a single ``resolve(...)`` call to make
    at the top of every review run.

    Construction via :meth:`from_runner_temp` covers the GitHub Actions
    cadence (``$RUNNER_TEMP/openrouter-review-state``); explicit
    constructors are used by tests and operator tooling.
    """

    def __init__(
        self,
        state_manager: ReviewStateManager,
        *,
        git_probe: GitDeltaProbe | None = None,
        max_inline_diff_lines: int = MAX_INLINE_INCREMENTAL_DIFF_LINES,
        debug: Callable[[int, str], None] | None = None,
    ) -> None:
        if max_inline_diff_lines < 0:
            raise ValueError(
                "max_inline_diff_lines must be >= 0; "
                f"got {max_inline_diff_lines!r}"
            )
        self.state_manager = state_manager
        self.git_probe: GitDeltaProbe = git_probe or DefaultGitDeltaProbe()
        self.max_inline_diff_lines = max_inline_diff_lines
        self._debug = debug or _no_debug

    # ---------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------

    @classmethod
    def from_runner_temp(
        cls,
        runner_temp: str | None,
        *,
        git_probe: GitDeltaProbe | None = None,
        debug: Callable[[int, str], None] | None = None,
    ) -> ShaDeltaResolver:
        """Build a resolver rooted at the conventional GitHub Actions cache dir."""
        state_manager = ReviewStateManager.from_runner_temp(runner_temp)
        return cls(state_manager, git_probe=git_probe, debug=debug)

    # ---------------------------------------------------------------
    # Decision entry point
    # ---------------------------------------------------------------

    def resolve(
        self,
        current_key: PriorReviewKey,
        *,
        previous_head_sha: str | None = None,
        generated_at: str | None = None,
    ) -> ShaDeltaResult:
        """Decide whether the next review run can be incremental.

        Algorithm:

        1. Ask the state manager for prior state under ``current_key``,
           with ``previous_head_sha`` as the fall-back hint. The
           manager returns one of three branches (current_sha hit,
           previous_sha neighbour, or initialised empty).
        2. **current_sha hit** → ``decision = same_sha``. The prior
           state already covers this exact HEAD, so no diff is
           required. ``incremental_diff`` and ``commit_shas`` stay
           empty.
        3. **previous_sha neighbour** → run the ancestry check. If the
           neighbour SHA is an ancestor of the current HEAD, compute
           the diff and commit list and return
           ``decision = incremental``. If not (force-push), return
           ``decision = fresh_review`` with the manager-supplied empty
           envelope (we *do not* return the orphan neighbour state on
           a fresh review — the seed constraint is explicit that we
           start over).
        4. **initialised** → ``decision = fresh_review``. First-time
           PR, evicted cache, model swap, etc.

        Errors from the git probe (missing SHAs, bad refs, transient
        failures) downgrade to ``fresh_review`` with a descriptive
        ``reason`` string rather than raising — the seed contract says
        a force-push that breaks ancestry must fall back gracefully and
        an ambiguous git failure is operationally identical.
        """
        normalized_previous = _normalise_previous_sha(previous_head_sha)

        lookup = self.state_manager.lookup_or_initialize(
            current_key,
            previous_head_sha=normalized_previous,
            generated_at=generated_at,
        )

        if lookup.source == LOOKUP_SOURCE_CURRENT_SHA:
            self._debug(
                1,
                f"SHA-delta: prior state hit on current HEAD {current_key.reviewed_head_sha}; "
                "no incremental diff required",
            )
            return ShaDeltaResult(
                decision=SHA_DELTA_DECISION_SAME_SHA,
                current_head_sha=current_key.reviewed_head_sha,
                prior_state=lookup.state,
                previous_reviewed_sha=current_key.reviewed_head_sha,
                incremental_diff=None,
                commit_shas=(),
                reason=(
                    "prior review state already keyed against the current HEAD"
                ),
                fallback_key=None,
            )

        if lookup.source == LOOKUP_SOURCE_PREVIOUS_SHA:
            return self._resolve_previous_sha_branch(
                current_key=current_key,
                lookup=lookup,
                generated_at=generated_at,
            )

        # LOOKUP_SOURCE_INITIALIZED — no record at all.
        reason = self._format_fresh_review_reason(
            current_key=current_key,
            previous_head_sha=normalized_previous,
        )
        self._debug(1, f"SHA-delta: fresh review required ({reason})")
        return ShaDeltaResult(
            decision=SHA_DELTA_DECISION_FRESH_REVIEW,
            current_head_sha=current_key.reviewed_head_sha,
            prior_state=lookup.state,
            previous_reviewed_sha=None,
            incremental_diff=None,
            commit_shas=(),
            reason=reason,
            fallback_key=None,
        )

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _resolve_previous_sha_branch(
        self,
        *,
        current_key: PriorReviewKey,
        lookup: ReviewStateLookupResult,
        generated_at: str | None,
    ) -> ShaDeltaResult:
        """Continuation path: previous-SHA neighbour found on disk.

        Validates ancestry and computes the SHA-delta if the neighbour
        is reachable from HEAD; otherwise falls back to a fresh review
        envelope (per the seed force-push constraint).
        """
        fallback_key = lookup.fallback_key
        if fallback_key is None:  # defensive — manager always sets it
            reason = (
                "previous-SHA lookup branch hit but fallback_key was missing; "
                "treating as fresh review"
            )
            self._debug(1, f"SHA-delta: {reason}")
            return self._fresh_envelope_result(
                current_key=current_key,
                reason=reason,
                generated_at=generated_at,
            )

        previous_reviewed_sha = fallback_key.reviewed_head_sha
        current_head_sha = current_key.reviewed_head_sha

        try:
            is_ancestor = self.git_probe.is_ancestor(
                previous_reviewed_sha, current_head_sha
            )
        except subprocess.CalledProcessError as exc:
            reason = (
                f"git ancestry probe failed for {previous_reviewed_sha}->"
                f"{current_head_sha} ({_summarize_called_process_error(exc)}); "
                "treating as fresh review"
            )
            self._debug(1, f"SHA-delta: {reason}")
            return self._fresh_envelope_result(
                current_key=current_key,
                reason=reason,
                generated_at=generated_at,
            )

        if not is_ancestor:
            reason = (
                f"previous reviewed SHA {previous_reviewed_sha} is not an "
                f"ancestor of current HEAD {current_head_sha} (force-push); "
                "falling back to fresh full review"
            )
            self._debug(1, f"SHA-delta: {reason}")
            return self._fresh_envelope_result(
                current_key=current_key,
                reason=reason,
                generated_at=generated_at,
            )

        revision_range = f"{previous_reviewed_sha}..{current_head_sha}"
        try:
            raw_diff = self.git_probe.diff_text(revision_range)
            commits = tuple(self.git_probe.commit_shas(revision_range))
        except subprocess.CalledProcessError as exc:
            reason = (
                f"git diff/commit-list probe failed for {revision_range} "
                f"({_summarize_called_process_error(exc)}); treating as fresh review"
            )
            self._debug(1, f"SHA-delta: {reason}")
            return self._fresh_envelope_result(
                current_key=current_key,
                reason=reason,
                generated_at=generated_at,
            )

        diff_line_count = len(raw_diff.splitlines())
        inline_diff: str | None
        if diff_line_count == 0 and not commits:
            # Empty delta. Two SHAs that compute as ancestors but have
            # no diff between them is the same logical case as
            # "same SHA" — flag it as such rather than emitting an
            # empty incremental block.
            self._debug(
                1,
                "SHA-delta: ancestry valid but range "
                f"{revision_range} produced no diff and no commits; "
                "treating as same-SHA",
            )
            return ShaDeltaResult(
                decision=SHA_DELTA_DECISION_SAME_SHA,
                current_head_sha=current_head_sha,
                prior_state=lookup.state,
                previous_reviewed_sha=previous_reviewed_sha,
                incremental_diff=None,
                commit_shas=(),
                reason=(
                    f"prior reviewed SHA {previous_reviewed_sha} resolves to "
                    f"the same tree as current HEAD {current_head_sha}"
                ),
                fallback_key=fallback_key,
            )

        if diff_line_count <= self.max_inline_diff_lines:
            stripped = raw_diff.strip()
            inline_diff = stripped or None
        else:
            inline_diff = None

        self._debug(
            1,
            "SHA-delta: incremental review enabled "
            f"({previous_reviewed_sha}..{current_head_sha}, "
            f"diff_lines={diff_line_count}, "
            f"commits={len(commits)}, "
            f"inline_diff={'yes' if inline_diff is not None else 'no'})",
        )
        return ShaDeltaResult(
            decision=SHA_DELTA_DECISION_INCREMENTAL,
            current_head_sha=current_head_sha,
            prior_state=lookup.state,
            previous_reviewed_sha=previous_reviewed_sha,
            incremental_diff=inline_diff,
            commit_shas=commits,
            reason=(
                f"prior reviewed SHA {previous_reviewed_sha} is an ancestor "
                f"of current HEAD {current_head_sha}; reviewing "
                f"{diff_line_count} diff lines across {len(commits)} commit(s)"
            ),
            fallback_key=fallback_key,
        )

    def _fresh_envelope_result(
        self,
        *,
        current_key: PriorReviewKey,
        reason: str,
        generated_at: str | None,
    ) -> ShaDeltaResult:
        """Build a fresh-review envelope when ancestry/diff fails.

        We do **not** surface the orphan neighbour state here — the
        seed constraint says force-push falls back to a fresh full
        review, so the workflow gets a clean empty envelope keyed
        against the current HEAD instead.
        """
        return ShaDeltaResult(
            decision=SHA_DELTA_DECISION_FRESH_REVIEW,
            current_head_sha=current_key.reviewed_head_sha,
            prior_state=PriorReviewState.empty(
                current_key, generated_at=generated_at
            ),
            previous_reviewed_sha=None,
            incremental_diff=None,
            commit_shas=(),
            reason=reason,
            fallback_key=None,
        )

    def _format_fresh_review_reason(
        self,
        *,
        current_key: PriorReviewKey,
        previous_head_sha: str | None,
    ) -> str:
        if previous_head_sha is None:
            return (
                f"no prior review state for {current_key.cache_key()!r} and "
                "no previous-SHA hint supplied"
            )
        if previous_head_sha == current_key.reviewed_head_sha:
            return (
                f"no prior review state for {current_key.cache_key()!r} "
                "(previous-SHA hint equals current HEAD; nothing to fall back to)"
            )
        return (
            f"no prior review state for {current_key.cache_key()!r} or for "
            f"previous SHA {previous_head_sha!r}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _normalise_previous_sha(value: str | None) -> str | None:
    """Strip and empty-string the supplied previous-SHA hint.

    Mirrors :func:`cli.core.review_state_manager._normalise_previous_sha`
    so the resolver and manager treat empty/whitespace inputs identically;
    duplicated locally to avoid importing a private symbol across modules.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _no_debug(_level: int, _message: str) -> None:
    """Default debug callback that swallows messages.

    The resolver accepts an optional ``debug`` callable so the workflow
    can pipe log output through its own ``make_debug``-shaped logger.
    Production callers always pass one; unit tests can omit it.
    """
    return None


def _summarize_called_process_error(exc: subprocess.CalledProcessError) -> str:
    """Render a one-line summary of a git ``CalledProcessError`` for logs."""
    cmd = exc.cmd
    if isinstance(cmd, (list, tuple)):
        command_text = " ".join(str(part) for part in cmd)
    else:
        command_text = str(cmd)
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    stderr_one_line = " ".join(stderr.split()) if stderr else ""
    if stderr_one_line:
        return f"`{command_text}` exited {exc.returncode}: {stderr_one_line}"
    return f"`{command_text}` exited {exc.returncode}"


__all__ = [
    "MAX_INLINE_INCREMENTAL_DIFF_LINES",
    "SHA_DELTA_DECISION_FRESH_REVIEW",
    "SHA_DELTA_DECISION_INCREMENTAL",
    "SHA_DELTA_DECISION_SAME_SHA",
    "DefaultGitDeltaProbe",
    "GitDeltaProbe",
    "ShaDeltaResolver",
    "ShaDeltaResult",
]
