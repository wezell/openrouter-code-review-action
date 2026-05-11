"""Tests for the SHA-delta detection layer.

Sub-AC 4.2 — exercises :mod:`cli.core.sha_delta`. The resolver decides,
for each review run, whether the workflow can re-use prior state and
review only the SHA-delta or whether it must fall back to a fresh full
review.

Coverage map
------------

* **Same-SHA re-run** — prior state already keyed against the current
  HEAD; resolver short-circuits without invoking the git probe.
* **Incremental review** — prior state on an ancestor SHA; resolver
  validates ancestry, computes the diff and commit list, and surfaces
  the incremental block. Diff truncation by line cap is verified.
* **Force-push fallback** — prior state exists but the previous SHA is
  no longer an ancestor of HEAD (force-push). Per the seed constraint,
  the resolver returns ``decision = fresh_review`` with an empty
  envelope keyed against the current HEAD.
* **First-time PR** — no prior state and no useful previous-SHA hint;
  resolver mints a fresh-review envelope without touching the git probe.
* **Probe failure tolerance** — git probe raises
  :class:`subprocess.CalledProcessError`; the resolver downgrades to a
  fresh review with a descriptive ``reason`` rather than propagating
  the exception (force-push and "git couldn't tell us" are
  operationally identical).
* **Empty-delta degeneracy** — ancestry valid but the range produces no
  diff or commits; resolver collapses to ``same_sha`` rather than
  emitting an empty incremental block.
* **Construction helpers** — ``from_runner_temp`` wires the resolver to
  the conventional GitHub Actions cache directory.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cli.core.review_state import (
    PersistedReviewFinding,
    PriorReviewKey,
    PriorReviewState,
)
from cli.core.review_state_manager import ReviewStateManager
from cli.core.sha_delta import (
    MAX_INLINE_INCREMENTAL_DIFF_LINES,
    SHA_DELTA_DECISION_FRESH_REVIEW,
    SHA_DELTA_DECISION_INCREMENTAL,
    SHA_DELTA_DECISION_SAME_SHA,
    DefaultGitDeltaProbe,
    GitDeltaProbe,
    ShaDeltaResolver,
    ShaDeltaResult,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _key(reviewed_head_sha: str = "head", *, pr_number: int = 42) -> PriorReviewKey:
    return PriorReviewKey(
        repository="owner/repo",
        pr_number=pr_number,
        review_model="anthropic/claude-opus-4.7",
        reviewed_head_sha=reviewed_head_sha,
    )


def _finding(line: int = 10, title: str = "bug") -> PersistedReviewFinding:
    return PersistedReviewFinding(
        path="src/example.py",
        start_line=line,
        end_line=line,
        title=title,
        body="body",
        severity="high",
        side="RIGHT",
    )


@dataclass
class _StubProbe:
    """:class:`GitDeltaProbe` stub that records calls and replies with canned data."""

    is_ancestor_map: dict[tuple[str, str], bool] = field(default_factory=dict)
    is_ancestor_raise: subprocess.CalledProcessError | None = None
    diff_text_map: dict[str, str] = field(default_factory=dict)
    diff_text_raise: subprocess.CalledProcessError | None = None
    commit_shas_map: dict[str, list[str]] = field(default_factory=dict)
    commit_shas_raise: subprocess.CalledProcessError | None = None
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def is_ancestor(self, older_sha: str, newer_sha: str) -> bool:
        self.calls.append(("is_ancestor", (older_sha, newer_sha)))
        if self.is_ancestor_raise is not None:
            raise self.is_ancestor_raise
        return self.is_ancestor_map.get((older_sha, newer_sha), False)

    def diff_text(self, revision_range: str) -> str:
        self.calls.append(("diff_text", (revision_range,)))
        if self.diff_text_raise is not None:
            raise self.diff_text_raise
        return self.diff_text_map.get(revision_range, "")

    def commit_shas(self, revision_range: str) -> list[str]:
        self.calls.append(("commit_shas", (revision_range,)))
        if self.commit_shas_raise is not None:
            raise self.commit_shas_raise
        return list(self.commit_shas_map.get(revision_range, []))


def _resolver(
    tmp_path: Path,
    probe: GitDeltaProbe,
    *,
    max_inline_diff_lines: int = MAX_INLINE_INCREMENTAL_DIFF_LINES,
    debug: list[tuple[int, str]] | None = None,
) -> ShaDeltaResolver:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    debug_cb = (
        (lambda level, message: debug.append((level, message))) if debug is not None else None
    )
    return ShaDeltaResolver(
        manager,
        git_probe=probe,
        max_inline_diff_lines=max_inline_diff_lines,
        debug=debug_cb,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_probe_implements_protocol() -> None:
    probe: GitDeltaProbe = DefaultGitDeltaProbe()
    assert callable(probe.is_ancestor)
    assert callable(probe.diff_text)
    assert callable(probe.commit_shas)


def test_from_runner_temp_anchors_state_manager_to_runner_temp(
    tmp_path: Path,
) -> None:
    resolver = ShaDeltaResolver.from_runner_temp(str(tmp_path))
    assert isinstance(resolver.state_manager, ReviewStateManager)
    assert resolver.state_manager.store.base_dir.parent == tmp_path.resolve()


def test_resolver_rejects_negative_diff_cap(tmp_path: Path) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    with pytest.raises(ValueError, match="max_inline_diff_lines"):
        ShaDeltaResolver(manager, max_inline_diff_lines=-1)


# ---------------------------------------------------------------------------
# Same-SHA branch
# ---------------------------------------------------------------------------


def test_resolve_same_sha_short_circuits_without_probing_git(tmp_path: Path) -> None:
    probe = _StubProbe()
    resolver = _resolver(tmp_path, probe)

    key = _key("sha-current")
    resolver.state_manager.record_review_run(
        key,
        findings=(_finding(),),
        summary_comment_id=99,
    )

    result = resolver.resolve(key)

    assert isinstance(result, ShaDeltaResult)
    assert result.decision == SHA_DELTA_DECISION_SAME_SHA
    assert result.is_same_sha is True
    assert result.previous_reviewed_sha == "sha-current"
    assert result.current_head_sha == "sha-current"
    assert result.incremental_diff is None
    assert result.commit_shas == ()
    assert result.fallback_key is None
    assert result.prior_state.summary_comment_id == 99
    # Same-SHA must not consult git at all.
    assert probe.calls == []


def test_resolve_same_sha_when_previous_hint_matches_current(tmp_path: Path) -> None:
    probe = _StubProbe()
    resolver = _resolver(tmp_path, probe)
    key = _key("sha-current")
    resolver.state_manager.record_review_run(key, findings=(_finding(),))

    result = resolver.resolve(key, previous_head_sha="sha-current")

    assert result.decision == SHA_DELTA_DECISION_SAME_SHA
    assert probe.calls == []


# ---------------------------------------------------------------------------
# Incremental branch — ancestry valid
# ---------------------------------------------------------------------------


def test_resolve_incremental_when_previous_sha_is_ancestor(tmp_path: Path) -> None:
    diff_text = "diff --git a/x b/x\n@@\n-old\n+new\n"
    probe = _StubProbe(
        is_ancestor_map={("sha-old", "sha-new"): True},
        diff_text_map={"sha-old..sha-new": diff_text},
        commit_shas_map={"sha-old..sha-new": ["c1", "c2"]},
    )
    debug: list[tuple[int, str]] = []
    resolver = _resolver(tmp_path, probe, debug=debug)

    # Persist prior state under sha-old.
    resolver.state_manager.record_review_run(
        _key("sha-old"),
        findings=(_finding(line=10, title="bug-A"),),
        summary_comment_id=10,
    )

    result = resolver.resolve(_key("sha-new"), previous_head_sha="sha-old")

    assert result.decision == SHA_DELTA_DECISION_INCREMENTAL
    assert result.is_incremental is True
    assert result.previous_reviewed_sha == "sha-old"
    assert result.current_head_sha == "sha-new"
    assert result.incremental_diff is not None
    assert "diff --git a/x b/x" in result.incremental_diff
    assert result.commit_shas == ("c1", "c2")
    assert result.fallback_key == _key("sha-old")
    # Prior state surfaces with its findings intact (workflow re-keys).
    assert result.prior_state.findings[0].title == "bug-A"
    # Probe was called in the expected order.
    assert probe.calls == [
        ("is_ancestor", ("sha-old", "sha-new")),
        ("diff_text", ("sha-old..sha-new",)),
        ("commit_shas", ("sha-old..sha-new",)),
    ]
    assert any("incremental review enabled" in msg for _, msg in debug)


def test_resolve_incremental_truncates_diff_above_line_cap(tmp_path: Path) -> None:
    big_diff = "\n".join(f"+ line {i}" for i in range(50))
    probe = _StubProbe(
        is_ancestor_map={("sha-old", "sha-new"): True},
        diff_text_map={"sha-old..sha-new": big_diff},
        commit_shas_map={"sha-old..sha-new": ["c1"]},
    )
    resolver = _resolver(tmp_path, probe, max_inline_diff_lines=10)
    resolver.state_manager.record_review_run(_key("sha-old"))

    result = resolver.resolve(_key("sha-new"), previous_head_sha="sha-old")

    assert result.decision == SHA_DELTA_DECISION_INCREMENTAL
    # Above the cap, inline diff is dropped but commits remain.
    assert result.incremental_diff is None
    assert result.commit_shas == ("c1",)
    # The result still reports the actual diff size for telemetry.
    assert "50 diff lines across 1 commit" in result.reason


def test_resolve_incremental_keeps_diff_at_exact_cap(tmp_path: Path) -> None:
    diff = "\n".join(f"+ line {i}" for i in range(10))
    probe = _StubProbe(
        is_ancestor_map={("sha-old", "sha-new"): True},
        diff_text_map={"sha-old..sha-new": diff},
        commit_shas_map={"sha-old..sha-new": ["c1"]},
    )
    resolver = _resolver(tmp_path, probe, max_inline_diff_lines=10)
    resolver.state_manager.record_review_run(_key("sha-old"))

    result = resolver.resolve(_key("sha-new"), previous_head_sha="sha-old")

    assert result.decision == SHA_DELTA_DECISION_INCREMENTAL
    assert result.incremental_diff is not None
    assert result.incremental_diff.count("+ line ") == 10


def test_resolve_collapses_to_same_sha_when_range_is_empty(tmp_path: Path) -> None:
    """Two SHAs that ancestry-match but have no diff are functionally
    a same-SHA scenario — emit ``same_sha`` instead of an empty
    incremental block so the workflow doesn't waste an LLM call."""
    probe = _StubProbe(
        is_ancestor_map={("sha-old", "sha-new"): True},
        diff_text_map={"sha-old..sha-new": ""},
        commit_shas_map={"sha-old..sha-new": []},
    )
    resolver = _resolver(tmp_path, probe)
    resolver.state_manager.record_review_run(
        _key("sha-old"),
        findings=(_finding(),),
    )

    result = resolver.resolve(_key("sha-new"), previous_head_sha="sha-old")

    assert result.decision == SHA_DELTA_DECISION_SAME_SHA
    assert result.is_same_sha is True
    assert result.incremental_diff is None
    assert result.commit_shas == ()
    # Prior state still surfaces (it's not "fresh review") and the
    # fallback_key is reported so the workflow knows where the data
    # came from.
    assert result.fallback_key == _key("sha-old")
    assert result.previous_reviewed_sha == "sha-old"


def test_diff_line_count_property_reports_inline_size(tmp_path: Path) -> None:
    diff = "line-a\nline-b\nline-c"
    probe = _StubProbe(
        is_ancestor_map={("o", "n"): True},
        diff_text_map={"o..n": diff},
        commit_shas_map={"o..n": ["c"]},
    )
    resolver = _resolver(tmp_path, probe)
    resolver.state_manager.record_review_run(_key("o"))

    result = resolver.resolve(_key("n"), previous_head_sha="o")
    assert result.diff_line_count == 3


# ---------------------------------------------------------------------------
# Force-push fallback
# ---------------------------------------------------------------------------


def test_resolve_force_push_falls_back_to_fresh_review(tmp_path: Path) -> None:
    probe = _StubProbe(
        is_ancestor_map={("sha-old", "sha-force"): False},
    )
    debug: list[tuple[int, str]] = []
    resolver = _resolver(tmp_path, probe, debug=debug)

    resolver.state_manager.record_review_run(
        _key("sha-old"),
        findings=(_finding(),),
    )

    result = resolver.resolve(_key("sha-force"), previous_head_sha="sha-old")

    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert result.is_fresh_review is True
    # The orphan neighbour state is NOT surfaced — fresh review means
    # an empty envelope under the current key.
    assert result.prior_state.findings == ()
    assert result.prior_state.key.reviewed_head_sha == "sha-force"
    assert result.previous_reviewed_sha is None
    assert result.incremental_diff is None
    assert result.commit_shas == ()
    assert result.fallback_key is None
    assert "force-push" in result.reason
    # diff/commit probes must not run after ancestry rejected.
    assert [name for name, _ in probe.calls] == ["is_ancestor"]
    assert any("force-push" in msg for _, msg in debug)


def test_resolve_treats_empty_previous_hint_as_fresh_review(tmp_path: Path) -> None:
    probe = _StubProbe()
    resolver = _resolver(tmp_path, probe)

    result = resolver.resolve(_key("sha-current"), previous_head_sha="   ")

    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert result.previous_reviewed_sha is None
    # No git probes for first-time PR.
    assert probe.calls == []
    assert "no previous-SHA hint" in result.reason or "no prior review state" in result.reason


def test_resolve_first_time_pr_initializes_fresh(tmp_path: Path) -> None:
    probe = _StubProbe()
    resolver = _resolver(tmp_path, probe)

    result = resolver.resolve(_key("sha-current"))

    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert result.prior_state.findings == ()
    assert result.prior_state.key == _key("sha-current")
    assert probe.calls == []


def test_resolve_no_neighbour_initialises_without_probing_git(tmp_path: Path) -> None:
    """Previous-SHA hint provided but no record on disk; resolver
    short-circuits to fresh review without touching git (we have no
    ``previous_reviewed_sha`` to validate)."""
    probe = _StubProbe()
    resolver = _resolver(tmp_path, probe)

    result = resolver.resolve(
        _key("sha-current"),
        previous_head_sha="ghost-sha",
    )

    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert probe.calls == []
    assert "ghost-sha" in result.reason


# ---------------------------------------------------------------------------
# Probe failure tolerance
# ---------------------------------------------------------------------------


def test_resolve_falls_back_when_ancestry_probe_raises(tmp_path: Path) -> None:
    err = subprocess.CalledProcessError(
        128,
        ["git", "merge-base", "--is-ancestor", "sha-old", "sha-new"],
        stderr="fatal: bad object",
    )
    probe = _StubProbe(is_ancestor_raise=err)
    resolver = _resolver(tmp_path, probe)
    resolver.state_manager.record_review_run(_key("sha-old"))

    result = resolver.resolve(_key("sha-new"), previous_head_sha="sha-old")

    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert "git ancestry probe failed" in result.reason
    assert "fatal: bad object" in result.reason
    assert result.prior_state.findings == ()


def test_resolve_falls_back_when_diff_probe_raises(tmp_path: Path) -> None:
    err = subprocess.CalledProcessError(
        128,
        ["git", "diff", "sha-old..sha-new"],
        stderr="fatal: bad revision",
    )
    probe = _StubProbe(
        is_ancestor_map={("sha-old", "sha-new"): True},
        diff_text_raise=err,
    )
    resolver = _resolver(tmp_path, probe)
    resolver.state_manager.record_review_run(_key("sha-old"))

    result = resolver.resolve(_key("sha-new"), previous_head_sha="sha-old")

    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert "git diff/commit-list probe failed" in result.reason


def test_resolve_falls_back_when_commit_probe_raises(tmp_path: Path) -> None:
    err = subprocess.CalledProcessError(
        128,
        ["git", "rev-list", "sha-old..sha-new"],
        stderr="fatal: ambiguous argument",
    )
    probe = _StubProbe(
        is_ancestor_map={("sha-old", "sha-new"): True},
        diff_text_map={"sha-old..sha-new": "diff"},
        commit_shas_raise=err,
    )
    resolver = _resolver(tmp_path, probe)
    resolver.state_manager.record_review_run(_key("sha-old"))

    result = resolver.resolve(_key("sha-new"), previous_head_sha="sha-old")

    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert "git diff/commit-list probe failed" in result.reason


def test_resolve_propagates_unexpected_probe_exceptions(tmp_path: Path) -> None:
    """A non-CalledProcessError from the probe (e.g. RuntimeError from
    ``git`` lookup itself) is *not* swallowed — those are bugs the
    workflow needs to see, not operational ancestry failures."""

    class _BoomProbe:
        def is_ancestor(self, older_sha: str, newer_sha: str) -> bool:  # noqa: ARG002
            raise RuntimeError("git binary not on PATH")

        def diff_text(self, revision_range: str) -> str:  # noqa: ARG002
            return ""

        def commit_shas(self, revision_range: str) -> list[str]:  # noqa: ARG002
            return []

    resolver = _resolver(tmp_path, _BoomProbe())
    resolver.state_manager.record_review_run(_key("sha-old"))

    with pytest.raises(RuntimeError, match="git binary"):
        resolver.resolve(_key("sha-new"), previous_head_sha="sha-old")


# ---------------------------------------------------------------------------
# State retention across decisions
# ---------------------------------------------------------------------------


def test_incremental_result_carries_prior_findings_for_workflow_to_rekey(
    tmp_path: Path,
) -> None:
    probe = _StubProbe(
        is_ancestor_map={("sha-old", "sha-new"): True},
        diff_text_map={"sha-old..sha-new": "diff"},
        commit_shas_map={"sha-old..sha-new": ["c1"]},
    )
    resolver = _resolver(tmp_path, probe)

    resolver.state_manager.record_review_run(
        _key("sha-old"),
        findings=(
            _finding(line=10, title="A"),
            _finding(line=20, title="B"),
        ),
        carried_forward_comment_ids=("legacy-id",),
        summary_comment_id=12,
    )

    result = resolver.resolve(_key("sha-new"), previous_head_sha="sha-old")

    assert result.decision == SHA_DELTA_DECISION_INCREMENTAL
    titles = tuple(f.title for f in result.prior_state.findings)
    assert titles == ("A", "B")
    assert result.prior_state.carried_forward_comment_ids == ("legacy-id",)
    assert result.prior_state.summary_comment_id == 12
    # The state surfaced is still keyed against the old SHA — it's the
    # workflow's job to call ``state_manager.migrate_to_sha`` after
    # validating ancestry; the resolver intentionally does not re-key.
    assert result.prior_state.key.reviewed_head_sha == "sha-old"


def test_resolve_propagates_generated_at_into_initialised_envelope(
    tmp_path: Path,
) -> None:
    probe = _StubProbe()
    resolver = _resolver(tmp_path, probe)

    result = resolver.resolve(
        _key("sha-current"),
        generated_at="2026-05-06T12:00:00Z",
    )

    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert result.prior_state.generated_at == "2026-05-06T12:00:00Z"


def test_resolve_propagates_generated_at_when_force_push_falls_back(
    tmp_path: Path,
) -> None:
    probe = _StubProbe(is_ancestor_map={("sha-old", "sha-force"): False})
    resolver = _resolver(tmp_path, probe)
    resolver.state_manager.record_review_run(_key("sha-old"))

    result = resolver.resolve(
        _key("sha-force"),
        previous_head_sha="sha-old",
        generated_at="2026-05-06T12:00:00Z",
    )

    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert result.prior_state.generated_at == "2026-05-06T12:00:00Z"


# ---------------------------------------------------------------------------
# End-to-end across multiple manager generations
# ---------------------------------------------------------------------------


def test_three_run_incremental_flow_with_workflow_rekey(tmp_path: Path) -> None:
    """Simulates the full multi-run flow:

    1. Run 1 records under sha1.
    2. Run 2 resolves ``incremental`` for sha1→sha2; workflow re-keys
       and records under sha2.
    3. Run 3 resolves ``incremental`` for sha2→sha3; workflow re-keys
       and records under sha3.

    Each step exercises both the resolver (decision + diff) and the
    manager (re-keying via ``migrate_to_sha`` and ``record_state``)
    against the same on-disk directory.
    """
    diffs = {
        "sha1..sha2": "diff-1\n",
        "sha2..sha3": "diff-2\n",
    }
    commits = {
        "sha1..sha2": ["c1"],
        "sha2..sha3": ["c2"],
    }
    probe = _StubProbe(
        is_ancestor_map={
            ("sha1", "sha2"): True,
            ("sha2", "sha3"): True,
        },
        diff_text_map=diffs,
        commit_shas_map=commits,
    )

    manager = ReviewStateManager.from_base_dir(tmp_path)
    resolver = ShaDeltaResolver(manager, git_probe=probe)

    # ---- Run 1: fresh review on sha1.
    run1_decision = resolver.resolve(_key("sha1"))
    assert run1_decision.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    manager.record_review_run(
        _key("sha1"),
        findings=(_finding(line=10, title="A"),),
        summary_comment_id=1,
    )

    # ---- Run 2: incremental sha1 → sha2.
    run2_decision = resolver.resolve(_key("sha2"), previous_head_sha="sha1")
    assert run2_decision.decision == SHA_DELTA_DECISION_INCREMENTAL
    rekeyed = manager.migrate_to_sha(run2_decision.prior_state, "sha2")
    accumulated = rekeyed.with_findings((*rekeyed.findings, _finding(line=20, title="B")))
    manager.record_state(accumulated)

    # ---- Run 3: incremental sha2 → sha3.
    run3_decision = resolver.resolve(_key("sha3"), previous_head_sha="sha2")
    assert run3_decision.decision == SHA_DELTA_DECISION_INCREMENTAL
    assert run3_decision.previous_reviewed_sha == "sha2"
    titles = tuple(f.title for f in run3_decision.prior_state.findings)
    assert titles == ("A", "B")
    assert run3_decision.commit_shas == ("c2",)


def test_force_push_after_incremental_run_resets_to_fresh_review(
    tmp_path: Path,
) -> None:
    probe = _StubProbe(
        is_ancestor_map={
            ("sha1", "sha2"): True,
            # After a force-push, sha2 is no longer reachable from
            # the new HEAD ``sha-fp``.
            ("sha2", "sha-fp"): False,
        },
        diff_text_map={"sha1..sha2": "diff"},
        commit_shas_map={"sha1..sha2": ["c1"]},
    )
    manager = ReviewStateManager.from_base_dir(tmp_path)
    resolver = ShaDeltaResolver(manager, git_probe=probe)

    manager.record_review_run(_key("sha1"), findings=(_finding(),))
    incremental = resolver.resolve(_key("sha2"), previous_head_sha="sha1")
    assert incremental.decision == SHA_DELTA_DECISION_INCREMENTAL
    rekeyed = manager.migrate_to_sha(incremental.prior_state, "sha2")
    manager.record_state(rekeyed)

    # Force-push: HEAD jumps to sha-fp, sha2 is no longer an ancestor.
    fresh = resolver.resolve(_key("sha-fp"), previous_head_sha="sha2")
    assert fresh.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    assert fresh.prior_state.findings == ()
    assert "force-push" in fresh.reason


# ---------------------------------------------------------------------------
# Probe stub equivalence — ensures stub respects protocol contract
# ---------------------------------------------------------------------------


def test_stub_probe_records_every_call_in_order(tmp_path: Path) -> None:
    probe = _StubProbe(
        is_ancestor_map={("a", "b"): True},
        diff_text_map={"a..b": "x"},
        commit_shas_map={"a..b": ["c1"]},
    )
    resolver = _resolver(tmp_path, probe)
    resolver.state_manager.record_review_run(_key("a"))

    resolver.resolve(_key("b"), previous_head_sha="a")
    names = [name for name, _ in probe.calls]
    assert names == ["is_ancestor", "diff_text", "commit_shas"]


# ---------------------------------------------------------------------------
# Public API smoke test
# ---------------------------------------------------------------------------


def test_public_exports_are_importable() -> None:
    from cli.core import sha_delta as module

    expected_names: Sequence[str] = (
        "MAX_INLINE_INCREMENTAL_DIFF_LINES",
        "SHA_DELTA_DECISION_FRESH_REVIEW",
        "SHA_DELTA_DECISION_INCREMENTAL",
        "SHA_DELTA_DECISION_SAME_SHA",
        "DefaultGitDeltaProbe",
        "GitDeltaProbe",
        "ShaDeltaResolver",
        "ShaDeltaResult",
    )
    for name in expected_names:
        assert hasattr(module, name), f"{name!r} not exported"
    # __all__ is the contract the module advertises.
    assert set(expected_names) == set(module.__all__)


def test_default_probe_delegates_to_git_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default probe is a thin pass-through; verify each method
    forwards to the corresponding ``cli.clients.git_ops`` function."""
    from cli.clients import git_ops

    is_ancestor_calls: list[tuple[str, str]] = []
    diff_calls: list[str] = []
    commit_calls: list[str] = []

    def fake_is_ancestor(older: str, newer: str) -> bool:
        is_ancestor_calls.append((older, newer))
        return True

    def fake_diff_text(rev_range: str, *, unified: int = 3) -> str:
        diff_calls.append(rev_range)
        assert unified == 3
        return "diff-output"

    def fake_commit_shas(rev_range: str) -> list[str]:
        commit_calls.append(rev_range)
        return ["c1", "c2"]

    monkeypatch.setattr(git_ops, "git_is_ancestor", fake_is_ancestor)
    monkeypatch.setattr(git_ops, "git_diff_text", fake_diff_text)
    monkeypatch.setattr(git_ops, "git_commit_shas", fake_commit_shas)

    probe = DefaultGitDeltaProbe()
    assert probe.is_ancestor("a", "b") is True
    assert probe.diff_text("a..b") == "diff-output"
    assert probe.commit_shas("a..b") == ["c1", "c2"]
    assert is_ancestor_calls == [("a", "b")]
    assert diff_calls == ["a..b"]
    assert commit_calls == ["a..b"]


# ---------------------------------------------------------------------------
# Reason-string ergonomics
# ---------------------------------------------------------------------------


def test_fresh_review_reason_distinguishes_no_hint_from_missing_neighbour(
    tmp_path: Path,
) -> None:
    probe = _StubProbe()
    resolver = _resolver(tmp_path, probe)

    no_hint = resolver.resolve(_key("sha"))
    with_hint = resolver.resolve(_key("sha"), previous_head_sha="other")

    assert "no previous-SHA hint" in no_hint.reason
    assert "other" in with_hint.reason
    # Reason strings must not be identical so operator log scanners can
    # tell the two situations apart.
    assert no_hint.reason != with_hint.reason


def test_incremental_reason_includes_diff_size_and_commit_count(
    tmp_path: Path,
) -> None:
    probe = _StubProbe(
        is_ancestor_map={("a", "b"): True},
        diff_text_map={"a..b": "+l1\n+l2\n+l3\n"},
        commit_shas_map={"a..b": ["c1"]},
    )
    resolver = _resolver(tmp_path, probe)
    resolver.state_manager.record_review_run(_key("a"))

    result = resolver.resolve(_key("b"), previous_head_sha="a")
    assert "3 diff lines" in result.reason
    assert "1 commit" in result.reason


def test_state_round_trips_through_resolver_without_mutation(tmp_path: Path) -> None:
    """The resolver must not mutate the persisted state when surfacing
    it on the incremental path — :class:`PriorReviewState` is frozen,
    but freezing is enforced at construction time, so confirm callers
    get the *same* object back rather than a re-derived shallow copy
    that drops fields."""
    probe = _StubProbe(
        is_ancestor_map={("o", "n"): True},
        diff_text_map={"o..n": "diff"},
        commit_shas_map={"o..n": ["c1"]},
    )
    resolver = _resolver(tmp_path, probe)

    saved_finding = _finding(line=10, title="cross-run")
    resolver.state_manager.record_review_run(
        _key("o"),
        findings=(saved_finding,),
        carried_forward_comment_ids=("legacy",),
        notes=("first",),
        generated_at="2026-05-06T12:00:00Z",
    )

    result = resolver.resolve(_key("n"), previous_head_sha="o")
    assert result.prior_state.findings == (saved_finding,)
    assert result.prior_state.carried_forward_comment_ids == ("legacy",)
    assert result.prior_state.notes == ("first",)
    assert result.prior_state.generated_at == "2026-05-06T12:00:00Z"
    # PriorReviewState is frozen so attempts to mutate must raise.
    with pytest.raises(Exception):  # noqa: BLE001 — dataclass-frozen FrozenInstanceError
        result.prior_state.findings = ()  # type: ignore[misc]


def test_empty_state_in_initialised_envelope_uses_current_key(tmp_path: Path) -> None:
    probe = _StubProbe()
    resolver = _resolver(tmp_path, probe)

    result = resolver.resolve(_key("sha-current"))
    assert result.decision == SHA_DELTA_DECISION_FRESH_REVIEW
    # The empty state envelope is keyed against the *current* HEAD so
    # the next ``record_review_run`` lands at the right cache file.
    assert result.prior_state.key == _key("sha-current")
    assert isinstance(result.prior_state, PriorReviewState)


# ---------------------------------------------------------------------------
# Sanity: API constants are not strings the workflow can typo silently
# ---------------------------------------------------------------------------


def test_decision_constants_are_distinct() -> None:
    decisions = {
        SHA_DELTA_DECISION_SAME_SHA,
        SHA_DELTA_DECISION_INCREMENTAL,
        SHA_DELTA_DECISION_FRESH_REVIEW,
    }
    assert len(decisions) == 3
    # No accidental whitespace / casing overlaps.
    for value in decisions:
        assert value.strip() == value
        assert value.lower() == value
