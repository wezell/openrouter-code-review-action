"""Tests for the prior-review state lookup/update manager.

Sub-AC 4.1.3 — verifies the service layer
(``cli/core/review_state_manager.py``) correctly implements the
state-lookup and state-update API keyed by PR identifier:

* First-time PRs initialise an empty :class:`PriorReviewState` envelope
  rather than raising; the workflow always receives a usable state
  object.
* Returning PRs hit the on-disk record keyed against the current head
  SHA.
* When the current SHA misses but a previous-SHA neighbour exists, the
  manager surfaces the older record and reports the fall-back key so
  the workflow can decide whether to re-key it.
* ``record_review_run`` always writes a fresh, fully-specified envelope
  atomically so the next run reads a consistent snapshot.
* ``migrate_to_sha`` clones a state record under a new head SHA without
  touching disk so callers can validate ancestry first.
* The ``from_runner_temp`` / ``from_base_dir`` constructors anchor the
  underlying store at the conventional GitHub Actions cache path while
  remaining usable in plain unit-test environments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.core.review_state import (
    REVIEW_STATE_SCHEMA_VERSION,
    PersistedReviewFinding,
    PriorReviewKey,
    PriorReviewState,
)
from cli.core.review_state_manager import (
    LOOKUP_SOURCE_CURRENT_SHA,
    LOOKUP_SOURCE_INITIALIZED,
    LOOKUP_SOURCE_PREVIOUS_SHA,
    ReviewStateLookupResult,
    ReviewStateManager,
)
from cli.core.review_state_store import (
    DEFAULT_STATE_DIRECTORY_NAME,
    ReviewStateStore,
)

# ---------------------------------------------------------------------------
# Sample factories
# ---------------------------------------------------------------------------


def _key(reviewed_head_sha: str = "headsha", *, pr_number: int = 42) -> PriorReviewKey:
    return PriorReviewKey(
        repository="owner/repo",
        pr_number=pr_number,
        review_model="anthropic/claude-opus-4.7",
        reviewed_head_sha=reviewed_head_sha,
    )


def _finding(line: int = 10, title: str = "Possible NPE") -> PersistedReviewFinding:
    return PersistedReviewFinding(
        path="src/example.py",
        start_line=line,
        end_line=line,
        title=title,
        body="Body",
        severity="high",
        suggestion="Add a guard clause",
        side="RIGHT",
        priority=1,
        confidence_score=0.9,
        inline_comment_id=4242,
        review_thread_id="PRRT_kwDOA",
    )


def _state(
    *,
    key: PriorReviewKey | None = None,
    findings: tuple[PersistedReviewFinding, ...] = (),
    summary_comment_id: int | None = 99,
) -> PriorReviewState:
    return PriorReviewState(
        schema_version=REVIEW_STATE_SCHEMA_VERSION,
        key=key or _key(),
        findings=findings,
        carried_forward_comment_ids=("comment-A",),
        summary_comment_id=summary_comment_id,
        generated_at="2026-05-06T12:00:00Z",
    )


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def test_from_runner_temp_anchors_at_default_cache_directory(tmp_path: Path) -> None:
    manager = ReviewStateManager.from_runner_temp(str(tmp_path))
    assert isinstance(manager.store, ReviewStateStore)
    assert manager.store.base_dir == tmp_path.resolve() / DEFAULT_STATE_DIRECTORY_NAME


def test_from_runner_temp_falls_back_to_cwd_when_runner_temp_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manager = ReviewStateManager.from_runner_temp(None)
    assert manager.store.base_dir.parent == tmp_path.resolve()
    assert manager.store.base_dir.name == DEFAULT_STATE_DIRECTORY_NAME


def test_from_base_dir_uses_explicit_path(tmp_path: Path) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path / "custom")
    assert manager.store.base_dir == tmp_path / "custom"


# ---------------------------------------------------------------------------
# Lookup: first-time initialization
# ---------------------------------------------------------------------------


def test_lookup_or_initialize_returns_empty_state_for_first_time_pr(
    tmp_path: Path,
) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    result = manager.lookup_or_initialize(_key("freshsha"))

    assert isinstance(result, ReviewStateLookupResult)
    assert result.initialized is True
    assert result.source == LOOKUP_SOURCE_INITIALIZED
    assert result.fallback_key is None

    # The envelope is empty but well-formed and keyed against the
    # caller-supplied identifier — no second call required.
    assert result.state.key == _key("freshsha")
    assert result.state.findings == ()
    assert result.state.carried_forward_comment_ids == ()
    assert result.state.summary_comment_id is None


def test_lookup_or_initialize_does_not_write_to_disk_on_initialize(
    tmp_path: Path,
) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    manager.lookup_or_initialize(_key())

    # Initialization must not eagerly persist; only record_review_run
    # writes. This keeps a "lookup, decide not to write" path side-effect
    # free so a dry-run can lookup safely.
    assert list(tmp_path.iterdir()) == [] or all(
        entry.is_dir() for entry in tmp_path.iterdir()
    )


def test_lookup_or_initialize_propagates_generated_at_for_fresh_envelope(
    tmp_path: Path,
) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    result = manager.lookup_or_initialize(
        _key(),
        generated_at="2026-05-06T12:34:56Z",
    )
    assert result.initialized is True
    assert result.state.generated_at == "2026-05-06T12:34:56Z"


# ---------------------------------------------------------------------------
# Lookup: hit on the current head SHA
# ---------------------------------------------------------------------------


def test_lookup_or_initialize_returns_existing_state_for_current_sha(
    tmp_path: Path,
) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    saved = _state(findings=(_finding(),))
    manager.store.save(saved)

    result = manager.lookup_or_initialize(saved.key)

    assert result.initialized is False
    assert result.source == LOOKUP_SOURCE_CURRENT_SHA
    assert result.fallback_key is None
    assert result.state == saved


def test_lookup_or_initialize_prefers_current_sha_over_previous_sha(
    tmp_path: Path,
) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    older = _state(key=_key("oldsha"), findings=(_finding(line=1, title="OLD"),))
    newer = _state(key=_key("newsha"), findings=(_finding(line=2, title="NEW"),))
    manager.store.save(older)
    manager.store.save(newer)

    result = manager.lookup_or_initialize(
        _key("newsha"),
        previous_head_sha="oldsha",
    )
    assert result.source == LOOKUP_SOURCE_CURRENT_SHA
    assert result.state == newer


# ---------------------------------------------------------------------------
# Lookup: fall-back to previous SHA
# ---------------------------------------------------------------------------


def test_lookup_or_initialize_falls_back_to_previous_sha_when_current_missing(
    tmp_path: Path,
) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    older = _state(key=_key("oldsha"), findings=(_finding(),))
    manager.store.save(older)

    result = manager.lookup_or_initialize(
        _key("newsha"),
        previous_head_sha="oldsha",
    )
    assert result.initialized is False
    assert result.source == LOOKUP_SOURCE_PREVIOUS_SHA
    assert result.fallback_key == _key("oldsha")
    # The state surfaced is the *original* — re-keying to the new SHA
    # is the workflow's responsibility because that decision depends on
    # ancestry checks the manager intentionally does not do.
    assert result.state == older


def test_lookup_or_initialize_initializes_when_previous_sha_also_missing(
    tmp_path: Path,
) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    result = manager.lookup_or_initialize(
        _key("newsha"),
        previous_head_sha="ghost-sha",
    )
    assert result.initialized is True
    assert result.source == LOOKUP_SOURCE_INITIALIZED
    assert result.fallback_key is None


def test_lookup_or_initialize_treats_empty_previous_sha_as_unset(
    tmp_path: Path,
) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    result = manager.lookup_or_initialize(
        _key("newsha"),
        previous_head_sha="   ",
    )
    assert result.source == LOOKUP_SOURCE_INITIALIZED


def test_lookup_or_initialize_ignores_previous_sha_equal_to_current(
    tmp_path: Path,
) -> None:
    # If the workflow accidentally passes the current SHA as the
    # previous SHA hint (e.g. re-running the same workflow without any
    # new commit) we must not double-search the same key.
    manager = ReviewStateManager.from_base_dir(tmp_path)
    result = manager.lookup_or_initialize(
        _key("samesha"),
        previous_head_sha="samesha",
    )
    assert result.source == LOOKUP_SOURCE_INITIALIZED
    assert result.fallback_key is None


# ---------------------------------------------------------------------------
# has_state_for
# ---------------------------------------------------------------------------


def test_has_state_for_reports_existence(tmp_path: Path) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    assert manager.has_state_for(_key()) is False

    manager.store.save(_state())
    assert manager.has_state_for(_key()) is True


# ---------------------------------------------------------------------------
# record_review_run
# ---------------------------------------------------------------------------


def test_record_review_run_writes_fresh_envelope(tmp_path: Path) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    record = manager.record_review_run(
        _key(),
        findings=(_finding(),),
        carried_forward_comment_ids=("comment-A",),
        summary_comment_id=99,
        generated_at="2026-05-06T12:00:00Z",
        notes=("first run",),
    )
    assert record.path.is_file()
    assert record.cache_key == _key().cache_key()

    loaded = manager.store.load(_key())
    assert loaded is not None
    assert loaded.findings == (_finding(),)
    assert loaded.carried_forward_comment_ids == ("comment-A",)
    assert loaded.summary_comment_id == 99
    assert loaded.notes == ("first run",)
    assert loaded.schema_version == REVIEW_STATE_SCHEMA_VERSION


def test_record_review_run_overwrites_prior_record_atomically(
    tmp_path: Path,
) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    manager.record_review_run(
        _key(),
        findings=(_finding(line=1, title="A"),),
    )
    manager.record_review_run(
        _key(),
        findings=(_finding(line=2, title="B"),),
        summary_comment_id=123,
    )

    loaded = manager.store.load(_key())
    assert loaded is not None
    assert len(loaded.findings) == 1
    assert loaded.findings[0].title == "B"
    assert loaded.summary_comment_id == 123

    # Atomic write must not leave a tempfile behind.
    files = sorted(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"


def test_record_review_run_accepts_zero_findings(tmp_path: Path) -> None:
    # A clean PR (no findings, no carried-forward) is a valid state to
    # persist — the next run still wants the summary comment ID and the
    # reviewed-head metadata.
    manager = ReviewStateManager.from_base_dir(tmp_path)
    record = manager.record_review_run(
        _key(),
        summary_comment_id=42,
        generated_at="2026-05-06T13:00:00Z",
    )
    loaded = manager.store.load(_key())
    assert record.path.is_file()
    assert loaded is not None
    assert loaded.findings == ()
    assert loaded.carried_forward_comment_ids == ()
    assert loaded.summary_comment_id == 42


def test_record_state_persists_a_prebuilt_envelope(tmp_path: Path) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    state = _state(findings=(_finding(),))
    record = manager.record_state(state)
    assert record.path == manager.store.path_for(state.key)
    loaded = manager.store.load(state.key)
    assert loaded == state


# ---------------------------------------------------------------------------
# migrate_to_sha
# ---------------------------------------------------------------------------


def test_migrate_to_sha_rekeys_state_without_writing(tmp_path: Path) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    state = _state(key=_key("oldsha"), findings=(_finding(),))
    manager.store.save(state)

    rekeyed = manager.migrate_to_sha(state, "newsha")

    # In-memory only — disk still has only the older record.
    assert rekeyed.key == _key("newsha")
    assert rekeyed.reviewed_head_sha == "newsha"
    assert rekeyed.findings == state.findings
    assert rekeyed.carried_forward_comment_ids == state.carried_forward_comment_ids
    assert rekeyed.summary_comment_id == state.summary_comment_id
    assert rekeyed.notes == state.notes
    # Disk untouched.
    assert manager.store.exists(_key("newsha")) is False
    assert manager.store.exists(_key("oldsha")) is True


def test_migrate_to_sha_preserves_generated_at_by_default() -> None:
    manager = ReviewStateManager.from_base_dir(Path("/tmp/unused"))
    state = _state()
    rekeyed = manager.migrate_to_sha(state, "newsha")
    assert rekeyed.generated_at == state.generated_at


def test_migrate_to_sha_overrides_generated_at_when_supplied() -> None:
    manager = ReviewStateManager.from_base_dir(Path("/tmp/unused"))
    state = _state()
    rekeyed = manager.migrate_to_sha(
        state,
        "newsha",
        generated_at="2026-06-07T00:00:00Z",
    )
    assert rekeyed.generated_at == "2026-06-07T00:00:00Z"


# ---------------------------------------------------------------------------
# discard
# ---------------------------------------------------------------------------


def test_discard_removes_existing_record(tmp_path: Path) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    manager.record_review_run(_key())
    assert manager.has_state_for(_key()) is True

    assert manager.discard(_key()) is True
    assert manager.has_state_for(_key()) is False


def test_discard_returns_false_for_missing_record(tmp_path: Path) -> None:
    manager = ReviewStateManager.from_base_dir(tmp_path)
    assert manager.discard(_key()) is False


# ---------------------------------------------------------------------------
# End-to-end: simulated cross-run flow
# ---------------------------------------------------------------------------


def test_first_run_initializes_then_records(tmp_path: Path) -> None:
    """First-time PR: lookup yields empty, run records, second lookup
    on the same SHA returns the record."""
    manager = ReviewStateManager.from_base_dir(tmp_path)
    key = _key("sha1")

    # First lookup — empty.
    first = manager.lookup_or_initialize(key)
    assert first.initialized is True
    assert first.state.findings == ()

    # The workflow did its review and recorded outputs.
    manager.record_review_run(
        key,
        findings=(_finding(line=10, title="bug"),),
        summary_comment_id=12345,
        generated_at="2026-05-06T12:00:00Z",
    )

    # Second lookup on the same SHA — hits the persisted record.
    second = manager.lookup_or_initialize(key)
    assert second.initialized is False
    assert second.source == LOOKUP_SOURCE_CURRENT_SHA
    assert second.state.summary_comment_id == 12345
    assert len(second.state.findings) == 1


def test_incremental_review_continuation_rekeys_through_manager(
    tmp_path: Path,
) -> None:
    """Returning PR with new commits: the previous-SHA record is found
    and the workflow re-keys it to the new SHA before persisting."""
    manager = ReviewStateManager.from_base_dir(tmp_path)

    # Initial review on sha1.
    initial = _state(
        key=_key("sha1"),
        findings=(_finding(line=10, title="bug-A"),),
    )
    manager.record_state(initial)

    # PR HEAD advances to sha2; lookup with previous-SHA hint.
    fallback = manager.lookup_or_initialize(
        _key("sha2"),
        previous_head_sha="sha1",
    )
    assert fallback.source == LOOKUP_SOURCE_PREVIOUS_SHA
    assert fallback.fallback_key == _key("sha1")
    assert fallback.state.findings[0].title == "bug-A"

    # Workflow validates ancestry, re-keys, records under sha2.
    rekeyed = manager.migrate_to_sha(fallback.state, "sha2")
    manager.record_state(rekeyed)

    # Both records now exist side by side: sha1 (original) and sha2
    # (rekeyed). The next lookup on sha2 hits directly.
    direct = manager.lookup_or_initialize(_key("sha2"))
    assert direct.source == LOOKUP_SOURCE_CURRENT_SHA
    assert direct.state.findings[0].title == "bug-A"
    assert manager.has_state_for(_key("sha1")) is True


def test_force_push_falls_back_to_fresh_review_when_no_neighbour(
    tmp_path: Path,
) -> None:
    """A force-push that breaks ancestry leaves no usable neighbour
    record, so the manager initialises an empty envelope and the
    workflow does a fresh full review (per seed constraint)."""
    manager = ReviewStateManager.from_base_dir(tmp_path)

    # No prior runs exist (e.g. cache evicted by force-push); the
    # manager must initialise rather than raise.
    result = manager.lookup_or_initialize(
        _key("force-pushed-sha"),
        previous_head_sha="abandoned-sha",
    )
    assert result.initialized is True
    assert result.source == LOOKUP_SOURCE_INITIALIZED
    assert result.state.findings == ()


def test_lookup_then_record_then_relookup_round_trip_across_managers(
    tmp_path: Path,
) -> None:
    """Two separate manager instances (simulating workflow runs across
    two GitHub Actions invocations restoring the same RUNNER_TEMP) see
    the same persisted state."""
    runner_temp = tmp_path

    run1 = ReviewStateManager.from_runner_temp(str(runner_temp))
    key = _key()
    run1.record_review_run(
        key,
        findings=(_finding(),),
        summary_comment_id=42,
    )

    # Simulate the next GitHub Action invocation by constructing a
    # fresh manager pointed at the same runner-temp directory (the way
    # actions/cache restores it).
    run2 = ReviewStateManager.from_runner_temp(str(runner_temp))
    result = run2.lookup_or_initialize(key)
    assert result.source == LOOKUP_SOURCE_CURRENT_SHA
    assert result.state.summary_comment_id == 42
