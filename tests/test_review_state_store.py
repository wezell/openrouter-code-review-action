"""Tests for the prior-review state storage backend.

Sub-AC 4.1.2 — verifies the file-backed storage layer
(``cli/core/review_state_store.py``) reads, writes, and persists
:class:`PriorReviewState` records across simulated GitHub Action runs:

* ``ReviewStateStore.save`` writes atomically and returns the
  deterministic cache-key path.
* ``ReviewStateStore.load`` round-trips to the same in-memory record.
* Missing files default to ``None`` so the workflow can fall back to a
  fresh review on cache miss.
* Corrupt JSON / unknown schema versions degrade to ``None`` by default
  but raise when callers explicitly opt out of tolerance.
* ``list_paths`` only reports files matching the
  ``openrouter-review-v1-`` namespace, so a stray legacy
  ``dotbot-review-v1-`` file is ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.core.exceptions import ReviewContractError, ReviewResumeError
from cli.core.review_state import (
    OPENROUTER_REVIEW_CACHE_PREFIX,
    REVIEW_STATE_SCHEMA_VERSION,
    PersistedReviewFinding,
    PriorReviewKey,
    PriorReviewState,
)
from cli.core.review_state_store import (
    DEFAULT_STATE_DIRECTORY_NAME,
    STATE_FILE_SUFFIX,
    ReviewStateNotFound,
    ReviewStateStore,
    WrittenStateRecord,
    deserialize_state_from_json,
    load_review_state,
    save_review_state,
    serialize_state_to_json,
)

# ---------------------------------------------------------------------------
# Test fixtures / sample factories
# ---------------------------------------------------------------------------


def _sample_key(reviewed_head_sha: str = "deadbeef") -> PriorReviewKey:
    return PriorReviewKey(
        repository="owner/repo",
        pr_number=42,
        review_model="anthropic/claude-opus-4.7",
        reviewed_head_sha=reviewed_head_sha,
    )


def _sample_finding() -> PersistedReviewFinding:
    return PersistedReviewFinding(
        path="src/example.py",
        start_line=10,
        end_line=12,
        title="Possible NPE",
        body="The variable can be null when X happens.",
        severity="high",
        suggestion="Add a guard clause",
        side="RIGHT",
        priority=1,
        confidence_score=0.8,
        inline_comment_id=4242,
        general_comment_id=None,
        review_thread_id="PRRT_kwDOA",
        finding_fingerprint="sha256:abc",
    )


def _sample_state(
    *,
    key: PriorReviewKey | None = None,
    findings: tuple[PersistedReviewFinding, ...] | None = None,
) -> PriorReviewState:
    return PriorReviewState(
        schema_version=REVIEW_STATE_SCHEMA_VERSION,
        key=key or _sample_key(),
        findings=findings if findings is not None else (_sample_finding(),),
        carried_forward_comment_ids=("comment-A",),
        summary_comment_id=99,
        generated_at="2026-05-06T12:00:00Z",
    )


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_default_directory_name_uses_openrouter_namespace() -> None:
    # The directory carved under RUNNER_TEMP must match the new
    # OpenRouter namespace, not the legacy "dotbot-review-state" path.
    assert DEFAULT_STATE_DIRECTORY_NAME == "openrouter-review-state"


def test_state_file_suffix_is_dotted_json() -> None:
    # File suffix must include the dot — path concatenation expects it.
    assert STATE_FILE_SUFFIX == ".json"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_runner_temp_uses_default_directory_name(tmp_path: Path) -> None:
    store = ReviewStateStore.from_runner_temp(str(tmp_path))
    assert store.base_dir == tmp_path.resolve() / DEFAULT_STATE_DIRECTORY_NAME


def test_from_runner_temp_falls_back_to_cwd_when_runner_temp_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    store = ReviewStateStore.from_runner_temp(None)
    assert store.base_dir.parent == tmp_path.resolve()
    assert store.base_dir.name == DEFAULT_STATE_DIRECTORY_NAME


def test_from_runner_temp_falls_back_to_cwd_when_runner_temp_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    store = ReviewStateStore.from_runner_temp("")
    assert store.base_dir.parent == tmp_path.resolve()


def test_from_runner_temp_accepts_custom_directory_name(tmp_path: Path) -> None:
    store = ReviewStateStore.from_runner_temp(str(tmp_path), directory_name="my-cache")
    assert store.base_dir == tmp_path.resolve() / "my-cache"


# ---------------------------------------------------------------------------
# Path arithmetic
# ---------------------------------------------------------------------------


def test_path_for_uses_cache_key_with_dot_json_suffix(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    key = _sample_key()
    expected = tmp_path / f"{key.cache_key()}{STATE_FILE_SUFFIX}"
    assert store.path_for(key) == expected
    # The cache key must also live inside the OpenRouter namespace.
    assert expected.name.startswith(OPENROUTER_REVIEW_CACHE_PREFIX)


def test_path_for_changes_when_sha_changes(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    old = store.path_for(_sample_key("oldsha"))
    new = store.path_for(_sample_key("newsha"))
    assert old != new


def test_ensure_directory_is_idempotent(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path / "nested" / "state")
    first = store.ensure_directory()
    second = store.ensure_directory()
    assert first == second
    assert store.base_dir.is_dir()


# ---------------------------------------------------------------------------
# Save → load round-trip
# ---------------------------------------------------------------------------


def test_save_round_trips_through_load(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    state = _sample_state()
    record = store.save(state)

    assert isinstance(record, WrittenStateRecord)
    assert record.path == store.path_for(state.key)
    assert record.cache_key == state.cache_key
    assert record.path.is_file()

    loaded = store.load(state.key)
    assert loaded == state


def test_save_creates_base_directory_on_demand(tmp_path: Path) -> None:
    target = tmp_path / "deeply" / "nested" / "state"
    store = ReviewStateStore(target)
    assert not target.exists()

    store.save(_sample_state())

    assert target.is_dir()


def test_save_overwrites_existing_record(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    initial = _sample_state()
    store.save(initial)
    updated = initial.with_findings(
        (
            _sample_finding(),
            PersistedReviewFinding(
                path="src/example.py",
                start_line=20,
                end_line=20,
                title="Second finding",
                body="Updated body",
            ),
        )
    )
    store.save(updated)

    loaded = store.load(updated.key)
    assert loaded == updated
    assert loaded is not None
    assert len(loaded.findings) == 2


def test_save_writes_pretty_printed_json_for_operator_inspection(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    state = _sample_state()
    record = store.save(state)

    raw = record.path.read_text(encoding="utf-8")
    # The file must be valid JSON.
    payload = json.loads(raw)
    assert payload["schema_version"] == REVIEW_STATE_SCHEMA_VERSION
    # And pretty-printed (default indent=2 → contains newlines).
    assert "\n" in raw


def test_save_uses_atomic_write_so_no_temp_file_remains(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    store.save(_sample_state())

    files = sorted(tmp_path.iterdir())
    # Only the final ``.json`` file should remain — write_text_atomic must
    # have cleaned up its same-directory temp file.
    assert len(files) == 1
    assert files[0].suffix == STATE_FILE_SUFFIX


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------


def test_exists_returns_false_when_directory_missing(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path / "does-not-exist")
    assert store.exists(_sample_key()) is False


def test_exists_returns_false_for_unsaved_key(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    store.save(_sample_state(key=_sample_key("oldsha")))
    assert store.exists(_sample_key("newsha")) is False


def test_exists_returns_true_for_saved_key(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    store.save(_sample_state())
    assert store.exists(_sample_key()) is True


# ---------------------------------------------------------------------------
# Load: missing files
# ---------------------------------------------------------------------------


def test_load_returns_none_when_record_missing(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    assert store.load(_sample_key()) is None


def test_load_returns_none_when_directory_missing(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path / "no-such-dir")
    assert store.load(_sample_key()) is None


def test_load_required_raises_when_missing(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    with pytest.raises(ReviewStateNotFound):
        store.load(_sample_key(), required=True)


def test_load_required_raises_when_file_is_empty(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    target = store.path_for(_sample_key())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")

    with pytest.raises(ReviewStateNotFound):
        store.load(_sample_key(), required=True)


def test_review_state_not_found_inherits_from_resume_error(tmp_path: Path) -> None:
    # Workflow code that catches ReviewResumeError must transparently
    # absorb the missing-state case so it can fall back to a fresh review.
    store = ReviewStateStore(tmp_path)
    with pytest.raises(ReviewResumeError):
        store.load(_sample_key(), required=True)


def test_load_returns_none_for_zero_byte_file(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    target = store.path_for(_sample_key())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")

    # A truncated cache restore should look identical to a missing file
    # so the workflow falls back to a fresh review.
    assert store.load(_sample_key()) is None


def test_load_returns_none_for_whitespace_only_file(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    target = store.path_for(_sample_key())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("   \n  \n", encoding="utf-8")

    assert store.load(_sample_key()) is None


# ---------------------------------------------------------------------------
# Load: corruption tolerance
# ---------------------------------------------------------------------------


def test_load_returns_none_for_invalid_json_by_default(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    target = store.path_for(_sample_key())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json", encoding="utf-8")

    assert store.load(_sample_key()) is None


def test_load_returns_none_for_unknown_schema_version_by_default(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    target = store.path_for(_sample_key())
    target.parent.mkdir(parents=True, exist_ok=True)
    bad_payload = _sample_state().to_dict()
    bad_payload["schema_version"] = "v0"
    target.write_text(json.dumps(bad_payload), encoding="utf-8")

    assert store.load(_sample_key()) is None


def test_load_raises_when_tolerate_corruption_is_false(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    target = store.path_for(_sample_key())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json", encoding="utf-8")

    with pytest.raises(ReviewContractError):
        store.load(_sample_key(), tolerate_corruption=False)


def test_load_corrupt_file_is_left_in_place_for_inspection(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    target = store.path_for(_sample_key())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json", encoding="utf-8")

    store.load(_sample_key())
    assert target.exists(), "Corrupt cache files must be preserved for operator review"


# ---------------------------------------------------------------------------
# load_path
# ---------------------------------------------------------------------------


def test_load_path_round_trips_a_saved_state(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    record = store.save(_sample_state())
    loaded = store.load_path(record.path)
    assert loaded == _sample_state()


def test_load_path_returns_none_for_missing_file(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    assert store.load_path(tmp_path / "ghost.json") is None


def test_load_path_returns_none_for_corrupt_json_by_default(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    target = tmp_path / "broken.json"
    target.write_text("{nope", encoding="utf-8")
    assert store.load_path(target) is None


def test_load_path_raises_when_tolerate_corruption_false(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    target = tmp_path / "broken.json"
    target.write_text("{nope", encoding="utf-8")
    with pytest.raises(ReviewContractError):
        store.load_path(target, tolerate_corruption=False)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_removes_existing_record(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    store.save(_sample_state())
    assert store.delete(_sample_key()) is True
    assert store.exists(_sample_key()) is False


def test_delete_returns_false_for_missing_record(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    assert store.delete(_sample_key()) is False


def test_delete_is_idempotent(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    store.save(_sample_state())
    assert store.delete(_sample_key()) is True
    assert store.delete(_sample_key()) is False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_paths_returns_empty_list_when_directory_missing(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path / "missing")
    assert store.list_paths() == []


def test_list_paths_returns_only_namespaced_files(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    store.ensure_directory()

    # One legitimate state file
    record = store.save(_sample_state())
    # A stray legacy file from the dotbot-review action
    (tmp_path / "dotbot-review-v1-old.json").write_text("{}", encoding="utf-8")
    # A non-JSON file someone dropped in
    (tmp_path / "openrouter-review-v1-readme.txt").write_text("note", encoding="utf-8")
    # A subdirectory that should be ignored
    (tmp_path / "openrouter-review-v1-subdir").mkdir()

    paths = store.list_paths()
    assert paths == [record.path]


def test_list_paths_returns_sorted_for_determinism(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    store.save(_sample_state(key=_sample_key("aaaa")))
    store.save(_sample_state(key=_sample_key("zzzz")))
    store.save(_sample_state(key=_sample_key("mmmm")))

    paths = store.list_paths()
    assert paths == sorted(paths)


def test_iter_states_yields_every_saved_state(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    a = _sample_state(key=_sample_key("aaaa"))
    b = _sample_state(key=_sample_key("bbbb"))
    store.save(a)
    store.save(b)

    states = list(store.iter_states())
    assert {s.reviewed_head_sha for s in states} == {"aaaa", "bbbb"}


def test_iter_states_skips_corrupt_entries_by_default(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    store.save(_sample_state(key=_sample_key("good")))
    bad_path = tmp_path / f"{_sample_key('bad').cache_key()}{STATE_FILE_SUFFIX}"
    bad_path.write_text("{not json", encoding="utf-8")

    states = list(store.iter_states())
    assert len(states) == 1
    assert states[0].reviewed_head_sha == "good"


def test_iter_states_propagates_corruption_when_tolerance_disabled(
    tmp_path: Path,
) -> None:
    store = ReviewStateStore(tmp_path)
    store.save(_sample_state(key=_sample_key("good")))
    bad_path = tmp_path / f"{_sample_key('bad').cache_key()}{STATE_FILE_SUFFIX}"
    bad_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ReviewContractError):
        list(store.iter_states(tolerate_corruption=False))


# ---------------------------------------------------------------------------
# Cross-run persistence simulation
# ---------------------------------------------------------------------------


def test_state_survives_simulated_action_run_round_trip(tmp_path: Path) -> None:
    """End-to-end: write in 'run 1', re-instantiate the store as if the
    runner restarted, read in 'run 2'."""
    runner_temp = tmp_path

    # Run 1: workflow writes its review state at end of execution.
    run1 = ReviewStateStore.from_runner_temp(str(runner_temp))
    state = _sample_state()
    record = run1.save(state)
    assert record.path.is_file()

    # actions/cache implicitly tarballs runner_temp/openrouter-review-state
    # and restores it before the next run; we model this by simply
    # constructing a new store object pointing at the same directory.
    run2 = ReviewStateStore.from_runner_temp(str(runner_temp))
    loaded = run2.load(state.key)

    assert loaded == state


def test_state_for_new_sha_is_distinct_record(tmp_path: Path) -> None:
    store = ReviewStateStore(tmp_path)
    old = _sample_state(key=_sample_key("oldsha"))
    new_state = _sample_state(key=_sample_key("newsha"))
    store.save(old)
    store.save(new_state)

    # Both records persist side by side — a new SHA does NOT evict the
    # prior one. This is what enables fall-back lookup under a secondary
    # restore key when the primary cache misses.
    assert store.load(_sample_key("oldsha")) == old
    assert store.load(_sample_key("newsha")) == new_state


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def test_save_review_state_function_round_trips_through_load_review_state(
    tmp_path: Path,
) -> None:
    state = _sample_state()
    record = save_review_state(tmp_path, state)
    assert record.path == ReviewStateStore(tmp_path).path_for(state.key)

    loaded = load_review_state(tmp_path, state.key)
    assert loaded == state


def test_load_review_state_required_raises_for_missing(tmp_path: Path) -> None:
    with pytest.raises(ReviewStateNotFound):
        load_review_state(tmp_path, _sample_key(), required=True)


def test_serialize_and_deserialize_round_trip() -> None:
    state = _sample_state()
    raw = serialize_state_to_json(state)
    parsed = deserialize_state_from_json(raw)
    assert parsed == state


def test_deserialize_returns_none_for_corrupt_input_when_tolerated() -> None:
    assert deserialize_state_from_json("{not json", tolerate_corruption=True) is None


def test_deserialize_raises_for_corrupt_input_by_default() -> None:
    with pytest.raises(ReviewContractError):
        deserialize_state_from_json("{not json")


def test_serialize_can_emit_compact_json() -> None:
    state = _sample_state()
    raw = serialize_state_to_json(state, indent=None)
    # No newlines means single-line compact output.
    assert "\n" not in raw
    # And it must still round-trip.
    assert deserialize_state_from_json(raw) == state
