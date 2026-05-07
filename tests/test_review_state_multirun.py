"""Cross-layer multi-run integration tests for prior-review state.

Sub-AC 4.1.4 — exercises the **schema** (``cli/core/review_state.py``),
**storage** (``cli/core/review_state_store.py``), and **manager**
(``cli/core/review_state_manager.py``) layers *together* across the kind
of multi-run scenarios a real GitHub Actions deployment produces.

The per-layer unit suites (``test_review_state.py``,
``test_review_state_store.py``, ``test_review_state_manager.py``) cover
each layer's individual contract. This file fills the gap between them:
end-to-end flows where a PR is reviewed, the action exits, the runner
goes away, ``actions/cache`` restores the state directory into a *fresh*
process, and a second (or third, or fourth) run reads what the prior run
wrote and either continues, re-keys, or falls back to a fresh review.

A "simulated run" here is modelled by tearing down all in-memory state
between operations (no shared store / manager objects across runs) and
re-instantiating against the same on-disk directory the way
``actions/cache`` would. That is what gives the tests their cross-cut
value: they exercise the JSON-on-disk format that survives the process
boundary, not just the in-memory dataclasses.

Scenarios covered
-----------------

1. **Greenfield → first review → re-run on same SHA.**
   The first lookup must initialise; the first record must persist; a
   re-run on the same SHA from a fresh manager must see the persisted
   record under the current-SHA branch (not initialise again).

2. **Fast-forward incremental review continuation.**
   Run 1 records under sha1; run 2 lookups sha2 with previous_head_sha
   hint, falls back to the sha1 neighbour, re-keys via
   ``migrate_to_sha``, accumulates a new finding, and persists under
   sha2. Run 3 with sha3 hints sha2 and inherits both findings.

3. **Force-push that breaks ancestry → fresh full review.**
   Per the seed constraint, when the previous SHA is not on disk the
   manager returns an initialised empty envelope so the workflow can
   do a fresh full review under the new SHA without raising.

4. **Schema bump corrupts an existing cache entry.**
   A future writer wrote a ``v999`` payload that the current reader
   doesn't understand; the storage layer's tolerance default returns
   ``None``, the manager initialises an empty envelope, the workflow
   does a fresh review, and the new payload supersedes the unreadable
   one without hand-deletion.

5. **Multi-PR isolation under one RUNNER_TEMP.**
   Reviews for different PRs in the same repo share the cache directory
   but never collide: each PR's record lives at a distinct cache key
   and the iter-states discovery reports them all.

6. **Model swap on the same PR produces a distinct cache key.**
   A consumer-controlled bump from ``anthropic/claude-opus-4.7`` to a
   different OpenRouter slug must not pick up the old model's findings;
   the seed constraint says "single model per call — no mid-run swap
   or fallback chain" so the SHAs of the two records must not collide.

7. **Carried-forward comment IDs accumulate across runs.**
   Each successive run merges the previous run's published-comment
   IDs into ``carried_forward_comment_ids`` so dismissed-but-still-
   applicable comments are not lost.

8. **Atomic-write contract under simulated mid-write crash recovery.**
   A pre-existing partially-written file in the directory must not
   corrupt a subsequent successful save; the next read must yield the
   newly persisted record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.core.exceptions import ReviewContractError
from cli.core.review_state import (
    OPENROUTER_REVIEW_CACHE_PREFIX,
    REVIEW_STATE_SCHEMA_VERSION,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    PersistedReviewFinding,
    PriorReviewKey,
    PriorReviewState,
)
from cli.core.review_state_manager import (
    LOOKUP_SOURCE_CURRENT_SHA,
    LOOKUP_SOURCE_INITIALIZED,
    LOOKUP_SOURCE_PREVIOUS_SHA,
    ReviewStateManager,
)
from cli.core.review_state_store import (
    DEFAULT_STATE_DIRECTORY_NAME,
    STATE_FILE_SUFFIX,
    ReviewStateStore,
    deserialize_state_from_json,
    load_review_state,
    save_review_state,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_DEFAULT_REPO = "owner/repo"
_DEFAULT_MODEL = "anthropic/claude-opus-4.7"


def _key(
    sha: str,
    *,
    pr_number: int = 42,
    repository: str = _DEFAULT_REPO,
    review_model: str = _DEFAULT_MODEL,
) -> PriorReviewKey:
    return PriorReviewKey(
        repository=repository,
        pr_number=pr_number,
        review_model=review_model,
        reviewed_head_sha=sha,
    )


def _finding(
    *,
    line: int = 10,
    title: str = "Possible NPE",
    severity: str = SEVERITY_HIGH,
    inline_comment_id: int | None = 4242,
    general_comment_id: int | None = None,
) -> PersistedReviewFinding:
    return PersistedReviewFinding(
        path="src/example.py",
        start_line=line,
        end_line=line,
        title=title,
        body="The variable can be null when X happens.",
        severity=severity,
        suggestion="Add a guard clause",
        side="RIGHT",
        priority=1,
        confidence_score=0.85,
        inline_comment_id=inline_comment_id,
        general_comment_id=general_comment_id,
        review_thread_id=(
            f"PRRT_{line:04d}" if inline_comment_id is not None else None
        ),
        finding_fingerprint=f"sha256:{line:08d}",
    )


def _new_manager(runner_temp: Path) -> ReviewStateManager:
    """Build a manager rooted at ``$RUNNER_TEMP/openrouter-review-state``.

    Each call returns a *fresh* manager — that's the whole point. No
    shared in-memory state between runs; the only thing that persists is
    the directory on disk.
    """
    return ReviewStateManager.from_runner_temp(str(runner_temp))


# ---------------------------------------------------------------------------
# Scenario 1: Greenfield → first review → re-run on same SHA
# ---------------------------------------------------------------------------


def test_greenfield_to_recorded_to_rerun_round_trip(tmp_path: Path) -> None:
    """First-time PR: lookup initialises empty, run records, fresh-process
    re-run on the same SHA hits the persisted record."""
    key = _key("sha1")

    # ---- Run 1: first lookup → initialised empty.
    run1 = _new_manager(tmp_path)
    first = run1.lookup_or_initialize(key, generated_at="2026-05-06T12:00:00Z")
    assert first.initialized is True
    assert first.source == LOOKUP_SOURCE_INITIALIZED
    assert first.state.findings == ()
    # Initialisation must not eagerly write — store dir does not yet exist
    # or, if it does, must contain no namespaced files.
    namespaced = (
        list(run1.store.list_paths()) if run1.store.base_dir.exists() else []
    )
    assert namespaced == []

    # ---- Run 1 (continued): workflow records its findings.
    run1.record_review_run(
        key,
        findings=(_finding(line=10, title="bug-A"),),
        carried_forward_comment_ids=(),
        summary_comment_id=12345,
        generated_at="2026-05-06T12:00:00Z",
        notes=("first review",),
    )
    # File now lives under the OpenRouter namespace.
    paths = run1.store.list_paths()
    assert len(paths) == 1
    assert paths[0].name.startswith(OPENROUTER_REVIEW_CACHE_PREFIX)
    assert paths[0].suffix == STATE_FILE_SUFFIX

    # ---- Run 2 (fresh process): same RUNNER_TEMP, same key.
    run2 = _new_manager(tmp_path)
    second = run2.lookup_or_initialize(key)
    assert second.initialized is False
    assert second.source == LOOKUP_SOURCE_CURRENT_SHA
    assert second.state.summary_comment_id == 12345
    assert len(second.state.findings) == 1
    assert second.state.findings[0].title == "bug-A"
    assert second.state.notes == ("first review",)


def test_greenfield_record_reads_back_through_module_level_helpers(
    tmp_path: Path,
) -> None:
    """The convenience ``save_review_state`` / ``load_review_state``
    free functions must integrate with the manager-written file layout
    so callers that bypass the manager still see the same bytes."""
    key = _key("sha1")
    run1 = _new_manager(tmp_path)
    run1.record_review_run(
        key,
        findings=(_finding(),),
        summary_comment_id=99,
    )

    # The free function must hit the same file the manager wrote.
    loaded = load_review_state(run1.store.base_dir, key)
    assert loaded is not None
    assert loaded.summary_comment_id == 99
    assert loaded.findings[0].inline_comment_id == 4242


# ---------------------------------------------------------------------------
# Scenario 2: Fast-forward incremental review continuation
# ---------------------------------------------------------------------------


def test_three_run_incremental_continuation_accumulates_findings(
    tmp_path: Path,
) -> None:
    """Three sequential runs: sha1 → sha2 → sha3, each adding a finding
    and re-keying through the previous-SHA fall-back path."""
    # ---- Run 1: review on sha1.
    run1 = _new_manager(tmp_path)
    run1.record_review_run(
        _key("sha1"),
        findings=(_finding(line=10, title="bug-A"),),
        summary_comment_id=10,
    )

    # ---- Run 2: PR HEAD advances to sha2; lookup with previous_head_sha.
    run2 = _new_manager(tmp_path)
    fallback_at_2 = run2.lookup_or_initialize(
        _key("sha2"), previous_head_sha="sha1"
    )
    assert fallback_at_2.source == LOOKUP_SOURCE_PREVIOUS_SHA
    assert fallback_at_2.fallback_key == _key("sha1")
    assert fallback_at_2.state.findings[0].title == "bug-A"

    rekeyed_2 = run2.migrate_to_sha(fallback_at_2.state, "sha2")
    # Workflow appends a new finding from this run's re-review.
    accumulated_2 = rekeyed_2.with_findings(
        (*rekeyed_2.findings, _finding(line=20, title="bug-B"))
    )
    run2.record_state(accumulated_2)

    # ---- Run 3: sha3 fall-back via sha2; both findings carry forward.
    run3 = _new_manager(tmp_path)
    fallback_at_3 = run3.lookup_or_initialize(
        _key("sha3"), previous_head_sha="sha2"
    )
    assert fallback_at_3.source == LOOKUP_SOURCE_PREVIOUS_SHA
    assert fallback_at_3.fallback_key == _key("sha2")
    titles = tuple(f.title for f in fallback_at_3.state.findings)
    assert titles == ("bug-A", "bug-B")

    rekeyed_3 = run3.migrate_to_sha(fallback_at_3.state, "sha3")
    accumulated_3 = rekeyed_3.with_findings(
        (*rekeyed_3.findings, _finding(line=30, title="bug-C"))
    )
    run3.record_state(accumulated_3)

    # All three SHAs must now be discoverable side by side — incremental
    # review continuation must not evict prior records, since they may
    # still serve as fall-back neighbours for future runs.
    discovery = _new_manager(tmp_path)
    by_sha = {state.reviewed_head_sha: state for state in discovery.store.iter_states()}
    assert set(by_sha.keys()) == {"sha1", "sha2", "sha3"}
    assert tuple(f.title for f in by_sha["sha3"].findings) == (
        "bug-A",
        "bug-B",
        "bug-C",
    )


def test_incremental_continuation_round_trips_severity_and_anchors(
    tmp_path: Path,
) -> None:
    """Mixed severities and anchor metadata must survive re-keying and
    re-persistence across a multi-run sequence."""
    run1 = _new_manager(tmp_path)
    run1.record_review_run(
        _key("sha1"),
        findings=(
            _finding(line=10, title="HIGH", severity=SEVERITY_HIGH),
            _finding(line=20, title="MED", severity=SEVERITY_MEDIUM),
        ),
    )

    run2 = _new_manager(tmp_path)
    fallback = run2.lookup_or_initialize(_key("sha2"), previous_head_sha="sha1")
    rekeyed = run2.migrate_to_sha(fallback.state, "sha2")
    run2.record_state(rekeyed)

    # Re-read from a third manager to prove it's the on-disk bytes
    # surviving the round-trip, not a happy in-memory accident.
    inspector = _new_manager(tmp_path)
    final = inspector.lookup_or_initialize(_key("sha2"))
    assert final.source == LOOKUP_SOURCE_CURRENT_SHA
    sevs = tuple(f.severity for f in final.state.findings)
    assert sevs == (SEVERITY_HIGH, SEVERITY_MEDIUM)
    lines = tuple((f.start_line, f.end_line) for f in final.state.findings)
    assert lines == ((10, 10), (20, 20))


# ---------------------------------------------------------------------------
# Scenario 3: Force-push that breaks ancestry → fresh full review
# ---------------------------------------------------------------------------


def test_force_push_with_no_neighbour_initialises_fresh_envelope(
    tmp_path: Path,
) -> None:
    """Per seed constraint: a force-push that breaks reviewed-SHA
    ancestry falls back to fresh full review — manager initialises
    rather than raising."""
    # Run 1: legitimate review on sha1.
    run1 = _new_manager(tmp_path)
    run1.record_review_run(_key("sha1"), findings=(_finding(),))

    # Run 2: force-push has obliterated history; the previous-SHA hint
    # the workflow was given (e.g. from the summary-comment metadata)
    # points to a SHA the runner *no longer has* on disk because the
    # cache was evicted before this run started.
    run2 = _new_manager(tmp_path)
    # Simulate cache eviction by deleting the run1 record.
    assert run2.discard(_key("sha1")) is True

    result = run2.lookup_or_initialize(
        _key("force-pushed-sha"), previous_head_sha="abandoned-sha"
    )
    assert result.initialized is True
    assert result.source == LOOKUP_SOURCE_INITIALIZED
    assert result.state.findings == ()


def test_force_push_records_fresh_envelope_under_new_sha(tmp_path: Path) -> None:
    """After the force-push fall-back, a fresh review must record under
    the new SHA without touching any leftover neighbour record."""
    # Run 1.
    run1 = _new_manager(tmp_path)
    run1.record_review_run(_key("orphan-sha"), findings=(_finding(line=10),))

    # Run 2: force-push, no neighbour, fresh full review under fp-sha.
    run2 = _new_manager(tmp_path)
    result = run2.lookup_or_initialize(
        _key("fp-sha"), previous_head_sha="phantom-sha"
    )
    assert result.initialized is True
    run2.record_review_run(
        _key("fp-sha"),
        findings=(_finding(line=99, title="post-force-push"),),
    )

    # Both records exist; the orphan is preserved (the manager has no
    # garbage-collection contract).
    inspector = _new_manager(tmp_path)
    assert inspector.has_state_for(_key("fp-sha")) is True
    assert inspector.has_state_for(_key("orphan-sha")) is True
    fresh = inspector.store.load(_key("fp-sha"))
    assert fresh is not None
    assert fresh.findings[0].title == "post-force-push"


# ---------------------------------------------------------------------------
# Scenario 4: Schema bump corrupts existing cache entry
# ---------------------------------------------------------------------------


def test_unknown_schema_version_on_disk_degrades_to_fresh_review(
    tmp_path: Path,
) -> None:
    """A future writer's payload using a schema version this reader
    doesn't understand must not crash the workflow — it must look like
    a cache miss so the manager initialises a fresh envelope."""
    key = _key("sha1")
    base_dir = tmp_path / DEFAULT_STATE_DIRECTORY_NAME
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / f"{key.cache_key()}{STATE_FILE_SUFFIX}"

    # Hand-craft a future-version payload that the current reader can't
    # parse. Real-world example: a downstream consumer pinned to v2.
    future_payload = {
        "schema_version": "v999",
        "key": key.to_dict(),
        "findings": [],
        "carried_forward_comment_ids": [],
        "summary_comment_id": None,
        "generated_at": None,
        "notes": [],
    }
    target.write_text(json.dumps(future_payload), encoding="utf-8")

    # Lookup must initialise (tolerate-corruption default in the store).
    manager = _new_manager(tmp_path)
    result = manager.lookup_or_initialize(key)
    assert result.initialized is True
    assert result.source == LOOKUP_SOURCE_INITIALIZED

    # Strict deserialise refuses the unknown version.
    raw = target.read_text(encoding="utf-8")
    with pytest.raises(ReviewContractError, match="schema version"):
        deserialize_state_from_json(raw)

    # The corrupt file is left in place for operator inspection.
    assert target.exists()

    # Recording a fresh run overwrites the unreadable file in place.
    manager.record_review_run(key, findings=(_finding(),))
    reloaded = manager.store.load(key)
    assert reloaded is not None
    assert reloaded.schema_version == REVIEW_STATE_SCHEMA_VERSION
    assert len(reloaded.findings) == 1


def test_corrupt_json_on_disk_degrades_to_fresh_review(tmp_path: Path) -> None:
    """A truncated or malformed JSON file in the cache directory must
    not abort the lookup; the workflow must initialise and continue."""
    key = _key("sha1")
    base_dir = tmp_path / DEFAULT_STATE_DIRECTORY_NAME
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / f"{key.cache_key()}{STATE_FILE_SUFFIX}"
    target.write_text("{not json", encoding="utf-8")

    manager = _new_manager(tmp_path)
    result = manager.lookup_or_initialize(key)
    assert result.initialized is True
    assert result.state.findings == ()


# ---------------------------------------------------------------------------
# Scenario 5: Multi-PR isolation under one RUNNER_TEMP
# ---------------------------------------------------------------------------


def test_multiple_prs_share_directory_without_collision(tmp_path: Path) -> None:
    """Two different PRs in the same repo using the same model share
    the cache directory; their cache keys differ on ``pr_number`` alone
    so no record can collide."""
    pr_a = _key("sha-A", pr_number=101)
    pr_b = _key("sha-B", pr_number=202)

    # Two independent run-1 invocations write under one RUNNER_TEMP.
    runner = tmp_path
    _new_manager(runner).record_review_run(
        pr_a, findings=(_finding(line=1, title="A-1"),)
    )
    _new_manager(runner).record_review_run(
        pr_b, findings=(_finding(line=2, title="B-1"),)
    )

    # A subsequent run for PR A must see only PR A's findings, never
    # PR B's, even though they share the directory.
    inspector = _new_manager(runner)
    a_state = inspector.lookup_or_initialize(pr_a).state
    b_state = inspector.lookup_or_initialize(pr_b).state
    assert tuple(f.title for f in a_state.findings) == ("A-1",)
    assert tuple(f.title for f in b_state.findings) == ("B-1",)

    # Discovery sees both records under the openrouter-review-v1 prefix.
    pr_numbers = {state.key.pr_number for state in inspector.store.iter_states()}
    assert pr_numbers == {101, 202}


def test_per_pr_cache_keys_are_distinct_files(tmp_path: Path) -> None:
    """The on-disk file path is derived from ``cache_key()``; PRs that
    differ in any keying component must land in different files."""
    pr_a = _key("sha-shared", pr_number=1)
    pr_b = _key("sha-shared", pr_number=2)

    store = ReviewStateStore.from_runner_temp(str(tmp_path))
    assert store.path_for(pr_a) != store.path_for(pr_b)

    save_review_state(store.base_dir, PriorReviewState.empty(pr_a))
    save_review_state(store.base_dir, PriorReviewState.empty(pr_b))
    listed = store.list_paths()
    assert len(listed) == 2
    # Each path must include the corresponding PR number in its name.
    names = sorted(p.name for p in listed)
    assert any("pr-1-" in name for name in names)
    assert any("pr-2-" in name for name in names)


# ---------------------------------------------------------------------------
# Scenario 6: Model swap on the same PR
# ---------------------------------------------------------------------------


def test_review_model_change_does_not_share_cache(tmp_path: Path) -> None:
    """Bumping ``review_model`` produces a new cache key; the old
    model's findings must not show up under the new model. The seed
    constraint forbids fallback chains, so the two records stay
    independent on disk."""
    same_sha = "sha-shared"
    old_model_key = _key(same_sha, review_model="anthropic/claude-opus-4.7")
    new_model_key = _key(same_sha, review_model="anthropic/claude-opus-5.0")

    runner = tmp_path
    _new_manager(runner).record_review_run(
        old_model_key, findings=(_finding(line=10, title="found-by-opus-4.7"),)
    )

    # Switch model — fresh manager, same SHA, new model slug.
    new_run = _new_manager(runner)
    fallback_attempt = new_run.lookup_or_initialize(
        new_model_key,
        # Even with a previous_head_sha hint, the model component of the
        # key is different so the fall-back lookup also misses.
        previous_head_sha=same_sha,
    )
    assert fallback_attempt.initialized is True
    assert fallback_attempt.source == LOOKUP_SOURCE_INITIALIZED
    assert fallback_attempt.state.findings == ()

    new_run.record_review_run(
        new_model_key,
        findings=(_finding(line=10, title="found-by-opus-5.0"),),
    )

    # Both records persist side by side.
    inspector = _new_manager(runner)
    by_model = {
        state.key.review_model: state for state in inspector.store.iter_states()
    }
    assert set(by_model.keys()) == {
        "anthropic/claude-opus-4.7",
        "anthropic/claude-opus-5.0",
    }
    assert by_model["anthropic/claude-opus-4.7"].findings[0].title == (
        "found-by-opus-4.7"
    )
    assert by_model["anthropic/claude-opus-5.0"].findings[0].title == (
        "found-by-opus-5.0"
    )


# ---------------------------------------------------------------------------
# Scenario 7: Carried-forward comment IDs accumulate across runs
# ---------------------------------------------------------------------------


def test_carried_forward_comment_ids_accumulate_through_record_review_run(
    tmp_path: Path,
) -> None:
    """The manager has no merge magic — the workflow is responsible for
    accumulating carried-forward comment IDs run-over-run by reading the
    prior state and explicitly passing the union to the next
    ``record_review_run``. This test pins that contract: state is the
    record of what the workflow said, not of what the manager decided."""
    key_1 = _key("sha1")

    run1 = _new_manager(tmp_path)
    run1.record_review_run(
        key_1,
        findings=(_finding(line=10, title="bug-A", inline_comment_id=11),),
        carried_forward_comment_ids=("legacy-123",),
        summary_comment_id=1,
    )

    # Run 2 reads the prior state and unions the new "still-applicable"
    # comment into the carried-forward set before saving under the new
    # SHA.
    run2 = _new_manager(tmp_path)
    fallback = run2.lookup_or_initialize(_key("sha2"), previous_head_sha="sha1")
    prior_carried = fallback.state.carried_forward_comment_ids
    prior_published = fallback.state.published_inline_comment_ids
    next_carried = tuple(prior_carried) + tuple(str(i) for i in prior_published)

    rekeyed = run2.migrate_to_sha(fallback.state, "sha2")
    run2.record_review_run(
        rekeyed.key,
        findings=rekeyed.findings,  # carry the prior anchor + new ones
        carried_forward_comment_ids=next_carried,
        summary_comment_id=2,
    )

    inspector = _new_manager(tmp_path)
    final = inspector.lookup_or_initialize(_key("sha2"))
    assert final.source == LOOKUP_SOURCE_CURRENT_SHA
    # The legacy ID and the run-1 inline comment ID both ended up in
    # the carried-forward set — manager preserved them verbatim.
    assert "legacy-123" in final.state.carried_forward_comment_ids
    assert "11" in final.state.carried_forward_comment_ids


# ---------------------------------------------------------------------------
# Scenario 8: Atomic-write contract under stray-tempfile recovery
# ---------------------------------------------------------------------------


def test_save_after_stray_partial_write_yields_clean_record(tmp_path: Path) -> None:
    """A leftover same-directory tempfile from a prior crashed write
    must not corrupt subsequent saves; the next manager-driven save
    must produce a well-formed record that round-trips cleanly."""
    key = _key("sha1")
    manager = _new_manager(tmp_path)
    base_dir = manager.store.ensure_directory()

    # Drop a stray ``*.tmp.<rand>`` file that mimics what an interrupted
    # ``write_text_atomic`` would leave behind on a power-cut runner.
    stray = base_dir / f"{key.cache_key()}{STATE_FILE_SUFFIX}.tmp.broken"
    stray.write_text("{this is half-written", encoding="utf-8")

    # The successful save must not be deflected by the stray.
    manager.record_review_run(
        key,
        findings=(_finding(line=10, title="post-recovery"),),
        summary_comment_id=10,
    )

    # The store's namespaced ``.json`` listing skips the stray.
    paths = manager.store.list_paths()
    assert len(paths) == 1
    assert paths[0].name.endswith(".json")
    assert ".tmp" not in paths[0].name

    # The record reads back lossless across a fresh manager.
    fresh = _new_manager(tmp_path)
    final = fresh.lookup_or_initialize(key)
    assert final.source == LOOKUP_SOURCE_CURRENT_SHA
    assert final.state.findings[0].title == "post-recovery"


# ---------------------------------------------------------------------------
# Scenario 9: End-to-end JSON-on-disk schema validation
# ---------------------------------------------------------------------------


def test_persisted_file_validates_against_schema_layer(tmp_path: Path) -> None:
    """The JSON the storage layer wrote must validate against the schema
    layer's ``from_mapping`` / ``from_json`` parsers without losing any
    field — including the ``priority``/``confidence_score`` pair that
    crosses the model-output ↔ persistence boundary and the GitHub
    comment IDs the posting layer sets."""
    key = _key("sha1")
    manager = _new_manager(tmp_path)
    findings = (
        _finding(line=10, title="HIGH", severity=SEVERITY_HIGH,
                 inline_comment_id=11, general_comment_id=None),
        _finding(line=20, title="GENERAL-FALLBACK", severity=SEVERITY_MEDIUM,
                 inline_comment_id=None, general_comment_id=22),
    )
    manager.record_review_run(
        key,
        findings=findings,
        carried_forward_comment_ids=("c1", "c2"),
        summary_comment_id=99,
        generated_at="2026-05-06T12:00:00Z",
        notes=("first review", "incremental"),
    )

    on_disk = manager.store.path_for(key).read_text(encoding="utf-8")
    payload = json.loads(on_disk)

    # ---- Schema-level invariants on the raw dict.
    assert payload["schema_version"] == REVIEW_STATE_SCHEMA_VERSION
    assert payload["key"]["repository"] == _DEFAULT_REPO
    assert payload["key"]["pr_number"] == 42
    assert payload["key"]["review_model"] == _DEFAULT_MODEL
    assert payload["key"]["reviewed_head_sha"] == "sha1"
    assert payload["summary_comment_id"] == 99
    assert payload["carried_forward_comment_ids"] == ["c1", "c2"]
    assert payload["notes"] == ["first review", "incremental"]
    assert len(payload["findings"]) == 2

    # ---- Round-trip the bytes back through the schema layer's
    # JSON parser. The result must match the in-memory state precisely.
    parsed = PriorReviewState.from_json(on_disk)
    assert parsed.schema_version == REVIEW_STATE_SCHEMA_VERSION
    assert parsed.key == key
    assert tuple(f.title for f in parsed.findings) == ("HIGH", "GENERAL-FALLBACK")
    assert parsed.findings[0].inline_comment_id == 11
    assert parsed.findings[0].general_comment_id is None
    assert parsed.findings[1].inline_comment_id is None
    assert parsed.findings[1].general_comment_id == 22
    assert parsed.carried_forward_comment_ids == ("c1", "c2")
    assert parsed.summary_comment_id == 99
    assert parsed.generated_at == "2026-05-06T12:00:00Z"
    assert parsed.notes == ("first review", "incremental")


def test_partial_payload_with_missing_optional_fields_round_trips(
    tmp_path: Path,
) -> None:
    """Workflows that only fill the required fields must still round-trip
    through schema → store → schema without losing the explicit
    ``None`` / empty-tuple defaults."""
    key = _key("sha1")
    state = PriorReviewState.empty(key, generated_at="2026-05-06T12:00:00Z")
    manager = _new_manager(tmp_path)
    manager.record_state(state)

    raw = manager.store.path_for(key).read_text(encoding="utf-8")
    parsed = PriorReviewState.from_json(raw)
    assert parsed == state
    assert parsed.findings == ()
    assert parsed.carried_forward_comment_ids == ()
    assert parsed.summary_comment_id is None
    assert parsed.notes == ()


# ---------------------------------------------------------------------------
# Scenario 10: Discovery across mixed-namespace cache directories
# ---------------------------------------------------------------------------


def test_iter_states_reports_only_openrouter_prefixed_files(
    tmp_path: Path,
) -> None:
    """A real RUNNER_TEMP may carry stray files from other actions or
    legacy runs; discovery must see only the OpenRouter-namespaced
    state files so the workflow's "list known prior reviews" output
    is not polluted with junk."""
    runner = tmp_path
    manager = _new_manager(runner)
    manager.record_review_run(_key("sha1"))

    # Drop a stray legacy codex-review file in the same directory.
    legacy = manager.store.base_dir / "codex-review-v1-old-record.json"
    legacy.write_text(json.dumps({"schema_version": "v1"}), encoding="utf-8")

    # Drop an unrelated scratch file.
    (manager.store.base_dir / "scratch.txt").write_text("ignore me", encoding="utf-8")

    inspector = _new_manager(runner)
    discovered = list(inspector.store.iter_states())
    assert len(discovered) == 1
    assert discovered[0].reviewed_head_sha == "sha1"


# ---------------------------------------------------------------------------
# Scenario 11: Re-record on the same key is atomic and replaces in place
# ---------------------------------------------------------------------------


def test_re_record_same_key_replaces_in_place(tmp_path: Path) -> None:
    """The "one writer, one reader" GitHub Actions cadence implies a
    workflow may re-record under the same key during a single run (e.g.
    after a self-review correction). Each save must atomically replace
    the prior bytes — no orphaned tempfiles, no stale fields."""
    key = _key("sha1")
    manager = _new_manager(tmp_path)

    manager.record_review_run(
        key,
        findings=(_finding(line=10, title="initial"),),
        summary_comment_id=1,
    )
    manager.record_review_run(
        key,
        findings=(_finding(line=99, title="corrected"),),
        summary_comment_id=2,
        notes=("self-review",),
    )

    final = manager.store.load(key)
    assert final is not None
    assert len(final.findings) == 1
    assert final.findings[0].title == "corrected"
    assert final.summary_comment_id == 2
    assert final.notes == ("self-review",)

    # Only the final record sits in the directory — atomic replace, no
    # stray tempfile residue.
    files = sorted(manager.store.base_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"
