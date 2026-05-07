"""Tests for the prior-review state schema and PR identifier keying convention.

Sub-AC 4.1.1 — verifies the dedicated types/schema module
(``cli/core/review_state.py``) for last-reviewed SHA, findings, and comment
IDs behaves as a stable contract:

* ``PriorReviewKey.cache_key()`` derives a sanitized, namespaced cache key
  with the ``openrouter-review-v1-`` prefix.
* ``PersistedReviewFinding`` validates anchor / content / comment-ID fields
  and round-trips through JSON without loss.
* ``PriorReviewState.to_json`` / ``from_json`` round-trips lossless and
  refuses unknown ``schema_version`` values.
* The summary metadata renderer/parser pair round-trips and tolerates
  malformed payloads.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from cli.core.exceptions import ReviewContractError
from cli.core.review_state import (
    KNOWN_SEVERITIES,
    OPENROUTER_REVIEW_CACHE_PREFIX,
    OPENROUTER_REVIEW_SUMMARY_MARKER,
    REVIEW_STATE_SCHEMA_VERSION,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    PersistedReviewFinding,
    PriorReviewKey,
    PriorReviewState,
    parse_reviewed_head_sha,
    render_review_summary_metadata,
)

# ---------------------------------------------------------------------------
# Module constants / canonical taxonomy
# ---------------------------------------------------------------------------


def test_module_constants_use_openrouter_namespace() -> None:
    # The new namespace must be distinct from the legacy "codex-review"
    # prefix; the cache restore-key boundary depends on it.
    assert OPENROUTER_REVIEW_CACHE_PREFIX == "openrouter-review-v1"
    assert OPENROUTER_REVIEW_SUMMARY_MARKER.startswith("OpenRouter")
    assert REVIEW_STATE_SCHEMA_VERSION == "v1"


def test_known_severities_cover_canonical_tiers() -> None:
    assert KNOWN_SEVERITIES == frozenset(
        {SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL}
    )


# ---------------------------------------------------------------------------
# Summary metadata round-trip
# ---------------------------------------------------------------------------


def test_summary_metadata_round_trips() -> None:
    metadata = render_review_summary_metadata("deadbeef")

    body = f"{OPENROUTER_REVIEW_SUMMARY_MARKER}\n{metadata}"
    assert parse_reviewed_head_sha(body) == "deadbeef"


def test_summary_metadata_uses_openrouter_marker_in_html_comment() -> None:
    rendered = render_review_summary_metadata("cafebabe")
    # Must use a marker distinct from the codex one so old and new
    # summary comments cannot be confused for each other.
    assert "openrouter-review-meta" in rendered
    assert "codex-review-meta" not in rendered


def test_parse_reviewed_head_sha_returns_none_on_malformed_payloads() -> None:
    # Malformed JSON inside the marker
    assert (
        parse_reviewed_head_sha("<!-- openrouter-review-meta {not json} -->") is None
    )
    # Wrong field type
    assert (
        parse_reviewed_head_sha(
            '<!-- openrouter-review-meta {"reviewed_head_sha": 123} -->'
        )
        is None
    )
    # No metadata block at all
    assert parse_reviewed_head_sha("just a comment body") is None


def test_parse_reviewed_head_sha_strips_whitespace() -> None:
    body = '<!-- openrouter-review-meta {"reviewed_head_sha": "  abcd  "} -->'
    assert parse_reviewed_head_sha(body) == "abcd"


def test_parse_reviewed_head_sha_returns_none_for_empty_string() -> None:
    body = '<!-- openrouter-review-meta {"reviewed_head_sha": "   "} -->'
    assert parse_reviewed_head_sha(body) is None


# ---------------------------------------------------------------------------
# PriorReviewKey
# ---------------------------------------------------------------------------


def test_prior_review_key_cache_key_uses_namespaced_prefix() -> None:
    key = PriorReviewKey(
        repository="owner/repo",
        pr_number=17,
        review_model="anthropic/claude-opus-4.7",
        reviewed_head_sha="deadbeef",
    )
    assert key.cache_key() == (
        "openrouter-review-v1-owner-repo-pr-17-"
        "anthropic-claude-opus-4.7-deadbeef"
    )


def test_prior_review_key_sanitizes_components() -> None:
    key = PriorReviewKey(
        repository="org/Repo Name!",
        pr_number=3,
        review_model="provider/model 1.0+beta",
        reviewed_head_sha="ABCD/EFGH",
    )
    cache_key = key.cache_key()
    # Every character must be in the cache-safe set.
    for char in cache_key:
        assert char.isalnum() or char in "._-"


def test_prior_review_key_with_sha_returns_new_key() -> None:
    original = PriorReviewKey(
        repository="owner/repo",
        pr_number=1,
        review_model="openai/gpt-5.4",
        reviewed_head_sha="oldsha",
    )
    updated = original.with_sha("newsha")

    assert updated.reviewed_head_sha == "newsha"
    assert original.reviewed_head_sha == "oldsha"  # original unchanged
    assert updated.cache_key() != original.cache_key()


def test_prior_review_key_round_trips_through_dict() -> None:
    key = PriorReviewKey(
        repository="owner/repo",
        pr_number=7,
        review_model="anthropic/claude-opus-4.7",
        reviewed_head_sha="abc123",
    )
    assert PriorReviewKey.from_mapping(key.to_dict()) == key


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repository": "no-slash"},
        {"repository": ""},
        {"pr_number": 0},
        {"pr_number": -1},
        {"review_model": ""},
        {"reviewed_head_sha": ""},
    ],
)
def test_prior_review_key_validation_rejects_bad_inputs(
    kwargs: dict[str, Any],
) -> None:
    base: dict[str, Any] = {
        "repository": "owner/repo",
        "pr_number": 1,
        "review_model": "anthropic/claude-opus-4.7",
        "reviewed_head_sha": "deadbeef",
    }
    base.update(kwargs)
    with pytest.raises(ReviewContractError):
        PriorReviewKey(**base)


def test_prior_review_key_from_mapping_rejects_missing_field() -> None:
    with pytest.raises(ReviewContractError, match="repository"):
        PriorReviewKey.from_mapping(
            {
                "pr_number": 1,
                "review_model": "x/y",
                "reviewed_head_sha": "abc",
            }
        )


# ---------------------------------------------------------------------------
# PersistedReviewFinding
# ---------------------------------------------------------------------------


def _sample_finding(**overrides: Any) -> PersistedReviewFinding:
    base: dict[str, Any] = {
        "path": "src/example.py",
        "start_line": 10,
        "end_line": 12,
        "title": "Possible NPE",
        "body": "The variable can be null when X happens.",
        "severity": SEVERITY_HIGH,
        "suggestion": "Add a guard clause",
        "side": "RIGHT",
        "priority": 1,
        "confidence_score": 0.8,
        "inline_comment_id": 4242,
        "general_comment_id": None,
        "review_thread_id": "PRRT_kwDOA",
        "finding_fingerprint": "sha256:abc",
    }
    base.update(overrides)
    return PersistedReviewFinding(**base)


def test_persisted_finding_round_trips_through_dict() -> None:
    finding = _sample_finding()
    restored = PersistedReviewFinding.from_mapping(finding.to_dict())
    assert restored == finding


def test_persisted_finding_round_trips_with_unpublished_state() -> None:
    finding = _sample_finding(
        inline_comment_id=None,
        general_comment_id=None,
        review_thread_id=None,
        finding_fingerprint=None,
    )
    assert finding.is_published is False
    assert finding.comment_id is None
    restored = PersistedReviewFinding.from_mapping(finding.to_dict())
    assert restored == finding
    assert restored.is_published is False


def test_persisted_finding_comment_id_prefers_inline_then_general() -> None:
    inline = _sample_finding(inline_comment_id=11, general_comment_id=22)
    assert inline.comment_id == 11
    general_only = _sample_finding(inline_comment_id=None, general_comment_id=22)
    assert general_only.comment_id == 22
    none = _sample_finding(inline_comment_id=None, general_comment_id=None)
    assert none.comment_id is None


def test_persisted_finding_is_published_when_review_thread_is_set() -> None:
    finding = _sample_finding(
        inline_comment_id=None,
        general_comment_id=None,
        review_thread_id="thread-1",
    )
    assert finding.is_published is True


@pytest.mark.parametrize(
    "overrides,expected_match",
    [
        ({"path": ""}, "path"),
        ({"start_line": 0}, "start_line"),
        ({"start_line": -1}, "start_line"),
        ({"start_line": 10, "end_line": 5}, "end_line"),
        ({"severity": ""}, "severity"),
        ({"side": "TOP"}, "side"),
    ],
)
def test_persisted_finding_validation_rejects_bad_inputs(
    overrides: dict[str, Any],
    expected_match: str,
) -> None:
    with pytest.raises(ReviewContractError, match=expected_match):
        _sample_finding(**overrides)


def test_persisted_finding_from_mapping_defaults_end_line_to_start_line() -> None:
    payload = {
        "path": "x.py",
        "start_line": 5,
        # end_line omitted on purpose
        "title": "t",
        "body": "b",
    }
    finding = PersistedReviewFinding.from_mapping(payload)
    assert finding.start_line == 5
    assert finding.end_line == 5


def test_persisted_finding_from_mapping_rejects_non_int_priority() -> None:
    payload = _sample_finding().to_dict()
    payload["priority"] = "high"
    with pytest.raises(ReviewContractError, match="priority"):
        PersistedReviewFinding.from_mapping(payload)


def test_persisted_finding_from_mapping_rejects_bool_priority() -> None:
    # Python's ``bool`` is a subclass of ``int``; the validator must reject it.
    payload = _sample_finding().to_dict()
    payload["priority"] = True
    with pytest.raises(ReviewContractError, match="priority"):
        PersistedReviewFinding.from_mapping(payload)


def test_persisted_finding_from_mapping_rejects_non_int_inline_id() -> None:
    payload = _sample_finding().to_dict()
    payload["inline_comment_id"] = "1234"
    with pytest.raises(ReviewContractError, match="inline_comment_id"):
        PersistedReviewFinding.from_mapping(payload)


# ---------------------------------------------------------------------------
# PriorReviewState
# ---------------------------------------------------------------------------


def _sample_key() -> PriorReviewKey:
    return PriorReviewKey(
        repository="owner/repo",
        pr_number=42,
        review_model="anthropic/claude-opus-4.7",
        reviewed_head_sha="abcd1234",
    )


def _sample_state() -> PriorReviewState:
    return PriorReviewState(
        schema_version=REVIEW_STATE_SCHEMA_VERSION,
        key=_sample_key(),
        findings=(_sample_finding(), _sample_finding(start_line=20, end_line=20)),
        carried_forward_comment_ids=("comment-A", "comment-B"),
        summary_comment_id=99,
        generated_at="2026-05-06T12:00:00Z",
        notes=("first note",),
    )


def test_prior_review_state_empty_constructs_zero_findings_envelope() -> None:
    state = PriorReviewState.empty(_sample_key(), generated_at="now")
    assert state.schema_version == REVIEW_STATE_SCHEMA_VERSION
    assert state.findings == ()
    assert state.carried_forward_comment_ids == ()
    assert state.summary_comment_id is None
    assert state.generated_at == "now"
    assert state.reviewed_head_sha == "abcd1234"
    assert state.cache_key.startswith(OPENROUTER_REVIEW_CACHE_PREFIX)


def test_prior_review_state_round_trips_through_json() -> None:
    state = _sample_state()
    raw = state.to_json()
    parsed = PriorReviewState.from_json(raw)
    assert parsed == state


def test_prior_review_state_from_json_accepts_unindented_payload() -> None:
    state = _sample_state()
    compact = state.to_json(indent=None)
    assert "\n" not in compact  # one-line JSON
    assert PriorReviewState.from_json(compact) == state


def test_prior_review_state_with_findings_replaces_only_findings() -> None:
    state = _sample_state()
    new_findings = (_sample_finding(start_line=99, end_line=99),)
    updated = state.with_findings(new_findings)

    assert updated.findings == new_findings
    assert updated.key == state.key
    assert updated.carried_forward_comment_ids == state.carried_forward_comment_ids
    assert updated.summary_comment_id == state.summary_comment_id
    # Original state must not be mutated.
    assert len(state.findings) == 2


def test_prior_review_state_helpers_collect_published_ids() -> None:
    findings = (
        _sample_finding(inline_comment_id=11, general_comment_id=None,
                        review_thread_id="t1"),
        _sample_finding(inline_comment_id=None, general_comment_id=22,
                        review_thread_id=None, start_line=20, end_line=20),
        _sample_finding(inline_comment_id=33, general_comment_id=None,
                        review_thread_id="t3", start_line=30, end_line=30),
    )
    state = _sample_state().with_findings(findings)

    assert state.published_inline_comment_ids == (11, 33)
    assert state.published_general_comment_ids == (22,)
    assert state.published_review_thread_ids == ("t1", "t3")


def test_prior_review_state_from_mapping_rejects_unknown_schema_version() -> None:
    raw = _sample_state().to_dict()
    raw["schema_version"] = "v0"
    with pytest.raises(ReviewContractError, match="schema version"):
        PriorReviewState.from_mapping(raw)


def test_prior_review_state_constructor_rejects_unknown_schema_version() -> None:
    with pytest.raises(ReviewContractError, match="schema version"):
        PriorReviewState(
            schema_version="v999",
            key=_sample_key(),
        )


def test_prior_review_state_from_mapping_rejects_non_array_findings() -> None:
    raw = _sample_state().to_dict()
    raw["findings"] = "not-an-array"
    with pytest.raises(ReviewContractError, match="findings"):
        PriorReviewState.from_mapping(raw)


def test_prior_review_state_from_mapping_rejects_non_string_carried_forward() -> None:
    raw = _sample_state().to_dict()
    raw["carried_forward_comment_ids"] = ["valid", 123]
    with pytest.raises(ReviewContractError, match="carried_forward_comment_ids"):
        PriorReviewState.from_mapping(raw)


def test_prior_review_state_from_mapping_rejects_non_int_summary_id() -> None:
    raw = _sample_state().to_dict()
    raw["summary_comment_id"] = "not-an-int"
    with pytest.raises(ReviewContractError, match="summary_comment_id"):
        PriorReviewState.from_mapping(raw)


def test_prior_review_state_from_json_rejects_invalid_json() -> None:
    with pytest.raises(ReviewContractError, match="JSON decode"):
        PriorReviewState.from_json("{not json")


def test_prior_review_state_from_json_rejects_non_object_root() -> None:
    with pytest.raises(ReviewContractError, match="root must be an object"):
        PriorReviewState.from_json("[]")


def test_prior_review_state_to_json_is_valid_json_serializable() -> None:
    state = _sample_state()
    raw = state.to_json()
    # The output must round-trip through json.loads/dumps without error.
    payload = json.loads(raw)
    assert payload["schema_version"] == REVIEW_STATE_SCHEMA_VERSION
    assert payload["key"]["repository"] == "owner/repo"
    assert payload["key"]["pr_number"] == 42
    # Findings preserve order and structure.
    assert len(payload["findings"]) == 2
    assert payload["findings"][0]["path"] == "src/example.py"
    assert payload["summary_comment_id"] == 99


def test_prior_review_state_cache_key_matches_key_cache_key() -> None:
    state = _sample_state()
    assert state.cache_key == state.key.cache_key()


def test_prior_review_state_supports_default_factory_for_notes() -> None:
    # ``notes`` defaults to an empty tuple via field(default_factory=tuple).
    state = PriorReviewState(
        schema_version=REVIEW_STATE_SCHEMA_VERSION,
        key=_sample_key(),
    )
    assert state.notes == ()


def test_prior_review_state_from_mapping_with_minimal_payload() -> None:
    minimal = {
        "schema_version": REVIEW_STATE_SCHEMA_VERSION,
        "key": _sample_key().to_dict(),
    }
    state = PriorReviewState.from_mapping(minimal)
    assert state.findings == ()
    assert state.carried_forward_comment_ids == ()
    assert state.summary_comment_id is None
    assert state.generated_at is None
    assert state.notes == ()
