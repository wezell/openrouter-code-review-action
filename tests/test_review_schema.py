"""Tests for the OpenRouter-facing review-output JSON schema and helpers.

These cover Sub-AC 3.1: define and enforce a JSON schema for inline review
comments returned by OpenRouter model calls (structured output /
``response_format``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from cli.core.exceptions import ReviewContractError
from cli.core.models import (
    CARRIED_FORWARD_COMMENT_SCHEMA,
    OPENROUTER_REVIEW_SCHEMA_NAME,
    REVIEW_FINDING_LOCATION_SCHEMA,
    REVIEW_FINDING_SCHEMA,
    REVIEW_OUTPUT_SCHEMA,
    ReviewFinding,
    ReviewRunResult,
    build_openrouter_response_format,
    openrouter_review_response_format,
    validate_review_finding,
    validate_review_payload,
)

# ---------------------------------------------------------------------------
# Strict-mode invariants
#
# OpenRouter forwards OpenAI-compatible Structured Output requests. Strict
# mode requires every object schema to (1) declare every property in
# `required`, (2) set `additionalProperties: false`, and (3) avoid omitted
# fields. Walk the schema tree and assert these invariants.
# ---------------------------------------------------------------------------


def _walk_object_schemas(schema: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Yield every object-type subschema reachable from ``schema``."""
    objects: list[Mapping[str, Any]] = []

    def _visit(node: Any) -> None:
        if isinstance(node, Mapping):
            node_type = node.get("type")
            if node_type == "object":
                objects.append(node)
            for value in node.values():
                _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(schema)
    return objects


@pytest.mark.parametrize(
    "schema",
    [
        REVIEW_OUTPUT_SCHEMA,
        REVIEW_FINDING_SCHEMA,
        REVIEW_FINDING_LOCATION_SCHEMA,
        CARRIED_FORWARD_COMMENT_SCHEMA,
    ],
)
def test_schema_is_strict_object(schema: Mapping[str, Any]) -> None:
    """Every object schema declares all properties as required and forbids extras."""
    for node in _walk_object_schemas(schema):
        properties = node.get("properties")
        required = node.get("required")
        assert isinstance(properties, Mapping), f"schema missing properties: {node}"
        assert isinstance(required, list), f"schema missing required list: {node}"

        # additionalProperties must be exactly False (not absent, not True).
        assert node.get("additionalProperties") is False, (
            f"object schema must set additionalProperties=False: {node}"
        )

        # Every property must appear in `required` for OpenRouter strict mode.
        property_names = set(properties.keys())
        required_names = set(required)
        missing = property_names - required_names
        assert not missing, f"properties not listed as required: {missing}"
        extra = required_names - property_names
        assert not extra, f"required entries with no property: {extra}"


def test_review_output_schema_top_level_shape() -> None:
    assert REVIEW_OUTPUT_SCHEMA["type"] == "object"
    properties = REVIEW_OUTPUT_SCHEMA["properties"]
    assert isinstance(properties, dict)
    assert set(properties.keys()) == {
        "findings",
        "carried_forward",
        "overall_correctness",
        "overall_explanation",
        "overall_confidence_score",
    }


def test_findings_items_use_inline_finding_schema() -> None:
    findings = REVIEW_OUTPUT_SCHEMA["properties"]["findings"]
    assert isinstance(findings, dict)
    assert findings["type"] == "array"
    # Items should be the same shape as REVIEW_FINDING_SCHEMA (single inline
    # comment), guaranteeing parity between the array and the standalone
    # inline-comment schema.
    items = findings["items"]
    assert isinstance(items, dict)
    assert items["type"] == "object"
    assert set(items["properties"].keys()) == set(REVIEW_FINDING_SCHEMA["properties"].keys())


def test_nullable_fields_use_array_form() -> None:
    """OpenRouter strict mode requires ``["type", "null"]`` for nullable fields."""
    confidence_field = REVIEW_FINDING_SCHEMA["properties"]["confidence_score"]
    assert confidence_field == {"type": ["number", "null"]}
    priority_field = REVIEW_FINDING_SCHEMA["properties"]["priority"]
    assert priority_field == {"type": ["integer", "null"]}
    overall_confidence_field = REVIEW_OUTPUT_SCHEMA["properties"]["overall_confidence_score"]
    assert overall_confidence_field == {"type": ["number", "null"]}


# ---------------------------------------------------------------------------
# response_format helper
# ---------------------------------------------------------------------------


def test_build_openrouter_response_format_produces_strict_payload() -> None:
    payload = build_openrouter_response_format(schema=REVIEW_FINDING_SCHEMA, name="finding")
    assert payload["type"] == "json_schema"
    inner = payload["json_schema"]
    assert inner["name"] == "finding"
    assert inner["strict"] is True
    assert inner["schema"] == REVIEW_FINDING_SCHEMA
    # Must be JSON-serialisable since this is sent over the wire.
    json.dumps(payload)


def test_build_openrouter_response_format_allows_disabling_strict() -> None:
    payload = build_openrouter_response_format(
        schema=REVIEW_FINDING_SCHEMA, name="finding", strict=False
    )
    assert payload["json_schema"]["strict"] is False


def test_build_openrouter_response_format_rejects_blank_name() -> None:
    with pytest.raises(ReviewContractError, match="non-empty"):
        build_openrouter_response_format(schema=REVIEW_FINDING_SCHEMA, name="")
    with pytest.raises(ReviewContractError, match="non-empty"):
        build_openrouter_response_format(schema=REVIEW_FINDING_SCHEMA, name="   ")


def test_build_openrouter_response_format_rejects_non_mapping_schema() -> None:
    with pytest.raises(ReviewContractError, match="must be a mapping"):
        build_openrouter_response_format(schema="not-a-schema", name="finding")  # type: ignore[arg-type]


def test_openrouter_review_response_format_uses_review_output_schema() -> None:
    payload = openrouter_review_response_format()
    assert payload["type"] == "json_schema"
    inner = payload["json_schema"]
    assert inner["name"] == OPENROUTER_REVIEW_SCHEMA_NAME
    assert inner["strict"] is True
    assert inner["schema"] == REVIEW_OUTPUT_SCHEMA


# ---------------------------------------------------------------------------
# Validation entry points
# ---------------------------------------------------------------------------


def _conformant_payload() -> dict[str, Any]:
    return {
        "overall_correctness": "ok",
        "overall_explanation": "fine",
        "overall_confidence_score": 0.5,
        "carried_forward": [
            {"comment_id": "c1", "current_evidence": "still applies"},
        ],
        "findings": [
            {
                "title": "Possible NPE",
                "body": "The variable can be null when X happens.",
                "confidence_score": 0.8,
                "priority": 1,
                "code_location": {
                    "absolute_file_path": "/repo/src/foo.py",
                    "line_range": {"start": 10, "end": 12},
                },
            }
        ],
    }


def test_validate_review_payload_accepts_conformant_output() -> None:
    result = validate_review_payload(_conformant_payload())
    assert isinstance(result, ReviewRunResult)
    assert len(result.findings) == 1
    assert result.findings[0].code_location.start_line == 10
    assert result.carried_forward_comment_ids == ["c1"]


def test_validate_review_payload_rejects_non_object() -> None:
    with pytest.raises(ReviewContractError, match="JSON object"):
        validate_review_payload("not an object")  # type: ignore[arg-type]


def test_validate_review_payload_rejects_missing_top_level_field() -> None:
    payload = _conformant_payload()
    payload.pop("overall_correctness")
    with pytest.raises(ReviewContractError, match="missing required fields"):
        validate_review_payload(payload)


def test_validate_review_payload_rejects_wrong_finding_field_type() -> None:
    payload = _conformant_payload()
    # title must be a string per the schema
    payload["findings"][0]["title"] = 42
    with pytest.raises(ReviewContractError, match="title"):
        validate_review_payload(payload)


def test_validate_review_payload_rejects_non_array_findings() -> None:
    payload = _conformant_payload()
    payload["findings"] = {"not": "an array"}
    with pytest.raises(ReviewContractError, match="findings"):
        validate_review_payload(payload)


def test_validate_review_finding_accepts_conformant_finding() -> None:
    finding = validate_review_finding(_conformant_payload()["findings"][0])
    assert isinstance(finding, ReviewFinding)
    assert finding.title == "Possible NPE"
    assert finding.priority == 1
    assert finding.confidence_score == pytest.approx(0.8)
    assert finding.code_location.absolute_file_path == "/repo/src/foo.py"


def test_validate_review_finding_rejects_non_object() -> None:
    with pytest.raises(ReviewContractError, match="JSON object"):
        validate_review_finding("oops")  # type: ignore[arg-type]


def test_validate_review_finding_rejects_missing_code_location() -> None:
    finding = _conformant_payload()["findings"][0]
    finding.pop("code_location")
    with pytest.raises(ReviewContractError, match="missing required fields"):
        validate_review_finding(finding)


def test_validate_review_finding_rejects_invalid_code_location_shape() -> None:
    finding = _conformant_payload()["findings"][0]
    # Empty path violates the contract — strict schema would reject it too.
    finding["code_location"] = {
        "absolute_file_path": "",
        "line_range": {"start": 1, "end": 2},
    }
    with pytest.raises(ReviewContractError, match="code_location"):
        validate_review_finding(finding)


def test_validate_review_payload_accepts_null_optional_numbers() -> None:
    """``confidence_score`` and ``priority`` are nullable by schema."""
    payload = _conformant_payload()
    payload["overall_confidence_score"] = None
    payload["findings"][0]["confidence_score"] = None
    payload["findings"][0]["priority"] = None
    result = validate_review_payload(payload)
    assert result.overall_confidence_score is None
    assert result.findings[0].confidence_score is None
    assert result.findings[0].priority is None
