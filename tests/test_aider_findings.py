"""Tests for review-finding → aider-edit-plan translation (Sub-AC 7.2.2)."""

from __future__ import annotations

import pytest

from cli.core.models import ReviewFinding, ReviewFindingLocation
from cli.workflows.aider_findings import (
    AiderEditPlan,
    FindingSuggestion,
    build_aider_edit_plan,
    build_aider_message,
    collect_finding_files,
    extract_finding_suggestions,
    extract_suggestion_blocks,
    render_finding_block,
)


def _finding(
    *,
    title: str = "Use enumerate",
    body: str = "Switch to enumerate.",
    path: str = "src/foo.py",
    start: int = 10,
    end: int | None = None,
    priority: int | None = 2,
    confidence: float | None = 0.9,
) -> ReviewFinding:
    return ReviewFinding(
        title=title,
        body=body,
        confidence_score=confidence,
        priority=priority,
        code_location=ReviewFindingLocation(
            absolute_file_path=path,
            start_line=start,
            end_line=end if end is not None else start,
        ),
    )


# ---------- extract_suggestion_blocks ----------------------------------------


def test_extract_suggestion_blocks_returns_inner_payload() -> None:
    body = "Use the indexed form.\n```suggestion\nfor i, x in enumerate(items):\n    pass\n```\n"
    blocks = extract_suggestion_blocks(body)
    assert len(blocks) == 1
    assert "enumerate(items)" in blocks[0]
    assert not blocks[0].startswith("```")
    assert not blocks[0].endswith("```")


def test_extract_suggestion_blocks_handles_multiple() -> None:
    body = (
        "First option:\n"
        "```suggestion\n"
        "alpha = 1\n"
        "```\n"
        "Second option:\n"
        "```suggestion\n"
        "beta = 2\n"
        "```\n"
    )
    blocks = extract_suggestion_blocks(body)
    assert len(blocks) == 2
    assert "alpha = 1" in blocks[0]
    assert "beta = 2" in blocks[1]


def test_extract_suggestion_blocks_drops_empty_blocks() -> None:
    body = "```suggestion\n\n```\n"
    assert extract_suggestion_blocks(body) == []


def test_extract_suggestion_blocks_ignores_inline_mention() -> None:
    body = "We could use ```suggestion``` here, but it's optional."
    assert extract_suggestion_blocks(body) == []


def test_extract_suggestion_blocks_empty_input() -> None:
    assert extract_suggestion_blocks("") == []


# ---------- extract_finding_suggestions --------------------------------------


def test_extract_finding_suggestions_carries_location() -> None:
    finding = _finding(
        body="Replace this:\n```suggestion\nreturn True\n```\n",
        path="pkg/module.py",
        start=42,
        end=44,
    )
    suggestions = extract_finding_suggestions(finding)
    assert len(suggestions) == 1
    s = suggestions[0]
    assert isinstance(s, FindingSuggestion)
    assert s.file_path == "pkg/module.py"
    assert s.start_line == 42
    assert s.end_line == 44
    assert s.replacement == "return True"
    assert s.is_multi_line


def test_extract_finding_suggestions_returns_empty_when_no_block() -> None:
    finding = _finding(body="No suggestion fence here.")
    assert extract_finding_suggestions(finding) == []


# ---------- collect_finding_files --------------------------------------------


def test_collect_finding_files_dedupes_preserving_order() -> None:
    findings = [
        _finding(path="b.py", start=1),
        _finding(path="a.py", start=2),
        _finding(path="b.py", start=3),
        _finding(path="./c.py", start=4),
    ]
    assert collect_finding_files(findings) == ["b.py", "a.py", "c.py"]


def test_collect_finding_files_skips_blank_paths() -> None:
    findings = [
        _finding(path="   ", start=1),
        _finding(path="real.py", start=2),
    ]
    assert collect_finding_files(findings) == ["real.py"]


def test_collect_finding_files_strips_leading_slash() -> None:
    findings = [_finding(path="/abs/foo.py")]
    assert collect_finding_files(findings) == ["abs/foo.py"]


# ---------- render_finding_block ---------------------------------------------


def test_render_finding_block_emits_xml_attributes() -> None:
    finding = _finding(
        title="Bug",
        body="Use enumerate.",
        path="src/foo.py",
        start=10,
        end=12,
        priority=1,
    )
    block = render_finding_block(finding, index=1)
    assert 'index="1"' in block
    assert 'path="src/foo.py"' in block
    assert 'start_line="10"' in block
    assert 'end_line="12"' in block
    assert 'priority="1"' in block
    assert "<title>Bug</title>" in block
    assert "Use enumerate." in block


def test_render_finding_block_truncates_long_body() -> None:
    finding = _finding(body="x" * 5000)
    block = render_finding_block(finding, index=1, body_limit=100)
    assert "… (truncated)" in block
    # Body section is small.
    assert len(block) < 600


def test_render_finding_block_falls_back_to_start_when_end_lower() -> None:
    finding = _finding(start=10, end=5)
    # Manually override end via dataclass replace not possible (frozen);
    # instead check construction with mismatched range fixed by renderer.
    finding = ReviewFinding(
        title="t",
        body="b",
        confidence_score=None,
        priority=None,
        code_location=ReviewFindingLocation(
            absolute_file_path="x.py",
            start_line=10,
            end_line=5,
        ),
    )
    block = render_finding_block(finding, index=1)
    assert 'start_line="10"' in block
    assert 'end_line="10"' in block


# ---------- build_aider_message ----------------------------------------------


def test_build_aider_message_has_overview_and_findings_sections() -> None:
    msg = build_aider_message([_finding()])
    assert "<act_overview>" in msg
    assert "non-interactive act mode" in msg
    assert "<findings>" in msg
    assert "</findings>" in msg
    assert "do not run git commit" in msg.lower()


def test_build_aider_message_includes_extra_instructions() -> None:
    msg = build_aider_message([_finding()], extra_instructions="Prefer comprehensions.")
    assert "<extra_instructions>" in msg
    assert "Prefer comprehensions." in msg


def test_build_aider_message_renders_each_finding_with_index() -> None:
    findings = [
        _finding(title="One", path="a.py", start=1),
        _finding(title="Two", path="b.py", start=2),
    ]
    msg = build_aider_message(findings)
    assert 'index="1"' in msg
    assert 'index="2"' in msg
    assert "<title>One</title>" in msg
    assert "<title>Two</title>" in msg


def test_build_aider_message_handles_empty_findings() -> None:
    msg = build_aider_message([])
    assert "<findings>" in msg
    assert "no findings supplied" in msg


def test_build_aider_message_truncates_when_over_budget() -> None:
    findings = [_finding(title=f"f{i}", body="x" * 500, start=i) for i in range(20)]
    msg = build_aider_message(findings, char_limit=2000)
    assert "Truncated after" in msg
    assert len(msg) <= 2500  # some slack for the truncation note


# ---------- build_aider_edit_plan --------------------------------------------


def test_build_aider_edit_plan_returns_message_files_and_suggestions() -> None:
    findings = [
        _finding(
            title="Use enumerate",
            body="```suggestion\nfor i, x in enumerate(items):\n    pass\n```\n",
            path="src/foo.py",
            start=10,
            end=11,
        ),
        _finding(
            title="Rename var",
            body="No fence here.",
            path="src/bar.py",
            start=3,
        ),
    ]
    plan = build_aider_edit_plan(findings, extra_instructions="Be terse.")
    assert isinstance(plan, AiderEditPlan)
    assert plan.files == ("src/foo.py", "src/bar.py")
    assert "Be terse." in plan.message
    assert "Use enumerate" in plan.message
    assert "Rename var" in plan.message
    assert len(plan.suggestions) == 1
    assert plan.suggestions[0].file_path == "src/foo.py"
    assert plan.suggestions[0].start_line == 10
    assert "enumerate(items)" in plan.suggestions[0].replacement
    assert not plan.is_empty


def test_build_aider_edit_plan_is_empty_with_no_findings() -> None:
    plan = build_aider_edit_plan([])
    assert plan.files == ()
    assert plan.suggestions == ()
    assert plan.is_empty


def test_build_aider_edit_plan_dedupes_files_when_findings_repeat_path() -> None:
    findings = [
        _finding(path="x.py", start=1),
        _finding(path="x.py", start=5),
        _finding(path="y.py", start=1),
    ]
    plan = build_aider_edit_plan(findings)
    assert plan.files == ("x.py", "y.py")


def test_build_aider_edit_plan_collects_multiple_suggestions_per_finding() -> None:
    finding = _finding(
        body=("Option A:\n```suggestion\na = 1\n```\nOption B:\n```suggestion\nb = 2\n```\n"),
    )
    plan = build_aider_edit_plan([finding])
    assert len(plan.suggestions) == 2
    assert plan.suggestions[0].replacement == "a = 1"
    assert plan.suggestions[1].replacement == "b = 2"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("./pkg/file.py", "pkg/file.py"),
        ("pkg//file.py", "pkg/file.py"),
        ("/abs/path.py", "abs/path.py"),
    ],
)
def test_collect_finding_files_normalises_paths(raw: str, expected: str) -> None:
    plan = build_aider_edit_plan([_finding(path=raw)])
    assert plan.files == (expected,)
