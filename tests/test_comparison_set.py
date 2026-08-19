"""Tests for the bake-off comparison reference set builder (Sub-AC 1.1.3).

These tests pin the *contract* the comparison set offers to the
downstream labeling (Sub-AC 1.4) and scoring (Sub-AC 1.5) sub-ACs:

* Per-PR JSON shape — normalized fields, predictable keys, no surprise
  values for downstream consumers to special-case.
* Run-state semantics — when only the dotbot side is captured the
  record reports ``awaiting_openrouter``; when neither is the rollup
  is ``pending``; both captured flips to ``ready_for_labeling``.
* Severity normalization — both the explicit ``priority`` field and
  the ``[Pn]`` title prefix should resolve to the same severity tag.
* Match-candidate scan — same path + line proximity surfaces a
  candidate; line-distance is reported and overlapping ranges are 0.
* On-disk reproducibility — ``--check`` flags drift on a hand-edit
  and zeroes back out after a rebuild.

Tests run against a *temporary* sample registry and runs/ tree
isolated under ``tmp_path`` so they never touch the committed
artifacts under ``eval/runs`` or ``evaluation/bakeoff/runs``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bakeoff import build_comparison_set as builder  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


_HEAD_SHA = "abc123def456" * 3 + "abcd"  # 40-char SHA-shaped string


def _eval_sample_doc() -> dict[str, Any]:
    return {
        "prs": [
            {
                "id": "fixture-pr-1",
                "repo": "owner/repo",
                "pr": 1,
                "head_sha": _HEAD_SHA,
                "languages": ["python"],
                "slots": ["real_bug_present"],
                "loc_changed": 42,
                "web_search_mode_for_run": "disabled",
                "reasoning_effort_for_run": "medium",
            },
            {
                "id": "fixture-pr-2",
                "repo": "owner/repo",
                "pr": 2,
                "head_sha": "f" * 40,
                "languages": ["typescript_or_javascript"],
                "slots": ["clean_diff"],
                "loc_changed": 10,
                "web_search_mode_for_run": "cached",
                "reasoning_effort_for_run": "high",
            },
        ]
    }


def _bakeoff_sample_doc() -> dict[str, Any]:
    return {
        "repo": "owner/repo",
        "prs": [
            {
                "number": 1,
                "repo": "owner/repo",
                "title": "fix: thing",
                "merge_commit": _HEAD_SHA,
                "base_ref": "main",
                "head_ref": "fix/thing",
                "additions": 30,
                "deletions": 12,
                "changed_files": 3,
                "surface": "backend",
                "change_type": "bug_fix",
                "security_signal": False,
                "why": "fixture: real bug present",
            },
            {
                "number": 2,
                "repo": "owner/repo",
                "title": "refactor: clean up",
                "merge_commit": "f" * 40,
                "base_ref": "main",
                "head_ref": "refactor/clean",
                "additions": 4,
                "deletions": 6,
                "changed_files": 2,
                "surface": "frontend",
                "change_type": "refactor",
                "security_signal": False,
                "why": "fixture: clean diff",
            },
        ],
    }


def _codex_artifact(
    *,
    head_sha: str,
    capture_status: str,
    findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    review_run_result: dict[str, Any] | None
    if capture_status == "captured":
        review_run_result = {
            "overall_correctness": "patch is correct",
            "overall_explanation": "Looks fine.",
            "overall_confidence_score": 0.85,
            "findings": findings or [],
            "carried_forward": [],
        }
    else:
        review_run_result = None
    return {
        "pr_id": "ignored-by-builder",
        "run": "codex",
        "head_sha": head_sha,
        "model": "gpt-5.4",
        "reasoning_effort": "medium",
        "web_search_mode": "disabled",
        "prior_review_state": None,
        "capture": {
            "status": capture_status,
            "runner_version": "1",
            "captured_at": "2026-05-07T00:00:00Z" if capture_status == "captured" else None,
            "command": "fixture",
            "notes": "fixture",
        },
        "review_run_result": review_run_result,
        "posted": {
            "summary_id": None,
            "inline_ids": [],
            "posting_outcome": {
                "batch_submitted": 0,
                "per_comment_fallback": 0,
                "skipped_after_422": 0,
            },
        },
    }


def _openrouter_artifact(
    *,
    head_sha: str,
    status: str,
    findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    review_run_result: dict[str, Any] | None
    if status == "completed":
        review_run_result = {
            "overall_correctness": "patch is correct",
            "overall_explanation": "All good.",
            "overall_confidence_score": 0.9,
            "findings": findings or [],
            "carried_forward": [],
        }
    else:
        review_run_result = None
    return {
        "pr_id": "ignored-by-builder",
        "run": "openrouter",
        "status": status,
        "head_sha": head_sha,
        "repo": "owner/repo",
        "pr_number": 1,
        "model": "anthropic/claude-opus-4.7",
        "reasoning_effort": "medium",
        "web_search_mode": "disabled",
        "prior_review_state": None,
        "review_run_result": review_run_result,
        "posted": None,
        "labels": [],
        "solo_labeled": None,
        "notes": "fixture",
        "telemetry": {},
        "captured_at": "2026-05-07T00:01:00Z" if status == "completed" else None,
        "runner": "fixture",
        "runner_version": "1.0.0",
    }


def _finding(
    *,
    path: str,
    line_start: int,
    line_end: int,
    title: str,
    body: str = "fixture body",
    priority: int | None = None,
    confidence_score: float | None = 0.8,
) -> dict[str, Any]:
    return {
        "title": title,
        "body": body,
        "confidence_score": confidence_score,
        "priority": priority,
        "code_location": {
            "absolute_file_path": path,
            "line_range": {"start": line_start, "end": line_end},
        },
    }


@pytest.fixture
def isolated_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Spin up a temp repo layout that mirrors the real one's relevant paths."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    bakeoff_dir = tmp_path / "evaluation" / "bakeoff"
    bakeoff_dir.mkdir(parents=True)
    (eval_dir / "sample-prs.yaml").write_text(yaml.safe_dump(_eval_sample_doc()))
    (bakeoff_dir / "sample-prs.yml").write_text(yaml.safe_dump(_bakeoff_sample_doc()))

    runs_codex = eval_dir / "runs"
    runs_codex.mkdir()
    runs_or = bakeoff_dir / "runs"
    runs_or.mkdir()
    output_dir = bakeoff_dir / "comparison"

    monkeypatch.setattr(builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(builder, "EVAL_SAMPLE_PATH", eval_dir / "sample-prs.yaml")
    monkeypatch.setattr(builder, "BAKEOFF_SAMPLE_PATH", bakeoff_dir / "sample-prs.yml")
    monkeypatch.setattr(builder, "CODEX_RUNS_DIR", runs_codex)
    monkeypatch.setattr(builder, "OPENROUTER_RUNS_DIR", runs_or)
    monkeypatch.setattr(builder, "OUTPUT_DIR", output_dir)
    return tmp_path


def _write_codex(tmp: Path, eval_id: str, payload: dict[str, Any]) -> None:
    pr_dir = tmp / "eval" / "runs" / eval_id
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "codex.json").write_text(json.dumps(payload))


def _write_openrouter(tmp: Path, pr_id: str, payload: dict[str, Any]) -> None:
    pr_dir = tmp / "evaluation" / "bakeoff" / "runs" / pr_id
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "openrouter.json").write_text(json.dumps(payload))


# --------------------------------------------------------------------------- #
# Sample registry join
# --------------------------------------------------------------------------- #


def test_load_sample_registry_joins_eval_and_bakeoff_metadata(isolated_layout: Path) -> None:
    entries = builder.load_sample_registry()
    assert [e.pr_id for e in entries] == ["pr-1", "pr-2"]
    pr1 = entries[0]
    assert pr1.eval_id == "fixture-pr-1"
    assert pr1.repo == "owner/repo"
    assert pr1.head_sha == _HEAD_SHA
    assert pr1.title == "fix: thing"  # from bakeoff registry
    assert pr1.surface == "backend"
    assert pr1.change_type == "bug_fix"
    assert pr1.languages == ["python"]
    assert pr1.slots == ["real_bug_present"]
    assert pr1.loc_changed == 42
    assert pr1.changed_files == 3


def test_artifact_paths_are_constructed_from_per_run_id_conventions(
    isolated_layout: Path,
) -> None:
    [pr1, _pr2] = builder.load_sample_registry()
    assert pr1.codex_artifact_path() == (
        isolated_layout / "eval" / "runs" / "fixture-pr-1" / "codex.json"
    )
    assert pr1.openrouter_artifact_path() == (
        isolated_layout / "evaluation" / "bakeoff" / "runs" / "pr-1" / "openrouter.json"
    )


# --------------------------------------------------------------------------- #
# Status rollups
# --------------------------------------------------------------------------- #


def test_pending_when_neither_run_has_real_findings(isolated_layout: Path) -> None:
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert rec["labeling"]["status"] == "pending"
    assert rec["runs"]["codex"]["capture_status"] == "missing"
    assert rec["runs"]["openrouter"]["capture_status"] == "missing"
    assert rec["findings"] == []
    assert rec["match_candidates"] == []


def test_awaiting_openrouter_when_only_codex_captured(isolated_layout: Path) -> None:
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(head_sha=_HEAD_SHA, capture_status="captured"),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert rec["labeling"]["status"] == "awaiting_openrouter"
    assert rec["runs"]["codex"]["capture_status"] == "captured"


def test_awaiting_codex_when_only_openrouter_completed(isolated_layout: Path) -> None:
    _write_openrouter(
        isolated_layout,
        "pr-1",
        _openrouter_artifact(head_sha=_HEAD_SHA, status="completed"),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert rec["labeling"]["status"] == "awaiting_codex"


def test_ready_for_labeling_when_both_runs_captured(isolated_layout: Path) -> None:
    finding = _finding(
        path="src/x.py",
        line_start=10,
        line_end=10,
        title="🔴 [P1] src/x.py:10 unguarded null deref",
        priority=1,
    )
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(head_sha=_HEAD_SHA, capture_status="captured", findings=[finding]),
    )
    _write_openrouter(
        isolated_layout,
        "pr-1",
        _openrouter_artifact(head_sha=_HEAD_SHA, status="completed", findings=[finding]),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert rec["labeling"]["status"] == "ready_for_labeling"
    assert len(rec["findings"]) == 2
    assert {f["run"] for f in rec["findings"]} == {"codex", "openrouter"}


# --------------------------------------------------------------------------- #
# Finding normalization
# --------------------------------------------------------------------------- #


def test_finding_fields_normalized_from_priority_field(isolated_layout: Path) -> None:
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(
            head_sha=_HEAD_SHA,
            capture_status="captured",
            findings=[
                _finding(
                    path="src/api.py",
                    line_start=42,
                    line_end=44,
                    title="🟡 [P2] src/api.py:42 missing nullness check",
                    body="…",
                    priority=2,
                    confidence_score=0.7,
                )
            ],
        ),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    f = rec["findings"][0]
    assert f["finding_id"] == "codex.f0001"
    assert f["run"] == "codex"
    assert f["raw_index"] == 0
    assert f["path"] == "src/api.py"
    assert f["line_start"] == 42
    assert f["line_end"] == 44
    assert f["severity"] == "P2"
    assert f["priority"] == 2
    assert f["confidence_score"] == 0.7
    assert f["label"] is None  # populated later by labeling pass
    assert f["matched_finding_id"] is None
    assert f["match_to_other_run"] is None


def test_severity_falls_back_to_title_tag_when_priority_missing(
    isolated_layout: Path,
) -> None:
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(
            head_sha=_HEAD_SHA,
            capture_status="captured",
            findings=[
                _finding(
                    path="src/a.py",
                    line_start=1,
                    line_end=1,
                    title="⚪ [P3] src/a.py:1 nit on naming",
                    priority=None,
                )
            ],
        ),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert rec["findings"][0]["severity"] == "P3"
    assert rec["findings"][0]["priority"] is None


def test_severity_unknown_when_neither_field_provides_one(isolated_layout: Path) -> None:
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(
            head_sha=_HEAD_SHA,
            capture_status="captured",
            findings=[
                _finding(
                    path="src/a.py",
                    line_start=1,
                    line_end=1,
                    title="some untagged finding",
                    priority=None,
                )
            ],
        ),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert rec["findings"][0]["severity"] is None


# --------------------------------------------------------------------------- #
# Match candidates
# --------------------------------------------------------------------------- #


def test_match_candidates_overlap_ranges(isolated_layout: Path) -> None:
    f_codex = _finding(
        path="src/x.py",
        line_start=10,
        line_end=20,
        title="🔴 [P1] x:10 issue",
        priority=1,
    )
    f_or = _finding(
        path="src/x.py",
        line_start=15,
        line_end=25,
        title="🔴 [P1] x:15 same issue",
        priority=1,
    )
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(head_sha=_HEAD_SHA, capture_status="captured", findings=[f_codex]),
    )
    _write_openrouter(
        isolated_layout,
        "pr-1",
        _openrouter_artifact(head_sha=_HEAD_SHA, status="completed", findings=[f_or]),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert len(rec["match_candidates"]) == 1
    cand = rec["match_candidates"][0]
    assert cand["codex_finding_id"] == "codex.f0001"
    assert cand["openrouter_finding_id"] == "openrouter.f0001"
    assert cand["line_distance"] == 0  # ranges overlap
    assert cand["severity_match"] is True


def test_match_candidates_within_tolerance_but_non_overlapping(
    isolated_layout: Path,
) -> None:
    f_codex = _finding(
        path="src/x.py",
        line_start=10,
        line_end=10,
        title="[P1] codex finding",
        priority=1,
    )
    f_or = _finding(
        path="src/x.py",
        line_start=12,
        line_end=12,
        title="[P2] openrouter finding",
        priority=2,
    )
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(head_sha=_HEAD_SHA, capture_status="captured", findings=[f_codex]),
    )
    _write_openrouter(
        isolated_layout,
        "pr-1",
        _openrouter_artifact(head_sha=_HEAD_SHA, status="completed", findings=[f_or]),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert len(rec["match_candidates"]) == 1
    cand = rec["match_candidates"][0]
    assert cand["line_distance"] == 2
    assert cand["severity_match"] is False  # P1 vs P2


def test_match_candidates_excluded_when_distance_exceeds_tolerance(
    isolated_layout: Path,
) -> None:
    f_codex = _finding(
        path="src/x.py", line_start=10, line_end=10, title="[P1] near top", priority=1
    )
    f_or_far = _finding(
        path="src/x.py",
        line_start=200,
        line_end=200,
        title="[P1] far away",
        priority=1,
    )
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(head_sha=_HEAD_SHA, capture_status="captured", findings=[f_codex]),
    )
    _write_openrouter(
        isolated_layout,
        "pr-1",
        _openrouter_artifact(head_sha=_HEAD_SHA, status="completed", findings=[f_or_far]),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert rec["match_candidates"] == []


def test_match_candidates_excluded_when_paths_differ(isolated_layout: Path) -> None:
    f_codex = _finding(path="src/a.py", line_start=10, line_end=10, title="[P1] a", priority=1)
    f_or = _finding(path="src/b.py", line_start=10, line_end=10, title="[P1] b", priority=1)
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(head_sha=_HEAD_SHA, capture_status="captured", findings=[f_codex]),
    )
    _write_openrouter(
        isolated_layout,
        "pr-1",
        _openrouter_artifact(head_sha=_HEAD_SHA, status="completed", findings=[f_or]),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert rec["match_candidates"] == []


# --------------------------------------------------------------------------- #
# head_sha drift detection
# --------------------------------------------------------------------------- #


def test_head_sha_mismatch_is_surfaced_in_run_view(isolated_layout: Path) -> None:
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(head_sha="0" * 40, capture_status="captured"),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    rec = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json").read_text()
    )
    assert rec["runs"]["codex"]["head_sha_at_capture"] == "0" * 40
    assert rec["runs"]["codex"]["head_sha_matches"] is False


# --------------------------------------------------------------------------- #
# Index rollup
# --------------------------------------------------------------------------- #


def test_index_summarizes_status_counts_and_per_pr_rows(isolated_layout: Path) -> None:
    _write_codex(
        isolated_layout,
        "fixture-pr-1",
        _codex_artifact(head_sha=_HEAD_SHA, capture_status="captured"),
    )
    builder.write_comparison_set(builder.load_sample_registry())
    index = json.loads(
        (isolated_layout / "evaluation" / "bakeoff" / "comparison" / "_index.json").read_text()
    )
    assert index["sample_size"] == 2
    assert index["status_counts"]["awaiting_openrouter"] == 1
    assert index["status_counts"]["pending"] == 1
    pr_ids = {row["pr_id"] for row in index["prs"]}
    assert pr_ids == {"pr-1", "pr-2"}


# --------------------------------------------------------------------------- #
# Idempotency / drift detection
# --------------------------------------------------------------------------- #


def test_check_passes_after_clean_build(isolated_layout: Path) -> None:
    builder.write_comparison_set(builder.load_sample_registry())
    drift = builder.check_comparison_set(builder.load_sample_registry())
    assert drift == []


def test_check_detects_hand_edited_drift(isolated_layout: Path) -> None:
    builder.write_comparison_set(builder.load_sample_registry())
    pr1_path = isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json"
    payload = json.loads(pr1_path.read_text())
    payload["title"] = "hand-edited title"
    pr1_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    drift = builder.check_comparison_set(builder.load_sample_registry())
    assert any("pr-1" in entry for entry in drift)


def test_rebuild_clears_drift(isolated_layout: Path) -> None:
    builder.write_comparison_set(builder.load_sample_registry())
    pr1_path = isolated_layout / "evaluation" / "bakeoff" / "comparison" / "pr-1.json"
    payload = json.loads(pr1_path.read_text())
    payload["title"] = "drifted"
    pr1_path.write_text(json.dumps(payload) + "\n")
    builder.write_comparison_set(builder.load_sample_registry())
    assert builder.check_comparison_set(builder.load_sample_registry()) == []


def test_only_filter_restricts_output(isolated_layout: Path) -> None:
    sample = builder.load_sample_registry()
    only_pr1 = [e for e in sample if e.pr_id == "pr-1"]
    builder.write_comparison_set(only_pr1)
    files = sorted(
        p.name for p in (isolated_layout / "evaluation" / "bakeoff" / "comparison").iterdir()
    )
    assert "pr-1.json" in files
    assert "_index.json" in files
    assert "pr-2.json" not in files


# --------------------------------------------------------------------------- #
# Real-repo smoke
# --------------------------------------------------------------------------- #


def test_real_repo_layout_loads_ten_pr_sample() -> None:
    """Smoke test against the real, committed sample registries.

    Failing this test means the comparison-set builder lost the
    ability to read the actual eval/sample-prs.yaml + the bakeoff
    sample-prs.yml — i.e. a contract regression with the upstream
    Sub-AC 1.1.1 deliverable.
    """
    entries = builder.load_sample_registry()
    assert len(entries) == 10
    assert {e.pr_id for e in entries}.issuperset({"pr-35567", "pr-35509", "pr-35498", "pr-35458"})
    pr_35567 = next(e for e in entries if e.pr_id == "pr-35567")
    assert pr_35567.repo == "dotCMS/core"
    assert pr_35567.surface == "backend"
    assert pr_35567.change_type == "bug_fix"
