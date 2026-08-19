"""Tests for ``eval.overlap_aggregate`` (Sub-AC 3.4).

Pins these contracts:

1. ``_read_sample_registry`` extracts ``id: dotcms-core-<n>`` lines
   from the YAML registry, ignores comments/blanks, dedupes, and
   tolerates a missing file.
2. ``write_aggregate`` writes a single JSON record at
   ``eval/overlap_aggregate.json`` covering every sample PR.
3. Headline micro-metrics aggregate TP/FP/FN across **only** PRs
   whose status is ``ready`` (head_sha drift / pending PRs do not
   contaminate the rate).
4. Macro metrics average per-PR P/R/F1 over ``ready`` PRs and report
   ``averaged_over_prs`` so a reviewer knows the sample size.
5. Per-category breakdown emits the five severity buckets
   (``P0..P3`` + ``unknown``), splits matched-pair counts by dotbot vs
   OpenRouter side, and computes per-bucket precision/recall.
6. ``check_aggregate`` is clean immediately after a write; mutating
   the on-disk file (or deleting it) surfaces a drift message.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from eval import compare_findings as cf
from eval import overlap_aggregate as oa

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _finding(
    *,
    title: str = "🔴 [P1] foo.py:10 thing",
    path: str = "foo.py",
    start: int = 10,
    end: int | None = None,
    priority: int | None = 1,
) -> dict[str, Any]:
    return {
        "title": title,
        "body": "details",
        "confidence_score": 0.9,
        "priority": priority,
        "side": "RIGHT",
        "code_location": {
            "absolute_file_path": path,
            "line_range": {
                "start": start,
                "end": end if end is not None else start,
            },
        },
    }


def _wrapper(
    *,
    pr_id: str,
    head_sha: str,
    findings: list[dict[str, Any]] | None,
    run: str,
    model: str,
    capture_status: str,
) -> dict[str, Any]:
    rrr: dict[str, Any] | None
    if findings is None:
        rrr = None
    else:
        rrr = {
            "overall_correctness": "ok",
            "overall_explanation": "ok",
            "overall_confidence_score": 0.8,
            "findings": findings,
            "carried_forward": [],
        }
    return {
        "pr_id": pr_id,
        "run": run,
        "head_sha": head_sha,
        "model": model,
        "reasoning_effort": "medium",
        "web_search_mode": "disabled",
        "prior_review_state": None,
        "review_run_result": rrr,
        "posted": {
            "summary_id": None,
            "inline_ids": [],
            "posting_outcome": {
                "batch_submitted": 0,
                "per_comment_fallback": 0,
                "skipped_after_422": 0,
            },
        },
        "capture": {
            "captured_at": None,
            "command": "synthetic",
            "notes": "synthetic",
            "runner_version": "1",
            "status": capture_status,
        },
        "source": {
            "eval_artifact": "eval/runs/synthetic.json",
            "produced_by": "synthetic",
            "mirrored_by": "synthetic",
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_registry(path: Path, runs_ids: list[str]) -> None:
    """Write a minimal sample-prs.yaml shape that exercises the parser."""
    lines = ["# synthetic registry", "prs:"]
    for rid in runs_ids:
        lines.append(f"  - id: {rid}")
        lines.append("    repo: dotcms/core")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    registry = tmp_path / "sample-prs.yaml"
    aggregate = tmp_path / "overlap_aggregate.json"
    baseline.mkdir()
    candidate.mkdir()
    monkeypatch.setattr(cf, "BASELINE_ROOT", baseline)
    monkeypatch.setattr(cf, "CANDIDATE_ROOT", candidate)
    monkeypatch.setattr(oa, "SAMPLE_REGISTRY_PATH", registry)
    monkeypatch.setattr(oa, "AGGREGATE_PATH", aggregate)
    return baseline, candidate, registry, aggregate


# ---------------------------------------------------------------------------
# Registry reader
# ---------------------------------------------------------------------------


def test_read_sample_registry_extracts_ids(tmp_path: Path) -> None:
    p = tmp_path / "registry.yaml"
    p.write_text(
        "# preamble\n"
        "required_slots:\n"
        "  - real_bug_present\n"
        "prs:\n"
        "  - id: dotcms-core-1\n"
        "    repo: x\n"
        "  - id: 'dotcms-core-2'\n"
        '  - id: "dotcms-core-3"\n',
        encoding="utf-8",
    )
    assert oa._read_sample_registry(p) == [
        "dotcms-core-1",
        "dotcms-core-2",
        "dotcms-core-3",
    ]


def test_read_sample_registry_dedupes_and_skips_non_dotcms(
    tmp_path: Path,
) -> None:
    p = tmp_path / "registry.yaml"
    p.write_text(
        "prs:\n  - id: dotcms-core-1\n  - id: dotcms-core-1\n  - id: other-repo-99\n",
        encoding="utf-8",
    )
    assert oa._read_sample_registry(p) == ["dotcms-core-1"]


def test_read_sample_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    assert oa._read_sample_registry(tmp_path / "nope.yaml") == []


# ---------------------------------------------------------------------------
# End-to-end: ready PR
# ---------------------------------------------------------------------------


def test_ready_pr_emits_micro_macro_and_per_category(
    staged: tuple[Path, Path, Path, Path],
) -> None:
    baseline, candidate, registry, aggregate = staged
    pr = "pr-1"
    runs_id = "dotcms-core-1"
    sha = "aa" * 20
    _write_registry(registry, [runs_id])
    # dotbot emits two findings; OpenRouter pairs one (matched, same
    # P1 severity), misses the other, and adds a P3 false positive.
    _write_json(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=[
                _finding(path="foo.py", start=10, priority=1, title="🔴 [P1] foo.py:10 a"),
                _finding(path="bar.py", start=20, priority=0, title="🔴 [P0] bar.py:20 b"),
            ],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write_json(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=[
                _finding(path="foo.py", start=10, priority=1, title="🔴 [P1] foo.py:10 a"),
                _finding(path="other.py", start=99, priority=3, title="🟡 [P3] other.py:99 c"),
            ],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    path = oa.write_aggregate(generated_at="2026-05-07T00:00:00Z")
    assert path == aggregate
    record = json.loads(path.read_text())

    assert record["sample_size"] == 1
    assert record["ready_count"] == 1
    assert record["target_source"] == "registry"
    assert record["status_counts"] == {"ready": 1}

    micro = record["metrics"]["micro"]
    # tp=1, fp=1 (other.py P3), fn=1 (bar.py P0 unmatched)
    assert micro["true_positive"] == 1
    assert micro["false_positive"] == 1
    assert micro["false_negative"] == 1
    assert micro["precision"] == 0.5
    assert micro["recall"] == 0.5
    assert micro["f1"] == 0.5

    macro = record["metrics"]["macro"]
    assert macro["averaged_over_prs"] == 1
    assert macro["precision"] == 0.5
    assert macro["recall"] == 0.5
    assert macro["f1"] == 0.5

    # Severity match: codex P1 ↔ openrouter P1 → 1 of 1 matches agree.
    assert record["totals"]["severity_match"] == 1
    assert record["metrics"]["severity_match_rate"] == 1.0

    cats = record["per_category"]
    # Buckets always present, even when empty.
    assert set(cats.keys()) == {"P0", "P1", "P2", "P3", "unknown"}
    assert cats["P0"]["codex_total"] == 1
    assert cats["P0"]["matched_codex"] == 0
    assert cats["P0"]["recall"] == 0.0
    assert cats["P1"]["codex_total"] == 1
    assert cats["P1"]["openrouter_total"] == 1
    assert cats["P1"]["matched_codex"] == 1
    assert cats["P1"]["matched_openrouter"] == 1
    assert cats["P1"]["recall"] == 1.0
    assert cats["P1"]["precision"] == 1.0
    assert cats["P3"]["openrouter_total"] == 1
    assert cats["P3"]["matched_openrouter"] == 0
    assert cats["P3"]["precision"] == 0.0

    rows = {row["pr_id"]: row for row in record["prs"]}
    assert runs_id in rows
    assert rows[runs_id]["fixture_pr_id"] == pr
    assert rows[runs_id]["precision"] == 0.5


def test_pending_prs_excluded_from_micro_and_per_category(
    staged: tuple[Path, Path, Path, Path],
) -> None:
    baseline, candidate, registry, aggregate = staged
    runs_ready = "dotcms-core-1"
    runs_pending = "dotcms-core-2"
    pr_ready = "pr-1"
    pr_pending = "pr-2"
    _write_registry(registry, [runs_ready, runs_pending])
    sha = "bb" * 20
    # Ready PR: 1 match at P2, no FP, no FN.
    _write_json(
        baseline / pr_ready / "codex-findings.json",
        _wrapper(
            pr_id=pr_ready,
            head_sha=sha,
            findings=[_finding(path="x.py", start=5, priority=2, title="🟠 [P2] x.py:5 a")],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write_json(
        candidate / pr_ready / "openrouter-findings.json",
        _wrapper(
            pr_id=pr_ready,
            head_sha=sha,
            findings=[_finding(path="x.py", start=5, priority=2, title="🟠 [P2] x.py:5 a")],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    # Pending PR: codex captured, openrouter unfilled.
    _write_json(
        baseline / pr_pending / "codex-findings.json",
        _wrapper(
            pr_id=pr_pending,
            head_sha=sha,
            findings=[
                _finding(path="loud.py", start=1, priority=0, title="🔴 [P0] loud.py:1 boom")
            ],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write_json(
        candidate / pr_pending / "openrouter-findings.json",
        _wrapper(
            pr_id=pr_pending,
            head_sha=sha,
            findings=None,
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="unfilled",
        ),
    )
    record = json.loads(oa.write_aggregate(generated_at="2026-05-07T00:00:00Z").read_text())
    assert record["sample_size"] == 2
    assert record["ready_count"] == 1
    assert record["status_counts"]["ready"] == 1
    assert record["status_counts"]["codex_only"] == 1

    # Micro: only the ready PR contributes.
    micro = record["metrics"]["micro"]
    assert micro["true_positive"] == 1
    assert micro["false_positive"] == 0
    assert micro["false_negative"] == 0
    assert micro["precision"] == 1.0
    assert micro["recall"] == 1.0

    # Per-category: only the ready PR's findings counted.
    cats = record["per_category"]
    assert cats["P0"]["codex_total"] == 0
    assert cats["P2"]["codex_total"] == 1
    assert cats["P2"]["recall"] == 1.0


def test_zero_ready_prs_yields_null_metrics(
    staged: tuple[Path, Path, Path, Path],
) -> None:
    baseline, candidate, registry, aggregate = staged
    _write_registry(registry, ["dotcms-core-1"])
    pr = "pr-1"
    sha = "cc" * 20
    _write_json(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=None,
            run="codex",
            model="gpt-5.4",
            capture_status="pending",
        ),
    )
    _write_json(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=None,
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="unfilled",
        ),
    )
    record = json.loads(oa.write_aggregate(generated_at="2026-05-07T00:00:00Z").read_text())
    assert record["ready_count"] == 0
    assert record["metrics"]["micro"]["precision"] is None
    assert record["metrics"]["micro"]["recall"] is None
    assert record["metrics"]["micro"]["f1"] is None
    assert record["metrics"]["macro"]["precision"] is None
    assert record["metrics"]["macro"]["recall"] is None
    assert record["metrics"]["macro"]["f1"] is None
    assert record["metrics"]["macro"]["averaged_over_prs"] == 0
    assert record["metrics"]["severity_match_rate"] is None
    # Every per-category bucket is null too because no ready PRs.
    for sev in ("P0", "P1", "P2", "P3", "unknown"):
        assert record["per_category"][sev]["precision"] is None
        assert record["per_category"][sev]["recall"] is None


def test_head_sha_drift_does_not_contaminate_micro(
    staged: tuple[Path, Path, Path, Path],
) -> None:
    baseline, candidate, registry, aggregate = staged
    _write_registry(registry, ["dotcms-core-1"])
    pr = "pr-1"
    _write_json(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="aa" * 20,
            findings=[_finding()],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write_json(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="bb" * 20,
            findings=[_finding()],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    record = json.loads(oa.write_aggregate(generated_at="2026-05-07T00:00:00Z").read_text())
    assert record["status_counts"] == {"head_sha_drift": 1}
    assert record["ready_count"] == 0
    assert record["metrics"]["micro"]["true_positive"] == 0
    assert record["metrics"]["micro"]["precision"] is None


# ---------------------------------------------------------------------------
# Drift check
# ---------------------------------------------------------------------------


def test_check_aggregate_clean_after_write(
    staged: tuple[Path, Path, Path, Path],
) -> None:
    baseline, candidate, registry, aggregate = staged
    _write_registry(registry, ["dotcms-core-1"])
    pr = "pr-1"
    sha = "dd" * 20
    _write_json(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=[_finding()],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write_json(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=[_finding()],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    oa.write_aggregate(generated_at="2026-05-07T00:00:00Z")
    assert oa.check_aggregate() == []


def test_check_aggregate_flags_missing_file(
    staged: tuple[Path, Path, Path, Path],
) -> None:
    baseline, candidate, registry, aggregate = staged
    _write_registry(registry, ["dotcms-core-1"])
    drift = oa.check_aggregate()
    assert drift
    assert any("missing" in m for m in drift)


def test_check_aggregate_flags_mutation(
    staged: tuple[Path, Path, Path, Path],
) -> None:
    baseline, candidate, registry, aggregate = staged
    _write_registry(registry, ["dotcms-core-1"])
    pr = "pr-1"
    sha = "ee" * 20
    _write_json(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=[_finding()],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write_json(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=[_finding()],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    oa.write_aggregate(generated_at="2026-05-07T00:00:00Z")
    record = json.loads(aggregate.read_text())
    record["metrics"]["micro"]["precision"] = 0.0  # corrupt
    aggregate.write_text(json.dumps(record), encoding="utf-8")
    drift = oa.check_aggregate()
    assert drift
    assert any("differs from rebuild" in m for m in drift)


# ---------------------------------------------------------------------------
# Discovery fallback when registry is missing
# ---------------------------------------------------------------------------


def test_discovery_fallback_when_registry_missing(
    staged: tuple[Path, Path, Path, Path],
) -> None:
    baseline, candidate, registry, aggregate = staged
    # Do not write a registry — discovery should kick in.
    pr = "pr-9"
    sha = "ff" * 20
    _write_json(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=[_finding()],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write_json(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=[_finding()],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    record = json.loads(oa.write_aggregate(generated_at="2026-05-07T00:00:00Z").read_text())
    assert record["target_source"] == "fixture_discovery"
    assert record["sample_size"] == 1
    assert record["prs"][0]["pr_id"] == "dotcms-core-9"
    assert record["prs"][0]["fixture_pr_id"] == pr
