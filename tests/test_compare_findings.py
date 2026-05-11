"""Tests for ``eval.compare_findings`` (Sub-AC 2.4).

Pins these contracts:

1. ``compare_pr`` produces a status-classified comparison even when
   one side or both are pending placeholders.
2. Proximity matching is greedy 1:1, honors the ± tolerance, and
   produces deterministic deltas / overlap counts.
3. The report writes to ``tests/fixtures/parity_report/<pr-id>.json``
   plus ``_summary.json`` and ``--check`` flags drift after a write.
4. ``head_sha_drift`` is detected when both sides are captured but on
   different SHAs and the overlap_ratio is suppressed (``null``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from eval import compare_findings as cf

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _finding(
    *,
    title: str = "🔴 [P1] foo.py:10 thing",
    path: str = "foo.py",
    start: int = 10,
    end: int | None = None,
    priority: int | None = 1,
    confidence: float | None = 0.9,
    side: str = "RIGHT",
) -> dict[str, Any]:
    return {
        "title": title,
        "body": "details",
        "confidence_score": confidence,
        "priority": priority,
        "side": side,
        "code_location": {
            "absolute_file_path": path,
            "line_range": {"start": start, "end": end if end is not None else start},
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
    notes: str = "synthetic",
) -> dict[str, Any]:
    rrr: dict[str, Any] | None
    if findings is None:
        rrr = None
    else:
        rrr = {
            "overall_correctness": "patch is correct",
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
            "notes": notes,
            "runner_version": "1",
            "status": capture_status,
        },
        "source": {
            "eval_artifact": "eval/runs/synthetic.json",
            "produced_by": "synthetic",
            "mirrored_by": "synthetic",
        },
    }


@pytest.fixture
def staged_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    report = tmp_path / "report"
    baseline.mkdir()
    candidate.mkdir()
    report.mkdir()
    monkeypatch.setattr(cf, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cf, "BASELINE_ROOT", baseline)
    monkeypatch.setattr(cf, "CANDIDATE_ROOT", candidate)
    monkeypatch.setattr(cf, "REPORT_ROOT", report)
    monkeypatch.setattr(cf, "SUMMARY_PATH", report / "_summary.json")
    return baseline, candidate, report


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------


def test_pending_when_both_sides_have_no_capture(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, _ = staged_fixtures
    pr = "pr-1"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="aa" * 20,
            findings=None,
            run="codex",
            model="gpt-5.4",
            capture_status="pending",
        ),
    )
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="aa" * 20,
            findings=None,
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="unfilled",
        ),
    )
    cmp = cf.compare_pr(pr, baseline_root=baseline, candidate_root=candidate)
    assert cmp.status == "pending"
    assert len(cmp.codex_findings) == 0
    assert len(cmp.openrouter_findings) == 0


def test_codex_only_status(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, _ = staged_fixtures
    pr = "pr-2"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="bb" * 20,
            findings=[_finding()],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="bb" * 20,
            findings=None,
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="unfilled",
        ),
    )
    cmp = cf.compare_pr(pr, baseline_root=baseline, candidate_root=candidate)
    assert cmp.status == "codex_only"


def test_head_sha_drift_when_captures_disagree(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, _ = staged_fixtures
    pr = "pr-3"
    _write(
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
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="cc" * 20,
            findings=[_finding()],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    cmp = cf.compare_pr(pr, baseline_root=baseline, candidate_root=candidate)
    assert cmp.status == "head_sha_drift"
    payload = cf._build_pr_report(cmp, generated_at="x", tolerance=3)
    # head_sha_drift suppresses overlap ratio so the number is never
    # mistaken for "0% parity".
    assert payload["overlap"]["overlap_ratio_codex_baseline"] is None
    assert payload["overlap"]["overlap_ratio_openrouter_candidate"] is None


# ---------------------------------------------------------------------------
# Proximity matching + counts
# ---------------------------------------------------------------------------


def test_exact_line_match_pairs_findings(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, _ = staged_fixtures
    pr = "pr-10"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="dd" * 20,
            findings=[_finding(start=42, end=44, priority=1)],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="dd" * 20,
            findings=[_finding(start=43, end=44, priority=1)],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    cmp = cf.compare_pr(pr, baseline_root=baseline, candidate_root=candidate)
    assert cmp.status == "ready"
    assert len(cmp.matches) == 1
    assert cmp.matches[0].line_distance == 0
    assert len(cmp.codex_only) == 0
    assert len(cmp.openrouter_only) == 0


def test_proximity_match_within_tolerance_outside_dropped(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, _ = staged_fixtures
    pr = "pr-11"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="ee" * 20,
            findings=[
                _finding(start=10, priority=1),  # near to OR's 12
                _finding(start=200, priority=2, title="🟡 [P2] foo.py:200 x"),
            ],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="ee" * 20,
            findings=[
                _finding(start=12, priority=1),  # within tolerance
                _finding(start=500, priority=1, title="🔴 [P1] foo.py:500 y"),
            ],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    cmp = cf.compare_pr(pr, baseline_root=baseline, candidate_root=candidate)
    assert len(cmp.matches) == 1
    assert cmp.matches[0].line_distance == 2
    # The other Codex finding (line 200) and OR finding (line 500) are
    # both "_only" because no cross-side proximity match exists.
    codex_only_starts = sorted(f.line_start for f in cmp.codex_only)
    or_only_starts = sorted(f.line_start for f in cmp.openrouter_only)
    assert codex_only_starts == [200]
    assert or_only_starts == [500]


def test_different_paths_never_match(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, _ = staged_fixtures
    pr = "pr-12"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="ff" * 20,
            findings=[_finding(path="a.py", start=10)],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="ff" * 20,
            findings=[_finding(path="b.py", start=10)],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    cmp = cf.compare_pr(pr, baseline_root=baseline, candidate_root=candidate)
    assert cmp.matches == []
    assert len(cmp.codex_only) == 1
    assert len(cmp.openrouter_only) == 1


def test_severity_mismatch_recorded_in_match_payload(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, _ = staged_fixtures
    pr = "pr-13"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="11" * 20,
            findings=[_finding(start=20, priority=1, title="🔴 [P1] foo.py:20 a")],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="11" * 20,
            findings=[_finding(start=20, priority=2, title="🟡 [P2] foo.py:20 a")],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    cmp = cf.compare_pr(pr, baseline_root=baseline, candidate_root=candidate)
    payload = cf._build_pr_report(cmp, generated_at="t", tolerance=3)
    assert payload["matches"][0]["severity_match"] is False
    assert payload["matches"][0]["codex_severity"] == "P1"
    assert payload["matches"][0]["openrouter_severity"] == "P2"


def test_delta_by_severity_signed_difference(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, _ = staged_fixtures
    pr = "pr-14"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="22" * 20,
            findings=[
                _finding(start=10, priority=1),
                _finding(start=20, priority=1, title="🔴 [P1] foo.py:20 b"),
            ],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="22" * 20,
            findings=[
                _finding(start=10, priority=1),
                _finding(start=200, priority=2, title="🟡 [P2] bar.py:200 c", path="bar.py"),
                _finding(start=300, priority=3, title="⚪ [P3] bar.py:300 d", path="bar.py"),
            ],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    cmp = cf.compare_pr(pr, baseline_root=baseline, candidate_root=candidate)
    payload = cf._build_pr_report(cmp, generated_at="t", tolerance=3)
    assert payload["counts"]["matched_pairs"] == 1
    assert payload["counts"]["codex_only"] == 1
    assert payload["counts"]["openrouter_only"] == 2
    assert payload["deltas"]["delta_total"] == 1
    deltas = payload["deltas"]["delta_by_severity"]
    # OpenRouter has one more P2, one more P3, one fewer P1.
    assert deltas["P1"] == -1
    assert deltas["P2"] == 1
    assert deltas["P3"] == 1


# ---------------------------------------------------------------------------
# Persistence + drift check
# ---------------------------------------------------------------------------


def test_write_report_emits_per_pr_and_summary(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, report = staged_fixtures
    pr = "pr-30"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="33" * 20,
            findings=[_finding(start=10, priority=1)],
            run="codex",
            model="gpt-5.4",
            capture_status="captured",
        ),
    )
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="33" * 20,
            findings=[_finding(start=10, priority=1)],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )

    comparisons = cf.compare_all([pr])
    written, summary_path = cf.write_report(
        comparisons,
        report_root=report,
        summary_path=report / "_summary.json",
        generated_at="2026-05-07T00:00:00Z",
    )
    assert len(written) == 1
    pr_report = json.loads(written[0].read_text(encoding="utf-8"))
    assert pr_report["status"] == "ready"
    assert pr_report["counts"] == {
        "codex_total": 1,
        "openrouter_total": 1,
        "matched_pairs": 1,
        "codex_only": 0,
        "openrouter_only": 0,
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["sample_size"] == 1
    assert summary["status_counts"] == {"ready": 1}
    assert summary["totals"]["matched_pairs_total"] == 1
    assert summary["ready_overlap"]["overlap_ratio_codex_baseline"] == 1.0


def test_check_mode_flags_drift_then_clears(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, report = staged_fixtures
    pr = "pr-40"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="44" * 20,
            findings=None,
            run="codex",
            model="gpt-5.4",
            capture_status="pending",
        ),
    )
    _write(
        candidate / pr / "openrouter-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="44" * 20,
            findings=None,
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="unfilled",
        ),
    )

    # CLI: --check before any write must report drift (missing files).
    assert cf.main(["--check"]) == 1
    # Persist report.
    assert cf.main([]) == 0
    # Re-running --check after the write must be clean.
    assert cf.main(["--check"]) == 0


def test_main_unknown_pr_id_errors(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, _ = staged_fixtures
    pr = "pr-50"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="55" * 20,
            findings=None,
            run="codex",
            model="gpt-5.4",
            capture_status="pending",
        ),
    )
    with pytest.raises(SystemExit):
        cf.main(["--pr-id", "pr-doesnotexist"])


def test_main_no_fixtures_is_noop_success(
    staged_fixtures: tuple[Path, Path, Path],
) -> None:
    # Both trees empty -> CLI exits 0 (fresh checkout is not a CI failure).
    assert cf.main([]) == 0
