"""Tests for ``eval.overlap_metrics`` (Sub-AC 3.3).

Pins these contracts:

1. ``compute_metrics`` produces precision/recall/F1 with the classic
   IR formulas, and emits ``None`` (→ JSON ``null``) when a denominator
   is zero so reviewers read "not measurable" rather than "0%".
2. ``runs_id_to_fixture_id`` round-trips with ``fixture_id_to_runs_id``
   and rejects unrecognized prefixes.
3. ``build_overlap_record`` reports metrics only when ``status ==
   "ready"``; for ``head_sha_drift`` / ``codex_only`` / ``pending``
   the headline metrics collapse to ``null`` while raw counts remain.
4. ``write_overlap`` persists ``eval/runs/<runs_id>/overlap.json`` and
   ``check_overlap`` flags drift after manual edit / missing files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from eval import compare_findings as cf
from eval import overlap_metrics as om

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
    if findings is None:
        rrr: dict[str, Any] | None = None
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


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    runs = tmp_path / "runs"
    baseline.mkdir()
    candidate.mkdir()
    runs.mkdir()
    monkeypatch.setattr(cf, "BASELINE_ROOT", baseline)
    monkeypatch.setattr(cf, "CANDIDATE_ROOT", candidate)
    monkeypatch.setattr(om, "RUNS_ROOT", runs)
    return baseline, candidate, runs


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def test_runs_id_to_fixture_id_round_trip() -> None:
    assert om.runs_id_to_fixture_id("dotcms-core-35449") == "pr-35449"
    assert om.fixture_id_to_runs_id("pr-35449") == "dotcms-core-35449"


def test_runs_id_rejects_bad_prefix() -> None:
    with pytest.raises(ValueError):
        om.runs_id_to_fixture_id("pr-35449")
    with pytest.raises(ValueError):
        om.runs_id_to_fixture_id("dotcms-core-")


def test_fixture_id_rejects_bad_prefix() -> None:
    with pytest.raises(ValueError):
        om.fixture_id_to_runs_id("dotcms-core-35449")
    with pytest.raises(ValueError):
        om.fixture_id_to_runs_id("pr-")


# ---------------------------------------------------------------------------
# Metrics math
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_overlap() -> None:
    m = om.compute_metrics(tp=5, fp=0, fn=0)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_compute_metrics_partial_overlap() -> None:
    # tp=3, fp=1, fn=2 -> precision=0.75, recall=0.6, f1≈0.6667
    m = om.compute_metrics(tp=3, fp=1, fn=2)
    assert m.precision == 0.75
    assert m.recall == 0.6
    assert m.f1 == round(2 * 0.75 * 0.6 / (0.75 + 0.6), 4)


def test_compute_metrics_zero_predictions_makes_precision_null() -> None:
    m = om.compute_metrics(tp=0, fp=0, fn=2)
    assert m.precision is None
    assert m.recall == 0.0
    assert m.f1 is None


def test_compute_metrics_zero_baseline_makes_recall_null() -> None:
    m = om.compute_metrics(tp=0, fp=2, fn=0)
    assert m.precision == 0.0
    assert m.recall is None
    assert m.f1 is None


def test_compute_metrics_all_zero_yields_all_null() -> None:
    m = om.compute_metrics(tp=0, fp=0, fn=0)
    assert m.precision is None
    assert m.recall is None
    assert m.f1 is None


def test_compute_metrics_negative_counts_raise() -> None:
    with pytest.raises(ValueError):
        om.compute_metrics(tp=-1, fp=0, fn=0)


# ---------------------------------------------------------------------------
# Record assembly: status-gated metrics
# ---------------------------------------------------------------------------


def test_ready_status_emits_metrics(
    staged: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, runs = staged
    pr = "pr-1"
    runs_id = "dotcms-core-1"
    sha = "aa" * 20
    # dotbot has 2 findings; OpenRouter has 2 findings, one matches at
    # the same path/line, the other is a false positive on a new file.
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha=sha,
            findings=[
                _finding(path="foo.py", start=10),
                _finding(path="bar.py", start=20),
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
            head_sha=sha,
            findings=[
                _finding(path="foo.py", start=10),  # match
                _finding(path="other.py", start=99),  # FP
            ],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    path = om.write_overlap(
        runs_id,
        baseline_root=baseline,
        candidate_root=candidate,
        runs_root=runs,
        generated_at="2026-05-07T00:00:00Z",
    )
    assert path == runs / runs_id / "overlap.json"
    record = json.loads(path.read_text())
    assert record["status"] == "ready"
    assert record["pr_id"] == runs_id
    assert record["fixture_pr_id"] == pr
    # tp=1, fp=1, fn=1 -> p=0.5, r=0.5, f1=0.5
    assert record["metrics"]["true_positive"] == 1
    assert record["metrics"]["false_positive"] == 1
    assert record["metrics"]["false_negative"] == 1
    assert record["metrics"]["precision"] == 0.5
    assert record["metrics"]["recall"] == 0.5
    assert record["metrics"]["f1"] == 0.5
    assert record["counts"]["matched_pairs"] == 1
    assert record["counts"]["openrouter_only"] == 1
    assert record["counts"]["codex_only"] == 1


def test_head_sha_drift_suppresses_metrics(
    staged: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, runs = staged
    pr = "pr-2"
    runs_id = "dotcms-core-2"
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
            head_sha="bb" * 20,  # different SHA
            findings=[_finding()],
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="completed",
        ),
    )
    path = om.write_overlap(
        runs_id,
        baseline_root=baseline,
        candidate_root=candidate,
        runs_root=runs,
        generated_at="2026-05-07T00:00:00Z",
    )
    record = json.loads(path.read_text())
    assert record["status"] == "head_sha_drift"
    assert record["metrics"]["precision"] is None
    assert record["metrics"]["recall"] is None
    assert record["metrics"]["f1"] is None
    # raw counts still surface so a reviewer can see what would have
    # matched if the SHAs had aligned
    assert record["counts"]["matched_pairs"] >= 0


def test_pending_status_metrics_null(
    staged: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, runs = staged
    pr = "pr-3"
    runs_id = "dotcms-core-3"
    _write(
        baseline / pr / "codex-findings.json",
        _wrapper(
            pr_id=pr,
            head_sha="cc" * 20,
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
            head_sha="cc" * 20,
            findings=None,
            run="openrouter",
            model="anthropic/claude-opus-4.7",
            capture_status="unfilled",
        ),
    )
    path = om.write_overlap(
        runs_id,
        baseline_root=baseline,
        candidate_root=candidate,
        runs_root=runs,
    )
    record = json.loads(path.read_text())
    assert record["status"] == "pending"
    assert record["metrics"]["precision"] is None
    assert record["metrics"]["recall"] is None
    assert record["metrics"]["f1"] is None
    assert record["counts"]["matched_pairs"] == 0


# ---------------------------------------------------------------------------
# Drift check
# ---------------------------------------------------------------------------


def test_check_overlap_clean_after_write(
    staged: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, runs = staged
    pr = "pr-4"
    runs_id = "dotcms-core-4"
    sha = "dd" * 20
    _write(
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
    _write(
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
    om.write_overlap(
        runs_id,
        baseline_root=baseline,
        candidate_root=candidate,
        runs_root=runs,
    )
    drift = om.check_overlap(
        [runs_id],
        baseline_root=baseline,
        candidate_root=candidate,
        runs_root=runs,
    )
    assert drift == []


def test_check_overlap_detects_missing_file(
    staged: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, runs = staged
    pr = "pr-5"
    runs_id = "dotcms-core-5"
    sha = "ee" * 20
    _write(
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
    _write(
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
    drift = om.check_overlap(
        [runs_id],
        baseline_root=baseline,
        candidate_root=candidate,
        runs_root=runs,
    )
    assert any("missing overlap.json" in d for d in drift)


def test_check_overlap_detects_edit(
    staged: tuple[Path, Path, Path],
) -> None:
    baseline, candidate, runs = staged
    pr = "pr-6"
    runs_id = "dotcms-core-6"
    sha = "ff" * 20
    _write(
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
    _write(
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
    path = om.write_overlap(
        runs_id,
        baseline_root=baseline,
        candidate_root=candidate,
        runs_root=runs,
    )
    # Tamper with the persisted record.
    record = json.loads(path.read_text())
    record["metrics"]["precision"] = 0.0
    path.write_text(json.dumps(record))
    drift = om.check_overlap(
        [runs_id],
        baseline_root=baseline,
        candidate_root=candidate,
        runs_root=runs,
    )
    assert any("differs from rebuild" in d for d in drift)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_runs_ids_filters_prefix(
    staged: tuple[Path, Path, Path],
) -> None:
    _, _, runs = staged
    (runs / "dotcms-core-1").mkdir()
    (runs / "dotcms-core-2").mkdir()
    (runs / "_summary").mkdir()  # leading-underscore non-PR dir
    (runs / "stray.txt").write_text("noise")
    found = om.discover_runs_ids(runs_root=runs)
    assert found == ["dotcms-core-1", "dotcms-core-2"]


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def test_main_writes_for_discovered_targets(
    staged: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline, candidate, runs = staged
    pr = "pr-7"
    runs_id = "dotcms-core-7"
    sha = "11" * 20
    _write(
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
    _write(
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
    # Make a runs dir so ``discover_runs_ids`` finds it.
    (runs / runs_id).mkdir()
    rc = om.main(["--pr-id", runs_id])
    assert rc == 0
    assert (runs / runs_id / "overlap.json").is_file()
    out = capsys.readouterr().out
    assert "overlap.json" in out


def test_main_check_flags_drift(
    staged: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline, candidate, runs = staged
    pr = "pr-8"
    runs_id = "dotcms-core-8"
    sha = "22" * 20
    _write(
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
    _write(
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
    (runs / runs_id).mkdir()
    # No overlap.json written yet; --check should report missing.
    rc = om.main(["--check", "--pr-id", runs_id])
    assert rc == 1
    err = capsys.readouterr().err
    assert "missing overlap.json" in err


def test_main_unknown_pr_id_exits(
    staged: tuple[Path, Path, Path],
) -> None:
    _, _, runs = staged
    (runs / "dotcms-core-9").mkdir()
    with pytest.raises(SystemExit):
        om.main(["--pr-id", "dotcms-core-does-not-exist"])
