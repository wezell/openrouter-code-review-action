"""Tests for ``eval.normalize_openrouter_findings`` (Sub-AC 2.3).

Pins three contracts:

1. ``normalize_review_run_result`` reshapes raw model output to the
   §1 envelope (defaulting omitted ``side`` to ``null``) and rejects
   payloads that violate the strict review schema.
2. ``normalize_one`` mirrors ``eval/runs/<id>/openrouter.json`` into
   ``tests/fixtures/candidate/<pr-id>/openrouter-findings.json``,
   adding the §4 ``source`` block and re-validating
   ``review_run_result``.
3. The CLI is idempotent: a second invocation reports no drift via
   ``--check`` after a clean write.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cli.core.exceptions import ReviewContractError
from eval import normalize_openrouter_findings as nrm

# ---------------------------------------------------------------------------
# Pure normalizer
# ---------------------------------------------------------------------------


def _valid_review_run_result(*, side: str | None = "RIGHT") -> dict[str, Any]:
    finding: dict[str, Any] = {
        "title": "🔴 [P1] foo.py:10 something",
        "body": "explain",
        "confidence_score": 0.9,
        "priority": 1,
        "code_location": {
            "absolute_file_path": "foo.py",
            "line_range": {"start": 10, "end": 10},
        },
    }
    if side is not None:
        finding["side"] = side
    return {
        "overall_correctness": "patch is correct",
        "overall_explanation": "looks good",
        "overall_confidence_score": 0.8,
        "findings": [finding],
        "carried_forward": [],
    }


def test_normalize_passes_through_none() -> None:
    assert nrm.normalize_review_run_result(None) is None


def test_normalize_canonicalizes_omitted_side() -> None:
    """Legacy fixtures predating Sub-AC 3.1 omit ``side``; the
    normalizer must re-emit it as ``null`` so the §1 envelope's
    required-keys contract holds."""
    raw = _valid_review_run_result(side=None)
    assert "side" not in raw["findings"][0]

    out = nrm.normalize_review_run_result(raw)
    assert out is not None
    assert out["findings"][0]["side"] is None
    # All §1 envelope keys present.
    assert set(out.keys()) == {
        "overall_correctness",
        "overall_explanation",
        "overall_confidence_score",
        "findings",
        "carried_forward",
    }


def test_normalize_rejects_schema_violation() -> None:
    bad = _valid_review_run_result()
    bad["findings"][0]["title"] = 12345  # type: ignore[assignment]
    with pytest.raises(ReviewContractError):
        nrm.normalize_review_run_result(bad)


def test_normalize_rejects_missing_envelope_key() -> None:
    bad = _valid_review_run_result()
    del bad["overall_correctness"]
    with pytest.raises(ReviewContractError):
        nrm.normalize_review_run_result(bad)


# ---------------------------------------------------------------------------
# Mirror loop / wrapper construction
# ---------------------------------------------------------------------------


def _wrap_artifact(
    rrr: dict[str, Any] | None, *, pr_id: str = "dotcms-core-99999"
) -> dict[str, Any]:
    return {
        "pr_id": pr_id,
        "run": "openrouter",
        "head_sha": "deadbeef" * 5,
        "model": "anthropic/claude-opus-4.7",
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
            "command": "python -m eval.run_openrouter_baseline --pr-id dotcms-core-99999",
            "notes": "synthetic",
            "runner_version": "1",
            "status": "captured" if rrr is not None else "pending",
        },
    }


@pytest.fixture
def staged_eval_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    eval_runs = tmp_path / "eval" / "runs"
    fixture_root = tmp_path / "tests" / "fixtures" / "candidate"
    eval_runs.mkdir(parents=True)
    fixture_root.mkdir(parents=True)
    monkeypatch.setattr(nrm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(nrm, "EVAL_RUNS", eval_runs)
    monkeypatch.setattr(nrm, "FIXTURE_ROOT", fixture_root)
    return tmp_path


def test_normalize_one_writes_fixture_with_source_block(staged_eval_dir: Path) -> None:
    eval_dir = staged_eval_dir / "eval" / "runs" / "dotcms-core-99999"
    eval_dir.mkdir()
    (eval_dir / "openrouter.json").write_text(
        json.dumps(_wrap_artifact(_valid_review_run_result(), pr_id="dotcms-core-99999")),
        encoding="utf-8",
    )

    fixture_id, changed = nrm.normalize_one(eval_dir, check_only=False)
    assert fixture_id == "pr-99999"
    assert changed is True

    written = json.loads(
        (
            staged_eval_dir
            / "tests"
            / "fixtures"
            / "candidate"
            / "pr-99999"
            / "openrouter-findings.json"
        ).read_text(encoding="utf-8")
    )
    assert written["pr_id"] == "pr-99999"
    assert written["run"] == "openrouter"
    assert written["source"] == {
        "eval_artifact": "eval/runs/dotcms-core-99999/openrouter.json",
        "produced_by": "eval.run_openrouter_baseline",
        "mirrored_by": "eval.normalize_openrouter_findings",
    }
    # review_run_result was re-validated and re-emitted via as_dict().
    assert written["review_run_result"]["findings"][0]["side"] == "RIGHT"


def test_normalize_one_idempotent_check_mode(staged_eval_dir: Path) -> None:
    eval_dir = staged_eval_dir / "eval" / "runs" / "dotcms-core-12345"
    eval_dir.mkdir()
    (eval_dir / "openrouter.json").write_text(
        json.dumps(_wrap_artifact(_valid_review_run_result(), pr_id="dotcms-core-12345")),
        encoding="utf-8",
    )

    nrm.normalize_one(eval_dir, check_only=False)
    # Second pass — nothing changed.
    fixture_id, changed = nrm.normalize_one(eval_dir, check_only=True)
    assert fixture_id == "pr-12345"
    assert changed is False


def test_normalize_one_handles_pending_placeholder(staged_eval_dir: Path) -> None:
    """A ``capture.status == 'pending'`` artifact has ``review_run_result`` =
    null. The normalizer must mirror it without choking on the missing
    payload — that lets us bootstrap the candidate fixture tree before
    the live OpenRouter capture has run on a host."""
    eval_dir = staged_eval_dir / "eval" / "runs" / "dotcms-core-77777"
    eval_dir.mkdir()
    (eval_dir / "openrouter.json").write_text(
        json.dumps(_wrap_artifact(None, pr_id="dotcms-core-77777")),
        encoding="utf-8",
    )

    fixture_id, changed = nrm.normalize_one(eval_dir, check_only=False)
    assert fixture_id == "pr-77777"
    assert changed is True

    written = json.loads(
        (
            staged_eval_dir
            / "tests"
            / "fixtures"
            / "candidate"
            / "pr-77777"
            / "openrouter-findings.json"
        ).read_text(encoding="utf-8")
    )
    assert written["review_run_result"] is None
    assert written["capture"]["status"] == "pending"


def test_normalize_one_rejects_schema_violating_artifact(staged_eval_dir: Path) -> None:
    """If the upstream artifact's review_run_result violates the schema,
    the normalizer must raise so the bake-off scoring inputs are never
    silently corrupted."""
    bad_rrr = _valid_review_run_result()
    bad_rrr["findings"][0]["confidence_score"] = "not-a-number"  # type: ignore[assignment]
    eval_dir = staged_eval_dir / "eval" / "runs" / "dotcms-core-55555"
    eval_dir.mkdir()
    (eval_dir / "openrouter.json").write_text(
        json.dumps(_wrap_artifact(bad_rrr, pr_id="dotcms-core-55555")),
        encoding="utf-8",
    )
    with pytest.raises(ReviewContractError):
        nrm.normalize_one(eval_dir, check_only=False)


def test_main_with_no_eval_runs_returns_zero(staged_eval_dir: Path) -> None:
    # No openrouter.json under eval/runs — must be a no-op success so a
    # fresh checkout doesn't trip CI.
    rc = nrm.main([])
    assert rc == 0


def test_main_check_mode_flags_drift_then_clears(staged_eval_dir: Path) -> None:
    eval_dir = staged_eval_dir / "eval" / "runs" / "dotcms-core-33333"
    eval_dir.mkdir()
    (eval_dir / "openrouter.json").write_text(
        json.dumps(_wrap_artifact(_valid_review_run_result(), pr_id="dotcms-core-33333")),
        encoding="utf-8",
    )

    # First check: nothing on disk → drift.
    assert nrm.main(["--check"]) == 1
    # Write fixtures, then check again: clean.
    assert nrm.main([]) == 0
    assert nrm.main(["--check"]) == 0


def test_main_pr_id_filter_accepts_both_id_styles(staged_eval_dir: Path) -> None:
    for slug in ("dotcms-core-11111", "dotcms-core-22222"):
        ed = staged_eval_dir / "eval" / "runs" / slug
        ed.mkdir()
        (ed / "openrouter.json").write_text(
            json.dumps(_wrap_artifact(_valid_review_run_result(), pr_id=slug)),
            encoding="utf-8",
        )

    assert nrm.main(["--pr-id", "pr-11111"]) == 0
    assert (
        staged_eval_dir
        / "tests"
        / "fixtures"
        / "candidate"
        / "pr-11111"
        / "openrouter-findings.json"
    ).is_file()
    # The other PR was *not* mirrored.
    assert not (
        staged_eval_dir
        / "tests"
        / "fixtures"
        / "candidate"
        / "pr-22222"
        / "openrouter-findings.json"
    ).exists()

    # Eval-id form also works.
    assert nrm.main(["--pr-id", "dotcms-core-22222"]) == 0
    assert (
        staged_eval_dir
        / "tests"
        / "fixtures"
        / "candidate"
        / "pr-22222"
        / "openrouter-findings.json"
    ).is_file()


def test_main_unknown_pr_id_errors(staged_eval_dir: Path) -> None:
    eval_dir = staged_eval_dir / "eval" / "runs" / "dotcms-core-44444"
    eval_dir.mkdir()
    (eval_dir / "openrouter.json").write_text(
        json.dumps(_wrap_artifact(_valid_review_run_result(), pr_id="dotcms-core-44444")),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        nrm.main(["--pr-id", "pr-99999"])
