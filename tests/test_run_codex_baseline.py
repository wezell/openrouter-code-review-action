"""Smoke tests for the Codex baseline runner (Sub-AC 1.2).

These tests pin the artifact contract: shape of the per-PR JSON
envelope, idempotency of placeholder writes, and correct handling of
selection by ``--pr-id``. They do NOT exercise the live Codex CLI —
that requires real API access and is covered by the bake-off run
itself, not by the unit suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import run_codex_baseline as runner  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_DOC = {
    "prs": [
        {
            "id": "fixture-pr-1",
            "repo": "owner/repo",
            "pr": 1,
            "head_sha": "deadbeef" * 5,  # 40-char SHA
            "web_search_mode_for_run": "disabled",
            "reasoning_effort_for_run": "medium",
        },
        {
            "id": "fixture-pr-2",
            "repo": "owner/repo",
            "pr": 2,
            "head_sha": "feedface" * 5,
            "web_search_mode_for_run": "live",
            "reasoning_effort_for_run": "high",
        },
    ]
}


@pytest.fixture
def isolated_runs(tmp_path, monkeypatch):
    """Redirect the runner to a tmp sample registry + tmp runs/ dir.

    Keeps the real ``eval/runs/`` artifacts untouched while the smoke
    test exercises the placeholder-write path.
    """
    sample_path = tmp_path / "sample-prs.yaml"
    sample_path.write_text(yaml.safe_dump(SAMPLE_DOC, sort_keys=False))
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(runner, "SAMPLE_PATH", sample_path)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    # Strip env so live capture path doesn't accidentally fire.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests — sample loading
# ---------------------------------------------------------------------------


def test_load_sample_parses_pinned_metadata(isolated_runs):
    sample = runner.load_sample()
    assert [pr.id for pr in sample] == ["fixture-pr-1", "fixture-pr-2"]
    pr1 = sample[0]
    assert pr1.repo == "owner/repo"
    assert pr1.pr == 1
    assert pr1.head_sha == "deadbeef" * 5
    assert pr1.web_search_mode == "disabled"
    assert pr1.reasoning_effort == "medium"
    # Default model pinned to action.yml's prior baseline so re-runs
    # against a fresh registry stay reproducible.
    assert pr1.model == runner.DEFAULT_BASELINE_MODEL


def test_load_sample_rejects_non_mapping_top_level(tmp_path, monkeypatch):
    bad = tmp_path / "sample.yaml"
    bad.write_text("- 1\n- 2\n")
    monkeypatch.setattr(runner, "SAMPLE_PATH", bad)
    with pytest.raises(SystemExit):
        runner.load_sample()


# ---------------------------------------------------------------------------
# Tests — placeholder write contract
# ---------------------------------------------------------------------------


def _read_artifact(pr_id: str) -> dict:
    return json.loads((runner.RUNS_DIR / pr_id / "codex.json").read_text())


def test_dry_run_only_writes_pending_placeholder_for_every_pr(isolated_runs):
    rc = runner.main(["--all", "--dry-run-only"])
    assert rc == 0

    for pr_id in ("fixture-pr-1", "fixture-pr-2"):
        artifact = _read_artifact(pr_id)
        # Required top-level keys per methodology §4.2.
        assert artifact["pr_id"] == pr_id
        assert artifact["run"] == "codex"
        assert artifact["model"] == runner.DEFAULT_BASELINE_MODEL
        assert artifact["prior_review_state"] is None
        # Placeholder has no review payload yet — downstream scoring
        # MUST treat this as not-yet-measurable.
        assert artifact["review_run_result"] is None
        assert artifact["capture"]["status"] == "pending"
        assert artifact["capture"]["runner_version"] == runner.RUNNER_VERSION
        # Posting telemetry shape matches AC 3's PostingOutcome dataclass —
        # downstream code that propagates batch_submitted /
        # per_comment_fallback / skipped_after_422 must find them present.
        outcome = artifact["posted"]["posting_outcome"]
        assert set(outcome) == {
            "batch_submitted",
            "per_comment_fallback",
            "skipped_after_422",
        }


def test_pending_placeholder_when_live_env_missing(isolated_runs):
    # Without OPENAI_API_KEY / GITHUB_TOKEN / codex module the runner
    # MUST fall back to writing a pending placeholder rather than
    # silently skipping or writing fabricated findings.
    rc = runner.main(["--all"])
    assert rc == 0
    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "pending"
    assert "OPENAI_API_KEY" in artifact["capture"]["notes"] or (
        "GITHUB_TOKEN" in artifact["capture"]["notes"]
    )


def test_idempotent_skips_already_captured(isolated_runs, monkeypatch):
    # Seed an already-captured artifact for fixture-pr-1.
    sample = runner.load_sample()
    pr1 = sample[0]
    runner.write_artifact(
        pr1,
        capture_status="captured",
        review_run_result={
            "findings": [],
            "carried_forward": [],
            "overall_correctness": "patch is correct",
            "overall_explanation": "no issues found",
            "overall_confidence_score": 0.9,
        },
        posted=None,
        notes="seeded by test",
    )

    # Spy on capture_one to confirm it short-circuits on captured PRs.
    captured_calls: list[str] = []
    real_capture = runner.capture_one

    def _spy(pr, *, dry_run_only, force):
        captured_calls.append(pr.id)
        return real_capture(pr, dry_run_only=dry_run_only, force=force)

    monkeypatch.setattr(runner, "capture_one", _spy)
    rc = runner.main(["--all", "--dry-run-only"])
    assert rc == 0
    assert "fixture-pr-1" in captured_calls  # called …
    # … but the captured artifact is unchanged.
    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "captured"
    assert artifact["capture"]["notes"] == "seeded by test"


def test_force_reruns_even_when_captured(isolated_runs):
    sample = runner.load_sample()
    pr1 = sample[0]
    runner.write_artifact(
        pr1,
        capture_status="captured",
        review_run_result={
            "findings": [],
            "carried_forward": [],
            "overall_correctness": "patch is correct",
            "overall_explanation": "no issues found",
            "overall_confidence_score": 0.9,
        },
        posted=None,
        notes="seeded by test",
    )
    rc = runner.main(["--pr-id", "fixture-pr-1", "--dry-run-only", "--force"])
    assert rc == 0
    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "pending"
    assert "Placeholder written by --dry-run-only" in artifact["capture"]["notes"]


def test_unknown_pr_id_errors(isolated_runs):
    with pytest.raises(SystemExit):
        runner.main(["--pr-id", "not-a-real-id", "--dry-run-only"])


# ---------------------------------------------------------------------------
# Tests — review-run-result extractor (the live-capture parser)
# ---------------------------------------------------------------------------


def test_extract_review_run_result_from_mixed_stdout():
    # The Codex CLI prints diagnostic lines plus a JSON envelope; the
    # extractor must locate the JSON among the noise.
    payload = {
        "findings": [],
        "carried_forward": [],
        "overall_correctness": "patch is correct",
        "overall_explanation": "no issues found",
        "overall_confidence_score": 0.9,
    }
    stdout = (
        "Loading config…\n"
        "Calling Codex…\n"
        f"{json.dumps(payload)}\n"
        "Review completed: patch is correct, 0 findings\n"
    )
    out = runner._extract_review_run_result(stdout)
    assert out == payload


def test_extract_review_run_result_skips_unrelated_braces():
    # First {…} is a metadata blob, NOT the review envelope. The
    # extractor must keep scanning until it finds the envelope keys.
    decoy = json.dumps({"unrelated": "data", "items": [1, 2, 3]})
    payload = {
        "findings": [
            {
                "title": "[P2] x",
                "body": "y",
                "confidence_score": 0.5,
                "priority": 2,
                "code_location": {
                    "absolute_file_path": "foo.py",
                    "line_range": {"start": 1, "end": 1},
                },
            }
        ],
        "carried_forward": [],
        "overall_correctness": "patch needs changes",
        "overall_explanation": "see findings",
        "overall_confidence_score": 0.7,
    }
    stdout = f"banner\n{decoy}\n--- review ---\n{json.dumps(payload)}\n"
    out = runner._extract_review_run_result(stdout)
    assert out == payload


def test_extract_review_run_result_raises_when_no_envelope():
    with pytest.raises(RuntimeError):
        runner._extract_review_run_result('no review here\n{"unrelated": "data"}\n')
