"""Smoke tests for the OpenRouter+aider review baseline runner (Sub-AC 2.2).

Pin the artifact contract: per-PR JSON envelope shape, idempotency,
selection by ``--pr-id``, and the placeholder fallback when live env
is missing. These tests do NOT exercise the real OpenRouter API or
the in-tree CLI subprocess — those require live API access and are
covered by the bake-off run itself, not by the unit suite.
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

from eval import run_openrouter_baseline as runner  # noqa: E402


SAMPLE_DOC = {
    "prs": [
        {
            "id": "fixture-pr-1",
            "repo": "owner/repo",
            "pr": 1,
            "head_sha": "deadbeef" * 5,
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
    sample_path = tmp_path / "sample-prs.yaml"
    sample_path.write_text(yaml.safe_dump(SAMPLE_DOC, sort_keys=False))
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(runner, "SAMPLE_PATH", sample_path)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    # Strip any inherited live-env so the live capture path can't fire.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return tmp_path


def _read_artifact(pr_id: str) -> dict:
    return json.loads((runner.RUNS_DIR / pr_id / "openrouter.json").read_text())


# ---------------------------------------------------------------------------
# Sample loader
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
    # Default review model pinned to seed.yaml's initial model so re-runs
    # against a fresh registry stay reproducible.
    assert pr1.model == runner.DEFAULT_REVIEW_MODEL


def test_load_sample_rejects_non_mapping_top_level(tmp_path, monkeypatch):
    bad = tmp_path / "sample.yaml"
    bad.write_text("- 1\n- 2\n")
    monkeypatch.setattr(runner, "SAMPLE_PATH", bad)
    with pytest.raises(SystemExit):
        runner.load_sample()


# ---------------------------------------------------------------------------
# Placeholder + idempotency contract
# ---------------------------------------------------------------------------


def test_dry_run_only_writes_pending_placeholder_for_every_pr(isolated_runs):
    rc = runner.main(["--all", "--dry-run-only"])
    assert rc == 0

    for pr_id in ("fixture-pr-1", "fixture-pr-2"):
        artifact = _read_artifact(pr_id)
        assert artifact["pr_id"] == pr_id
        assert artifact["run"] == "openrouter"
        assert artifact["model"]  # resolved at call time
        assert artifact["prior_review_state"] is None
        # Placeholder has no review payload yet — downstream scoring
        # MUST treat this as not-yet-measurable.
        assert artifact["review_run_result"] is None
        assert artifact["capture"]["status"] == "pending"
        assert artifact["capture"]["runner_version"] == runner.RUNNER_VERSION
        # Posting telemetry shape matches AC 3's PostingOutcome dataclass —
        # keys must be present even on placeholders.
        outcome = artifact["posted"]["posting_outcome"]
        assert set(outcome) == {
            "batch_submitted",
            "per_comment_fallback",
            "skipped_after_422",
        }


def test_pending_placeholder_when_live_env_missing(isolated_runs):
    # Without OPENROUTER_API_KEY / GITHUB_TOKEN the runner MUST fall
    # back to a pending placeholder rather than fabricating findings
    # or silently skipping.
    rc = runner.main(["--all"])
    assert rc == 0
    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "pending"
    notes = artifact["capture"]["notes"]
    assert "OPENROUTER_API_KEY" in notes or "GITHUB_TOKEN" in notes


def test_idempotent_skips_already_captured(isolated_runs, monkeypatch):
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

    rc = runner.main(["--all", "--dry-run-only"])
    assert rc == 0
    artifact = _read_artifact("fixture-pr-1")
    # Already-captured artifact untouched on a non-forced re-run.
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
# ReviewRunResult extractor — guards the live-capture parser contract
# ---------------------------------------------------------------------------


def test_extract_review_run_result_from_mixed_stdout():
    payload = {
        "findings": [],
        "carried_forward": [],
        "overall_correctness": "patch is correct",
        "overall_explanation": "no issues found",
        "overall_confidence_score": 0.9,
    }
    stdout = (
        "Loading config…\n"
        "Calling OpenRouter…\n"
        f"{json.dumps(payload)}\n"
        "Review completed: patch is correct, 0 findings\n"
    )
    out = runner._extract_review_run_result(stdout)
    assert out == payload


def test_extract_review_run_result_skips_unrelated_braces():
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
        runner._extract_review_run_result(
            'no review here\n{"unrelated": "data"}\n'
        )


# ---------------------------------------------------------------------------
# Live env check — guarantees both required secrets gate the live path
# ---------------------------------------------------------------------------


def test_live_env_ok_requires_openrouter_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_x")
    ok, why = runner._live_env_ok()
    assert ok is False
    assert "OPENROUTER_API_KEY" in why


def test_live_env_ok_requires_github_token(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_x")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    ok, why = runner._live_env_ok()
    assert ok is False
    assert "GITHUB_TOKEN" in why


def test_live_env_ok_passes_when_both_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_x")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_x")
    ok, why = runner._live_env_ok()
    assert ok is True
    assert why == ""


# ---------------------------------------------------------------------------
# capture_one — exercises the full capture path with the CLI mocked out
# ---------------------------------------------------------------------------


def test_capture_one_live_path_writes_captured_artifact(isolated_runs, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_test")

    sample = runner.load_sample()
    pr1 = sample[0]

    # Avoid real git over the network.
    monkeypatch.setattr(runner, "_checkout_pr", lambda pr, dest: None)

    fake_payload = {
        "findings": [],
        "carried_forward": [],
        "overall_correctness": "patch is correct",
        "overall_explanation": "no issues found",
        "overall_confidence_score": 0.9,
    }

    def fake_invoke(pr, repo_path, *, review_model, reasoning_effort, web_search_mode):
        # Confirms the harness threads model + reasoning + web-search through.
        assert review_model == runner.DEFAULT_REVIEW_MODEL
        assert reasoning_effort == "medium"
        assert web_search_mode == "disabled"
        return fake_payload

    monkeypatch.setattr(runner, "_invoke_review_cli", fake_invoke)

    status = runner.capture_one(
        pr1,
        dry_run_only=False,
        force=False,
        review_model=runner.DEFAULT_REVIEW_MODEL,
        reasoning_effort="medium",
        web_search_mode="disabled",
    )
    assert status == "captured"

    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "captured"
    assert artifact["review_run_result"] == fake_payload
    assert artifact["capture"]["captured_at"] is not None


def test_capture_one_live_path_records_failure(isolated_runs, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_test")
    monkeypatch.setattr(runner, "_checkout_pr", lambda pr, dest: None)

    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "_invoke_review_cli", boom)

    sample = runner.load_sample()
    status = runner.capture_one(
        sample[0],
        dry_run_only=False,
        force=False,
        review_model=runner.DEFAULT_REVIEW_MODEL,
        reasoning_effort="medium",
        web_search_mode="disabled",
    )
    assert status == "failed"

    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "failed"
    assert "boom" in artifact["capture"]["notes"]


# ---------------------------------------------------------------------------
# Model resolution — swap-speed contract
# ---------------------------------------------------------------------------


def test_resolve_review_model_returns_defaults_when_loader_unavailable(
    monkeypatch, tmp_path
):
    # Point repo_root at an empty dir so load_model_config has nothing
    # to read; resolver MUST fall back to the seed.yaml-pinned defaults.
    rm, re_, ws = runner.resolve_review_model(tmp_path)
    assert rm == runner.DEFAULT_REVIEW_MODEL
    assert re_ in {"minimal", "low", "medium", "high"}
    assert ws in {"disabled", "cached", "live"}
