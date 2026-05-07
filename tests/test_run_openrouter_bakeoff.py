"""Smoke + behaviour tests for the OpenRouter bake-off runner (Sub-AC 1.2).

These tests pin the runner's *contract* — the per-PR artifact shape it
writes for each mode (``scaffold``/``dry``/``replay``/``live``) — without
talking to OpenRouter or to GitHub. The point is that the harness is
exercisable offline with deterministic outputs, so the parity sub-ACs
downstream can read the artifacts without depending on network state.

Live-mode credential failure is covered too, because the most common
operator error on a fresh machine is ``--mode live`` without
``OPENROUTER_API_KEY`` set; the runner must fail loudly, not silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bakeoff import run_openrouter as runner  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


SAMPLE_PR = runner.SamplePR(
    pr_id="pr-123",
    repo="owner/repo",
    number=123,
    title="fixture: add a knob",
    head_sha="cafebabe" * 5,
    base_ref="main",
    head_ref="feature/knob",
    surface="backend",
    change_type="feature",
    security_signal=False,
    notes="fixture",
)


# A schema-conforming ReviewRunResult payload. Mirrors the JSON the
# OpenRouter chat-completions endpoint returns when run with the strict
# response_format from openrouter_review_response_format(). One real-ish
# finding so the replay test can assert that findings round-trip.
VALID_REVIEW_PAYLOAD = {
    "overall_correctness": "patch is correct",
    "overall_explanation": "No correctness or security issues identified in the diff.",
    "overall_confidence_score": 0.92,
    "findings": [
        {
            "title": "[P2] Minor: log line uses positional formatting",
            "body": "The `_emit_event` call concatenates `event` with the log "
            "level via `%`. Consider switching to structured logging.",
            "confidence_score": 0.7,
            "priority": 2,
            "code_location": {
                "absolute_file_path": "core/handlers.py",
                "line_range": {"start": 42, "end": 44},
            },
        }
    ],
    "carried_forward": [],
}


def _live_response_envelope(model: str = "anthropic/claude-opus-4.7:online") -> dict:
    """Return a faithful OpenRouter chat-completions envelope for the payload above."""
    return {
        "id": "gen-fixture-0001",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(VALID_REVIEW_PAYLOAD),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1234,
            "completion_tokens": 567,
            "total_tokens": 1801,
        },
    }


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Redirect _REPO_ROOT-anchored paths into a tmp tree and stub diff fetch."""
    runs_dir = tmp_path / "runs"
    replay_dir = tmp_path / "replay"
    runs_dir.mkdir()
    replay_dir.mkdir()

    monkeypatch.setattr(runner, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "DEFAULT_REPLAY_DIR", "replay")

    # The runner reads PROMPT_PATH; point it at an in-memory stub so the
    # tests don't depend on prompts/review.md at this exact path.
    prompt_stub = tmp_path / "prompts_review.md"
    prompt_stub.write_text("system prompt", encoding="utf-8")
    monkeypatch.setattr(runner, "PROMPT_PATH", prompt_stub)

    return {"root": tmp_path, "runs_dir": runs_dir, "replay_dir": replay_dir}


# ---------------------------------------------------------------------------
# Scaffold mode — the canonical checked-in placeholder shape.
# ---------------------------------------------------------------------------


def test_scaffold_writes_unfilled_artifact_with_resolved_run_config(
    isolated_dirs,
):
    artifact, status = runner._handle_pr(
        SAMPLE_PR,
        mode="scaffold",
        out_dir=isolated_dirs["runs_dir"],
        model="anthropic/claude-opus-4.7",
        reasoning_effort="medium",
        web_search_mode="live",
        api_key=None,
    )
    assert status == "unfilled"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    assert doc["status"] == "unfilled"
    assert doc["model"] == "anthropic/claude-opus-4.7"
    assert doc["reasoning_effort"] == "medium"
    assert doc["web_search_mode"] == "live"
    assert doc["review_run_result"] is None
    assert doc["telemetry"] == {}


# ---------------------------------------------------------------------------
# Dry mode — fetches diff, builds payload, persists payload-shape telemetry.
# ---------------------------------------------------------------------------


def test_dry_mode_persists_payload_shape_telemetry(isolated_dirs, monkeypatch):
    monkeypatch.setattr(runner, "fetch_pr_diff", lambda pr: "diff --git a/x b/x\n+ change\n")

    artifact, status = runner._handle_pr(
        SAMPLE_PR,
        mode="dry",
        out_dir=isolated_dirs["runs_dir"],
        model="anthropic/claude-opus-4.7",
        reasoning_effort="medium",
        web_search_mode="live",
        api_key=None,
    )
    assert status == "skipped_dry_run"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    assert doc["status"] == "skipped_dry_run"
    # Sub-AC 1.2 evidence: the runner ran end-to-end up to the POST.
    tel = doc["telemetry"]
    assert tel["diff_chars"] > 0
    # web_search_mode=live → :online suffix wired into the resolved model.
    assert tel["model_resolved"].endswith(":online")
    # System + user messages.
    assert tel["payload_messages"] == 2
    # Strict-schema response_format.
    assert tel["response_format_kind"] == "json_schema"
    # Bake-off runner deliberately does not stream.
    assert tel["stream"] is False
    assert tel["reasoning_effort_resolved"] == "medium"
    # Live-only field stays null on dry runs.
    assert tel["usage"] is None
    # No findings synthesis on dry runs (per methodology §4.1, §5).
    assert doc["review_run_result"] is None


def test_dry_mode_records_diff_fetch_failure_per_pr(isolated_dirs, monkeypatch):
    """One bad diff fetch shouldn't sink the whole sweep."""

    def boom(pr):
        raise RuntimeError("gh pr diff failed: 404")

    monkeypatch.setattr(runner, "fetch_pr_diff", boom)

    artifact, status = runner._handle_pr(
        SAMPLE_PR,
        mode="dry",
        out_dir=isolated_dirs["runs_dir"],
        model="anthropic/claude-opus-4.7",
        reasoning_effort="medium",
        web_search_mode="live",
        api_key=None,
    )
    # Failure on a single PR's diff fetch is recorded but does NOT raise.
    assert status == "skipped_dry_run"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    assert doc["telemetry"]["dry_run_error"].startswith("RuntimeError:")
    assert doc["telemetry"]["diff_chars"] is None
    assert "failed for this PR" in doc["notes"]


# ---------------------------------------------------------------------------
# Replay mode — re-derive findings from a committed live response.
# ---------------------------------------------------------------------------


def test_replay_with_recorded_response_persists_validated_findings(
    isolated_dirs,
):
    replay_path = isolated_dirs["replay_dir"] / f"{SAMPLE_PR.pr_id}.json"
    replay_path.write_text(json.dumps(_live_response_envelope()), encoding="utf-8")

    artifact, status = runner._handle_pr(
        SAMPLE_PR,
        mode="replay",
        out_dir=isolated_dirs["runs_dir"],
        model="anthropic/claude-opus-4.7",
        reasoning_effort="medium",
        web_search_mode="live",
        api_key=None,
    )
    assert status == "replayed"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    assert doc["status"] == "replayed"
    # Findings round-trip from the recorded response, *via* schema validation.
    assert doc["review_run_result"]["overall_correctness"] == "patch is correct"
    assert len(doc["review_run_result"]["findings"]) == 1
    finding = doc["review_run_result"]["findings"][0]
    assert finding["priority"] == 2
    assert finding["code_location"]["absolute_file_path"] == "core/handlers.py"
    # Telemetry surfaces the replay provenance.
    tel = doc["telemetry"]
    assert tel["model_resolved"] == "anthropic/claude-opus-4.7:online"
    assert tel["usage"]["total_tokens"] == 1801
    assert tel["replay_source"].endswith(f"{SAMPLE_PR.pr_id}.json")


def test_replay_without_recorded_response_writes_unfilled(isolated_dirs):
    artifact, status = runner._handle_pr(
        SAMPLE_PR,
        mode="replay",
        out_dir=isolated_dirs["runs_dir"],
        model="anthropic/claude-opus-4.7",
        reasoning_effort="medium",
        web_search_mode="live",
        api_key=None,
    )
    assert status == "unfilled"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    assert doc["status"] == "unfilled"
    assert "no recorded OpenRouter response" in doc["notes"]
    assert doc["review_run_result"] is None


def test_replay_with_invalid_recorded_response_records_error(isolated_dirs):
    """A malformed replay file is captured as status=error, not crashing the sweep."""
    replay_path = isolated_dirs["replay_dir"] / f"{SAMPLE_PR.pr_id}.json"
    # Schema-violating payload: missing required overall_correctness.
    bad_envelope = {
        "id": "gen-bad",
        "model": "anthropic/claude-opus-4.7",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"findings": [], "carried_forward": []}),
                },
            }
        ],
    }
    replay_path.write_text(json.dumps(bad_envelope), encoding="utf-8")

    artifact, status = runner._handle_pr(
        SAMPLE_PR,
        mode="replay",
        out_dir=isolated_dirs["runs_dir"],
        model="anthropic/claude-opus-4.7",
        reasoning_effort="medium",
        web_search_mode="live",
        api_key=None,
    )
    assert status == "error"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    assert doc["status"] == "error"
    assert "Replay failed" in doc["notes"]
    assert doc["review_run_result"] is None


# ---------------------------------------------------------------------------
# Live mode — must fail loudly without an API key.
# ---------------------------------------------------------------------------


def test_live_mode_without_api_key_raises(isolated_dirs):
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        runner._handle_pr(
            SAMPLE_PR,
            mode="live",
            out_dir=isolated_dirs["runs_dir"],
            model="anthropic/claude-opus-4.7",
            reasoning_effort="medium",
            web_search_mode="live",
            api_key=None,
        )


# ---------------------------------------------------------------------------
# CLI surface — replay added to --mode choices, scaffold remains default.
# ---------------------------------------------------------------------------


def test_cli_mode_choices_include_replay():
    args = runner.parse_args(["--mode", "replay", "--only", "pr-123"])
    assert args.mode == "replay"
    assert args.only == ["pr-123"]


def test_cli_default_mode_is_scaffold():
    args = runner.parse_args([])
    assert args.mode == "scaffold"
