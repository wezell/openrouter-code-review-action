"""Tests for the act-mode entrypoint that drives the aider wrapper.

Sub-AC 7.2.3 — verifies that:

* OpenRouter model config (review_model / act_model) and the
  ``OPENROUTER_API_KEY`` action secret flow from :class:`ReviewConfig`
  into :class:`AiderClient` invocations without re-plumbing.
* stdout, stderr, exit codes, and an applied-edit summary are surfaced
  in a structured :class:`ActAgentResult`.
* Failures from aider are translated into ``ActAgentResult`` instances
  with ``succeeded`` set to ``False`` and the exception preserved,
  rather than bubbling raw subprocess errors at the workflow layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from cli.clients.aider_client import (
    AiderClient,
    AiderExecutionError,
    AiderResult,
    DEFAULT_OPENROUTER_API_BASE,
)
from cli.core.config import ReviewConfig
from cli.core.exceptions import ConfigurationError
from cli.workflows.act_runner import (
    ActAgentResult,
    ActModeRunner,
    _extract_applied_files,
)


class _FakeAiderClient:
    """Stand-in for :class:`AiderClient` capturing every invocation."""

    def __init__(
        self,
        *,
        result: AiderResult | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._result = result
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> AiderResult:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result


def _make_config(tmp_path: Path, **overrides: Any) -> ReviewConfig:
    base: dict[str, Any] = {
        "github_token": "ghp_test",
        "repository": "owner/repo",
        "pr_number": 1,
        "mode": "act",
        "model_provider": "openai",
        "openai_api_key": "sk-openai",
        "openrouter_api_key": "sk-or-test",
        "act_model": "anthropic/claude-opus-4.7",
        "review_model": "anthropic/claude-opus-4.7",
        "act_instructions": "fix the lint",
        "stream_output": True,
        "repo_root": tmp_path,
    }
    base.update(overrides)
    cfg = ReviewConfig(**{k: v for k, v in base.items() if k in ReviewConfig.__dataclass_fields__})
    return cfg


def _make_aider_result(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    cwd: Path | None = None,
) -> AiderResult:
    return AiderResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        command=("aider", "--message", "x"),
        cwd=cwd or Path("/tmp"),
    )


# ---------------------------------------------------------------------------
# Construction + config validation
# ---------------------------------------------------------------------------


def test_constructor_uses_aider_client_with_config_secrets(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    runner = ActModeRunner(cfg)
    assert isinstance(runner._aider_client, AiderClient)
    # The runner pulls OpenRouter API base + key directly from config.
    assert runner._aider_client.api_base == DEFAULT_OPENROUTER_API_BASE


def test_execute_rejects_review_mode(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, mode="review")
    runner = ActModeRunner(
        cfg, aider_client=_FakeAiderClient(result=_make_aider_result())
    )
    with pytest.raises(ConfigurationError, match="expected 'act'"):
        runner.execute("do stuff")


def test_execute_requires_non_empty_prompt(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    runner = ActModeRunner(
        cfg, aider_client=_FakeAiderClient(result=_make_aider_result())
    )
    with pytest.raises(ConfigurationError, match="prompt is empty"):
        runner.execute("   ")


def test_execute_requires_act_model(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, act_model="")
    runner = ActModeRunner(
        cfg, aider_client=_FakeAiderClient(result=_make_aider_result())
    )
    with pytest.raises(ConfigurationError, match="Act model is empty"):
        runner.execute("fix things")


def test_execute_requires_openrouter_api_key(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, openrouter_api_key="")
    runner = ActModeRunner(
        cfg, aider_client=_FakeAiderClient(result=_make_aider_result())
    )
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        runner.execute("fix things")


# ---------------------------------------------------------------------------
# Wiring: action inputs → aider arguments
# ---------------------------------------------------------------------------


def test_execute_passes_act_model_and_prompt_to_aider(tmp_path: Path) -> None:
    cfg = _make_config(
        tmp_path,
        act_model="anthropic/claude-opus-4.7",
    )
    fake = _FakeAiderClient(result=_make_aider_result(cwd=tmp_path))
    runner = ActModeRunner(cfg, aider_client=fake)

    runner.execute("fix bug", files=["a.py", "sub/b.py"])

    assert len(fake.calls) == 1
    kw = fake.calls[0]
    assert kw["model"] == "anthropic/claude-opus-4.7"
    assert kw["message"] == "fix bug"
    assert kw["files"] == ["a.py", "sub/b.py"]
    assert kw["repo_path"] == tmp_path
    # Aider's auto-commits are off so EditWorkflow controls git.
    assert kw["auto_commit"] is False
    # check=False so we surface a non-zero exit as a result rather than
    # a raised exception.
    assert kw["check"] is False


def test_execute_threads_stream_output_flag(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, stream_output=False)
    fake = _FakeAiderClient(result=_make_aider_result())
    runner = ActModeRunner(cfg, aider_client=fake)
    runner.execute("fix")
    assert fake.calls[0]["stream"] is False


# ---------------------------------------------------------------------------
# Surfaced output: stdout / stderr / returncode / applied edits
# ---------------------------------------------------------------------------


def test_execute_returns_structured_result_with_applied_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stdout = (
        "Loading model anthropic/claude-opus-4.7\n"
        "Applied edit to src/foo.py\n"
        "Applied edit to src/bar.py\n"
        "Applied edit to src/foo.py\n"  # duplicate dedupes
        "Done.\n"
    )
    cfg = _make_config(tmp_path)
    fake = _FakeAiderClient(
        result=_make_aider_result(returncode=0, stdout=stdout, stderr="", cwd=tmp_path)
    )
    runner = ActModeRunner(cfg, aider_client=fake)

    result = runner.execute("fix things")

    assert isinstance(result, ActAgentResult)
    assert result.returncode == 0
    assert result.succeeded is True
    assert result.stdout == stdout
    assert result.applied_files == ("src/foo.py", "src/bar.py")
    summary = result.render_summary()
    assert "aider exited with status 0" in summary
    assert "Applied edits:" in summary
    assert "- src/foo.py" in summary
    # Streams are also tee'd to the host process for runner logs.
    captured = capsys.readouterr()
    assert "Applied edit to src/foo.py" in captured.out


def test_execute_translates_failure_into_result(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    fake = _FakeAiderClient(
        raises=AiderExecutionError(
            "Aider exited with status 2",
            returncode=2,
            stdout="partial output\nApplied edit to src/done.py\n",
            stderr="boom: openrouter rate limit",
        )
    )
    runner = ActModeRunner(cfg, aider_client=fake)

    result = runner.execute("fix things")

    assert result.returncode == 2
    assert result.succeeded is False
    assert result.error_message is not None
    assert "rate limit" in result.stderr
    assert result.applied_files == ("src/done.py",)
    summary = result.render_summary()
    assert "aider failed" in summary
    assert "stderr:" in summary
    assert "rate limit" in summary


def test_execute_handles_no_applied_edits(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    fake = _FakeAiderClient(
        result=_make_aider_result(stdout="No edits suggested.", cwd=tmp_path)
    )
    runner = ActModeRunner(cfg, aider_client=fake)
    result = runner.execute("fix")
    assert result.applied_files == ()
    assert "Applied edits: (none)" in result.render_summary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_extract_applied_files_dedupes_in_order() -> None:
    text = (
        "Applied edit to a.py\n"
        "Applied edit to b.py\n"
        "Applied edit to a.py\n"
        "Applied edit to c.py\n"
    )
    assert _extract_applied_files(text) == ["a.py", "b.py", "c.py"]


def test_extract_applied_files_ignores_inline_mentions() -> None:
    text = "the model would have applied edit to nowhere if we asked\n"
    assert _extract_applied_files(text) == []


def test_render_summary_truncates_long_output(tmp_path: Path) -> None:
    big = "x" * 20_000
    result = ActAgentResult(
        returncode=0,
        stdout=big,
        stderr="",
        command=("aider",),
        cwd=tmp_path,
        applied_files=(),
    )
    summary = result.render_summary(max_output_chars=200)
    # Truncation marker present and length stays within bounds.
    assert "truncated" in summary
    assert len(summary) < 1_000
