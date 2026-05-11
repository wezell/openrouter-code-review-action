"""Tests for the aider invocation wrapper (Sub-AC 7.2.1)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cli.clients.aider_client import (
    DEFAULT_OPENROUTER_API_BASE,
    AiderClient,
    AiderExecutionError,
    AiderResult,
    build_aider_command,
)


class _FakeRunner:
    """Minimal stand-in for ``subprocess.run`` capturing every call."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        raises: Exception | None = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": list(argv), **kwargs})
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            args=argv,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file.py").write_text("print('hi')\n")
    return repo


@pytest.fixture
def fake_aider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stage a fake aider binary on PATH so shutil.which resolves it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "aider"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return fake


# ---------- build_aider_command ---------------------------------------------


def test_build_command_includes_model_message_and_defaults() -> None:
    argv = build_aider_command(
        model="anthropic/claude-opus-4.7",
        message="refactor this",
    )
    assert argv[0] == "aider"
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "openrouter/anthropic/claude-opus-4.7"
    assert "--message" in argv
    assert argv[argv.index("--message") + 1] == "refactor this"
    # Non-interactive defaults
    assert "--yes-always" in argv
    assert "--no-pretty" in argv
    assert "--no-show-model-warnings" in argv
    # Default API base is OpenRouter
    assert argv[argv.index("--openai-api-base") + 1] == DEFAULT_OPENROUTER_API_BASE


def test_build_command_idempotent_openrouter_prefix() -> None:
    argv = build_aider_command(
        model="openrouter/anthropic/claude-opus-4.7",
        message="x",
    )
    idx = argv.index("--model")
    assert argv[idx + 1] == "openrouter/anthropic/claude-opus-4.7"


def test_build_command_rejects_empty_model() -> None:
    with pytest.raises(ValueError):
        build_aider_command(model="  ", message="x")


def test_build_command_appends_files_at_end() -> None:
    argv = build_aider_command(
        model="anthropic/claude-opus-4.7",
        message="edit",
        files=["a.py", Path("sub/b.py")],
    )
    # Files are positional and must be at the tail
    assert argv[-2:] == ["a.py", "sub/b.py"]


def test_build_command_passes_api_key_via_argv() -> None:
    argv = build_aider_command(
        model="anthropic/claude-opus-4.7",
        message="edit",
        api_key="sk-or-test-1234",
    )
    idx = argv.index("--api-key")
    assert argv[idx + 1] == "openrouter=sk-or-test-1234"


def test_build_command_omits_api_key_when_absent() -> None:
    argv = build_aider_command(model="anthropic/claude-opus-4.7", message="edit")
    assert "--api-key" not in argv


def test_build_command_stream_and_commit_flags() -> None:
    on = build_aider_command(
        model="anthropic/claude-opus-4.7",
        message="edit",
        stream=True,
        auto_commit=True,
    )
    assert "--stream" in on
    assert "--auto-commits" in on

    off = build_aider_command(
        model="anthropic/claude-opus-4.7",
        message="edit",
        stream=False,
        auto_commit=False,
    )
    assert "--no-stream" in off
    assert "--no-auto-commits" in off


def test_build_command_extra_args_passed_through() -> None:
    argv = build_aider_command(
        model="anthropic/claude-opus-4.7",
        message="edit",
        extra_args=["--map-tokens", "1024"],
    )
    assert "--map-tokens" in argv
    assert argv[argv.index("--map-tokens") + 1] == "1024"


def test_build_command_omits_api_base_when_none() -> None:
    argv = build_aider_command(
        model="anthropic/claude-opus-4.7",
        message="edit",
        api_base=None,
    )
    assert "--openai-api-base" not in argv


def test_build_command_respects_explicit_binary() -> None:
    argv = build_aider_command(
        model="anthropic/claude-opus-4.7",
        message="edit",
        aider_binary="/opt/aider",
    )
    assert argv[0] == "/opt/aider"


# ---------- AiderClient.run --------------------------------------------------


def test_run_invokes_subprocess_with_resolved_argv_and_cwd(
    repo_dir: Path, fake_aider: Path
) -> None:
    runner = _FakeRunner(returncode=0, stdout="ok", stderr="")
    client = AiderClient(api_key="sk-or-x", runner=runner)

    result = client.run(
        model="anthropic/claude-opus-4.7",
        message="fix bug",
        repo_path=repo_dir,
        files=["file.py"],
    )

    assert isinstance(result, AiderResult)
    assert result.succeeded
    assert result.returncode == 0
    assert result.cwd == repo_dir.resolve()

    assert len(runner.calls) == 1
    call = runner.calls[0]
    argv = call["argv"]
    # Binary resolved to fake aider absolute path
    assert argv[0] == str(fake_aider)
    # Cwd passed through to subprocess
    assert call["cwd"] == str(repo_dir.resolve())
    # Files appear at the tail
    assert argv[-1] == "file.py"


def test_run_rejects_nonexistent_repo_path(tmp_path: Path) -> None:
    client = AiderClient(runner=_FakeRunner())
    missing = tmp_path / "does-not-exist"
    with pytest.raises(AiderExecutionError, match="does not exist"):
        client.run(
            model="anthropic/claude-opus-4.7",
            message="fix",
            repo_path=missing,
        )


def test_run_raises_on_missing_aider_binary(
    monkeypatch: pytest.MonkeyPatch, repo_dir: Path
) -> None:
    # Empty PATH so shutil.which cannot find aider.
    monkeypatch.setenv("PATH", "")
    client = AiderClient(runner=_FakeRunner())
    with pytest.raises(AiderExecutionError, match="not found on PATH"):
        client.run(
            model="anthropic/claude-opus-4.7",
            message="fix",
            repo_path=repo_dir,
        )


def test_run_raises_on_nonzero_exit_when_check_true(
    repo_dir: Path, fake_aider: Path
) -> None:
    runner = _FakeRunner(returncode=2, stdout="boom", stderr="ouch")
    client = AiderClient(runner=runner)
    with pytest.raises(AiderExecutionError) as info:
        client.run(
            model="anthropic/claude-opus-4.7",
            message="fix",
            repo_path=repo_dir,
            check=True,
        )
    assert info.value.returncode == 2
    assert info.value.stderr == "ouch"


def test_run_returns_failure_when_check_false(
    repo_dir: Path, fake_aider: Path
) -> None:
    runner = _FakeRunner(returncode=3, stdout="", stderr="bad")
    client = AiderClient(runner=runner)
    result = client.run(
        model="anthropic/claude-opus-4.7",
        message="fix",
        repo_path=repo_dir,
        check=False,
    )
    assert not result.succeeded
    assert result.returncode == 3
    assert result.stderr == "bad"


def test_run_translates_timeout(repo_dir: Path, fake_aider: Path) -> None:
    runner = _FakeRunner(raises=subprocess.TimeoutExpired(cmd="aider", timeout=1))
    client = AiderClient(runner=runner)
    with pytest.raises(AiderExecutionError, match="timed out"):
        client.run(
            model="anthropic/claude-opus-4.7",
            message="fix",
            repo_path=repo_dir,
            timeout=1,
        )


def test_run_sets_openrouter_env_for_subprocess(
    repo_dir: Path, fake_aider: Path
) -> None:
    runner = _FakeRunner(returncode=0)
    client = AiderClient(api_key="sk-or-y", runner=runner)
    client.run(
        model="anthropic/claude-opus-4.7",
        message="fix",
        repo_path=repo_dir,
        env={"PATH": os.environ.get("PATH", "")},
    )
    env = runner.calls[0]["env"]
    assert env["OPENROUTER_API_BASE"] == DEFAULT_OPENROUTER_API_BASE
    assert env["OPENROUTER_API_KEY"] == "sk-or-y"


def test_run_does_not_clobber_existing_api_key_in_env(
    repo_dir: Path, fake_aider: Path
) -> None:
    runner = _FakeRunner(returncode=0)
    client = AiderClient(api_key="sk-or-from-config", runner=runner)
    client.run(
        model="anthropic/claude-opus-4.7",
        message="fix",
        repo_path=repo_dir,
        env={
            "PATH": os.environ.get("PATH", ""),
            "OPENROUTER_API_KEY": "sk-or-existing",
        },
    )
    env = runner.calls[0]["env"]
    # Pre-existing env value wins so callers can override per-run.
    assert env["OPENROUTER_API_KEY"] == "sk-or-existing"


def test_build_command_via_client_returns_dry_run_argv(
    repo_dir: Path, fake_aider: Path
) -> None:
    client = AiderClient(api_key="sk-or-z")
    argv = client.build_command(
        model="anthropic/claude-opus-4.7",
        message="dry",
        files=[Path("file.py")],
    )
    assert "--message" in argv
    assert argv[argv.index("--message") + 1] == "dry"
    assert "openrouter=sk-or-z" in argv
    assert argv[-1] == "file.py"
