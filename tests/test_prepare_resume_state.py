from __future__ import annotations

from pathlib import Path

from cli.review.prepare_resume_state import _resolve_cache_model_name


def _write_config(workspace: Path, model: str) -> None:
    (workspace / ".openrouter-review.yml").write_text(
        f"review:\n  model: {model}\n", encoding="utf-8"
    )


def test_openrouter_provider_reads_model_from_in_repo_config(
    monkeypatch,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_config(workspace, "openai/gpt-5")

    monkeypatch.setenv("CODEX_PROVIDER", "openrouter")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.delenv("OPENROUTER_REVIEW_CONFIG", raising=False)
    monkeypatch.setenv("OPENROUTER_REVIEW_MODEL", "google/gemini-2.5-pro")

    assert _resolve_cache_model_name() == "openai/gpt-5"


def test_openrouter_provider_falls_back_to_env_when_no_config_file(
    monkeypatch,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("CODEX_PROVIDER", "openrouter")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.delenv("OPENROUTER_REVIEW_CONFIG", raising=False)
    monkeypatch.setenv("OPENROUTER_REVIEW_MODEL", "google/gemini-2.5-pro")

    assert _resolve_cache_model_name() == "google/gemini-2.5-pro"


def test_openrouter_provider_defaults_without_config_or_env(
    monkeypatch,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("CODEX_PROVIDER", "openrouter")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.delenv("OPENROUTER_REVIEW_CONFIG", raising=False)
    monkeypatch.delenv("OPENROUTER_REVIEW_MODEL", raising=False)

    assert _resolve_cache_model_name() == "anthropic/claude-opus-4.7"


def test_openrouter_provider_honors_config_path_override(
    monkeypatch,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    custom = tmp_path / "custom-models.yml"
    custom.write_text("review:\n  model: meta-llama/llama-4\n", encoding="utf-8")
    # Default config location exists too — the override must win.
    _write_config(workspace, "openai/gpt-5")

    monkeypatch.setenv("CODEX_PROVIDER", "openrouter")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.setenv("OPENROUTER_REVIEW_CONFIG", str(custom))

    assert _resolve_cache_model_name() == "meta-llama/llama-4"


def test_openai_provider_uses_codex_model_input(
    monkeypatch,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_config(workspace, "openai/gpt-5")

    monkeypatch.setenv("CODEX_PROVIDER", "openai")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.setenv("CODEX_MODEL_INPUT", "gpt-5.4")

    assert _resolve_cache_model_name() == "gpt-5.4"


def test_provider_defaults_to_openrouter_when_unset(
    monkeypatch,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_config(workspace, "openai/gpt-5")

    monkeypatch.delenv("CODEX_PROVIDER", raising=False)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.delenv("OPENROUTER_REVIEW_MODEL", raising=False)

    assert _resolve_cache_model_name() == "openai/gpt-5"


def test_missing_workspace_still_resolves_env_model(
    monkeypatch,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    monkeypatch.delenv("CODEX_PROVIDER", raising=False)
    monkeypatch.delenv("GITHUB_WORKSPACE", raising=False)
    monkeypatch.setenv("OPENROUTER_REVIEW_MODEL", "google/gemini-2.5-pro")
    # Without GITHUB_WORKSPACE the config path resolves against the CWD;
    # point it at an empty dir so no in-repo config file interferes.
    monkeypatch.chdir(tmp_path)

    assert _resolve_cache_model_name() == "google/gemini-2.5-pro"
