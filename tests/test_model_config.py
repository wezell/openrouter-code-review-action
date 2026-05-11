"""Tests for the in-repo model config file (.openrouter-review.yml).

These tests pin AC 8 from seed.yaml: "Changing review or act model is a single
edit to the in-repo config file." They assert that:

* Editing only the YAML file flips ``ReviewConfig.selected_model`` for both
  modes — no Python code changes, no action input changes.
* Defaults (``anthropic/claude-opus-4.7``) apply when the file is absent.
* Malformed files surface a ``ConfigurationError`` instead of silently using
  the defaults.
* Explicit ``from_args`` overrides still win over the file (so a workflow can
  pin a model for a one-off run) but in normal use the file is the source of
  truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.core.config import ReviewConfig
from cli.core.exceptions import ConfigurationError
from cli.core.model_config import (
    DEFAULT_ACT_MODEL,
    DEFAULT_CONFIG_PATH,
    DEFAULT_REVIEW_MODEL,
    ONLINE_SUFFIX,
    WEB_SEARCH_MODES,
    ModelConfig,
    apply_web_search_mode,
    load_model_config,
    resolve_config_path,
)


def _write_config(repo_root: Path, body: str) -> Path:
    target = repo_root / DEFAULT_CONFIG_PATH
    target.write_text(body, encoding="utf-8")
    return target


# --- pure loader -----------------------------------------------------------


def test_load_model_config_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_model_config(repo_root=tmp_path)

    assert cfg.review_model == DEFAULT_REVIEW_MODEL
    assert cfg.act_model == DEFAULT_ACT_MODEL
    assert cfg.source_path is None


def test_load_model_config_reads_per_mode_models(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "review:\n  model: anthropic/claude-opus-4.7\nact:\n  model: openai/gpt-5.4\n",
    )

    cfg = load_model_config(repo_root=tmp_path)

    assert cfg.review_model == "anthropic/claude-opus-4.7"
    assert cfg.act_model == "openai/gpt-5.4"
    assert cfg.source_path == (tmp_path / DEFAULT_CONFIG_PATH).resolve()


def test_load_model_config_partial_review_only_keeps_act_default(tmp_path: Path) -> None:
    _write_config(tmp_path, "review:\n  model: anthropic/claude-opus-4.7\n")

    cfg = load_model_config(repo_root=tmp_path)

    assert cfg.review_model == "anthropic/claude-opus-4.7"
    assert cfg.act_model == DEFAULT_ACT_MODEL


def test_load_model_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    _write_config(tmp_path, ":\nthis: is: : invalid")

    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_model_config(repo_root=tmp_path)


def test_load_model_config_rejects_unknown_keys(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "review:\n  model: anthropic/claude-opus-4.7\n  bogus: 1\n",
    )

    with pytest.raises(ConfigurationError, match="unknown 'review' keys: bogus"):
        load_model_config(repo_root=tmp_path)


def test_load_model_config_rejects_invalid_reasoning_effort(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "review:\n  model: anthropic/claude-opus-4.7\n  reasoning_effort: extreme\n",
    )

    with pytest.raises(ConfigurationError, match="reasoning_effort"):
        load_model_config(repo_root=tmp_path)


def test_load_model_config_rejects_web_search_mode_under_act(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "act:\n  model: anthropic/claude-opus-4.7\n  web_search_mode: live\n",
    )

    with pytest.raises(ConfigurationError, match="web_search_mode"):
        load_model_config(repo_root=tmp_path)


def test_load_model_config_rejects_top_level_non_mapping(tmp_path: Path) -> None:
    _write_config(tmp_path, "- a\n- b\n")

    with pytest.raises(ConfigurationError, match="must be a YAML mapping"):
        load_model_config(repo_root=tmp_path)


def test_resolve_config_path_uses_default_when_unset(tmp_path: Path) -> None:
    assert resolve_config_path(tmp_path, None) == (tmp_path / DEFAULT_CONFIG_PATH).resolve()
    assert resolve_config_path(tmp_path, "") == (tmp_path / DEFAULT_CONFIG_PATH).resolve()


def test_resolve_config_path_honors_absolute_path(tmp_path: Path) -> None:
    abs_path = tmp_path / "elsewhere" / "models.yml"
    assert resolve_config_path(tmp_path, str(abs_path)) == abs_path


def test_model_config_for_mode_picks_correct_slug() -> None:
    cfg = ModelConfig(review_model="r/x", act_model="a/y")
    assert cfg.model_for_mode("review") == "r/x"
    assert cfg.model_for_mode("act") == "a/y"
    # Unknown mode falls back to review (defensive — schema disallows other modes)
    assert cfg.model_for_mode("anything-else") == "r/x"


# --- ReviewConfig integration (the AC 8 contract) --------------------------


@pytest.fixture()
def base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Minimal env that lets ReviewConfig.from_args succeed in either mode."""

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    # Avoid leaking host env vars into the loader
    monkeypatch.delenv("OPENROUTER_REVIEW_CONFIG", raising=False)
    monkeypatch.delenv("OPENROUTER_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_ACT_MODEL", raising=False)
    return tmp_path


def test_review_config_uses_defaults_when_no_in_repo_file(base_env: Path) -> None:
    config = ReviewConfig.from_args(pr_number=1)

    assert config.review_model == DEFAULT_REVIEW_MODEL
    assert config.act_model == DEFAULT_ACT_MODEL
    assert config.selected_model == DEFAULT_REVIEW_MODEL


def test_review_config_picks_up_review_model_from_in_repo_file(base_env: Path) -> None:
    _write_config(base_env, "review:\n  model: anthropic/claude-opus-4.7\n")

    config = ReviewConfig.from_args(pr_number=1, mode="review")

    assert config.selected_model == "anthropic/claude-opus-4.7"


def test_swap_review_model_is_single_file_edit(base_env: Path) -> None:
    """The full AC 8 contract — swapping models requires only a YAML edit."""

    target = base_env / DEFAULT_CONFIG_PATH

    target.write_text("review:\n  model: anthropic/claude-opus-4.7\n", encoding="utf-8")
    first = ReviewConfig.from_args(pr_number=1, mode="review")
    assert first.selected_model == "anthropic/claude-opus-4.7"

    # The "single edit" — change one line in the YAML, nothing else.
    target.write_text("review:\n  model: openai/gpt-5.4\n", encoding="utf-8")
    second = ReviewConfig.from_args(pr_number=1, mode="review")

    assert second.selected_model == "openai/gpt-5.4"
    # No other config drift from that single edit:
    assert second.review_model == "openai/gpt-5.4"
    assert second.act_model == DEFAULT_ACT_MODEL


def test_swap_act_model_is_single_file_edit(
    base_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same contract for act mode — the in-repo file is the source of truth."""

    monkeypatch.setenv("CODEX_MODE", "act")
    target = base_env / DEFAULT_CONFIG_PATH

    target.write_text("act:\n  model: anthropic/claude-opus-4.7\n", encoding="utf-8")
    first = ReviewConfig.from_args(pr_number=1, mode="act", act_instructions="x")
    assert first.selected_model == "anthropic/claude-opus-4.7"

    target.write_text("act:\n  model: google/gemini-2.5-pro\n", encoding="utf-8")
    second = ReviewConfig.from_args(pr_number=1, mode="act", act_instructions="x")
    assert second.selected_model == "google/gemini-2.5-pro"


def test_explicit_kwarg_still_wins_over_in_repo_file(base_env: Path) -> None:
    _write_config(base_env, "review:\n  model: anthropic/claude-opus-4.7\n")

    config = ReviewConfig.from_args(pr_number=1, mode="review", review_model="openai/gpt-5.4")

    assert config.selected_model == "openai/gpt-5.4"


def test_in_repo_file_can_override_reasoning_effort_and_web_search(
    base_env: Path,
) -> None:
    _write_config(
        base_env,
        "review:\n"
        "  model: anthropic/claude-opus-4.7\n"
        "  reasoning_effort: high\n"
        "  web_search_mode: cached\n",
    )

    config = ReviewConfig.from_args(pr_number=1, mode="review")

    assert config.reasoning_effort == "high"
    assert config.web_search_mode == "cached"


def test_custom_config_path_resolved_relative_to_repo_root(
    base_env: Path,
) -> None:
    custom = base_env / "ci" / "models.yml"
    custom.parent.mkdir(parents=True)
    custom.write_text("review:\n  model: anthropic/claude-sonnet-4.5\n", encoding="utf-8")

    config = ReviewConfig.from_args(pr_number=1, mode="review", config_path="ci/models.yml")

    assert config.selected_model == "anthropic/claude-sonnet-4.5"


def test_invalid_in_repo_file_raises_configuration_error(base_env: Path) -> None:
    _write_config(base_env, "review:\n  model: 12345\n")

    with pytest.raises(ConfigurationError, match="must be a non-empty string"):
        ReviewConfig.from_args(pr_number=1, mode="review")


# --- AC 5: web_search_mode → OpenRouter :online slug mapping ---------------
#
# AC 5 requires that the disabled/cached/live web-search knob is honored by
# routing live mode through OpenRouter's ``:online`` slug variant, while
# disabled and cached leave the slug untouched. The helper is the single
# choke point that both the bake-off runner and the production review path
# call into, so swapping models or modes never silently flips web search.


def test_web_search_modes_enum_is_three_values() -> None:
    assert WEB_SEARCH_MODES == frozenset({"disabled", "cached", "live"})


def test_apply_web_search_mode_live_appends_online_suffix() -> None:
    assert (
        apply_web_search_mode("anthropic/claude-opus-4.7", "live")
        == f"anthropic/claude-opus-4.7{ONLINE_SUFFIX}"
    )
    assert (
        apply_web_search_mode("openai/gpt-5.4", "live") == f"openai/gpt-5.4{ONLINE_SUFFIX}"
    )


def test_apply_web_search_mode_live_is_idempotent() -> None:
    already = f"anthropic/claude-opus-4.7{ONLINE_SUFFIX}"
    # Repeated application must not produce ":online:online".
    assert apply_web_search_mode(already, "live") == already


@pytest.mark.parametrize("mode", ["disabled", "cached"])
def test_apply_web_search_mode_disabled_and_cached_leave_slug_untouched(mode: str) -> None:
    slug = "anthropic/claude-opus-4.7"
    assert apply_web_search_mode(slug, mode) == slug
    # And do not strip an existing :online if the operator pinned it
    # explicitly in the slug — the helper is non-destructive.
    pinned = f"{slug}{ONLINE_SUFFIX}"
    assert apply_web_search_mode(pinned, mode) == pinned


def test_apply_web_search_mode_handles_whitespace_and_unknown_modes() -> None:
    # Whitespace in the slug is trimmed.
    assert (
        apply_web_search_mode("  anthropic/claude-opus-4.7  ", "live")
        == f"anthropic/claude-opus-4.7{ONLINE_SUFFIX}"
    )
    # None / empty / unknown modes default to no-op (do not silently turn on
    # web search). Validation of the *config value* happens at file-load time.
    slug = "anthropic/claude-opus-4.7"
    assert apply_web_search_mode(slug, None) == slug
    assert apply_web_search_mode(slug, "") == slug
    assert apply_web_search_mode(slug, "bogus") == slug


def test_apply_web_search_mode_returns_input_when_not_a_string() -> None:
    # Defensive: any non-string slug (e.g. ``None`` from a misconfigured
    # caller) is returned unchanged so we never produce ``None:online``.
    assert apply_web_search_mode("", "live") == ""
    assert apply_web_search_mode(None, "live") is None  # type: ignore[arg-type]


def test_apply_web_search_mode_works_for_any_provider_slug() -> None:
    # AC 5 is provider-agnostic — swap-speed means flipping the slug to a
    # different vendor must keep the :online mapping intact.
    for slug in (
        "google/gemini-2.5-pro",
        "openai/gpt-5.4",
        "deepseek/deepseek-chat",
        "anthropic/claude-sonnet-4.5",
    ):
        assert apply_web_search_mode(slug, "live") == f"{slug}{ONLINE_SUFFIX}"
        assert apply_web_search_mode(slug, "disabled") == slug
        assert apply_web_search_mode(slug, "cached") == slug


def test_review_config_routes_live_through_apply_web_search_mode(base_env: Path) -> None:
    """End-to-end: a YAML edit setting web_search_mode: live makes the
    helper produce the :online slug for the resolved review model."""

    _write_config(
        base_env,
        "review:\n"
        "  model: anthropic/claude-opus-4.7\n"
        "  web_search_mode: live\n",
    )
    config = ReviewConfig.from_args(pr_number=1, mode="review")

    assert config.web_search_mode == "live"
    assert (
        apply_web_search_mode(config.selected_model, config.web_search_mode)
        == f"anthropic/claude-opus-4.7{ONLINE_SUFFIX}"
    )


@pytest.mark.parametrize("mode", ["disabled", "cached"])
def test_review_config_disabled_and_cached_skip_online_suffix(
    base_env: Path, mode: str
) -> None:
    _write_config(
        base_env,
        f"review:\n"
        f"  model: anthropic/claude-opus-4.7\n"
        f"  web_search_mode: {mode}\n",
    )
    config = ReviewConfig.from_args(pr_number=1, mode="review")

    assert config.web_search_mode == mode
    assert (
        apply_web_search_mode(config.selected_model, config.web_search_mode)
        == "anthropic/claude-opus-4.7"
    )
