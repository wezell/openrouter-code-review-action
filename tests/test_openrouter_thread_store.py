from __future__ import annotations

import os
from pathlib import Path

from cli.clients.openrouter_thread_store import OpenRouterThreadStore


def _store(tmp_path: Path) -> OpenRouterThreadStore:
    return OpenRouterThreadStore(tmp_path / "threads")


def test_default_directory_honors_env_override(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("OPENROUTER_THREAD_DIR", "/custom/threads")
    assert OpenRouterThreadStore.default_directory() == Path("/custom/threads")


def test_default_directory_prefers_runner_temp(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("OPENROUTER_THREAD_DIR", raising=False)
    monkeypatch.setenv("RUNNER_TEMP", "/runner/tmp")
    assert OpenRouterThreadStore.default_directory() == Path(
        "/runner/tmp/openrouter-review-threads"
    )


def test_default_directory_falls_back_to_home(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("OPENROUTER_THREAD_DIR", raising=False)
    monkeypatch.delenv("RUNNER_TEMP", raising=False)
    expected = Path.home() / ".openrouter-review" / "threads"
    assert OpenRouterThreadStore.default_directory() == expected


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    store.save(
        "thread-1",
        messages,
        repository="owner/repo",
        pr_number=7,
        model="anthropic/claude-opus-4.7",
    )

    loaded = store.load("thread-1")
    assert loaded == messages


def test_load_missing_thread_returns_none(tmp_path: Path) -> None:
    assert _store(tmp_path).load("nope") is None


def test_load_ignores_corrupt_and_foreign_files(tmp_path: Path) -> None:
    directory = tmp_path / "threads"
    directory.mkdir(parents=True)
    (directory / "openrouter-thread-v1-owner-repo-pr-7-m-thread-9.json").write_text(
        "not json", encoding="utf-8"
    )
    (directory / "unrelated.json").write_text("{}", encoding="utf-8")

    store = OpenRouterThreadStore(directory)
    assert store.load("thread-9") is None


def test_latest_thread_id_scopes_to_repo_pr_and_model(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(
        "thread-a",
        [{"role": "user", "content": "a"}],
        repository="owner/repo",
        pr_number=7,
        model="anthropic/claude-opus-4.7",
    )
    store.save(
        "thread-other-model",
        [{"role": "user", "content": "b"}],
        repository="owner/repo",
        pr_number=7,
        model="openai/gpt-5",
    )

    latest = store.latest_thread_id(
        repository="owner/repo",
        pr_number=7,
        model="anthropic/claude-opus-4.7",
    )
    assert latest == "thread-a"

    assert (
        store.latest_thread_id(
            repository="owner/repo",
            pr_number=8,
            model="anthropic/claude-opus-4.7",
        )
        is None
    )


def test_latest_thread_id_picks_most_recent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(
        "thread-old",
        [{"role": "user", "content": "old"}],
        repository="owner/repo",
        pr_number=7,
        model="m",
    )
    store.save(
        "thread-new",
        [{"role": "user", "content": "new"}],
        repository="owner/repo",
        pr_number=7,
        model="m",
    )

    latest = store.latest_thread_id(repository="owner/repo", pr_number=7, model="m")
    assert latest == "thread-new"


def test_save_overwrites_same_thread(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(
        "thread-1",
        [{"role": "user", "content": "before"}],
        repository="owner/repo",
        pr_number=1,
        model="m",
    )
    store.save(
        "thread-1",
        [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "after"},
        ],
        repository="owner/repo",
        pr_number=1,
        model="m",
    )
    loaded = store.load("thread-1")
    assert loaded is not None
    assert len(loaded) == 2


def test_save_rejects_empty_thread_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.save(
            "  ",
            [],
            repository="owner/repo",
            pr_number=1,
            model="m",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty thread_id")


def test_directory_created_lazily(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert not store.directory.exists()
    store.save(
        "thread-1",
        [{"role": "user", "content": "x"}],
        repository="owner/repo",
        pr_number=1,
        model="m",
    )
    assert store.directory.is_dir()


def test_save_sanitizes_unsafe_components(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.save(
        "thread/1",
        [{"role": "user", "content": "x"}],
        repository="owner/../repo",
        pr_number=1,
        model="a/b",
    )
    assert path.parent == store.directory
    assert os.sep not in path.stem.replace("-", "")
