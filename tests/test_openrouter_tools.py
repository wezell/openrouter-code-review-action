from __future__ import annotations

from pathlib import Path

from cli.clients.openrouter_tools import TOOL_SPECS, execute_tool


def test_tool_specs_expose_the_three_tools() -> None:
    names = {spec["function"]["name"] for spec in TOOL_SPECS}
    assert names == {"read_file", "write_file", "run_command"}


def test_read_file_returns_content(tmp_path: Path) -> None:
    target = tmp_path / "src.py"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = execute_tool(
        "read_file",
        '{"path": "src.py"}',
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert result == "one\ntwo\nthree\n"


def test_read_file_supports_offset_and_limit(tmp_path: Path) -> None:
    target = tmp_path / "src.py"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = execute_tool(
        "read_file",
        '{"path": "src.py", "offset": 2, "limit": 1}',
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert result == "two\n"


def test_read_file_missing_file_is_tool_output_not_exception(tmp_path: Path) -> None:
    result = execute_tool(
        "read_file",
        '{"path": "nope.py"}',
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert result.startswith("error:")


def test_write_file_creates_nested_paths(tmp_path: Path) -> None:
    result = execute_tool(
        "write_file",
        '{"path": "nested/dir/out.txt", "content": "hello"}',
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert result.startswith("wrote ")
    assert (tmp_path / "nested" / "dir" / "out.txt").read_text(encoding="utf-8") == "hello"


def test_write_file_rejects_path_escape(tmp_path: Path) -> None:
    result = execute_tool(
        "write_file",
        '{"path": "../escape.txt", "content": "nope"}',
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert result.startswith("error:")
    assert not (tmp_path.parent / "escape.txt").exists()


def test_read_file_rejects_absolute_path_escape(tmp_path: Path) -> None:
    result = execute_tool(
        "read_file",
        '{"path": "/etc/passwd"}',
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert result.startswith("error:")


def test_run_command_runs_in_repo_root(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("hi", encoding="utf-8")
    result = execute_tool(
        "run_command",
        '{"command": "cat marker.txt"}',
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert "exit_code=0" in result
    assert "hi" in result


def test_run_command_reports_nonzero_exit(tmp_path: Path) -> None:
    result = execute_tool(
        "run_command",
        '{"command": "exit 3"}',
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert "exit_code=3" in result


def test_read_only_sandbox_blocks_mutating_tools(tmp_path: Path) -> None:
    write_result = execute_tool(
        "write_file",
        '{"path": "out.txt", "content": "nope"}',
        repo_root=tmp_path,
        sandbox_mode="read-only",
    )
    command_result = execute_tool(
        "run_command",
        '{"command": "echo hi"}',
        repo_root=tmp_path,
        sandbox_mode="read-only",
    )
    assert write_result.startswith("error: sandbox is read-only")
    assert command_result.startswith("error: sandbox is read-only")
    assert not (tmp_path / "out.txt").exists()


def test_read_only_sandbox_allows_read_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
    result = execute_tool(
        "read_file",
        '{"path": "a.txt"}',
        repo_root=tmp_path,
        sandbox_mode="read-only",
    )
    assert result == "ok"


def test_unknown_tool_returns_error_output(tmp_path: Path) -> None:
    result = execute_tool(
        "delete_everything",
        "{}",
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert result.startswith("error: unknown tool")


def test_invalid_json_arguments_return_error_output(tmp_path: Path) -> None:
    result = execute_tool(
        "read_file",
        "{not json",
        repo_root=tmp_path,
        sandbox_mode="danger-full-access",
    )
    assert result.startswith("error:")
