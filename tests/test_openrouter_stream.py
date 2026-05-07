"""Tests for AC-6: token streaming reaches GitHub Action logs.

These tests verify that ``OpenRouterStreamPrinter``:

* Emits each content delta to stdout as it arrives, *not* only at the
  end of the stream.
* Flushes stdout after every delta so the GH Actions runner forwards
  partial lines to the live log.
* Aggregates the full text into the returned ``StreamingResult``.
* Tolerates the OpenRouter SSE framing quirks: chunk boundaries that
  split mid-frame, ``: keep-alive`` comments, ``[DONE]`` sentinel,
  and ``message`` payloads on the final event.
* Honours ``stream_to_logs=False`` (the legacy ``suppress_stream``
  contract used by the schema-enforced JSON turn).
* Merges streamed tool-call deltas by index (used by the schema turn).
"""

from __future__ import annotations

import io
import json
from typing import Any

from cli.clients.openrouter_stream import (
    OpenRouterStreamPrinter,
    StreamingResult,
    _iter_sse_events,
    _merge_tool_call_deltas,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sse_frame(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _content_chunk(text: str, *, finish_reason: str | None = None) -> dict[str, Any]:
    delta: dict[str, Any] = {"content": text}
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {"choices": [choice]}


class _FlushTrackingStream(io.StringIO):
    """StringIO that records the buffer state at each flush."""

    def __init__(self) -> None:
        super().__init__()
        self.flush_snapshots: list[str] = []
        self.flush_count = 0

    def flush(self) -> None:  # type: ignore[override]
        self.flush_count += 1
        self.flush_snapshots.append(self.getvalue())
        super().flush()


# ----------------------------------------------------------------------
# Streaming behaviour
# ----------------------------------------------------------------------


def test_emits_content_deltas_incrementally() -> None:
    sink = _FlushTrackingStream()
    printer = OpenRouterStreamPrinter(stdout=sink, stream_to_logs=True)

    chunks = [
        _sse_frame(_content_chunk("Hello")),
        _sse_frame(_content_chunk(", ")),
        _sse_frame(_content_chunk("world!", finish_reason="stop")),
        b"data: [DONE]\n\n",
    ]
    result = printer.consume_bytes(chunks)

    assert result.text == "Hello, world!"
    assert result.finish_reason == "stop"

    # Each delta must appear in the buffer mid-stream — proving
    # streaming, not batch flushing at the end. We use the snapshot
    # taken on each flush call.
    assert "Hello" in sink.flush_snapshots[0]
    assert "Hello, " in sink.flush_snapshots[1]
    assert "Hello, world!" in sink.flush_snapshots[2]


def test_flushes_after_every_delta() -> None:
    sink = _FlushTrackingStream()
    printer = OpenRouterStreamPrinter(stdout=sink, stream_to_logs=True)

    chunks = [
        _sse_frame(_content_chunk("a")),
        _sse_frame(_content_chunk("b")),
        _sse_frame(_content_chunk("c", finish_reason="stop")),
        b"data: [DONE]\n\n",
    ]
    printer.consume_bytes(chunks)

    # Three content deltas + one terminating newline flush = 4 flushes.
    assert sink.flush_count >= 4, (
        "AC-6 requires per-delta flushing so partial lines reach the "
        f"GH Actions log buffer; observed flush count={sink.flush_count}"
    )


def test_appends_trailing_newline_so_gh_actions_commits_last_line() -> None:
    sink = _FlushTrackingStream()
    printer = OpenRouterStreamPrinter(stdout=sink, stream_to_logs=True)

    printer.consume_bytes([_sse_frame(_content_chunk("final-token"))])

    # Trailing newline matters: GH Actions log lines are committed on
    # newline. Without it, the last token can be held in the runner's
    # pipe until process exit.
    assert sink.getvalue().endswith("\n")
    assert "final-token\n" == sink.getvalue()


def test_suppress_stream_does_not_write_to_stdout_but_still_aggregates() -> None:
    sink = _FlushTrackingStream()
    printer = OpenRouterStreamPrinter(stdout=sink, stream_to_logs=False)

    chunks = [
        _sse_frame(_content_chunk("hidden ")),
        _sse_frame(_content_chunk("output", finish_reason="stop")),
    ]
    result = printer.consume_bytes(chunks)

    assert result.text == "hidden output"
    assert sink.getvalue() == "", (
        "stream_to_logs=False must not produce live output (this is the "
        "schema-enforced JSON turn that posts inline comments)"
    )
    # No trailing newline either — nothing was streamed.
    assert sink.flush_count == 0


# ----------------------------------------------------------------------
# Wire-format tolerance
# ----------------------------------------------------------------------


def test_handles_chunk_boundaries_mid_frame() -> None:
    """Chunk boundaries from httpx/requests are arbitrary — a single
    SSE frame may be split across multiple ``iter_bytes`` chunks."""
    sink = _FlushTrackingStream()
    printer = OpenRouterStreamPrinter(stdout=sink, stream_to_logs=True)

    full = _sse_frame(_content_chunk("split-token", finish_reason="stop"))
    # Split the bytes at an arbitrary mid-frame position.
    cut = len(full) // 2
    chunks = [full[:cut], full[cut:], b"data: [DONE]\n\n"]
    result = printer.consume_bytes(chunks)

    assert result.text == "split-token"
    assert "split-token" in sink.getvalue()


def test_ignores_keepalive_comments_and_blank_lines() -> None:
    sink = _FlushTrackingStream()
    printer = OpenRouterStreamPrinter(stdout=sink, stream_to_logs=True)

    chunks = [
        b": keep-alive\n",
        b"\n",
        _sse_frame(_content_chunk("ok", finish_reason="stop")),
        b"\n\n",
        b"data: [DONE]\n\n",
    ]
    result = printer.consume_bytes(chunks)

    assert result.text == "ok"


def test_done_sentinel_terminates_iteration_even_with_trailing_data() -> None:
    """Anything after ``data: [DONE]`` must be ignored to match
    OpenAI/OpenRouter wire semantics."""
    chunks = [
        _sse_frame(_content_chunk("x")),
        b"data: [DONE]\n\n",
        # This frame must NOT be consumed; if it were, we'd see "y".
        _sse_frame(_content_chunk("y")),
    ]
    events = list(_iter_sse_events(chunks))
    # Only the first content event should appear.
    assert len(events) == 1
    delta = events[0]["choices"][0]["delta"]
    assert delta["content"] == "x"


def test_handles_final_message_payload_without_delta() -> None:
    """Some providers send a final ``message`` instead of ``delta`` on
    the terminal event. Streaming must not lose those tokens."""
    sink = _FlushTrackingStream()
    printer = OpenRouterStreamPrinter(stdout=sink, stream_to_logs=True)

    chunks = [
        _sse_frame(_content_chunk("partial ")),
        _sse_frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "tail"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
    ]
    result = printer.consume_bytes(chunks)

    assert result.text == "partial tail"
    assert "tail" in sink.getvalue()


def test_skips_malformed_json_frames() -> None:
    """A single corrupt frame must not abort the stream — review must
    still complete with whatever valid tokens arrived."""
    chunks = [
        _sse_frame(_content_chunk("good ")),
        b"data: not-json\n\n",
        _sse_frame(_content_chunk("more", finish_reason="stop")),
    ]
    sink = io.StringIO()
    printer = OpenRouterStreamPrinter(stdout=sink, stream_to_logs=True)
    result = printer.consume_bytes(chunks)
    assert result.text == "good more"


def test_captures_usage_when_provider_emits_it() -> None:
    chunks = [
        _sse_frame(_content_chunk("hi", finish_reason="stop")),
        _sse_frame(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ),
    ]
    printer = OpenRouterStreamPrinter(stdout=io.StringIO(), stream_to_logs=False)
    result = printer.consume_bytes(chunks)
    assert isinstance(result.usage, dict)
    assert result.usage["total_tokens"] == 15


# ----------------------------------------------------------------------
# Reasoning + tool-call paths
# ----------------------------------------------------------------------


def test_reasoning_deltas_dispatch_to_callback_not_stdout() -> None:
    """Reasoning ('thinking') tokens must NOT clutter the live log by
    default. They are routed to the optional callback so callers can
    surface them at debug level 2 only."""
    sink = io.StringIO()
    captured: list[str] = []
    printer = OpenRouterStreamPrinter(
        stdout=sink,
        stream_to_logs=True,
        on_reasoning_delta=captured.append,
    )

    chunks = [
        _sse_frame({"choices": [{"index": 0, "delta": {"reasoning": "step 1 "}}]}),
        _sse_frame({"choices": [{"index": 0, "delta": {"reasoning": "step 2"}}]}),
        _sse_frame(_content_chunk("answer", finish_reason="stop")),
    ]
    result = printer.consume_bytes(chunks)

    assert result.text == "answer"
    assert captured == ["step 1 ", "step 2"]
    assert "step" not in sink.getvalue()


def test_tool_call_deltas_merge_by_index() -> None:
    """The schema-enforced JSON turn relies on tool calls arriving as
    streaming argument deltas keyed by index."""
    chunks = [
        _sse_frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_42",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_review",
                                        "arguments": '{"findings"',
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        _sse_frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ": []}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
    ]
    printer = OpenRouterStreamPrinter(stdout=io.StringIO(), stream_to_logs=False)
    result = printer.consume_bytes(chunks)

    assert result.tool_calls == {
        0: {
            "id": "call_42",
            "type": "function",
            "function": {
                "name": "submit_review",
                "arguments": '{"findings": []}',
            },
        }
    }


def test_merge_tool_call_deltas_handles_multiple_indices() -> None:
    accumulator: dict[int, dict[str, Any]] = {}
    _merge_tool_call_deltas(
        accumulator,
        [
            {"index": 0, "id": "a", "function": {"name": "f1", "arguments": "{"}},
            {"index": 1, "id": "b", "function": {"name": "f2", "arguments": "["}},
        ],
    )
    _merge_tool_call_deltas(
        accumulator,
        [
            {"index": 0, "function": {"arguments": "}"}},
            {"index": 1, "function": {"arguments": "]"}},
        ],
    )
    assert accumulator[0]["function"]["arguments"] == "{}"
    assert accumulator[1]["function"]["arguments"] == "[]"


# ----------------------------------------------------------------------
# StreamingResult dataclass
# ----------------------------------------------------------------------


def test_streaming_result_default_values() -> None:
    result = StreamingResult(text="x")
    assert result.text == "x"
    assert result.finish_reason is None
    assert result.usage is None
    assert result.tool_calls == {}


def test_empty_stream_does_not_emit_trailing_newline() -> None:
    """If the model produces no tokens, do not write a stray newline
    that would generate an empty line in the action log."""
    sink = io.StringIO()
    printer = OpenRouterStreamPrinter(stdout=sink, stream_to_logs=True)
    result = printer.consume_bytes([b"data: [DONE]\n\n"])
    assert result.text == ""
    assert sink.getvalue() == ""
