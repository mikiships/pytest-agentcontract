"""SDK interceptors for automatic recording from OpenAI/Anthropic clients."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from agentcontract.recorder.core import Recorder


def patch_openai(client: Any, recorder: Recorder) -> Callable[[], None]:
    """Monkey-patch an OpenAI client to record all chat completion calls.

    Returns an unpatch function.

    NOTE: Tool call *results* are not available in the API response -- they come
    from your application code in subsequent messages.  The interceptor records
    tool call requests (function + arguments) only.  To capture results, either:
    1. Use ``recorder.add_turn()`` manually for tool-result messages, or
    2. Post-process the cassette to backfill results from your tool execution.
    """
    original_create = client.chat.completions.create

    @functools.wraps(original_create)
    def recording_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = original_create(*args, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        # Extract tool calls from response
        choices = _get_field(response, "choices", [])
        choice = choices[0] if choices else None
        message = _get_field(choice, "message")
        tool_calls_data: list[dict[str, Any]] = []
        for tc in _get_field(message, "tool_calls", []) or []:
            fn = _get_field(tc, "function")
            tool_calls_data.append(
                {
                    "id": _get_field(tc, "id", ""),
                    "function": _get_field(fn, "name", ""),
                    "arguments": _safe_parse_json(_get_field(fn, "arguments")),
                    # result is NOT available here; see docstring
                }
            )

        # Record the assistant turn
        content = _get_field(message, "content")

        usage = _get_field(response, "usage")
        recorder.add_turn(
            role="assistant",
            content=content,
            tool_calls=tool_calls_data or None,
            latency_ms=latency_ms,
            prompt_tokens=int(_get_field(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(_get_field(usage, "completion_tokens", 0) or 0),
        )

        # Update model info from response
        model_name = _get_field(response, "model")
        if model_name:
            recorder.run.model.model = model_name
            recorder.run.model.provider = "openai"

        return response

    client.chat.completions.create = recording_create

    def unpatch() -> None:
        client.chat.completions.create = original_create

    return unpatch


def patch_anthropic(client: Any, recorder: Recorder) -> Callable[[], None]:
    """Monkey-patch an Anthropic client to record all message creation calls.

    Returns an unpatch function.

    NOTE: Like the OpenAI interceptor, tool *results* are provided by your code
    in subsequent ``tool_result`` messages.  The interceptor records tool_use
    requests only.
    """
    original_create = client.messages.create

    @functools.wraps(original_create)
    def recording_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = original_create(*args, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        # Extract content and tool calls
        content_text = ""
        tool_calls_data: list[dict[str, Any]] = []
        for block in _get_field(response, "content", []) or []:
            block_type = _get_field(block, "type")
            if block_type == "text":
                content_text += str(_get_field(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls_data.append(
                    {
                        "id": _get_field(block, "id", ""),
                        "function": _get_field(block, "name", ""),
                        "arguments": _get_field(block, "input", {}),
                    }
                )

        usage = _get_field(response, "usage")
        recorder.add_turn(
            role="assistant",
            content=content_text or None,
            tool_calls=tool_calls_data or None,
            latency_ms=latency_ms,
            prompt_tokens=int(_get_field(usage, "input_tokens", 0) or 0),
            completion_tokens=int(_get_field(usage, "output_tokens", 0) or 0),
        )

        model_name = _get_field(response, "model")
        if model_name:
            recorder.run.model.model = model_name
            recorder.run.model.provider = "anthropic"

        return response

    client.messages.create = recording_create

    def unpatch() -> None:
        client.messages.create = original_create

    return unpatch


def _safe_parse_json(s: str | None) -> dict[str, Any]:
    """Parse JSON string, returning raw string in dict on failure."""
    import json

    if s is None:
        return {}
    try:
        result = json.loads(s)
        # OpenAI tool call arguments should always be a dict, but guard against
        # edge cases where the model produces a non-object JSON value.
        if not isinstance(result, dict):
            return {"_value": result}
        return result
    except (json.JSONDecodeError, TypeError):
        return {"_raw": s}


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from dict-like or object-like SDK responses."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
