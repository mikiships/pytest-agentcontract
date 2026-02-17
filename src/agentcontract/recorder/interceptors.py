"""SDK interceptors for automatic recording from OpenAI/Anthropic clients."""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Awaitable, Callable
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

        if inspect.isawaitable(response):
            return _record_openai_async(response, recorder, start)

        latency_ms = (time.monotonic() - start) * 1000
        _record_openai_response(response, recorder, latency_ms)
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

        if inspect.isawaitable(response):
            return _record_anthropic_async(response, recorder, start)

        latency_ms = (time.monotonic() - start) * 1000
        _record_anthropic_response(response, recorder, latency_ms)
        return response

    client.messages.create = recording_create

    def unpatch() -> None:
        client.messages.create = original_create

    return unpatch


async def _record_openai_async(
    response: Awaitable[Any], recorder: Recorder, start_time: float
) -> Any:
    resolved = await response
    latency_ms = (time.monotonic() - start_time) * 1000
    _record_openai_response(resolved, recorder, latency_ms)
    return resolved


def _record_openai_response(response: Any, recorder: Recorder, latency_ms: float) -> None:
    model_name = _get_field(response, "model")
    if model_name:
        recorder.run.model.model = model_name
        recorder.run.model.provider = "openai"

    # Extract tool calls from response
    choices = _get_field(response, "choices", [])
    if not isinstance(choices, (list, tuple)) or not choices:
        # Unsupported response shape (for example streaming objects); skip to avoid
        # recording a bogus empty assistant turn.
        return

    choice = choices[0] if choices else None
    message = _get_field(choice, "message")
    if message is None:
        return

    raw_tool_calls = _get_field(message, "tool_calls", []) or []
    if not isinstance(raw_tool_calls, (list, tuple)):
        raw_tool_calls = []

    tool_calls_data: list[dict[str, Any]] = []
    for tc in raw_tool_calls:
        fn = _get_field(tc, "function")
        function_name = _get_field(fn, "name", "")
        if not function_name:
            continue

        tool_calls_data.append(
            {
                "id": _get_field(tc, "id", ""),
                "function": function_name,
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
        prompt_tokens=_coerce_int(_get_field(usage, "prompt_tokens", 0), 0),
        completion_tokens=_coerce_int(_get_field(usage, "completion_tokens", 0), 0),
    )


async def _record_anthropic_async(
    response: Awaitable[Any], recorder: Recorder, start_time: float
) -> Any:
    resolved = await response
    latency_ms = (time.monotonic() - start_time) * 1000
    _record_anthropic_response(resolved, recorder, latency_ms)
    return resolved


def _record_anthropic_response(response: Any, recorder: Recorder, latency_ms: float) -> None:
    model_name = _get_field(response, "model")
    if model_name:
        recorder.run.model.model = model_name
        recorder.run.model.provider = "anthropic"

    # Extract content and tool calls
    blocks = _get_field(response, "content", []) or []
    if not isinstance(blocks, (list, tuple)) or not blocks:
        return

    content_text = ""
    tool_calls_data: list[dict[str, Any]] = []
    saw_replayable_content = False
    for block in blocks:
        block_type = _get_field(block, "type")
        if block_type == "text":
            saw_replayable_content = True
            content_text += str(_get_field(block, "text", ""))
        elif block_type == "tool_use":
            saw_replayable_content = True
            function_name = _get_field(block, "name", "")
            if not function_name:
                continue
            tool_calls_data.append(
                {
                    "id": _get_field(block, "id", ""),
                    "function": function_name,
                    "arguments": _coerce_tool_arguments(_get_field(block, "input")),
                }
            )

    if not saw_replayable_content:
        return

    usage = _get_field(response, "usage")
    recorder.add_turn(
        role="assistant",
        content=content_text or None,
        tool_calls=tool_calls_data or None,
        latency_ms=latency_ms,
        prompt_tokens=_coerce_int(_get_field(usage, "input_tokens", 0), 0),
        completion_tokens=_coerce_int(_get_field(usage, "output_tokens", 0), 0),
    )


def _safe_parse_json(value: Any) -> dict[str, Any]:
    """Parse tool arguments payloads to dicts."""
    import json

    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        result = json.loads(value)
        # OpenAI tool call arguments should always be a dict, but guard against
        # edge cases where the model produces a non-object JSON value.
        if not isinstance(result, dict):
            return {"_value": result}
        return result
    except (json.JSONDecodeError, TypeError):
        return {"_raw": value}


def _coerce_tool_arguments(value: Any) -> dict[str, Any]:
    """Normalize anthropic tool input payloads to dicts."""
    if isinstance(value, dict):
        return value
    return {}


def _coerce_int(value: Any, default: int = 0) -> int:
    """Normalize numeric usage fields to integers without raising."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from dict-like or object-like SDK responses."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
