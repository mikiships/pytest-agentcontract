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
        choice = response.choices[0] if response.choices else None
        tool_calls_data: list[dict[str, Any]] = []
        if choice and choice.message and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls_data.append(
                    {
                        "id": tc.id,
                        "function": tc.function.name,
                        "arguments": _safe_parse_json(tc.function.arguments),
                        # result is NOT available here; see docstring
                    }
                )

        # Record the assistant turn
        content = None
        if choice and choice.message:
            content = choice.message.content

        usage = response.usage
        recorder.add_turn(
            role="assistant",
            content=content,
            tool_calls=tool_calls_data or None,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )

        # Update model info from response
        if hasattr(response, "model") and response.model:
            recorder.run.model.model = response.model
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
        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_text += getattr(block, "text", "")
            elif block_type == "tool_use":
                tool_calls_data.append(
                    {
                        "id": getattr(block, "id", ""),
                        "function": getattr(block, "name", ""),
                        "arguments": getattr(block, "input", {}),
                    }
                )

        usage = getattr(response, "usage", None)
        recorder.add_turn(
            role="assistant",
            content=content_text or None,
            tool_calls=tool_calls_data or None,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

        if hasattr(response, "model") and response.model:
            recorder.run.model.model = response.model
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
