"""SDK interceptors for automatic recording from OpenAI/Anthropic clients."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from agentcontract.recorder.core import Recorder


def patch_openai(client: Any, recorder: Recorder) -> Callable[[], None]:
    """Monkey-patch an OpenAI client to record all chat completion calls.

    Returns an unpatch function.
    """
    original_create = client.chat.completions.create

    @functools.wraps(original_create)
    def recording_create(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", args[0] if args else [])

        # Record user/system messages that haven't been recorded yet
        start = time.monotonic()
        response = original_create(*args, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        # Extract tool calls from response
        choice = response.choices[0] if response.choices else None
        tool_calls_data = []
        if choice and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "function": tc.function.name,
                    "arguments": _safe_parse_json(tc.function.arguments),
                })

        # Record the assistant turn
        usage = response.usage
        recorder.add_turn(
            role="assistant",
            content=choice.message.content if choice else None,
            tool_calls=tool_calls_data or None,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

        # Update model info from response
        if response.model:
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
    """
    original_create = client.messages.create

    @functools.wraps(original_create)
    def recording_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = original_create(*args, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        # Extract content and tool calls
        content_text = ""
        tool_calls_data = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls_data.append({
                    "id": block.id,
                    "function": block.name,
                    "arguments": block.input,
                })

        usage = response.usage
        recorder.add_turn(
            role="assistant",
            content=content_text or None,
            tool_calls=tool_calls_data or None,
            latency_ms=latency_ms,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
        )

        recorder.run.model.model = response.model
        recorder.run.model.provider = "anthropic"

        return response

    client.messages.create = recording_create

    def unpatch() -> None:
        client.messages.create = original_create

    return unpatch


def _safe_parse_json(s: str) -> dict[str, Any]:
    """Parse JSON string, returning raw string in dict on failure."""
    import json

    try:
        return json.loads(s)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        return {"_raw": s}
