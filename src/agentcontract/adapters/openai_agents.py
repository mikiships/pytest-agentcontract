"""OpenAI Agents SDK adapter for pytest-agentcontract.

Records agent trajectories from the OpenAI Agents SDK (openai-agents)
by wrapping Runner.run / Runner.run_sync.

Usage:
    from agentcontract.adapters.openai_agents import record_runner

    recorder = Recorder(scenario="triage-agent")
    with recorder.recording():
        unpatch = record_runner(recorder)
        result = Runner.run_sync(agent, "I need help with billing")
        unpatch()
        recorder.save("tests/scenarios/triage-agent.agentrun.json")
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from agentcontract.recorder.core import Recorder


def record_runner(recorder: Recorder) -> Callable[[], None]:
    """Patch the OpenAI Agents SDK Runner class to record trajectories.

    Intercepts ``Runner.run()`` and ``Runner.run_sync()`` at the class level
    to capture agent runs as AgentRun trajectories.

    Returns an unpatch function that restores original methods.

    Args:
        recorder: A Recorder instance to capture the trajectory.
    """
    if not isinstance(recorder, Recorder):
        raise TypeError("recorder must be a Recorder instance")

    try:
        from agents import Runner  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
        ) from e

    originals: dict[str, Any] = {}

    # Patch run (async)
    original_run = getattr(Runner, "run", None)
    if original_run is not None and not callable(original_run):
        raise TypeError("Runner.run must be callable")
    if original_run is not None:
        originals["run"] = original_run

        @functools.wraps(original_run)
        async def recording_run(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            result = await original_run(*args, **kwargs)
            latency_ms = (time.monotonic() - start) * 1000
            _extract_from_result(result, recorder, latency_ms)
            return result

        Runner.run = recording_run  # type: ignore[assignment]

    # Patch run_sync
    original_run_sync = getattr(Runner, "run_sync", None)
    if original_run_sync is not None and not callable(original_run_sync):
        raise TypeError("Runner.run_sync must be callable")
    if original_run_sync is not None:
        originals["run_sync"] = original_run_sync

        @functools.wraps(original_run_sync)
        def recording_run_sync(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            result = original_run_sync(*args, **kwargs)
            latency_ms = (time.monotonic() - start) * 1000
            _extract_from_result(result, recorder, latency_ms)
            return result

        Runner.run_sync = recording_run_sync  # type: ignore[assignment]

    if not originals:
        raise ValueError("Runner must define run and/or run_sync")

    def unpatch() -> None:
        for name, orig in originals.items():
            setattr(Runner, name, orig)

    return unpatch


def _extract_from_result(result: Any, recorder: Recorder, latency_ms: float) -> None:
    """Extract turns from a RunResult.

    The OpenAI Agents SDK RunResult contains:
    - result.final_output: the agent's final response
    - result.new_items: list of RunItem objects (messages, tool calls, handoffs)
    - result.raw_responses: list of ModelResponse objects
    """
    if result is None:
        return

    # Extract from new_items (preferred -- gives full trajectory)
    new_items = getattr(result, "new_items", None)
    if new_items and isinstance(new_items, (list, tuple)):
        _extract_from_items(new_items, recorder, latency_ms)
        return

    # Fallback: just record the final output
    final = getattr(result, "final_output", None)
    if final is not None:
        content = str(final) if not isinstance(final, str) else final
        recorder.add_turn(
            role="assistant",
            content=content,
            latency_ms=latency_ms,
        )


def _extract_from_items(
    items: list[Any] | tuple[Any, ...], recorder: Recorder, latency_ms: float
) -> None:
    """Extract turns from RunItem list."""
    for item in items:
        item_type = type(item).__name__

        if item_type == "MessageOutputItem":
            # Agent message output
            agent_msg = getattr(item, "raw_item", None)
            content = _extract_message_content(agent_msg)
            tool_calls = _extract_message_tool_calls(agent_msg)
            if content or tool_calls:
                recorder.add_turn(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls or None,
                    latency_ms=latency_ms,
                )

        elif item_type == "ToolCallItem":
            # Individual tool call
            raw = getattr(item, "raw_item", None)
            if raw is not None:
                name = getattr(raw, "name", "") or _get_nested(raw, "function", "name", default="")
                args = _get_tool_arguments(raw)
                if name:
                    recorder.add_turn(
                        role="assistant",
                        tool_calls=[{
                            "id": str(getattr(raw, "id", "") or getattr(raw, "call_id", "") or ""),
                            "function": str(name),
                            "arguments": args,
                        }],
                    )

        elif item_type == "ToolCallOutputItem":
            # Tool result
            output = getattr(item, "output", None)
            recorder.add_turn(
                role="tool",
                content=str(output) if output is not None else None,
            )

        elif item_type == "HandoffCallItem":
            # Agent handoff
            target = getattr(item, "target_agent", None)
            target_name = getattr(target, "name", str(target)) if target else "unknown"
            recorder.add_turn(
                role="assistant",
                content=f"[handoff to {target_name}]",
            )


def _extract_message_content(msg: Any) -> str | None:
    """Extract text content from an Agents SDK message."""
    if msg is None:
        return None

    # Check for content list (OpenAI format)
    content = getattr(msg, "content", None)
    if content is None:
        return None

    if isinstance(content, str):
        return content or None

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("output_text", "text"):
                    texts.append(str(block.get("text", "")))
            else:
                block_type = getattr(block, "type", "")
                if block_type in ("output_text", "text"):
                    texts.append(str(getattr(block, "text", "")))
        joined = "".join(texts)
        return joined or None

    return str(content) or None


def _extract_message_tool_calls(msg: Any) -> list[dict[str, Any]] | None:
    """Extract tool calls from an Agents SDK message."""
    if msg is None:
        return None

    raw_calls = getattr(msg, "tool_calls", None)
    if not raw_calls or not isinstance(raw_calls, (list, tuple)):
        return None

    calls: list[dict[str, Any]] = []
    for tc in raw_calls:
        name = getattr(tc, "name", "") or _get_nested(tc, "function", "name", default="")
        if name:
            calls.append({
                "id": str(getattr(tc, "id", "") or getattr(tc, "call_id", "") or ""),
                "function": str(name),
                "arguments": _get_tool_arguments(tc),
            })

    return calls or None


def _get_tool_arguments(tc: Any) -> dict[str, Any]:
    """Extract arguments dict from a tool call object."""
    # Try direct args/arguments
    for attr in ("args", "arguments", "input"):
        val = getattr(tc, attr, None)
        if isinstance(val, dict):
            return val
        if isinstance(val, str) and val:
            import json

            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                return {"_raw": val}

    # Try function.arguments (OpenAI chat format)
    fn = getattr(tc, "function", None)
    if fn:
        fn_args = getattr(fn, "arguments", None)
        if isinstance(fn_args, dict):
            return fn_args
        if isinstance(fn_args, str) and fn_args:
            import json

            try:
                parsed = json.loads(fn_args)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass

    return {}


def _get_nested(obj: Any, *attrs: str, default: Any = None) -> Any:
    """Safely traverse nested attributes."""
    current = obj
    for attr in attrs:
        if current is None:
            return default
        current = getattr(current, attr, None)
    return current if current is not None else default
