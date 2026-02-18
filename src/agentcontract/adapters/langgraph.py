"""LangGraph adapter for pytest-agentcontract.

Records agent trajectories from LangGraph graph executions by wrapping
the graph's stream/invoke methods.

Usage:
    from agentcontract.adapters.langgraph import record_graph

    recorder = Recorder(scenario="customer-support")
    with recorder.recording():
        unpatch = record_graph(graph, recorder)
        result = graph.invoke({"messages": [("user", "I need a refund")]})
        unpatch()
        recorder.save("tests/scenarios/customer-support.agentrun.json")
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from agentcontract.recorder.core import Recorder


def record_graph(graph: Any, recorder: Recorder) -> Callable[[], None]:
    """Wrap a LangGraph CompiledGraph to record trajectories.

    Intercepts ``invoke()`` and ``ainvoke()`` to capture agent turns,
    tool calls, and message flow as an AgentRun.

    Returns an unpatch function that restores original methods.

    Args:
        graph: A LangGraph CompiledGraph (from ``graph.compile()``).
        recorder: A Recorder instance to capture the trajectory.
    """
    if not isinstance(recorder, Recorder):
        raise TypeError("recorder must be a Recorder instance")

    original_invoke = getattr(graph, "invoke", None)
    original_ainvoke = getattr(graph, "ainvoke", None)
    if original_invoke is not None and not callable(original_invoke):
        raise TypeError("graph.invoke must be callable")
    if original_ainvoke is not None and not callable(original_ainvoke):
        raise TypeError("graph.ainvoke must be callable")
    if original_invoke is None and original_ainvoke is None:
        raise ValueError("graph must define invoke and/or ainvoke")

    if original_invoke is not None:

        @functools.wraps(original_invoke)
        def recording_invoke(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            result = original_invoke(*args, **kwargs)
            latency_ms = (time.monotonic() - start) * 1000
            _extract_turns(result, recorder, latency_ms)
            return result

        graph.invoke = recording_invoke

    if original_ainvoke is not None:

        @functools.wraps(original_ainvoke)
        async def recording_ainvoke(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            result = await original_ainvoke(*args, **kwargs)
            latency_ms = (time.monotonic() - start) * 1000
            _extract_turns(result, recorder, latency_ms)
            return result

        graph.ainvoke = recording_ainvoke

    def unpatch() -> None:
        if original_invoke is not None:
            graph.invoke = original_invoke
        if original_ainvoke is not None:
            graph.ainvoke = original_ainvoke

    return unpatch


def _extract_turns(result: Any, recorder: Recorder, latency_ms: float) -> None:
    """Extract turns from a LangGraph result dict.

    LangGraph returns a state dict. The standard ``messages`` key contains
    the conversation as a list of LangChain message objects.
    """
    if not isinstance(result, dict):
        return

    messages = result.get("messages", [])
    if not isinstance(messages, (list, tuple)):
        return

    for msg in messages:
        role = _get_role(msg)
        content = _get_content(msg)
        tool_calls = _get_tool_calls(msg)

        if role in ("user", "assistant", "system", "tool"):
            recorder.add_turn(
                role=role,
                content=content,
                tool_calls=tool_calls or None,
                latency_ms=latency_ms if role == "assistant" else None,
            )


def _get_role(msg: Any) -> str:
    """Extract role from a LangChain message object or dict."""
    if isinstance(msg, dict):
        return str(msg.get("role", msg.get("type", "")))

    # LangChain message classes: HumanMessage, AIMessage, SystemMessage, ToolMessage
    type_attr = getattr(msg, "type", None)
    if type_attr:
        role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
        return role_map.get(str(type_attr), str(type_attr))

    return ""


def _get_content(msg: Any) -> str | None:
    """Extract text content from a message."""
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

    if content is None:
        return None
    if isinstance(content, str):
        return content or None
    # LangChain can return list of content blocks
    if isinstance(content, list):
        texts = [str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content]
        joined = "".join(texts)
        return joined or None
    return str(content) or None


def _get_tool_calls(msg: Any) -> list[dict[str, Any]] | None:
    """Extract tool calls from a LangChain AIMessage."""
    raw = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", None)

    if not raw or not isinstance(raw, (list, tuple)):
        return None

    calls: list[dict[str, Any]] = []
    for tc in raw:
        if isinstance(tc, dict):
            name = tc.get("name", "")
            args = tc.get("args", tc.get("arguments", {}))
            call_id = tc.get("id", "")
        else:
            name = getattr(tc, "name", "")
            args = getattr(tc, "args", getattr(tc, "arguments", {}))
            call_id = getattr(tc, "id", "")

        if name:
            calls.append({
                "id": str(call_id) if call_id else "",
                "function": str(name),
                "arguments": args if isinstance(args, dict) else {},
            })

    return calls or None
