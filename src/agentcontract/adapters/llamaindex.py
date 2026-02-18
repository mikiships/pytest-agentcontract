"""LlamaIndex adapter for pytest-agentcontract.

Records agent trajectories from LlamaIndex AgentRunner / ReActAgent
by wrapping the chat/query methods.

Usage:
    from agentcontract.adapters.llamaindex import record_agent

    recorder = Recorder(scenario="rag-qa")
    with recorder.recording():
        unpatch = record_agent(agent, recorder)
        response = agent.chat("What's the refund policy?")
        unpatch()
        recorder.save("tests/scenarios/rag-qa.agentrun.json")
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from agentcontract.recorder.core import Recorder


def record_agent(agent: Any, recorder: Recorder) -> Callable[[], None]:
    """Wrap a LlamaIndex agent to record trajectories.

    Intercepts ``chat()``, ``achat()``, ``query()``, and ``aquery()``
    to capture the full interaction as an AgentRun.

    Returns an unpatch function that restores original methods.

    Args:
        agent: A LlamaIndex AgentRunner, ReActAgent, or similar.
        recorder: A Recorder instance to capture the trajectory.
    """
    if not isinstance(recorder, Recorder):
        raise TypeError("recorder must be a Recorder instance")

    originals: dict[str, Any] = {}
    methods = ["chat", "achat", "query", "aquery"]

    for method_name in methods:
        original = getattr(agent, method_name, None)
        if original is None:
            continue
        if not callable(original):
            raise TypeError(f"agent.{method_name} must be callable")
        originals[method_name] = original

        if method_name.startswith("a"):
            # Async variant
            @functools.wraps(original)
            async def async_wrapper(
                *args: Any,
                _orig: Any = original,
                _name: str = method_name,
                **kwargs: Any,
            ) -> Any:
                start = time.monotonic()
                result = await _orig(*args, **kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                _extract_from_response(result, recorder, latency_ms, agent)
                return result

            setattr(agent, method_name, async_wrapper)
        else:
            # Sync variant
            @functools.wraps(original)
            def sync_wrapper(
                *args: Any,
                _orig: Any = original,
                _name: str = method_name,
                **kwargs: Any,
            ) -> Any:
                start = time.monotonic()
                result = _orig(*args, **kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                _extract_from_response(result, recorder, latency_ms, agent)
                return result

            setattr(agent, method_name, sync_wrapper)

    if not originals:
        raise ValueError("agent must define at least one of: chat, achat, query, aquery")

    def unpatch() -> None:
        for name, orig in originals.items():
            setattr(agent, name, orig)

    return unpatch


def _extract_from_response(
    response: Any, recorder: Recorder, latency_ms: float, agent: Any
) -> None:
    """Extract turns from a LlamaIndex response.

    LlamaIndex agents return AgentChatResponse or Response objects.
    We also check the agent's chat_history/memory for the full trajectory.
    """
    # Try to get the response text
    response_text = None
    if hasattr(response, "response"):
        response_text = str(response.response) if response.response else None
    elif hasattr(response, "message"):
        msg = response.message
        response_text = _get_content(msg)

    # Extract tool calls from sources/source_nodes (tool outputs)
    tool_calls = _extract_tool_calls(response)

    if response_text or tool_calls:
        recorder.add_turn(
            role="assistant",
            content=response_text,
            tool_calls=tool_calls or None,
            latency_ms=latency_ms,
        )


def _extract_tool_calls(response: Any) -> list[dict[str, Any]] | None:
    """Extract tool calls from LlamaIndex response sources."""
    calls: list[dict[str, Any]] = []

    # AgentChatResponse has .sources which are ToolOutput objects
    sources = getattr(response, "sources", None)
    if sources and isinstance(sources, (list, tuple)):
        for source in sources:
            tool_name = getattr(source, "tool_name", None)
            if not tool_name:
                continue
            raw_input = getattr(source, "raw_input", {})
            raw_output = getattr(source, "raw_output", None)

            calls.append({
                "id": "",
                "function": str(tool_name),
                "arguments": raw_input if isinstance(raw_input, dict) else {},
                "result": str(raw_output) if raw_output is not None else None,
            })

    # Also check source_nodes for retrieval-based responses
    source_nodes = getattr(response, "source_nodes", None)
    if source_nodes and isinstance(source_nodes, (list, tuple)):
        for node in source_nodes:
            node_id = getattr(node, "node_id", "") or getattr(node, "id_", "")
            score = getattr(node, "score", None)
            text = ""
            inner = getattr(node, "node", None) or getattr(node, "text", None)
            if inner:
                text = getattr(inner, "text", str(inner))[:200]

            calls.append({
                "id": str(node_id) if node_id else "",
                "function": "_retrieve",
                "arguments": {"score": score} if score is not None else {},
                "result": text or None,
            })

    return calls or None


def _get_content(msg: Any) -> str | None:
    """Extract text from a LlamaIndex ChatMessage."""
    if msg is None:
        return None
    content = getattr(msg, "content", None)
    if content is None:
        return None
    if isinstance(content, str):
        return content or None
    return str(content) or None
