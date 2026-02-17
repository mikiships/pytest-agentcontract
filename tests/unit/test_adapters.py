"""Tests for framework adapters (LangGraph, LlamaIndex, OpenAI Agents SDK).

All tests use mock objects -- no actual framework imports needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentcontract.recorder.core import Recorder

# ── LangGraph adapter tests ──


class TestLangGraphAdapter:
    def _make_graph(self, result: dict[str, Any]) -> MagicMock:
        graph = MagicMock()
        graph.invoke.return_value = result
        return graph

    def test_records_user_and_assistant_messages(self) -> None:
        from agentcontract.adapters.langgraph import record_graph

        @dataclass
        class FakeMessage:
            type: str
            content: str
            tool_calls: list[Any] = field(default_factory=list)

        messages = [
            FakeMessage(type="human", content="Hello"),
            FakeMessage(type="ai", content="Hi there!"),
        ]
        graph = self._make_graph({"messages": messages})
        recorder = Recorder(scenario="test-langgraph")

        unpatch = record_graph(graph, recorder)
        graph.invoke({"messages": [("user", "Hello")]})
        unpatch()

        assert len(recorder.run.turns) == 2
        assert recorder.run.turns[0].role.value == "user"
        assert recorder.run.turns[0].content == "Hello"
        assert recorder.run.turns[1].role.value == "assistant"
        assert recorder.run.turns[1].content == "Hi there!"

    def test_records_tool_calls(self) -> None:
        from agentcontract.adapters.langgraph import record_graph

        @dataclass
        class FakeToolCall:
            name: str
            args: dict[str, Any]
            id: str = ""

        @dataclass
        class FakeAIMessage:
            type: str = "ai"
            content: str = ""
            tool_calls: list[Any] = field(default_factory=list)

        msg = FakeAIMessage(
            content="Let me look that up.",
            tool_calls=[FakeToolCall(name="search", args={"query": "refund policy"}, id="tc1")],
        )
        graph = self._make_graph({"messages": [msg]})
        recorder = Recorder(scenario="test-tools")

        unpatch = record_graph(graph, recorder)
        graph.invoke({})
        unpatch()

        assert len(recorder.run.turns) == 1
        turn = recorder.run.turns[0]
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].function == "search"
        assert turn.tool_calls[0].arguments == {"query": "refund policy"}

    def test_unpatch_restores_original(self) -> None:
        from agentcontract.adapters.langgraph import record_graph

        graph = self._make_graph({"messages": []})
        original = graph.invoke
        recorder = Recorder(scenario="test-unpatch")

        unpatch = record_graph(graph, recorder)
        assert graph.invoke is not original

        unpatch()
        assert graph.invoke is original

    def test_handles_dict_messages(self) -> None:
        from agentcontract.adapters.langgraph import record_graph

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        graph = self._make_graph({"messages": messages})
        recorder = Recorder(scenario="test-dict")

        unpatch = record_graph(graph, recorder)
        graph.invoke({})
        unpatch()

        assert len(recorder.run.turns) == 2

    def test_handles_non_dict_result(self) -> None:
        from agentcontract.adapters.langgraph import record_graph

        graph = self._make_graph("not a dict")
        graph.invoke.return_value = "not a dict"
        recorder = Recorder(scenario="test-nodict")

        unpatch = record_graph(graph, recorder)
        graph.invoke({})
        unpatch()

        assert len(recorder.run.turns) == 0


# ── LlamaIndex adapter tests ──


class TestLlamaIndexAdapter:
    def test_records_response(self) -> None:
        from agentcontract.adapters.llamaindex import record_agent

        agent = MagicMock()

        @dataclass
        class FakeSource:
            tool_name: str
            raw_input: dict[str, Any]
            raw_output: str

        @dataclass
        class FakeResponse:
            response: str
            sources: list[Any] = field(default_factory=list)

        resp = FakeResponse(
            response="Your refund has been processed.",
            sources=[
                FakeSource(
                    tool_name="process_refund", raw_input={"order_id": "123"}, raw_output="ok"
                )
            ],
        )
        agent.chat = MagicMock(return_value=resp)

        recorder = Recorder(scenario="test-llamaindex")

        unpatch = record_agent(agent, recorder)
        agent.chat("Process my refund")
        unpatch()

        assert len(recorder.run.turns) == 1
        turn = recorder.run.turns[0]
        assert turn.role.value == "assistant"
        assert turn.content == "Your refund has been processed."
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].function == "process_refund"
        assert turn.tool_calls[0].arguments == {"order_id": "123"}

    def test_records_source_nodes(self) -> None:
        from agentcontract.adapters.llamaindex import record_agent

        agent = MagicMock()

        @dataclass
        class FakeNode:
            text: str

        @dataclass
        class FakeSourceNode:
            node_id: str
            score: float
            node: Any

        @dataclass
        class FakeResponse:
            response: str
            sources: list[Any] = field(default_factory=list)
            source_nodes: list[Any] = field(default_factory=list)

        resp = FakeResponse(
            response="Found it.",
            source_nodes=[
                FakeSourceNode(node_id="n1", score=0.95, node=FakeNode(text="relevant text"))
            ],
        )
        agent.chat = MagicMock(return_value=resp)
        recorder = Recorder(scenario="test-rag")

        unpatch = record_agent(agent, recorder)
        agent.chat("find info")
        unpatch()

        assert len(recorder.run.turns) == 1
        turn = recorder.run.turns[0]
        assert any(tc.function == "_retrieve" for tc in turn.tool_calls)

    def test_unpatch_restores_methods(self) -> None:
        from agentcontract.adapters.llamaindex import record_agent

        agent = MagicMock()
        original_chat = agent.chat
        recorder = Recorder(scenario="test-unpatch")

        unpatch = record_agent(agent, recorder)
        assert agent.chat is not original_chat

        unpatch()
        assert agent.chat is original_chat


# ── OpenAI Agents SDK adapter tests ──


class TestOpenAIAgentsAdapter:
    def test_extract_from_result_with_final_output(self) -> None:
        from agentcontract.adapters.openai_agents import _extract_from_result

        @dataclass
        class FakeResult:
            final_output: str
            new_items: None = None

        recorder = Recorder(scenario="test-agents-sdk")
        result = FakeResult(final_output="Here's your answer.")
        _extract_from_result(result, recorder, 150.0)

        assert len(recorder.run.turns) == 1
        assert recorder.run.turns[0].content == "Here's your answer."
        assert recorder.run.turns[0].role.value == "assistant"

    def test_extract_from_items(self) -> None:
        from agentcontract.adapters.openai_agents import _extract_from_items

        class MessageOutputItem:
            def __init__(self, content: str) -> None:
                self.raw_item = MagicMock()
                self.raw_item.content = content
                self.raw_item.tool_calls = None

        class ToolCallOutputItem:
            def __init__(self, output: str) -> None:
                self.output = output

        items = [
            MessageOutputItem("I'll look that up."),
            ToolCallOutputItem("result data"),
        ]

        recorder = Recorder(scenario="test-items")
        _extract_from_items(items, recorder, 200.0)

        assert len(recorder.run.turns) == 2
        assert recorder.run.turns[0].role.value == "assistant"
        assert recorder.run.turns[0].content == "I'll look that up."
        assert recorder.run.turns[1].role.value == "tool"
        assert recorder.run.turns[1].content == "result data"

    def test_handles_none_result(self) -> None:
        from agentcontract.adapters.openai_agents import _extract_from_result

        recorder = Recorder(scenario="test-none")
        _extract_from_result(None, recorder, 100.0)

        assert len(recorder.run.turns) == 0

    def test_handoff_item(self) -> None:
        from agentcontract.adapters.openai_agents import _extract_from_items

        class HandoffCallItem:
            def __init__(self, target_name: str) -> None:
                self.target_agent = MagicMock()
                self.target_agent.name = target_name

        items = [HandoffCallItem("billing-agent")]
        recorder = Recorder(scenario="test-handoff")
        _extract_from_items(items, recorder, 100.0)

        assert len(recorder.run.turns) == 1
        assert "[handoff to billing-agent]" in recorder.run.turns[0].content


# ── Lazy import tests ──


class TestAdaptersInit:
    def test_lazy_import_record_graph(self) -> None:
        from agentcontract.adapters import record_graph

        assert callable(record_graph)

    def test_lazy_import_record_agent(self) -> None:
        from agentcontract.adapters import record_agent

        assert callable(record_agent)

    def test_lazy_import_record_runner(self) -> None:
        from agentcontract.adapters import record_runner

        assert callable(record_runner)

    def test_lazy_import_unknown_raises(self) -> None:
        with pytest.raises(AttributeError):
            from agentcontract import adapters

            adapters.nonexistent_function  # noqa: B018
