"""Focused tests for the OpenAI Agents SDK adapter."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from agentcontract.recorder.core import Recorder


def _install_agents_module(monkeypatch, runner_cls: type[Any]) -> None:
    module = ModuleType("agents")
    module.Runner = runner_cls
    monkeypatch.setitem(sys.modules, "agents", module)


class TestRecordRunner:
    def test_record_runner_patches_and_unpatches_run_sync(self, monkeypatch) -> None:
        from agentcontract.adapters.openai_agents import record_runner

        class Runner:
            @staticmethod
            def run_sync(*args: Any, **kwargs: Any) -> Any:
                return SimpleNamespace(final_output="sync result", new_items=None)

        _install_agents_module(monkeypatch, Runner)
        recorder = Recorder(scenario="sync")
        original = Runner.run_sync

        unpatch = record_runner(recorder)
        result = Runner.run_sync("agent", "prompt")
        unpatch()

        assert result.final_output == "sync result"
        assert Runner.run_sync is original
        assert [turn.content for turn in recorder.run.turns] == ["sync result"]

    def test_record_runner_patches_and_unpatches_async_run(self, monkeypatch) -> None:
        from agentcontract.adapters.openai_agents import record_runner

        class Runner:
            @staticmethod
            async def run(*args: Any, **kwargs: Any) -> Any:
                return SimpleNamespace(final_output="async result", new_items=None)

        _install_agents_module(monkeypatch, Runner)
        recorder = Recorder(scenario="async")
        original = Runner.run
        unpatch = record_runner(recorder)

        result = asyncio.run(Runner.run("agent", "prompt"))
        unpatch()

        assert result.final_output == "async result"
        assert Runner.run is original
        assert [turn.content for turn in recorder.run.turns] == ["async result"]

    def test_record_runner_rejects_non_recorder_instances(self) -> None:
        from agentcontract.adapters.openai_agents import record_runner

        with pytest.raises(TypeError, match="Recorder instance"):
            record_runner(object())  # type: ignore[arg-type]

    def test_record_runner_reports_missing_dependency(self, monkeypatch) -> None:
        from agentcontract.adapters.openai_agents import record_runner

        monkeypatch.setitem(sys.modules, "agents", None)

        with pytest.raises(ImportError, match="OpenAI Agents SDK not installed"):
            record_runner(Recorder(scenario="missing"))

    def test_record_runner_rejects_non_callable_runner_methods(self, monkeypatch) -> None:
        from agentcontract.adapters.openai_agents import record_runner

        class Runner:
            run = "not-callable"

        _install_agents_module(monkeypatch, Runner)

        with pytest.raises(TypeError, match="Runner.run must be callable"):
            record_runner(Recorder(scenario="bad-runner"))

    def test_record_runner_requires_at_least_one_runner_method(self, monkeypatch) -> None:
        from agentcontract.adapters.openai_agents import record_runner

        class Runner:
            pass

        _install_agents_module(monkeypatch, Runner)

        with pytest.raises(ValueError, match="run and/or run_sync"):
            record_runner(Recorder(scenario="empty-runner"))


class TestExtractionHelpers:
    def test_extract_message_content_handles_string_list_and_object_blocks(self) -> None:
        from agentcontract.adapters.openai_agents import _extract_message_content

        object_block = SimpleNamespace(type="text", text=" from object")
        dict_blocks = [{"type": "output_text", "text": "hello"}, {"type": "ignored", "text": "x"}]

        assert _extract_message_content(SimpleNamespace(content="plain")) == "plain"
        assert _extract_message_content(
            SimpleNamespace(content=dict_blocks + [object_block])
        ) == "hello from object"
        assert _extract_message_content(SimpleNamespace(content=123)) == "123"
        assert _extract_message_content(SimpleNamespace(content=None)) is None

    def test_extract_message_tool_calls_reads_direct_and_nested_fields(self) -> None:
        from agentcontract.adapters.openai_agents import _extract_message_tool_calls

        direct = SimpleNamespace(id="call-1", name="lookup", arguments='{"id": 1}')
        nested = SimpleNamespace(
            call_id="call-2",
            function=SimpleNamespace(name="refund", arguments={"approved": True}),
        )

        calls = _extract_message_tool_calls(SimpleNamespace(tool_calls=[direct, nested]))

        assert calls == [
            {"id": "call-1", "function": "lookup", "arguments": {"id": 1}},
            {"id": "call-2", "function": "refund", "arguments": {"approved": True}},
        ]

    def test_get_tool_arguments_parses_json_and_falls_back_to_raw_values(self) -> None:
        from agentcontract.adapters.openai_agents import _get_tool_arguments

        assert _get_tool_arguments(SimpleNamespace(args={"a": 1})) == {"a": 1}
        assert _get_tool_arguments(SimpleNamespace(arguments='{"a": 1}')) == {"a": 1}
        assert _get_tool_arguments(SimpleNamespace(arguments="not-json")) == {"_raw": "not-json"}
        assert _get_tool_arguments(
            SimpleNamespace(function=SimpleNamespace(arguments='{"b": 2}'))
        ) == {"b": 2}
        assert _get_tool_arguments(
            SimpleNamespace(function=SimpleNamespace(arguments="still-bad"))
        ) == {}

    def test_get_nested_returns_default_for_missing_attributes(self) -> None:
        from agentcontract.adapters.openai_agents import _get_nested

        obj = SimpleNamespace(child=SimpleNamespace(value=5))

        assert _get_nested(obj, "child", "value") == 5
        assert _get_nested(obj, "child", "missing", default="fallback") == "fallback"
        assert _get_nested(None, "child", default="fallback") == "fallback"

    def test_extract_from_result_prefers_items_and_falls_back_to_final_output(self) -> None:
        from agentcontract.adapters.openai_agents import _extract_from_result

        class MessageOutputItem:
            def __init__(self) -> None:
                self.raw_item = SimpleNamespace(content="from items", tool_calls=None)

        recorder_with_items = Recorder(scenario="items")
        _extract_from_result(
            SimpleNamespace(new_items=(MessageOutputItem(),), final_output="ignored"),
            recorder_with_items,
            12.5,
        )

        recorder_with_final = Recorder(scenario="final")
        _extract_from_result(
            SimpleNamespace(new_items=[], final_output=SimpleNamespace(text="fallback")),
            recorder_with_final,
            7.0,
        )

        assert [turn.content for turn in recorder_with_items.run.turns] == ["from items"]
        assert [turn.content for turn in recorder_with_final.run.turns] == [
            "namespace(text='fallback')"
        ]

    def test_extract_from_items_handles_message_tool_call_output_and_handoff_variants(self) -> None:
        from agentcontract.adapters.openai_agents import _extract_from_items

        class MessageOutputItem:
            def __init__(self) -> None:
                self.raw_item = SimpleNamespace(
                    content=[{"type": "text", "text": "Checking"}],
                    tool_calls=[
                        SimpleNamespace(
                            id="tc-1",
                            function=SimpleNamespace(name="lookup", arguments='{"order": "1"}'),
                        )
                    ],
                )

        class ToolCallItem:
            def __init__(self) -> None:
                self.raw_item = SimpleNamespace(
                    call_id="tc-2",
                    function=SimpleNamespace(name="refund", arguments='{"approved": true}'),
                )

        class ToolCallOutputItem:
            def __init__(self, output: Any) -> None:
                self.output = output

        class HandoffCallItem:
            def __init__(self, target_agent: Any) -> None:
                self.target_agent = target_agent

        recorder = Recorder(scenario="mixed")
        _extract_from_items(
            [
                MessageOutputItem(),
                ToolCallItem(),
                ToolCallOutputItem(None),
                HandoffCallItem(None),
            ],
            recorder,
            25.0,
        )

        turns = recorder.run.turns
        assert turns[0].content == "Checking"
        assert turns[0].tool_calls[0].function == "lookup"
        assert turns[1].tool_calls[0].arguments == {"approved": True}
        assert turns[2].role.value == "tool"
        assert turns[2].content is None
        assert turns[3].content == "[handoff to unknown]"
