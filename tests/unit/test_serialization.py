"""Tests for serialization/deserialization edge cases."""

import json
from datetime import datetime
from pathlib import Path

from agentcontract.serialization import run_from_dict, run_to_dict
from agentcontract.types import AgentRun, ToolCall, Turn, TurnRole


def test_run_from_dict_handles_null_top_level_sections():
    run = run_from_dict(
        {
            "schema_version": "1.0.0",
            "source": None,
            "model": None,
            "metadata": {"scenario": "null-sections", "tags": None},
            "summary": None,
            "turns": None,
        }
    )

    assert run.metadata.scenario == "null-sections"
    assert run.metadata.tags == []
    assert run.summary.total_turns == 0
    assert run.turns == []


def test_run_from_dict_handles_null_turn_optional_sections():
    run = run_from_dict(
        {
            "turns": [
                {
                    "index": 0,
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": None,
                    "timing": None,
                    "tokens": None,
                }
            ]
        }
    )

    turn = run.turns[0]
    assert turn.content == "done"
    assert turn.tool_calls == []
    assert turn.timing is None
    assert turn.tokens is None


def test_run_from_dict_defaults_missing_turn_index():
    run = run_from_dict(
        {
            "turns": [
                {
                    "role": "assistant",
                    "content": "done",
                }
            ]
        }
    )

    turn = run.turns[0]
    assert turn.index == 0
    assert turn.role.value == "assistant"
    assert turn.content == "done"


def test_run_from_dict_coerces_non_object_tool_arguments():
    run = run_from_dict(
        {
            "turns": [
                {
                    "index": 0,
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "tc1", "function": "lookup_order", "arguments": None},
                        {"id": "tc2", "function": "lookup_order", "arguments": ["bad"]},
                        None,
                    ],
                }
            ]
        }
    )

    turn = run.turns[0]
    assert len(turn.tool_calls) == 2
    assert turn.tool_calls[0].arguments == {}
    assert turn.tool_calls[1].arguments == {}


def test_run_from_dict_coerces_nullable_and_string_numeric_fields():
    run = run_from_dict(
        {
            "model": {
                "temperature": None,
                "top_p": "0.75",
                "max_tokens": "8192",
                "seed": "7",
            },
            "summary": {
                "total_turns": None,
                "total_duration_ms": "12.5",
                "total_tokens": {"prompt": None, "completion": "9", "total": "bad"},
                "total_tool_calls": None,
                "estimated_cost_usd": None,
            },
            "turns": [
                {
                    "index": "3",
                    "role": "assistant",
                    "content": {"text": "done"},
                    "tool_calls": [
                        {"id": 1, "function": None, "arguments": {}, "duration_ms": "5.5"}
                    ],
                    "timing": {"latency_ms": "10.25", "time_to_first_token_ms": None},
                    "tokens": {"prompt": "4", "completion": None, "total": "6"},
                }
            ],
        }
    )

    assert run.model.temperature == 0.0
    assert run.model.top_p == 0.75
    assert run.model.max_tokens == 8192
    assert run.model.seed == 7

    assert run.summary.total_turns == 0
    assert run.summary.total_duration_ms == 12.5
    assert run.summary.total_tokens.prompt == 0
    assert run.summary.total_tokens.completion == 9
    assert run.summary.total_tokens.total == 0
    assert run.summary.total_tool_calls == 0
    assert run.summary.estimated_cost_usd == 0.0

    turn = run.turns[0]
    assert turn.index == 3
    assert isinstance(turn.content, str)
    assert "done" in turn.content
    assert turn.tool_calls[0].id == "1"
    assert turn.tool_calls[0].function == ""
    assert turn.tool_calls[0].duration_ms == 5.5
    assert turn.timing is not None
    assert turn.timing.latency_ms == 10.25
    assert turn.timing.time_to_first_token_ms is None
    assert turn.tokens is not None
    assert turn.tokens.prompt == 4
    assert turn.tokens.completion == 0
    assert turn.tokens.total == 6


def test_run_to_dict_coerces_non_json_tool_payloads():
    run = AgentRun(
        turns=[
            Turn(
                index=0,
                role=TurnRole.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        function="lookup_order",
                        arguments={
                            "requested_at": datetime(2026, 1, 1, 12, 0, 0),
                            "path": Path("orders/ORD-123"),
                        },
                        result={"raw_values": {3, 1, 2}},
                    )
                ],
            )
        ]
    )

    data = run_to_dict(run)
    tool_call = data["turns"][0]["tool_calls"][0]
    assert tool_call["arguments"]["requested_at"] == "2026-01-01T12:00:00"
    assert tool_call["arguments"]["path"] == "orders/ORD-123"
    assert tool_call["result"]["raw_values"] == [1, 2, 3]
    json.dumps(data)  # should not raise
