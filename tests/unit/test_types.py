"""Tests for core dataclass and enum types."""

import importlib

from agentcontract.types import (
    AgentRun,
    RunMetadata,
    Timing,
    TokenUsage,
    ToolCall,
    Turn,
    TurnRole,
)


def test_types_module_reloads_cleanly() -> None:
    import agentcontract.types as types_module

    reloaded = importlib.reload(types_module)

    assert reloaded.AgentRun is not None
    assert reloaded.TurnRole.ASSISTANT.value == "assistant"


def test_turn_role_values_are_stable_strings() -> None:
    assert [role.value for role in TurnRole] == ["system", "user", "assistant", "tool"]


def test_dataclass_defaults_use_independent_nested_factories() -> None:
    first = AgentRun()
    second = AgentRun()

    first.metadata.tags.append("unit")
    first.summary.total_tokens.prompt = 7
    first.turns.append(Turn(index=0, role=TurnRole.USER, content="hello"))

    assert first.sdk == "agentcontract-python"
    assert first.model.provider == ""
    assert second.metadata == RunMetadata()
    assert second.summary.total_tokens == TokenUsage()
    assert second.turns == []


def test_turn_related_defaults_are_optional_and_empty_by_default() -> None:
    tool_call = ToolCall(id="tc1", function="lookup_order", arguments={"order_id": "123"})
    turn = Turn(index=1, role=TurnRole.ASSISTANT)
    timing = Timing()

    assert tool_call.result is None
    assert tool_call.duration_ms is None
    assert turn.content is None
    assert turn.tool_calls == []
    assert turn.timing is None
    assert turn.tokens is None
    assert timing.latency_ms is None
    assert timing.time_to_first_token_ms is None


def test_agent_run_to_dict_delegates_to_serialization_module(monkeypatch) -> None:
    run = AgentRun(run_id="run-123")

    def fake_run_to_dict(arg: AgentRun) -> dict[str, str]:
        assert arg is run
        return {"run_id": "run-123"}

    monkeypatch.setattr("agentcontract.serialization.run_to_dict", fake_run_to_dict)

    assert run.to_dict() == {"run_id": "run-123"}


def test_agent_run_from_dict_delegates_to_serialization_module(monkeypatch) -> None:
    expected = AgentRun(run_id="run-456")

    def fake_run_from_dict(data: dict[str, str]) -> AgentRun:
        assert data == {"run_id": "run-456"}
        return expected

    monkeypatch.setattr("agentcontract.serialization.run_from_dict", fake_run_from_dict)

    assert AgentRun.from_dict({"run_id": "run-456"}) is expected
