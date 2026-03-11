"""Tests for core types and package export modules."""

from __future__ import annotations

import importlib

import pytest


def test_turn_role_enum_values_are_stable() -> None:
    types_module = importlib.reload(importlib.import_module("agentcontract.types"))

    assert types_module.TurnRole.SYSTEM.value == "system"
    assert types_module.TurnRole.USER.value == "user"
    assert types_module.TurnRole.ASSISTANT.value == "assistant"
    assert types_module.TurnRole.TOOL.value == "tool"


def test_agent_run_defaults_create_independent_dataclasses() -> None:
    types_module = importlib.reload(importlib.import_module("agentcontract.types"))
    first = types_module.AgentRun()
    second = types_module.AgentRun()

    assert first.schema_version == "1.0.0"
    assert first.sdk == "agentcontract-python"
    assert isinstance(first.model, types_module.ModelInfo)
    assert first.model.max_tokens == 4096
    assert isinstance(first.metadata, types_module.RunMetadata)
    assert first.metadata.tags == []
    assert isinstance(first.summary, types_module.RunSummary)
    assert isinstance(first.summary.total_tokens, types_module.TokenUsage)
    assert first.summary.total_tokens.total == 0
    assert first.turns == []
    assert first.metadata is not second.metadata
    assert first.summary is not second.summary
    assert first.turns is not second.turns


def test_agent_run_to_dict_delegates_to_serialization(monkeypatch) -> None:
    types_module = importlib.reload(importlib.import_module("agentcontract.types"))
    import agentcontract.serialization as serialization

    run = types_module.AgentRun(run_id="run-123")
    monkeypatch.setattr(serialization, "run_to_dict", lambda value: {"run_id": value.run_id})

    assert run.to_dict() == {"run_id": "run-123"}


def test_agent_run_from_dict_delegates_to_serialization(monkeypatch) -> None:
    types_module = importlib.reload(importlib.import_module("agentcontract.types"))
    import agentcontract.serialization as serialization

    expected = types_module.AgentRun(run_id="from-dict")
    monkeypatch.setattr(serialization, "run_from_dict", lambda data: expected)

    assert types_module.AgentRun.from_dict({"run_id": "ignored"}) is expected


def test_top_level_package_lazy_exports_and_attribute_error() -> None:
    package = importlib.reload(importlib.import_module("agentcontract"))

    assert package.Recorder.__name__ == "Recorder"
    assert package.ReplayEngine.__name__ == "ReplayEngine"
    assert package.AssertionEngine.__name__ == "AssertionEngine"
    assert package.AgentContractConfig.__name__ == "AgentContractConfig"
    assert package.__all__ == ["Recorder", "ReplayEngine", "AssertionEngine", "AgentContractConfig"]

    with pytest.raises(AttributeError, match="nonexistent"):
        package.nonexistent  # noqa: B018


def test_export_modules_reexport_public_classes() -> None:
    assertions = importlib.reload(importlib.import_module("agentcontract.assertions"))
    recorder = importlib.reload(importlib.import_module("agentcontract.recorder"))
    replay = importlib.reload(importlib.import_module("agentcontract.replay"))

    assert assertions.AssertionEngine.__name__ == "AssertionEngine"
    assert recorder.Recorder.__name__ == "Recorder"
    assert replay.ReplayEngine.__name__ == "ReplayEngine"
