"""Tests for uncovered public functions."""

import pytest

import agentcontract
import agentcontract.serialization as serialization
from agentcontract.types import AgentRun


def test_package_getattr_lazy_loads_public_api_and_rejects_unknown_name() -> None:
    recorder_cls = agentcontract.Recorder

    assert recorder_cls.__name__ == "Recorder"

    with pytest.raises(AttributeError, match="does_not_exist"):
        agentcontract.__getattr__("does_not_exist")


def test_agent_run_to_dict_delegates_to_serializer_and_bubbles_serializer_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = AgentRun(run_id="run-123")
    calls: list[AgentRun] = []

    def fake_run_to_dict(arg: AgentRun) -> dict[str, str]:
        calls.append(arg)
        return {"run_id": arg.run_id}

    def raising_run_to_dict(arg: AgentRun) -> dict[str, str]:
        raise RuntimeError(f"cannot serialize {arg.run_id}")

    monkeypatch.setattr(serialization, "run_to_dict", fake_run_to_dict)
    assert run.to_dict() == {"run_id": "run-123"}
    assert calls == [run]

    monkeypatch.setattr(serialization, "run_to_dict", raising_run_to_dict)
    with pytest.raises(RuntimeError, match="cannot serialize run-123"):
        run.to_dict()


def test_agent_run_from_dict_delegates_to_serializer_and_bubbles_serializer_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = {"run_id": "run-456"}
    expected = AgentRun(run_id="run-456")
    calls: list[dict[str, str]] = []

    def fake_run_from_dict(arg: dict[str, str]) -> AgentRun:
        calls.append(arg)
        return expected

    def raising_run_from_dict(arg: dict[str, str]) -> AgentRun:
        raise ValueError(f"bad payload: {arg['run_id']}")

    monkeypatch.setattr(serialization, "run_from_dict", fake_run_from_dict)
    assert AgentRun.from_dict(data) is expected
    assert calls == [data]

    monkeypatch.setattr(serialization, "run_from_dict", raising_run_from_dict)
    with pytest.raises(ValueError, match="bad payload: run-456"):
        AgentRun.from_dict(data)
