"""Tests for the Recorder."""

from pathlib import Path

from agentcontract.recorder.core import Recorder
from agentcontract.serialization import load_run
from agentcontract.types import TurnRole


def test_recorder_basic():
    """Record a simple trajectory and verify structure."""
    rec = Recorder(scenario="test-basic", tags=["unit"])

    with rec.recording():
        rec.add_turn(role="user", content="Hello")
        rec.add_turn(
            role="assistant",
            content="Let me check.",
            tool_calls=[
                {
                    "id": "tc1",
                    "function": "lookup",
                    "arguments": {"key": "abc"},
                    "result": {"found": True},
                }
            ],
            latency_ms=150.0,
            prompt_tokens=10,
            completion_tokens=20,
        )
        rec.add_turn(role="assistant", content="Done!")

    run = rec.run
    assert run.metadata.scenario == "test-basic"
    assert run.metadata.tags == ["unit"]
    assert run.summary.total_turns == 3
    assert run.summary.total_tool_calls == 1
    assert run.summary.total_tokens.prompt == 10
    assert run.summary.total_tokens.completion == 20
    assert run.summary.total_duration_ms > 0

    assert run.turns[0].role == TurnRole.USER
    assert run.turns[1].tool_calls[0].function == "lookup"
    assert run.turns[1].tool_calls[0].result == {"found": True}


def test_recorder_save_load(tmp_path: Path):
    """Record, save, and reload a trajectory."""
    rec = Recorder(scenario="save-test", model_provider="openai", model_name="gpt-4o")

    with rec.recording():
        rec.add_turn(role="user", content="Test input")
        rec.add_turn(role="assistant", content="Test output")

    path = rec.save(tmp_path / "test.agentrun.json")
    assert path.exists()

    loaded = load_run(path)
    assert loaded.metadata.scenario == "save-test"
    assert loaded.model.provider == "openai"
    assert loaded.model.model == "gpt-4o"
    assert len(loaded.turns) == 2
    assert loaded.turns[0].content == "Test input"
    assert loaded.turns[1].content == "Test output"


def test_recorder_coerces_non_object_tool_arguments():
    rec = Recorder(scenario="coerce-tool-args")

    with rec.recording():
        rec.add_turn(
            role="assistant",
            content="Running lookup.",
            tool_calls=[
                {"id": "tc1", "function": "lookup", "arguments": None},
                {"id": "tc2", "function": "lookup", "arguments": ["bad"]},
            ],
        )

    turn = rec.run.turns[0]
    assert turn.tool_calls[0].arguments == {}
    assert turn.tool_calls[1].arguments == {}


def test_recorder_ignores_non_dict_tool_calls_and_coerces_scalar_fields():
    rec = Recorder(scenario="malformed-tool-calls")

    with rec.recording():
        rec.add_turn(
            role="assistant",
            content="Running lookup.",
            tool_calls=[
                None,  # type: ignore[list-item]
                {"id": 123, "arguments": {"order_id": "abc"}},
                {"id": "tc2", "function": "lookup", "arguments": {"order_id": "xyz"}},
            ],
        )

    turn = rec.run.turns[0]
    assert len(turn.tool_calls) == 2
    assert turn.tool_calls[0].id == "123"
    assert turn.tool_calls[0].function == ""
    assert turn.tool_calls[1].function == "lookup"


def test_recorder_coerces_non_string_content_and_numeric_fields():
    rec = Recorder(scenario="coerce-input-types")

    with rec.recording():
        rec.add_turn(
            role="assistant",
            content={"message": "done"},  # type: ignore[arg-type]
            tool_calls=[
                {
                    "id": "tc1",
                    "function": "lookup",
                    "arguments": {"order_id": "123"},
                    "duration_ms": "5.25",
                }
            ],
            latency_ms="10.5",  # type: ignore[arg-type]
            prompt_tokens="3",  # type: ignore[arg-type]
            completion_tokens="2",  # type: ignore[arg-type]
        )

    turn = rec.run.turns[0]
    assert turn.content == "{'message': 'done'}"
    assert turn.timing is not None
    assert turn.timing.latency_ms == 10.5
    assert turn.tool_calls[0].duration_ms == 5.25
    assert turn.tokens is not None
    assert turn.tokens.prompt == 3
    assert turn.tokens.completion == 2
    assert turn.tokens.total == 5
