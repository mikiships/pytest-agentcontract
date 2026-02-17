"""Tests for the Recorder."""

import json
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
