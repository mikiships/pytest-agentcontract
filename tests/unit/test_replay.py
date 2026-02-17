"""Tests for the ReplayEngine."""

import pytest

from agentcontract.replay.engine import ReplayEngine, ToolStubArgumentsMismatch, ToolStubExhausted
from agentcontract.types import AgentRun, RunMetadata, ToolCall, Turn, TurnRole


def _make_run() -> AgentRun:
    """Create a simple test run with tool calls."""
    return AgentRun(
        metadata=RunMetadata(scenario="test"),
        turns=[
            Turn(index=0, role=TurnRole.USER, content="Check order 123"),
            Turn(
                index=1,
                role=TurnRole.ASSISTANT,
                content="Looking it up.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        function="lookup_order",
                        arguments={"order_id": "123"},
                        result={"status": "delivered", "total": 49.99},
                    )
                ],
            ),
            Turn(
                index=2,
                role=TurnRole.ASSISTANT,
                content="Processing refund.",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        function="process_refund",
                        arguments={"order_id": "123", "amount": 49.99},
                        result={"success": True, "refund_id": "ref_001"},
                    )
                ],
            ),
            Turn(index=3, role=TurnRole.ASSISTANT, content="Refund complete!"),
        ],
    )


def test_tool_stub_returns_recorded_results():
    run = _make_run()
    engine = ReplayEngine(run)
    stub = engine.tool_stub

    result1 = stub.get_result("lookup_order")
    assert result1 == {"status": "delivered", "total": 49.99}

    result2 = stub.get_result("process_refund")
    assert result2 == {"success": True, "refund_id": "ref_001"}


def test_tool_stub_exhausted():
    run = _make_run()
    engine = ReplayEngine(run)
    stub = engine.tool_stub

    stub.get_result("lookup_order")  # consume the only result

    with pytest.raises(ToolStubExhausted):
        stub.get_result("lookup_order")  # no more results


def test_tool_stub_has_results():
    run = _make_run()
    engine = ReplayEngine(run)
    stub = engine.tool_stub

    assert stub.has_results("lookup_order")
    assert stub.has_results("process_refund")
    assert not stub.has_results("nonexistent_tool")

    stub.get_result("lookup_order")
    assert not stub.has_results("lookup_order")


def test_tool_stub_validates_arguments_in_recorded_order():
    run = AgentRun(
        metadata=RunMetadata(scenario="ordered-args"),
        turns=[
            Turn(
                index=0,
                role=TurnRole.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        function="lookup_order",
                        arguments={"order_id": "123"},
                        result={"order_id": "123"},
                    ),
                    ToolCall(
                        id="tc2",
                        function="lookup_order",
                        arguments={"order_id": "456"},
                        result={"order_id": "456"},
                    ),
                ],
            )
        ],
    )
    engine = ReplayEngine(run)
    stub = engine.tool_stub

    with pytest.raises(ToolStubArgumentsMismatch):
        stub.get_result("lookup_order", {"order_id": "456"})

    # Mismatch should not consume the recorded call.
    assert stub.get_result("lookup_order", {"order_id": "123"}) == {"order_id": "123"}


def test_replay_finish_matching():
    run = _make_run()
    engine = ReplayEngine(run)

    # Simulate exact same turns
    result = engine.finish(actual_turns=run.turns)
    assert result.ok
    assert result.matched_tools == 2
    assert result.mismatched_tools == 0


def test_replay_finish_mismatch():
    run = _make_run()
    engine = ReplayEngine(run)

    # Simulate turns with wrong tool function
    bad_turns = [
        Turn(index=0, role=TurnRole.USER, content="Check order 123"),
        Turn(
            index=1,
            role=TurnRole.ASSISTANT,
            content="Looking it up.",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    function="wrong_tool",  # different!
                    arguments={"order_id": "123"},
                    result={"status": "delivered"},
                )
            ],
        ),
        Turn(
            index=2,
            role=TurnRole.ASSISTANT,
            content="Processing.",
            tool_calls=[
                ToolCall(
                    id="tc2",
                    function="process_refund",
                    arguments={"order_id": "123", "amount": 49.99},
                    result={"success": True},
                )
            ],
        ),
        Turn(index=3, role=TurnRole.ASSISTANT, content="Done"),
    ]

    result = engine.finish(actual_turns=bad_turns)
    assert not result.ok
    assert result.mismatched_tools >= 1
    assert any("wrong_tool" in e for e in result.errors)


def test_replay_finish_counts_extra_and_missing_tools():
    run = _make_run()
    engine = ReplayEngine(run)

    # Turn 1 has one extra tool call and we also drop the final two recorded turns.
    actual_turns = [
        run.turns[0],
        Turn(
            index=1,
            role=TurnRole.ASSISTANT,
            content="Looking it up.",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    function="lookup_order",
                    arguments={"order_id": "123"},
                    result={"status": "delivered", "total": 49.99},
                ),
                ToolCall(
                    id="tc_extra",
                    function="lookup_order",
                    arguments={"order_id": "123"},
                    result={"status": "delivered", "total": 49.99},
                ),
            ],
        ),
    ]

    result = engine.finish(actual_turns=actual_turns)

    assert not result.ok
    assert result.extra_tools == 1
    assert result.missing_tools == 1
    assert result.mismatched_tools >= 2
