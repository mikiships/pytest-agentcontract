"""Example tests demonstrating pytest-agentcontract with a customer support agent.

Run these tests:
    # Record trajectories (creates cassettes)
    pytest examples/customer_support/test_support.py --ac-record -v

    # Replay from cassettes (deterministic, no external calls)
    pytest examples/customer_support/test_support.py --ac-replay -v

    # Run with contract assertions
    pytest examples/customer_support/test_support.py -v
"""

from __future__ import annotations

import pytest

from agentcontract.config import AssertionSpec, PolicySpec

# Import our example agent (relative import for running from repo root)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent import run_support_agent  # noqa: E402


@pytest.mark.agentcontract("refund-eligible")
def test_refund_happy_path(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    """Test: customer requests refund for a delivered order within return window."""
    if ac_mode == "replay" and ac_replay_engine is not None:
        # In replay mode, just verify the recorded trajectory passes contracts
        run = ac_replay_engine.recorded_run
    else:
        # Run the agent and record turns
        turns = run_support_agent("I'd like a refund for order ORD-123 please")

        for turn in turns:
            ac_recorder.add_turn(
                role=turn["role"],
                content=turn.get("content"),
                tool_calls=turn.get("tool_calls"),
            )
        run = ac_recorder.run

    # Assert the contract
    result = ac_check_contract(
        run,
        extra_assertions=[
            # Final response should mention the refund amount
            AssertionSpec(type="contains", target="final_response", value="$79.99"),
            # process_refund must have been called with correct schema
            AssertionSpec(
                type="json_schema",
                target="tool_call:process_refund:arguments",
                schema={
                    "type": "object",
                    "required": ["order_id", "amount", "method"],
                },
            ),
        ],
    )
    assert result.passed, [f.message for f in result.failures()]


@pytest.mark.agentcontract("refund-not-delivered")
def test_refund_denied_not_delivered(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    """Test: customer requests refund for an order that hasn't been delivered yet."""
    if ac_mode == "replay" and ac_replay_engine is not None:
        run = ac_replay_engine.recorded_run
    else:
        turns = run_support_agent("Refund my order ORD-456")

        for turn in turns:
            ac_recorder.add_turn(
                role=turn["role"],
                content=turn.get("content"),
                tool_calls=turn.get("tool_calls"),
            )
        run = ac_recorder.run

    # Assert: process_refund should NOT have been called
    result = ac_check_contract(
        run,
        extra_assertions=[
            AssertionSpec(type="not_called", target="tool:process_refund"),
            AssertionSpec(type="contains", target="final_response", value="isn't eligible"),
        ],
    )
    assert result.passed, [f.message for f in result.failures()]


@pytest.mark.agentcontract("refund-with-policies")
def test_refund_with_policy_enforcement(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    """Test: verify agent respects tool allowlist and confirmation policies."""
    if ac_mode == "replay" and ac_replay_engine is not None:
        run = ac_replay_engine.recorded_run
    else:
        turns = run_support_agent("I'd like a refund for order ORD-123 please")

        for turn in turns:
            ac_recorder.add_turn(
                role=turn["role"],
                content=turn.get("content"),
                tool_calls=turn.get("tool_calls"),
            )
        run = ac_recorder.run

    # Check with policies from config (or define inline)
    from agentcontract.assertions.engine import AssertionEngine

    engine = AssertionEngine()
    result = engine.check(
        run,
        policies=[
            PolicySpec(
                name="allowed-tools",
                type="tool_allowlist",
                tools=["lookup_order", "check_refund_eligibility", "process_refund"],
            ),
            PolicySpec(
                name="confirm-before-refund",
                type="requires_confirmation",
                tools=["process_refund"],
            ),
        ],
    )
    assert result.passed, [f.message for f in result.failures()]
