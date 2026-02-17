"""Tests for the AssertionEngine."""

from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import AssertionSpec, PolicySpec
from agentcontract.types import AgentRun, RunMetadata, ToolCall, Turn, TurnRole


def _make_run() -> AgentRun:
    return AgentRun(
        metadata=RunMetadata(scenario="test-assertions"),
        turns=[
            Turn(index=0, role=TurnRole.USER, content="I want a refund"),
            Turn(
                index=1,
                role=TurnRole.ASSISTANT,
                content="Let me look up your order.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        function="lookup_order",
                        arguments={"order_id": "123"},
                        result={"status": "delivered", "total": 49.99},
                    )
                ],
            ),
            Turn(index=2, role=TurnRole.USER, content="Yes, please process it"),
            Turn(
                index=3,
                role=TurnRole.ASSISTANT,
                content="Your refund of $49.99 has been processed.",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        function="process_refund",
                        arguments={"order_id": "123", "amount": 49.99, "method": "original"},
                        result={"success": True, "refund_id": "ref_001"},
                    )
                ],
            ),
        ],
    )


class TestContains:
    def test_pass(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[AssertionSpec(type="contains", target="final_response", value="refund")],
        )
        assert result.passed

    def test_fail(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[AssertionSpec(type="contains", target="final_response", value="denied")],
        )
        assert not result.passed


class TestExact:
    def test_pass(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[
                AssertionSpec(
                    type="exact",
                    target="final_response",
                    value="Your refund of $49.99 has been processed.",
                )
            ],
        )
        assert result.passed


class TestRegex:
    def test_pass(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[
                AssertionSpec(type="regex", target="final_response", value=r"\$\d+\.\d{2}")
            ],
        )
        assert result.passed


class TestJsonSchema:
    def test_pass(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[
                AssertionSpec(
                    type="json_schema",
                    target="tool_call:process_refund:arguments",
                    schema={
                        "type": "object",
                        "required": ["order_id", "amount", "method"],
                    },
                )
            ],
        )
        assert result.passed

    def test_fail(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[
                AssertionSpec(
                    type="json_schema",
                    target="tool_call:process_refund:arguments",
                    schema={
                        "type": "object",
                        "required": ["order_id", "amount", "method", "reason"],  # reason missing
                    },
                )
            ],
        )
        assert not result.passed


class TestNotCalled:
    def test_pass(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[AssertionSpec(type="not_called", target="tool:delete_account")],
        )
        assert result.passed

    def test_fail(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[AssertionSpec(type="not_called", target="tool:process_refund")],
        )
        assert not result.passed


class TestCalledWith:
    def test_pass(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[
                AssertionSpec(
                    type="called_with",
                    target="tool:lookup_order",
                    schema={"order_id": "123"},
                )
            ],
        )
        assert result.passed


class TestToolAllowlistPolicy:
    def test_pass(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            policies=[
                PolicySpec(
                    name="allowed-tools",
                    type="tool_allowlist",
                    tools=["lookup_order", "process_refund"],
                )
            ],
        )
        assert result.passed

    def test_fail(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            policies=[
                PolicySpec(
                    name="allowed-tools",
                    type="tool_allowlist",
                    tools=["lookup_order"],  # process_refund not allowed
                )
            ],
        )
        assert not result.passed


class TestRequiresConfirmationPolicy:
    def test_pass(self):
        """process_refund at turn 3 is preceded by user turn 2."""
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            policies=[
                PolicySpec(
                    name="confirm-before-refund",
                    type="requires_confirmation",
                    tools=["process_refund"],
                )
            ],
        )
        assert result.passed

    def test_fail(self):
        """Make a run where process_refund is NOT preceded by user turn."""
        run = AgentRun(
            metadata=RunMetadata(scenario="no-confirm"),
            turns=[
                Turn(index=0, role=TurnRole.USER, content="Refund me"),
                Turn(
                    index=1,
                    role=TurnRole.ASSISTANT,
                    content="Looking up.",
                    tool_calls=[
                        ToolCall(id="tc1", function="lookup_order", arguments={"order_id": "123"})
                    ],
                ),
                Turn(
                    index=2,
                    role=TurnRole.ASSISTANT,  # no user confirmation before this!
                    content="Refunding.",
                    tool_calls=[
                        ToolCall(
                            id="tc2",
                            function="process_refund",
                            arguments={"order_id": "123", "amount": 49.99},
                        )
                    ],
                ),
            ],
        )
        engine = AssertionEngine()
        result = engine.check(
            run,
            policies=[
                PolicySpec(
                    name="confirm-before-refund",
                    type="requires_confirmation",
                    tools=["process_refund"],
                )
            ],
        )
        assert not result.passed
