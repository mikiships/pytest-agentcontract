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

    def test_final_response_can_be_empty_string(self):
        engine = AssertionEngine()
        run = AgentRun(
            metadata=RunMetadata(scenario="empty-final-response"),
            turns=[
                Turn(index=0, role=TurnRole.USER, content="Hi"),
                Turn(index=1, role=TurnRole.ASSISTANT, content="Interim response"),
                Turn(index=2, role=TurnRole.ASSISTANT, content=""),
            ],
        )
        result = engine.check(
            run,
            assertions=[AssertionSpec(type="exact", target="final_response", value="")],
        )
        assert result.passed

    def test_fail_when_value_missing(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[AssertionSpec(type="exact", target="final_response")],
        )
        assert not result.passed
        assert result.results[0].message != ""


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

    def test_fail_when_schema_missing(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[AssertionSpec(type="called_with", target="tool:lookup_order")],
        )
        assert not result.passed

    def test_handles_non_dict_recorded_arguments(self):
        run = AgentRun(
            metadata=RunMetadata(scenario="called-with-non-dict-args"),
            turns=[
                Turn(
                    index=0,
                    role=TurnRole.ASSISTANT,
                    tool_calls=[
                        ToolCall(id="tc1", function="lookup_order", arguments=None),  # type: ignore[arg-type]
                    ],
                )
            ],
        )
        engine = AssertionEngine()
        result = engine.check(
            run,
            assertions=[
                AssertionSpec(
                    type="called_with",
                    target="tool:lookup_order",
                    schema={"order_id": "123"},
                )
            ],
        )
        assert not result.passed
        assert result.results[0].message != ""


class TestCalledCount:
    def test_pass(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[AssertionSpec(type="called_count", target="tool:lookup_order", value="1")],
        )
        assert result.passed

    def test_fail_when_value_missing(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[AssertionSpec(type="called_count", target="tool:lookup_order")],
        )
        assert not result.passed
        assert "requires an integer" in result.results[0].message

    def test_fail_when_value_is_not_integer(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            assertions=[
                AssertionSpec(type="called_count", target="tool:lookup_order", value="abc")
            ],
        )
        assert not result.passed
        assert "expects an integer" in result.results[0].message


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


class TestPolicyErrors:
    def test_unknown_policy_type_fails_closed(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            policies=[PolicySpec(name="unknown", type="not_a_real_policy")],
        )
        assert not result.passed
        assert result.results[0].message == "Unknown policy type: not_a_real_policy"

    def test_policy_errors_are_reported_without_raising(self):
        engine = AssertionEngine()
        result = engine.check(
            _make_run(),
            policies=[
                PolicySpec(
                    name="broken-allowlist",
                    type="tool_allowlist",
                    tools=None,  # type: ignore[arg-type]
                )
            ],
        )
        assert not result.passed
        assert result.results[0].message.startswith("Policy error:")
