"""Tests for cassette PII scanning."""

from agentcontract.pii import scan_run_for_pii
from agentcontract.types import (
    AgentRun,
    ModelInfo,
    RunMetadata,
    RunSummary,
    ToolCall,
    Turn,
    TurnRole,
)


def test_scan_run_for_pii_detects_nested_turn_and_tool_payloads() -> None:
    run = _build_run(
        turns=[
            Turn(index=0, role=TurnRole.USER, content="Reach me at alice@example.com"),
            Turn(
                index=1,
                role=TurnRole.ASSISTANT,
                content="Calling the customer at (212) 555-0188",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        function="lookup_customer",
                        arguments={"profile": {"ssn": "123-45-6789"}},
                        result={"cards": [{"number": "4111-1111-1111-1111"}]},
                    )
                ],
            ),
        ]
    )

    report = scan_run_for_pii(run)

    assert report.total_findings == 4
    assert report.detector_counts == {
        "email": 1,
        "phone": 1,
        "ssn": 1,
        "payment_card": 1,
    }
    assert report.findings[0].field_path == "content"
    assert report.findings[1].field_path == "content"
    assert report.findings[2].field_path == "tool_calls[0].arguments.profile.ssn"
    assert report.findings[2].tool_function == "lookup_customer"
    assert report.findings[3].field_path == "tool_calls[0].result.cards[0].number"


def test_scan_run_for_pii_redacts_previews() -> None:
    run = _build_run(
        turns=[
            Turn(
                index=2,
                role=TurnRole.ASSISTANT,
                content="Use alice@example.com and 4111 1111 1111 1111 for testing.",
            )
        ]
    )

    report = scan_run_for_pii(run)
    previews = [finding.preview for finding in report.findings]

    assert "alice@example.com" not in "".join(previews)
    assert "4111 1111 1111 1111" not in "".join(previews)
    assert any("a****@e******.com" in preview for preview in previews)
    assert any("**** **** **** 1111" in preview for preview in previews)


def test_scan_run_for_pii_avoids_common_false_positives() -> None:
    run = _build_run(
        turns=[
            Turn(
                index=0,
                role=TurnRole.USER,
                content=(
                    "Order ORD-123 ships on 2026-03-13. Ref 123456789. "
                    "Alt ref 4111-1111-1111-1112."
                ),
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        function="lookup_order",
                        arguments={"order_id": "ORD-123", "ticket": 1234567890},
                        result={"processed_at": "2026-03-13T12:45:00Z"},
                    )
                ],
            )
        ]
    )

    report = scan_run_for_pii(run)

    assert report.total_findings == 0


def _build_run(turns: list[Turn]) -> AgentRun:
    return AgentRun(
        run_id="run-1",
        model=ModelInfo(provider="openai", model="gpt-4o-mini"),
        metadata=RunMetadata(scenario="pii-test"),
        summary=RunSummary(
            total_turns=len(turns),
            total_tool_calls=sum(len(turn.tool_calls) for turn in turns),
        ),
        turns=turns,
    )
