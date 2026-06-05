"""Tests for PII exposure scanning."""

from pathlib import Path

from agentcontract.pii import (
    luhn_valid,
    scan_agent_run,
    scan_cassette_file,
    scan_payload,
    scan_text,
)
from agentcontract.serialization import save_run
from agentcontract.types import AgentRun, RunMetadata, ToolCall, Turn, TurnRole


def test_scan_text_detects_email() -> None:
    result = scan_text("Contact alice@example.com", location="turns[0].content")

    assert result.finding_count == 1
    finding = result.findings[0]
    assert finding.category == "email"
    assert finding.location == "turns[0].content"
    assert finding.snippet == "a***@e***.com"
    assert "alice@example.com" not in finding.snippet


def test_scan_text_detects_us_phone_number() -> None:
    result = scan_text("Call (212) 555-0198 today", location="turns[0].content")

    assert result.finding_count == 1
    assert result.findings[0].category == "phone"
    assert result.findings[0].snippet == "(***) ***-0198"


def test_scan_text_detects_ssn() -> None:
    result = scan_text("SSN: 123-45-6789", location="turns[0].content")

    assert result.finding_count == 1
    assert result.findings[0].category == "ssn"
    assert result.findings[0].snippet == "***-**-6789"


def test_scan_text_detects_luhn_valid_credit_card() -> None:
    result = scan_text("Card 4111 1111 1111 1111", location="turns[0].content")

    assert result.finding_count == 1
    assert result.findings[0].category == "credit_card"
    assert result.findings[0].snippet == "**** **** **** 1111"


def test_credit_card_detector_requires_luhn_valid_number() -> None:
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")

    result = scan_text("Card 4111 1111 1111 1112", location="turns[0].content")

    assert not result.has_findings


def test_scan_payload_recurses_nested_payloads_with_stable_locations() -> None:
    result = scan_payload(
        {
            "customer": {"email": "alice@example.com"},
            "phones": ["212-555-0198"],
        },
        location="arguments",
    )

    assert {finding.location for finding in result.findings} == {
        "arguments.customer.email",
        "arguments.phones[0]",
    }
    assert {finding.category for finding in result.findings} == {"email", "phone"}


def test_scan_payload_scans_dict_keys_without_raw_location_leak() -> None:
    result = scan_payload(
        {
            "alice@example.com": "customer record",
            "orders": [{"212-555-0198": "callback"}],
        },
        location="arguments",
    )

    assert {finding.category for finding in result.findings} == {"email", "phone"}
    assert {finding.location for finding in result.findings} == {
        "arguments[<key:0>].__key__",
        "arguments.orders[0][<key:0>].__key__",
    }
    for finding in result.findings:
        assert "alice@example.com" not in finding.location
        assert "212-555-0198" not in finding.location


def test_scan_payload_masks_pii_key_locations_for_nested_value_findings() -> None:
    result = scan_payload(
        {"alice@example.com": {"ssn": "123-45-6789"}},
        location="arguments",
        categories=["ssn"],
    )

    assert result.finding_count == 1
    finding = result.findings[0]
    assert finding.category == "ssn"
    assert finding.location == "arguments[<key:0>].ssn"
    assert "alice@example.com" not in finding.location
    assert "123-45-6789" not in finding.snippet


def test_scan_payload_supports_category_filtering() -> None:
    result = scan_payload(
        {"email": "alice@example.com", "ssn": "123-45-6789"},
        location="arguments",
        categories=["ssn"],
    )

    assert result.finding_count == 1
    assert result.findings[0].category == "ssn"
    assert result.findings[0].location == "arguments.ssn"


def test_scan_agent_run_covers_metadata_turns_tool_arguments_and_results() -> None:
    run = AgentRun(
        metadata=RunMetadata(
            scenario="support-alice@example.com",
            description="Customer card 4111 1111 1111 1111",
            tags=["123-45-6789"],
        ),
        turns=[
            Turn(index=0, role=TurnRole.USER, content="Call me at 212-555-0198"),
            Turn(
                index=1,
                role=TurnRole.ASSISTANT,
                content="Looking up the account",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        function="lookup_customer",
                        arguments={
                            "customer": {"email": "alice@example.com"},
                            "billing_4111 1111 1111 1111": "card key",
                        },
                        result={
                            "phone": "(212) 555-0198",
                            "alice@example.com": "email key",
                        },
                    )
                ],
            ),
        ],
    )

    result = scan_agent_run(run)

    assert {finding.location for finding in result.findings} == {
        "metadata.scenario",
        "metadata.description",
        "metadata.tags[0]",
        "turns[0].content",
        "turns[1].tool_calls[0].arguments.customer.email",
        "turns[1].tool_calls[0].arguments[<key:1>].__key__",
        "turns[1].tool_calls[0].result.phone",
        "turns[1].tool_calls[0].result[<key:1>].__key__",
    }
    assert all("alice@example.com" not in finding.location for finding in result.findings)
    assert all("4111 1111 1111 1111" not in finding.location for finding in result.findings)


def test_scan_cassette_file_sets_cassette_path(tmp_path: Path) -> None:
    cassette = tmp_path / "pii.agentrun.json"
    save_run(
        AgentRun(
            metadata=RunMetadata(scenario="pii"),
            turns=[Turn(index=0, role=TurnRole.USER, content="alice@example.com")],
        ),
        cassette,
    )

    result = scan_cassette_file(cassette)

    assert result.scanned_files == [str(cassette)]
    assert result.findings[0].cassette_path == str(cassette)
    assert "alice@example.com" not in result.findings[0].snippet
