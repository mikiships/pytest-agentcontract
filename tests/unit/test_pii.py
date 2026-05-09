"""Tests for PII scanning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcontract.pii import PiiScanError, scan_data, scan_file


def test_scan_data_detects_common_pii_patterns() -> None:
    findings = scan_data(
        {
            "turns": [
                {
                    "index": 0,
                    "role": "user",
                    "content": (
                        "Contact jane.doe@example.com. SSN 123-45-6789, "
                        "or phone +14155552671"
                    ),
                },
                {
                    "index": 1,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "arguments": {"card": "4111 1111 1111 1111"},
                            "result": {"alternate": "(212) 555-0100"},
                        }
                    ],
                },
            ]
        },
        Path("case.agentrun.json"),
    )

    assert [finding.kind for finding in findings] == [
        "email",
        "ssn",
        "phone",
        "credit_card",
        "phone",
    ]


def test_scan_data_reports_recursive_json_locations() -> None:
    findings = scan_data(
        {
            "metadata": {"description": "owner@example.com is outside scan scope"},
            "turns": [
                {
                    "index": 0,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "arguments": {"safe": "ORD-123"},
                            "result": {
                                "customer": {
                                    "contacts": [
                                        {"email": "alice@example.com"},
                                    ]
                                }
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert len(findings) == 1
    assert (
        findings[0].json_path
        == "$.turns[0].tool_calls[0].result.customer.contacts[0].email"
    )


def test_scan_data_scans_object_keys_and_masks_pii_in_json_paths() -> None:
    findings = scan_data(
        {
            "turns": [
                {
                    "index": 0,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "arguments": {
                                "alice@example.com": {
                                    "phone (415) 555-2671": "safe value",
                                }
                            },
                        }
                    ],
                }
            ]
        }
    )

    assert [finding.kind for finding in findings] == ["email", "phone"]
    assert (
        findings[0].json_path
        == '$.turns[0].tool_calls[0].arguments["a***@e***.com"]'
    )
    assert (
        findings[1].json_path
        == '$.turns[0].tool_calls[0].arguments["a***@e***.com"]'
        '["phone ***-***-2671"]'
    )
    rendered = "\n".join(
        f"{finding.json_path} {finding.preview}" for finding in findings
    )
    assert "alice@example.com" not in rendered
    assert "(415) 555-2671" not in rendered


def test_scan_data_masks_previews_without_full_values() -> None:
    findings = scan_data(
        {
            "turns": [
                {
                    "index": 0,
                    "role": "user",
                    "content": "Email alice@example.com and use card 4111-1111-1111-1111",
                }
            ]
        }
    )

    previews = {finding.kind: finding.preview for finding in findings}
    assert previews["email"] == "a***@e***.com"
    assert previews["credit_card"] == "**** **** **** 1111"
    assert "alice@example.com" not in previews["email"]
    assert "4111-1111-1111-1111" not in previews["credit_card"]


def test_scan_data_filters_credit_cards_with_luhn() -> None:
    findings = scan_data(
        {
            "turns": [
                {
                    "index": 0,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "arguments": {
                                "invalid": "4111 1111 1111 1112",
                                "valid": 4111111111111111,
                            },
                            "result": None,
                        }
                    ],
                }
            ]
        }
    )

    assert len(findings) == 1
    assert findings[0].kind == "credit_card"
    assert findings[0].json_path == "$.turns[0].tool_calls[0].arguments.valid"


def test_scan_data_returns_empty_for_clean_partial_cassette() -> None:
    findings = scan_data(
        {
            "turns": [
                {"index": 0, "role": "user", "content": "Refund order ORD-123"},
                {
                    "index": 1,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "arguments": {"order_id": "ORD-123"},
                            "result": {"delivered_at": "2026-02-10", "amount": 79.99},
                        }
                    ],
                },
            ]
        }
    )

    assert findings == []


def test_scan_file_raises_clean_error_for_invalid_json(tmp_path: Path) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text("{")

    with pytest.raises(PiiScanError):
        scan_file(cassette)


def test_scan_file_reads_raw_json_without_schema_validation(tmp_path: Path) -> None:
    cassette = tmp_path / "partial.agentrun.json"
    cassette.write_text(
        json.dumps({"turns": [{"content": "Call 415-555-2671", "tool_calls": None}]})
    )

    findings = scan_file(cassette)

    assert len(findings) == 1
    assert findings[0].json_path == "$.turns[0].content"
