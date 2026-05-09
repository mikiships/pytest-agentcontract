"""Tests for CLI commands."""

import json
from pathlib import Path

from agentcontract.cli import main


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_scan_pii_clean_cassette_exits_zero(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "clean.agentrun.json"
    _write_cassette(
        cassette,
        {
            "turns": [
                {"index": 0, "role": "user", "content": "Refund order ORD-123"},
                {
                    "index": 1,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "arguments": {"order_id": "ORD-123"},
                            "result": {"status": "delivered", "delivered_at": "2026-02-10"},
                        }
                    ],
                },
            ]
        },
    )

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No potential PII found in 1 cassette(s)." in captured.out
    assert captured.err == ""


def test_scan_pii_findings_exit_one_with_masked_report(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "pii.agentrun.json"
    _write_cassette(
        cassette,
        {
            "turns": [
                {
                    "index": 0,
                    "role": "user",
                    "content": "Please email jane.doe@example.com",
                },
                {
                    "index": 1,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "arguments": {
                                "support@example.com": "safe value",
                                "card": "4111 1111 1111 1111",
                            },
                            "result": {"phone": "(415) 555-2671"},
                        }
                    ],
                },
            ]
        },
    )

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Potential PII found: 4 finding(s) in 1 file(s)." in captured.out
    assert "$.turns[0].content [email]" in captured.out
    assert '$.turns[1].tool_calls[0].arguments["s***@e***.com"] [email]' in captured.out
    assert "$.turns[1].tool_calls[0].arguments.card [credit_card]" in captured.out
    assert "$.turns[1].tool_calls[0].result.phone [phone]" in captured.out
    assert "jane.doe@example.com" not in captured.out
    assert "support@example.com" not in captured.out
    assert "4111 1111 1111 1111" not in captured.out
    assert "(415) 555-2671" not in captured.out
    assert captured.err == ""


def test_scan_pii_directory_discovers_nested_cassettes(tmp_path: Path, capsys) -> None:
    root = tmp_path / "scenarios"
    nested = root / "nested"
    nested.mkdir(parents=True)
    _write_cassette(
        nested / "pii.agentrun.json",
        {"turns": [{"index": 0, "role": "user", "content": "SSN 123-45-6789"}]},
    )
    (nested / "ignored.json").write_text('{"turns": []}')

    exit_code = main(["scan-pii", str(root)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert str(nested / "pii.agentrun.json") in captured.out
    assert "$.turns[0].content [ssn]" in captured.out
    assert "ignored.json" not in captured.out


def test_scan_pii_missing_path_exits_two(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.agentrun.json"

    exit_code = main(["scan-pii", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert f"Error: {missing} not found" in captured.err


def test_scan_pii_invalid_json_exits_two(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text("{")

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "failed to read cassette" in captured.err


def _write_cassette(path: Path, data: object) -> None:
    path.write_text(json.dumps(data))
