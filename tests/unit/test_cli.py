"""Tests for CLI commands."""

import json
from pathlib import Path

from agentcontract.cli import main
from agentcontract.serialization import save_run
from agentcontract.types import AgentRun, RunMetadata, ToolCall, Turn, TurnRole


def _write_cassette(path: Path, content: str) -> None:
    save_run(
        AgentRun(
            metadata=RunMetadata(scenario=path.stem),
            turns=[Turn(index=0, role=TurnRole.USER, content=content)],
        ),
        path,
    )


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_scan_pii_returns_zero_for_clean_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "clean.agentrun.json"
    _write_cassette(cassette, "No sensitive customer details here")

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "no PII findings" in captured.out


def test_scan_pii_returns_nonzero_and_masks_findings(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "pii.agentrun.json"
    _write_cassette(cassette, "Customer email is alice@example.com")

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "email" in captured.out
    assert "turns[0].content" in captured.out
    assert "a***@e***.com" in captured.out
    assert "alice@example.com" not in captured.out


def test_scan_pii_accepts_multiple_input_files(tmp_path: Path, capsys) -> None:
    clean = tmp_path / "clean.agentrun.json"
    pii = tmp_path / "pii.agentrun.json"
    _write_cassette(clean, "No sensitive customer details here")
    _write_cassette(pii, "Customer email is alice@example.com")

    exit_code = main(["scan-pii", str(clean), str(pii)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "1 finding(s) across 2 cassette(s)" in captured.out
    assert str(pii) in captured.out
    assert "alice@example.com" not in captured.out


def test_scan_pii_accepts_glob_input(tmp_path: Path, capsys) -> None:
    clean = tmp_path / "clean.agentrun.json"
    pii = tmp_path / "pii.agentrun.json"
    ignored = tmp_path / "ignored.json"
    _write_cassette(clean, "No sensitive customer details here")
    _write_cassette(pii, "Call 212-555-0198")
    ignored.write_text('{"content":"alice@example.com"}')

    exit_code = main(["scan-pii", str(tmp_path / "*.agentrun.json")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "1 finding(s) across 2 cassette(s)" in captured.out
    assert "phone" in captured.out
    assert str(pii) in captured.out
    assert "ignored.json" not in captured.out


def test_scan_pii_masks_pii_dict_key_locations(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "pii-key.agentrun.json"
    save_run(
        AgentRun(
            metadata=RunMetadata(scenario="pii-key"),
            turns=[
                Turn(
                    index=0,
                    role=TurnRole.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            function="lookup_customer",
                            arguments={"alice@example.com": "customer record"},
                        )
                    ],
                )
            ],
        ),
        cassette,
    )

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "turns[0].tool_calls[0].arguments[<key:0>].__key__" in captured.out
    assert "a***@e***.com" in captured.out
    assert "alice@example.com" not in captured.out


def test_scan_pii_recurses_directories(tmp_path: Path, capsys) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    cassette = nested / "pii.agentrun.json"
    _write_cassette(cassette, "Call 212-555-0198")
    ignored = nested / "ignored.json"
    ignored.write_text('{"content":"alice@example.com"}')

    exit_code = main(["scan-pii", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "phone" in captured.out
    assert str(cassette) in captured.out
    assert "ignored.json" not in captured.out


def test_scan_pii_json_output(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "pii.agentrun.json"
    _write_cassette(cassette, "SSN: 123-45-6789")

    exit_code = main(["scan-pii", "--json", str(cassette)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["finding_count"] == 1
    assert payload["findings"][0]["category"] == "ssn"
    assert payload["findings"][0]["snippet"] == "***-**-6789"
    assert "123-45-6789" not in captured.out


def test_scan_pii_returns_error_for_missing_path(tmp_path: Path, capsys) -> None:
    exit_code = main(["scan-pii", str(tmp_path / "missing.agentrun.json")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "not found" in captured.err


def test_scan_pii_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to scan" in captured.err


def test_scan_pii_invalid_cassette_error_does_not_echo_raw_values(
    tmp_path: Path, capsys
) -> None:
    cassette = tmp_path / "bad-pii.agentrun.json"
    cassette.write_text('{"turns":[{"index":0,"role":"alice@example.com"}]}')

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to scan" in captured.err
    assert "alice@example.com" not in captured.err
