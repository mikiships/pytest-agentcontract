"""Tests for CLI commands."""

from pathlib import Path

from agentcontract.cli import main
from agentcontract.serialization import save_run
from agentcontract.types import (
    AgentRun,
    ModelInfo,
    RunMetadata,
    RunSummary,
    ToolCall,
    Turn,
    TurnRole,
)


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_scan_pii_returns_success_when_no_findings(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "clean.agentrun.json"
    save_run(_build_run("No sensitive content here."), cassette)

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No potential PII findings detected" in captured.out


def test_scan_pii_returns_findings_exit_code_and_summary(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "pii.agentrun.json"
    save_run(
        _build_run(
            "Email me at alice@example.com",
            tool_calls=[
                ToolCall(
                    id="tc_lookup",
                    function="lookup_customer",
                    arguments={"callback": "(212) 555-0199"},
                    result={"card": "4111 1111 1111 1111"},
                )
            ],
        ),
        cassette,
    )

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Potential PII findings: 3" in captured.out
    assert "  email: 1" in captured.out
    assert "  phone: 1" in captured.out
    assert "  payment_card: 1" in captured.out
    assert "tool:lookup_customer tool_calls[0].arguments.callback" in captured.out
    assert "alice@example.com" not in captured.out
    assert "4111 1111 1111 1111" not in captured.out


def test_scan_pii_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["scan-pii", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to scan cassette" in captured.err


def _build_run(content: str, tool_calls: list[ToolCall] | None = None) -> AgentRun:
    calls = tool_calls or []
    return AgentRun(
        run_id="run-1",
        model=ModelInfo(provider="openai", model="gpt-4o-mini"),
        metadata=RunMetadata(scenario="cli-test"),
        summary=RunSummary(total_turns=1, total_tool_calls=len(calls)),
        turns=[
            Turn(
                index=0,
                role=TurnRole.ASSISTANT,
                content=content,
                tool_calls=calls,
            )
        ],
    )
