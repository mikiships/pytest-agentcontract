"""Tests for CLI commands."""

from pathlib import Path

import pytest

from agentcontract.cli import main
from agentcontract.serialization import save_run
from agentcontract.types import AgentRun, RunMetadata, ToolCall, Turn, TurnRole


def test_info_returns_error_for_invalid_cassette(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_security_returns_success_for_clean_cassette(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cassette = tmp_path / "clean.agentrun.json"
    save_run(
        AgentRun(
            metadata=RunMetadata(scenario="clean"),
            turns=[
                Turn(
                    index=0,
                    role=TurnRole.ASSISTANT,
                    content="Done.",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            function="lookup_order",
                            arguments={"order_id": "123"},
                            result={"status": "delivered"},
                        )
                    ],
                )
            ],
        ),
        cassette,
    )

    exit_code = main(["security", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No security foot-guns found" in captured.out


def test_security_returns_failure_for_detected_footgun(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-1234567890abcdefghijkl"
    cassette = tmp_path / "leak.agentrun.json"
    save_run(
        AgentRun(
            metadata=RunMetadata(scenario="leak"),
            turns=[Turn(index=0, role=TurnRole.ASSISTANT, content=f"token={secret}")],
        ),
        cassette,
    )

    exit_code = main(["security", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Security foot-guns found" in captured.out
    assert "secret at turns[0].content" in captured.out
    assert secret not in captured.out


def test_security_returns_error_for_invalid_cassette(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["security", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err
