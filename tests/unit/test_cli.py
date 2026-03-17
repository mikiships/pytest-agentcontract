"""Tests for CLI commands."""

import json
from pathlib import Path

import pytest

from agentcontract.cli import main
from agentcontract.serialization import run_to_dict
from agentcontract.types import AgentRun, ModelInfo, RunMetadata, RunSummary, TokenUsage


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def _write_valid_cassette(tmp_path: Path, scenario: str = "refund-flow") -> Path:
    cassette = tmp_path / f"{scenario}.agentrun.json"
    cassette.write_text(
        json.dumps(
            run_to_dict(
                AgentRun(
                    run_id="run-123",
                    recorded_at="2026-03-17T00:00:00Z",
                    model=ModelInfo(provider="openai", model="gpt-4o"),
                    metadata=RunMetadata(scenario=scenario),
                    summary=RunSummary(
                        total_turns=2,
                        total_duration_ms=1250,
                        total_tokens=TokenUsage(total=42),
                        total_tool_calls=1,
                        estimated_cost_usd=0.0123,
                    ),
                )
            )
        )
    )
    return cassette


def test_info_prints_summary_for_valid_cassette(tmp_path: Path, capsys) -> None:
    cassette = _write_valid_cassette(tmp_path)

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Scenario:    refund-flow" in captured.out
    assert "Model:       openai/gpt-4o" in captured.out
    assert "Tokens:      42" in captured.out


@pytest.mark.parametrize("command", ["info", "validate"])
def test_commands_report_missing_files(command: str, tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.agentrun.json"

    exit_code = main([command, str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert f"Error: {missing} not found" in captured.err


def test_validate_prints_success_for_valid_cassette(tmp_path: Path, capsys) -> None:
    cassette = _write_valid_cassette(tmp_path, scenario="validated-flow")

    exit_code = main(["validate", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Valid cassette: validated-flow (0 turns)" in captured.out


def test_validate_reports_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["validate", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Invalid cassette" in captured.err


def test_init_creates_starter_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (tmp_path / "agentcontract.yml").exists()
    assert 'version: "1"' in (tmp_path / "agentcontract.yml").read_text()
    assert "Created agentcontract.yml" in captured.out


def test_init_refuses_to_overwrite_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agentcontract.yml").write_text("version: \"existing\"\n")

    exit_code = main(["init"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "already exists" in captured.err


def test_init_reports_write_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_write_text(self: Path, text: str) -> int:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "write_text", fake_write_text)

    exit_code = main(["init"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to write agentcontract.yml" in captured.err


def test_main_prints_help_when_no_command_is_given(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage: agentcontract" in captured.out
