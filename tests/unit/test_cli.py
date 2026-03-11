"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

from agentcontract.cli import main
from agentcontract.recorder.core import Recorder


def _write_cassette(path: Path, scenario: str = "refund-flow") -> Path:
    recorder = Recorder(scenario=scenario, model_provider="openai", model_name="gpt-4o-mini")
    with recorder.recording():
        recorder.add_turn(role="user", content="Need a refund")
        recorder.add_turn(
            role="assistant",
            content="Refund approved",
            prompt_tokens=5,
            completion_tokens=7,
        )
    recorder.run.summary.estimated_cost_usd = 0.0123
    recorder.save(path)
    return path


def test_main_prints_help_for_no_command(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage: agentcontract" in captured.out
    assert "init" in captured.out


def test_info_prints_summary_for_valid_cassette(tmp_path: Path, capsys) -> None:
    cassette = _write_cassette(tmp_path / "valid.agentrun.json")

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Scenario:    refund-flow" in captured.out
    assert "Turns:       2" in captured.out
    assert "Tokens:      12" in captured.out
    assert "Est. cost:   $0.0123" in captured.out


def test_info_returns_error_for_missing_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "missing.agentrun.json"

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert f"Error: {cassette} not found" in captured.err


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_validate_accepts_valid_cassette(tmp_path: Path, capsys) -> None:
    cassette = _write_cassette(tmp_path / "valid.agentrun.json", scenario="triage")

    exit_code = main(["validate", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Valid cassette: triage (2 turns)" in captured.out


def test_validate_rejects_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0,"role":"ghost"}]}')

    exit_code = main(["validate", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Invalid cassette" in captured.err


def test_init_creates_template_in_current_directory(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init"])
    captured = capsys.readouterr()

    created = tmp_path / "agentcontract.yml"
    assert exit_code == 0
    assert created.exists()
    assert 'version: "1"' in created.read_text()
    assert "Created agentcontract.yml" in captured.out


def test_init_rejects_existing_file(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "agentcontract.yml"
    target.write_text("existing: true\n")

    exit_code = main(["init"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "already exists" in captured.err


def test_init_reports_write_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    def broken_write_text(self: Path, data: str, *args: object, **kwargs: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", broken_write_text)

    exit_code = main(["init"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to write agentcontract.yml" in captured.err
