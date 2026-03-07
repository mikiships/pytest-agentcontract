"""Tests for CLI commands."""

from pathlib import Path

from agentcontract.cli import main


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_init_writes_slimmer_starter_config(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init"])
    captured = capsys.readouterr()
    config_text = (tmp_path / "agentcontract.yml").read_text()

    assert exit_code == 0
    assert "Created agentcontract.yml" in captured.out
    assert "reporting:" not in config_text
    assert "budgets:" not in config_text
    assert "replay:" in config_text
    assert "policies:" in config_text
