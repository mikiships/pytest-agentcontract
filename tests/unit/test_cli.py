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


def test_init_writes_minimal_supported_template(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init"])
    captured = capsys.readouterr()
    content = (tmp_path / "agentcontract.yml").read_text()

    assert exit_code == 0
    assert "Created agentcontract.yml" in captured.out
    assert "budgets:" not in content
    assert "reporting:" not in content
    assert 'concurrency: 5' in content
