"""Tests for CLI commands."""

from pathlib import Path

from agentcontract.cli import main


def test_init_creates_supported_starter_config(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Created agentcontract.yml" in captured.out
    assert (
        (tmp_path / "agentcontract.yml").read_text()
        == """\
version: "1"

scenarios:
  include: ["tests/scenarios/**/*.agentrun.json"]

replay:
  stub_tools: true
  concurrency: 5

defaults:
  assertions:
    - type: contains
      target: final_response
      value: ""  # customize this

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: []  # list your agent's tools here
"""
    )


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err
