"""Tests for CLI commands."""

from pathlib import Path

import yaml

from agentcontract.cli import main


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_init_writes_supported_template(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init"])
    captured = capsys.readouterr()
    config = yaml.safe_load((tmp_path / "agentcontract.yml").read_text())

    assert exit_code == 0
    assert "Created agentcontract.yml" in captured.out
    assert config == {
        "version": "1",
        "scenarios": {"include": ["tests/scenarios/**/*.agentrun.json"]},
        "replay": {"stub_tools": True},
        "defaults": {
            "assertions": [
                {
                    "type": "contains",
                    "target": "final_response",
                    "value": "",
                }
            ]
        },
        "policies": [
            {
                "name": "allowed-tools",
                "type": "tool_allowlist",
                "tools": [],
            }
        ],
    }
    rendered = (tmp_path / "agentcontract.yml").read_text()
    assert "concurrency" not in rendered
    assert "budgets" not in rendered
    assert "reporting" not in rendered
    assert "baseline" not in rendered
