"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

from agentcontract.cli import main


def test_validate_returns_error_for_missing_cassette(capsys) -> None:
    exit_code = main(["validate", "missing.agentrun.json"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "missing.agentrun.json not found" in captured.err


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_init_creates_config_file(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (tmp_path / "agentcontract.yml").exists()
    assert "Created agentcontract.yml" in captured.out


def test_test_gap_reports_success_without_fail_flag(tmp_path: Path, capsys) -> None:
    _write(
        tmp_path / "src/agentcontract/plugin.py",
        "def pytest_addoption() -> None:\n    return None\n",
    )
    (tmp_path / "tests/unit").mkdir(parents=True)

    exit_code = main(["test-gap", "--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "agentcontract.plugin" in captured.out


def test_test_gap_returns_nonzero_when_requested(tmp_path: Path, capsys) -> None:
    _write(
        tmp_path / "src/agentcontract/plugin.py",
        "def pytest_addoption() -> None:\n    return None\n",
    )
    (tmp_path / "tests/unit").mkdir(parents=True)

    exit_code = main(["test-gap", "--root", str(tmp_path), "--fail-on-gaps"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Missing coverage:" in captured.out


def test_test_gap_returns_error_for_invalid_root(tmp_path: Path, capsys) -> None:
    exit_code = main(["test-gap", "--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "package directory not found" in captured.err


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
