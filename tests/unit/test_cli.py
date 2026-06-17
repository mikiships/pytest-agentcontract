"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentcontract.cli import main
from agentcontract.test_gap import ModuleGap, TestGapSummary


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_gaps_prints_ranked_report(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    summary = TestGapSummary(
        source_root=Path("src/agentcontract"),
        test_root=Path("tests"),
        coverage_file=Path(".coverage"),
        analyzed_module_count=2,
        test_root_exists=True,
        modules=(
            ModuleGap(
                module_name="agentcontract.plugin",
                source_path=Path("src/agentcontract/plugin.py"),
                total_line_count=20,
                missing_line_count=8,
                executed_line_count=12,
                coverage_percent=60.0,
                companion_tests=(Path("tests/unit/test_plugin.py"),),
            ),
            ModuleGap(
                module_name="agentcontract.types",
                source_path=Path("src/agentcontract/types.py"),
                total_line_count=15,
                missing_line_count=5,
                executed_line_count=10,
                coverage_percent=66.7,
                companion_tests=(),
            ),
        ),
    )

    monkeypatch.setattr("agentcontract.test_gap.analyze_test_gaps", lambda **_: summary)

    exit_code = main(["gaps", "--limit", "1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Coverage gaps in src/agentcontract" in captured.out
    assert "Showing top 1 of 2 modules with gaps" in captured.out
    assert "agentcontract.plugin: 60.0% covered" in captured.out
    assert "tests/unit/test_plugin.py" in captured.out
    assert "agentcontract.types" not in captured.out


def test_gaps_reports_missing_test_root(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    summary = TestGapSummary(
        source_root=Path("src/agentcontract"),
        test_root=Path("tests"),
        coverage_file=Path(".coverage"),
        analyzed_module_count=1,
        test_root_exists=False,
        modules=(
            ModuleGap(
                module_name="agentcontract.config",
                source_path=Path("src/agentcontract/config.py"),
                total_line_count=12,
                missing_line_count=4,
                executed_line_count=8,
                coverage_percent=66.7,
                companion_tests=(),
            ),
        ),
    )

    monkeypatch.setattr("agentcontract.test_gap.analyze_test_gaps", lambda **_: summary)

    exit_code = main(["gaps"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Companion test scan skipped missing directory: tests" in captured.out
    assert "Companion tests: none found" in captured.out


def test_gaps_returns_error_for_invalid_limit(capsys) -> None:
    exit_code = main(["gaps", "--limit", "0"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--limit must be greater than zero" in captured.err


def test_gaps_returns_error_for_coverage_failures(capsys) -> None:
    exit_code = main(["gaps", "--coverage-file", "missing.coverage"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "coverage data file not found" in captured.err
