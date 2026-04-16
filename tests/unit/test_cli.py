"""Tests for CLI commands."""

from pathlib import Path

import pytest

from agentcontract.cli import main
from agentcontract.test_gap import CoverageDataError, GapReport, ModuleGap


def test_info_returns_error_for_invalid_cassette(tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text('{"turns":[{"index":0}]}')

    exit_code = main(["info", str(cassette)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to read cassette" in captured.err


def test_gaps_prints_ranked_report(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    captured_args: dict[str, Path | int] = {}

    def fake_analyze(*, coverage_file: Path, source_root: Path, test_root: Path) -> GapReport:
        captured_args.update(
            {
                "coverage_file": coverage_file,
                "source_root": source_root,
                "test_root": test_root,
            }
        )
        return GapReport(
            coverage_file=coverage_file.resolve(),
            source_root=source_root.resolve(),
            test_root=test_root.resolve(),
            analyzed_module_count=3,
            modules=(
                ModuleGap(
                    module_name="agentcontract.plugin",
                    source_path=source_root.resolve() / "plugin.py",
                    statement_count=10,
                    missing_lines=(3, 4, 5, 6),
                    obvious_test_paths=(),
                ),
                ModuleGap(
                    module_name="agentcontract.config",
                    source_path=source_root.resolve() / "config.py",
                    statement_count=8,
                    missing_lines=(7,),
                    obvious_test_paths=(test_root.resolve() / "unit" / "test_config.py",),
                ),
            ),
        )

    monkeypatch.setattr("agentcontract.test_gap.analyze_test_gaps", fake_analyze)

    exit_code = main(
        [
            "gaps",
            "--coverage-file",
            "custom.coverage",
            "--source-root",
            "src/agentcontract",
            "--test-root",
            "tests",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured_args == {
        "coverage_file": Path("custom.coverage"),
        "source_root": Path("src/agentcontract"),
        "test_root": Path("tests"),
    }
    assert "Coverage gaps from" in captured.out
    assert "Top 1 modules by uncovered lines" in captured.out
    assert "agentcontract.plugin" in captured.out
    assert "agentcontract.config" not in captured.out
    assert "No obvious companion tests: agentcontract.plugin" in captured.out
    assert "2 modules with gaps" in captured.out


def test_gaps_returns_error_when_analysis_fails(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def fake_analyze(*, coverage_file: Path, source_root: Path, test_root: Path) -> GapReport:
        raise CoverageDataError("broken coverage data")

    monkeypatch.setattr("agentcontract.test_gap.analyze_test_gaps", fake_analyze)

    exit_code = main(["gaps"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "broken coverage data" in captured.err


def test_gaps_reports_when_every_module_is_fully_covered(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def fake_analyze(*, coverage_file: Path, source_root: Path, test_root: Path) -> GapReport:
        return GapReport(
            coverage_file=coverage_file.resolve(),
            source_root=source_root.resolve(),
            test_root=test_root.resolve(),
            analyzed_module_count=4,
            modules=(),
        )

    monkeypatch.setattr("agentcontract.test_gap.analyze_test_gaps", fake_analyze)

    exit_code = main(["gaps"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No coverage gaps found across 4 modules." in captured.out


def test_gaps_rejects_non_positive_limit() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["gaps", "--limit", "0"])

    assert excinfo.value.code == 2
