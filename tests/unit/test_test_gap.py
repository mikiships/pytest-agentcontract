"""Tests for coverage gap analysis."""

from __future__ import annotations

import runpy
import textwrap
from pathlib import Path

import pytest
from coverage import Coverage

from agentcontract.test_gap import CoverageDataError, analyze_test_gaps, find_companion_tests


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n")


def _create_coverage(repo_root: Path, *executed_files: Path) -> Path:
    coverage_path = repo_root / ".coverage"
    coverage = Coverage(source=[str(repo_root / "src")], data_file=str(coverage_path))
    coverage.start()
    for index, source_file in enumerate(executed_files):
        runpy.run_path(str(source_file), run_name=f"__test_gap_{index}__")
    coverage.stop()
    coverage.save()
    coverage.get_data().close()
    return coverage_path


def test_analyze_test_gaps_ranks_modules_and_reports_companion_tests(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "agentcontract"
    test_root = tmp_path / "tests" / "unit"

    _write(
        source_root / "plugin.py",
        """
        def covered():
            value = 1
            return value


        def missing():
            hidden = 2
            return hidden


        covered()
        """,
    )
    _write(
        source_root / "types.py",
        """
        def missing_one():
            first = 1
            second = first + 1
            return second


        def missing_two():
            third = 3
            fourth = third + 1
            return fourth
        """,
    )
    _write(
        source_root / "config.py",
        """
        def covered():
            number = 1
            return number


        covered()
        """,
    )
    _write(test_root / "test_config.py", "def test_placeholder():\n    assert True\n")

    coverage_path = _create_coverage(
        tmp_path,
        source_root / "plugin.py",
        source_root / "config.py",
    )

    report = analyze_test_gaps(
        coverage_file=coverage_path,
        source_root=source_root,
        test_root=tmp_path / "tests",
    )

    assert [module.module_name for module in report.modules] == [
        "agentcontract.types",
        "agentcontract.plugin",
    ]
    assert report.modules[0].missing_line_count > report.modules[1].missing_line_count
    assert report.analyzed_module_count == 3
    assert report.modules[0].obvious_test_paths == ()
    assert [module.module_name for module in report.modules_without_obvious_tests] == [
        "agentcontract.types",
        "agentcontract.plugin",
    ]


def test_find_companion_tests_matches_generic_module_to_parent_test(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "agentcontract"
    test_root = tmp_path / "tests"

    _write(source_root / "replay" / "engine.py", "def replay():\n    return 'ok'\n")
    _write(test_root / "unit" / "test_replay.py", "def test_placeholder():\n    assert True\n")

    matched_tests = find_companion_tests(
        source_path=source_root / "replay" / "engine.py",
        source_root=source_root,
        test_root=test_root,
    )

    assert [path.name for path in matched_tests] == ["test_replay.py"]


def test_find_companion_tests_matches_grouped_adapter_tests(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "agentcontract"
    test_root = tmp_path / "tests"

    _write(source_root / "adapters" / "openai_agents.py", "def record():\n    return 'ok'\n")
    _write(test_root / "unit" / "test_adapters.py", "def test_placeholder():\n    assert True\n")

    matched_tests = find_companion_tests(
        source_path=source_root / "adapters" / "openai_agents.py",
        source_root=source_root,
        test_root=test_root,
    )

    assert [path.name for path in matched_tests] == ["test_adapters.py"]


def test_analyze_test_gaps_omits_fully_covered_modules(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "agentcontract"
    test_root = tmp_path / "tests"

    _write(
        source_root / "config.py",
        """
        def covered():
            value = 1
            return value


        covered()
        """,
    )
    _write(test_root / "unit" / "test_config.py", "def test_placeholder():\n    assert True\n")

    coverage_path = _create_coverage(tmp_path, source_root / "config.py")

    report = analyze_test_gaps(
        coverage_file=coverage_path,
        source_root=source_root,
        test_root=test_root,
    )

    assert report.analyzed_module_count == 1
    assert report.modules == ()


def test_analyze_test_gaps_errors_when_coverage_is_missing(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "agentcontract"
    test_root = tmp_path / "tests"
    source_root.mkdir(parents=True)
    test_root.mkdir(parents=True)

    with pytest.raises(CoverageDataError, match="coverage data not found"):
        analyze_test_gaps(
            coverage_file=tmp_path / ".coverage",
            source_root=source_root,
            test_root=test_root,
        )


def test_analyze_test_gaps_errors_when_coverage_is_malformed(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "agentcontract"
    test_root = tmp_path / "tests"
    source_root.mkdir(parents=True)
    test_root.mkdir(parents=True)
    (tmp_path / ".coverage").write_text("not a sqlite database")

    with pytest.raises(CoverageDataError, match="failed to load coverage data"):
        analyze_test_gaps(
            coverage_file=tmp_path / ".coverage",
            source_root=source_root,
            test_root=test_root,
        )
