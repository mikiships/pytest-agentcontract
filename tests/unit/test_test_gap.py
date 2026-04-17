"""Tests for the coverage gap analyzer."""

from __future__ import annotations

import runpy
from pathlib import Path
from textwrap import dedent

import pytest
from coverage import Coverage

from agentcontract.test_gap import CoverageDataError, analyze_test_gaps, find_companion_tests


def test_analyze_test_gaps_ranks_missing_modules_and_filters_fully_covered(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src" / "samplepkg"
    test_root = tmp_path / "tests" / "unit"

    alpha = _write_file(
        source_root / "alpha.py",
        """
        def covered() -> int:
            return 1

        def missing() -> int:
            value = 2
            return value

        covered()
        """,
    )
    beta = _write_file(
        source_root / "beta.py",
        """
        def one() -> int:
            return 1

        def two() -> int:
            value = 2
            return value

        def three() -> int:
            value = 3
            return value

        one()
        """,
    )
    gamma = _write_file(
        source_root / "gamma.py",
        """
        def fully_covered() -> int:
            return 1

        fully_covered()
        """,
    )
    _write_file(test_root / "test_alpha.py", "def test_alpha() -> None:\n    assert True\n")
    _write_file(test_root / "beta_test.py", "def test_beta() -> None:\n    assert True\n")

    coverage_file = tmp_path / ".coverage"
    _record_coverage(coverage_file, source_root, alpha, beta, gamma)

    summary = analyze_test_gaps(
        source_root=source_root,
        test_root=tmp_path / "tests",
        coverage_file=coverage_file,
    )

    assert summary.analyzed_module_count == 3
    assert [gap.module_name for gap in summary.modules] == ["samplepkg.beta", "samplepkg.alpha"]
    assert summary.modules[0].source_path == Path("beta.py")
    assert summary.modules[0].missing_line_count > summary.modules[1].missing_line_count
    assert summary.modules[0].executed_line_count == (
        summary.modules[0].statement_count - summary.modules[0].missing_line_count
    )
    assert summary.modules[0].companion_tests == (Path("unit/beta_test.py"),)
    assert summary.modules[1].companion_tests == (Path("unit/test_alpha.py"),)
    assert all(gap.module_name != "samplepkg.gamma" for gap in summary.modules)


def test_analyze_test_gaps_handles_missing_test_directory(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "samplepkg"
    delta = _write_file(
        source_root / "delta.py",
        """
        def covered() -> int:
            return 1

        def missing() -> int:
            value = 2
            return value

        covered()
        """,
    )

    coverage_file = tmp_path / ".coverage"
    _record_coverage(coverage_file, source_root, delta)

    summary = analyze_test_gaps(
        source_root=source_root,
        test_root=tmp_path / "tests",
        coverage_file=coverage_file,
    )

    assert summary.test_root_exists is False
    assert len(summary.modules) == 1
    assert summary.modules[0].module_name == "samplepkg.delta"
    assert summary.modules[0].companion_tests == ()


def test_find_companion_tests_matches_grouped_and_suffix_patterns(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "samplepkg"
    test_root = tmp_path / "tests" / "unit"
    _write_file(
        source_root / "adapters" / "openai_agents.py",
        "def record() -> str:\n    return 'ok'\n",
    )
    _write_file(test_root / "test_adapters.py", "def test_group() -> None:\n    assert True\n")
    _write_file(test_root / "openai_agents_test.py", "def test_leaf() -> None:\n    assert True\n")
    _write_file(test_root / "test_replay.py", "def test_replay() -> None:\n    assert True\n")
    _write_file(test_root / "helpers.py", "HELPER = True\n")

    adapter_matches = find_companion_tests(
        source_root / "adapters" / "openai_agents.py",
        source_root=source_root,
        test_root=tmp_path / "tests",
    )
    replay_matches = find_companion_tests(
        Path("replay/engine.py"),
        test_root=tmp_path / "tests",
    )

    assert adapter_matches == (
        Path("unit/openai_agents_test.py"),
        Path("unit/test_adapters.py"),
    )
    assert replay_matches == (Path("unit/test_replay.py"),)


def test_analyze_test_gaps_raises_for_missing_source_or_coverage_data(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src" / "samplepkg"

    with pytest.raises(CoverageDataError, match="source root not found"):
        analyze_test_gaps(
            source_root=source_root,
            test_root=tmp_path / "tests",
            coverage_file=tmp_path / ".coverage",
        )

    _write_file(source_root / "alpha.py", "def covered() -> int:\n    return 1\n")

    missing_coverage = tmp_path / ".coverage"
    with pytest.raises(CoverageDataError, match="coverage data file not found"):
        analyze_test_gaps(
            source_root=source_root,
            test_root=tmp_path / "tests",
            coverage_file=missing_coverage,
        )


def test_analyze_test_gaps_raises_for_malformed_coverage_data(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "samplepkg"
    _write_file(source_root / "alpha.py", "def covered() -> int:\n    return 1\n")

    malformed_coverage = tmp_path / "broken.coverage"
    malformed_coverage.write_text("not valid coverage data")

    with pytest.raises(CoverageDataError, match="failed to load coverage data from"):
        analyze_test_gaps(
            source_root=source_root,
            test_root=tmp_path / "tests",
            coverage_file=malformed_coverage,
        )


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip())
    return path


def _record_coverage(coverage_file: Path, source_root: Path, *paths: Path) -> None:
    coverage = Coverage(
        source=[str(source_root.parent)],
        data_file=str(coverage_file),
    )
    coverage.start()
    try:
        for index, path in enumerate(paths):
            runpy.run_path(str(path), run_name=f"__test_gap_{index}__")
    finally:
        coverage.stop()
    coverage.save()
    coverage.get_data().close()
