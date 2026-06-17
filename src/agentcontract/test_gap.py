"""Coverage gap analysis for agentcontract source modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coverage import Coverage
from coverage.exceptions import CoverageException


class CoverageDataError(ValueError):
    """Raised when local coverage data cannot be analyzed."""


@dataclass(frozen=True)
class ModuleGap:
    """Coverage details for a single source module with missing lines."""

    module_name: str
    source_path: Path
    total_line_count: int
    missing_line_count: int
    executed_line_count: int
    coverage_percent: float
    companion_tests: tuple[Path, ...]

    @property
    def has_companion_tests(self) -> bool:
        """Whether the analyzer found plausible matching pytest files."""
        return bool(self.companion_tests)


@dataclass(frozen=True)
class TestGapSummary:
    """Deterministic summary of under-tested modules in a checkout."""

    __test__ = False

    source_root: Path
    test_root: Path
    coverage_file: Path
    analyzed_module_count: int
    test_root_exists: bool
    modules: tuple[ModuleGap, ...]


def analyze_test_gaps(
    source_root: Path | str = Path("src/agentcontract"),
    test_root: Path | str = Path("tests"),
    coverage_file: Path | str = Path(".coverage"),
) -> TestGapSummary:
    """Rank modules with uncovered lines using local coverage.py data."""
    source_root = Path(source_root)
    test_root = Path(test_root)
    coverage_file = Path(coverage_file)

    if not source_root.exists():
        raise CoverageDataError(f"source root not found: {source_root}")
    if not coverage_file.exists():
        raise CoverageDataError(f"coverage data file not found: {coverage_file}")

    source_files = sorted(
        path
        for path in source_root.rglob("*.py")
        if path.is_file() and path.name != "__init__.py"
    )
    if not source_files:
        raise CoverageDataError(f"no Python modules found under {source_root}")

    coverage = Coverage(data_file=str(coverage_file))
    try:
        coverage.load()
    except CoverageException as exc:
        raise CoverageDataError(f"failed to load coverage data: {exc}") from exc

    modules: list[ModuleGap] = []
    for source_path in source_files:
        try:
            _, statements, _, missing, _ = coverage.analysis2(str(source_path))
        except CoverageException as exc:
            raise CoverageDataError(
                f"failed to analyze coverage for {source_path}: {exc}"
            ) from exc

        total_line_count = len(statements)
        if total_line_count == 0:
            continue

        missing_line_count = len(missing)
        if missing_line_count == 0:
            continue

        executed_line_count = total_line_count - missing_line_count
        relative_module_path = source_path.relative_to(source_root)
        modules.append(
            ModuleGap(
                module_name=_module_name_for_path(source_root, relative_module_path),
                source_path=source_path,
                total_line_count=total_line_count,
                missing_line_count=missing_line_count,
                executed_line_count=executed_line_count,
                coverage_percent=(executed_line_count / total_line_count) * 100,
                companion_tests=tuple(find_companion_tests(relative_module_path, test_root)),
            )
        )

    modules.sort(
        key=lambda gap: (
            -gap.missing_line_count,
            gap.coverage_percent,
            gap.module_name,
        )
    )

    return TestGapSummary(
        source_root=source_root,
        test_root=test_root,
        coverage_file=coverage_file,
        analyzed_module_count=len(source_files),
        test_root_exists=test_root.exists(),
        modules=tuple(modules),
    )


def find_companion_tests(
    module_path: Path | str,
    test_root: Path | str = Path("tests"),
) -> list[Path]:
    """Find likely pytest files that exercise a source module."""
    module_path = Path(module_path)
    test_root = Path(test_root)

    if not test_root.exists():
        return []

    candidate_stems = _candidate_test_stems(module_path)
    matches: list[Path] = []
    for candidate in sorted(path for path in test_root.rglob("*.py") if path.is_file()):
        test_stem = _pytest_test_stem(candidate)
        if test_stem and test_stem in candidate_stems:
            matches.append(candidate)
    return matches


def _module_name_for_path(source_root: Path, module_path: Path) -> str:
    parts = (source_root.name, *module_path.with_suffix("").parts)
    return ".".join(parts)


def _candidate_test_stems(module_path: Path) -> set[str]:
    module_parts = module_path.with_suffix("").parts
    if not module_parts:
        return set()

    stems = {
        module_parts[-1],
        "_".join(module_parts),
    }
    if len(module_parts) > 1:
        stems.add(module_parts[-2])
    return stems


def _pytest_test_stem(path: Path) -> str | None:
    name = path.name
    if name.startswith("test_") and name.endswith(".py"):
        return name[5:-3]
    if name.endswith("_test.py"):
        return name[:-8]
    return None
