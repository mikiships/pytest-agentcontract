"""Coverage gap analysis for agentcontract source modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_GENERIC_MODULE_NAMES = frozenset({"core", "engine"})


class CoverageDataError(RuntimeError):
    """Raised when local coverage data cannot be analyzed."""


@dataclass(frozen=True, slots=True)
class ModuleGap:
    """Coverage details for a single source module with missing lines."""

    module_name: str
    source_path: Path
    statement_count: int
    executed_line_count: int
    missing_line_count: int
    coverage_percent: float
    missing_lines: tuple[int, ...]
    companion_tests: tuple[Path, ...]

    @property
    def has_companion_tests(self) -> bool:
        """Whether the analyzer found plausible matching pytest files."""
        return bool(self.companion_tests)


@dataclass(frozen=True, slots=True)
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
    source_root_path = Path(source_root)
    test_root_path = Path(test_root)
    coverage_file_path = Path(coverage_file)

    if not source_root_path.is_dir():
        raise CoverageDataError(f"source root not found: {source_root_path}")
    if not coverage_file_path.exists():
        raise CoverageDataError(f"coverage data file not found: {coverage_file_path}")

    source_files = _discover_source_modules(source_root_path)
    if not source_files:
        raise CoverageDataError(f"no Python modules found under {source_root_path}")

    test_files = _discover_test_files(test_root_path) if test_root_path.is_dir() else ()
    coverage = _load_coverage(coverage_file_path)
    try:
        modules = [
            gap
            for source_path in source_files
            if (gap := _build_module_gap(coverage, source_root_path, source_path, test_files))
            is not None
        ]
    finally:
        _close_coverage(coverage)

    modules.sort(key=_gap_sort_key)
    return TestGapSummary(
        source_root=source_root_path,
        test_root=test_root_path,
        coverage_file=coverage_file_path,
        analyzed_module_count=len(source_files),
        test_root_exists=test_root_path.is_dir(),
        modules=tuple(modules),
    )


def find_companion_tests(
    source_path: Path | str,
    source_root: Path | str = Path("src/agentcontract"),
    test_root: Path | str = Path("tests"),
) -> tuple[Path, ...]:
    """Find likely pytest files that exercise a source module."""
    source_root_path = Path(source_root)
    test_root_path = Path(test_root)

    if not test_root_path.is_dir():
        return ()

    relative_source_path = _relative_source_path(Path(source_path), source_root_path)
    return _match_test_files(relative_source_path, _discover_test_files(test_root_path))


def _load_coverage(coverage_file: Path):
    try:
        from coverage import Coverage
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise CoverageDataError(
            "coverage.py is required to analyze test gaps; install coverage or pytest-cov"
        ) from exc

    coverage = Coverage(data_file=str(coverage_file))
    try:
        coverage.load()
    except Exception as exc:  # pragma: no cover - backend-specific failure shapes vary
        _close_coverage(coverage)
        raise CoverageDataError(
            f"failed to load coverage data from {coverage_file}: {exc}"
        ) from exc
    return coverage


def _discover_source_modules(source_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in source_root.rglob("*.py")
            if path.is_file() and path.name != "__init__.py"
        )
    )


def _discover_test_files(test_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path.relative_to(test_root)
            for path in test_root.rglob("*.py")
            if path.is_file() and _pytest_test_stem(path) is not None
        )
    )


def _build_module_gap(
    coverage,
    source_root: Path,
    source_path: Path,
    test_files: tuple[Path, ...],
) -> ModuleGap | None:
    try:
        _, statements, _, missing, _ = coverage.analysis2(str(source_path))
    except Exception as exc:  # pragma: no cover - backend-specific failure shapes vary
        raise CoverageDataError(
            f"failed to analyze coverage for {source_path}: {exc}"
        ) from exc

    statement_count = len(statements)
    if statement_count == 0:
        return None

    missing_lines = tuple(sorted(missing))
    missing_line_count = len(missing_lines)
    if missing_line_count == 0:
        return None

    relative_source_path = source_path.relative_to(source_root)
    executed_line_count = statement_count - missing_line_count
    return ModuleGap(
        module_name=_module_name_for_path(source_root, relative_source_path),
        source_path=relative_source_path,
        statement_count=statement_count,
        executed_line_count=executed_line_count,
        missing_line_count=missing_line_count,
        coverage_percent=(executed_line_count / statement_count) * 100.0,
        missing_lines=missing_lines,
        companion_tests=_match_test_files(relative_source_path, test_files),
    )


def _relative_source_path(source_path: Path, source_root: Path) -> Path:
    try:
        return source_path.relative_to(source_root)
    except ValueError:
        pass

    source_root_path = source_root.resolve()
    source_path_obj = source_path.resolve()
    try:
        return source_path_obj.relative_to(source_root_path)
    except ValueError as exc:
        if not source_path.is_absolute():
            return source_path
        raise CoverageDataError(
            f"source path {source_path_obj} is not under source root {source_root_path}"
        ) from exc


def _match_test_files(
    relative_source_path: Path,
    test_files: tuple[Path, ...],
) -> tuple[Path, ...]:
    candidate_stems = _candidate_test_stems(relative_source_path)
    return tuple(
        test_file
        for test_file in test_files
        if (test_stem := _pytest_test_stem(test_file)) is not None and test_stem in candidate_stems
    )


def _candidate_test_stems(relative_source_path: Path) -> frozenset[str]:
    module_parts = relative_source_path.with_suffix("").parts
    if not module_parts:
        return frozenset()

    module_name = module_parts[-1]
    candidates = {module_name, "_".join(module_parts)}
    if len(module_parts) > 1:
        candidates.add(module_parts[-2])
        if module_name in _GENERIC_MODULE_NAMES:
            candidates.discard(module_name)
    return frozenset(candidates)


def _pytest_test_stem(path: Path) -> str | None:
    name = path.name
    if name.startswith("test_") and name.endswith(".py"):
        return name[5:-3]
    if name.endswith("_test.py"):
        return name[:-8]
    return None


def _module_name_for_path(source_root: Path, relative_source_path: Path) -> str:
    return ".".join((source_root.name, *relative_source_path.with_suffix("").parts))


def _gap_sort_key(module: ModuleGap) -> tuple[float | str, ...]:
    return (
        -module.missing_line_count,
        module.coverage_percent,
        module.module_name,
    )


def _close_coverage(coverage) -> None:
    try:
        data = coverage.get_data()
    except Exception:  # pragma: no cover - cleanup best effort
        return

    close = getattr(data, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - cleanup best effort
            return


__all__ = [
    "CoverageDataError",
    "ModuleGap",
    "TestGapSummary",
    "analyze_test_gaps",
    "find_companion_tests",
]
