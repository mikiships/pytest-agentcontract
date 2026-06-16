"""Coverage-based reporting for weakly tested modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_GENERIC_MODULE_NAMES = frozenset({"core", "engine"})


class CoverageDataError(RuntimeError):
    """Raised when coverage data cannot be loaded or analyzed."""


@dataclass(frozen=True, slots=True)
class ModuleGap:
    """Coverage gap details for a single source module."""

    module_name: str
    source_path: Path
    statement_count: int
    missing_lines: tuple[int, ...]
    obvious_test_paths: tuple[Path, ...]

    @property
    def executed_line_count(self) -> int:
        """Return the number of covered statements."""
        return self.statement_count - self.missing_line_count

    @property
    def missing_line_count(self) -> int:
        """Return the number of uncovered statements."""
        return len(self.missing_lines)

    @property
    def coverage_percent(self) -> float:
        """Return line coverage as a percentage."""
        if self.statement_count == 0:
            return 100.0
        return (self.executed_line_count / self.statement_count) * 100.0

    @property
    def has_obvious_tests(self) -> bool:
        """Whether a likely companion test file was found."""
        return bool(self.obvious_test_paths)


@dataclass(frozen=True, slots=True)
class GapReport:
    """Deterministic summary of coverage gaps for a source tree."""

    coverage_file: Path
    source_root: Path
    test_root: Path
    modules: tuple[ModuleGap, ...]

    @property
    def module_count(self) -> int:
        """Return the number of analyzed modules."""
        return len(self.modules)

    @property
    def total_statement_count(self) -> int:
        """Return the total statement count across analyzed modules."""
        return sum(module.statement_count for module in self.modules)

    @property
    def total_missing_line_count(self) -> int:
        """Return the total number of uncovered statements."""
        return sum(module.missing_line_count for module in self.modules)

    @property
    def modules_without_obvious_tests(self) -> tuple[ModuleGap, ...]:
        """Return modules lacking a clearly named companion test file."""
        return tuple(module for module in self.modules if not module.has_obvious_tests)


def analyze_test_gaps(
    coverage_file: Path | str = Path(".coverage"),
    source_root: Path | str = Path("src/agentcontract"),
    test_root: Path | str = Path("tests"),
) -> GapReport:
    """Analyze coverage data and rank under-tested modules."""
    coverage_path = Path(coverage_file).resolve()
    source_root_path = Path(source_root).resolve()
    test_root_path = Path(test_root).resolve()

    if not source_root_path.is_dir():
        raise CoverageDataError(f"source root not found: {source_root_path}")
    if not test_root_path.is_dir():
        raise CoverageDataError(f"test root not found: {test_root_path}")

    coverage = _load_coverage(coverage_path)
    test_files = _discover_test_files(test_root_path)
    try:
        modules = [
            _build_module_gap(
                coverage=coverage,
                source_root=source_root_path,
                source_path=source_path,
                test_files=test_files,
            )
            for source_path in _discover_source_modules(source_root_path)
        ]
    finally:
        _close_coverage(coverage)

    ranked_modules = tuple(sorted(modules, key=_gap_sort_key))
    return GapReport(
        coverage_file=coverage_path,
        source_root=source_root_path,
        test_root=test_root_path,
        modules=ranked_modules,
    )


def find_companion_tests(
    source_path: Path | str,
    source_root: Path | str = Path("src/agentcontract"),
    test_root: Path | str = Path("tests"),
) -> tuple[Path, ...]:
    """Find clearly named companion tests for a source module."""
    source_path_obj = Path(source_path).resolve()
    source_root_path = Path(source_root).resolve()
    test_root_path = Path(test_root).resolve()

    if not source_root_path.is_dir():
        raise CoverageDataError(f"source root not found: {source_root_path}")
    if not test_root_path.is_dir():
        raise CoverageDataError(f"test root not found: {test_root_path}")

    return _match_test_files(
        source_path=source_path_obj,
        source_root=source_root_path,
        test_files=_discover_test_files(test_root_path),
    )


def _load_coverage(coverage_path: Path):
    try:
        from coverage import Coverage
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise CoverageDataError(
            "coverage.py is required to analyze test gaps; install coverage or pytest-cov"
        ) from exc

    if not coverage_path.exists():
        raise CoverageDataError(f"coverage data not found: {coverage_path}")

    coverage = Coverage(data_file=str(coverage_path))
    try:
        coverage.load()
    except Exception as exc:  # pragma: no cover - exact exception varies by backend
        _close_coverage(coverage)
        raise CoverageDataError(
            f"failed to load coverage data from {coverage_path}: {exc}"
        ) from exc
    return coverage


def _discover_source_modules(source_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path.resolve()
            for path in source_root.rglob("*.py")
            if path.is_file() and path.name != "__init__.py"
        )
    )


def _discover_test_files(test_root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path.resolve() for path in test_root.rglob("test_*.py") if path.is_file()))


def _build_module_gap(
    coverage,
    source_root: Path,
    source_path: Path,
    test_files: tuple[Path, ...],
) -> ModuleGap:
    try:
        _, statements, _, missing, _ = coverage.analysis2(str(source_path))
    except Exception as exc:  # pragma: no cover - exact exception varies by backend
        raise CoverageDataError(f"failed to analyze coverage for {source_path}: {exc}") from exc

    return ModuleGap(
        module_name=_module_name(source_path, source_root),
        source_path=source_path,
        statement_count=len(statements),
        missing_lines=tuple(sorted(missing)),
        obvious_test_paths=_match_test_files(
            source_path=source_path,
            source_root=source_root,
            test_files=test_files,
        ),
    )


def _match_test_files(
    source_path: Path,
    source_root: Path,
    test_files: tuple[Path, ...],
) -> tuple[Path, ...]:
    candidate_stems = _candidate_test_stems(source_path.relative_to(source_root))
    return tuple(test_file for test_file in test_files if test_file.stem in candidate_stems)


def _candidate_test_stems(relative_source_path: Path) -> frozenset[str]:
    stem = relative_source_path.stem
    candidates = {f"test_{stem}"}
    if stem in _GENERIC_MODULE_NAMES and relative_source_path.parent.name:
        candidates.add(f"test_{relative_source_path.parent.name}")
    return frozenset(candidates)


def _module_name(source_path: Path, source_root: Path) -> str:
    relative_path = source_path.relative_to(source_root).with_suffix("")
    return ".".join((source_root.name, *relative_path.parts))


def _gap_sort_key(module: ModuleGap) -> tuple[float | bool | str, ...]:
    return (
        -module.missing_line_count,
        module.coverage_percent,
        module.has_obvious_tests,
        module.module_name,
    )


def _close_coverage(coverage) -> None:
    try:
        coverage.get_data().close()
    except Exception:  # pragma: no cover - cleanup best effort
        return
