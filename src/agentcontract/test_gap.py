"""Static test-gap finder for the agentcontract source tree."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path("src/agentcontract")
TESTS_DIR = Path("tests/unit")
HOTSPOT_MODULES = {"cli", "plugin"}
CLI_COMMAND_PATTERN = re.compile(r'add_parser\("([^"]+)"')
CLI_TEST_HIT_PATTERN = re.compile(
    r"""\bmain\(\s*[\[(]\s*["']([^"']+)["']""",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ModuleGap:
    """Coverage signal for a source module."""

    module_name: str
    source_path: Path
    expected_tests: tuple[Path, ...]
    matched_tests: tuple[Path, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TestGapReport:
    """Structured result for static test-gap inspection."""

    root: Path
    package_dir: Path
    tests_dir: Path
    scanned_modules: tuple[str, ...]
    scanned_test_files: tuple[Path, ...]
    missing: tuple[ModuleGap, ...]
    weak: tuple[ModuleGap, ...]

    @property
    def has_gaps(self) -> bool:
        """Return True when missing or weak coverage signals were found."""
        return bool(self.missing or self.weak)


def find_test_gaps(root: Path) -> TestGapReport:
    """Compare source modules to unit-test layout and report obvious gaps."""
    repo_root = root.resolve()
    package_dir = repo_root / PACKAGE_DIR
    tests_dir = repo_root / TESTS_DIR

    if not package_dir.is_dir():
        raise ValueError(f"package directory not found: {package_dir}")
    if not tests_dir.is_dir():
        raise ValueError(f"test directory not found: {tests_dir}")

    test_files = tuple(sorted(tests_dir.rglob("test_*.py")))
    test_lookup = {path.relative_to(tests_dir).as_posix(): path for path in test_files}

    modules = tuple(_discover_source_modules(package_dir))
    package_counts = _count_top_level_packages(package_dir, modules)

    missing: list[ModuleGap] = []
    weak: list[ModuleGap] = []

    for module_path in modules:
        relative = module_path.relative_to(package_dir)
        module_name = ".".join(("agentcontract", *relative.with_suffix("").parts))
        candidates = tuple(_expected_test_candidates(relative))
        matched = tuple(
            sorted(
                (
                    test_lookup[candidate.as_posix()]
                    for candidate in candidates
                    if candidate.as_posix() in test_lookup
                ),
                key=lambda path: path.relative_to(tests_dir).as_posix(),
            )
        )

        notes = tuple(
            _coverage_notes(
                module_path=module_path,
                relative_module_path=relative,
                tests_dir=tests_dir,
                matched_tests=matched,
                package_counts=package_counts,
            )
        )
        gap = ModuleGap(
            module_name=module_name,
            source_path=module_path,
            expected_tests=candidates,
            matched_tests=matched,
            notes=notes,
        )
        if not matched:
            missing.append(gap)
        elif notes:
            weak.append(gap)

    return TestGapReport(
        root=repo_root,
        package_dir=package_dir,
        tests_dir=tests_dir,
        scanned_modules=tuple(
            sorted(
                ".".join(("agentcontract", *path.relative_to(package_dir).with_suffix("").parts))
                for path in modules
            )
        ),
        scanned_test_files=tuple(sorted(test_files)),
        missing=tuple(missing),
        weak=tuple(weak),
    )


def format_test_gap_report(report: TestGapReport) -> str:
    """Render a human-readable report for the CLI."""
    lines = [
        "Scanned "
        f"{len(report.scanned_modules)} source modules against "
        f"{len(report.scanned_test_files)} unit test files.",
    ]
    if not report.has_gaps:
        lines.append("No structural test gaps found.")
        return "\n".join(lines)

    if report.missing:
        lines.append("Missing coverage:")
        for gap in report.missing:
            expected = ", ".join(candidate.as_posix() for candidate in gap.expected_tests[:3])
            lines.append(
                f"- {gap.module_name} ({gap.source_path.relative_to(report.root).as_posix()})"
                f" -> expected {expected}"
            )

    if report.weak:
        lines.append("Weak coverage signals:")
        for gap in report.weak:
            lines.append(
                f"- {gap.module_name} ({gap.source_path.relative_to(report.root).as_posix()}): "
                + "; ".join(gap.notes)
            )

    lines.append(
        "v1 limitation: this checks structural test coverage, not line or branch coverage."
    )
    return "\n".join(lines)


def _discover_source_modules(package_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(package_dir.rglob("*.py"))
        if "__pycache__" not in path.parts and path.name != "__init__.py"
    ]


def _count_top_level_packages(package_dir: Path, modules: tuple[Path, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module_path in modules:
        relative = module_path.relative_to(package_dir).parts
        key = relative[0] if len(relative) > 1 else module_path.stem
        counts[key] = counts.get(key, 0) + 1
    return counts


def _expected_test_candidates(relative_module_path: Path) -> tuple[Path, ...]:
    module_parts = relative_module_path.with_suffix("").parts
    stem = module_parts[-1]
    top_level = module_parts[0]

    candidates = {
        Path(f"test_{stem}.py"),
        Path(f"test_{top_level}.py"),
        Path(f"test_{'_'.join(module_parts)}.py"),
    }
    if len(module_parts) > 1:
        candidates.add(Path(*module_parts[:-1]) / f"test_{stem}.py")

    return tuple(sorted(candidates, key=lambda path: path.as_posix()))


def _coverage_notes(
    *,
    module_path: Path,
    relative_module_path: Path,
    tests_dir: Path,
    matched_tests: tuple[Path, ...],
    package_counts: dict[str, int],
) -> list[str]:
    notes: list[str] = []
    module_key = relative_module_path.stem

    if module_key in HOTSPOT_MODULES:
        hotspot_notes = _hotspot_notes(
            module_key=module_key,
            module_path=module_path,
            tests_dir=tests_dir,
            matched_tests=matched_tests,
        )
        notes.extend(hotspot_notes)

    if matched_tests and _shared_package_only(
        relative_module_path, tests_dir, matched_tests, package_counts
    ):
        shared = ", ".join(path.relative_to(tests_dir).as_posix() for path in matched_tests)
        notes.append(f"only matched shared package-level test file(s): {shared}")

    return sorted(set(notes))


def _hotspot_notes(
    *,
    module_key: str,
    module_path: Path,
    tests_dir: Path,
    matched_tests: tuple[Path, ...],
) -> list[str]:
    if module_key == "plugin":
        if _has_dedicated_test_match(module_key, tests_dir, matched_tests):
            return []
        return ["pytest plugin entry points do not have a matching unit test file"]

    if module_key != "cli":
        return []

    covered_commands = _covered_cli_commands(matched_tests)
    declared_commands = _declared_cli_commands(module_path)
    missing_commands = sorted(
        command for command in declared_commands if command not in covered_commands
    )
    if missing_commands:
        return [f"CLI subcommands without explicit test hits: {', '.join(missing_commands)}"]

    if not matched_tests:
        return ["CLI entry points do not have a matching unit test file"]

    return []


def _shared_package_only(
    relative_module_path: Path,
    tests_dir: Path,
    matched_tests: tuple[Path, ...],
    package_counts: dict[str, int],
) -> bool:
    if len(relative_module_path.parts) <= 1:
        return False

    top_level = relative_module_path.parts[0]
    if package_counts.get(top_level, 0) <= 1:
        return False

    module_import_path = ".".join(("agentcontract", *relative_module_path.with_suffix("").parts))
    package_test = Path(f"test_{top_level}.py")
    dedicated_matches = {
        Path(f"test_{relative_module_path.stem}.py"),
        Path(f"test_{'_'.join(relative_module_path.with_suffix('').parts)}.py"),
        Path(*relative_module_path.parts[:-1]) / f"test_{relative_module_path.stem}.py",
    }
    relative_matches = {path.relative_to(tests_dir) for path in matched_tests}
    if package_test not in relative_matches or not relative_matches.isdisjoint(dedicated_matches):
        return False

    package_level_matches = [
        path for path in matched_tests if path.relative_to(tests_dir) == package_test
    ]
    return not any(
        _test_file_explicitly_targets_module(test_path, module_import_path)
        for test_path in package_level_matches
    )


def _declared_cli_commands(cli_path: Path) -> set[str]:
    return set(CLI_COMMAND_PATTERN.findall(cli_path.read_text()))


def _covered_cli_commands(matched_tests: tuple[Path, ...]) -> set[str]:
    covered: set[str] = set()
    for test_path in matched_tests:
        contents = test_path.read_text()
        for command in CLI_TEST_HIT_PATTERN.findall(contents):
            covered.add(command)
    return covered


def _has_dedicated_test_match(
    module_key: str,
    tests_dir: Path,
    matched_tests: tuple[Path, ...],
) -> bool:
    expected_name = Path(f"test_{module_key}.py")
    return any(path.relative_to(tests_dir) == expected_name for path in matched_tests)


def _test_file_explicitly_targets_module(test_path: Path, module_import_path: str) -> bool:
    try:
        tree = ast.parse(test_path.read_text(), filename=str(test_path))
    except (OSError, SyntaxError):
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_import_path or alias.name.startswith(
                    module_import_path + "."
                ):
                    return True
        elif isinstance(node, ast.ImportFrom) and (
            node.module == module_import_path
            or (node.module and node.module.startswith(module_import_path + "."))
        ):
                return True

    return False
