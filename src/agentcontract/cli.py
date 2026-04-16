"""CLI entry point for agentcontract commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="agentcontract",
        description="pytest-agentcontract: Deterministic CI tests for LLM agent trajectories",
    )
    subparsers = parser.add_subparsers(dest="command")

    # info command
    info_parser = subparsers.add_parser("info", help="Show cassette info")
    info_parser.add_argument("path", type=Path, help="Path to .agentrun.json file")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a cassette file")
    validate_parser.add_argument("path", type=Path, help="Path to .agentrun.json file")

    # init command
    subparsers.add_parser("init", help="Create a starter agentcontract.yml")

    # gaps command
    gaps_parser = subparsers.add_parser(
        "gaps",
        help="Report weakly tested modules from local coverage data",
    )
    gaps_parser.add_argument(
        "--coverage-file",
        type=Path,
        default=Path(".coverage"),
        help="Path to a coverage.py data file (default: .coverage)",
    )
    gaps_parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("src/agentcontract"),
        help="Source tree to inspect (default: src/agentcontract)",
    )
    gaps_parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("tests"),
        help="Test tree used for companion-test matching (default: tests)",
    )
    gaps_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=10,
        help="Maximum modules to show (default: 10)",
    )

    args = parser.parse_args(argv)

    if args.command == "info":
        return _cmd_info(args.path)
    elif args.command == "validate":
        return _cmd_validate(args.path)
    elif args.command == "init":
        return _cmd_init()
    elif args.command == "gaps":
        return _cmd_gaps(
            coverage_file=args.coverage_file,
            source_root=args.source_root,
            test_root=args.test_root,
            limit=args.limit,
        )
    else:
        parser.print_help()
        return 0


def _cmd_info(path: Path) -> int:
    """Print summary info about a cassette."""
    from agentcontract.serialization import load_run

    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        return 1

    try:
        run = load_run(path)
        print(f"Scenario:    {run.metadata.scenario}")
        print(f"Run ID:      {run.run_id}")
        print(f"Recorded:    {run.recorded_at}")
        print(f"Model:       {run.model.provider}/{run.model.model}")
        print(f"Turns:       {run.summary.total_turns}")
        print(f"Tool calls:  {run.summary.total_tool_calls}")
        print(f"Duration:    {run.summary.total_duration_ms:.0f}ms")
        print(f"Tokens:      {run.summary.total_tokens.total}")
        print(f"Est. cost:   ${run.summary.estimated_cost_usd:.4f}")
        return 0
    except (OSError, ValueError, TypeError) as e:
        print(
            f"Error: failed to read cassette '{path}' ({type(e).__name__}): {e}",
            file=sys.stderr,
        )
        return 1


def _cmd_validate(path: Path) -> int:
    """Validate a cassette file structure."""
    from agentcontract.serialization import load_run

    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        return 1

    try:
        run = load_run(path)
        print(f"✓ Valid cassette: {run.metadata.scenario} ({len(run.turns)} turns)")
        return 0
    except (OSError, ValueError, TypeError) as e:
        print(f"✗ Invalid cassette ({type(e).__name__}): {e}", file=sys.stderr)
        return 1


def _cmd_init() -> int:
    """Create a starter agentcontract.yml in the current directory."""
    target = Path("agentcontract.yml")
    if target.exists():
        print(f"Error: {target} already exists", file=sys.stderr)
        return 1

    template = """\
version: "1"

scenarios:
  include: ["tests/scenarios/**/*.agentrun.json"]

replay:
  stub_tools: true
  concurrency: 5

defaults:
  assertions:
    - type: contains
      target: final_response
      value: ""  # customize this

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: []  # list your agent's tools here

budgets:
  per_scenario:
    max_cost_usd: 0.05
    max_turns: 15

reporting:
  github_comment: true
  artifact_path: "agentci-results/"
"""
    try:
        target.write_text(template)
    except OSError as e:
        print(f"Error: failed to write {target}: {e}", file=sys.stderr)
        return 1
    print(f"Created {target}")
    return 0


def _cmd_gaps(coverage_file: Path, source_root: Path, test_root: Path, limit: int) -> int:
    """Print a ranked coverage gap report for source modules."""
    from agentcontract.test_gap import CoverageDataError, analyze_test_gaps

    try:
        report = analyze_test_gaps(
            coverage_file=coverage_file,
            source_root=source_root,
            test_root=test_root,
        )
    except CoverageDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    display_count = min(limit, report.module_count)
    print(f"Coverage gaps from {_display_path(report.coverage_file)}")
    print(f"Source root: {_display_path(report.source_root)}")
    print(f"Test root:   {_display_path(report.test_root)}")

    if report.analyzed_module_count == 0:
        print("\nNo Python modules found under the requested source root.")
        return 0

    if display_count == 0:
        print(f"\nNo coverage gaps found across {report.analyzed_module_count} modules.")
        return 0

    print(f"\nTop {display_count} modules by uncovered lines:")
    print(f"{'Module':<40} {'Cov':>6} {'Miss':>6}  Tests")
    for module in report.modules[:display_count]:
        tests = (
            ", ".join(_display_path(test_path).name for test_path in module.obvious_test_paths)
            if module.has_obvious_tests
            else "none"
        )
        print(
            f"{module.module_name:<40} {module.coverage_percent:>5.1f}% "
            f"{module.missing_line_count:>6}  {tests}"
        )

    without_tests = report.modules_without_obvious_tests
    if without_tests:
        names = ", ".join(module.module_name for module in without_tests[:5])
        suffix = " ..." if len(without_tests) > 5 else ""
        print(f"\nNo obvious companion tests: {names}{suffix}")

    print(
        "\nSummary: "
        f"{report.module_count} modules with gaps, "
        f"{report.total_missing_line_count} uncovered lines, "
        f"{len(without_tests)} modules without obvious companion tests"
    )
    return 0


def _display_path(path: Path) -> Path:
    """Display paths relative to the current working directory when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return resolved


if __name__ == "__main__":
    sys.exit(main())
