"""CLI entry point for agentcontract commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentcontract.test_gap import TestGapSummary


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
        help="Report the source modules with the weakest test coverage",
    )
    gaps_parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("src/agentcontract"),
        help="Directory containing source modules to analyze",
    )
    gaps_parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("tests"),
        help="Directory containing pytest files used for companion matching",
    )
    gaps_parser.add_argument(
        "--coverage-file",
        type=Path,
        default=Path(".coverage"),
        help="coverage.py data file to analyze",
    )
    gaps_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of modules to print",
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
            source_root=args.source_root,
            test_root=args.test_root,
            coverage_file=args.coverage_file,
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


def _cmd_gaps(
    source_root: Path,
    test_root: Path,
    coverage_file: Path,
    limit: int,
) -> int:
    """Print ranked source modules with missing coverage."""
    from agentcontract.test_gap import CoverageDataError, analyze_test_gaps

    if limit <= 0:
        print("Error: --limit must be greater than zero", file=sys.stderr)
        return 1

    try:
        summary = analyze_test_gaps(
            source_root=source_root,
            test_root=test_root,
            coverage_file=coverage_file,
        )
    except CoverageDataError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    _print_gap_summary(summary, limit)
    return 0


def _print_gap_summary(summary: TestGapSummary, limit: int) -> None:
    print(f"Coverage gaps in {_display_path(summary.source_root)}")
    print(f"Using coverage data from {_display_path(summary.coverage_file)}")

    if not summary.test_root_exists:
        print(f"Companion test scan skipped missing directory: {_display_path(summary.test_root)}")

    if not summary.modules:
        print("No coverage gaps found.")
        return

    visible_modules = summary.modules[:limit]
    if len(summary.modules) > limit:
        print(f"Showing top {len(visible_modules)} of {len(summary.modules)} modules with gaps")

    for gap in visible_modules:
        print(
            f"{gap.module_name}: {gap.coverage_percent:.1f}% covered "
            f"({gap.missing_line_count} missing / {gap.total_line_count} statements)"
        )
        print(f"  Source: {_display_path(gap.source_path)}")
        if gap.companion_tests:
            companion_list = ", ".join(_display_path(path) for path in gap.companion_tests)
            print(f"  Companion tests: {companion_list}")
        else:
            print("  Companion tests: none found")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
