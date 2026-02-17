"""CLI entry point for agentcontract commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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

    args = parser.parse_args(argv)

    if args.command == "info":
        return _cmd_info(args.path)
    elif args.command == "validate":
        return _cmd_validate(args.path)
    elif args.command == "init":
        return _cmd_init()
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
    except Exception as e:
        print(f"Error: failed to read cassette '{path}': {e}", file=sys.stderr)
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
    except Exception as e:
        print(f"✗ Invalid cassette: {e}", file=sys.stderr)
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
    target.write_text(template)
    print(f"Created {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
