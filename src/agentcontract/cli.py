"""CLI entry point for agentcontract commands."""

from __future__ import annotations

import argparse
import glob
import json
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

    # scan-pii command
    scan_pii_parser = subparsers.add_parser(
        "scan-pii",
        help="Scan cassette files for potential PII exposure",
    )
    scan_pii_parser.add_argument(
        "paths",
        nargs="+",
        help="Cassette path(s), directory path(s), or glob pattern(s)",
    )
    scan_pii_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print machine-readable JSON output",
    )

    args = parser.parse_args(argv)

    if args.command == "info":
        return _cmd_info(args.path)
    elif args.command == "validate":
        return _cmd_validate(args.path)
    elif args.command == "init":
        return _cmd_init()
    elif args.command == "scan-pii":
        return _cmd_scan_pii(args.paths, json_output=args.json_output)
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
    - type: no_pii
      target: full_run
      block: [email, phone, ssn, credit_card]

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


def _cmd_scan_pii(paths: list[str], *, json_output: bool = False) -> int:
    """Scan cassette files for potential PII exposure."""
    from agentcontract.pii import PiiScanResult, scan_cassette_path

    result = PiiScanResult()
    try:
        scan_paths = _expand_scan_pii_paths(paths)
    except FileNotFoundError:
        print("Error: no cassette paths matched", file=sys.stderr)
        return 1

    for path in scan_paths:
        try:
            result.extend(scan_cassette_path(path))
        except FileNotFoundError:
            print(f"Error: {path} not found", file=sys.stderr)
            return 1
        except (OSError, ValueError, TypeError):
            print(f"Error: failed to scan '{path}'", file=sys.stderr)
            return 1

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
        return 1 if result.has_findings else 0

    cassette_count = len(result.scanned_files)
    if not result.has_findings:
        print(f"Scanned {cassette_count} cassette(s); no PII findings.")
        return 0

    print(f"PII findings: {result.finding_count} finding(s) across {cassette_count} cassette(s)")
    for finding in result.findings:
        cassette = finding.cassette_path or ""
        print(f"- {finding.category} {cassette} {finding.location}: {finding.snippet}")
    return 1


def _expand_scan_pii_paths(paths: list[str]) -> list[Path]:
    """Expand CLI path and glob inputs while preserving order."""
    expanded: list[Path] = []
    seen: set[str] = set()

    for raw_path in paths:
        if glob.has_magic(raw_path):
            matches = [Path(match) for match in sorted(glob.glob(raw_path, recursive=True))]
            if not matches:
                raise FileNotFoundError(raw_path)
        else:
            matches = [Path(raw_path)]

        for path in matches:
            key = str(path)
            if key not in seen:
                seen.add(key)
                expanded.append(path)

    if not expanded:
        raise FileNotFoundError
    return expanded


if __name__ == "__main__":
    sys.exit(main())
