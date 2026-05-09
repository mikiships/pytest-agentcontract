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

    # scan-pii command
    scan_pii_parser = subparsers.add_parser(
        "scan-pii", help="Scan cassette files for likely PII exposure"
    )
    scan_pii_parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more .agentrun.json files or directories to scan",
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    if args.command == "info":
        return _cmd_info(args.path)
    elif args.command == "validate":
        return _cmd_validate(args.path)
    elif args.command == "init":
        return _cmd_init()
    elif args.command == "scan-pii":
        return _cmd_scan_pii(args.paths)
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


def _cmd_scan_pii(paths: list[Path]) -> int:
    """Scan cassette files for likely PII exposure."""
    from agentcontract.pii import PiiFinding, PiiScanError, scan_file

    targets, errors = _discover_pii_targets(paths)
    findings_by_file: dict[Path, list[PiiFinding]] = {}

    for target in targets:
        try:
            findings = scan_file(target)
        except PiiScanError as exc:
            errors.append(f"Error: {exc}")
            continue

        if findings:
            findings_by_file[target] = findings

    if findings_by_file:
        total_findings = sum(len(findings) for findings in findings_by_file.values())
        total_files = len(findings_by_file)
        print(f"Potential PII found: {total_findings} finding(s) in {total_files} file(s).")
        for target, findings in findings_by_file.items():
            print(f"{target}:")
            for finding in findings:
                print(f"  - {finding.json_path} [{finding.kind}] {finding.preview}")
    elif not errors:
        if targets:
            print(f"No potential PII found in {len(targets)} cassette(s).")
        else:
            print("No .agentrun.json files found.")

    for error in errors:
        print(error, file=sys.stderr)

    if errors:
        return 2
    if findings_by_file:
        return 1
    return 0


def _discover_pii_targets(paths: list[Path]) -> tuple[list[Path], list[str]]:
    targets: list[Path] = []
    errors: list[str] = []
    seen: set[Path] = set()

    for path in paths:
        if not path.exists():
            errors.append(f"Error: {path} not found")
            continue

        if path.is_dir():
            candidates = sorted(candidate for candidate in path.rglob("*.agentrun.json"))
        elif path.is_file():
            candidates = [path]
        else:
            errors.append(f"Error: {path} is not a file or directory")
            continue

        for candidate in candidates:
            if not candidate.is_file():
                continue
            key = candidate.resolve()
            if key in seen:
                continue
            seen.add(key)
            targets.append(candidate)

    return targets, errors


if __name__ == "__main__":
    sys.exit(main())
