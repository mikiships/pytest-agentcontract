# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI source in `src/agentcontract/cli.py` and inspect help with `.venv/bin/agentcontract --help`.
2. This project uses `argparse` through `main(argv: list[str] | None = None) -> int`. Check existing subcommands for patterns.
3. CLI commands should:
   - Register subparsers with clear `--help` text
   - Return integer exit codes, with non-zero codes on failure
   - Print user-facing errors to `stderr`
   - Keep output human-readable unless adding and testing an explicit machine-readable flag
4. Test CLI changes with: `.venv/bin/agentcontract <command> --help` and `.venv/bin/pytest tests/unit/test_cli.py -q`.
5. If adding a new command, add its subparser and dispatch branch in `cli.py`.
