# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure: `uv run agentcontract --help`
2. This project uses `argparse`. Check existing subcommands for patterns.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Only document JSON output if a `--json` flag is implemented and tested
4. Test CLI changes with: `uv run agentcontract <command> --help`
5. There is no `src/agentcontract/__main__.py`; use the installed `agentcontract` script unless you add and test a module entrypoint.
6. If adding a new command, add it to the `argparse` subparsers in `cli.py`, update README CLI examples if user-facing, and add coverage in `tests/unit/test_cli.py`
