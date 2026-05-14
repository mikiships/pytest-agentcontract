# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure: `uv run agentcontract --help`
2. This project uses `argparse` and exposes the console script through `pyproject.toml`.
3. There is no `src/agentcontract/__main__.py`, so use the console script unless the task adds module execution support.
4. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Print user-facing errors to stderr
   - Preserve existing plain-text output unless the task explicitly adds a new output mode
5. Test CLI changes with: `uv run agentcontract <command> --help`
6. Add or update tests in `tests/unit/test_cli.py` for behavior changes.
7. If adding a new command, add it as an `argparse` subparser in `cli.py`.
