# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure with the installed console script: `agentcontract --help`
2. This project uses `argparse` in `src/agentcontract/cli.py`. Check existing subparsers for patterns.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Keep output stable enough for tests and docs
4. Test CLI changes with: `agentcontract <command> --help` when installed, or call `agentcontract.cli.main([...])` from tests.
5. If adding a new command, add a subparser in `cli.py` and cover the command in `tests/unit/test_cli.py`
