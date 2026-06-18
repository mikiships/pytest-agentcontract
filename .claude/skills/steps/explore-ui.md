# Explore: CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure: `.venv/bin/python -m agentcontract --help`
2. This project uses `argparse`. Check existing subparsers and command helpers for patterns.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
4. Test CLI changes with: `.venv/bin/python -m agentcontract <command> --help`
5. If adding a new command, add an `argparse` subparser in `src/agentcontract/cli.py` and dispatch it from `main()`
