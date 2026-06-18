# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure: `.venv/bin/python -m agentcontract --help`
2. This project uses `argparse` with subparsers. Check existing commands for parser patterns.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Print human-readable output unless the implementation explicitly adds another format
4. Test CLI changes with: `.venv/bin/python -m agentcontract <command> --help`
5. If adding a new command, add a subparser in `cli.py` and dispatch it from `main()`
