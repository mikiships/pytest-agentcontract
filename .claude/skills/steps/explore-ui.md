# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure: `agentcontract --help`
2. This project uses `argparse` for CLI parsing. Check existing commands for patterns.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Print user-facing errors to stderr
4. Test CLI changes with: `agentcontract <command> --help`
5. If adding a new command, add it to the argparse subparsers in `cli.py`
