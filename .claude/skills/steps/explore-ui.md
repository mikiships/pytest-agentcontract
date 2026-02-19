# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure: `python -m agentcontract --help`
2. This project uses Click for CLI. Check existing commands for patterns.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Output JSON when `--json` flag is present
4. Test CLI changes with: `.venv/bin/python -m agentcontract <command> --help`
5. If adding a new command, add it to the Click group in `cli.py`
