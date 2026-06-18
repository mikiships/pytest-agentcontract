# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure: `.venv/bin/agentcontract --help`
2. This project uses `argparse` through the `agentcontract` console script in `pyproject.toml`. Follow the existing subparser and dispatch pattern.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Use plain text output unless you explicitly add and test a structured output flag
4. Test CLI changes with:
   - `.venv/bin/agentcontract --help`
   - `.venv/bin/agentcontract <command> --help`
   - `.venv/bin/agentcontract validate <cassette.agentrun.json>` for behavior changes
5. If adding a new command, add an `argparse` subparser and dispatch branch in `main()`
