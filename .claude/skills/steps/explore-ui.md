# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure: `.venv/bin/agentcontract --help`
2. This project uses `argparse`, not Click. Check `main()` and the existing subparser setup for patterns.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Preserve the current text output unless the task explicitly adds a new interface
4. Test CLI changes with:
   - `.venv/bin/agentcontract --help`
   - `.venv/bin/agentcontract info tests/scenarios/refund-eligible.agentrun.json`
   - `.venv/bin/agentcontract validate tests/scenarios/refund-eligible.agentrun.json`
5. Validate the repo with:
   - `.venv/bin/pytest tests/ -x -q`
   - `.venv/bin/ruff check src/ tests/`
   - `.venv/bin/mypy src/`
6. If adding a new command, register it in the `argparse` subparsers in `cli.py`
