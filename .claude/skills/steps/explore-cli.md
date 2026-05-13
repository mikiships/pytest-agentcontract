# Explore: CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the README CLI section and the current command structure: `.venv/bin/agentcontract --help`
2. This project uses `argparse` subcommands. Check existing parsers and helper functions for patterns.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Match the README command surface: `agentcontract info`, `agentcontract validate`, and `agentcontract init`
4. Test CLI changes with: `.venv/bin/agentcontract <command> --help`
5. Add or update focused coverage in `tests/unit/test_cli.py`
6. If adding a new command, add a subparser in `main()` and update the README CLI section
