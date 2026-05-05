# Explore: UI/CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current CLI structure: `agentcontract --help`
2. This project uses `argparse` in `src/agentcontract/cli.py`. Check the existing parser, subparsers, and command dispatch for patterns.
3. CLI commands should:
   - Have clear `--help` text
   - Return non-zero exit codes on failure
   - Print user-facing errors to stderr
   - Cover both text and JSON behavior if you add a `--json` flag
4. Test CLI changes with the installed entrypoint: `.venv/bin/agentcontract <command> --help`
5. If the entrypoint is not installed, use the module path that exists: `.venv/bin/python -m agentcontract.cli <command> --help`
6. Do not use `python -m agentcontract` unless a package-level `src/agentcontract/__main__.py` has been added.
7. If adding a new command, add a subparser in `main()` and a dispatch branch for the parsed command.
