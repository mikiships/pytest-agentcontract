# Explore: CLI Work

Use this reference when a task changes user-visible command-line behavior.

1. Read `src/agentcontract/cli.py`.
2. The CLI currently uses Python `argparse`, with the console script configured in `pyproject.toml` as `agentcontract = "agentcontract.cli:main"`.
3. Check `tests/unit/test_cli.py` for existing coverage.
4. CLI commands should:
   - Have clear help text.
   - Return non-zero exit codes on failure.
   - Print user-facing errors to stderr.
   - Keep output compatible with the existing text-based commands unless the task explicitly asks for a new output mode.
5. Test command help with `agentcontract <command> --help` after installing the package, or `.venv/bin/agentcontract <command> --help` when using the local virtualenv.
6. If adding a new command, add a subparser in `src/agentcontract/cli.py` and cover it in `tests/unit/test_cli.py`.
