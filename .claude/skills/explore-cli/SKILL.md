---
name: explore-cli
description: Guides pytest-agentcontract CLI exploration and changes in src/agentcontract/cli.py. Use when modifying agentcontract info, agentcontract validate, agentcontract init, argparse command behavior, CLI help text, exit codes, or user-facing terminal workflows.
---

# Explore: CLI Work

When modifying the CLI (`src/agentcontract/cli.py`):

1. Read the current `argparse` structure and the command handlers `_cmd_info`, `_cmd_validate`, and `_cmd_init`.
2. Keep the README-documented commands current:
   - `agentcontract info cassette.agentrun.json`
   - `agentcontract validate cassette.agentrun.json`
   - `agentcontract init`
3. CLI commands should:
   - Have clear `argparse` help text
   - Return integer exit codes through `main()`
   - Print failures to stderr and return non-zero exit codes on failure
   - Avoid documenting new flags until they are implemented and tested
4. Test CLI changes with the available local entry point:
   - `.venv/bin/agentcontract --help`
   - `.venv/bin/agentcontract <command> --help`
   - `.venv/bin/python -m agentcontract.cli <command> --help`
5. If adding a new command, add its parser, dispatch branch, README entry, and coverage in `tests/unit/test_cli.py`.
6. For cassette-based behavior, use files under `tests/scenarios/` as fixtures.
