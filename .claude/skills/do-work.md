# /do-work

Entrypoint for feature and bugfix work on pytest-agentcontract, a Python pytest plugin and CLI for recording, replaying, and asserting LLM agent trajectories.

## Steps

### 1. Plan
- Read the task/prompt carefully
- Identify which files need to change
- If the change touches public API (`src/agentcontract/__init__.py` exports, pytest plugin behavior, or CLI commands), note it
- If unsure about approach, check existing patterns in nearby code first

### 2. Explore
- Read the files you'll modify. Don't guess at structure.
- Check `tests/unit/` and `tests/scenarios/` for existing coverage of the area
- If the change involves a new assertion type or adapter, read an existing one as template
- Use `examples/customer_support/` as the current end-to-end example for record, replay, and assert behavior

For CLI work: see `.claude/skills/steps/explore-cli.md`
For new adapter work: see `.claude/skills/steps/explore-adapter.md`

### 3. Build
- Smallest possible diff that solves the problem
- Follow existing code style (check the file you're editing)
- Keep strict typing on public Python functions
- If adding a public module or symbol, update `src/agentcontract/__init__.py` exports only when the symbol is intended as public API
- Preserve README workflows: `pytest --ac-record`, `pytest --ac-replay`, `agentcontract validate`, and `agentcontract init`

For refactoring: see `.claude/skills/steps/build-refactor.md`

### 4. Validate
- Run focused pytest coverage for the touched area; default to `.venv/bin/pytest tests/ -x -q` when the local venv exists, otherwise `pytest tests/ -x -q`
- For replay/cassette behavior, exercise the README workflow with `pytest --ac-replay`; use `pytest --ac-record -k <test>` only when intentionally refreshing a cassette
- For changed or generated cassettes, run `agentcontract validate <cassette>`
- For broader source changes, also run the dev checks configured in `pyproject.toml` when available: `.venv/bin/ruff check src/ tests/` and `.venv/bin/mypy src/`
- If any test fails, fix it before proceeding
- If you added new public functionality, write at least one test (happy path + edge case)

### 5. Commit
- One commit per logical change
- Conventional commit format: `fix:`, `feat:`, `refactor:`, `test:`, `docs:`
- Message describes what changed, not what you did ("fix: handle None in replay engine" not "updated replay.py")
- Never commit failing tests or lint errors

## Rules
- Do NOT move utility functions between modules
- Do NOT refactor unless the task explicitly asks for it
- Do NOT touch files unrelated to the task
- If stuck for 5+ minutes, try a simpler approach
