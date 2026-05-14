# /do-work

Entrypoint for all feature and bugfix work on pytest-agentcontract.

## Steps

### 1. Plan
- Read the task/prompt carefully
- Identify which files need to change
- If the change touches public API (anything in `__init__.py` exports), note it
- If unsure about approach, check existing patterns in nearby code first

### 2. Explore
- Read the files you'll modify. Don't guess at structure.
- Use `README.md` as the source of truth for advertised commands and workflows
- Check `tests/unit/` for existing test coverage of the area
- Check `examples/` when changing record/replay examples or public workflow docs
- If the change involves a new assertion type or adapter, read an existing one as a template

For frontend/UI work: see `.claude/skills/steps/explore-ui.md`
For new adapter work: see `.claude/skills/steps/explore-adapter.md`

### 3. Build
- Smallest possible diff that solves the problem
- Follow existing code style (check the file you're editing)
- Type hints on all public functions
- No untyped public Python functions
- If adding public API, update the appropriate `__init__.py` lazy export and `__all__`
- If adding a public adapter, register it in `src/agentcontract/adapters/__init__.py`

For refactoring: see `.claude/skills/steps/build-refactor.md`

### 4. Validate
- Run: `uv run pytest tests/ -x -q`
- Run: `uv run ruff check src/ tests/`
- Run: `uv run mypy src/`
- For record/replay changes, also run targeted examples with `uv run pytest --ac-record` or `uv run pytest --ac-replay` as appropriate
- For CLI changes, smoke-test the console script, for example `uv run agentcontract --help`
- If tests or lint fail, fix them before proceeding
- If mypy has known baseline failures, document the existing failures and do not add new ones
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
