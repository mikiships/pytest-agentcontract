# /do-work

Entrypoint for all feature and bugfix work on pytest-agentcontract.

## Steps

### 1. Plan
- Read the task/prompt carefully
- Identify which files need to change
- If the change touches public API (anything exported from `src/agentcontract/__init__.py`), note it
- If unsure about approach, check existing patterns in nearby code first

### 2. Explore
- Read the files you'll modify. Don't guess at structure.
- Check `tests/unit/` for existing test coverage of the area
- If the change involves a new assertion type or adapter, read an existing one as template

For frontend/UI work: see `.claude/skills/steps/explore-ui.md`
For new adapter work: see `.claude/skills/steps/explore-adapter.md`

### 3. Build
- Smallest possible diff that solves the problem
- Follow existing code style (check the file you're editing)
- Add type hints to new public functions unless the local code deliberately suppresses them
- Do not introduce untyped public Python APIs; avoid `Any` in new public APIs unless the existing pattern requires it
- If adding a new public module or symbol, update `src/agentcontract/__init__.py` only when the package exposes it as API

For refactoring: see `.claude/skills/steps/build-refactor.md`

### 4. Validate
- Run: `.venv/bin/pytest tests/ -x -q`
- Run: `.venv/bin/ruff check src/ tests/`
- Run: `.venv/bin/mypy src/`
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
