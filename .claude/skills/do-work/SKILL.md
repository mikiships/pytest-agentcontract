---
name: do-work
description: Project workflow for implementing pytest-agentcontract feature, bugfix, CLI, adapter, refactoring, and validation changes. Use for repository tasks that require reading current code patterns, updating tests, running README-documented commands, and committing scoped changes.
---

# Project Work

Entrypoint for all feature and bugfix work on pytest-agentcontract.

## Steps

### 1. Plan
- Read the task/prompt carefully
- Identify which files need to change
- Use `README.md` as the primary source for user-facing commands, architecture, and workflow wording
- If the change touches public API (anything in `__init__.py` exports), note it
- If unsure about approach, check existing patterns in nearby code first

### 2. Explore
- Read the files you'll modify. Don't guess at structure.
- Check `tests/unit/` for existing test coverage of the area
- If the change involves a new assertion type or adapter, read an existing one as template
- For CLI work, use the `explore-cli` skill
- For new adapter work, use the `explore-adapter` skill

### 3. Build
- Smallest possible diff that solves the problem
- Follow existing code style (check the file you're editing)
- Type hints on all public functions
- No untyped public functions in Python
- If adding a new public module or symbol, add it to the appropriate `__init__.py` export

For refactoring, use the `build-refactor` skill.

### 4. Validate
- Record/replay workflows from the README:
  - `pytest --ac-record`
  - `pytest --ac-replay`
- CLI workflows from the README:
  - `agentcontract info cassette.agentrun.json`
  - `agentcontract validate cassette.agentrun.json`
  - `agentcontract init`
- Standard repository checks:
  - `.venv/bin/pytest tests/ -x -q`
  - `.venv/bin/ruff check src/ tests/`
  - `.venv/bin/mypy src/`
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
