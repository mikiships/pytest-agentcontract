# /do-work

Entrypoint for all feature and bugfix work on pytest-agentcontract.

## Project context

- Python 3.10+ pytest plugin with source under `src/agentcontract/`
- Unit tests live in `tests/unit/`; cassette fixtures live in `tests/scenarios/`
- Examples live in `examples/`
- CLI entry point is the installed `agentcontract` command from `agentcontract.cli:main`
- Pytest modes are `pytest --ac-record` for recording and `pytest --ac-replay` for replay
- Framework adapters are exported from `agentcontract.adapters`: `record_graph`, `record_agent`, and `record_runner`

## Steps

### 1. Plan
- Read the task/prompt carefully
- Identify which files need to change
- If the change touches public API exports in `src/agentcontract/__init__.py` or `src/agentcontract/adapters/__init__.py`, note it
- If unsure about approach, check existing patterns in nearby code first

### 2. Explore
- Read the files you'll modify. Don't guess at structure.
- Use `README.md` and `pyproject.toml` as the primary references for supported commands, package layout, and tooling
- Check `tests/unit/` for existing test coverage of the area
- If the change involves a new assertion type or adapter, read an existing one as template

For frontend/UI work: see `.claude/skills/steps/explore-ui.md`
For new adapter work: see `.claude/skills/steps/explore-adapter.md`

### 3. Build
- Smallest possible diff that solves the problem
- Follow existing code style (check the file you're editing)
- Type hints on all public functions
- No untyped public functions in Python
- If adding a new public module or adapter, add the appropriate lazy export in `__init__.py`

For refactoring: see `.claude/skills/steps/build-refactor.md`

### 4. Validate
- Run: `pytest tests/ -x -q`
- Run: `ruff check src/ tests/`
- Run: `mypy src/`
- For record/replay behavior, use focused checks with `pytest --ac-record` and `pytest --ac-replay`
- For CLI behavior, check the installed command form: `agentcontract --help`, `agentcontract info`, `agentcontract validate`, or `agentcontract init`
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
