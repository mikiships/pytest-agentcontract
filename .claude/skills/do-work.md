# /do-work

Entrypoint for all feature and bugfix work on pytest-agentcontract.

## Project context
- Python 3.10+ pytest plugin with package code under `src/agentcontract/`
- Tests live under `tests/`; cassette fixtures live under `tests/scenarios/*.agentrun.json`
- Examples live under `examples/`
- README-advertised pytest modes are `pytest --ac-record` and `pytest --ac-replay`
- CLI entrypoint is the installed `agentcontract` script; current commands are `info`, `validate`, and `init`

## Steps

### 1. Plan
- Read the task/prompt carefully
- Identify which files need to change
- If the change touches public API, note the required `__init__.py` exports
- If unsure about approach, check existing patterns in nearby code first

### 2. Explore
- Read the files you'll modify. Don't guess at structure.
- Check `tests/` for existing test coverage of the area
- If the change involves a new assertion type or adapter, read an existing one as template

For frontend/UI work: see `.claude/skills/steps/explore-ui.md`
For new adapter work: see `.claude/skills/steps/explore-adapter.md`

### 3. Build
- Smallest possible diff that solves the problem
- Follow existing code style (check the file you're editing)
- Type hints on all public functions
- Avoid new `Any` unless the code is wrapping dynamic third-party SDK objects, as the adapters do
- If adding public package API, update `src/agentcontract/__init__.py` lazy imports and `__all__`
- If adding public adapter API, update `src/agentcontract/adapters/__init__.py` lazy imports and `__all__`

For refactoring: see `.claude/skills/steps/build-refactor.md`

### 4. Validate
- Run: `uv run pytest tests/ -x -q`
- Run: `uv run ruff check src/ tests/`
- Run: `uv run mypy src/`
- For record/replay changes, run the relevant targeted flow with `uv run pytest --ac-record -k <test>` and `uv run pytest --ac-replay -k <test>`
- For CLI changes, run `uv run agentcontract --help` and the relevant command help
- If any test fails because of your change, fix it before proceeding
- If a validation command reports a known baseline failure, record the exact command and failure summary
- If you added new public functionality, write at least one test (happy path + edge case)

### 5. Commit
- One commit per logical change
- Conventional commit format: `fix:`, `feat:`, `refactor:`, `test:`, `docs:`
- Message describes what changed, not what you did ("fix: handle None in replay engine" not "updated replay.py")
- Never commit newly introduced failing tests or lint errors

## Rules
- Do NOT move utility functions between modules
- Do NOT refactor unless the task explicitly asks for it
- Do NOT touch files unrelated to the task
- If stuck for 5+ minutes, try a simpler approach
