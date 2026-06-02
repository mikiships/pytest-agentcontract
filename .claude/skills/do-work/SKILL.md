---
name: do-work
description: Follow the pytest-agentcontract project workflow for feature work, bug fixes, tests, refactors, validation, and commits. Use when implementing code or documentation changes in this repository.
---

# Do Work

Entrypoint for feature, bugfix, test, refactor, and documentation work on pytest-agentcontract.

## Workflow

1. Plan
   - Read the task carefully.
   - Identify the smallest file set that needs to change.
   - If the change touches public API exports, especially `src/agentcontract/__init__.py`, note it before editing.
   - Prefer existing patterns in nearby code over new abstractions.

2. Explore
   - Read the files you will modify before editing.
   - Check `tests/unit/` for existing test coverage of the area.
   - If the change involves an assertion type or adapter, read the closest existing implementation first.
   - For CLI-visible work, read `references/explore-cli.md`.
   - For framework adapter work, read `references/explore-adapter.md`.

3. Build
   - Make the smallest diff that solves the task.
   - Follow the current code style in the file being edited.
   - Add type hints on public Python functions.
   - Avoid untyped public functions.
   - If adding a new public module or adapter entrypoint, update the relevant `__init__.py` exports.
   - For explicit refactoring tasks, read `references/build-refactor.md` before changing files.

4. Validate
   - For most code changes, run `pytest tests/ -x -q`; prefix with `.venv/bin/` when using the repo-local virtualenv.
   - Run `ruff check src/ tests/` when Python source or tests changed; prefix with `.venv/bin/` when using the repo-local virtualenv.
   - Treat type checking as a separate, non-blocking CI baseline unless the task explicitly asks for mypy cleanup. When needed, use `mypy src/agentcontract/ --ignore-missing-imports` to match CI.
   - For record/replay behavior, run the relevant test once with `pytest --ac-record` and once with `pytest --ac-replay`; prefix with `.venv/bin/` when using the repo-local virtualenv.
   - If docs-only scope makes automated validation unnecessary, record exactly what was skipped and why.
   - If adding public functionality, add focused tests for a happy path and an edge case.

5. Commit
   - Use one commit per logical change.
   - Use conventional commit prefixes such as `fix:`, `feat:`, `refactor:`, `test:`, or `docs:`.
   - Describe what changed, not the editing activity.
   - Do not commit failing tests or lint errors unless the failure is explicitly documented as pre-existing or unavailable.

## Project Facts

- Main package: `src/agentcontract/`.
- Pytest plugin entrypoint: `agentcontract.plugin`.
- CLI script: `agentcontract`, implemented in `src/agentcontract/cli.py`.
- Framework adapters: `src/agentcontract/adapters/`.
- Unit tests: `tests/unit/`.
- Example project: `examples/customer_support/`.

## Rules

- Do not move utility functions between modules unless the task explicitly requires it.
- Do not refactor unless the task explicitly asks for it.
- Do not touch files unrelated to the task.
- If stuck, try a simpler approach and reduce scope.
