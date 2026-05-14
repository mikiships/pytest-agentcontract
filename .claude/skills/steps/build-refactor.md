# Build: Refactoring

Only read this if the task EXPLICITLY asks for refactoring.

1. Before a broad refactor: run `uv run pytest tests/ -x -q` and save the result
2. Refactor rule: never change behavior and structure in the same commit
   - First commit: structural change (move code, rename), tests still pass
   - Second commit: behavior change (new logic), tests updated
3. If moving a function between modules:
   - Add a re-export from the old location for backward compatibility
   - Update all internal imports
   - Update public lazy imports and `__all__` if the function is exported
   - Run targeted tests after each logical move, not only at the end
4. Maximum scope: 3 files per refactor commit. If touching more, split into multiple commits.
5. NEVER refactor test files as part of source refactoring. Separate commits.
6. Before committing, run `uv run pytest tests/ -x -q`, `uv run ruff check src/ tests/`, and `uv run mypy src/`; document any known baseline failures separately from the refactor.
