# Build: Refactoring

Only read this if the task EXPLICITLY asks for refactoring.

1. Before touching anything: run full test suite and save output
2. Refactor rule: never change behavior and structure in the same commit
   - First commit: structural change (move code, rename), tests still pass
   - Second commit: behavior change (new logic), tests updated
3. If moving a function between modules:
   - Add a re-export from the old location for backward compatibility
   - Update all internal imports
   - Run tests after EACH file change, not at the end
4. Maximum scope: 3 files per refactor commit. If touching more, split into multiple commits.
5. NEVER refactor test files as part of source refactoring. Separate commits.
