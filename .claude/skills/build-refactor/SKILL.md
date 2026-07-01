---
name: build-refactor
description: Guides safe pytest-agentcontract refactoring. Use only when a task explicitly asks to refactor Python source, reorganize modules, move functions, rename internal APIs, split behavior across files, or separate structural changes from behavior changes.
---

# Build: Refactoring

Only read this if the task EXPLICITLY asks for refactoring.

1. Before touching anything: run `.venv/bin/pytest tests/ -x -q` and save the output as the baseline when dependencies are available
2. Refactor rule: never change behavior and structure in the same commit
   - First commit: structural change (move code, rename), tests still pass
   - Second commit: behavior change (new logic), tests updated
3. If moving a function between modules:
   - Add a re-export from the old location when the symbol is public or imported externally
   - Update all internal imports
   - Run focused tests after each file change, not only at the end
4. Maximum scope: 3 files per refactor commit. If touching more, split into multiple commits.
5. NEVER refactor test files as part of source refactoring. Separate commits.
