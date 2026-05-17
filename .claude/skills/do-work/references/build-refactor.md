# Build: Refactoring

Only use this reference when the task explicitly asks for refactoring.

1. Before touching files, run the relevant tests and save the outcome.
2. Do not change behavior and structure in the same commit:
   - First commit: structural change such as moving code or renaming; tests still pass.
   - Second commit: behavior change; tests updated.
3. If moving a public function between modules:
   - Add a re-export from the old location for backward compatibility.
   - Update all internal imports.
   - Run tests after each file change.
4. Keep each refactor commit to a small, reviewable scope. If more than three files need changes, split the work.
5. Do not refactor tests as part of source refactoring. Use a separate commit when test cleanup is needed.
