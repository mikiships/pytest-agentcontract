# Current-checkout PII audit

Audited the tracked text files at base `17c5203` using local email, formatted US
phone, and US SSN pattern matching. No history or external services were used.
Reviewed recorder/core.py's save path and serialization.py: messages, metadata,
and nested tool arguments/results are persisted without PII filtering. The new
scanner reads raw JSON so deserialization cannot discard fields before scanning.

| Location | Category | Review |
| --- | --- | --- |
| examples/customer_support/agent.py:16, :24 | email | Synthetic example-domain contacts |
| tests/scenarios/refund-eligible.agentrun.json:52 | email | Synthetic example-domain contact |
| tests/scenarios/refund-not-delivered.agentrun.json:52 | email | Synthetic example-domain contact |
| tests/scenarios/refund-with-policies.agentrun.json:52 | email | Synthetic example-domain contact |
| examples/customer_support/scenarios/refund-eligible.agentrun.json:52 | email | Same synthetic fixture contact |
| examples/customer_support/scenarios/refund-not-delivered.agentrun.json:52 | email | Same synthetic fixture contact |
| examples/customer_support/scenarios/refund-with-policies.agentrun.json:52 | email | Same synthetic fixture contact |

Every match above uses a reserved example domain. No confirmed accidental PII
was found, so no fixture or example remediation was made. Both scenario directory
scans complete with three email findings each and exit 1, as expected. New scanner
tests intentionally use reserved-domain emails, fictional phone examples, and an
invalid all-zero SSN pattern. These test data should continue to trigger detectors.

## Intentional public attribution

The author name in pyproject.toml and copyright attribution in LICENSE are public
project attribution, not recorded customer data. Repository URLs and contributor
handles are also intentional attribution. These were left unchanged.

This limited heuristic review does not certify that the checkout is PII-free.
Binary assets and unsupported PII categories are not covered.

## Iteration 2 verification

Repeated the current-checkout audit after extending parenthesized phone detection
and sanitizing CLI argument errors. The eight original email locations remain
reserved example-domain contacts; no fixture remediation is needed. The additional
README phone example and scanner regression inputs are intentionally synthetic.
Both scenario directories still produce three email findings each without errors.
Invalid argument tests cover missing input, surplus paths, and unknown options;
none may echo supplied values. Parenthesized phone tests cover compact subscriber
numbers and reject adjoining letters or extra digits.


## Iteration 3 verification

The root parser and every subparser now use fixed argument-error diagnostics.
This closes the remaining leak when an unknown option or invalid positional token
appears before `scan-pii`. Regression tests exercise both explicit argument lists
and `sys.argv`, surplus arguments after `--`, attached help-option values, and the
module entry point. Normal root and scanner help remain available.

Repeated local pattern matching over tracked text files and both cassette trees:
the original eight email locations still use reserved example domains. No confirmed
accidental exposure or fixture changes were required. Public attribution remains
unchanged. Scanner test values and the README phone example are intentionally
synthetic detector inputs.

Validation: 131 tests pass, including 55 focused scanner/CLI tests; CI-scoped Ruff
passes. The CI Mypy command reports nine errors in four unchanged files, and
repository-wide Ruff reports four errors in unchanged examples. Both sets were
reproduced against an isolated export of base `17c5203` with the same tools.
