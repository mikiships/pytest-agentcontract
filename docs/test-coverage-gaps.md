# Test coverage gaps

Audit of `origin/main` at `17c5203`, measured on 2026-10-23. Production code is unchanged.

## Reproduction and measurement limits

Environment: macOS, CPython 3.12.14, pytest 9.0.3, pytest-cov 7.1.0,
coverage.py 7.14.0. Installed the declared development extra into a separate venv
with `uv pip install --python .venv/bin/python -e '.[dev]'`; no lockfile was changed.

Required baseline and final command (activate the venv first):

```sh
pytest tests --cov=agentcontract --cov-branch --cov-report=term-missing --cov-report=json
```

Baseline: **77 passed**. After: **106 passed**, including **29 plugin tests**.
No baseline test failures. This command reports a `module-not-measured` warning:
the installed pytest entry point imports plugin dependencies before pytest-cov starts.
Thus missed import-time definitions are measurement artifacts, not proof of missing tests.
Its combined statement/branch totals are 55.26% → 60.33%.

A second measurement starts coverage before pytest imports its entry points, using
identical commands before and after the test additions:

```sh
python -m coverage run --branch --source=agentcontract -m pytest tests -q
python -m coverage json -o /tmp/test-gap-startup.json
```

These startup measurements underpin the table and prioritization below. Combined
coverage rises **70.41% → 75.20%**, statement coverage **74.46% → 78.90%**,
and branch coverage **59.09% → 64.88%** (286 → 314 of 484 branch exits).
Coverage JSON and data files are local artifacts, excluded from the commit.

Only Python 3.12 was run locally; CI declares Python 3.10–3.13. Optional provider
SDKs were not installed and no network calls or credentials were used in the tests.
Existing adapter/interceptor tests use fake objects; they do not prove compatibility
with real installed SDK versions. Percentages describe exercised code, not correctness.

## Measured module coverage

Percentages combine covered statements and branch exits. Paths are relative to
`src/agentcontract/`; both columns use startup measurement.

| Module | Before | After |
| --- | ---: | ---: |
| `__init__.py` | 27.27% | 27.27% |
| `adapters/__init__.py` | 100.00% | 100.00% |
| `adapters/langgraph.py` | 71.54% | 71.54% |
| `adapters/llamaindex.py` | 69.17% | 69.17% |
| `adapters/openai_agents.py` | 28.32% | 28.32% |
| `assertions/__init__.py` | 100.00% | 100.00% |
| `assertions/engine.py` | 76.78% | 77.53% |
| `cli.py` | 35.37% | 35.37% |
| `config.py` | 87.37% | 88.95% |
| `plugin.py` | 28.83% | 99.10% |
| `recorder/__init__.py` | 100.00% | 100.00% |
| `recorder/core.py` | 91.49% | 91.49% |
| `recorder/interceptors.py` | 89.62% | 89.62% |
| `replay/__init__.py` | 100.00% | 100.00% |
| `replay/engine.py` | 79.87% | 83.22% |
| `serialization.py` | 89.07% | 89.07% |
| `types.py` | 94.37% | 94.37% |

## Prioritized behaviors

Locations refer to the audited source revision. Relevant tests are under `tests/unit/`.
P1 prioritizes deterministic offline execution and actionable diagnostics; P2 covers
additional normalization and optional integrations.

| Priority / status | Source location | Existing relevant tests | Missing behavior and recommended cases |
| --- | --- | --- | --- |
| P1 addressed | `plugin.py:19–187` | New `test_plugin.py` | Pytester exercises live/record/replay and both-flag precedence, explicit/discovered configuration, strict markers, both aliases and keyword names, fallback names, default/overridden paths, serialized round trips, missing/malformed cassettes, save/load OSError/ValueError/TypeError diagnostics and setup/teardown outcomes, defaults/overrides/extras/policies and repeated-check immutability. |
| P1 remaining | `cli.py:33–39,47–61,72–84,89–130` | `test_cli.py::test_info_returns_error_for_invalid_cassette` | Add public `main()` tests for info success/missing path, validate success/malformed/missing input, init creation/refusal/write failure, and no-command help; assert exit codes and stdout/stderr. |
| P1 remaining | `replay/engine.py:14–20,131–136,148–170,187–188` | `test_replay.py` matching/mismatch/count tests; plugin cassette loading | Test partially consumed stubs with no actual turns, role mismatch, extra/missing turns without tools, argument mismatch in `finish()`, long/unprintable content diagnostics. Loading and empty successful replay are now indirectly covered by plugin tests. |
| P1 remaining | `assertions/engine.py:87–96,113–159,188–216,251,284–290,364` | `test_assertions.py`; new contract fixture tests | Exercise unknown assertion types, checker exceptions, malformed turn/tool targets, full conversation, missing values, non-dict expected arguments, boolean/fractional call counts, and protected tools after non-user turns. Assert failure messages, not only booleans. |
| P1 remaining | `adapters/openai_agents.py:38–91,189–270` | `test_adapters.py` extraction tests | Add fake runner sync/async wrapper tests, exception cleanup and restoration, usage/raw response extraction. Most wrapper paths are genuinely untested; real SDK compatibility additionally requires an optional dependency CI job. |
| P2 remaining | `config.py:218–265` | `test_config.py` coercion tests; plugin discovery/loading tests | Parameterize numeric booleans, fractional/invalid/overflow integer inputs, invalid floats and defaults. Discovery success now has plugin integration coverage; malformed YAML/config diagnostics remain useful additions. |
| P2 remaining | `serialization.py:68–89,113–116,170,191–207,291–292`; `types.py:108–117` | `test_serialization.py`; recorder save/load; new plugin round trips | Test failed numeric conversions, recursive/custom values, malformed list entries, file read failures and public `AgentRun.to_dict/from_dict` round trips. Serialization already has indirect coverage; plugin I/O injections test diagnostics rather than the serializer's OS error paths. |
| P2 remaining | `recorder/core.py:34–45,187`; `recorder/interceptors.py:29,63,110–114,178–217` | `test_recorder.py`, `test_interceptors.py`; new fixture lifecycle tests | Add invalid numeric normalization, extensionless save path, missing SDK methods and malformed SDK fields; verify patch restoration after exceptions. |
| P2 remaining | `adapters/langgraph.py:40–71,122–153`; `adapters/llamaindex.py:40–66,108–110,168–175` | `test_adapters.py` fake graph/agent tests | Test asynchronous wrappers, restoration on exceptions, unusual message/result shapes and empty inputs using fakes; separately test installed SDK compatibility. |
| P2 remaining | `__init__.py:11–22` | Adapter lazy-import tests only | Test public lazy exports and unknown attribute errors; imports alone do not cover export dispatch. |

## Plugin interpretation and remaining limitations

Startup coverage for the plugin is **99.10%**: 86/87 statements and 24/24 branch
exits. The single missed line (`plugin.py:138`) is `return None` after
`pytest.skip()`, which raises; it is unreachable under normal pytest behavior.
Do not force execution by replacing skip with a non-raising fake.

The fixture tests use local `pytester` activation, disable entry-point autoload for
nested runs, and explicitly load `agentcontract.plugin` once. Runs are in-process
to retain outer coverage, isolated by pytester temporary directories. Each mode test
asserts a single plugin instance. Tests synthesize trajectories and use public
`load_run`/`save_run` APIs to verify cassette contents.

Current semantics are preserved: `ac_mode` returns record when both flags are set,
while `ac_replay_engine` independently tests the replay flag. The cassette directory
fixtures use `--ac-scenarios` or `tests/scenarios`, not config include patterns.
Those semantics may deserve future product decisions; this audit does not change them.
Additional integration cases could cover both flags with a requested replay fixture,
conflicting markers, parent-directory discovery, and recording after a failing test.

## Validation

- Focused plugin suite: 29 passed.
- Full suite with requested branch coverage: 106 passed.
- Full suite with startup branch coverage: 106 passed.
- CI lint command `ruff check src/ tests/`: passed; changed Python file also passes.
- CI advisory type check `mypy src/agentcontract/ --ignore-missing-imports`:
  9 existing errors in 4 unchanged source files: missing return annotation in
  `__init__.py`, Any return in serialization, four optional-access errors in
  interceptors, and three unused ignores in the OpenAI agents adapter.
  CI marks this check `continue-on-error`; no production fixes were added.
