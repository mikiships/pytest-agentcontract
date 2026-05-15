# Test Gap Report

## Scope

This report identifies missing or thin test coverage in the current
`pytest-agentcontract` test suite. It does not add production code or new tests.

Starting branch recorded for this task: `nightshift/lint-fix-codex`.
Pre-existing dirty worktree entry before this task: `uv.lock` was modified and
was intentionally not touched.

Default branch used as the task base: `origin/main`.

## Coverage Run

Command used:

```sh
uv run --extra dev pytest --cov=agentcontract --cov-report=term-missing --cov-report=json
```

Result:

- `77 passed`
- Total measured coverage: `54%` (`728/1351` statements)
- Coverage artifact generated locally: `coverage.json`
- Coverage warning observed:
  `Module agentcontract was previously imported, but not measured`

The `--extra dev` flag is required from a clean checkout because `pytest-cov`
is declared in the optional `dev` dependency group, not in the runtime
dependencies.

The warning appears to come from the package being imported by pytest plugin
loading before coverage begins. The per-file results are still useful for gap
finding, but future coverage runs may need pytest-cov startup configuration if
the project starts enforcing coverage thresholds.

## Source to Test Mapping

| Source module | Existing direct tests | Coverage | Assessment |
| --- | --- | ---: | --- |
| `agentcontract.__init__` | None apparent | `0%` | No direct public API export tests. |
| `agentcontract.adapters.__init__` | `tests/unit/test_adapters.py` | `100%` | Direct lazy import coverage. |
| `agentcontract.adapters.langgraph` | `tests/unit/test_adapters.py` | `77%` | Partial coverage; async and error paths missing. |
| `agentcontract.adapters.llamaindex` | `tests/unit/test_adapters.py` | `74%` | Partial coverage; async and malformed response paths missing. |
| `agentcontract.adapters.openai_agents` | `tests/unit/test_adapters.py` | `32%` | High-risk gaps around patching `Runner` and item extraction. |
| `agentcontract.assertions.engine` | `tests/unit/test_assertions.py` | `62%` | Core behavior covered, but many failure and target-resolution branches missing. |
| `agentcontract.cli` | `tests/unit/test_cli.py` | `38%` | Only invalid `info` cassette path is directly tested. |
| `agentcontract.config` | `tests/unit/test_config.py` | `45%` | Core parsing covered; discovery and invalid/coercion edge cases remain. |
| `agentcontract.plugin` | None apparent | `2%` | Pytest fixtures, modes, save/load behavior, and marker handling lack direct tests. |
| `agentcontract.recorder.core` | `tests/unit/test_recorder.py` | `65%` | Basic recording covered; invalid role/save suffix and summary edge cases remain. |
| `agentcontract.recorder.interceptors` | `tests/unit/test_interceptors.py` | `93%` | Strong coverage; only unsupported client and parse edge branches remain. |
| `agentcontract.replay.engine` | `tests/unit/test_replay.py` | `57%` | Tool stubs and mismatch cases covered; finish/consumption edge cases remain. |
| `agentcontract.serialization` | `tests/unit/test_serialization.py` | `73%` | Good coercion coverage; file I/O and invalid role/error paths remain. |
| `agentcontract.types` | Indirect only | `1%` | Dataclass construction is indirect; `to_dict`/`from_dict` convenience methods not direct. |

## Highest-Value Gaps

1. `agentcontract.plugin` pytest integration

   User impact: high. This package is primarily a pytest plugin, but the
   fixtures and CLI options are almost untested directly.

   Recommended tests:

   - Use `pytester` to verify `--ac-record`, `--ac-replay`, and live mode values
     from the `ac_mode` fixture.
   - Verify `ac_recorder` derives scenario names from `agentcontract` and
     `agent_scenario` markers and writes cassettes only under `--ac-record`.
   - Verify `ac_replay_engine` skips when a cassette is missing, fails clearly on
     invalid cassettes, and loads a matching cassette from `--ac-scenarios`.
   - Verify `ac_check_contract` merges default assertions, scenario overrides,
     extra assertions, and policies.

2. CLI command behavior

   User impact: high. The installed `agentcontract` command is a public entry
   point, but only one error path is tested.

   Recommended tests:

   - `agentcontract info` on a valid cassette prints scenario, model, turns,
     tokens, duration, and estimated cost.
   - `agentcontract validate` returns success for valid cassette files and
     failure for malformed files.
   - `agentcontract init` writes the starter `agentcontract.yml`, and refuses to
     overwrite an existing file.
   - The no-command path prints help and returns `0`.

3. OpenAI Agents SDK adapter patching

   User impact: medium-high. Adapter behavior is integration-heavy and currently
   most of `record_runner` is uncovered.

   Recommended tests:

   - Inject a fake `agents.Runner` module with `run` and `run_sync`, then verify
     both methods are patched, record turns, and are restored by `unpatch`.
   - Verify missing optional SDK import raises the documented `ImportError`.
   - Verify non-callable `Runner.run` or `Runner.run_sync` fails with `TypeError`.
   - Cover `ToolCallItem`, message content blocks, function argument JSON parsing,
     invalid JSON fallback, and nested `function.arguments` extraction.

4. Public API exports and type helpers

   User impact: medium. The README-facing API appears to rely on lazy exports and
   dataclass convenience methods.

   Recommended tests:

   - Import `Recorder`, `ReplayEngine`, `AssertionEngine`, and
     `AgentContractConfig` from `agentcontract` and verify unknown attributes
     raise `AttributeError`.
   - Call `AgentRun.to_dict()` and `AgentRun.from_dict()` directly from
     `agentcontract.types`.
   - Confirm `__all__` remains aligned with the documented public API.

5. Assertion target and failure edge cases

   User impact: medium. Existing tests cover happy paths and some failures, but
   contract debugging depends on clear failure behavior.

   Recommended tests:

   - `_resolve_target` for `full_conversation`, `turn:N`, invalid turn indexes,
     `tool_call:name:result`, missing tool calls, and non-string target values.
   - Unknown assertion types fail closed.
   - `regex`, `json_schema`, `contains`, and tool assertions handle `None`,
     invalid schemas, non-dict schemas, boolean counts, and non-integral counts.
   - `requires_confirmation` fails for a protected tool called at turn `0`.

6. Replay engine completion edge cases

   User impact: medium. Replay mismatch messages are central to CI debugging.

   Recommended tests:

   - `ReplayEngine.finish()` without `actual_turns` reports unconsumed recorded
     tool calls.
   - Extra turns with tool calls increment both `extra_tools` and
     `mismatched_tools`.
   - Role mismatches produce actionable errors even when tool calls match.
   - Argument mismatches and missing/extra tool-call count combinations are
     reported in the same run.

7. Config and serialization validation boundaries

   User impact: medium. These modules accept user-authored YAML and cassette
   JSON, so error shape matters.

   Recommended tests:

   - `AgentContractConfig.discover()` returns defaults when no config exists and
     walks parent directories deterministically.
   - Invalid assertion and policy entries missing required `type` or `name`
     produce intentional errors or are skipped by documented behavior.
   - `load_run()` and `save_run()` cover file I/O behavior, suffix handling, and
     invalid turn roles.
   - Serialization coercion for non-finite or overflow numeric inputs is defined
     by tests.

## Lower-Priority Gaps

- Adapter async paths for LangGraph and LlamaIndex are not directly covered.
- Optional framework dependency coverage is represented by mocks only; no tests
  exercise real LangGraph, LlamaIndex, OpenAI, Anthropic, or OpenAI Agents SDK
  objects.
- Package `__init__.py` files for subpackages show low measured coverage because
  they contain tiny export surfaces. They should get lightweight export tests
  only after the plugin and CLI gaps are covered.

## Regenerating Findings

Run the coverage command from the repository root:

```sh
uv run --extra dev pytest --cov=agentcontract --cov-report=term-missing --cov-report=json
```

Use the terminal report for missing line ranges and inspect `coverage.json` for
machine-readable details. Do not commit generated coverage artifacts unless the
project intentionally adds a coverage publishing workflow.
