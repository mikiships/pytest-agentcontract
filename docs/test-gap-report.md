# Test Coverage Gap Report

Generated from the current repository state on 2026-06-15.

## Scope

This report identifies the highest-value test coverage gaps only. It does not
change runtime code, add dependencies, or attempt to close the gaps in this
branch.

The reliable coverage command is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -p pytest_cov --cov=agentcontract --cov-report=term-missing
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is important here because normal pytest
plugin autoload imports `agentcontract` before coverage starts. Disabling
autoload and explicitly loading `pytest_cov` gives a cleaner package coverage
baseline.

## Baseline

The coverage run passed:

```text
77 passed in 0.27s
TOTAL 1351 statements, 377 missed, 72% covered
```

Highest gaps from the missing-line report:

| Area | Statements | Missed | Coverage | Missing lines |
| --- | ---: | ---: | ---: | --- |
| `src/agentcontract/plugin.py` | 87 | 87 | 0% | 3-183 |
| `src/agentcontract/adapters/openai_agents.py` | 146 | 100 | 32% | 38-91, 108-109, 144-149, 183, 188, 193-206, 212, 218-230, 236-266, 271-276 |
| `src/agentcontract/cli.py` | 68 | 42 | 38% | 33-39, 47-48, 52-61, 72-84, 89-130, 134 |
| `src/agentcontract/adapters/llamaindex.py` | 82 | 21 | 74% | 40, 48, 50, 62-66, 87, 108-110, 134, 172-179 |
| `src/agentcontract/adapters/langgraph.py` | 88 | 20 | 77% | 40, 45, 47, 49, 67-71, 95, 122, 130, 134-138, 151-153 |

Moderate remaining gaps are in `assertions/engine.py` at 82%,
`replay/engine.py` at 83%, `serialization.py` at 89%, `config.py` at 91%, and
the package lazy-import surface in `__init__.py` at 33%.

## Priority 1: pytest Plugin Surface

`src/agentcontract/plugin.py` is the largest user-facing gap: the shipped pytest
entry point has no coverage. This file owns the public pytest options, markers,
fixtures, record/replay behavior, cassette loading, cassette saving, and
contract-checking fixture.

Why it matters:

- Broken `--ac-record`, `--ac-replay`, `--ac-config`, or `--ac-scenarios`
  handling would affect the core pytest workflow.
- Scenario name resolution differs between `@pytest.mark.agentcontract(...)`,
  `@pytest.mark.agent_scenario(name=...)`, and fallback test names.
- `ac_recorder` auto-save failures and `ac_replay_engine` load failures call
  `pytest.fail`; missing cassettes call `pytest.skip`.
- `ac_check_contract` merges default assertions, scenario overrides, and
  extra assertions before delegating to `AssertionEngine`.

Recommended follow-up tests:

- Add `tests/unit/test_plugin.py` with pytest fixture/hook tests.
- Cover option registration and marker registration.
- Cover `ac_mode` for record, replay, and live modes.
- Cover scenario-name resolution for positional marker args, `name=` marker
  kwargs, and fallback node names.
- Cover `ac_recorder` auto-save path selection and save failure behavior.
- Cover `ac_replay_engine` when replay mode is off, cassette is missing,
  cassette loading succeeds, and cassette loading fails.
- Cover `ac_check_contract` assertion merge order and policy forwarding with a
  small fake config and fake run.

## Priority 2: CLI Success and Error Paths

`src/agentcontract/cli.py` has one existing test, and it only covers an invalid
`info` cassette. Most command dispatch, happy paths, missing-file branches, and
`init` behavior are uncovered.

Why it matters:

- `agentcontract info` and `agentcontract validate` are user-facing diagnostics
  for cassette files.
- Missing-path and invalid-cassette errors should be stable because they are
  likely to appear in CI logs.
- `agentcontract init` writes configuration into the current working directory,
  so overwrite prevention and write errors are important.

Recommended follow-up tests:

- Extend `tests/unit/test_cli.py`.
- Cover `main([])` help behavior and dispatch for `info`, `validate`, and
  `init`.
- Cover `info` success using a small valid cassette.
- Cover `info` missing-file errors separately from malformed-cassette errors.
- Cover `validate` success, missing-file failure, and malformed-cassette
  failure.
- Cover `init` success in `tmp_path`, existing-file refusal, and an `OSError`
  from writing the target file.

## Priority 3: OpenAI Agents SDK Adapter

`src/agentcontract/adapters/openai_agents.py` is at 32% coverage. Current tests
exercise some extraction helpers, but the public `record_runner` patching path
and many SDK-shape variants are uncovered.

Why it matters:

- `record_runner` monkeypatches SDK class methods globally and returns an
  unpatch callback. Bugs here can leak state across tests or production code.
- The adapter supports both async `Runner.run` and sync `Runner.run_sync`.
- The extraction helpers normalize multiple object shapes: final output,
  message output items, tool call items, tool call output items, handoffs,
  nested `function.name`, direct arguments, stringified JSON arguments, and
  invalid JSON fallbacks.

Recommended follow-up tests:

- Extend `tests/unit/test_adapters.py`, preserving the no-real-SDK mock style.
- Fake an `agents` module in `sys.modules` to cover `record_runner` without
  installing OpenAI Agents SDK.
- Cover non-`Recorder` inputs, missing SDK import, non-callable `Runner.run`,
  non-callable `Runner.run_sync`, and a `Runner` with neither method.
- Cover sync and async patch wrappers recording latency and restoring originals.
- Cover `_extract_from_result` when `new_items` is present.
- Cover `ToolCallItem` extraction, empty tool-call names, nested function names,
  dict arguments, JSON string arguments, invalid JSON strings, and missing
  nested attributes.
- Cover message content lists containing both dict content blocks and object
  content blocks.
- Cover `_extract_message_tool_calls` for object calls and empty call lists.

## Priority 4: Async and Edge Paths in Other Adapters

The LangGraph and LlamaIndex adapters have useful happy-path coverage, but the
async wrappers and validation branches are still lightly covered.

Why it matters:

- These adapters patch framework objects in place, so unpatch reliability and
  validation errors are important.
- Async framework entry points are common in agent applications.
- Extraction helpers need to tolerate partial framework objects without
  crashing.

Recommended follow-up tests:

- Extend `tests/unit/test_adapters.py`.
- For LangGraph, cover invalid recorder input, non-callable `invoke`, non-callable
  `ainvoke`, no available methods, async `ainvoke`, non-list `messages`, missing
  role/type, `None` content, list content blocks, dict tool calls, and non-dict
  tool-call arguments.
- For LlamaIndex, cover invalid recorder input, absent methods, non-callable
  methods, async `achat`/`aquery`, response `.message` extraction, empty message
  content, source entries without `tool_name`, source raw input that is not a
  dict, and source-node fallbacks using `id_` or direct `text`.

## Priority 5: Smaller Core Edge Cases

The core assertion, replay, serialization, config, and lazy-import modules have
moderate coverage. These are lower priority than the plugin, CLI, and adapters,
but the missing lines still map to realistic edge cases.

Recommended follow-up tests:

- In `tests/unit/test_assertions.py`, cover unknown assertion types, checker
  exceptions, non-string targets, invalid `turn:` targets, out-of-range turn
  targets, malformed `tool_call:` targets, `tool_call:*:result`, missing regex
  values, missing JSON schemas, non-dict `called_with` schemas, bool and
  non-integral float `called_count` values, unknown policy types, policy
  exceptions, and protected tool calls at turn 0.
- In `tests/unit/test_replay.py`, cover `_preview_content(None)`, unprintable
  content, long-content truncation, `recorded_run`, `finish(None)` with
  unconsumed tool calls, role mismatches, argument mismatches, extra turns with
  tool calls, and missing turns with no missing tool calls.
- In `tests/unit/test_serialization.py`, cover non-dict `run_from_dict` input,
  fallback role serialization for invalid/empty roles, timing and token
  serialization, invalid optional int/float coercions, invalid turn roles, and
  `save_run`/`load_run` file round trips.
- In `tests/unit/test_config.py`, cover the missing config-loading and coercion
  branches from the coverage report.
- Add a small `tests/unit/test_public_api.py` for `agentcontract.__getattr__`
  lazy imports and unknown attribute errors.

## Suggested Test Order

1. Add plugin tests first. They cover the fully untested pytest integration and
   protect the main package promise.
2. Expand CLI tests next. They are cheap, deterministic, and close a visible
   user-facing gap.
3. Add OpenAI Agents SDK adapter patching tests with fake modules and objects.
4. Add async and validation tests for LangGraph and LlamaIndex adapters.
5. Sweep the smaller assertion, replay, serialization, config, and public API
   edge cases.
