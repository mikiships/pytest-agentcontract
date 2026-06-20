# Test Coverage Gap Report

## Scope

This report identifies current Python package test coverage gaps only. It does
not add or remediate tests.

Coverage was measured with pytest plugin autoload disabled so this package's own
pytest plugin does not load before coverage starts and skew the result:

```shell
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -p pytest_cov --cov=agentcontract --cov-report=term-missing --cov-report=json:coverage.json
```

`coverage.json` and `.coverage` are local analysis artifacts from this command
and should not be committed unless intentionally needed for a separate workflow.

## Baseline

Recorded on 2026-06-20 from the current package test suite:

- 77 tests passed.
- Total package coverage: 72%.
- The largest uncovered surfaces are pytest plugin integration, CLI commands,
  OpenAI Agents SDK patching and extraction branches, adapter edge paths, and
  selected assertion/replay branches.

| Module | Coverage | Missing statements | Primary gap |
| --- | ---: | ---: | --- |
| `src/agentcontract/plugin.py` | 0% | 87 | No direct tests for pytest options, fixtures, markers, record/replay behavior, or contract assertion fixture wiring. |
| `src/agentcontract/adapters/openai_agents.py` | 32% | 100 | Helper extraction has limited tests; SDK patching, async/sync wrapping, error paths, and many item/argument variants are uncovered. |
| `src/agentcontract/cli.py` | 38% | 42 | Only one invalid-cassette `info` path is covered; normal `info`, `validate`, `init`, missing files, and write errors are untested. |
| `src/agentcontract/adapters/llamaindex.py` | 74% | 21 | Adapter validation, async methods, message fallback, absent methods, and source edge cases are lightly covered. |
| `src/agentcontract/adapters/langgraph.py` | 77% | 20 | Adapter validation, async invocation, malformed message collections, content-list handling, and object tool-call variants are lightly covered. |
| `src/agentcontract/assertions/engine.py` | 82% | 34 | Target parsing and error branches remain untested. |
| `src/agentcontract/replay/engine.py` | 83% | 18 | Replay completion without actual turns, role/argument mismatch branches, and preview formatting edge cases remain untested. |

`src/agentcontract/__init__.py` also reports low line coverage, but the user-facing
risk is lower than the command and plugin surfaces above.

## Prioritized Gaps

### 1. Pytest Plugin Fixtures and Options

Risk: high. This is the package's main pytest-facing integration point, and the
coverage command intentionally disables plugin autoload. The resulting 0%
coverage is a real signal that no tests import or exercise `plugin.py` directly.

Recommended next file: `tests/unit/test_plugin.py`.

Add focused tests for:

- `pytest_addoption` registering `--ac-record`, `--ac-replay`, `--ac-config`,
  and `--ac-scenarios`.
- `pytest_configure` registering the `agentcontract` and `agent_scenario`
  markers.
- `ac_mode` resolving `record`, `replay`, and default `live`.
- `ac_config` using an explicit `--ac-config` path and discovery fallback.
- `ac_recorder` scenario name selection from `agentcontract`,
  `agent_scenario`, and the test node name.
- Record-mode auto-save success and save failure reporting.
- `ac_replay_engine` replay-mode skip when a cassette is absent and failure
  when a cassette cannot be loaded.
- `ac_check_contract` merging default assertions, scenario overrides, extra
  assertions, and policies before calling `AssertionEngine.check`.

### 2. CLI Happy and Error Paths

Risk: high. Users interact with `agentcontract info`, `validate`, and `init`
directly, but `tests/unit/test_cli.py` currently covers only one invalid
`info` cassette path.

Recommended next file: expand `tests/unit/test_cli.py`.

Add tests for:

- `main([])` or an unknown command printing help and returning success.
- `info` on a valid cassette, asserting scenario, run id, model, turn count,
  tool-call count, duration, tokens, and estimated cost output.
- `info` and `validate` missing-file errors.
- `validate` valid and invalid cassette paths.
- `init` writing the starter `agentcontract.yml`.
- `init` refusing to overwrite an existing config.
- `init` write failure handling, using a patched `Path.write_text`.

### 3. OpenAI Agents SDK Adapter

Risk: medium-high. `record_runner` is the integration point that patches
OpenAI Agents SDK `Runner.run` and `Runner.run_sync`; current tests cover only
some extraction helpers and do not exercise the patch lifecycle.

Recommended next file: expand `tests/unit/test_adapters.py` or split out
`tests/unit/test_openai_agents_adapter.py`.

Add tests for:

- Import error when `agents` is unavailable.
- Rejecting non-`Recorder` values.
- Rejecting non-callable `Runner.run` or `Runner.run_sync`.
- Rejecting a `Runner` with neither method.
- Patching and unpatching `run_sync`.
- Patching and unpatching async `run`.
- Extracting from `new_items` before falling back to `final_output`.
- `ToolCallItem` extraction with `name`, `function.name`, `id`, `call_id`,
  dict arguments, JSON string arguments, invalid JSON arguments, and empty
  names.
- `MessageOutputItem` content lists with dict and object blocks, plus tool-call
  lists.
- `ToolCallOutputItem` with `None` output and `HandoffCallItem` without a
  target agent.

### 4. Adapter Edge Paths

Risk: medium. LangGraph and LlamaIndex have solid basic coverage but limited
negative and async coverage.

Recommended next file: expand `tests/unit/test_adapters.py`.

Add LangGraph tests for:

- Non-`Recorder` rejection.
- Non-callable `invoke` or `ainvoke`.
- No `invoke` or `ainvoke`.
- Async `ainvoke` wrapper recording.
- `messages` present but not a list/tuple.
- content-list extraction and non-string content conversion.
- dict and object tool-call variants with non-dict arguments.

Add LlamaIndex tests for:

- Non-`Recorder` rejection.
- Non-callable adapter methods.
- No supported methods.
- Async `achat` and `aquery` wrappers.
- Response `.message` fallback.
- sources without `tool_name`, non-dict `raw_input`, missing `raw_output`.
- source nodes using `id_`, missing scores, and direct text fallback.

### 5. Assertion and Replay Branch Coverage

Risk: medium. Core behavior has broad coverage, but several failure and parsing
branches are still untested. These are useful regression tests because they
define failure messages for contract authors and CI replay users.

Recommended next files: expand `tests/unit/test_assertions.py` and
`tests/unit/test_replay.py`.

Add assertion tests for:

- Unknown assertion type.
- Assertion checker exceptions being reported rather than raised.
- Non-string targets.
- `final_response` when no assistant content exists.
- `full_conversation`.
- `turn:N` valid, invalid, and out-of-range targets.
- `tool_call:name:result`, missing tool-call name, and unknown fields.
- `contains`, `regex`, and `json_schema` missing target/value/schema paths.
- `called_with` with non-dict `schema`.
- `called_count` with bool and non-integer float values.
- `requires_confirmation` when a protected tool is called at turn 0.

Add replay tests for:

- `finish(actual_turns=None)` reporting unconsumed recorded tool calls.
- `recorded_run` property.
- Role mismatch reporting.
- Tool argument mismatch inside `finish`.
- Extra turns with extra tool calls.
- `_preview_content` handling `None`, truncating long content, and handling
  unprintable objects.

## Suggested Follow-Up Order

1. Add `tests/unit/test_plugin.py` with direct fixture and option coverage.
2. Expand `tests/unit/test_cli.py` to cover user-visible commands.
3. Add focused OpenAI Agents SDK adapter patching tests.
4. Fill LangGraph and LlamaIndex edge paths.
5. Add assertion and replay branch regression tests.
