# Test Gap Report

Date: 2026-06-01

## Scope

This report identifies the largest test coverage gaps in the current repository state. It is
intentionally limited to findings and recommended follow-up tests; it does not add broad coverage
or change production code.

Coverage was measured with the local pytest plugin disabled so `agentcontract` is not imported
before coverage starts. The coverage-relevant pytest invocation is:

```bash
uv run pytest -p no:agentcontract --cov=agentcontract --cov-report=term-missing
```

On a clean `origin/main` checkout, `pytest-cov` is only available through the optional `dev` extra
and a plain `uv run pytest ... --cov` rejects the `--cov` options. To keep this documentation-only
branch from changing `uv.lock`, the observed baseline below was collected with `pytest-cov` supplied
ephemerally:

```bash
uv run --frozen --with pytest-cov pytest -p no:agentcontract --cov=agentcontract --cov-report=term-missing
```

## Baseline

The baseline suite passes:

- Result: 77 passed
- Total coverage: 72%
- Python: 3.11.14
- Pytest: 9.0.2
- pytest-cov: 7.0.0

Largest coverage gaps:

| Priority | Module | Coverage | Missing behavior |
| --- | --- | ---: | --- |
| P0 | `src/agentcontract/plugin.py` | 0% | Pytest options, markers, fixtures, cassette loading, recorder save, assertion merging |
| P1 | `src/agentcontract/cli.py` | 38% | Most command success and failure paths |
| P1 | `src/agentcontract/adapters/openai_agents.py` | 32% | Runner patching plus most item extraction helpers |
| P2 | `src/agentcontract/adapters/llamaindex.py` | 74% | Async wrappers and response edge cases |
| P2 | `src/agentcontract/adapters/langgraph.py` | 77% | Async wrapper and invalid target edge cases |
| P3 | `src/agentcontract/__init__.py` | 33% | Lazy public API imports and unknown attribute path |
| P3 | `src/agentcontract/assertions/engine.py` | 82% | Exception, unknown target, and rare assertion branches |
| P3 | `src/agentcontract/replay/engine.py` | 83% | Role mismatch and missing-turn diff branches |

## P0: Pytest Plugin

`src/agentcontract/plugin.py` has no direct unit coverage, but it is the package's primary pytest
integration surface. This is the highest-priority gap because regressions here can break the normal
user workflow even when lower-level recorder, replay, config, and assertion tests pass.

Recommended file: `tests/unit/test_plugin.py`

Recommended scenarios:

- `pytest_addoption` registers `--ac-record`, `--ac-replay`, `--ac-config`, and `--ac-scenarios`.
- `pytest_configure` registers both `agentcontract` and `agent_scenario` markers.
- `ac_config` loads an explicit `--ac-config` path and falls back to discovery when no path is set.
- `ac_mode` returns `record`, `replay`, or `live` from the option combinations.
- `ac_recorder` chooses the scenario from marker args, marker `name`, or the test node name.
- `ac_recorder` auto-saves only in record mode and respects `--ac-scenarios`.
- `ac_recorder` reports save failures through `pytest.fail`.
- `ac_replay_engine` returns `None` outside replay mode.
- `ac_replay_engine` skips when the cassette is missing.
- `ac_replay_engine` fails clearly when cassette loading raises `OSError`, `ValueError`, or
  `TypeError`.
- `ac_replay_engine` returns a `ReplayEngine` for a valid cassette.
- `ac_check_contract` merges default assertions, scenario override assertions, and extra assertions,
  then passes configured policies through to `AssertionEngine.check`.

## P1: CLI

`tests/unit/test_cli.py` currently covers only one invalid-cassette path. The CLI should get focused
command-level tests because it is a public entry point and has several simple branches that are
cheap to cover.

Recommended file: expand `tests/unit/test_cli.py`

Recommended scenarios:

- `main([])` prints help and exits with `0`.
- `info` returns `1` for a missing file.
- `info` prints scenario, run id, model, turn count, tool calls, duration, tokens, and estimated cost
  for a valid cassette.
- `info` returns `1` and writes a useful error for an invalid cassette.
- `validate` returns `1` for a missing file.
- `validate` returns `0` and prints a valid-cassette summary for a valid cassette.
- `validate` returns `1` and writes a useful error for an invalid cassette.
- `init` writes the starter `agentcontract.yml` in an empty directory.
- `init` refuses to overwrite an existing `agentcontract.yml`.
- `init` returns `1` when `Path.write_text` raises `OSError`.

## P1: OpenAI Agents SDK Adapter

`src/agentcontract/adapters/openai_agents.py` is mostly uncovered. Current tests exercise a few
private extraction helpers, but they do not cover `record_runner`, import/type validation, method
patching, unpatching, or many item-shape edge cases.

Recommended file: split to `tests/unit/test_openai_agents_adapter.py` or extend
`tests/unit/test_adapters.py` if the single adapter test file remains preferred.

Recommended scenarios:

- `record_runner` rejects a non-`Recorder` argument.
- `record_runner` raises a clear `ImportError` when `agents` is unavailable.
- `record_runner` rejects non-callable `Runner.run` and `Runner.run_sync` attributes.
- `record_runner` raises `ValueError` when neither method exists.
- Sync `Runner.run_sync` is patched, records latency-backed turns, returns the original result, and
  is restored by `unpatch`.
- Async `Runner.run` is patched, records latency-backed turns, returns the original result, and is
  restored by `unpatch`.
- `_extract_from_result` prefers `new_items` over `final_output` and ignores empty or unsupported
  result shapes.
- `_extract_message_content` handles string content, dict content blocks, object content blocks,
  empty content, and non-string fallback content.
- `_extract_message_tool_calls` handles direct `name`, nested `function.name`, `id`, `call_id`, and
  empty tool-call lists.
- `_get_tool_arguments` handles dict args, JSON string args, invalid JSON strings, `input`,
  `function.arguments` as dict, and `function.arguments` as JSON string.
- `_extract_from_items` covers `ToolCallItem` with direct and nested names, empty-name tool calls,
  `ToolCallOutputItem` with `None` output, and `HandoffCallItem` without a target agent.

## P2: LangGraph And LlamaIndex Adapters

The LangGraph and LlamaIndex adapters have useful happy-path coverage, but several boundary paths
remain untested. These are medium priority because they protect optional framework integrations and
can be covered with lightweight fake objects.

Recommended file: expand `tests/unit/test_adapters.py` or split into
`tests/unit/test_langgraph_adapter.py` and `tests/unit/test_llamaindex_adapter.py`.

Recommended LangGraph scenarios:

- `record_graph` rejects a non-`Recorder` argument.
- `record_graph` rejects non-callable `invoke` or `ainvoke` attributes.
- `record_graph` raises `ValueError` when neither method exists.
- `ainvoke` wrapper records turns and restores the original method.
- Non-list `messages` values are ignored.
- Dict and object tool calls with non-dict `args` or `arguments` are coerced to `{}`.
- Content block lists are joined for dict and object message shapes.
- Unknown message roles are ignored.

Recommended LlamaIndex scenarios:

- `record_agent` rejects a non-`Recorder` argument.
- `record_agent` rejects non-callable `chat`, `achat`, `query`, or `aquery` attributes.
- `record_agent` raises `ValueError` when no supported method exists.
- Async `achat` and `aquery` wrappers record turns and restore originals.
- Response extraction handles `.message` content when `.response` is absent.
- Source entries without `tool_name` are ignored.
- Source entries with non-dict `raw_input` use empty arguments.
- `source_nodes` fall back from `node_id` to `id_`.
- Retrieval source-node text fallback works when `node` is missing and `text` is present.

## P3: Public API And Edge Cases

These gaps are lower priority because core happy paths already have coverage, but they are still
worth filling after the plugin, CLI, and OpenAI adapter are covered.

Recommended scenarios:

- Add `agentcontract.__getattr__` tests for `Recorder`, `ReplayEngine`, `AssertionEngine`,
  `AgentContractConfig`, and an unknown attribute.
- Add `AgentRun.to_dict` and `AgentRun.from_dict` delegation tests in `tests/unit/test_serialization.py`
  or a new `tests/unit/test_types.py`.
- Add assertion engine tests for exception handling, unknown assertion types, unknown or unsupported
  targets, invalid regex, and malformed JSON schema branches.
- Add replay tests for role mismatch messages, missing turn diffs, empty recorded runs, and
  non-string missing-turn content handling.
- Add serialization tests for save or load write failures, invalid top-level JSON, non-object
  decoded JSON, and non-serializable fallback branches that are not already covered.

## Recommended Order

1. Add plugin tests first. This covers the package's pytest integration contract and should produce
   the largest immediate risk reduction.
2. Expand CLI tests next. The branch count is small, the scenarios are deterministic, and the tests
   should be quick to write.
3. Split and deepen OpenAI Agents SDK adapter tests. This protects the largest optional integration
   gap and should be kept mock-only.
4. Fill LangGraph and LlamaIndex edge cases.
5. Add focused public API, replay, assertion, and serialization edge-case tests.
