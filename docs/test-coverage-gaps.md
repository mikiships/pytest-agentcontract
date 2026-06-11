# Test Coverage Gaps

Baseline collected on 2026-06-11 from `origin/main` using:

```bash
uv run --extra dev coverage erase
uv run --extra dev coverage run -m pytest
uv run --extra dev coverage report -m --include='src/agentcontract/*'
```

`coverage run -m pytest` is the authoritative invocation for gap analysis in
this repository because it starts coverage before pytest imports the
`agentcontract` pytest plugin. Use `--extra dev` from a clean checkout so `uv`
installs the development tooling that provides `pytest-cov` and its transitive
`coverage` dependency.

## Baseline

The authoritative source-package baseline is 74% coverage with 77 tests passing.

| Module | Statements | Missing | Coverage | Missing lines |
| --- | ---: | ---: | ---: | --- |
| `src/agentcontract/adapters/openai_agents.py` | 146 | 100 | 32% | 38-91, 108-109, 144-149, 179, 184, 189-202, 208, 214-224, 230-260, 265-270 |
| `src/agentcontract/plugin.py` | 87 | 55 | 37% | 61-64, 70-74, 83-106, 119-146, 152, 169-187 |
| `src/agentcontract/cli.py` | 68 | 42 | 38% | 33-39, 47-48, 52-61, 72-84, 89-130, 134 |
| `src/agentcontract/adapters/llamaindex.py` | 82 | 21 | 74% | 40, 48, 50, 62-66, 87, 108-110, 134, 168-175 |
| `src/agentcontract/adapters/langgraph.py` | 88 | 20 | 77% | 40, 45, 47, 49, 67-71, 95, 122, 130, 134-138, 151-153 |
| `src/agentcontract/assertions/engine.py` | 185 | 34 | 82% | 38, 41, 87, 95-96, 113-115, 124, 127-131, 134-141, 146, 155-159, 188, 203, 216, 251, 284, 290, 364 |
| `src/agentcontract/replay/engine.py` | 107 | 18 | 83% | 14, 17-18, 20, 120, 131-140, 148-149, 160, 170, 187-188 |
| `src/agentcontract/serialization.py` | 131 | 15 | 89% | 68-69, 78-79, 88-89, 113-116, 170, 191, 196, 207, 291-292 |
| `src/agentcontract/config.py` | 148 | 14 | 91% | 163, 218, 222, 233, 235, 238-239, 247, 249, 252-253, 261, 264-265 |

The package total for `src/agentcontract/*` is 1,351 statements, 345 missing,
74% coverage.

## Measurement Caveat

The pytest-cov invocation below currently underreports coverage for modules
imported during pytest plugin discovery:

```bash
uv run --extra dev pytest --cov=agentcontract --cov-report=term-missing
```

The run still passes, but coverage emits:

```text
CoverageWarning: Module agentcontract was previously imported, but not measured (module-not-measured)
```

In the same checkout, the pytest-cov invocation reported 54% total package
coverage and marked modules such as `agentcontract.plugin`,
`agentcontract.types`, and several package `__init__` modules as mostly
uncovered. Treat those numbers as a measurement artifact unless the test
invocation is changed so coverage starts before pytest loads the
`agentcontract` plugin.

## Ranked Gaps

1. `src/agentcontract/adapters/openai_agents.py` - 32%

   Existing tests cover `None` results, final-output fallback, simple message
   extraction, tool output items, and handoff items. The real gaps are the
   public `record_runner()` patching behavior and most tool-call extraction
   branches.

   Add coverage for:

   - `record_runner()` rejecting a non-`Recorder`, raising a helpful
     `ImportError` when `agents.Runner` cannot be imported, rejecting
     non-callable `Runner.run` or `Runner.run_sync`, and raising `ValueError`
     when neither method exists.
   - Sync and async runner patching, including original method restoration,
     result passthrough, latency capture, and extraction for both `run()` and
     `run_sync()`.
   - `new_items` dispatch in `_extract_from_result()` so list and tuple item
     collections use `_extract_from_items()`.
   - `ToolCallItem` extraction, including direct `name`, nested
     `function.name`, `id`, `call_id`, and empty-name skip behavior.
   - `_extract_message_content()` branches for `None`, missing content, empty
     string, list blocks as dicts, list blocks as objects, non-string content,
     and unsupported block types.
   - `_extract_message_tool_calls()` branches for missing calls, direct call
     names, nested function names, object and tuple collections, empty-name
     filtering, and empty final lists.
   - `_get_tool_arguments()` branches for dict values, JSON strings, malformed
     JSON strings that become `_raw`, `input`, `function.arguments` as dicts,
     valid function JSON, malformed function JSON, and no-argument fallback.
   - `_get_nested()` traversal when an intermediate value is `None` and when
     the default is returned.

   Recommended follow-up file: extend `tests/unit/test_adapters.py` with a
   focused `TestOpenAIAgentsRunnerPatching` class, or split this into
   `tests/unit/test_openai_agents_adapter.py` if the adapter tests become too
   large.

2. `src/agentcontract/plugin.py` - 37%

   There are no dedicated tests for the pytest plugin even though it is loaded
   through the `pytest11` entry point. The missing coverage is mostly fixture
   behavior and option handling.

   Add coverage for:

   - `pytest_addoption()` registering `--ac-record`, `--ac-replay`,
     `--ac-config`, and `--ac-scenarios`.
   - `pytest_configure()` registering `agentcontract` and `agent_scenario`
     markers.
   - `ac_config` loading an explicit config path and falling back to
     `AgentContractConfig.discover()`.
   - `ac_mode` returning `record`, `replay`, and `live`.
   - `ac_recorder` scenario name selection from marker positional args,
     `agent_scenario(name=...)`, and the test node name.
   - `ac_recorder` auto-saving only under `--ac-record`, honoring
     `--ac-scenarios`, and surfacing save errors via `pytest.fail()`.
   - `ac_replay_engine` returning `None` outside replay mode, skipping when the
     cassette is absent, loading a valid cassette, and failing clearly when
     cassette loading raises.
   - `ac_assert` and `ac_check_contract`, including default assertions,
     per-scenario overrides, extra assertions, and policy forwarding.

   Recommended follow-up file: add `tests/unit/test_plugin.py` using `pytester`
   for end-to-end plugin behavior plus direct fixture unit tests where simpler.

3. `src/agentcontract/cli.py` - 38%

   `tests/unit/test_cli.py` currently covers only `info` with an invalid
   cassette. Most subcommand dispatch and success/error paths are untested.

   Add coverage for:

   - `main([])` printing help and returning `0`.
   - `main(["info", path])` for missing files, valid cassette summaries, and
     loader failures.
   - `main(["validate", path])` for missing files, valid cassettes, and invalid
     cassette errors.
   - `main(["init"])` creating the starter `agentcontract.yml` in an empty
     working directory.
   - `init` refusing to overwrite an existing config.
   - `init` handling `OSError` from `Path.write_text()`.
   - The module entry point path guarded by `if __name__ == "__main__"` if CLI
     process-level coverage is desired.

   Recommended follow-up file: extend `tests/unit/test_cli.py`; use
   `tmp_path`, `monkeypatch.chdir()`, `capsys`, and small fixture cassettes from
   `tests/scenarios`.

4. `src/agentcontract/adapters/llamaindex.py` - 74%

   Existing tests cover a sync `chat()` response with tool sources, source
   nodes, and unpatching. The remaining risk is mostly validation, async
   wrappers, alternate response shapes, and skipped source entries.

   Add coverage for:

   - Non-`Recorder` validation.
   - Missing methods raising `ValueError`.
   - Non-callable agent method raising `TypeError`.
   - `achat()` and `aquery()` wrappers awaiting originals and preserving
     results.
   - `response.message` extraction through `_get_content()`.
   - Empty or missing response content with tool-only turns.
   - Source entries without `tool_name` being skipped.
   - `_get_content()` handling `None`, missing content, empty strings, and
     non-string content.

   Recommended follow-up file: extend `tests/unit/test_adapters.py` or create
   `tests/unit/test_llamaindex_adapter.py`.

5. `src/agentcontract/adapters/langgraph.py` - 77%

   Existing tests cover sync `invoke()`, dict messages, object messages,
   simple tool calls, unpatching, and non-dict results. Untested areas are
   validation, async invocation, non-list messages, and helper edge cases.

   Add coverage for:

   - Non-`Recorder` validation.
   - Non-callable `invoke` and `ainvoke` attributes.
   - Graphs without either method raising `ValueError`.
   - `ainvoke()` wrapper awaiting the original and restoring via `unpatch()`.
   - Results where `messages` is not a list or tuple.
   - Message objects with unknown or missing roles.
   - Content extraction for `None`, empty string, list content blocks, and
     non-string content.
   - Tool call extraction for dict tool calls using `arguments`, object calls
     using `arguments`, non-dict arguments, missing names, and empty final
     lists.

   Recommended follow-up file: extend `tests/unit/test_adapters.py` or create
   `tests/unit/test_langgraph_adapter.py`.

## Other Moderate Gaps

- `src/agentcontract/assertions/engine.py` is at 82%. The missing branches
  appear to be assertion/policy edge cases: unknown or malformed specs,
  negative checks, policy failures, and formatting paths.
- `src/agentcontract/replay/engine.py` is at 83%. The missing branches include
  tool-stub exhaustion and mismatch formatting, repeated call behavior, and
  replay lookup edge cases.
- `src/agentcontract/serialization.py` is at 89%. The missing branches are
  mostly coercion/default handling and malformed nested structures.
- `src/agentcontract/config.py` is at 91%. The missing branches are mostly
  discovery and optional config-section parsing edge cases.

These are lower priority than the adapter/plugin/CLI gaps because they already
have focused unit test files and materially higher coverage.

## Suggested Test Order

1. Add `tests/unit/test_plugin.py` with `pytester` and direct fixture tests.
   This fills an entire untested public integration surface.
2. Extend CLI tests in `tests/unit/test_cli.py` for all subcommands and return
   codes. This is low-cost and will quickly raise user-facing coverage.
3. Split OpenAI Agents SDK adapter tests into patching tests and helper
   extraction tests. This is the largest real behavioral gap and should use fake
   `agents.Runner` modules inserted into `sys.modules` rather than requiring the
   optional SDK.
4. Add async and validation cases for LangGraph and LlamaIndex adapters.
5. Sweep remaining assertion, replay, serialization, and config edge branches
   after the public integration surfaces are covered.
