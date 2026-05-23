# Test Coverage Gap Report

Date: 2026-05-23

## Baseline

Command:

```bash
uv run pytest --cov=agentcontract --cov-report=term-missing
```

Observed result:

- 77 tests passed.
- Total coverage: 54%.
- Coverage emitted `module-not-measured` for `agentcontract` because the package is imported by
  pytest plugin discovery before coverage starts.

## Coverage Noise

Some of the lowest percentages are import-timing artifacts rather than direct product risk. The
pytest entry point imports `agentcontract.plugin` before coverage measurement, and that import pulls
in package exports and dataclass modules. Treat these files as lower-signal coverage indicators
unless their runtime behavior is also untested:

- `src/agentcontract/__init__.py`
- `src/agentcontract/types.py`
- `src/agentcontract/assertions/__init__.py`
- `src/agentcontract/recorder/__init__.py`
- `src/agentcontract/replay/__init__.py`

The uncovered fixture bodies and branch logic in `src/agentcontract/plugin.py` are not just noise.
They are the main user-facing pytest integration path and need direct tests.

## Highest-Risk Gaps

| File | Coverage | Gap | Risk |
| --- | ---: | --- | --- |
| `src/agentcontract/plugin.py` | 2% | Fixture behavior is essentially untested: `ac_mode`, `ac_config`, `ac_recorder`, `ac_replay_engine`, and `ac_check_contract` are not exercised under pytest. | The package is primarily a pytest plugin, so regressions here can break record/replay workflows while unit tests stay green. |
| `src/agentcontract/cli.py` | 38% | Only one invalid `info` cassette path is tested. Success paths and most error paths for `info`, `validate`, and `init` are uncovered. | CLI regressions would affect quick validation, cassette inspection, and first-run setup. |
| `src/agentcontract/adapters/openai_agents.py` | 32% | Broad adapter behavior is untested: `record_runner`, sync/async patching, missing SDK handling, invalid Runner shapes, tool-call item extraction, content block parsing, and JSON argument coercion. | OpenAI Agents SDK support is an integration surface where SDK shape changes are likely. |
| `src/agentcontract/config.py` | 45% | Core parsing has coverage, but edge cases around malformed sections, missing required assertion/policy fields, discovery fallbacks, and coercion failures need more direct tests. | Bad config should fail or fall back predictably instead of causing confusing pytest failures. |
| `src/agentcontract/replay/engine.py` | 57% | Replay covers happy paths and several mismatches, but misses unconsumed stub checks, role mismatch reporting, argument-difference reporting, and extra/missing turn branches. | Replay diagnostics are central to deterministic CI failures. |
| `src/agentcontract/recorder/core.py` | 65% | Basic recording is covered, but save suffix behavior, invalid roles, save failures, and coercion edge cases remain uncovered. | Recorder errors should be clear because they create the cassette artifact used by replay. |

## Priority Table

| Priority | Area | Target files | Why first | Suggested tests |
| --- | --- | --- | --- | --- |
| P0 | Pytest plugin integration | `src/agentcontract/plugin.py`, new `tests/unit/test_plugin.py` | This is the primary public API and the current coverage does not exercise actual fixture usage. | Use `pytester` to run tiny test modules with `--ac-record`, `--ac-replay`, `--ac-config`, and `--ac-scenarios`; assert marker scenario selection, auto-save behavior, missing cassette skip, invalid cassette failure, and `ac_check_contract` merging defaults with overrides. |
| P1 | CLI command behavior | `src/agentcontract/cli.py`, `tests/unit/test_cli.py` | The CLI is small, stable, and high leverage to cover; it guards user setup and cassette inspection. | Add success and missing-file tests for `info` and `validate`; test `init` creates the expected YAML, refuses to overwrite, and reports write failures via monkeypatching. |
| P2 | OpenAI Agents SDK adapter branches | `src/agentcontract/adapters/openai_agents.py`, `tests/unit/test_adapters.py` | Adapter code handles many SDK object shapes and currently tests only a narrow extraction subset. | Inject a fake `agents.Runner` module through `sys.modules`; test sync and async patch/unpatch, non-Recorder TypeError, missing SDK ImportError, missing Runner methods, `ToolCallItem`, message content block lists, `tool_calls`, invalid JSON arguments, and `function.arguments`. |
| P3 | Config, replay, and recorder edge cases | `src/agentcontract/config.py`, `src/agentcontract/replay/engine.py`, `src/agentcontract/recorder/core.py` | These are important but lower-risk than the public plugin and command surfaces. | Add focused tests for config malformed scalar/list fallbacks and missing keys; replay `finish(None)` unconsumed tools, role mismatches, argument mismatches, extra turns with tools, and missing turns; recorder invalid role handling, implicit `.agentrun.json` suffix, and save error propagation. |

## Recommended Next Step

Start with a narrow plugin test file using pytest's `pytester` fixture. A small suite that proves
the CLI flags, markers, record auto-save path, replay cassette loading, and contract assertion
merging work together would cover the highest-risk behavior and should move total coverage more than
isolated import/export tests.
