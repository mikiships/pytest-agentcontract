# Pytest Plugin Reference

`pytest-agentcontract` registers a pytest plugin entry point named `agentcontract`. Installing the package makes the plugin available automatically in pytest.

This page documents the user-facing pieces implemented in [`src/agentcontract/plugin.py`](../src/agentcontract/plugin.py): markers, flags, fixtures, scenario naming, and the record/replay workflow.

## Core Workflow

The built-in plugin supports three modes:

- `live`: default when no agentcontract flag is set.
- `record`: enabled with `--ac-record`.
- `replay`: enabled with `--ac-replay`.

Typical flow:

1. Mark a test with a scenario name.
2. Use `ac_recorder` to capture turns in record or live mode.
3. Use `ac_replay_engine` to load the matching cassette in replay mode.
4. Run `ac_check_contract(run)` to apply default assertions, scenario overrides, and policies from `agentcontract.yml`.

## Markers

The plugin registers two equivalent markers:

| Marker | Example | Notes |
| --- | --- | --- |
| `@pytest.mark.agentcontract("refund-eligible")` | positional scenario name | Preferred and used throughout the repository. |
| `@pytest.mark.agent_scenario("refund-eligible")` | alias | The fixture logic treats it the same as `agentcontract`. |
| `@pytest.mark.agent_scenario(name="refund-eligible")` | keyword scenario name | Supported because the plugin checks `marker.kwargs["name"]`. |

If neither marker is present, fixtures fall back to the pytest test function name as the scenario name.

## Scenario Names And Cassette Paths

`ac_recorder` and `ac_replay_engine` both derive the cassette filename from the scenario name:

```text
<scenarios_dir>/<scenario>.agentrun.json
```

The scenarios directory comes from:

1. `--ac-scenarios`, if provided.
2. Otherwise `tests/scenarios`.

Examples:

- `@pytest.mark.agentcontract("refund-eligible")` with default settings writes or reads `tests/scenarios/refund-eligible.agentrun.json`.
- The repository's example suite uses committed cassettes under `examples/customer_support/scenarios`, so those commands should pass `--ac-scenarios examples/customer_support/scenarios`.

## Command-Line Flags

| Flag | Effect |
| --- | --- |
| `--ac-record` | Sets `ac_mode` to `"record"` and causes `ac_recorder` to auto-save a cassette after the test finishes. |
| `--ac-replay` | Enables `ac_replay_engine`, which loads the cassette for the current scenario. |
| `--ac-config PATH` | Makes `ac_config` load that file instead of using automatic discovery. |
| `--ac-scenarios PATH` | Overrides the directory used for cassette save/load. |

Notes:

- `--ac-record` and `--ac-replay` are intended to be used separately.
- If a replay cassette is missing, `ac_replay_engine` skips the test instead of failing it.
- `--ac-config` affects `ac_config` and anything built on top of it, especially `ac_check_contract`.

## Fixtures

### `ac_config`

Returns an `AgentContractConfig`.

- Uses `AgentContractConfig.from_file(Path(...))` when `--ac-config` is set.
- Otherwise uses `AgentContractConfig.discover()`.

### `ac_mode`

Returns one of:

- `"record"`
- `"replay"`
- `"live"`

This is the simplest switch for branching your test body between live execution and replay.

### `ac_recorder`

Returns a `Recorder` inside its `recording()` context manager.

Behavior:

- Derives the scenario name from `agentcontract`, `agent_scenario`, or the test name.
- Records turns you add with `ac_recorder.add_turn(...)`.
- Auto-saves at the end of the test only when `--ac-record` is set.
- Writes to `<scenarios_dir>/<scenario>.agentrun.json`.

The fixture does not automatically intercept SDKs for you. If you want auto-recording, patch your SDK or framework with the helper functions described in the README.

### `ac_replay_engine`

Returns a `ReplayEngine | None`.

Behavior:

- Returns `None` unless `--ac-replay` is set.
- Loads the cassette matching the current scenario.
- Calls `pytest.skip(...)` if the cassette file does not exist.
- Exposes `recorded_run` and `tool_stub`.

The simplest replay pattern is to reuse the recorded run directly for assertions:

```python
if ac_mode == "replay" and ac_replay_engine is not None:
    run = ac_replay_engine.recorded_run
else:
    ...
```

If your agent loop can swap real tool calls for recorded outputs, you can use `ac_replay_engine.tool_stub.get_result(...)` and later call `ac_replay_engine.finish(actual_turns)` to diff the live replay against the cassette.

### `ac_assert`

Returns a fresh `AssertionEngine`.

Use this when you want to bypass config merging and call `AssertionEngine.check(...)` directly.

### `ac_check_contract`

Returns a callable:

```python
result = ac_check_contract(run, extra_assertions=[...])
```

The callable merges:

1. `ac_config.default_assertions`
2. `ac_config.overrides[run.metadata.scenario].assertions`, if present
3. `extra_assertions`, if provided

It then calls `AssertionEngine.check(run, assertions=..., policies=ac_config.policies)`.

## Example: Customer Support Suite

The repository example in [`examples/customer_support/test_support.py`](../examples/customer_support/test_support.py) uses these scenario names:

- `refund-eligible`
- `refund-not-delivered`
- `refund-with-policies`

Those cassettes live under [`examples/customer_support/scenarios`](../examples/customer_support/scenarios), so the matching commands are:

```bash
pytest examples/customer_support/test_support.py \
  --ac-record \
  --ac-scenarios examples/customer_support/scenarios \
  -v

pytest examples/customer_support/test_support.py \
  --ac-replay \
  --ac-scenarios examples/customer_support/scenarios \
  -v
```

A minimal pattern that matches the example tests:

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_happy_path(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay" and ac_replay_engine is not None:
        run = ac_replay_engine.recorded_run
    else:
        turns = run_support_agent("I'd like a refund for order ORD-123 please")
        for turn in turns:
            ac_recorder.add_turn(
                role=turn["role"],
                content=turn.get("content"),
                tool_calls=turn.get("tool_calls"),
            )
        run = ac_recorder.run

    result = ac_check_contract(run)
    assert result.passed, [failure.message for failure in result.failures()]
```

## Related References

- [Configuration Reference](configuration.md)
- [Repository README](../README.md)
