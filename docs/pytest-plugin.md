# Pytest Plugin Reference

The pytest plugin is registered through the `pytest11` entry point and adds a small set of CLI options, markers, and fixtures.

## CLI Options

| Option | Effect |
| --- | --- |
| `--ac-record` | Puts `ac_mode` into `"record"` and makes `ac_recorder` auto-save a cassette after the test. |
| `--ac-replay` | Puts `ac_mode` into `"replay"` and makes `ac_replay_engine` load the matching cassette. |
| `--ac-config PATH` | Loads `AgentContractConfig` from the given file instead of discovery. |
| `--ac-scenarios PATH` | Overrides the cassette directory used by `ac_recorder` and `ac_replay_engine`. |

The plugin does not expand `scenarios.include` globs when locating a cassette for a test. It always uses a single scenario name plus a directory:

`<scenarios_dir>/<scenario>.agentrun.json`

The directory is `tests/scenarios` by default, or the value of `--ac-scenarios` when provided.

## Markers

Two markers are registered:

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(...):
    ...


@pytest.mark.agent_scenario(name="refund-eligible")
def test_refund_flow_alias(...):
    ...
```

Scenario resolution follows this order:

1. `@pytest.mark.agentcontract("name")`
2. `@pytest.mark.agent_scenario(name="name")`
3. `request.node.name`

The fallback is the pytest node name, not just the Python function name. For parametrized tests that means the cassette filename includes the parametrization suffix, for example `test_refund_flow[ORD-123].agentrun.json`.

## Fixtures

### `ac_config`

Returns an `AgentContractConfig`.

- With `--ac-config`, it loads that exact file.
- Without it, `AgentContractConfig.discover()` walks up from the current working directory until it finds `agentcontract.yml`.
- If no config file exists, the fixture returns a config populated with defaults.

### `ac_mode`

Returns one of:

- `"record"` when `--ac-record` is set
- `"replay"` when `--ac-replay` is set
- `"live"` otherwise

If both flags are supplied, `--ac-record` wins because the fixture checks it first.

### `ac_recorder`

Returns a `Recorder` initialized with the resolved scenario name and wraps the test in `with recorder.recording():`.

When `--ac-record` is active, the fixture saves the run after the test to:

`tests/scenarios/<scenario>.agentrun.json`

or to:

`<ac-scenarios>/<scenario>.agentrun.json`

if `--ac-scenarios` is set.

Inside the test, `ac_recorder.run` already contains the turns you added. The summary fields are finalized when the fixture exits, which is the same point where auto-save happens.

### `ac_replay_engine`

Returns `ReplayEngine` in replay mode and `None` otherwise.

Behavior in replay mode:

- The fixture resolves the same scenario name and cassette path pattern as `ac_recorder`.
- If the cassette file is missing, the test is skipped.
- If the cassette cannot be loaded, the fixture fails the test.

The fixture only loads `ReplayEngine(run)`. It does not automatically execute your agent or stub tools on your behalf.

### `ac_assert`

Returns a plain `AssertionEngine`.

Use it when you want to bypass config merging and call `engine.check(run, assertions=..., policies=...)` yourself.

### `ac_check_contract`

Returns a callable with this shape:

```python
result = ac_check_contract(run, extra_assertions=None)
```

It merges:

1. `ac_config.default_assertions`
2. Scenario-specific `ac_config.overrides[run.metadata.scenario].assertions`, if present
3. Any `extra_assertions` passed at call time

It then evaluates those assertions plus `ac_config.policies` through `AssertionEngine.check(...)`.

## Replay Patterns

### Assert the recorded run directly

This is the pattern used in `examples/customer_support/test_support.py`:

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
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
    assert result.passed, result.failures()
```

### Drive your own replay loop

Use `tool_stub` if you want to exercise your own agent logic offline:

```python
def test_refund_replay(ac_replay_engine):
    assert ac_replay_engine is not None

    tool_stub = ac_replay_engine.tool_stub
    order = tool_stub.get_result("lookup_order", {"order_id": "123"})
    refund = tool_stub.get_result("process_refund", {"order_id": "123", "amount": 49.99})

    actual_turns = [...]
    result = ac_replay_engine.finish(actual_turns)
    assert result.ok, result.errors
```

`ToolStub.get_result(...)` returns recorded results in call order. If you request a tool too many times it raises `ToolStubExhausted`; if the arguments do not match the next recorded call it raises `ToolStubArgumentsMismatch`.
