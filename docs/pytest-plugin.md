# Pytest Plugin Reference

The pytest plugin is registered through the `pytest11` entry point as `agentcontract`. It adds markers, fixtures, and CLI flags for recording and replaying agent trajectories.

## Markers

### `@pytest.mark.agentcontract("scenario-name")`

Primary marker for agent contract tests. The first positional argument becomes the cassette name.

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(...):
    ...
```

### `@pytest.mark.agent_scenario(...)`

Alias for `agentcontract`. The plugin accepts either the first positional argument or `name=...`.

```python
@pytest.mark.agent_scenario("refund-eligible")
def test_refund_flow(...):
    ...

@pytest.mark.agent_scenario(name="refund-eligible")
def test_refund_flow_named(...):
    ...
```

If neither marker is present, `ac_recorder` and `ac_replay_engine` fall back to the pytest test function name for the cassette filename.

## Fixtures

### `ac_config`

Returns an `AgentContractConfig`.

- If `--ac-config PATH` is set, the plugin loads that file.
- Otherwise it searches upward from the current working directory for `agentcontract.yml`.
- If no config file is found, default values are used.

### `ac_mode`

Returns one of:

- `"record"` when `--ac-record` is set
- `"replay"` when `--ac-replay` is set
- `"live"` otherwise

### `ac_recorder`

Returns a `Recorder` wrapped in `with recorder.recording():`.

- The fixture is available in every mode.
- In `--ac-record` mode it auto-saves after the test finishes.
- The saved cassette path is `tests/scenarios/<scenario>.agentrun.json` by default.
- `--ac-scenarios PATH` overrides the base directory used for save and replay lookup.

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_check_contract):
    ac_recorder.add_turn(role="user", content="I want a refund")
    ac_recorder.add_turn(
        role="assistant",
        content="Checking your order.",
        tool_calls=[
            {
                "id": "tc1",
                "function": "lookup_order",
                "arguments": {"order_id": "123"},
                "result": {"status": "delivered"},
            }
        ],
    )

    result = ac_check_contract(ac_recorder.run)
    assert result.passed, result.failures()
```

### `ac_replay_engine`

Returns `ReplayEngine` in replay mode, otherwise `None`.

- Replay loads `tests/scenarios/<scenario>.agentrun.json` unless `--ac-scenarios` overrides the directory.
- If the cassette file is missing, the test is skipped.
- Use `ac_replay_engine.recorded_run` to assert directly against the cassette.
- Use `ac_replay_engine.tool_stub.get_result(function, arguments)` if your test drives your own replay loop.
- Use `ac_replay_engine.finish(actual_turns)` to compare an actual replayed turn list against the recording.

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay" and ac_replay_engine is not None:
        run = ac_replay_engine.recorded_run
    else:
        run_my_agent(ac_recorder)
        run = ac_recorder.run

    result = ac_check_contract(run)
    assert result.passed, result.failures()
```

### `ac_assert`

Returns a bare `AssertionEngine`. Use it when you want to evaluate assertions or policies manually instead of going through config-driven defaults.

```python
def test_manual_assertions(ac_assert, ac_recorder):
    result = ac_assert.check(ac_recorder.run, assertions=[...], policies=[...])
    assert result.passed
```

### `ac_check_contract`

Returns a callable with signature:

```python
ac_check_contract(run, extra_assertions=None)
```

The callable:

- starts with `ac_config.default_assertions`
- appends `overrides[run.metadata.scenario].assertions` when present
- appends any `extra_assertions`
- runs all configured policies from `ac_config.policies`

It returns `ContractResult`.

## Cassette Naming and Paths

Scenario resolution order:

1. First positional argument from `agentcontract` or `agent_scenario`
2. `name=` keyword argument from either marker
3. `request.node.name`

Path resolution:

1. `--ac-scenarios PATH` when provided
2. `tests/scenarios` otherwise

The final filename is always `<scenario>.agentrun.json`.

## Pytest CLI Options

### `--ac-record`

Enables recording mode. The test still runs your live agent code, but `ac_recorder` auto-saves a cassette after the test.

```bash
pytest --ac-record -k refund_flow
```

### `--ac-replay`

Enables replay mode. `ac_replay_engine` loads the matching cassette, and tests can assert against `recorded_run` or consume `tool_stub`.

```bash
pytest --ac-replay -k refund_flow
```

### `--ac-config PATH`

Loads configuration from a specific YAML file instead of discovery.

```bash
pytest --ac-config config/agentcontract.yml --ac-replay
```

### `--ac-scenarios PATH`

Overrides the base cassette directory for both recording and replay.

```bash
pytest --ac-record --ac-scenarios examples/customer_support/scenarios
```

## Record and Replay Workflow

1. Mark the test with a stable scenario name.
2. In live or record mode, run your real agent and add turns to `ac_recorder`.
3. Run `pytest --ac-record` once to save the cassette.
4. Run `pytest --ac-replay` in CI to load the cassette and assert against `ac_replay_engine.recorded_run` or your own replay loop.

The plugin does not automatically execute your agent in replay mode. Your test decides whether to assert against the recorded run directly or to drive your code with `ReplayEngine.tool_stub`.
