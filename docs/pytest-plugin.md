# Pytest Plugin Guide

`pytest-agentcontract` ships as a pytest plugin entry point. The plugin adds pytest flags, two markers, and six fixtures for record/replay contract testing.

## CLI Flags

| Flag | Behavior |
| --- | --- |
| `--ac-record` | Puts `ac_mode` into `"record"` and makes `ac_recorder` auto-save a cassette after the test. |
| `--ac-replay` | Puts `ac_mode` into `"replay"` and makes `ac_replay_engine` load the matching cassette. |
| `--ac-config PATH` | Loads `AgentContractConfig` from the given file instead of discovery. |
| `--ac-scenarios PATH` | Overrides the cassette directory used by `ac_recorder` and `ac_replay_engine`. |

Notes:

- The default mode is `"live"` when neither `--ac-record` nor `--ac-replay` is set.
- The plugin checks `--ac-record` before `--ac-replay`. Do not combine both flags in the same run.
- `--ac-scenarios` only changes cassette save/load paths. It does not change `scenarios.include` or `scenarios.exclude` values inside `agentcontract.yml`.

## Markers

The plugin registers two equivalent markers:

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(...):
    ...


@pytest.mark.agent_scenario(name="refund-eligible")
def test_refund_flow_alias(...):
    ...
```

Scenario resolution for `ac_recorder` and `ac_replay_engine` is:

1. First positional marker argument, if present.
2. Marker keyword argument `name`, if present.
3. `request.node.name` as a fallback.

That resolved scenario name becomes the cassette filename.

## Fixtures

### `ac_config`

Returns an `AgentContractConfig`.

- With `--ac-config PATH`, it loads exactly that file.
- Otherwise it walks upward from the current working directory looking for `agentcontract.yml`.
- If no file is found, it returns a config object populated with defaults.

### `ac_mode`

Returns one of:

- `"live"`
- `"record"`
- `"replay"`

Use this to branch between real execution and cassette-backed execution inside a test.

### `ac_recorder`

Yields a `Recorder` already wrapped in `with recorder.recording():`.

Typical usage:

```python
def test_agent(ac_recorder):
    turns = run_my_agent(...)
    for turn in turns:
        ac_recorder.add_turn(
            role=turn["role"],
            content=turn.get("content"),
            tool_calls=turn.get("tool_calls"),
        )
```

Behavior:

- In all modes, it gives you a live `Recorder` object and finalizes run metadata after the test.
- In `--ac-record` mode, it auto-saves to `<scenarios_dir>/<scenario>.agentrun.json`.
- The default `scenarios_dir` is `tests/scenarios`.
- `Recorder.save()` creates missing parent directories, so a new scenario directory can be created on first record.

### `ac_replay_engine`

Returns a `ReplayEngine` in `--ac-replay` mode and `None` otherwise.

Behavior in replay mode:

- It loads `<scenarios_dir>/<scenario>.agentrun.json`.
- If the cassette is missing, the test is skipped.
- If the cassette cannot be parsed, the test fails immediately.

Useful properties and methods:

- `ac_replay_engine.recorded_run`: the recorded `AgentRun`.
- `ac_replay_engine.tool_stub.get_result(function, arguments)`: fetches recorded tool results in call order.
- `ac_replay_engine.finish(actual_turns=None)`: checks stub consumption or compares actual turns against the recorded run.

There is no `run()` helper on `ReplayEngine`; replay is driven by your test or your agent loop.

### `ac_assert`

Returns a bare `AssertionEngine`.

Use it when you want complete control over which assertions or policies are evaluated. It does not merge config defaults for you.

### `ac_check_contract`

Returns a callable:

```python
result = ac_check_contract(run, extra_assertions=[...])
```

It evaluates:

1. `defaults.assertions` from config.
2. `overrides[run.metadata.scenario].assertions`, if present.
3. Any `extra_assertions` passed at call time.
4. All configured `policies`.

This is the highest-level fixture when you want config-driven assertions.

## Cassette Naming And Location

The cassette path is always derived from the resolved scenario name:

```text
<scenarios_dir>/<scenario>.agentrun.json
```

Examples:

- `tests/scenarios/refund-eligible.agentrun.json`
- `examples/customer_support/scenarios/refund-not-delivered.agentrun.json`

Use `--ac-scenarios examples/customer_support/scenarios` when you want tests to read or write cassettes outside the default `tests/scenarios` directory.

## Recording Strategies

### Manual turn recording

Best when your app already exposes structured turns:

```python
for turn in turns:
    ac_recorder.add_turn(
        role=turn["role"],
        content=turn.get("content"),
        tool_calls=turn.get("tool_calls"),
    )
```

### SDK interceptors

Available helpers:

- `patch_openai(client, recorder)`
- `patch_anthropic(client, recorder)`

These record assistant messages and tool-call requests from the SDK response. They do not automatically capture tool results produced later by your application code.

### Framework adapters

Available helpers:

- `record_graph(graph, recorder)` for LangGraph
- `record_agent(agent, recorder)` for LlamaIndex
- `record_runner(recorder)` for the OpenAI Agents SDK

All three return an `unpatch()` function.

## Recommended Test Structure

### Live mode

Run the real agent and assert on the current run:

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_check_contract):
    assert ac_mode == "live"
    turns = run_my_agent(...)
    for turn in turns:
        ac_recorder.add_turn(
            role=turn["role"],
            content=turn.get("content"),
            tool_calls=turn.get("tool_calls"),
        )

    result = ac_check_contract(ac_recorder.run)
    assert result.passed, [failure.message for failure in result.failures()]
```

### Record mode

Use the same test body, but invoke pytest with `--ac-record`. The fixture auto-saves the finished run after the test.

```bash
pytest path/to/test_file.py --ac-record
```

### Replay mode

Pick one of two patterns:

1. Contract-only verification against the stored run:

```python
if ac_mode == "replay" and ac_replay_engine is not None:
    run = ac_replay_engine.recorded_run
```

2. Full deterministic tool replay inside your agent loop:

```python
stub = ac_replay_engine.tool_stub
order = stub.get_result("lookup_order", {"order_id": "ORD-123"})
```

If you build real replay turns, call `ac_replay_engine.finish(actual_turns)` to compare them against the recording.
