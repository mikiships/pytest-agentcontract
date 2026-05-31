# Record And Replay

## Cassette Lifecycle

A cassette is a JSON file ending in `.agentrun.json`. It stores:

- run metadata, including the scenario name and recorder version
- model metadata, when an interceptor or recorder call captures it
- ordered turns with `system`, `user`, `assistant`, or `tool` roles
- tool calls with function names, argument objects, optional results, and optional durations
- summary data such as turn count, tool call count, token totals, and estimated cost

The pytest marker determines the cassette name. For `@pytest.mark.agentcontract("refund-eligible")`, the default path is:

```text
tests/scenarios/refund-eligible.agentrun.json
```

If a test has no `agentcontract` or `agent_scenario` marker, pytest-agentcontract uses the pytest node name as the scenario name.

## Record Mode

Run pytest with `--ac-record` to save the current recorder run after each marked test:

```bash
pytest --ac-record
```

`ac_recorder` records within a `Recorder.recording()` context managed by the fixture. Add turns manually:

```python
ac_recorder.add_turn(
    role="assistant",
    content="Let me look up order ORD-123.",
    tool_calls=[
        {
            "id": "tc_lookup",
            "function": "lookup_order",
            "arguments": {"order_id": "ORD-123"},
            "result": {"status": "delivered", "total": 79.99},
        }
    ],
)
```

The recorder accepts `role`, optional `content`, optional `tool_calls`, optional `latency_ms`, and token counts. Tool call arguments must be objects; non-object values are normalized to `{}` during recording or serialization.

Use `--ac-scenarios` to change where cassettes are written:

```bash
pytest --ac-record --ac-scenarios examples/customer_support/scenarios
```

## Replay Mode

Run pytest with `--ac-replay` to load the matching cassette:

```bash
pytest --ac-replay
```

In replay mode, `ac_replay_engine` is a `ReplayEngine` loaded from the cassette. Use `recorded_run` when you only need to assert the recorded trajectory:

```python
if ac_mode == "replay" and ac_replay_engine is not None:
    run = ac_replay_engine.recorded_run
```

Use `tool_stub` when your agent loop can replace real tool execution with recorded results:

```python
result = ac_replay_engine.tool_stub.get_result(
    "lookup_order",
    {"order_id": "ORD-123"},
)
```

`ToolStub.get_result()` returns recorded results in call order. It raises when the tool is exhausted or when provided arguments differ from the recorded arguments.

Call `ReplayEngine.finish(actual_turns)` to compare actual replay turns with the recorded turns. Calling `finish()` without turns checks whether all recorded tool results were consumed.

## CI Usage

A typical CI job runs replay mode only:

```bash
pytest --ac-replay
```

Replay mode should not need provider API keys when tests use `ac_replay_engine.recorded_run` or route tool calls through `ac_replay_engine.tool_stub`. Commit the `.agentrun.json` files needed by CI.

## Validation

Validate cassette structure before relying on a newly recorded file:

```bash
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
agentcontract info tests/scenarios/refund-eligible.agentrun.json
```

The CLI loads cassettes through the same serializer used by replay. For contract behavior, keep assertions focused on stable agent decisions: required tool calls, critical arguments, policy-sensitive operations, and final response invariants.
