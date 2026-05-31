# Record And Replay

pytest-agentcontract stores each trajectory as a JSON cassette with the
`.agentrun.json` suffix. A cassette contains run metadata, model metadata,
summary counts, and ordered turns. Turns may include assistant content, tool
calls, tool arguments, tool results, timing, and token counts.

## Cassette Lifecycle

1. Add a scenario marker to a test.
2. Run the test with `--ac-record`.
3. The `ac_recorder` fixture records turns during the test.
4. After the test body finishes, the fixture saves the cassette.
5. Commit the cassette with the test.
6. Run the test later with `--ac-replay` to load the cassette and avoid live
   provider or tool calls.

By default, cassettes are saved to:

```text
tests/scenarios/<scenario>.agentrun.json
```

The scenario is resolved from `@pytest.mark.agentcontract("name")`,
`@pytest.mark.agent_scenario("name")`, or the pytest test name. Use
`--ac-scenarios <dir>` to change the cassette directory for both recording and
replay.

## Record Mode

```bash
pytest --ac-record -k refund_happy_path
```

In record mode:

- `ac_mode` is `"record"`.
- `ac_recorder` starts an in-memory `Recorder`.
- Your test runs the real agent, tools, and provider calls.
- The fixture writes `<scenario>.agentrun.json` after the test.

Manual recording uses `Recorder.add_turn(...)`:

```python
ac_recorder.add_turn(
    role="assistant",
    content="Your refund has been processed.",
    tool_calls=[
        {
            "id": "tc1",
            "function": "process_refund",
            "arguments": {"order_id": "123", "amount": 49.99},
            "result": {"success": True},
        }
    ],
)
```

The valid roles are `system`, `user`, `assistant`, and `tool`.

## Replay Mode

```bash
pytest --ac-replay
```

In replay mode:

- `ac_mode` is `"replay"`.
- `ac_replay_engine` loads the matching cassette.
- If the cassette is missing, the fixture skips the test.
- If the cassette cannot be loaded, the fixture fails the test.

The replay engine exposes the original run and a tool stub:

```python
run = ac_replay_engine.recorded_run

result = ac_replay_engine.tool_stub.get_result(
    "lookup_order",
    {"order_id": "123"},
)
```

`ToolStub.get_result(function, arguments)` returns recorded tool results in
recorded order. If arguments are supplied, they must exactly match the recorded
arguments for that call. A missing recorded result raises `ToolStubExhausted`;
an argument mismatch raises `ToolStubArgumentsMismatch`.

After driving your own agent loop with stubbed tools, compare actual turns:

```python
replay_result = ac_replay_engine.finish(actual_turns)
assert replay_result.ok, replay_result.errors
```

When `finish()` is called without `actual_turns`, it checks that all recorded
tool stubs were consumed.

## CI Usage

Record cassettes locally or in a trusted environment with real credentials:

```bash
pytest --ac-record
```

Then run deterministic replay in CI:

```bash
pytest --ac-replay
```

Replay mode does not need provider API keys when your tests use the recorded run
or wire agent tool calls through `ac_replay_engine.tool_stub`. It is still your
test's responsibility to avoid live network calls in replay paths.

## Validation Guidance

Use the CLI to check cassette structure:

```bash
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
agentcontract info tests/scenarios/refund-eligible.agentrun.json
```

`validate` loads the cassette through the same deserializer used by replay. It
does not prove semantic correctness; keep assertions and policies in your tests
or config to validate behavior.
