# Recording And Replay

The pytest plugin provides fixtures for recording, loading cassettes, replaying tool results, and checking contracts.

## Pytest Options

```bash
pytest --ac-record
pytest --ac-replay
pytest --ac-config path/to/agentcontract.yml
pytest --ac-scenarios tests/scenarios
```

`--ac-record` switches `ac_mode` to `"record"` and makes `ac_recorder` save a cassette after the test.

`--ac-replay` switches `ac_mode` to `"replay"` and makes `ac_replay_engine` load the matching cassette.

`--ac-config` points config discovery at a specific YAML file.

`--ac-scenarios` overrides the scenario directory used by both `ac_recorder` and `ac_replay_engine`.

Without `--ac-record` or `--ac-replay`, `ac_mode` is `"live"` and `ac_recorder` records in memory without auto-saving.

## Fixtures

`ac_config` returns the parsed `AgentContractConfig`. If `--ac-config` is omitted, discovery walks up from the current working directory looking for `agentcontract.yml`; otherwise defaults are used.

`ac_mode` returns `"record"`, `"replay"`, or `"live"`.

`ac_recorder` returns a `Recorder` for the current scenario. The scenario comes from `@pytest.mark.agentcontract("name")`, `@pytest.mark.agent_scenario("name")`, or the test node name. In record mode it auto-saves to `<scenarios_dir>/<scenario>.agentrun.json`.

`ac_replay_engine` returns a `ReplayEngine` in replay mode. Outside replay mode it returns `None`. If the cassette is missing in replay mode, the fixture skips the test.

`ac_assert` returns a plain `AssertionEngine`.

`ac_check_contract` returns a callable that applies default assertions, matching scenario override assertions, optional `extra_assertions`, and policies.

## Manual Recorder Usage

```python
from agentcontract.recorder.core import Recorder

recorder = Recorder(
    scenario="refund-eligible",
    model_provider="openai",
    model_name="gpt-4o",
    seed=42,
)

with recorder.recording():
    recorder.add_turn(role="user", content="Refund order ORD-123")
    recorder.add_turn(
        role="assistant",
        content="Looking up your order.",
        tool_calls=[
            {
                "id": "tc_lookup",
                "function": "lookup_order",
                "arguments": {"order_id": "ORD-123"},
                "result": {"status": "delivered", "total": 49.99},
                "duration_ms": 8.5,
            }
        ],
        latency_ms=150.0,
        prompt_tokens=100,
        completion_tokens=40,
    )

recorder.save("tests/scenarios/refund-eligible.agentrun.json")
```

`Recorder.add_turn()` accepts roles `system`, `user`, `assistant`, and `tool`. It normalizes non-string content to strings, ignores non-dict tool-call entries, coerces missing/non-object tool arguments to `{}`, and records optional timing and token usage.

## ReplayEngine

The replay engine does not run your agent by itself. It gives your test access to the recorded run and a tool stub:

```python
from agentcontract.replay.engine import ReplayEngine

engine = ReplayEngine(recorded_run)
stub = engine.tool_stub

order = stub.get_result("lookup_order", {"order_id": "ORD-123"})
refund = stub.get_result("process_refund", {"order_id": "ORD-123", "amount": 49.99})

result = engine.finish()
assert result.ok, result.errors
```

`ReplayEngine.tool_stub.get_result(function, arguments=None)` returns recorded results in order per function. If `arguments` is provided, it must exactly match the recorded arguments for the next call.

`ReplayEngine.tool_stub.has_results(function)` reports whether more recorded results remain for a function.

`ReplayEngine.finish()` with no arguments checks that every recorded tool result was consumed. `ReplayEngine.finish(actual_turns=[...])` compares actual turns against recorded turns by role, tool-call count, tool function, and tool arguments. It returns a `ReplayResult` with `ok`, `matched_tools`, `mismatched_tools`, `missing_tools`, `extra_tools`, and `errors`.

## Lifecycle Guidance

Keep one cassette per scenario and commit it with the test that owns it.

Use record mode only when intentionally updating expected behavior. Review cassette diffs like code, especially tool names, arguments, results, token counts, and final responses.

Use replay mode in CI. The common pattern is to assert against `ac_replay_engine.recorded_run` for contract-only tests, or wire `ac_replay_engine.tool_stub` into your agent loop when you want to exercise more application code without calling external tools.

When using SDK interceptors, remember that OpenAI and Anthropic interceptors record model tool-call requests but not tool results. Add tool-result turns manually or backfill results if your replay checks need them.
