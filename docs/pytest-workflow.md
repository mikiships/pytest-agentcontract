# Pytest recording and replay workflow

The plugin provides fixtures for recording turns and checking saved trajectories.
Installing `pytest-agentcontract` registers it with pytest automatically. The
[README quick start](../README.md#quick-start) and
[customer-support example](../examples/customer_support/test_support.py) select an
`AgentRun` according to the mode and then check that run.

## Complete offline example

Save the following as `test_refund.py` in a scratch directory. It needs only
`pytest-agentcontract` installed in your active Python environment. Its turns and
tool results are deterministic in live and record modes; run IDs, timestamps,
and measured duration will vary between recordings.

```python
import pytest

from agentcontract.config import AssertionSpec


@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay":
        assert ac_replay_engine is not None
        run = ac_replay_engine.recorded_run
    else:
        # Deterministic stand-in for an agent: no SDK or credentials needed.
        ac_recorder.add_turn(role="user", content="Refund order 123")
        ac_recorder.add_turn(
            role="assistant",
            tool_calls=[{
                "id": "refund-1",
                "function": "process_refund",
                "arguments": {"order_id": "123", "amount": 49.99},
                "result": {"status": "refunded"},
            }],
        )
        ac_recorder.add_turn(
            role="assistant", content="Your refund of $49.99 has been processed."
        )
        run = ac_recorder.run

    contract = ac_check_contract(
        run,
        extra_assertions=[
            AssertionSpec(type="contains", target="final_response", value="$49.99"),
            AssertionSpec(
                type="called_with", target="tool:process_refund",
                schema={"order_id": "123", "amount": 49.99},
            ),
            AssertionSpec(type="called_count", target="tool:process_refund", value="1"),
        ],
    )
    assert contract.passed, [failure.message for failure in contract.failures()]
```

Run these commands from that scratch directory (POSIX shell):

```bash
# Isolate this example from any discovered project assertions or policies.
printf 'version: "1"\n' > agentcontract.yml
SCENARIOS_DIR="$(mktemp -d)"

# Live is the default: populate the recorder, check it, do not auto-save.
python -m pytest test_refund.py --ac-config agentcontract.yml -q

# Record: save at fixture teardown, after the test has checked its turns.
python -m pytest test_refund.py --ac-config agentcontract.yml --ac-record --ac-scenarios "$SCENARIOS_DIR" -q
test -f "$SCENARIOS_DIR/refund-eligible.agentrun.json"

# Replay: select the saved AgentRun and check it without credentials.
python -m pytest test_refund.py --ac-config agentcontract.yml --ac-replay --ac-scenarios "$SCENARIOS_DIR" -q

# A directory with no matching cassette causes a skip, shown by -rs.
python -m pytest test_refund.py --ac-config agentcontract.yml --ac-replay --ac-scenarios "$SCENARIOS_DIR/missing" -q -rs
```

The first three pytest commands each pass one test. The last skips one test with
`No cassette found at <directory>/refund-eligible.agentrun.json`.

In replay mode this example checks historical data against current assertions;
it **does not execute the current agent implementation**. It therefore cannot
catch changes to agent behavior unless you execute that implementation separately.
The flag itself neither patches SDK calls nor blocks network access. Real agent
integrations must explicitly record turns or install recording interceptors, and
must arrange their own offline model/tool behavior during replay.

## Fixtures

All six fixtures have function scope.

| Fixture | Behavior |
| --- | --- |
| `ac_config` | Loads an `AgentContractConfig` from `--ac-config` or discovery. |
| `ac_mode` | Returns `"live"` by default, `"record"` for `--ac-record`, or `"replay"` for `--ac-replay`. Record takes precedence when both flags are supplied. |
| `ac_recorder` | Yields a `Recorder` inside its recording context in every mode. Add turns with `add_turn(...)`; access the current `AgentRun` through `.run`. |
| `ac_replay_engine` | Returns `None` without `--ac-replay`; otherwise loads the scenario cassette into a `ReplayEngine`. A missing cassette skips the requesting test during fixture setup. Invalid or unreadable cassettes fail setup. |
| `ac_assert` | Provides an `AssertionEngine`; call `.check(run, assertions=[...], policies=[...])` for explicit checks. It depends on `ac_config` but does not apply its defaults automatically. |
| `ac_check_contract` | Callable accepting `run` and optional `extra_assertions`. Combines configured default assertions, assertions under `overrides[run.metadata.scenario]`, and extras, then checks configured policies. Returns a `ContractResult`; assert `.passed` and inspect `.failures()`. |

Scenario overrides append assertions; they do not replace defaults. With no
assertions or policies, a contract check passes without checking anything. Use
explicit assertions as above or configure meaningful checks.

## Markers and cassette names

Use `@pytest.mark.agentcontract("refund-eligible")` or its alias
`@pytest.mark.agent_scenario(name="refund-eligible")`. Both accept the first
positional argument as the scenario name, then a truthy `name=` keyword if no
positional argument exists. Otherwise the name falls back to `request.node.name`
(the pytest test name, including parametrization suffixes). Other marker keyword
arguments are not used by these fixtures. If both markers are present,
`agentcontract` takes priority.

Recording and replay use `<scenarios-directory>/<scenario>.agentrun.json`.
The directory defaults to `tests/scenarios`, relative to the current working
directory. Give tests distinct scenario names to avoid overwriting one another.

At `ac_recorder` teardown, the recording context finalizes the summary (turn count,
tool count, tokens, and duration). With `--ac-record`, the fixture then creates
parent directories and writes the cassette, replacing an existing file at that
path. Saving is not conditional on the test passing. Live and replay modes do not
auto-save unless `--ac-record` is also supplied. The finalized summary and saved
file are available after teardown, not while the test body is still running.
Save errors are reported as pytest teardown failures.

## Options and configuration

| Option | Behavior |
| --- | --- |
| `--ac-record` | Makes `ac_mode` return `"record"` and enables recorder auto-save. |
| `--ac-replay` | Enables cassette loading when `ac_replay_engine` is requested. |
| `--ac-config PATH` | Loads this YAML file instead of discovering configuration; relative paths use the current working directory. |
| `--ac-scenarios DIR` | Sets the directory used by both recorder and replay fixtures. Relative paths use the current working directory. |

Use one mode flag at a time. Although `ac_mode` gives record precedence,
`ac_replay_engine` independently checks `--ac-replay`; requesting it with both
flags still loads a cassette or skips if none exists.

Without `--ac-config`, discovery searches from the current working directory up
through its parents for the first `agentcontract.yml`. If none exists, default
configuration is used (including empty assertions and policies). An explicit
config path must exist and be readable.

The pytest cassette fixtures do not consult `scenarios.include` or
`scenarios.exclude`; `--ac-scenarios` or `tests/scenarios` determines their paths.
Likewise, the fixtures do not use the configured `replay` settings to patch tools,
select a model, set a seed, or schedule concurrent execution.

## Integrating replay into agent execution

`ReplayEngine` has no `run()` method. Its `recorded_run` property returns the
loaded trajectory. For actual execution, explicitly route your agent's tool
calls through `engine.tool_stub.get_result(function, arguments)` and collect the
new turns yourself. For example, inside a replay test using the cassette above:

```python
engine = ac_replay_engine
assert engine is not None
refund = engine.tool_stub.get_result(
    "process_refund", {"order_id": "123", "amount": 49.99}
)
assert refund == {"status": "refunded"}
consumption = engine.finish()
assert consumption.ok, consumption.errors
```

The stub returns recorded results in order **per function**, not in global
cross-tool order. Supplying arguments checks exact dictionary equality against
the next recorded call; omitting arguments (or passing `None`) skips this check.
An argument mismatch raises `ToolStubArgumentsMismatch` without consuming the
result. Unknown tools or exhausted results raise `ToolStubExhausted`. Both
exceptions live in `agentcontract.replay.engine`. `has_results(function)` reports
whether that function has unconsumed results.

`finish()` with no arguments checks only that all recorded tool results were
consumed. `finish(actual_turns)` instead compares the supplied `list[Turn]` against
the recording by position: roles, number of tool calls, function names, and
arguments, including extra or missing turns. It does not compare message content,
tool results, IDs, timing, or tokens, and does not check stub consumption in this
form. Call both forms if you need both checks. Assert the returned `.ok` and use
`.errors` for diagnostics; `.run` on that result is still the original recording.
Use explicit contract assertions on the newly collected `AgentRun` for response
content or other requirements. Tool stubs alone do not replace LLM calls.
