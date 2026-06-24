# Record and replay

pytest-agentcontract adds pytest markers, fixtures, and CLI options for recording
agent trajectories once and replaying them deterministically later.

## Mark a scenario

Use `agentcontract` to name the cassette for a test:

```python
import pytest


@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay" and ac_replay_engine is not None:
        run = ac_replay_engine.recorded_run
    else:
        run_my_agent(ac_recorder)
        run = ac_recorder.run

    result = ac_check_contract(run)
    assert result.passed, [failure.message for failure in result.failures()]
```

`@pytest.mark.agent_scenario(name="refund-eligible")` is also registered as an
alias. If neither marker is present, the pytest node name is used as the scenario
name.

## Fixtures

- `ac_mode` returns `"record"`, `"replay"`, or `"live"`.
- `ac_recorder` provides a `Recorder` and saves its run automatically after the
  test when `--ac-record` is active.
- `ac_replay_engine` returns a `ReplayEngine` only when `--ac-replay` is active.
  Outside replay mode it returns `None`.
- `ac_config` loads `agentcontract.yml` with discovery, or the file passed to
  `--ac-config`.
- `ac_assert` provides an `AssertionEngine`.
- `ac_check_contract(run, extra_assertions=None)` checks default assertions,
  scenario override assertions, optional extra assertions, and configured
  policies.

## Pytest options

```bash
pytest --ac-record
pytest --ac-replay
pytest --ac-config path/to/agentcontract.yml
pytest --ac-scenarios path/to/scenarios
```

`--ac-record` and `--ac-replay` are independent boolean flags. In normal live
mode, neither flag is set and `ac_mode` returns `"live"`.

## Cassette naming and location

Record mode writes one cassette per scenario:

```text
<scenarios-dir>/<scenario>.agentrun.json
```

The scenario comes from `@pytest.mark.agentcontract("...")`, then
`@pytest.mark.agent_scenario(name="...")`, then the pytest node name. The
scenario directory defaults to `tests/scenarios` and can be overridden with
`--ac-scenarios`.

Example:

```bash
pytest tests/test_refunds.py --ac-record --ac-scenarios tests/scenarios
# writes tests/scenarios/refund-eligible.agentrun.json
```

Replay mode loads the same path. If the file does not exist, the fixture skips
the test with `No cassette found at ...`.

## Replay patterns

For contract-only tests, use the recorded run directly:

```python
if ac_mode == "replay" and ac_replay_engine is not None:
    run = ac_replay_engine.recorded_run
else:
    run_my_agent(ac_recorder)
    run = ac_recorder.run
```

For tests that execute agent code during replay, replace real tool execution
with `ReplayEngine.tool_stub`:

```python
stub = ac_replay_engine.tool_stub
order = stub.get_result("lookup_order", {"order_id": "ORD-123"})
eligibility = stub.get_result("check_refund_eligibility", {"order_id": "ORD-123"})

result = ac_replay_engine.finish(actual_turns)
assert result.ok, result.errors
```

`ToolStub.get_result()` returns recorded tool results in the order they were
recorded. Passing arguments is optional, but when arguments are provided they
must exactly match the next recorded call for that tool.

## CI usage

Commit `.agentrun.json` cassettes and run replay in CI:

```bash
pytest --ac-replay
```

Replay does not require live LLM credentials when your tests use
`recorded_run`, or when your agent under test is wired to `tool_stub` instead of
real tools.

## Common failure modes

- Missing cassette: `ac_replay_engine` skips the test when the expected file is
  absent.
- Bad cassette JSON or invalid turn roles: loading fails the test with the path
  and exception type.
- Replay argument mismatch: `tool_stub.get_result("tool", args)` raises
  `ToolStubArgumentsMismatch` when `args` do not equal the next recorded
  arguments for that tool.
- Tool stub exhausted: `ToolStubExhausted` is raised when replay requests more
  calls than were recorded.
- Missing or extra turns: `ReplayEngine.finish(actual_turns)` reports
  `missing_tools`, `extra_tools`, `mismatched_tools`, and human-readable
  `errors`.
