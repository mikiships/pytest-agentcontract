# pytest integration

Installing `pytest-agentcontract` registers its pytest plugin automatically.
Fixtures are opt-in: a marker or mode flag alone does not run an agent, record
calls, or check a contract. See the [README](../README.md) for installation.

## Complete offline example

Save the following as `test_refund.py` in a directory without an inherited
`agentcontract.yml`. It uses only the package and its standard dependencies.
The turn content is deterministic; recording metadata such as IDs and timestamps
will differ between recordings.

```python
import pytest

from agentcontract.config import AssertionSpec


@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay":
        run = ac_replay_engine.recorded_run
    else:
        # Deterministic stand-in for your agent; no SDK or service required.
        ac_recorder.add_turn(
            role="assistant", content="Your refund of $49.99 has been processed."
        )
        run = ac_recorder.run

    contract = ac_check_contract(
        run,
        extra_assertions=[
            AssertionSpec(type="contains", target="final_response", value="$49.99")
        ],
    )
    assert contract.passed, [failure.message for failure in contract.failures()]
```

Run from that directory:

```bash
pytest test_refund.py --ac-record --ac-scenarios tests/scenarios
pytest test_refund.py --ac-replay --ac-scenarios tests/scenarios
pytest test_refund.py                         # live: no cassette is saved
pytest test_refund.py --ac-replay --ac-scenarios missing-cassettes -rs
```

The first command creates `tests/scenarios/refund-eligible.agentrun.json` at
fixture teardown. The second checks its stored response without external
services. The last command skips the test with `No cassette found at ...`,
provided `missing-cassettes` has no matching cassette.

## Fixtures

All six fixtures have function scope and are created when requested by a test
or another fixture. They are not autouse.

| Fixture | Value supplied to the test | Behavior |
| --- | --- | --- |
| `ac_config` | `AgentContractConfig` | Loads an explicit config or discovers one; returns defaults if discovery finds none. |
| `ac_mode` | `str` | Returns `"record"`, `"replay"`, or `"live"`. |
| `ac_recorder` | `Recorder` | Yields a fresh recorder inside its `recording()` context in every mode. Access its `AgentRun` via `.run` and append turns with `.add_turn(...)`. |
| `ac_replay_engine` | `ReplayEngine` or `None` | Loads the matching cassette when `--ac-replay` is set; otherwise returns `None`. A missing cassette skips the test instead of returning `None`. |
| `ac_assert` | `AssertionEngine` | Creates a fresh engine; depends on `ac_config` but does not apply its assertions automatically. Call `.check(run, assertions=..., policies=...)` directly. |
| `ac_check_contract` | Callable `(run: AgentRun, extra_assertions: list[AssertionSpec] or None = None) -> ContractResult` | Combines configured and extra assertions, then checks them and configured policies using `ac_assert`. |

The types come from `agentcontract.config` (`AgentContractConfig`, `AssertionSpec`),
`agentcontract.recorder.core` (`Recorder`), `agentcontract.replay.engine`
(`ReplayEngine`), `agentcontract.types` (`AgentRun`), and
`agentcontract.assertions.engine` (`AssertionEngine`, `ContractResult`).

At teardown, `ac_recorder` finalizes duration, token, turn, and tool-call summary
fields. If `--ac-record` is set, it saves the cassette, creating parent directories
and replacing an existing cassette at that path. Saving also occurs after a test
assertion fails; this is not a save-only-on-success mechanism. Other modes do not
auto-save. During the test, `.run` is available but its summary is not finalized.
Save errors of type `OSError`, `ValueError`, or `TypeError` become pytest teardown
failures with `Failed to save cassette ...`.

`ac_check_contract` appends assertions in this order:

1. `defaults.assertions` from configuration.
2. `overrides[run.metadata.scenario].assertions`, if present.
3. The call's `extra_assertions`, if supplied.

Scenario assertions append to defaults; they do not replace them. The scenario
lookup uses the supplied run's metadata, including when it came from a cassette.
Configured policies run after these assertions. Neither check helper fails the
test automatically: assert `result.passed` and use `result.failures()` (a list of
`AssertionResult` objects) for diagnostics, as above. With no assertions or policies,
the empty result passes. Unsupported assertion/policy types and checker errors
are reported as failed results.

## Options and modes

| Pytest option | Default | Implemented effect |
| --- | --- | --- |
| `--ac-record` | Off | Makes `ac_mode` return `"record"` and enables recorder auto-save. It does not itself require or call a live API. |
| `--ac-replay` | Off | Makes `ac_mode` return `"replay"` unless record is also set; enables cassette loading by `ac_replay_engine`. |
| `--ac-config PATH` | Unset | Loads that YAML file through `ac_config` instead of discovery. Relative paths use the current working directory. |
| `--ac-scenarios DIR` | `tests/scenarios` | Selects the directory used by recorder and replay fixtures. Relative paths use the current working directory. |

With neither mode flag, `ac_mode` is `"live"`: the recorder remains usable in
memory, the replay fixture returns `None`, and no cassette is automatically saved.
Your test controls agent execution in every mode. For offline replay, branch on
the mode as in the example or explicitly wire in stubs; replay is not a network
sandbox or an SDK interceptor.

Supply only one mode flag. If both are supplied, `ac_mode` gives record precedence,
but `ac_replay_engine` independently sees the replay flag and still tries to load
a cassette. It can skip or fail during setup before the test runs, while an
initialized recorder still auto-saves at teardown because the record flag is set.

When replay is requested, a nonexistent cassette causes `pytest.skip`. An existing
cassette that cannot be read or parsed is not treated as missing: load errors of
type `OSError`, `ValueError`, or `TypeError` cause `pytest.fail` with
`Failed to load cassette ...` during fixture setup (reported as a pytest error).
Malformed JSON is one example. Loading validates supported serialization structure;
it does not itself check your assertions or policies.

## Markers and cassette names

Both `@pytest.mark.agentcontract("refund-eligible")` and
`@pytest.mark.agent_scenario("refund-eligible")` are supported. Either also accepts
`name="refund-eligible"`. The fixtures select the closest `agentcontract` marker
if present, otherwise the closest `agent_scenario` marker. The first positional
argument wins over `name`; without positional arguments or a truthy `name`, they
fall back to `request.node.name` (including parametrization suffixes).
Other marker keyword arguments are not used by these fixtures.

Both fixtures use `<directory>/<scenario>.agentrun.json`. Use distinct scenario
names to avoid tests overwriting each other's cassettes. Despite the option help
text saying “overrides config”, the fixture directory comes only from
`--ac-scenarios` or the hardcoded `tests/scenarios` default. Configuration
`scenarios.include` and `scenarios.exclude` do not select fixture cassettes.

## Configuration discovery

Without `--ac-config`, `ac_config` searches the current working directory and
then each parent for the first `agentcontract.yml`. It does not start from the
test file or pytest root directory and does not search for `agentcontract.yaml`.
If none exists, it returns default configuration, with no assertions or policies.
An explicit missing/unreadable config or invalid YAML surfaces as a fixture setup
error rather than falling back to defaults.

The pytest fixtures use configuration assertions, scenario overrides, and policies
through `ac_check_contract`. Replay configuration such as `stub_tools`, model,
seed, and concurrency is not passed to `ReplayEngine` by these fixtures and does
not arrange agent execution or interception.

## Explicit replay integration

`ac_replay_engine.recorded_run` is the loaded `AgentRun`. Checking it, as above,
validates a stored trajectory; it does not test a new execution of your agent.
There is no `ac_replay_engine.run()` method.

To exercise your agent's tool dispatch, explicitly replace real tool execution
with `engine.tool_stub.get_result(function, arguments)`. For example, inside your
own replay-mode dispatch code, using a cassette that contains `lookup_order`:

```python
result = ac_replay_engine.tool_stub.get_result("lookup_order", {"order_id": "123"})
# Return result to your agent instead of calling the real lookup_order service.
```

`get_result` returns the stored result (`Any`) in recording order **per function**;
it does not enforce global ordering between different tools. Supplying arguments
requires exact equality with the next recorded argument dictionary. Omitting them
(or passing `None`) bypasses that check. A mismatch raises
`ToolStubArgumentsMismatch` without consuming the result; an unknown or exhausted
tool raises `ToolStubExhausted`. Both exceptions are in
`agentcontract.replay.engine`.

After your loop, call `finish()` and assert its `ReplayResult.ok`:

```python
replay_result = ac_replay_engine.finish()
assert replay_result.ok, replay_result.errors
```

With no argument, `finish()` checks that all recorded tool results were consumed.
Alternatively, `finish(actual_turns)` accepts a `list[Turn]` from your new execution
and compares turn counts, roles, tool counts, function names, and arguments. It
does not compare response text or tool result values, and this form does not also
check stub consumption. Use both calls if both checks matter, and use contract
assertions on the newly recorded run for response content. `finish` returns a
result rather than automatically failing pytest; its `.run` remains the original
recorded run. The fixture does not call `finish` on teardown. Any LLM calls in your
agent still need your own offline replacement to keep this integration offline.
