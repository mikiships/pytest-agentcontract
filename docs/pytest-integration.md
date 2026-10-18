# pytest integration

Installing `pytest-agentcontract` registers the plugin automatically with pytest.
Tests opt into its behavior by requesting fixtures; markers alone do not record,
replay, or check contracts. See the [README](../README.md) for installation.

## Options and modes

| Option | Default | Behavior |
| --- | --- | --- |
| `--ac-record` | Off | Makes `ac_mode` return `"record"` and saves requested recorders during teardown. |
| `--ac-replay` | Off | Makes `ac_mode` return `"replay"` unless recording is also enabled; requested replay engines load cassettes. |
| `--ac-config PATH` | Unset | Loads this YAML file instead of discovering `agentcontract.yml`. |
| `--ac-scenarios DIR` | `tests/scenarios` | Directory used for both saving and loading cassettes. |

Without either mode flag, `ac_mode` is `"live"`: the recorder is available in
memory but is not automatically saved, and `ac_replay_engine` is `None`.
Your test decides whether to execute an agent; mode flags do not intercept SDK
calls or prevent network access in arbitrary test code.

If both flags are supplied, `ac_mode` returns `"record"` and the recorder saves.
However, `ac_replay_engine` independently checks `--ac-replay`, so requesting it
still loads a cassette (or skips if missing), even before the test body runs.
Use one mode flag at a time.

## Markers, names, and paths

Both `@pytest.mark.agentcontract("refund-eligible")` and
`@pytest.mark.agent_scenario("refund-eligible")` name a scenario. Both also accept
`name="refund-eligible"`. Other keyword arguments have no effect on these fixtures.

The recorder and replay fixtures choose a name using this precedence:

1. Select the closest `agentcontract` marker; only if absent, select the closest
   `agent_scenario` marker. Thus `agentcontract` wins when both types are present.
2. Use the selected marker's first positional argument, if present.
3. Otherwise use its truthy `name` keyword argument.
4. Otherwise use `request.node.name`, including parameter IDs for parametrized tests.

The path is `<scenarios directory>/<scenario>.agentrun.json`. Relative paths,
including `--ac-config` and `--ac-scenarios`, are relative to the process working
directory, not the config file's directory or necessarily pytest's root directory.
Names are used directly as paths; choose distinct, filesystem-friendly names.
Tests sharing a name share a cassette, and recording overwrites an existing file.
Parent directories are created when saving.

## Fixtures

All six fixtures have function scope and are created only when requested,
including through fixture dependencies.

| Fixture | Return value and lifecycle |
| --- | --- |
| `ac_config` | An `AgentContractConfig`, loaded anew for the test from an explicit path or discovery. |
| `ac_mode` | The string `"live"`, `"record"`, or `"replay"`. |
| `ac_recorder` | A fresh `Recorder`, with the selected scenario in `recorder.run.metadata.scenario`. Its recording context surrounds the test; `recorder.run` is the mutable `AgentRun`. Teardown finalizes timing, token, and turn summaries, then saves only if `--ac-record` is set. |
| `ac_replay_engine` | A fresh `ReplayEngine` holding the loaded cassette when `--ac-replay` is set; otherwise `None`. `engine.recorded_run` is the loaded `AgentRun`. There is no automatic replay execution or teardown comparison. |
| `ac_assert` | An `AssertionEngine`. It depends on `ac_config`, but `check(run, assertions=..., policies=...)` uses only the specifications you pass explicitly. |
| `ac_check_contract` | A callable `check(run, extra_assertions=None)` returning a `ContractResult`; it combines configured assertions and policies as described below. Depends on `ac_config` and `ac_assert`. |

Requesting `ac_replay_engine` with a missing cassette skips the test with
`No cassette found at ...`; it does not return `None` in this case. Loading errors
of type `OSError`, `ValueError`, or `TypeError` fail fixture setup with the path and
error type. Examples include malformed JSON and invalid turn roles. Loading is
not full JSON Schema validation: some absent or malformed fields are normalized.

The recorder saves in normal pytest teardown even if the test assertion fails.
Save errors of type `OSError`, `ValueError`, or `TypeError` fail teardown with the
path and error type. During the test, turns are available immediately, but the
recording context's final summary is not populated until teardown. A recorder
requested in replay mode is still a fresh recorder; it is not the loaded cassette.

`ReplayEngine` also offers `tool_stub.get_result(function, arguments)` for callers
that explicitly wire recorded tool results into their own agent loop, and
`finish(actual_turns)` for comparing turns. Calling `finish()` without turns checks
stub consumption. These methods return results or raise errors as appropriate;
the fixture does not call them for you. The example below checks the stored
trajectory only, without rerunning an agent or consuming stubs.

## Configuration and contract composition

Without `--ac-config`, discovery starts at the current working directory and
walks upward, loading the first `agentcontract.yml`. No file means an
`AgentContractConfig` with defaults, including empty assertions and policies.
An explicit missing or unreadable config file raises an error; it does not fall
back to discovery. Config loading is triggered by `ac_config`, `ac_assert`, or
`ac_check_contract`, not by the mode, recorder, or replay fixture alone.

For each call, `ac_check_contract` evaluates:

1. `defaults.assertions`.
2. Assertions from `overrides[run.metadata.scenario].assertions`, if present.
   Despite the name, overrides append to defaults rather than replace them.
3. The supplied `extra_assertions` (a list of `AssertionSpec` objects).
4. All configured `policies`.

The scenario key comes from the supplied run's metadata, including cassette
metadata during replay, rather than directly from the test marker. Every result
must pass for `ContractResult.passed` to be true; an empty set passes. The caller
must use `assert result.passed, result.failures()` to fail the pytest test.
Policies check the supplied trajectory; they do not block live tool execution.

These assertions and policies are the configuration settings consumed by the
pytest contract helper. Although parsed and exposed through `ac_config`,
`scenarios.include` / `exclude` do not collect tests or select cassette paths.
The plugin uses `--ac-scenarios` or `tests/scenarios` for paths. Likewise, `replay`
(model, seed, stub_tools, concurrency), `thresholds`, `budgets`, `baseline`, and
`reporting` are not automatically applied or enforced by these pytest fixtures.

## Manual recording example

This example needs no external SDK, credentials, or network access after package
installation. In an empty directory, install `pytest-agentcontract` and save the
following as `agentcontract.yml`:

```yaml
version: "1"
defaults:
  assertions:
    - type: contains
      target: final_response
      value: "refund"
overrides:
  refund-eligible:
    assertions:
      - type: called_with
        target: tool:process_refund
        schema:
          order_id: "123"
policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [process_refund]
```

Save this as `test_refund.py`:

```python
import pytest

from agentcontract.config import AssertionSpec


@pytest.mark.agentcontract("refund-eligible")
def test_refund(ac_mode, ac_recorder, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay":
        assert ac_replay_engine is not None
        run = ac_replay_engine.recorded_run
    else:  # record or live: manually construct a trajectory
        ac_recorder.add_turn(role="user", content="Refund order 123")
        ac_recorder.add_turn(
            role="assistant",
            tool_calls=[{
                "id": "refund-1",
                "function": "process_refund",
                "arguments": {"order_id": "123"},
                "result": {"refunded": True},
            }],
        )
        ac_recorder.add_turn(role="assistant", content="Your refund is complete.")
        run = ac_recorder.run

    result = ac_check_contract(
        run,
        extra_assertions=[
            AssertionSpec(type="called_count", target="tool:process_refund", value="1")
        ],
    )
    assert result.passed, result.failures()
```

Run from that directory:

```bash
python -m pytest -q --ac-record test_refund.py
# Saves tests/scenarios/refund-eligible.agentrun.json during teardown.
python -m pytest -q --ac-replay test_refund.py
# Checks the loaded trajectory against the same contract, offline.
python -m pytest -q test_refund.py
# Live mode: constructs and checks the trajectory without saving.
```

To confirm failure reporting, change the extra assertion's `value="1"` to
`value="2"` and rerun the replay command: the contract assertion fails. Restore
`value="1"` afterward. Replaying before recording (or with `--ac-scenarios` pointing
to an empty directory) skips the test because there is no matching cassette.
