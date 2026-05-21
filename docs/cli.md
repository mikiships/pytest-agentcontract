# CLI

The package installs the `agentcontract` command and a pytest plugin.

## `agentcontract init`

Create a starter `agentcontract.yml` in the current directory:

```bash
agentcontract init
```

The command fails if `agentcontract.yml` already exists. The generated file
includes scenario includes, replay defaults, one placeholder assertion, one
allowlist policy, budget defaults, and reporting defaults.

## `agentcontract info`

Print a cassette summary:

```bash
agentcontract info tests/scenarios/refund-eligible.agentrun.json
```

Output includes scenario, run ID, recorded timestamp, model, turn count, tool
call count, duration, token total, and estimated cost. The command returns exit
code `1` when the file does not exist or cannot be loaded.

## `agentcontract validate`

Validate that a cassette can be loaded:

```bash
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
```

This is a structural load check using `load_run()`. It prints the scenario and
turn count on success and returns exit code `1` for missing or invalid files.

## Pytest Options

`--ac-record`

Sets `ac_mode` to `record`. The `ac_recorder` fixture auto-saves to:

```text
tests/scenarios/<scenario>.agentrun.json
```

`--ac-replay`

Sets `ac_mode` to `replay`. The `ac_replay_engine` fixture loads the matching
cassette and returns a `ReplayEngine`. If the cassette is missing, the test is
skipped.

`--ac-config PATH`

Loads a specific `agentcontract.yml` instead of discovering one by walking up
from the current working directory.

`--ac-scenarios PATH`

Overrides the scenario directory used for pytest record/replay fixture paths.

## Pytest Markers

Use a marker to name the scenario:

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(...):
    ...
```

The alias `@pytest.mark.agent_scenario("refund-eligible")` is also supported.
If neither marker is present, the test node name is used as the scenario name.

## Pytest Fixtures

- `ac_config`: parsed `AgentContractConfig`.
- `ac_mode`: `record`, `replay`, or `live`.
- `ac_recorder`: `Recorder` that auto-saves in record mode.
- `ac_replay_engine`: `ReplayEngine` in replay mode, otherwise `None`.
- `ac_assert`: `AssertionEngine`.
- `ac_check_contract`: callable that checks a run against configured and extra
  assertions plus configured policies.
