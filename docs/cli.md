# CLI Reference

pytest-agentcontract exposes the `agentcontract` command and pytest options.

## `agentcontract init`

Create a starter `agentcontract.yml` in the current directory:

```bash
agentcontract init
```

The command exits with an error if `agentcontract.yml` already exists.

## `agentcontract info`

Print a cassette summary:

```bash
agentcontract info tests/scenarios/refund-eligible.agentrun.json
```

Output includes the scenario, run ID, recorded timestamp, model provider/name, turn count, tool call count, duration, tokens, and estimated cost.

## `agentcontract validate`

Load and validate cassette structure:

```bash
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
```

The command succeeds when the cassette can be deserialized into an `AgentRun`. It reports an error for missing files or invalid cassette structure.

## Pytest Options

### `--ac-record`

Record agent trajectories and auto-save cassettes after tests using `ac_recorder` finish:

```bash
pytest --ac-record
```

### `--ac-replay`

Load matching cassettes and expose a `ReplayEngine` through `ac_replay_engine`:

```bash
pytest --ac-replay
```

If no cassette exists for a replayed scenario, the fixture skips that test.

### `--ac-config`

Use a specific config file instead of discovery:

```bash
pytest --ac-config path/to/agentcontract.yml
```

Without this option, `AgentContractConfig.discover()` walks up from the current working directory looking for `agentcontract.yml`.

### `--ac-scenarios`

Override the cassette directory used by pytest fixtures:

```bash
pytest --ac-record --ac-scenarios examples/customer_support/scenarios
pytest --ac-replay --ac-scenarios examples/customer_support/scenarios
```

The fixture writes or reads `<scenario>.agentrun.json` inside that directory. Scenario names come from `@pytest.mark.agentcontract("name")`, `@pytest.mark.agent_scenario("name")`, or the pytest node name when no marker is present.
