# Documentation

`pytest-agentcontract` ships with a small set of reference docs for the core workflow:

1. Write a pytest contract test with the plugin fixtures and markers in [Pytest Plugin Reference](pytest-plugin.md).
2. Configure shared assertions, policies, budgets, and reporting in [Configuration Reference](configuration.md).
3. Record and replay `.agentrun.json` cassettes, then inspect them with the CLI in [Cassette Format and CLI Reference](cassette-format.md).

## User Journey

### Writing tests

Start with a normal pytest test and mark it with `@pytest.mark.agentcontract("scenario-name")`. The plugin gives you fixtures for recording runs, loading config, replaying recorded tool results, and checking contracts.

- [Pytest Plugin Reference](pytest-plugin.md)

### Configuring contracts

Place `agentcontract.yml` in your project root to define default assertions, per-scenario overrides, policies, budgets, baseline settings, and reporting output.

- [Configuration Reference](configuration.md)

### Recording and replaying

Use `pytest --ac-record` to save a cassette under `tests/scenarios/` by default, then use `pytest --ac-replay` to run against recorded tool results without live provider access.

- [Pytest Plugin Reference](pytest-plugin.md#record-and-replay-workflow)
- [Cassette Format and CLI Reference](cassette-format.md)

### Inspecting runs

Each cassette is a serialized `AgentRun`. The `agentcontract info` and `agentcontract validate` commands read that file directly, and `agentcontract init` writes a starter `agentcontract.yml`.

- [Cassette Format and CLI Reference](cassette-format.md#cli-commands)
