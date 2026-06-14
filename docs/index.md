# Documentation

The README covers the basic record, replay, and assert workflow. These pages cover the public surfaces that matter once you start wiring the plugin into real tests.

## Start Here

- [Pytest plugin reference](pytest-plugin.md): markers, fixtures, cassette naming, and the pytest CLI flags.
- [Configuration reference](configuration.md): every key parsed from `agentcontract.yml`, including which fields are enforced today and which are only stored in `ac_config`.
- [Cassette format and CLI reference](cassette-format.md): the on-disk `.agentrun.json` schema, how replay reads it, and what the `agentcontract` CLI commands do.

## Workflow

1. Write a test that records turns into `ac_recorder` or uses one of the SDK/framework interceptors.
2. Record a cassette with `pytest --ac-record`.
3. Replay by loading the cassette through `ac_replay_engine` during `pytest --ac-replay`.
4. Check the resulting `AgentRun` with `ac_check_contract` or call `AssertionEngine` directly.

`pytest --ac-replay` does not automatically run your agent against recorded tool results. The plugin loads the matching cassette into `ReplayEngine`; your test can either assert against `ac_replay_engine.recorded_run` or drive its own replay loop with `ac_replay_engine.tool_stub` and `ac_replay_engine.finish(...)`.
