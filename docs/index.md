# pytest-agentcontract documentation

pytest-agentcontract records LLM agent trajectories as `.agentrun.json` cassettes,
replays them in pytest, and checks the recorded behavior against assertions and
policies.

## Start here

1. Install the package:

   ```bash
   pip install pytest-agentcontract
   ```

2. Add an `@pytest.mark.agentcontract("scenario-name")` test and run it once
   with `--ac-record`.
3. Commit the generated cassette and run `pytest --ac-replay` in CI.
4. Add config defaults, per-scenario overrides, or inline `AssertionSpec` checks
   as your contract grows.

## Guides

- [Record and replay](record-replay.md) covers pytest markers, fixtures, CLI
  options, cassette naming, CI usage, and replay failure modes.
- [Configuration](configuration.md) documents the supported `agentcontract.yml`
  schema and parser defaults.
- [Assertions and policies](assertions-and-policies.md) documents assertion
  types, target syntax, policy behavior, and YAML/Python examples.
- [Adapters and interceptors](adapters-and-interceptors.md) covers manual
  recording, OpenAI and Anthropic interceptors, and framework adapters.
- [Customer support example](../examples/customer_support/README.md) explains
  the bundled deterministic example and the exact commands to record and replay
  it.

## CLI quick reference

```bash
agentcontract init
agentcontract info tests/scenarios/refund-eligible.agentrun.json
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
pytest --ac-record
pytest --ac-replay
```

The pytest plugin is registered through the `agentcontract` pytest entry point.
The standalone `agentcontract` command exposes cassette inspection and starter
config helpers.
