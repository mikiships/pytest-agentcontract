# pytest-agentcontract Documentation

pytest-agentcontract records LLM agent trajectories into `.agentrun.json`
cassettes, replays them offline, and checks user-defined contracts over the
recorded turns and tool calls.

## Guides

- [Getting started](getting-started.md): installation, pytest markers, fixtures,
  record/replay commands, and a compact customer-support example.
- [Record and replay](record-replay.md): cassette lifecycle, file naming,
  replay stubs, CI usage, and validation.
- [Assertions and policies](assertions-and-policies.md): supported assertion
  types, target syntax, `called_with`, `called_count`, JSON Schema, and policy
  checks.
- [Configuration](configuration.md): `agentcontract.yml` fields parsed by
  `AgentContractConfig`.
- [Adapters and interceptors](adapters.md): OpenAI and Anthropic SDK
  interceptors plus LangGraph, LlamaIndex, and OpenAI Agents SDK adapters.
- [CLI](cli.md): `agentcontract` commands and pytest plugin options.

## Demo Assets

- [demo.gif](demo.gif): visual walkthrough of record, replay, and assert.
- [demo.cast](demo.cast): terminal recording source for the demo.

## Public API Entry Points

The top-level package lazy-loads these public classes:

```python
from agentcontract import AgentContractConfig, AssertionEngine, Recorder, ReplayEngine
```

Most pytest tests use the plugin fixtures instead:

- `ac_config`
- `ac_mode`
- `ac_recorder`
- `ac_replay_engine`
- `ac_assert`
- `ac_check_contract`
