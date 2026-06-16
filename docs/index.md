# pytest-agentcontract Documentation

pytest-agentcontract records LLM agent trajectories as `.agentrun.json` cassettes, replays them deterministically in pytest, and checks user-defined contracts over the recorded turns and tool calls.

Use these guides when moving from the README demo into a real test suite:

- [Quickstart](quickstart.md): install the package, write a small pytest contract, record a cassette, and replay it.
- [Recording and replay](recording-and-replay.md): use the pytest fixtures, `Recorder`, `ReplayEngine.tool_stub`, and replay lifecycle.
- [Configuration](configuration.md): document `agentcontract.yml` sections parsed by `AgentContractConfig`.
- [Assertions and policies](assertions-and-policies.md): define content assertions, tool-call assertions, target syntax, and policies.
- [Cassette format](cassette-format.md): understand the `.agentrun.json` structure saved by the serializer.
- [SDK and framework adapters](sdk-and-framework-adapters.md): auto-record OpenAI, Anthropic, LangGraph, LlamaIndex, and OpenAI Agents SDK runs.
- [CLI and CI](cli-and-ci.md): use `agentcontract init`, `info`, `validate`, and common local/CI pytest commands.

## Core Workflows

**Record:** run pytest with `--ac-record` and execute your live agent. The `ac_recorder` fixture saves `tests/scenarios/<scenario>.agentrun.json` unless you override the scenario directory.

**Replay:** run pytest with `--ac-replay`. The `ac_replay_engine` fixture loads the matching cassette and exposes the recorded run and a tool stub for deterministic replay.

**Assert:** call `ac_check_contract(run)` to apply default config assertions, scenario overrides, and policies. You can pass `extra_assertions` for test-specific checks.

**CI:** commit cassettes with the tests that own them, then run replay mode in CI so contract checks do not require LLM credentials, network access, or token spend.
