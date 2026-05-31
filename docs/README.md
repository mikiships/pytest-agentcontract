# pytest-agentcontract Documentation

pytest-agentcontract records LLM agent trajectories into `.agentrun.json` cassettes, replays them without live provider calls, and checks behavior with assertions and policies.

## Guides

- [Getting started](getting-started.md): install the package, mark pytest scenarios, record/replay cassettes, and write a compact end-to-end test.
- [Record and replay](record-replay.md): understand cassette lifecycle, record mode, replay mode, CI usage, and validation.
- [Assertions and policies](assertions-and-policies.md): configure contract assertions, target syntax, tool argument checks, and policy checks.
- [Configuration](configuration.md): define `agentcontract.yml` with scenarios, replay settings, defaults, overrides, policies, budgets, baseline, and reporting fields.
- [Adapters and interceptors](adapters.md): capture calls from OpenAI, Anthropic, LangGraph, LlamaIndex, and the OpenAI Agents SDK.
- [CLI reference](cli.md): use `agentcontract init`, `info`, `validate`, and the pytest options.

## Demo Assets

- [demo.gif](demo.gif)
- [demo.cast](demo.cast)

## Example Project

The runnable customer support example lives in [`examples/customer_support`](../examples/customer_support). It demonstrates the `refund-eligible`, `refund-not-delivered`, and `refund-with-policies` scenarios with committed cassettes under `examples/customer_support/scenarios/`.
