# Show HN: pytest-agentcontract -- Deterministic CI tests for LLM agent trajectories

**URL:** https://github.com/mikiships/pytest-agentcontract

**Text:**

I built a pytest plugin that lets you record an LLM agent's trajectory (the sequence of tool calls, arguments, and results), then replay it deterministically in CI. No API keys, no network, no flakes.

The problem: agent tests are either flaky (live API calls, rate limits, nondeterminism) or meaningless (everything mocked away). You can't tell if a prompt change broke tool selection or argument formatting.

The approach: treat the tool-call sequence as the agent's contract. Record it once against real APIs, save it as a cassette (.agentrun.json), then replay it offline. Assert on what the agent decided to do, not what HTTP calls were made underneath.

```bash
pytest --ac-record    # Record once (hits real APIs)
pytest --ac-replay    # Replay in CI forever (deterministic, sub-second)
```

What you can assert:
- Tool call sequences (called_with, called_count, not_called)
- Response content (contains, exact, regex, json_schema)
- Policies (tool_allowlist, requires_confirmation)

Works with OpenAI, Anthropic, LangGraph, LlamaIndex, and the OpenAI Agents SDK. Auto-records via SDK interceptors or framework adapters -- one line to instrument your existing agent.

77 tests, MIT license, zero config to start (convention over configuration, but agentcontract.yml if you want it).

Would love feedback on the assertion model and whether the record/replay approach matches how you think about agent testing.
