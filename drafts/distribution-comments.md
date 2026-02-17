# Distribution Comments

## LangChain Issue #4629 (Testing/Evaluation)

**Context:** This is likely a discussion about testing LangChain agents.

**Comment:**

I built a pytest plugin for this: [pytest-agentcontract](https://github.com/mikiships/pytest-agentcontract).

The idea: record your agent's tool-call trajectory once against real APIs, save it as a cassette, then replay it deterministically in CI. No API keys, no network, no flakes.

It ships with a LangGraph adapter that wraps `invoke()`/`ainvoke()`:

```python
from agentcontract.adapters import record_graph

unpatch = record_graph(compiled_graph, recorder)
result = compiled_graph.invoke({"messages": [("user", "Refund order 123")]})
unpatch()
```

Then assert contracts: tool sequences, argument schemas, allowlist policies.

```bash
pytest --ac-record    # Once
pytest --ac-replay    # In CI forever
```

MIT, works with LangGraph/LlamaIndex/OpenAI. Feedback welcome: https://github.com/mikiships/pytest-agentcontract

---

## LlamaIndex Issue #20448 (Testing)

**Comment:**

Built something for this: [pytest-agentcontract](https://github.com/mikiships/pytest-agentcontract) -- a pytest plugin for deterministic agent testing.

Record a trajectory (tool calls, arguments, results) once, replay offline in CI. Comes with a LlamaIndex adapter:

```python
from agentcontract.adapters import record_agent

unpatch = record_agent(agent, recorder)
response = agent.chat("What's the refund policy?")
unpatch()
```

Works with ReAct agents, captures tool outputs from `.sources`, and retrieval results from `.source_nodes`.

MIT, 77 tests, zero config to start. Would appreciate any feedback on the approach.

---

## r/Python Post

**Title:** pytest-agentcontract: Deterministic CI tests for LLM agent trajectories (record once, replay offline)

**Body:**

I built a pytest plugin for testing LLM agents deterministically in CI.

**The problem:** Agent tests are either flaky (live API calls) or meaningless (everything mocked). You can't tell if a prompt change broke tool selection.

**The solution:** Record the agent's trajectory (tool calls, arguments, results) once against real APIs. Replay it offline in CI. Assert contracts on what the agent decided to do.

```bash
pip install pytest-agentcontract
pytest --ac-record    # Record once
pytest --ac-replay    # CI forever -- no network, no keys, sub-second
```

Features:
- 7 assertion types (contains, exact, regex, json_schema, called_with, called_count, not_called)
- Policy enforcement (tool allowlists, confirmation gates)
- Auto-recording via SDK interceptors (OpenAI, Anthropic)
- Framework adapters (LangGraph, LlamaIndex, OpenAI Agents SDK)
- 77 tests, MIT license

GitHub: https://github.com/mikiships/pytest-agentcontract
PyPI: https://pypi.org/project/pytest-agentcontract/

Feedback welcome -- especially on the assertion model and whether record/replay is the right abstraction for agent testing.

---

## r/MachineLearning Post

**Title:** [P] pytest-agentcontract: Record-replay testing for LLM agent tool-call trajectories

**Body:**

Sharing a pytest plugin I built for deterministic testing of LLM agents.

Most agent "tests" either hit live APIs (flaky, expensive, nondeterministic) or mock everything (meaningless). This plugin records the full trajectory -- every tool call, argument, and result -- then replays it offline in CI.

The key insight: the tool-call sequence IS the agent's contract. If your refund agent calls `lookup_order` → `check_eligibility` → `process_refund` with the right arguments, the contract holds. Test that, not the HTTP layer.

Works with OpenAI, Anthropic, LangGraph, LlamaIndex, and the OpenAI Agents SDK via one-line interceptors.

GitHub: https://github.com/mikiships/pytest-agentcontract

Interested in feedback from anyone doing agent evaluation or CI for LLM systems.
