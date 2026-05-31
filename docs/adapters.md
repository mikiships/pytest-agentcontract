# Adapters And Interceptors

pytest-agentcontract can record trajectories manually through `Recorder.add_turn()` or automatically by wrapping SDK and framework clients. Every wrapper returns an `unpatch()` callable that restores the original method.

Use wrappers inside the lifetime of a `Recorder.recording()` context or the `ac_recorder` fixture.

## OpenAI SDK Interceptor

Install the OpenAI extra:

```bash
pip install pytest-agentcontract[openai]
```

Patch an OpenAI client instance:

```python
import openai

from agentcontract.recorder.interceptors import patch_openai


def test_openai_agent(ac_recorder):
    client = openai.OpenAI()
    unpatch = patch_openai(client, ac_recorder)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Refund order ORD-123"}],
            tools=[...],
        )
    finally:
        unpatch()
```

The interceptor wraps `client.chat.completions.create`. It records assistant content, tool call names, parsed tool arguments, latency, usage tokens, and model metadata when present. Tool results are not present in OpenAI chat completion responses; add tool-result turns manually or backfill results from your application layer when you need replay stubs.

Async `create` calls are supported when the wrapped method returns an awaitable. Unsupported response shapes, including streaming objects without replayable choices, are skipped.

## Anthropic SDK Interceptor

Install the Anthropic extra:

```bash
pip install pytest-agentcontract[anthropic]
```

Patch an Anthropic client instance:

```python
from agentcontract.recorder.interceptors import patch_anthropic


def test_anthropic_agent(ac_recorder, anthropic_client):
    unpatch = patch_anthropic(anthropic_client, ac_recorder)
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Refund order ORD-123"}],
            tools=[...],
        )
    finally:
        unpatch()
```

The interceptor wraps `client.messages.create`. It records text blocks, `tool_use` blocks, input/output token counts, latency, and model metadata. Anthropic tool results arrive in later `tool_result` messages from your application code, so record those manually when replay needs them.

## LangGraph Adapter

Import from the adapter package:

```python
from agentcontract.adapters import record_graph


def test_langgraph(ac_recorder, compiled_graph):
    unpatch = record_graph(compiled_graph, ac_recorder)
    try:
        result = compiled_graph.invoke({"messages": [("user", "I need a refund")]})
    finally:
        unpatch()
```

`record_graph(graph, recorder)` wraps `invoke()` and `ainvoke()` when present. It expects a LangGraph state dictionary with a `messages` list and records LangChain-style message roles, content, tool calls, and assistant latency.

## LlamaIndex Adapter

```python
from agentcontract.adapters import record_agent


def test_llamaindex(ac_recorder, agent):
    unpatch = record_agent(agent, ac_recorder)
    try:
        response = agent.chat("What's the refund policy?")
    finally:
        unpatch()
```

`record_agent(agent, recorder)` wraps any available `chat`, `achat`, `query`, and `aquery` methods. It records assistant response text, tool outputs from `response.sources`, and retrieval entries from `response.source_nodes` as `_retrieve` tool calls.

## OpenAI Agents SDK Adapter

Install the OpenAI Agents SDK in your project:

```bash
pip install openai-agents
```

Patch the SDK `Runner` class:

```python
from agents import Runner
from agentcontract.adapters import record_runner


def test_agents_sdk(ac_recorder, agent):
    unpatch = record_runner(ac_recorder)
    try:
        result = Runner.run_sync(agent, "Help with billing")
    finally:
        unpatch()
```

`record_runner(recorder)` imports `agents.Runner` and patches class-level `run()` and `run_sync()` methods when present. It extracts turns from `RunResult.new_items` when available and falls back to `final_output`.

Because this patch is class-level, always call `unpatch()` in a `finally` block to avoid affecting later tests.

## Patch Hygiene

- Keep patches scoped to a single test or fixture.
- Use `try/finally` so `unpatch()` runs after failures.
- Avoid nesting multiple wrappers around the same method unless you control the order.
- Record tool results manually when an SDK response only includes tool requests.
