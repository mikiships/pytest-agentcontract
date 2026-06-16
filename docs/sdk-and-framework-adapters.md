# SDK And Framework Adapters

Adapters wrap SDK or framework methods and add turns to a `Recorder`. Each adapter returns an `unpatch()` function; call it after the agent run to restore the original methods.

## OpenAI SDK

Install the extra when your project uses the OpenAI SDK:

```bash
pip install "pytest-agentcontract[openai]"
```

```python
from agentcontract.recorder.interceptors import patch_openai

unpatch = patch_openai(client, recorder)
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Refund order ORD-123"}],
        tools=[...],
    )
finally:
    unpatch()
```

`patch_openai` wraps `client.chat.completions.create`. It supports sync calls and awaitable responses. It records the first choice message as an assistant turn, including content, tool-call requests, latency, prompt tokens, and completion tokens when available.

Current limitation: OpenAI chat completion responses contain tool-call requests, not your application tool results. The interceptor records function name and parsed arguments, but it cannot record tool results unless your application adds them manually with `recorder.add_turn()` or you backfill the cassette.

## Anthropic SDK

Install the extra when your project uses the Anthropic SDK:

```bash
pip install "pytest-agentcontract[anthropic]"
```

```python
from agentcontract.recorder.interceptors import patch_anthropic

unpatch = patch_anthropic(client, recorder)
try:
    response = client.messages.create(
        model="claude-model-name",
        messages=[{"role": "user", "content": "Refund order ORD-123"}],
        tools=[...],
    )
finally:
    unpatch()
```

`patch_anthropic` wraps `client.messages.create`. It records text blocks and `tool_use` blocks as an assistant turn, including input/output token counts when available.

Current limitation: Anthropic `tool_use` blocks represent requests. Tool results are supplied later by your application in `tool_result` messages, so record or backfill those results separately if replay stubbing needs them.

## LangGraph

```python
from agentcontract.adapters import record_graph

unpatch = record_graph(compiled_graph, recorder)
try:
    result = compiled_graph.invoke({"messages": [("user", "I need a refund")]})
finally:
    unpatch()
```

`record_graph` wraps `invoke()` and `ainvoke()` on a LangGraph compiled graph. It expects a result dict with a `messages` list and records supported roles plus LangChain-style `tool_calls`.

## LlamaIndex

```python
from agentcontract.adapters import record_agent

unpatch = record_agent(agent, recorder)
try:
    response = agent.chat("What's the refund policy?")
finally:
    unpatch()
```

`record_agent` wraps `chat()`, `achat()`, `query()`, and `aquery()` when present. It records response text, tool outputs from `sources`, and retrieval information from `source_nodes` as `_retrieve` tool calls.

## OpenAI Agents SDK

The OpenAI Agents SDK adapter imports `Runner` from the `agents` package. Install that package separately:

```bash
pip install openai-agents
```

```python
from agents import Runner
from agentcontract.adapters import record_runner

unpatch = record_runner(recorder)
try:
    result = Runner.run_sync(agent, "Help with billing")
finally:
    unpatch()
```

`record_runner` patches `Runner.run()` and `Runner.run_sync()` at the class level. It records `MessageOutputItem`, `ToolCallItem`, `ToolCallOutputItem`, and `HandoffCallItem` data when available, with a fallback to `final_output`.

## General Adapter Guidance

Always pair patching with `try/finally` so test failures do not leave SDK or framework methods patched.

Review generated cassettes before committing them. Adapter extraction depends on framework response shapes, and unsupported streaming or custom message formats may need manual `Recorder.add_turn()` calls.

For replay stubbing, make sure recorded tool calls include `result`. OpenAI and Anthropic SDK interceptors do not capture results by themselves.
