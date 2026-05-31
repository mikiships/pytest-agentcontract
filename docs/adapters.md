# Adapters And Interceptors

pytest-agentcontract can record manually through `Recorder.add_turn(...)`, or it
can wrap common SDK and framework entry points. Every wrapper returns an
`unpatch` callable. Call it in a `finally` block or immediately after the agent
run so later tests are not affected.

## SDK Interceptors

### OpenAI

```python
import openai

from agentcontract.recorder.interceptors import patch_openai


client = openai.OpenAI()
unpatch = patch_openai(client, ac_recorder)
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Refund order 123"}],
        tools=[...],
    )
finally:
    unpatch()
```

`patch_openai(client, recorder)` patches `client.chat.completions.create`.
It records supported sync and awaitable responses as assistant turns, including:

- provider `openai`
- response model name
- assistant content
- tool call id, function name, and parsed JSON arguments
- latency
- prompt and completion token counts when present

OpenAI tool call results are not present in the chat completion response. Record
tool results manually with `recorder.add_turn(...)` or backfill them from your
tool execution layer.

### Anthropic

```python
from agentcontract.recorder.interceptors import patch_anthropic


unpatch = patch_anthropic(client, ac_recorder)
try:
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Refund order 123"}],
        tools=[...],
    )
finally:
    unpatch()
```

`patch_anthropic(client, recorder)` patches `client.messages.create`. It records
text blocks and `tool_use` blocks as assistant turns, including:

- provider `anthropic`
- response model name
- text content
- tool use id, name, and input object
- latency
- input and output token counts when present

As with OpenAI, tool results come from your application code, not from the model
response, so record them manually when you need replay stubs to return results.

## Framework Adapters

Adapters live under `agentcontract.adapters` and can be imported directly or via
the lazy adapter package:

```python
from agentcontract.adapters import record_agent, record_graph, record_runner
```

### LangGraph

```python
from agentcontract.adapters import record_graph


unpatch = record_graph(compiled_graph, ac_recorder)
try:
    result = compiled_graph.invoke({"messages": [("user", "I need a refund")]})
finally:
    unpatch()
```

`record_graph(graph, recorder)` patches `invoke()` and `ainvoke()` when present.
The graph must define at least one of those methods. The adapter expects a
LangGraph-style result dictionary with a `messages` list. It records messages
whose roles resolve to `user`, `assistant`, `system`, or `tool`, and extracts
LangChain-style `tool_calls` from assistant messages.

### LlamaIndex

```python
from agentcontract.adapters import record_agent


unpatch = record_agent(agent, ac_recorder)
try:
    response = agent.chat("What's the refund policy?")
finally:
    unpatch()
```

`record_agent(agent, recorder)` patches any available `chat()`, `achat()`,
`query()`, and `aquery()` methods. The agent must define at least one of them.
The adapter records assistant response text and tool calls from response
`sources`. Retrieval source nodes are recorded as `_retrieve` tool calls.

### OpenAI Agents SDK

```python
from agents import Runner
from agentcontract.adapters import record_runner


unpatch = record_runner(ac_recorder)
try:
    result = Runner.run_sync(agent, "Help with billing")
finally:
    unpatch()
```

`record_runner(recorder)` imports `Runner` from the `agents` package and patches
`Runner.run()` and `Runner.run_sync()` at the class level. Install the OpenAI
Agents SDK first:

```bash
pip install openai-agents
```

The adapter prefers `result.new_items` because it can include messages, tool
calls, tool outputs, and handoffs. If those items are unavailable, it records
`result.final_output` as an assistant turn.

## Patch And Unpatch Expectations

- Patch only around the code under test.
- Always call the returned `unpatch` function.
- Use `try/finally` when the wrapped call can raise.
- Keep one recorder per scenario.
- For SDK interceptors, manually record tool result turns if you need replay
  stubs to return those results.
