# Integrations

pytest-agentcontract can record trajectories manually, through SDK
interceptors, or through framework adapters.

## Manual Recording

Manual recording is the most explicit path and works with any agent:

```python
from agentcontract.recorder.core import Recorder

recorder = Recorder(
    scenario="refund-eligible",
    model_provider="openai",
    model_name="gpt-4o",
    temperature=0.0,
)

with recorder.recording():
    recorder.add_turn(role="user", content="Refund order ORD-123")
    recorder.add_turn(
        role="assistant",
        content="I'll check your order.",
        tool_calls=[
            {
                "id": "call_1",
                "function": "lookup_order",
                "arguments": {"order_id": "ORD-123"},
                "result": {"status": "delivered"},
                "duration_ms": 18.4,
            }
        ],
        prompt_tokens=120,
        completion_tokens=30,
    )

recorder.save("tests/scenarios/refund-eligible.agentrun.json")
```

Use this when your agent framework already exposes a clean event stream or when
you want full control over recorded tool results.

## OpenAI SDK Interceptor

Install the OpenAI extra:

```bash
pip install pytest-agentcontract[openai]
```

Patch an OpenAI client:

```python
from agentcontract.recorder.interceptors import patch_openai

client = openai.OpenAI()
unpatch = patch_openai(client, ac_recorder)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Refund order ORD-123"}],
    tools=[...],
)

unpatch()
```

The interceptor records `chat.completions.create` assistant responses, latency,
usage, and tool call requests. Tool results are not present in the OpenAI
response; add those manually with `recorder.add_turn()` or backfill them from
your tool execution path.

## Anthropic SDK Interceptor

Install the Anthropic extra:

```bash
pip install pytest-agentcontract[anthropic]
```

Patch an Anthropic client:

```python
from agentcontract.recorder.interceptors import patch_anthropic

unpatch = patch_anthropic(client, ac_recorder)
response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Refund order ORD-123"}],
    tools=[...],
)
unpatch()
```

The interceptor records text blocks, `tool_use` requests, latency, and usage.
As with OpenAI, tool result messages come from application code and should be
recorded manually if you need replayable tool results.

## LangGraph Adapter

```python
from agentcontract.adapters import record_graph

unpatch = record_graph(compiled_graph, ac_recorder)
result = compiled_graph.invoke({"messages": [("user", "I need a refund")]})
unpatch()
```

The adapter wraps `invoke()` and `ainvoke()` on a compiled graph and records
messages from the returned state dictionary's `messages` key.

## LlamaIndex Adapter

```python
from agentcontract.adapters import record_agent

unpatch = record_agent(agent, ac_recorder)
response = agent.chat("What's the refund policy?")
unpatch()
```

The adapter wraps `chat()`, `achat()`, `query()`, and `aquery()` when present.
It records assistant response text and tool or retrieval outputs exposed on the
response object.

## OpenAI Agents SDK Adapter

Install the OpenAI Agents SDK in your project:

```bash
pip install openai-agents
```

Then patch the `Runner` class:

```python
from agentcontract.adapters import record_runner
from agents import Runner

unpatch = record_runner(ac_recorder)
result = Runner.run_sync(agent, "Help with billing")
unpatch()
```

The adapter wraps `Runner.run()` and `Runner.run_sync()`. It records run items
when available, including message outputs, tool calls, tool outputs, handoffs,
and final output fallback.

## Unpatching

Every interceptor and adapter returns an `unpatch` callable. Call it in a
`finally` block for long-running tests or shared clients:

```python
unpatch = patch_openai(client, ac_recorder)
try:
    run_agent()
finally:
    unpatch()
```
