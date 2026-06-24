# Adapters and interceptors

You can record trajectories manually with `Recorder`, through SDK
interceptors, or through framework adapters. All approaches produce the same
`AgentRun` shape and can be saved as `.agentrun.json`.

## Manual recording

Manual recording is the most explicit path and is the easiest to make fully
replayable because you provide tool results yourself.

```python
from agentcontract.recorder.core import Recorder


recorder = Recorder(scenario="refund-eligible", model_provider="openai", model_name="gpt-4o")

with recorder.recording():
    recorder.add_turn(role="user", content="I want a refund for order ORD-123")
    recorder.add_turn(
        role="assistant",
        content="Let me look up that order.",
        tool_calls=[
            {
                "id": "tc_lookup",
                "function": "lookup_order",
                "arguments": {"order_id": "ORD-123"},
                "result": {"status": "delivered", "total": 79.99},
            }
        ],
    )

recorder.save("tests/scenarios/refund-eligible.agentrun.json")
```

`Recorder.add_turn()` accepts roles `system`, `user`, `assistant`, and `tool`.
Tool call dictionaries may include `id`, `function`, `arguments`, `result`, and
`duration_ms`.

## OpenAI SDK interceptor

Install the optional dependency:

```bash
pip install pytest-agentcontract[openai]
```

Patch a client instance:

```python
from agentcontract.recorder.interceptors import patch_openai


with recorder.recording():
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

The interceptor wraps `client.chat.completions.create` and supports sync and
async responses. It records the assistant content, model name, token usage when
available, and tool call requests.

Known limitation: OpenAI chat completion responses contain tool requests, not
your application's tool results. Add tool-result turns manually or backfill
results before relying on replay stubs.

## Anthropic SDK interceptor

Install the optional dependency:

```bash
pip install pytest-agentcontract[anthropic]
```

Patch a client instance:

```python
from agentcontract.recorder.interceptors import patch_anthropic


with recorder.recording():
    unpatch = patch_anthropic(client, recorder)
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            messages=[{"role": "user", "content": "Refund order ORD-123"}],
            tools=[...],
        )
    finally:
        unpatch()
```

The interceptor wraps `client.messages.create` and supports sync and async
responses. It records text blocks, `tool_use` requests, model name, and token
usage when available.

Known limitation: Anthropic message responses expose `tool_use` requests, while
tool results are produced later by your application. Add tool-result turns
manually or backfill results before relying on replay stubs.

## LangGraph adapter

```python
from agentcontract.adapters import record_graph


with recorder.recording():
    unpatch = record_graph(compiled_graph, recorder)
    try:
        result = compiled_graph.invoke({"messages": [("user", "I need a refund")]})
    finally:
        unpatch()
```

`record_graph()` wraps `invoke()` and `ainvoke()` on a LangGraph compiled graph.
It expects the result to be a dict with a `messages` list and records user,
assistant, system, and tool messages. LangChain `AIMessage.tool_calls` entries
are recorded as tool call requests.

## LlamaIndex adapter

```python
from agentcontract.adapters import record_agent


with recorder.recording():
    unpatch = record_agent(agent, recorder)
    try:
        response = agent.chat("What's the refund policy?")
    finally:
        unpatch()
```

`record_agent()` wraps `chat()`, `achat()`, `query()`, and `aquery()` when those
methods exist. It records assistant response text and tool calls from
`response.sources`. Retrieval `source_nodes` are recorded as `_retrieve` tool
calls with score metadata when available.

## OpenAI Agents SDK adapter

Install the SDK separately:

```bash
pip install openai-agents
```

Patch `agents.Runner` at the class level:

```python
from agentcontract.adapters import record_runner
from agents import Runner


with recorder.recording():
    unpatch = record_runner(recorder)
    try:
        result = Runner.run_sync(agent, "Help with billing")
    finally:
        unpatch()
```

`record_runner()` wraps `Runner.run()` and `Runner.run_sync()`. It records
`RunResult.new_items` when present, including message output items, tool call
items, tool call output items, and handoff items. If `new_items` is unavailable,
it records `final_output` as an assistant turn.

## Import shortcuts

The top-level adapter package lazy-loads framework adapters:

```python
from agentcontract.adapters import record_agent, record_graph, record_runner
```

You can also import concrete modules directly:

```python
from agentcontract.adapters.langgraph import record_graph
from agentcontract.adapters.llamaindex import record_agent
from agentcontract.adapters.openai_agents import record_runner
```
