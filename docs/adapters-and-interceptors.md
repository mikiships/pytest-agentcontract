# Adapters and Interceptors

`pytest-agentcontract` can record trajectories manually or by wrapping SDK and
framework methods. All wrappers return an `unpatch` function. Call it in a
`finally` block when possible.

## Manual Recorder Usage

```python
from agentcontract.recorder.core import Recorder

recorder = Recorder(
    scenario="refund-eligible",
    tags=["refund"],
    model_provider="openai",
    model_name="gpt-4o",
)

with recorder.recording():
    recorder.add_turn(role="user", content="Refund order ORD-123")
    recorder.add_turn(
        role="assistant",
        content="Your refund has been processed.",
        tool_calls=[
            {
                "id": "tc_refund",
                "function": "process_refund",
                "arguments": {"order_id": "ORD-123", "amount": 79.99},
                "result": {"success": True},
            }
        ],
        latency_ms=150.0,
        prompt_tokens=20,
        completion_tokens=12,
    )

recorder.save("tests/scenarios/refund-eligible.agentrun.json")
```

Valid roles are `system`, `user`, `assistant`, and `tool`. The pytest
`ac_recorder` fixture already enters `recorder.recording()` for the duration of
the test and auto-saves in `--ac-record` mode.

## OpenAI Interceptor

```python
from agentcontract.recorder.interceptors import patch_openai

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

`patch_openai()` wraps `client.chat.completions.create`. It supports sync and
async responses, records the first choice message, extracts OpenAI tool-call
IDs, function names, JSON arguments, usage tokens, provider, model, and latency.

Streaming or otherwise unexpected response shapes are skipped instead of
recording empty assistant turns.

OpenAI chat completion responses do not include your application's tool results.
The interceptor records tool-call requests only. Add tool-result turns manually
or post-process the cassette if replay needs `ToolStub` results.

## Anthropic Interceptor

```python
from agentcontract.recorder.interceptors import patch_anthropic

unpatch = patch_anthropic(client, ac_recorder)
try:
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Refund order ORD-123"}],
        tools=[...],
    )
finally:
    unpatch()
```

`patch_anthropic()` wraps `client.messages.create`. It supports sync and async
responses, concatenates text content blocks, records `tool_use` blocks as tool
calls, and captures usage tokens, provider, model, and latency when present.

Like the OpenAI interceptor, it records tool-use requests but not later
`tool_result` messages.

## LangGraph Adapter

```python
from agentcontract.adapters import record_graph

unpatch = record_graph(compiled_graph, ac_recorder)
try:
    result = compiled_graph.invoke({"messages": [("user", "I need a refund")]})
finally:
    unpatch()
```

`record_graph()` wraps a LangGraph compiled graph's `invoke()` and/or
`ainvoke()` methods. It expects a result dictionary with a `messages` list. It
understands LangChain-style message types (`human`, `ai`, `system`, `tool`) and
dict messages with `role`/`type` and `content` fields.

Tool calls are extracted from message `tool_calls` entries. Tool result messages
can be recorded as `tool` turns when present in the message list; they are not
automatically attached to the earlier assistant tool call's `result` field.

## LlamaIndex Adapter

```python
from agentcontract.adapters import record_agent

unpatch = record_agent(agent, ac_recorder)
try:
    response = agent.chat("What's the refund policy?")
finally:
    unpatch()
```

`record_agent()` wraps `chat()`, `achat()`, `query()`, and `aquery()` when those
methods exist. It records assistant response text from response objects and
extracts tool calls from `sources` entries with `tool_name`, `raw_input`, and
`raw_output`.

Retrieval `source_nodes` are recorded as `_retrieve` tool calls with score
metadata and a short text result.

## OpenAI Agents SDK Adapter

```python
from agentcontract.adapters import record_runner
from agents import Runner

unpatch = record_runner(ac_recorder)
try:
    result = Runner.run_sync(agent, "Help with billing")
finally:
    unpatch()
```

`record_runner()` imports `Runner` from the `agents` package and patches
`Runner.run()` and/or `Runner.run_sync()` at the class level. Install the SDK
with `pip install openai-agents`.

The adapter records:

- `MessageOutputItem` as assistant turns.
- `ToolCallItem` as assistant turns containing tool calls.
- `ToolCallOutputItem` as separate `tool` turns.
- `HandoffCallItem` as assistant turns with handoff text.

Because patching is class-level, all `Runner` calls in the process are affected
until `unpatch()` is called.

## Replay Limitations

The replay `ToolStub` returns values from `ToolCall.result`. Some wrappers record
tool results as separate `tool` turns or cannot observe them at all. If your
replay path needs exact tool stubbing, make sure the cassette's assistant tool
calls contain `result` values.
