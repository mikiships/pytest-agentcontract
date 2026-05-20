# Adapters

pytest-agentcontract supports manual recording, SDK interceptors, and lightweight adapters for common agent frameworks. All automatic adapters return an `unpatch()` function. Call it in `finally` or immediately after the agent run so later tests do not inherit patched behavior.

## Manual Recording

Manual recording gives the most complete and predictable cassettes because your test controls every turn and tool result:

```python
from agentcontract.recorder.core import Recorder


recorder = Recorder(scenario="refund-eligible")
with recorder.recording():
    recorder.add_turn(role="user", content="Refund order ORD-123")
    recorder.add_turn(
        role="assistant",
        content="Checking refund eligibility...",
        tool_calls=[
            {
                "id": "tc_eligibility",
                "function": "check_refund_eligibility",
                "arguments": {"order_id": "ORD-123"},
                "result": {"eligible": True, "amount": 79.99},
            }
        ],
    )
recorder.save("tests/scenarios/refund-eligible.agentrun.json")
```

Use manual recording when application-specific tool outputs are important to assertions or replay.

## OpenAI SDK Interceptor

```python
import openai
from agentcontract.recorder.interceptors import patch_openai


def test_openai_agent(ac_recorder):
    client = openai.OpenAI()
    unpatch = patch_openai(client, ac_recorder)
    try:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Refund order ORD-123"}],
            tools=[...],
        )
    finally:
        unpatch()
```

`patch_openai()` wraps `client.chat.completions.create` and supports sync and async calls. It records assistant content, tool call ids, function names, parsed argument objects, usage tokens, latency, and model metadata when those fields are present.

Limitations:

- It does not record user messages.
- It does not record tool results because OpenAI chat completion responses contain tool call requests, not application tool outputs.
- Unsupported response shapes, such as stream-like objects without a `choices` list, are skipped to avoid recording empty turns.

## Anthropic SDK Interceptor

```python
from agentcontract.recorder.interceptors import patch_anthropic


unpatch = patch_anthropic(client, ac_recorder)
try:
    client.messages.create(
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Refund order ORD-123"}],
        tools=[...],
    )
finally:
    unpatch()
```

`patch_anthropic()` wraps `client.messages.create` and supports sync and async calls. It records text blocks, `tool_use` ids, tool names, input objects, usage tokens, latency, and model metadata when present.

Limitations:

- It does not record user messages.
- It records `tool_use` requests, not later `tool_result` payloads.
- Unsupported response shapes without a content block list are skipped.

## LangGraph

```python
from agentcontract.adapters import record_graph


unpatch = record_graph(compiled_graph, recorder)
try:
    result = compiled_graph.invoke({"messages": [("user", "I need a refund")]})
finally:
    unpatch()
```

`record_graph()` wraps a LangGraph compiled graph's `invoke()` and `ainvoke()` methods. It expects the graph result to be a dict with a `messages` list. It records messages with roles mapped from LangChain message types:

| LangChain type | Recorded role |
| --- | --- |
| `human` | `user` |
| `ai` | `assistant` |
| `system` | `system` |
| `tool` | `tool` |

Tool calls are extracted from message `tool_calls` entries using `name`, `args` or `arguments`, and `id`. Tool result payloads are only captured if they are present in the returned messages in a form the adapter can read.

## LlamaIndex

```python
from agentcontract.adapters import record_agent


unpatch = record_agent(agent, recorder)
try:
    response = agent.chat("What's the refund policy?")
finally:
    unpatch()
```

`record_agent()` wraps `chat()`, `achat()`, `query()`, and `aquery()` when those methods exist. It records assistant response text and tool calls extracted from response `sources` and `source_nodes`.

For `sources`, the adapter reads:

| Source field | Recorded field |
| --- | --- |
| `tool_name` | `function` |
| `raw_input` | `arguments` when it is a dict |
| `raw_output` | `result` as a string when present |

For `source_nodes`, the adapter records `_retrieve` tool calls with optional score metadata and a short text result.

## OpenAI Agents SDK

```python
from agentcontract.adapters import record_runner
from agents import Runner


unpatch = record_runner(recorder)
try:
    result = Runner.run_sync(agent, "Help with billing")
finally:
    unpatch()
```

`record_runner()` imports `agents.Runner` and patches `Runner.run()` and `Runner.run_sync()` at the class level. Install the OpenAI Agents SDK separately with `pip install openai-agents`.

When the result exposes `new_items`, the adapter records:

| Item type | Recorded turn |
| --- | --- |
| `MessageOutputItem` | Assistant content and message tool calls. |
| `ToolCallItem` | Assistant turn containing one tool call. |
| `ToolCallOutputItem` | Tool turn with output as content. |
| `HandoffCallItem` | Assistant turn with handoff text. |

If `new_items` is not present, the adapter falls back to recording `final_output` as one assistant turn.

Tool call outputs are recorded as separate `tool` turns, not as the `result` field of a preceding `ToolCall`.

## Replay With Tool Stubs

The replay engine provides recorded tool results in cassette order:

```python
stub = ac_replay_engine.tool_stub
order = stub.get_result("lookup_order", {"order_id": "ORD-123"})
refund = stub.get_result(
    "process_refund",
    {"order_id": "ORD-123", "amount": 79.99, "method": "original"},
)
result = ac_replay_engine.finish()
assert result.ok, result.errors
```

When arguments are provided to `get_result()`, they must exactly match the next recorded call for that function. A mismatch raises `ToolStubArgumentsMismatch` and does not consume the recorded call.
