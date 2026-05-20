# Cassette Format

Agent trajectories are stored as `.agentrun.json` files. The top-level Python type is `AgentRun`, serialized by `agentcontract.serialization.run_to_dict()` and loaded by `run_from_dict()`.

## Top-Level Fields

```json
{
  "schema_version": "1.0.0",
  "run_id": "6c151616-23ee-45bc-8b29-ff91a569acab",
  "source": {
    "recorded_at": "2026-02-18T18:33:33.305409+00:00",
    "recorder_version": "0.1.0",
    "sdk": "agentcontract-python"
  },
  "model": {
    "provider": "",
    "model": "",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 4096,
    "seed": null
  },
  "metadata": {
    "scenario": "refund-eligible",
    "tags": [],
    "description": ""
  },
  "summary": {
    "total_turns": 5,
    "total_duration_ms": 1.03,
    "total_tokens": {
      "prompt": 0,
      "completion": 0,
      "total": 0
    },
    "total_tool_calls": 3,
    "estimated_cost_usd": 0.0
  },
  "turns": []
}
```

| Field | Meaning |
| --- | --- |
| `schema_version` | Cassette schema version. The default is `1.0.0`. |
| `run_id` | Recorder-generated UUID for the run. |
| `source.recorded_at` | UTC timestamp from the recorder. |
| `source.recorder_version` | Package version used by the recorder. |
| `source.sdk` | SDK identifier, defaulting to `agentcontract-python`. |
| `model` | Provider, model name, sampling settings, token limit, and seed metadata. |
| `metadata.scenario` | Scenario name, usually from the pytest marker. |
| `metadata.tags` | Optional scenario tags. |
| `metadata.description` | Optional human description. |
| `summary` | Aggregate turn, duration, token, tool-call, and cost fields. |
| `turns` | Ordered list of recorded turns. |

`Recorder.recording()` computes the summary when the context manager exits.

## Turns

Each turn has this shape:

```json
{
  "index": 1,
  "role": "assistant",
  "content": "Let me look up order ORD-123.",
  "tool_calls": [
    {
      "id": "tc_lookup",
      "function": "lookup_order",
      "arguments": {
        "order_id": "ORD-123"
      },
      "result": {
        "status": "delivered",
        "total": 79.99
      },
      "duration_ms": null
    }
  ],
  "timing": {
    "latency_ms": 150.0,
    "time_to_first_token_ms": null
  },
  "tokens": {
    "prompt": 10,
    "completion": 20,
    "total": 30
  }
}
```

| Field | Meaning |
| --- | --- |
| `index` | Turn position recorded by the recorder. |
| `role` | One of `system`, `user`, `assistant`, or `tool`. |
| `content` | Optional text content. Non-string content is coerced to a string on load. |
| `tool_calls` | Optional list of tool calls made in this turn. |
| `timing.latency_ms` | Optional turn latency. |
| `timing.time_to_first_token_ms` | Optional time to first token. |
| `tokens.prompt` | Prompt or input token count. |
| `tokens.completion` | Completion or output token count. |
| `tokens.total` | Total token count for the turn. |

The loader requires a valid `role` for every turn. Missing turn indexes default to `0`.

## Tool Calls

Tool calls are attached to turns:

```json
{
  "id": "tc_refund",
  "function": "process_refund",
  "arguments": {
    "order_id": "ORD-123",
    "amount": 79.99,
    "method": "original"
  },
  "result": {
    "success": true,
    "refund_id": "REF-ORD-123"
  },
  "duration_ms": null
}
```

| Field | Meaning |
| --- | --- |
| `id` | Tool call identifier from the agent framework or recorder. |
| `function` | Tool or function name. |
| `arguments` | JSON object of tool arguments. Non-object arguments load as `{}`. |
| `result` | Recorded result payload. It may be any JSON-compatible value or `null`. |
| `duration_ms` | Optional tool duration. |

Some automatic interceptors can record tool call requests but not tool results because SDK responses do not contain application-level tool outputs. Use manual recording or framework adapters that expose tool outputs when result capture matters.

## Serialization Behavior

The serializer normalizes values so `json.dump()` can write the cassette:

| Input value | Serialized as |
| --- | --- |
| `str`, `int`, `float`, `bool`, `None` | Unchanged. |
| `dict` | Recursively normalized with string keys. |
| `list` or `tuple` | Recursively normalized list. |
| `set` or `frozenset` | Sorted list using `repr` ordering. |
| `pathlib.Path` | String path. |
| Objects with `isoformat()` | ISO string when callable without extra arguments. |
| Other objects | `str(value)`. |

When loading, nullable optional sections such as `source`, `model`, `summary`, `turns`, `tool_calls`, `timing`, and `tokens` are coerced to sensible defaults. Unknown keys are ignored by the current loader, so additive fields are generally safe for readers that only rely on the public dataclasses.

## Compatibility Expectations

Use `.agentrun.json` for cassette files and keep `schema_version` present. Existing consumers should tolerate additive fields, but producers should continue writing the documented fields so replay, assertions, and CLI commands can operate on the cassette.

For deterministic replay, preserve the order of `turns` and the order of `tool_calls` within each turn. `ToolStub` returns recorded results by function name in the order they appear in the cassette.
