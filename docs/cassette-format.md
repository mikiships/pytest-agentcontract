# Cassette Format Reference

This page documents the `.agentrun.json` structure serialized by `src/agentcontract/serialization.py` from `AgentRun`, `Turn`, and `ToolCall`.

## Top-Level Shape

Serialized files use this structure:

```json
{
  "schema_version": "1.0.0",
  "run_id": "...",
  "source": {
    "recorded_at": "2026-01-01T12:00:00+00:00",
    "recorder_version": "0.1.1",
    "sdk": "agentcontract-python"
  },
  "model": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 4096,
    "seed": 42
  },
  "metadata": {
    "scenario": "refund-eligible",
    "tags": [],
    "description": ""
  },
  "summary": {
    "total_turns": 4,
    "total_duration_ms": 812.5,
    "total_tokens": {
      "prompt": 120,
      "completion": 44,
      "total": 164
    },
    "total_tool_calls": 2,
    "estimated_cost_usd": 0.0
  },
  "turns": []
}
```

## Field Reference

| Path | Type | Notes |
| --- | --- | --- |
| `schema_version` | string | Default `"1.0.0"` |
| `run_id` | string | Recorder sets UUID |
| `source.recorded_at` | string | ISO timestamp |
| `source.recorder_version` | string | Package version |
| `source.sdk` | string | Default `"agentcontract-python"` |
| `model.provider` | string | LLM provider |
| `model.model` | string | Model name |
| `model.temperature` | number | Default `0.0` |
| `model.top_p` | number | Default `1.0` |
| `model.max_tokens` | integer | Default `4096` |
| `model.seed` | integer or null | Optional |
| `metadata.scenario` | string | Scenario name |
| `metadata.tags` | array | Defaults to `[]` |
| `metadata.description` | string | Optional description |
| `summary.total_turns` | integer | Usually filled by recorder context manager |
| `summary.total_duration_ms` | number | Runtime duration |
| `summary.total_tokens.prompt` | integer | Prompt tokens |
| `summary.total_tokens.completion` | integer | Completion tokens |
| `summary.total_tokens.total` | integer | Prompt + completion |
| `summary.total_tool_calls` | integer | Count of tool calls |
| `summary.estimated_cost_usd` | number | Defaults to `0.0` unless set |
| `turns` | array of turn objects | Ordered trajectory |

## Turn Object

A turn has:

```json
{
  "index": 0,
  "role": "assistant",
  "content": "...",
  "tool_calls": [],
  "timing": {
    "latency_ms": 10.2,
    "time_to_first_token_ms": null
  },
  "tokens": {
    "prompt": 10,
    "completion": 4,
    "total": 14
  }
}
```

Rules:
- `role` is required when loading and must be one of: `system`, `user`, `assistant`, `tool`.
- `content` is optional and omitted when `None` during serialization.
- `tool_calls` is optional and omitted when empty.
- `timing` and `tokens` are optional and omitted when absent.

## Tool Call Object

Each `tool_calls[]` entry is serialized as:

```json
{
  "id": "tc1",
  "function": "lookup_order",
  "arguments": {"order_id": "123"},
  "result": {"status": "delivered"},
  "duration_ms": 5.4
}
```

Rules:
- `arguments` is always normalized to an object (`{}` if missing/non-object on load).
- `result` may be any JSON-compatible value or `null`.
- `duration_ms` is optional (`number` or `null`).

## Load-Time Normalization

`run_from_dict(...)` is permissive:
- Non-object `source/model/metadata/summary` become empty objects.
- Non-list `turns` and `tool_calls` become empty lists.
- Numeric/string fields are coerced to expected scalar types where possible.
- Non-string `content` is converted to string.
- Unknown fields are ignored.

Failure cases:
- Missing turn `role` raises `ValueError`.
- Invalid turn role value raises `ValueError`.

## Assertion Target Syntax (Contract Checks)

Assertion targets are resolved by `AssertionEngine._resolve_target(...)`:
- `final_response`: last assistant turn with non-`None` content
- `full_conversation`: newline-joined `<role>: <content>` for all turns with content
- `turn:N`: content from turn index `N`
- `tool_call:<function>:arguments`: first matching tool-call arguments
- `tool_call:<function>:result`: first matching tool-call result
- `tool_call:<function>`: same as `tool_call:<function>:arguments`

For `not_called`, `called_with`, and `called_count`, target format is either:
- `tool:<function>`
- `<function>`
