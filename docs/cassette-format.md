# Cassette Format

Recorded trajectories are saved as JSON files with the `.agentrun.json` suffix.
The serializer writes a top-level `AgentRun` object.

## Top-Level Structure

```json
{
  "schema_version": "1.0.0",
  "run_id": "bd3fdc7e-5bcc-496d-9db6-6501fd818482",
  "source": {
    "recorded_at": "2026-02-17T20:21:47.251735+00:00",
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
    "total_duration_ms": 0.86,
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

## Source Metadata

`source.recorded_at` is the UTC timestamp captured by `Recorder`.
`source.recorder_version` is the package version known to the recorder.
`source.sdk` defaults to `agentcontract-python`.

## Model

`model` contains provider information and generation settings:

- `provider`
- `model`
- `temperature`
- `top_p`
- `max_tokens`
- `seed`

Manual `Recorder(...)` construction accepts `model_provider`, `model_name`,
`temperature`, and `seed`. SDK interceptors may fill provider and model from the
response.

## Metadata

`metadata.scenario` is the scenario name used by pytest to match a test to a
cassette. `tags` and `description` can be set when constructing a `Recorder`.

## Summary

`summary` stores aggregate counts and timing:

- `total_turns`
- `total_duration_ms`
- `total_tokens.prompt`
- `total_tokens.completion`
- `total_tokens.total`
- `total_tool_calls`
- `estimated_cost_usd`

The core recorder currently computes duration, turn count, tool-call count, and
token totals. Estimated cost defaults to `0.0`.

## Turns

Each turn has:

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
        "status": "delivered"
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

Supported roles are `system`, `user`, `assistant`, and `tool`.

`content` is omitted when it is `None`. `tool_calls`, `timing`, and `tokens` are
also omitted when absent. Tool-call `arguments` are always normalized to a JSON
object; non-object arguments become `{}`.

## Tool Calls

Tool calls contain:

- `id`
- `function`
- `arguments`
- `result`
- `duration_ms`

The replay `ToolStub` indexes tool calls by `function` and returns `result` in
recorded order. If a recorder or interceptor did not capture a result, replaying
that tool call returns `null`/`None`.

## Compatibility Notes

- `schema_version` defaults to `1.0.0`.
- `load_run()` is tolerant of missing optional sections and null list/object
  fields.
- A turn must have a valid `role`; missing or unknown roles raise `ValueError`.
- Non-string turn content is converted to a string when loading.
- `save_run()` creates parent directories and writes indented JSON.
