# Cassette Format and CLI Reference

Recorded runs are stored as `.agentrun.json` files. The in-memory type is `AgentRun`, but the serialized JSON shape uses a top-level `source` object for recording metadata.

## On-Disk Shape

```json
{
  "schema_version": "1.0.0",
  "run_id": "efa47af3-80cd-412e-8398-08830a44b0ad",
  "source": {
    "recorded_at": "2026-02-17T20:21:47.254216+00:00",
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
    "scenario": "refund-with-policies",
    "tags": [],
    "description": ""
  },
  "summary": {
    "total_turns": 5,
    "total_duration_ms": 0.3860419965349138,
    "total_tokens": {
      "prompt": 0,
      "completion": 0,
      "total": 0
    },
    "total_tool_calls": 3,
    "estimated_cost_usd": 0.0
  },
  "turns": [
    {
      "index": 0,
      "role": "user",
      "content": "I'd like a refund for order ORD-123 please"
    }
  ]
}
```

## Top-Level Fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Cassette schema version written by the recorder |
| `run_id` | UUID for the captured run |
| `source.recorded_at` | ISO-8601 timestamp from the recorder |
| `source.recorder_version` | Package version captured at record time |
| `source.sdk` | SDK label, currently `agentcontract-python` by default |
| `model` | Provider and model metadata from the recorder |
| `metadata.scenario` | Scenario name used by the pytest plugin and examples |
| `metadata.tags` | Free-form tags supplied to `Recorder(...)` |
| `metadata.description` | Optional description supplied to `Recorder(...)` |
| `summary` | Aggregated counts and cost/timing totals |
| `turns` | Ordered list of recorded turns |

`Recorder.save(...)` appends `.agentrun.json` when the provided path has no suffix.

## `model`

`Recorder(...)` can populate:

- `provider`
- `model`
- `temperature`
- `top_p`
- `max_tokens`
- `seed`

Only `provider`, `model`, `temperature`, and `seed` are configurable through the current `Recorder` constructor. `top_p` and `max_tokens` stay at their defaults unless you build `AgentRun` objects yourself.

## `summary`

`summary` contains:

- `total_turns`
- `total_duration_ms`
- `total_tokens.prompt`
- `total_tokens.completion`
- `total_tokens.total`
- `total_tool_calls`
- `estimated_cost_usd`

When you use `Recorder.recording()`, the summary is finalized when that context exits. With the pytest `ac_recorder` fixture, that happens during fixture teardown after the test body has finished.

## Turns

Each turn has:

- `index`
- `role`: one of `system`, `user`, `assistant`, `tool`
- `content`: optional string
- `tool_calls`: optional list
- `timing`: optional object
- `tokens`: optional object

### `tool_calls`

Each tool call contains:

- `id`
- `function`
- `arguments`
- `result`
- `duration_ms`

Replay indexes tool calls by `function` and replays them in recorded order. `ToolStub.get_result(function, arguments)` optionally checks the next recorded `arguments` object before returning `result`.

### `timing`

If present, timing currently contains:

- `latency_ms`
- `time_to_first_token_ms`

The recorder populates `latency_ms` when you pass it to `add_turn(...)`. `time_to_first_token_ms` is part of the serialized schema and deserializer, but the current recorder does not set it.

### `tokens`

If present, tokens contains:

- `prompt`
- `completion`
- `total`

The recorder adds per-turn token info only when `prompt_tokens` or `completion_tokens` are passed to `add_turn(...)`.

## Serialization Behavior

`load_run(...)` and `run_from_dict(...)` are intentionally forgiving:

- Missing object sections such as `source`, `model`, `summary`, or `tool_calls` are normalized to defaults.
- Non-object tool arguments are coerced to `{}`.
- Numeric-looking strings are coerced to numbers where the schema expects them.
- Non-string turn content is stringified.
- Missing turn indexes default to `0`.
- Missing or invalid `role` values still raise `ValueError`.

That makes `agentcontract validate` a structural validation pass, not a strict schema validator.

## CLI Commands

The `agentcontract` CLI currently exposes three commands.

### `agentcontract info <path>`

Loads the cassette with `load_run(...)` and prints:

- `Scenario`
- `Run ID`
- `Recorded`
- `Model`
- `Turns`
- `Tool calls`
- `Duration`
- `Tokens`
- `Est. cost`

It exits with status `1` if the file is missing or cannot be parsed.

### `agentcontract validate <path>`

Loads the cassette with `load_run(...)`.

- On success it prints `Valid cassette: <scenario> (<turn-count> turns)` and exits `0`.
- On failure it prints an error to stderr and exits `1`.

Because it reuses `load_run(...)`, validation accepts the same normalized inputs described above.

### `agentcontract init`

Creates `agentcontract.yml` in the current directory using the built-in starter template.

- It exits `1` if `agentcontract.yml` already exists.
- It does not inspect or modify any cassette files.
