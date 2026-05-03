# Replay and Cassettes

pytest-agentcontract stores recorded trajectories as `.agentrun.json` files.
These cassettes capture agent-level behavior: turns, tool calls, arguments,
results, timing, token counts, and run metadata.

## Cassette Structure

A cassette has this top-level shape:

```json
{
  "schema_version": "1.0.0",
  "run_id": "uuid",
  "source": {
    "recorded_at": "2026-05-03T12:00:00+00:00",
    "recorder_version": "0.1.1",
    "sdk": "agentcontract-python"
  },
  "model": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 4096,
    "seed": 42
  },
  "metadata": {
    "scenario": "refund-eligible",
    "tags": [],
    "description": ""
  },
  "summary": {
    "total_turns": 5,
    "total_duration_ms": 250.0,
    "total_tokens": {
      "prompt": 100,
      "completion": 50,
      "total": 150
    },
    "total_tool_calls": 3,
    "estimated_cost_usd": 0.0
  },
  "turns": [
    {
      "index": 0,
      "role": "user",
      "content": "I'd like a refund for order ORD-123"
    },
    {
      "index": 1,
      "role": "assistant",
      "content": "I'll look that up.",
      "tool_calls": [
        {
          "id": "call_1",
          "function": "lookup_order",
          "arguments": {"order_id": "ORD-123"},
          "result": {"status": "delivered"},
          "duration_ms": 12.0
        }
      ]
    }
  ]
}
```

Valid turn roles are `system`, `user`, `assistant`, and `tool`.

## How Recorder Saves Runs

The `Recorder` creates an `AgentRun`, appends turns with `add_turn`, computes a
summary when the `recording()` context exits, and writes JSON with `save()`.

```python
from agentcontract.recorder.core import Recorder

recorder = Recorder(scenario="refund-eligible")

with recorder.recording():
    recorder.add_turn(role="user", content="Refund order ORD-123")
    recorder.add_turn(
        role="assistant",
        content="I'll check the order.",
        tool_calls=[
            {
                "id": "call_1",
                "function": "lookup_order",
                "arguments": {"order_id": "ORD-123"},
                "result": {"status": "delivered"},
            }
        ],
    )

recorder.save("tests/scenarios/refund-eligible.agentrun.json")
```

In pytest record mode, the `ac_recorder` fixture auto-saves to:

```text
tests/scenarios/<scenario>.agentrun.json
```

Use `--ac-scenarios` to override that directory.

## Replay Engine and ToolStub

`ReplayEngine` loads a recorded run and exposes a `tool_stub` that returns
recorded tool results in the original order.

```python
stub = ac_replay_engine.tool_stub

order = stub.get_result("lookup_order", {"order_id": "ORD-123"})
eligibility = stub.get_result(
    "check_refund_eligibility",
    {"order_id": "ORD-123"},
)
```

If a tool is called more times than recorded, `ToolStubExhausted` is raised. If
arguments do not match the recorded arguments for that call position,
`ToolStubArgumentsMismatch` is raised.

After replaying an agent path, call `finish()` to detect unconsumed tool calls:

```python
result = ac_replay_engine.finish()
assert result.ok, result.errors
```

`finish(actual_turns)` can also compare actual turns to the recorded trajectory,
including role, tool function, and tool argument differences.

## Validation Commands

Inspect a cassette:

```bash
agentcontract info tests/scenarios/refund-eligible.agentrun.json
```

Validate that it can be loaded:

```bash
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
```

The validation command checks deserialization and required turn role values. It
does not validate business assertions; run pytest for that.

## CI Usage

A typical CI job replays committed cassettes:

```yaml
name: tests

on: [push, pull_request]

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: pytest --ac-replay
```

Record cassettes locally or in a controlled environment with API keys, review
the generated `.agentrun.json` files, then commit them with the tests.
