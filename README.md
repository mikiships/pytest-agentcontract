# pytest-agentcontract

**Deterministic CI tests for LLM agent trajectories.** Record once, replay offline, assert contracts.

[![PyPI](https://img.shields.io/pypi/v/pytest-agentcontract)](https://pypi.org/project/pytest-agentcontract/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

LLM agent tests are either **flaky** (live API calls, rate limits, nondeterminism) or **meaningless** (everything mocked away). You can't tell if a prompt change broke tool selection, argument formatting, or the overall flow.

## The Solution

```python
import pytest
from agentcontract import Recorder

@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay":
        # Deterministic: no network, no API keys, no cost
        stub = ac_replay_engine.tool_stub
        result = stub.get_result("lookup_order")
        assert result["status"] == "delivered"
    else:
        # Live or record mode: hits real APIs
        # ... run your agent ...
        ac_recorder.add_turn(role="user", content="I want a refund for order 123")
        ac_recorder.add_turn(
            role="assistant",
            content="Let me check that order.",
            tool_calls=[{
                "id": "tc_1",
                "function": "lookup_order",
                "arguments": {"order_id": "123"},
                "result": {"status": "delivered", "total": 49.99},
            }],
        )

    # Assert the contract: tool sequence, schemas, policies
    contract = ac_check_contract(ac_recorder.run)
    assert contract.passed, contract.failures()
```

```bash
# Record once (hits real APIs)
pytest --ac-record

# Replay in CI forever (no network, no keys, deterministic)
pytest --ac-replay
```

## Install

```bash
pip install pytest-agentcontract
```

With SDK interceptors:
```bash
pip install pytest-agentcontract[openai]     # Auto-record OpenAI calls
pip install pytest-agentcontract[anthropic]   # Auto-record Anthropic calls
pip install pytest-agentcontract[all]         # Everything
```

## How It Works

1. **Record** a trajectory: run your agent, capture every turn + tool call + result as a `.agentrun.json` cassette
2. **Replay** in CI: tool calls return recorded results, zero network, zero tokens
3. **Assert contracts**: validate tool sequences, argument schemas, policies (allowlists, confirmation gates)

## Configuration

Create `agentcontract.yml`:

```yaml
version: "1"

scenarios:
  include: ["tests/scenarios/**/*.agentrun.json"]

replay:
  stub_tools: true

defaults:
  assertions:
    - type: contains
      target: final_response
      value: "refund"

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_eligibility, process_refund]

  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

Or generate a starter config:
```bash
agentcontract init
```

## Assertion Types

| Type | What It Does |
|------|-------------|
| `exact` | Exact string match on target |
| `contains` | Substring check |
| `regex` | Regex pattern match |
| `json_schema` | JSON Schema validation on tool args/results |
| `not_called` | Assert a tool was NOT invoked |
| `called_with` | Assert a tool was called with specific args |
| `called_count` | Assert exact invocation count |

## Policies

| Policy | What It Does |
|--------|-------------|
| `tool_allowlist` | Only listed tools may be called |
| `requires_confirmation` | Protected tools must be preceded by user confirmation |

## Target Syntax

- `final_response` — last assistant message
- `turn:N` — specific turn by index
- `full_conversation` — all turns concatenated
- `tool_call:function_name:arguments` — tool call arguments
- `tool_call:function_name:result` — tool call result

## SDK Auto-Recording

Instead of manually adding turns, intercept real SDK calls:

```python
from agentcontract.recorder.interceptors import patch_openai

def test_with_auto_record(ac_recorder):
    import openai
    client = openai.OpenAI()

    unpatch = patch_openai(client, ac_recorder)
    try:
        # All chat.completions.create calls are automatically recorded
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Check order 123"}],
            tools=[...],
        )
    finally:
        unpatch()
```

## CLI

```bash
agentcontract info tests/scenarios/refund.agentrun.json   # Show cassette summary
agentcontract validate tests/scenarios/refund.agentrun.json  # Validate structure
agentcontract init                                          # Create starter config
```

## Why Not Just VCR / pytest-recording?

VCR records **HTTP requests**. pytest-agentcontract records **agent trajectories**: the full sequence of turns, tool calls, arguments, and results. This lets you assert on *what the agent decided to do*, not just what HTTP calls were made.

- VCR: "did the HTTP request match?" → brittle, breaks on any API change
- agentcontract: "did the agent call the right tools in the right order with the right args?" → tests actual behavior

## License

MIT
