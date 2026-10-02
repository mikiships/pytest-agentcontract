# pytest-agentcontract

**Deterministic CI tests for LLM agent trajectories.** Record once, replay offline, assert contracts.

[![PyPI](https://img.shields.io/pypi/v/pytest-agentcontract)](https://pypi.org/project/pytest-agentcontract/)
[![CI](https://github.com/mikiships/pytest-agentcontract/actions/workflows/ci.yml/badge.svg)](https://github.com/mikiships/pytest-agentcontract/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

<p align="center">
  <img src="docs/demo.gif" alt="pytest-agentcontract demo: record, replay, assert" width="600">
</p>

Your agent calls `lookup_order`, then `check_eligibility`, then `process_refund`. Every time. That's the contract. Test it like any other interface.

```bash
# Record a trajectory (hits real APIs once)
pytest --ac-record

# Check saved trajectories in CI (the test selects the recorded run)
pytest --ac-replay
```

```
tests/scenarios/refund-eligible.agentrun.json
├── turn 0: user → "I want a refund for order 123"
├── turn 1: assistant → lookup_order(order_id="123")
├── turn 2: assistant → check_eligibility(order_id="123")
├── turn 3: assistant → process_refund(order_id="123", amount=49.99)
└── turn 4: assistant → "Your refund of $49.99 has been processed."
```

## Install

```bash
pip install pytest-agentcontract
```

With auto-recording interceptors:
```bash
pip install pytest-agentcontract[openai]      # OpenAI SDK
pip install pytest-agentcontract[anthropic]    # Anthropic SDK
pip install pytest-agentcontract[all]          # Everything
```

Framework adapters (LangGraph, LlamaIndex, OpenAI Agents SDK) are included -- no extras needed.

## Quick Start

### 1. Write a test

Save as `test_refund.py`. This self-contained example uses deterministic turns;
replace the recording branch with your agent integration when ready.

```python
import pytest

from agentcontract.config import AssertionSpec


@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay":
        assert ac_replay_engine is not None
        run = ac_replay_engine.recorded_run
    else:
        # Deterministic stand-in for an agent: no SDK or credentials needed.
        ac_recorder.add_turn(role="user", content="Refund order 123")
        ac_recorder.add_turn(
            role="assistant",
            tool_calls=[{
                "id": "refund-1",
                "function": "process_refund",
                "arguments": {"order_id": "123", "amount": 49.99},
                "result": {"status": "refunded"},
            }],
        )
        ac_recorder.add_turn(
            role="assistant", content="Your refund of $49.99 has been processed."
        )
        run = ac_recorder.run

    contract = ac_check_contract(
        run,
        extra_assertions=[
            AssertionSpec(type="contains", target="final_response", value="$49.99"),
            AssertionSpec(
                type="called_with", target="tool:process_refund",
                schema={"order_id": "123", "amount": 49.99},
            ),
            AssertionSpec(type="called_count", target="tool:process_refund", value="1"),
        ],
    )
    assert contract.passed, [failure.message for failure in contract.failures()]
```

### 2. Record once

```bash
pytest --ac-record -k test_refund_flow
# Creates tests/scenarios/refund-eligible.agentrun.json
```

### 3. Replay in CI

```bash
pytest --ac-replay
# The example checks the saved trajectory without calling an SDK.
```

Checking `recorded_run` validates saved data; it does not execute your current agent.
`--ac-replay` does not automatically intercept SDK calls or block network access.
See the [pytest workflow guide](docs/pytest-workflow.md) for fixtures, configuration,
and explicit tool-stub integration.

## SDK Auto-Recording

Intercept real SDK calls instead of manually building turns:

```python
from agentcontract.recorder.interceptors import patch_openai

def test_with_real_agent(ac_recorder):
    client = openai.OpenAI()
    unpatch = patch_openai(client, ac_recorder)

    # Every chat.completions.create call is recorded automatically
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Refund order 123"}],
        tools=[...],
    )
    unpatch()
```

Works with Anthropic too:
```python
from agentcontract.recorder.interceptors import patch_anthropic

unpatch = patch_anthropic(client, ac_recorder)
```

## Framework Adapters

Drop-in recording for popular agent frameworks:

```python
# LangGraph
from agentcontract.adapters import record_graph
unpatch = record_graph(compiled_graph, recorder)
result = compiled_graph.invoke({"messages": [("user", "I need a refund")]})
unpatch()

# LlamaIndex
from agentcontract.adapters import record_agent
unpatch = record_agent(agent, recorder)
response = agent.chat("What's the refund policy?")
unpatch()

# OpenAI Agents SDK
from agentcontract.adapters import record_runner
unpatch = record_runner(recorder)
result = Runner.run_sync(agent, "Help with billing")
unpatch()
```

## Configuration

`agentcontract.yml` in your project root:

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
    - type: called_with
      target: "tool:process_refund"
      schema:
        order_id: "123"

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_eligibility, process_refund]

  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

Generate a starter config:
```bash
agentcontract init
```

## Assertions

| Type | What It Checks |
|------|---------------|
| `exact` | Exact string match |
| `contains` | Substring present |
| `regex` | Pattern match |
| `json_schema` | JSON Schema validation on tool args/results |
| `not_called` | Tool was NOT invoked |
| `called_with` | Tool called with specific arguments |
| `called_count` | Exact invocation count |

## Policies

| Policy | What It Enforces |
|--------|-----------------|
| `tool_allowlist` | Only listed tools may be called |
| `requires_confirmation` | Protected tools must follow user confirmation |

## Target Syntax

- `final_response` -- last assistant message
- `turn:N` -- specific turn by index
- `full_conversation` -- all turns concatenated
- `tool_call:function_name:arguments` -- tool call arguments
- `tool_call:function_name:result` -- tool call result

## CLI

```bash
agentcontract info cassette.agentrun.json       # Cassette summary
agentcontract validate cassette.agentrun.json   # Structure check
agentcontract init                               # Starter config
```

## Why Not VCR / pytest-recording?

VCR records **HTTP requests**. This records **agent decisions**.

- VCR: "did the HTTP request match?" -- brittle, breaks on any provider API change
- agentcontract: "did the agent call the right tools with the right args?" -- tests actual behavior

Your agent's contract is: given this input, it calls these tools in this order with these arguments. That's what you want to regression-test, not the HTTP layer underneath.

## How It Works

```
┌─────────┐     ┌──────────┐     ┌───────────────┐
│  pytest  │────▶│ Recorder │────▶│ .agentrun.json│
│ --record │     │          │     │  (cassette)   │
└─────────┘     └──────────┘     └───────┬───────┘
                                         │
┌─────────┐     ┌──────────┐             │
│  pytest  │────▶│  Replay  │◀────────────┘
│ --replay │     │  Engine  │
└─────────┘     └────┬─────┘
                     │
                ┌────▼─────┐
                │Assertion │──▶ pass / fail
                │ Engine   │
                └──────────┘
```

1. **Record**: Run your agent against real APIs. The recorder captures every turn, tool call, argument, and result into a `.agentrun.json` cassette.
2. **Replay**: Check the saved trajectory, or explicitly integrate the replay engine’s tool stubs into your agent. Offline execution depends on that integration.
3. **Assert**: The assertion engine checks contracts -- tool sequences, argument schemas, response content, policies.

## See Also

- **[coderace](https://github.com/mikiships/coderace)** -- Race coding agents (Claude Code, Codex, Aider, Gemini CLI) against each other on real tasks in your repo. Automated scoring with tests, lint, time, and lines changed.

## License

MIT
