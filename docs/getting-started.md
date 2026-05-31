# Getting Started

## Install

```bash
pip install pytest-agentcontract
```

Install SDK extras only when you want automatic SDK interception:

```bash
pip install pytest-agentcontract[openai]
pip install pytest-agentcontract[anthropic]
pip install pytest-agentcontract[all]
```

Framework adapters for LangGraph, LlamaIndex, and the OpenAI Agents SDK are included in the base package. The corresponding framework packages must still be installed in your project.

## Mark A Scenario

Use the `agentcontract` pytest marker to name the cassette for a test:

```python
import pytest

@pytest.mark.agentcontract("refund-eligible")
def test_refund_happy_path(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    ...
```

`@pytest.mark.agent_scenario(name)` is also registered as an alias.

## Core Fixtures

- `ac_mode`: returns `"record"`, `"replay"`, or `"live"` based on pytest options.
- `ac_recorder`: records turns and auto-saves in record mode.
- `ac_replay_engine`: loads the matching cassette in replay mode, or skips when it is missing.
- `ac_check_contract`: checks a run against default config assertions, per-scenario overrides, optional extra assertions, and policies.
- `ac_config`: returns the discovered or explicitly configured `AgentContractConfig`.
- `ac_assert`: returns an `AssertionEngine` instance.

## Record And Replay

Record one scenario:

```bash
pytest examples/customer_support/test_support.py --ac-record -k refund_happy_path
```

Replay existing cassettes:

```bash
pytest examples/customer_support/test_support.py --ac-replay
```

By default, pytest fixtures save and load cassettes under `tests/scenarios/<scenario>.agentrun.json`. Override that directory with `--ac-scenarios`:

```bash
pytest examples/customer_support/test_support.py --ac-record --ac-scenarios examples/customer_support/scenarios
pytest examples/customer_support/test_support.py --ac-replay --ac-scenarios examples/customer_support/scenarios
```

## End-To-End Example

This pattern mirrors `examples/customer_support/test_support.py`:

```python
import pytest

from agentcontract.config import AssertionSpec
from examples.customer_support.agent import run_support_agent


@pytest.mark.agentcontract("refund-eligible")
def test_refund_happy_path(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay" and ac_replay_engine is not None:
        run = ac_replay_engine.recorded_run
    else:
        turns = run_support_agent("I'd like a refund for order ORD-123 please")
        for turn in turns:
            ac_recorder.add_turn(
                role=turn["role"],
                content=turn.get("content"),
                tool_calls=turn.get("tool_calls"),
            )
        run = ac_recorder.run

    result = ac_check_contract(
        run,
        extra_assertions=[
            AssertionSpec(type="contains", target="final_response", value="$79.99"),
            AssertionSpec(
                type="json_schema",
                target="tool_call:process_refund:arguments",
                schema={
                    "type": "object",
                    "required": ["order_id", "amount", "method"],
                },
            ),
        ],
    )
    assert result.passed, [failure.message for failure in result.failures()]
```

Run it in record mode once, commit the resulting `.agentrun.json`, and use replay mode in CI.
