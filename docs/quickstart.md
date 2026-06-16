# Quickstart

This guide creates one contract test, records its trajectory, and replays it from a cassette.

## Install

```bash
pip install pytest-agentcontract
```

Install SDK extras only when you want automatic SDK interception:

```bash
pip install "pytest-agentcontract[openai]"
pip install "pytest-agentcontract[anthropic]"
pip install "pytest-agentcontract[all]"
```

Framework adapters for LangGraph, LlamaIndex, and the OpenAI Agents SDK live in the base package. You still need the framework package installed in your application if you use that adapter.

## Write A Test

Mark each contract test with a scenario name. The scenario name controls the default cassette path:

```python
import pytest

from agentcontract.config import AssertionSpec


def run_refund_agent(order_id: str):
    order = {"id": order_id, "status": "delivered", "total": 49.99}
    refund = {"success": True, "refund_id": "ref_001"}
    return [
        {"role": "user", "content": f"Refund order {order_id}"},
        {
            "role": "assistant",
            "content": f"Looking up order {order_id}.",
            "tool_calls": [
                {
                    "id": "tc_lookup",
                    "function": "lookup_order",
                    "arguments": {"order_id": order_id},
                    "result": order,
                }
            ],
        },
        {"role": "user", "content": "Yes, please process it."},
        {
            "role": "assistant",
            "content": "Your refund of $49.99 has been processed.",
            "tool_calls": [
                {
                    "id": "tc_refund",
                    "function": "process_refund",
                    "arguments": {"order_id": order_id, "amount": 49.99},
                    "result": refund,
                }
            ],
        },
    ]


@pytest.mark.agentcontract("refund-eligible")
def test_refund_contract(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay" and ac_replay_engine is not None:
        run = ac_replay_engine.recorded_run
    else:
        for turn in run_refund_agent("ORD-123"):
            ac_recorder.add_turn(
                role=turn["role"],
                content=turn.get("content"),
                tool_calls=turn.get("tool_calls"),
            )
        run = ac_recorder.run

    result = ac_check_contract(
        run,
        extra_assertions=[
            AssertionSpec(type="contains", target="final_response", value="$49.99"),
            AssertionSpec(
                type="called_with",
                target="tool:process_refund",
                schema={"order_id": "ORD-123"},
            ),
        ],
    )
    assert result.passed, [failure.message for failure in result.failures()]
```

`ac_mode` is `"record"` with `--ac-record`, `"replay"` with `--ac-replay`, and `"live"` otherwise. The `agentcontract` marker can also be written as `@pytest.mark.agent_scenario("refund-eligible")`.

## Record Once

```bash
pytest tests/test_refund.py --ac-record -k refund_contract
```

By default, the `ac_recorder` fixture saves:

```text
tests/scenarios/refund-eligible.agentrun.json
```

Use `--ac-scenarios path/to/scenarios` to override that directory for both record and replay mode.

## Replay

```bash
pytest tests/test_refund.py --ac-replay -k refund_contract
```

Replay mode loads the cassette matching the marker name. If no cassette exists, the fixture skips the test with the missing path. To validate the cassette file can be parsed:

```bash
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
agentcontract info tests/scenarios/refund-eligible.agentrun.json
```

The replay result is deterministic because the test checks recorded turns and tool calls instead of calling a live model.
