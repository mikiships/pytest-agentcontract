# Getting Started

`pytest-agentcontract` records agent trajectories as `.agentrun.json` cassettes,
then lets pytest replay or assert those trajectories deterministically.

## Install

```bash
pip install pytest-agentcontract
```

Optional SDK/framework dependencies are split by extra:

```bash
pip install pytest-agentcontract[openai]
pip install pytest-agentcontract[anthropic]
pip install pytest-agentcontract[langchain]
pip install pytest-agentcontract[llamaindex]
pip install pytest-agentcontract[all]
```

The OpenAI Agents SDK adapter imports the `agents` package at runtime. Install
`openai-agents` separately if you use that adapter.

## Minimal Pytest Test

The pytest plugin provides the recorder, replay engine, mode, configuration, and
contract-checking fixtures.

```python
import pytest

from agentcontract.config import AssertionSpec


@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay":
        assert ac_replay_engine is not None
        run = ac_replay_engine.recorded_run
    else:
        ac_recorder.add_turn(role="user", content="Refund order ORD-123")
        ac_recorder.add_turn(
            role="assistant",
            content="Your refund of $79.99 has been processed.",
            tool_calls=[
                {
                    "id": "tc_refund",
                    "function": "process_refund",
                    "arguments": {"order_id": "ORD-123", "amount": 79.99},
                    "result": {"success": True},
                }
            ],
        )
        run = ac_recorder.run

    result = ac_check_contract(
        run,
        extra_assertions=[
            AssertionSpec(type="contains", target="final_response", value="$79.99"),
            AssertionSpec(
                type="called_with",
                target="tool:process_refund",
                schema={"order_id": "ORD-123"},
            ),
        ],
    )
    assert result.passed, [failure.message for failure in result.failures()]
```

`ac_mode` is:

- `record` when pytest is run with `--ac-record`.
- `replay` when pytest is run with `--ac-replay`.
- `live` when neither flag is provided.

The `ac_recorder` fixture already wraps the test in a recording context. Add
turns directly with `ac_recorder.add_turn(...)`; the summary is computed when
the test fixture exits.

## Record Once

```bash
pytest --ac-record -k test_refund_flow
```

When the test is marked with `@pytest.mark.agentcontract("refund-eligible")`,
record mode writes:

```text
tests/scenarios/refund-eligible.agentrun.json
```

If the marker is omitted, the test name is used as the scenario name. The alias
`@pytest.mark.agent_scenario("refund-eligible")` is also supported.

Use `--ac-scenarios` to write to a different directory:

```bash
pytest --ac-record --ac-scenarios examples/customer_support/scenarios
```

## Replay

```bash
pytest --ac-replay
```

Replay mode loads the matching cassette from the scenarios directory. If the
cassette is missing, the fixture skips the test. In replay mode, use
`ac_replay_engine.recorded_run` when your test only needs to assert the recorded
trajectory.

For agent code that calls tools during replay, use the recorded tool stub:

```python
result = ac_replay_engine.tool_stub.get_result(
    "lookup_order",
    {"order_id": "ORD-123"},
)
```

`ToolStub.get_result()` returns recorded results in call order. If arguments are
provided, they must exactly match the recorded arguments for that call.

## Customer Support Example

From the repository root:

```bash
uv run pytest examples/customer_support/test_support.py -v
uv run pytest examples/customer_support/test_support.py --ac-replay \
  --ac-scenarios examples/customer_support/scenarios -v
uv run pytest examples/customer_support/test_support.py --ac-record \
  --ac-scenarios examples/customer_support/scenarios -v
```

The example is a deterministic refund agent. Its README explains each scenario:
[`examples/customer_support/README.md`](../examples/customer_support/README.md).
