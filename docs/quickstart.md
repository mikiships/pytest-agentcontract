# Quickstart

## Install

```bash
pip install pytest-agentcontract
```

Optional SDK interceptors can be installed with extras:

```bash
pip install pytest-agentcontract[openai]
pip install pytest-agentcontract[anthropic]
pip install pytest-agentcontract[all]
```

Framework adapters for LangGraph, LlamaIndex, and the OpenAI Agents SDK are
included in the base package. The framework itself must still be installed by
your project.

## Pytest Options

pytest-agentcontract registers these pytest options:

| Option | Purpose |
| --- | --- |
| `--ac-record` | Run tests in record mode and save cassettes after each marked test. |
| `--ac-replay` | Load matching cassettes and replay without live model or tool calls. |
| `--ac-config PATH` | Load a specific `agentcontract.yml`. |
| `--ac-scenarios PATH` | Override the cassette directory used by fixtures. |

If neither `--ac-record` nor `--ac-replay` is set, `ac_mode` is `"live"`.

## Markers and Fixtures

Mark each contract test with a scenario name:

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(...):
    ...
```

`@pytest.mark.agent_scenario("refund-eligible")` is also accepted as an alias.
If no marker is present, the test name is used as the scenario name.

Core fixtures:

| Fixture | Purpose |
| --- | --- |
| `ac_mode` | `"record"`, `"replay"`, or `"live"`. |
| `ac_recorder` | A `Recorder` that captures turns and auto-saves in record mode. |
| `ac_replay_engine` | A `ReplayEngine` loaded from the matching cassette in replay mode. |
| `ac_config` | The discovered or explicitly configured `AgentContractConfig`. |
| `ac_assert` | An `AssertionEngine`. |
| `ac_check_contract` | Callable that merges config assertions, overrides, policies, and extra assertions. |

## Example

This mirrors the customer-support example in `examples/customer_support`.

```python
import pytest

from agentcontract.config import AssertionSpec
from my_agent import run_support_agent


@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
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

## Record Once

```bash
pytest examples/customer_support/test_support.py --ac-record -v
```

By default, the fixture writes to:

```text
tests/scenarios/<scenario>.agentrun.json
```

Use `--ac-scenarios` when your project stores cassettes somewhere else:

```bash
pytest --ac-record --ac-scenarios examples/customer_support/scenarios
```

## Replay in CI

```bash
pytest --ac-replay
```

Replay mode loads the cassette for each marked scenario. If a cassette is
missing, the fixture skips that test with the missing path in the skip reason.
