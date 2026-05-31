# Getting Started

## Install

```bash
pip install pytest-agentcontract
```

Install SDK extras when you want automatic recording around provider clients:

```bash
pip install pytest-agentcontract[openai]
pip install pytest-agentcontract[anthropic]
pip install pytest-agentcontract[all]
```

Framework adapters are imported lazily from `agentcontract.adapters`. Install the
framework you plan to run, such as LangGraph, LlamaIndex, or `openai-agents`, in
the same environment as your tests.

## Pytest Marker

Mark each contract test with a scenario name. The scenario becomes the cassette
file name.

```python
import pytest


@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder):
    ...
```

`@pytest.mark.agent_scenario("refund-eligible")` is also registered as an alias.
If neither marker is present, pytest-agentcontract uses the pytest test name as
the scenario.

## Core Fixtures

| Fixture | Purpose |
| --- | --- |
| `ac_config` | Parsed `agentcontract.yml`, or defaults when no config is found. |
| `ac_mode` | One of `"record"`, `"replay"`, or `"live"` based on pytest options. |
| `ac_recorder` | A `Recorder` bound to the scenario; auto-saves only with `--ac-record`. |
| `ac_replay_engine` | A `ReplayEngine` loaded from the matching cassette in replay mode, otherwise `None`. |
| `ac_assert` | An `AssertionEngine` instance. |
| `ac_check_contract` | Callable that merges config defaults, scenario overrides, extra assertions, and policies. |

## Record And Replay Commands

```bash
# Record one scenario against live code and APIs.
pytest examples/customer_support/test_support.py --ac-record -k refund_happy_path

# Replay all contract tests from saved cassettes.
pytest examples/customer_support/test_support.py --ac-replay

# Use a custom cassette directory.
pytest --ac-record --ac-scenarios tests/scenarios
pytest --ac-replay --ac-scenarios tests/scenarios

# Use a specific config file.
pytest --ac-config path/to/agentcontract.yml
```

Record mode writes `tests/scenarios/<scenario>.agentrun.json` by default.
`--ac-scenarios` changes that directory.

## Compact Example

This example follows `examples/customer_support/test_support.py`: run the agent
when recording or running live, load the recorded run when replaying, then assert
the contract.

```python
import pytest

from agentcontract.config import AssertionSpec
from agent import run_support_agent


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

For lower-level replay tests, use `ac_replay_engine.tool_stub.get_result(...)`
to feed recorded tool results into your own agent loop, then call
`ac_replay_engine.finish(actual_turns)` to compare actual turns with the
recorded trajectory.
