# Getting Started

pytest-agentcontract records agent trajectories as `.agentrun.json` cassettes, then replays and checks those trajectories in pytest. A trajectory is a sequence of turns plus any tool calls, arguments, results, timing, and token usage that were captured.

## Install

```bash
pip install pytest-agentcontract
```

Optional integrations can be installed with extras:

```bash
pip install pytest-agentcontract[openai]      # OpenAI SDK interceptor
pip install pytest-agentcontract[anthropic]   # Anthropic SDK interceptor
pip install pytest-agentcontract[langchain]   # LangChain Core message types
pip install pytest-agentcontract[llamaindex]  # LlamaIndex Core response types
pip install pytest-agentcontract[all]         # All declared optional dependencies above
```

The adapter modules for LangGraph, LlamaIndex, and the OpenAI Agents SDK are included in the package. Your application still needs to install the framework it uses. For example, install `langgraph` for a LangGraph app and install `openai-agents` for the OpenAI Agents SDK adapter.

## Minimal Pytest Flow

Mark each contract test with a scenario name. The marker name becomes the cassette filename:

```python
import pytest


@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_recorder, ac_mode, ac_replay_engine, ac_check_contract):
    if ac_mode == "replay":
        assert ac_replay_engine is not None
        run = ac_replay_engine.recorded_run
    else:
        turns = run_my_agent("I'd like a refund for order ORD-123 please")
        for turn in turns:
            ac_recorder.add_turn(
                role=turn["role"],
                content=turn.get("content"),
                tool_calls=turn.get("tool_calls"),
            )
        run = ac_recorder.run

    contract = ac_check_contract(run)
    assert contract.passed, [failure.message for failure in contract.failures()]
```

`@pytest.mark.agent_scenario(name="refund-eligible")` is also accepted as an alias.

## Record

```bash
pytest --ac-record -k test_refund_flow
```

In record mode, the `ac_recorder` fixture saves the cassette after the test exits. By default it writes to `tests/scenarios/<scenario>.agentrun.json`.

Use `--ac-scenarios` to choose a different cassette directory:

```bash
pytest --ac-record --ac-scenarios examples/customer_support/scenarios
```

## Replay

```bash
pytest --ac-replay
```

In replay mode, `ac_replay_engine` loads the matching cassette from the scenario directory. If the cassette does not exist, the fixture skips the test. The replay engine exposes:

| API | Use |
| --- | --- |
| `recorded_run` | Access the loaded `AgentRun` for assertions. |
| `tool_stub.get_result(function, arguments=None)` | Return recorded tool results in call order. |
| `finish(actual_turns=None)` | Check stub consumption or compare actual turns with the recording. |

The replay engine does not expose a method named `run`. Your test either asserts the recorded run directly or wires `tool_stub` into your own agent loop.

## Live Mode

When neither `--ac-record` nor `--ac-replay` is provided, `ac_mode` is `"live"`. The recorder still exists, but the pytest fixture does not auto-save a cassette. This is useful for running contract assertions against deterministic local examples.

## Fixtures

| Fixture | Provides |
| --- | --- |
| `ac_config` | Parsed `agentcontract.yml`, or defaults when no config is found. |
| `ac_mode` | `"record"`, `"replay"`, or `"live"`. |
| `ac_recorder` | A `Recorder` for manual turn capture and auto-save in record mode. |
| `ac_replay_engine` | A `ReplayEngine` in replay mode, otherwise `None`. |
| `ac_assert` | A raw `AssertionEngine`. |
| `ac_check_contract` | A helper that applies default assertions, scenario overrides, extra assertions, and policies. |

## Manual Recording

Manual recording is the most explicit way to capture complete tool results:

```python
from agentcontract.recorder.core import Recorder


recorder = Recorder(scenario="refund-eligible")
with recorder.recording():
    recorder.add_turn(role="user", content="Refund order ORD-123")
    recorder.add_turn(
        role="assistant",
        content="Let me look up that order.",
        tool_calls=[
            {
                "id": "tc_lookup",
                "function": "lookup_order",
                "arguments": {"order_id": "ORD-123"},
                "result": {"status": "delivered", "total": 79.99},
            }
        ],
    )

recorder.save("tests/scenarios/refund-eligible.agentrun.json")
```

`role` must be one of `system`, `user`, `assistant`, or `tool`.

## Customer Support Example

The repository includes a deterministic customer support example:

```bash
pytest examples/customer_support/test_support.py -v
pytest examples/customer_support/test_support.py --ac-replay \
  --ac-scenarios examples/customer_support/scenarios -v
pytest examples/customer_support/test_support.py --ac-record \
  --ac-scenarios examples/customer_support/scenarios -v
```

The example agent in `examples/customer_support/agent.py` emits turns manually. The tests in `examples/customer_support/test_support.py` show default contract checks, inline assertions, and policy enforcement.

## Useful CLI Commands

```bash
agentcontract info tests/scenarios/refund-eligible.agentrun.json
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
agentcontract init
```

`agentcontract init` creates `agentcontract.yml` in the current directory and fails if that file already exists.
