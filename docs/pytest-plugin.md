# Pytest Plugin Reference

This page documents the public pytest plugin surface implemented in `src/agentcontract/plugin.py`.

## CLI Options

| Option | Default | Behavior |
| --- | --- | --- |
| `--ac-record` | `false` | Enables record mode. `ac_recorder` auto-saves a cassette after the test. |
| `--ac-replay` | `false` | Enables replay mode. `ac_replay_engine` loads the matching cassette. |
| `--ac-config PATH` | `None` | `ac_config` loads this file instead of discovery. |
| `--ac-scenarios PATH` | `None` | Overrides cassette directory for auto-save/load. Default directory is `tests/scenarios`. |

Mode precedence (`ac_mode`) is deterministic:
- `record` if `--ac-record` is set
- `replay` if `--ac-replay` is set and `--ac-record` is not set
- `live` otherwise

## Markers

The plugin registers two equivalent markers:
- `@pytest.mark.agentcontract("scenario-name")`
- `@pytest.mark.agent_scenario("scenario-name")`

`agent_scenario` also supports keyword form: `@pytest.mark.agent_scenario(name="scenario-name")`.

Scenario name resolution order used by `ac_recorder` and `ac_replay_engine`:
1. First positional arg on marker
2. `name=` kwarg on marker
3. Pytest test node name

## Fixtures

### `ac_config`
Returns `AgentContractConfig`.

Behavior:
- If `--ac-config` is provided, loads that exact path via `AgentContractConfig.from_file(...)`
- Otherwise runs config discovery (`AgentContractConfig.discover()`) by walking up from cwd for `agentcontract.yml`

### `ac_mode`
Returns one of `"record"`, `"replay"`, `"live"` based on CLI flags.

### `ac_recorder`
Returns a `Recorder` in a recording context and yields it for the test.

Post-test behavior:
- In `--ac-record` mode, auto-saves to `<scenarios_dir>/<scenario>.agentrun.json`
- `scenarios_dir` is `--ac-scenarios` if set, else `tests/scenarios`
- Save failures fail the test (`pytest.fail`)

### `ac_replay_engine`
Returns `ReplayEngine | None`.

Behavior:
- Returns `None` when not in replay mode
- In replay mode, loads `<scenarios_dir>/<scenario>.agentrun.json`
- If cassette is missing, test is skipped
- Load failures fail the test (`pytest.fail`)

### `ac_assert`
Returns a new `AssertionEngine`.

### `ac_check_contract`
Returns a callable:

```python
check(run: AgentRun, extra_assertions: list[Any] | None = None) -> ContractResult
```

Assertion merge order:
1. `ac_config.default_assertions`
2. Scenario override assertions from `ac_config.overrides[run.metadata.scenario]` (if any)
3. `extra_assertions` passed at call time

The merged assertions are evaluated with `ac_config.policies`.

## Minimal Usage Example

```python
import pytest


@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(ac_mode, ac_recorder, ac_replay_engine, ac_check_contract):
    if ac_mode == "record":
        run_my_agent(ac_recorder)
    elif ac_mode == "replay":
        assert ac_replay_engine is not None
        ac_replay_engine.run()

    result = ac_check_contract(ac_recorder.run)
    assert result.passed, result.failures()
```
