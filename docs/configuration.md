# Configuration

Configuration lives in `agentcontract.yml`. The pytest fixture loads it from `--ac-config` when provided, otherwise `AgentContractConfig.discover()` walks up from the current working directory looking for `agentcontract.yml`. If no file is found, defaults are used.

## Starter Config

Create a starter file:

```bash
agentcontract init
```

The command writes `agentcontract.yml` in the current directory and exits with an error if the file already exists.

## Complete Shape

```yaml
version: "1"

scenarios:
  include: ["tests/scenarios/**/*.agentrun.json"]
  exclude: []

replay:
  model: ""
  seed: 42
  stub_tools: true
  concurrency: 5

defaults:
  assertions:
    - type: contains
      target: final_response
      value: "refund"
    - type: called_with
      target: "tool:process_refund"
      schema:
        order_id: "ORD-123"

overrides:
  refund-not-delivered:
    assertions:
      - type: not_called
        target: "tool:process_refund"

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]
  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]

thresholds:
  suite_pass_rate: 1.0

budgets:
  per_scenario:
    max_cost_usd: 0.05
    max_latency_ms: 10000
    max_turns: 15
  suite:
    max_cost_usd: 2.0

baseline:
  branch: main
  show_deltas: true

reporting:
  github_comment: true
  artifact_path: agentci-results/
```

## Supported Fields

| YAML path | Parsed as | Default |
| --- | --- | --- |
| `version` | Config schema version string. | `"1"` |
| `scenarios.include` | Parsed cassette glob patterns; current pytest fixtures do not use them to locate a marked test. | `["tests/scenarios/**/*.agentrun.json"]` |
| `scenarios.exclude` | Parsed cassette globs to exclude; current pytest fixtures do not use them to locate a marked test. | `[]` |
| `replay.model` | Parsed model metadata; the current `ReplayEngine` does not select a model from config. | `""` |
| `replay.seed` | Parsed seed metadata; replay itself is driven by the cassette. | `42` |
| `replay.stub_tools` | Parsed setting; tests still call `tool_stub` explicitly when they want tool results. | `true` |
| `replay.concurrency` | Parsed concurrency value; current pytest fixtures do not schedule concurrent replay. | `5` |
| `defaults.assertions` | Assertions applied to every `ac_check_contract()` call. | `[]` |
| `overrides.<scenario>.assertions` | Assertions appended for a named scenario. | `{}` |
| `policies` | Policies applied by `ac_check_contract()`. | `[]` |
| `thresholds.suite_pass_rate` | Parsed suite threshold; not enforced by the assertion engine. | `1.0` |
| `budgets.per_scenario.max_cost_usd` | Parsed per-scenario cost budget; not enforced by the assertion engine. | `0.05` |
| `budgets.per_scenario.max_latency_ms` | Parsed per-scenario latency budget; not enforced by the assertion engine. | `10000` |
| `budgets.per_scenario.max_turns` | Parsed per-scenario turn budget; not enforced by the assertion engine. | `15` |
| `budgets.suite.max_cost_usd` | Parsed suite cost budget; not enforced by the assertion engine. | `2.0` |
| `baseline.branch` | Parsed baseline branch name. | `"main"` |
| `baseline.show_deltas` | Parsed delta-display setting. | `true` |
| `reporting.github_comment` | Parsed GitHub comment setting; no comment is posted by the current pytest fixtures. | `true` |
| `reporting.artifact_path` | Parsed report artifact path; no artifact is written by the current pytest fixtures. | `"agentci-results/"` |

The current pytest fixtures use `--ac-scenarios` or `tests/scenarios` to locate a matching cassette for a marked test. The `scenarios.include` and `scenarios.exclude` fields are parsed into `ac_config` for callers that inspect config directly, but they are not used to collect pytest tests.

Budget, threshold, baseline, and reporting values are parsed and exposed on `ac_config`. The current pytest fixtures and assertion engine do not enforce budgets, compare baselines, write artifacts, or post GitHub comments by themselves.

## Scenario Names And Discovery

The scenario name comes from the nearest marker:

```python
@pytest.mark.agentcontract("refund-eligible")
def test_refund_flow(...):
    ...
```

The alias form is:

```python
@pytest.mark.agent_scenario(name="refund-eligible")
def test_refund_flow(...):
    ...
```

In record mode, the cassette path is `<scenario_dir>/<scenario>.agentrun.json`. In replay mode, the same path is loaded.

## Assertions And Overrides

`ac_check_contract(run, extra_assertions=None)` merges assertions in this order:

1. `defaults.assertions`
2. `overrides.<run.metadata.scenario>.assertions`
3. `extra_assertions` passed by the test

Policies from `policies` are then checked against the same run.

Assertion entries support the fields defined by `AssertionSpec`: `type`, `target`, `value`, `threshold`, `prompt`, `schema`, `judge_model`, `tools`, and `block`. The current assertion engine uses `type`, `target`, `value`, and `schema`.

Policy entries support `name`, `type`, `target`, `tools`, and `block`. The current policy engine uses `name`, `type`, and `tools`.

See [Assertions And Policies](assertions-and-policies.md) for supported assertion and policy types.

## Replay Settings

The replay engine stubs tool results from recorded cassettes:

```python
result = ac_replay_engine.tool_stub.get_result(
    "lookup_order",
    {"order_id": "ORD-123"},
)
```

`stub_tools`, `model`, `seed`, and `concurrency` are configuration fields, but the low-level `ReplayEngine` API is explicit: tests call `tool_stub.get_result()` and `finish()` directly.
