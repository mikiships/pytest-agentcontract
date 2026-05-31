# Configuration

pytest-agentcontract discovers `agentcontract.yml` by walking up from the current directory. Pass `--ac-config path/to/agentcontract.yml` to use a specific file.

Generate a starter file:

```bash
agentcontract init
```

## Example

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
      target: tool:process_refund
      schema:
        order_id: "ORD-123"
        method: "original"

overrides:
  refund-not-delivered:
    assertions:
      - type: not_called
        target: tool:process_refund

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

## Parsed Fields

| YAML path | Python field | Default |
| --- | --- | --- |
| `version` | `AgentContractConfig.version` | `"1"` |
| `scenarios.include` | `scenario_include` | `["tests/scenarios/**/*.agentrun.json"]` |
| `scenarios.exclude` | `scenario_exclude` | `[]` |
| `replay.model` | `replay.model` | `""` |
| `replay.seed` | `replay.seed` | `42` |
| `replay.stub_tools` | `replay.stub_tools` | `true` |
| `replay.concurrency` | `replay.concurrency` | `5` |
| `defaults.assertions` | `default_assertions` | `[]` |
| `overrides.<scenario>.assertions` | `overrides` | `{}` |
| `policies` | `policies` | `[]` |
| `thresholds.suite_pass_rate` | `suite_pass_rate` | `1.0` |
| `budgets.per_scenario.max_cost_usd` | `per_scenario_budget.max_cost_usd` | `0.05` |
| `budgets.per_scenario.max_latency_ms` | `per_scenario_budget.max_latency_ms` | `10000` |
| `budgets.per_scenario.max_turns` | `per_scenario_budget.max_turns` | `15` |
| `budgets.suite.max_cost_usd` | `suite_budget_usd` | `2.0` |
| `baseline.branch` | `baseline_branch` | `"main"` |
| `baseline.show_deltas` | `show_deltas` | `true` |
| `reporting.github_comment` | `github_comment` | `true` |
| `reporting.artifact_path` | `artifact_path` | `"agentci-results/"` |

The current pytest contract checker uses `defaults.assertions`, matching `overrides.<scenario>.assertions`, and `policies` when `ac_check_contract()` runs. Other parsed fields are available on `AgentContractConfig` for replay tooling, reporting, and downstream integrations.

## Assertions

Assertion entries are parsed into `AssertionSpec`:

```yaml
- type: contains
  target: final_response
  value: "refund"
```

Supported keys are:

- `type`
- `target`
- `value`
- `threshold`
- `prompt`
- `schema`
- `judge_model`
- `tools`
- `block`

Unknown assertion types fail closed in the built-in assertion engine.

## Policies

Policy entries are parsed into `PolicySpec`:

```yaml
- name: allowed-tools
  type: tool_allowlist
  tools: [lookup_order, check_refund_eligibility, process_refund]
  block: []
```

Supported keys are `name`, `type`, `target`, `tools`, and `block`. The built-in policy types are `tool_allowlist` and `requires_confirmation`.

## Scenario Paths

`scenarios.include` and `scenarios.exclude` are parsed into config, while the pytest fixtures use the `--ac-scenarios` option to choose the directory for saving and loading cassettes during record/replay. Without `--ac-scenarios`, fixtures use `tests/scenarios`.
