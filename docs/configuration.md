# Configuration

pytest-agentcontract discovers `agentcontract.yml` by walking up from the
current working directory. Pass `--ac-config path/to/agentcontract.yml` to load a
specific file.

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
      value: refund

overrides:
  refund-not-delivered:
    assertions:
      - type: not_called
        target: tool:process_refund

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]

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

| YAML field | `AgentContractConfig` field | Default |
| --- | --- | --- |
| `version` | `version` | `"1"` |
| `scenarios.include` | `scenario_include` | `["tests/scenarios/**/*.agentrun.json"]` |
| `scenarios.exclude` | `scenario_exclude` | `[]` |
| `replay.model` | `replay.model` | `""` |
| `replay.seed` | `replay.seed` | `42` |
| `replay.stub_tools` | `replay.stub_tools` | `true` |
| `replay.concurrency` | `replay.concurrency` | `5` |
| `defaults.assertions` | `default_assertions` | `[]` |
| `overrides.<scenario>.assertions` | `overrides[scenario].assertions` | `{}` |
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

The current pytest fixture uses config assertions and policies when
`ac_check_contract(...)` runs. Scenario include/exclude, budgets, baseline, and
reporting fields are parsed into the config object for runners or reporting code
that consume `AgentContractConfig`.

## Assertions

Config assertions use the same shape as `AssertionSpec`:

```yaml
defaults:
  assertions:
    - type: contains
      target: final_response
      value: refund
    - type: json_schema
      target: tool_call:process_refund:arguments
      schema:
        type: object
        required: [order_id, amount, method]
```

Supported assertion fields are:

- `type`
- `target`
- `value`
- `threshold`
- `prompt`
- `schema`
- `judge_model`
- `tools`
- `block`

Only the implemented assertion types are enforced by `AssertionEngine`; unknown
types fail closed.

## Overrides

Overrides are keyed by scenario name. When `ac_check_contract(...)` evaluates a
run, it appends assertions from `overrides.<scenario>.assertions` after the
defaults.

```yaml
overrides:
  refund-not-delivered:
    assertions:
      - type: not_called
        target: tool:process_refund
```

## Policies

Policies use the `PolicySpec` fields:

- `name`
- `type`
- `target`
- `tools`
- `block`

The implemented policy types are `tool_allowlist` and `requires_confirmation`.

```yaml
policies:
  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

## Type Coercion

The config loader is permissive around YAML input:

- Missing or `null` sections fall back to defaults.
- Scalar values are coerced to strings, booleans, integers, or floats according
  to the target field.
- `policies[].tools` and `policies[].block` become empty lists when omitted or
  `null`.
