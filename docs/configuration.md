# Configuration Reference

This page documents `agentcontract.yml` as parsed by `AgentContractConfig` in `src/agentcontract/config.py`.

## Discovery and Explicit Path

- With `--ac-config PATH`, the plugin loads that file directly.
- Without `--ac-config`, discovery walks up from current working directory for `agentcontract.yml`.
- If no file is found, built-in defaults are used.

## Full Structure and Defaults

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
  assertions: []

overrides:
  scenario-name:
    assertions: []

policies: []

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
  branch: "main"
  show_deltas: true

reporting:
  github_comment: true
  artifact_path: "agentci-results/"
```

## Top-Level Fields

| YAML Path | Parsed Field | Default |
| --- | --- | --- |
| `version` | `config.version` | `"1"` |
| `scenarios.include` | `config.scenario_include` | `['tests/scenarios/**/*.agentrun.json']` |
| `scenarios.exclude` | `config.scenario_exclude` | `[]` |
| `replay.model` | `config.replay.model` | `""` |
| `replay.seed` | `config.replay.seed` | `42` |
| `replay.stub_tools` | `config.replay.stub_tools` | `true` |
| `replay.concurrency` | `config.replay.concurrency` | `5` |
| `defaults.assertions` | `config.default_assertions` | `[]` |
| `overrides.<scenario>.assertions` | `config.overrides[scenario].assertions` | `[]` |
| `policies` | `config.policies` | `[]` |
| `thresholds.suite_pass_rate` | `config.suite_pass_rate` | `1.0` |
| `budgets.per_scenario.max_cost_usd` | `config.per_scenario_budget.max_cost_usd` | `0.05` |
| `budgets.per_scenario.max_latency_ms` | `config.per_scenario_budget.max_latency_ms` | `10000` |
| `budgets.per_scenario.max_turns` | `config.per_scenario_budget.max_turns` | `15` |
| `budgets.suite.max_cost_usd` | `config.suite_budget_usd` | `2.0` |
| `baseline.branch` | `config.baseline_branch` | `"main"` |
| `baseline.show_deltas` | `config.show_deltas` | `true` |
| `reporting.github_comment` | `config.github_comment` | `true` |
| `reporting.artifact_path` | `config.artifact_path` | `"agentci-results/"` |

## Assertion Entries

`defaults.assertions` and `overrides.<scenario>.assertions` parse into `AssertionSpec`.

Required keys:
- `type`

Optional keys:
- `target` (default `""`)
- `value`
- `threshold`
- `prompt`
- `schema`
- `judge_model`
- `tools`
- `block`

Built-in assertion types in the engine:
- `exact`
- `contains`
- `regex`
- `json_schema`
- `not_called`
- `called_with`
- `called_count`

## Policy Entries

`policies` parse into `PolicySpec`.

Required keys:
- `name`
- `type`

Optional keys:
- `target` (default `""`)
- `tools` (defaults to `[]`)
- `block` (defaults to `[]`)

Built-in policy types in the engine:
- `tool_allowlist`
- `requires_confirmation`

## Coercion and Defaulting Behavior

The parser is intentionally tolerant:
- Non-object sections (`scenarios`, `replay`, `defaults`, `thresholds`, `budgets`, `baseline`, `reporting`) are treated as empty objects.
- List fields use defaults when value is not a list.
- String fields convert non-null values with `str(...)`.
- Boolean fields accept YAML booleans plus string/int-like values (`true/false`, `1/0`, `yes/no`, `on/off`). Invalid values fall back to defaults.
- Integer fields reject bools and non-integer floats; invalid values fall back to defaults.
- Float fields use `float(...)` when possible; invalid values fall back to defaults.
- Policy `tools` and `block` entries are coerced to `list[str]`.
- Unknown keys are ignored.

If required assertion/policy keys are missing (`type` for assertions, `name`/`type` for policies), parsing raises a `KeyError`.
