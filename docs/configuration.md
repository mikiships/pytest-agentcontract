# Configuration Reference

`agentcontract.yml` is loaded by `AgentContractConfig.from_file()` or `AgentContractConfig.discover()`. The parser accepts more fields than the built-in pytest plugin currently enforces, so this page separates accepted keys from the keys the plugin actively consumes today.

## Where It Is Loaded

- `ac_config` loads the file passed with `--ac-config`.
- If `--ac-config` is not set, discovery walks up from the current working directory looking for `agentcontract.yml`.
- `agentcontract init` writes a starter file in the current directory.

## Example

This example matches the current parser and the customer-support scenarios shipped in this repository:

```yaml
version: "1"

scenarios:
  include:
    - examples/customer_support/scenarios/**/*.agentrun.json
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

overrides:
  refund-eligible:
    assertions:
      - type: called_with
        target: tool:process_refund
        schema:
          order_id: ORD-123
          amount: 79.99
  refund-not-delivered:
    assertions:
      - type: not_called
        target: tool:process_refund
      - type: contains
        target: final_response
        value: "isn't eligible"

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools:
      - lookup_order
      - check_refund_eligibility
      - process_refund
  - name: confirm-before-refund
    type: requires_confirmation
    tools:
      - process_refund

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

## Top-Level Keys

| Key | Default | Notes |
| --- | --- | --- |
| `version` | `"1"` | Parsed as a string. |
| `scenarios.include` | `["tests/scenarios/**/*.agentrun.json"]` | Parsed and exposed as `ac_config.scenario_include`. |
| `scenarios.exclude` | `[]` | Parsed and exposed as `ac_config.scenario_exclude`. |
| `replay` | see below | Parsed into `ReplayConfig`. |
| `defaults.assertions` | `[]` | Used by `ac_check_contract`. |
| `overrides` | `{}` | Used by `ac_check_contract` when `run.metadata.scenario` matches a key. |
| `policies` | `[]` | Used by `ac_check_contract`. |
| `thresholds.suite_pass_rate` | `1.0` | Parsed only. Not enforced by the built-in plugin. |
| `budgets` | see below | Parsed only. Not enforced by the built-in plugin. |
| `baseline` | see below | Parsed only. Not enforced by the built-in plugin. |
| `reporting` | see below | Parsed only. Not enforced by the built-in plugin. |

## `scenarios`

```yaml
scenarios:
  include:
    - tests/scenarios/**/*.agentrun.json
  exclude: []
```

- `include` and `exclude` are accepted lists of glob-like strings.
- The parser preserves both lists on `AgentContractConfig`.
- The current pytest plugin does not use these keys to choose cassette paths for fixtures. Fixture save/load paths are controlled by the scenario marker and `--ac-scenarios`.

## `replay`

```yaml
replay:
  model: ""
  seed: 42
  stub_tools: true
  concurrency: 5
```

| Key | Default | Notes |
| --- | --- | --- |
| `model` | `""` | Parsed as a string. |
| `seed` | `42` | Parsed as an optional integer, but the loader defaults nulls back to `42`. |
| `stub_tools` | `true` | Parsed as a boolean. |
| `concurrency` | `5` | Parsed as an integer. |

The built-in replay fixture currently loads the recorded run into `ReplayEngine`, but it does not automatically read these values back out to change replay behavior.

## `defaults.assertions`

Default assertions are applied by `ac_check_contract(run)` before any per-call `extra_assertions`.

Supported assertion types in the built-in `AssertionEngine`:

| Type | Required Fields | Behavior |
| --- | --- | --- |
| `exact` | `target`, `value` | Exact string/value comparison against a resolved target. |
| `contains` | `target`, `value` | Substring containment check. |
| `regex` | `target`, `value` | Regular expression search against the resolved target. |
| `json_schema` | `target`, `schema` | Validates a target against JSON Schema. |
| `not_called` | `target` | Passes when a tool was not called. |
| `called_with` | `target`, `schema` | Passes when a matching tool call contains the key/value pairs in `schema`. |
| `called_count` | `target`, `value` | Passes when a tool was called exactly `value` times. |

Accepted assertion fields:

| Field | Parsed | Used By Built-In Engine |
| --- | --- | --- |
| `type` | yes | yes |
| `target` | yes | yes |
| `value` | yes | yes |
| `schema` | yes | yes |
| `threshold` | yes | no |
| `prompt` | yes | no |
| `judge_model` | yes | no |
| `tools` | yes | no |
| `block` | yes | no |

### Target Syntax

The built-in engine resolves these target strings:

- `final_response`
- `full_conversation`
- `turn:N`
- `tool_call:<function_name>:arguments`
- `tool_call:<function_name>:result`

For `not_called`, `called_with`, and `called_count`, the target may also be written as either `tool:<function_name>` or just `<function_name>`.

## `overrides`

```yaml
overrides:
  refund-not-delivered:
    assertions:
      - type: not_called
        target: tool:process_refund
```

- Each key under `overrides` is matched against `run.metadata.scenario`.
- Matching override assertions are appended after `defaults.assertions`.
- `extra_assertions=` passed directly to `ac_check_contract()` are appended last.

Merge order in `ac_check_contract()`:

1. `defaults.assertions`
2. `overrides[run.metadata.scenario].assertions`
3. `extra_assertions`

## `policies`

Supported policy types in the built-in `AssertionEngine`:

| Type | Required Fields | Behavior |
| --- | --- | --- |
| `tool_allowlist` | `name`, `tools` | Fails if any recorded tool call is not in `tools`. |
| `requires_confirmation` | `name`, `tools` | Fails if a protected tool call is not immediately preceded by a user turn. |

Accepted policy fields:

| Field | Parsed | Used By Built-In Engine |
| --- | --- | --- |
| `name` | yes | yes |
| `type` | yes | yes |
| `tools` | yes | yes |
| `target` | yes | no |
| `block` | yes | no |

## `thresholds`

```yaml
thresholds:
  suite_pass_rate: 1.0
```

- Parsed into `ac_config.suite_pass_rate`.
- Default is `1.0`.
- The built-in plugin does not automatically fail a run based on this threshold. Use it from your own suite or reporting code if you need suite-level gating.

## `budgets`

```yaml
budgets:
  per_scenario:
    max_cost_usd: 0.05
    max_latency_ms: 10000
    max_turns: 15
  suite:
    max_cost_usd: 2.0
```

Parsed fields:

| Key | Default |
| --- | --- |
| `budgets.per_scenario.max_cost_usd` | `0.05` |
| `budgets.per_scenario.max_latency_ms` | `10000` |
| `budgets.per_scenario.max_turns` | `15` |
| `budgets.suite.max_cost_usd` | `2.0` |

These values are available on:

- `ac_config.per_scenario_budget.max_cost_usd`
- `ac_config.per_scenario_budget.max_latency_ms`
- `ac_config.per_scenario_budget.max_turns`
- `ac_config.suite_budget_usd`

The built-in plugin does not enforce these budgets automatically.

## `baseline`

```yaml
baseline:
  branch: main
  show_deltas: true
```

Parsed fields:

| Key | Default |
| --- | --- |
| `baseline.branch` | `"main"` |
| `baseline.show_deltas` | `true` |

These values are exposed on `ac_config.baseline_branch` and `ac_config.show_deltas`. They are parsed today but not consumed by the built-in pytest plugin.

## `reporting`

```yaml
reporting:
  github_comment: true
  artifact_path: agentci-results/
```

Parsed fields:

| Key | Default |
| --- | --- |
| `reporting.github_comment` | `true` |
| `reporting.artifact_path` | `"agentci-results/"` |

These values are exposed on `ac_config.github_comment` and `ac_config.artifact_path`. They are parsed today but not consumed by the built-in pytest plugin.

## Related References

- [Pytest Plugin Reference](pytest-plugin.md)
- [`agentcontract init` starter template in the CLI](../src/agentcontract/cli.py)
