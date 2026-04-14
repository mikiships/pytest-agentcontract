# Configuration Reference

`agentcontract.yml` is parsed by `AgentContractConfig`. This page documents the full schema the current code accepts, plus which parts are emitted by `agentcontract init` and which parts are actively consumed by the plugin/runtime today.

## Discovery

`ac_config` uses this resolution order:

1. `--ac-config PATH`, if supplied.
2. Walk upward from the current working directory until `agentcontract.yml` is found.
3. Fall back to built-in defaults if no file exists.

## Full Parsed Schema

This example includes every top-level section the current parser understands:

```yaml
version: "1"

scenarios:
  include:
    - "tests/scenarios/**/*.agentrun.json"
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
  refund-not-delivered:
    assertions:
      - type: not_called
        target: tool:process_refund

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools:
      - lookup_order
      - check_refund_eligibility
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

## What The Runtime Uses Today

The current pytest/runtime integration actively consumes:

- `defaults.assertions`
- `overrides.<scenario>.assertions`
- `policies`

The rest of the schema is still parsed into the `AgentContractConfig` object and available to your own tooling, but the shipped pytest plugin and CLI do not currently enforce or act on those values.

That matters for:

- `scenarios.include` and `scenarios.exclude`: parsed only; cassette save/load paths still come from `--ac-scenarios` or the default `tests/scenarios`.
- `replay.*`: parsed only; `ReplayEngine` does not read these values.
- `thresholds`, `budgets`, `baseline`, and `reporting`: parsed only in this version.

## Key Reference

| Key | Default | `init` writes it | Current runtime uses it |
| --- | --- | --- | --- |
| `version` | `"1"` | Yes | No |
| `scenarios.include` | `["tests/scenarios/**/*.agentrun.json"]` | Yes | No |
| `scenarios.exclude` | `[]` | No | No |
| `replay.model` | `""` | No | No |
| `replay.seed` | `42` | No | No |
| `replay.stub_tools` | `true` | Yes | No |
| `replay.concurrency` | `5` | Yes | No |
| `defaults.assertions` | `[]` | Yes | Yes |
| `overrides` | `{}` | No | Yes |
| `policies` | `[]` | Yes | Yes |
| `thresholds.suite_pass_rate` | `1.0` | No | No |
| `budgets.per_scenario.max_cost_usd` | `0.05` | Yes | No |
| `budgets.per_scenario.max_latency_ms` | `10000` | No | No |
| `budgets.per_scenario.max_turns` | `15` | Yes | No |
| `budgets.suite.max_cost_usd` | `2.0` | No | No |
| `baseline.branch` | `"main"` | No | No |
| `baseline.show_deltas` | `true` | No | No |
| `reporting.github_comment` | `true` | Yes | No |
| `reporting.artifact_path` | `"agentci-results/"` | Yes | No |

## Assertion Specs

Each item under `defaults.assertions` or `overrides.<scenario>.assertions` is parsed into an `AssertionSpec`.

Supported assertion types in the current `AssertionEngine`:

- `exact`
- `contains`
- `regex`
- `json_schema`
- `not_called`
- `called_with`
- `called_count`

Common fields:

| Field | Meaning |
| --- | --- |
| `type` | Assertion type. Required. |
| `target` | What to inspect, such as `final_response`, `turn:3`, `tool_call:process_refund:arguments`, or `tool:process_refund`. |
| `value` | Used by `exact`, `contains`, `regex`, and `called_count`. |
| `schema` | Used by `json_schema` and `called_with`. |

Additional parsed fields that are accepted in config but not consumed by the current assertion engine:

- `threshold`
- `prompt`
- `judge_model`
- `tools`
- `block`

### Target Syntax

Targets supported by the current assertion engine:

- `final_response`
- `full_conversation`
- `turn:N`
- `tool_call:<function>:arguments`
- `tool_call:<function>:result`
- `tool:<function>` for `not_called`, `called_with`, and `called_count`

Examples:

```yaml
defaults:
  assertions:
    - type: contains
      target: final_response
      value: "processed"
    - type: json_schema
      target: tool_call:process_refund:arguments
      schema:
        type: object
        required: [order_id, amount, method]
    - type: called_count
      target: tool:lookup_order
      value: "1"
```

## Policy Specs

Each item under `policies` is parsed into a `PolicySpec`.

Supported policy types in the current `AssertionEngine`:

- `tool_allowlist`
- `requires_confirmation`

Policy fields:

| Field | Meaning |
| --- | --- |
| `name` | Policy label shown in assertion output. Required. |
| `type` | Policy type. Required. |
| `tools` | Tool names the policy applies to. |
| `target` | Parsed but unused by the current engine. |
| `block` | Parsed but unused by the current engine. |

Examples:

```yaml
policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]

  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

`requires_confirmation` currently means: if one of the protected tools appears in a turn, the immediately previous turn must be a user turn.

## What `agentcontract init` Creates

`agentcontract init` writes this starter file:

```yaml
version: "1"

scenarios:
  include: ["tests/scenarios/**/*.agentrun.json"]

replay:
  stub_tools: true
  concurrency: 5

defaults:
  assertions:
    - type: contains
      target: final_response
      value: ""

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: []

budgets:
  per_scenario:
    max_cost_usd: 0.05
    max_turns: 15

reporting:
  github_comment: true
  artifact_path: "agentci-results/"
```

It does not emit:

- `scenarios.exclude`
- `replay.model`
- `replay.seed`
- `overrides`
- `thresholds`
- `budgets.per_scenario.max_latency_ms`
- `budgets.suite`
- `baseline`

## Practical Guidance

For the current release, treat `agentcontract.yml` as two layers:

1. Actively enforced contract rules: `defaults.assertions`, `overrides`, and `policies`.
2. Parsed metadata for future tooling or your own wrappers: everything else.

If you need cassette location control in pytest today, use `--ac-scenarios`, not `scenarios.include`.
