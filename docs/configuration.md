# Configuration

`agentcontract.yml` is optional. When present, the pytest fixtures discover it
by walking up from the current working directory. Use `--ac-config` to load a
specific file.

Generate a starter file:

```bash
agentcontract init
```

## What pytest uses today

The current pytest integration uses:

- `defaults.assertions` in `ac_check_contract`
- `overrides.<scenario>.assertions` in `ac_check_contract`
- `policies` in `ac_check_contract`
- `--ac-scenarios` for cassette save/load location

Other schema sections below are parsed into `AgentContractConfig` and exposed to
callers, but are not currently enforced by the pytest plugin itself.

## Complete schema

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

overrides:
  refund-not-delivered:
    assertions:
      - type: not_called
        target: "tool:process_refund"

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
  artifact_path: "agentci-results/"
```

## Section reference

### `version`

Default: `"1"`

Stored as a string.

### `scenarios`

Defaults:

```yaml
scenarios:
  include: ["tests/scenarios/**/*.agentrun.json"]
  exclude: []
```

`AgentContractConfig` stores these glob lists as `scenario_include` and
`scenario_exclude`. The pytest fixtures do not currently use these globs for
cassette discovery; record/replay path selection comes from `--ac-scenarios` or
the default `tests/scenarios`.

### `replay`

Defaults:

```yaml
replay:
  model: ""
  seed: 42
  stub_tools: true
  concurrency: 5
```

These values are parsed into `ReplayConfig`. The built-in `ReplayEngine`
constructs a `ToolStub` from a loaded cassette; tests decide whether to use
`tool_stub` or inspect `recorded_run` directly.

### `defaults.assertions`

Default: `[]`

Each assertion is parsed as an `AssertionSpec` and checked by
`ac_check_contract` for every scenario.

```yaml
defaults:
  assertions:
    - type: contains
      target: final_response
      value: "processed"
    - type: called_with
      target: "tool:lookup_order"
      schema:
        order_id: "ORD-123"
```

Supported assertion fields are `type`, `target`, `value`, `threshold`,
`prompt`, `schema`, `judge_model`, `tools`, and `block`. The current assertion
engine uses `type`, `target`, `value`, and `schema`.

### `overrides`

Default: `{}`

Keys are scenario names. Matching override assertions are appended after default
assertions in `ac_check_contract`.

```yaml
overrides:
  refund-not-delivered:
    assertions:
      - type: not_called
        target: "tool:process_refund"
```

### `policies`

Default: `[]`

Each policy is parsed as a `PolicySpec`.

```yaml
policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]

  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

Supported policy fields are `name`, `type`, `target`, `tools`, and `block`.
The current policy engine uses `name`, `type`, and `tools`.

### `thresholds`

Defaults:

```yaml
thresholds:
  suite_pass_rate: 1.0
```

Parsed as `suite_pass_rate`.

### `budgets`

Defaults:

```yaml
budgets:
  per_scenario:
    max_cost_usd: 0.05
    max_latency_ms: 10000
    max_turns: 15
  suite:
    max_cost_usd: 2.0
```

Parsed as `per_scenario_budget` and `suite_budget_usd`.

### `baseline`

Defaults:

```yaml
baseline:
  branch: main
  show_deltas: true
```

Parsed as `baseline_branch` and `show_deltas`.

### `reporting`

Defaults:

```yaml
reporting:
  github_comment: true
  artifact_path: "agentci-results/"
```

Parsed as `github_comment` and `artifact_path`.

## Starter template

`agentcontract init` creates this starter subset:

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
      value: ""  # customize this

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: []  # list your agent's tools here

budgets:
  per_scenario:
    max_cost_usd: 0.05
    max_turns: 15

reporting:
  github_comment: true
  artifact_path: "agentci-results/"
```
