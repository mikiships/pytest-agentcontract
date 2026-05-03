# Configuration

pytest-agentcontract looks for `agentcontract.yml` by walking upward from the
current working directory. Use `--ac-config` to point pytest at a specific file.

Create a starter file with:

```bash
agentcontract init
```

## Full Example

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
  refund-eligible:
    assertions:
      - type: called_with
        target: tool:process_refund
        schema:
          order_id: ORD-123

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]

budgets:
  per_scenario:
    max_cost_usd: 0.05
    max_latency_ms: 10000
    max_turns: 15
  suite:
    max_cost_usd: 2.0

thresholds:
  suite_pass_rate: 1.0

baseline:
  branch: main
  show_deltas: true

reporting:
  github_comment: true
  artifact_path: agentci-results/
```

## Supported Sections

| Section | Fields |
| --- | --- |
| `version` | String config version. Defaults to `"1"`. |
| `scenarios` | `include`, `exclude`. Defaults to `["tests/scenarios/**/*.agentrun.json"]` and `[]`. |
| `replay` | `model`, `seed`, `stub_tools`, `concurrency`. |
| `defaults.assertions` | Assertion specs applied to every checked run. |
| `overrides` | Per-scenario assertion specs appended after defaults. |
| `policies` | Policy specs applied to every checked run. |
| `budgets.per_scenario` | `max_cost_usd`, `max_latency_ms`, `max_turns`. |
| `budgets.suite` | `max_cost_usd`. |
| `thresholds` | `suite_pass_rate`. |
| `baseline` | `branch`, `show_deltas`. |
| `reporting` | `github_comment`, `artifact_path`. |

## Defaults

When a section is omitted or set to `null`, AgentContractConfig falls back to
these defaults:

| Config value | Default |
| --- | --- |
| `scenarios.include` | `["tests/scenarios/**/*.agentrun.json"]` |
| `scenarios.exclude` | `[]` |
| `replay.model` | `""` |
| `replay.seed` | `42` |
| `replay.stub_tools` | `true` |
| `replay.concurrency` | `5` |
| `defaults.assertions` | `[]` |
| `overrides` | `{}` |
| `policies` | `[]` |
| `thresholds.suite_pass_rate` | `1.0` |
| `budgets.per_scenario.max_cost_usd` | `0.05` |
| `budgets.per_scenario.max_latency_ms` | `10000` |
| `budgets.per_scenario.max_turns` | `15` |
| `budgets.suite.max_cost_usd` | `2.0` |
| `baseline.branch` | `main` |
| `baseline.show_deltas` | `true` |
| `reporting.github_comment` | `true` |
| `reporting.artifact_path` | `agentci-results/` |

## How Config Is Applied

`ac_check_contract(run)` evaluates:

1. `defaults.assertions`
2. assertions from `overrides.<scenario>.assertions`, when the run metadata
   scenario matches an override key
3. any `extra_assertions` passed directly to `ac_check_contract`
4. every configured policy

The current pytest fixtures use `--ac-scenarios` for cassette load/save paths.
The `scenarios.include` and `scenarios.exclude` values are parsed into config
for suite-level tooling and reporting.
