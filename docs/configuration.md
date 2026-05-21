# Configuration

`agentcontract.yml` is loaded by the `ac_config` pytest fixture. By default,
`AgentContractConfig.discover()` walks up from the current working directory and
uses the first `agentcontract.yml` it finds. Pass `--ac-config path/to/file.yml`
to use a specific file.

Create a starter config with:

```bash
agentcontract init
```

## Complete Shape

These are the sections parsed by `AgentContractConfig`:

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

## Sections

`version`

Configuration format version. The current parser defaults to `"1"` and stores
the value as a string.

`scenarios`

`include` and `exclude` are parsed as lists. The default include pattern is
`tests/scenarios/**/*.agentrun.json`. The current pytest fixtures use
`--ac-scenarios` for the scenario directory and do not expand these glob patterns
themselves.

`replay`

Parsed fields are `model`, `seed`, `stub_tools`, and `concurrency`. Defaults are
an empty model string, seed `42`, `stub_tools: true`, and concurrency `5`.

`defaults`

`defaults.assertions` is a list of assertion specs applied by
`ac_check_contract()` to every run.

`overrides`

Keys are scenario names. When `run.metadata.scenario` matches an override key,
`ac_check_contract()` appends that override's assertions after the defaults.

`policies`

List of policy specs passed to the assertion engine by `ac_check_contract()`.
Built-in policy types are `tool_allowlist` and `requires_confirmation`.

`thresholds`

`suite_pass_rate` is parsed as a float and defaults to `1.0`. It is available on
the config object as `suite_pass_rate`; the current pytest plugin does not
enforce it automatically.

`budgets`

`budgets.per_scenario` parses `max_cost_usd`, `max_latency_ms`, and `max_turns`.
`budgets.suite.max_cost_usd` is parsed as `suite_budget_usd`. These fields are
available on the config object for reporting or custom checks; the current
pytest plugin does not enforce them automatically.

`baseline`

Parses `branch` and `show_deltas`, defaulting to `main` and `true`.

`reporting`

Parses `github_comment` and `artifact_path`, defaulting to `true` and
`agentci-results/`.

## Assertion Specs

Assertion specs support these fields:

```yaml
type: contains
target: final_response
value: refund
schema: {}
threshold: 0.8
prompt: ""
judge_model: ""
tools: []
block: []
```

The current assertion engine uses `type`, `target`, `value`, and `schema` for
the built-in assertion types. Other fields are parsed but are not used by the
current built-in checks.

## Type Coercion

The parser is forgiving for common YAML input:

- Null sections are treated as empty sections.
- Scalar values such as `version`, `baseline.branch`, and `reporting.artifact_path`
  are converted to strings.
- Boolean strings such as `true`, `false`, `yes`, `no`, `1`, and `0` are
  accepted.
- Invalid numeric fields fall back to their defaults.
