# Configuration Reference

`AgentContractConfig` is loaded from `agentcontract.yml`. If no file is found, the library uses built-in defaults.

## Full Schema

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
  branch: main
  show_deltas: true

reporting:
  github_comment: true
  artifact_path: agentci-results/
```

## Top-Level Fields

### `version`

String version for the config format. Default: `"1"`.

### `scenarios`

Controls discovery patterns for scenario files.

- `include`: list of glob patterns. Default: `["tests/scenarios/**/*.agentrun.json"]`
- `exclude`: list of glob patterns. Default: `[]`

The current pytest plugin uses `--ac-scenarios` or `tests/scenarios` for direct cassette lookup during record and replay. `scenarios.include` and `scenarios.exclude` are parsed into config for higher-level tooling and reporting.

### `replay`

Replay settings parsed into `ReplayConfig`.

- `model`: string, default `""`
- `seed`: integer or null, default `42`
- `stub_tools`: boolean, default `true`
- `concurrency`: integer, default `5`

These values are available through `ac_config.replay`. The current pytest plugin and `ReplayEngine` do not change behavior based on `model`, `seed`, or `concurrency`; they are preserved for callers and future tooling.

### `defaults`

Shared assertions applied by `ac_check_contract`.

- `assertions`: list of assertion objects

### `overrides`

Per-scenario assertion additions keyed by `run.metadata.scenario`.

```yaml
overrides:
  refund-not-delivered:
    assertions:
      - type: not_called
        target: tool:process_refund
```

`ac_check_contract` appends these assertions after `defaults.assertions`.

### `policies`

List of policy objects. `ac_check_contract` passes the full list to `AssertionEngine.check(...)` for every run.

### `thresholds`

Suite-level thresholds.

- `suite_pass_rate`: float, default `1.0`

### `budgets`

Budget values parsed into config.

- `per_scenario.max_cost_usd`: float, default `0.05`
- `per_scenario.max_latency_ms`: float, default `10000`
- `per_scenario.max_turns`: integer, default `15`
- `suite.max_cost_usd`: float, default `2.0`

### `baseline`

Baseline comparison settings.

- `branch`: string, default `"main"`
- `show_deltas`: boolean, default `true`

### `reporting`

Output settings.

- `github_comment`: boolean, default `true`
- `artifact_path`: string, default `"agentci-results/"`

## Assertion Objects

Assertion objects are parsed into `AssertionSpec`.

### Common fields

- `type`: required string
- `target`: optional string, default `""`
- `value`: optional scalar
- `schema`: optional object

### Additional parsed fields

These keys are accepted and preserved on `AssertionSpec`, even though the built-in `AssertionEngine` does not currently read them:

- `threshold`
- `prompt`
- `judge_model`
- `tools`
- `block`

## Supported Assertion Types

The built-in `AssertionEngine` supports these `type` values:

### `exact`

Checks that the resolved `target` is exactly equal to `value`.

### `contains`

Checks that `value` is a substring of the resolved `target`.

### `regex`

Checks that the regular expression in `value` matches the resolved `target`.

### `json_schema`

Validates the resolved `target` against `schema` using `jsonschema`.

### `not_called`

Checks that a tool was never called. Use `target: tool:<function_name>` or just the function name.

### `called_with`

Checks that a tool was called with argument values that contain all key/value pairs in `schema`. Use `target: tool:<function_name>` or just the function name.

### `called_count`

Checks that a tool was called exactly `value` times. `value` must be an integer or an integer-like string.

## Supported Assertion Targets

These targets are resolved by the built-in engine:

- `final_response`
- `turn:<index>`
- `full_conversation`
- `tool_call:<function_name>:arguments`
- `tool_call:<function_name>:result`

Tool-oriented assertion types also accept `tool:<function_name>` as the `target` for `not_called`, `called_with`, and `called_count`.

## Policy Objects

Policy objects are parsed into `PolicySpec`.

### Parsed fields

- `name`: required string
- `type`: required string
- `target`: optional string, default `""`
- `tools`: optional list of strings, default `[]`
- `block`: optional list of strings, default `[]`

`target` and `block` are parsed and preserved, but the built-in policy checkers currently use only `tools`.

## Supported Policy Types

### `tool_allowlist`

Fails if any recorded tool call name is not in `tools`.

```yaml
policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, process_refund]
```

### `requires_confirmation`

Fails if any protected tool in `tools` is called without the immediately previous turn being a user turn.

```yaml
policies:
  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

## Type Coercion and Defaults

The loader is intentionally forgiving:

- missing or `null` sections fall back to defaults
- stringified numbers and booleans are coerced where possible
- malformed values fall back to the section default

That behavior is covered by `tests/unit/test_config.py`.
