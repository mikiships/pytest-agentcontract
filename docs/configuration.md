# Configuration Reference

`AgentContractConfig` parses `agentcontract.yml` into a typed config object. Some keys are enforced by the pytest plugin and assertion engine today; others are parsed and exposed through `ac_config` for your own tooling.

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

overrides:
  refund-denied:
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

## Top-Level Keys

| Key | Parsed into | Default | Used by current implementation |
| --- | --- | --- | --- |
| `version` | `config.version` | `"1"` | Stored only |
| `scenarios` | `config.scenario_include`, `config.scenario_exclude` | include `["tests/scenarios/**/*.agentrun.json"]`, exclude `[]` | Stored only |
| `replay` | `config.replay` | See below | Stored only |
| `defaults.assertions` | `config.default_assertions` | `[]` | Yes, through `ac_check_contract` |
| `overrides` | `config.overrides` | `{}` | Yes, through `ac_check_contract` |
| `policies` | `config.policies` | `[]` | Yes, through `ac_check_contract` or direct `AssertionEngine.check(...)` |
| `thresholds.suite_pass_rate` | `config.suite_pass_rate` | `1.0` | Stored only |
| `budgets` | `config.per_scenario_budget`, `config.suite_budget_usd` | See below | Stored only |
| `baseline` | `config.baseline_branch`, `config.show_deltas` | `main`, `true` | Stored only |
| `reporting` | `config.github_comment`, `config.artifact_path` | `true`, `agentci-results/` | Stored only |

The pytest plugin does not currently use `scenarios.include`, `scenarios.exclude`, `replay`, `thresholds`, `budgets`, `baseline`, or `reporting` to change runtime behavior. Those values are still available through `ac_config`.

## `scenarios`

```yaml
scenarios:
  include: ["tests/scenarios/**/*.agentrun.json"]
  exclude: []
```

- `include` becomes `config.scenario_include`
- `exclude` becomes `config.scenario_exclude`

These are plain lists. If omitted or invalid, the parser falls back to the defaults above.

## `replay`

```yaml
replay:
  model: ""
  seed: 42
  stub_tools: true
  concurrency: 5
```

Parsed fields:

- `model: str`
- `seed: int | null`
- `stub_tools: bool`
- `concurrency: int`

These values land in `config.replay`, but `ReplayEngine` and the pytest plugin do not read them automatically. If you want them to affect your replay loop, read them from `ac_config` inside your test or helper code.

## Assertions

Assertions are defined under `defaults.assertions` or under `overrides.<scenario>.assertions`.

Each assertion is parsed into:

- `type`
- `target`
- `value`
- `threshold`
- `prompt`
- `schema`
- `judge_model`
- `tools`
- `block`

The current `AssertionEngine` evaluates these assertion types:

| Type | Required fields | Notes |
| --- | --- | --- |
| `exact` | `target`, `value` | Exact string equality against the resolved target |
| `contains` | `target`, `value` | Substring match |
| `regex` | `target`, `value` | `re.search(...)` against the resolved target |
| `json_schema` | `target`, `schema` | Uses `jsonschema.validate(...)` |
| `not_called` | `target` | `target` can be `tool:name` or just `name` |
| `called_with` | `target`, `schema` | Treats `schema` as the expected argument subset |
| `called_count` | `target`, `value` | `value` must coerce to an integer |

`threshold`, `prompt`, `judge_model`, `tools`, and `block` are parsed but not consumed by the current assertion engine.

### Supported Targets

| Target | Resolves to |
| --- | --- |
| `final_response` | The last assistant turn with non-`None` content |
| `full_conversation` | Every turn joined as `role: content` lines |
| `turn:N` | The content of turn index `N` |
| `tool_call:function:arguments` | The first matching tool call arguments object |
| `tool_call:function:result` | The first matching tool call result payload |

## `overrides`

Overrides are keyed by scenario name:

```yaml
overrides:
  refund-eligible:
    assertions:
      - type: contains
        target: final_response
        value: processed
```

`ac_check_contract(run, ...)` uses `run.metadata.scenario` to decide whether an override applies.

## Policies

Policies are parsed into `PolicySpec` objects with these fields:

- `name`
- `type`
- `target`
- `tools`
- `block`

The current engine enforces two policy types:

| Type | Required fields | Behavior |
| --- | --- | --- |
| `tool_allowlist` | `tools` | Fails if any recorded tool call is not listed |
| `requires_confirmation` | `tools` | Fails if a listed tool call is not immediately preceded by a user turn |

`target` and `block` are parsed but not used by the current policy checks.

The `requires_confirmation` policy is intentionally simple: it checks only the previous turn’s role, not the text of the confirmation message.

## `thresholds`

```yaml
thresholds:
  suite_pass_rate: 1.0
```

- `suite_pass_rate` becomes `config.suite_pass_rate`

The parser coerces numeric-looking strings such as `"0.75"` into floats.

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

- `config.per_scenario_budget.max_cost_usd`
- `config.per_scenario_budget.max_latency_ms`
- `config.per_scenario_budget.max_turns`
- `config.suite_budget_usd`

These values are stored on the config object only.

## `baseline`

```yaml
baseline:
  branch: main
  show_deltas: true
```

Parsed fields:

- `config.baseline_branch`
- `config.show_deltas`

These are stored only.

## `reporting`

```yaml
reporting:
  github_comment: true
  artifact_path: agentci-results/
```

Parsed fields:

- `config.github_comment`
- `config.artifact_path`

These are stored only.

## Discovery and Overrides

- `AgentContractConfig.discover()` walks upward from the current working directory until it finds `agentcontract.yml`.
- If discovery starts from a file path, it first switches to that file’s parent directory.
- `--ac-config` bypasses discovery and loads the exact file you pass to pytest.
- Invalid or missing sections fall back to defaults instead of raising during parse.
