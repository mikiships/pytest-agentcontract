# Configuration

pytest-agentcontract discovers `agentcontract.yml` by walking up from the current working directory. You can also pass an explicit path:

```bash
pytest --ac-config path/to/agentcontract.yml --ac-replay
```

The pytest fixtures currently use config assertions, scenario overrides, and policies when `ac_check_contract(run)` is called. Other parsed sections, such as thresholds, budgets, baseline, and reporting, are available on `AgentContractConfig` for runners and future workflow integration, but the built-in pytest fixtures do not enforce them automatically.

## Complete Parser-Default Config

This sample shows every section parsed by `AgentContractConfig` with the parser defaults filled in:

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

overrides: {}

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

## Sections

`version` is stored as a string and defaults to `"1"`.

`scenarios.include` and `scenarios.exclude` are lists of cassette glob patterns. The parser default include pattern is `tests/scenarios/**/*.agentrun.json`. The built-in pytest fixtures use `--ac-scenarios` or `tests/scenarios` for direct marker-to-file lookup.

`replay.model` stores the replay model name if your runner needs one. The built-in `ReplayEngine` replays recorded tool results and does not call a model by itself.

`replay.seed` defaults to `42` and accepts an integer or `null`.

`replay.stub_tools` defaults to `true`. In the built-in engine, tool stubbing is exposed through `ReplayEngine.tool_stub`.

`replay.concurrency` defaults to `5` for runners that execute scenarios in parallel. The built-in pytest fixtures do not schedule concurrency themselves.

`defaults.assertions` is a list of assertion specs applied by `ac_check_contract(run)` for every scenario.

`overrides.<scenario>.assertions` adds scenario-specific assertions when `run.metadata.scenario` matches the override key.

`policies` is a list of policy specs applied by `ac_check_contract(run)`.

`thresholds.suite_pass_rate` defaults to `1.0`. It is parsed and exposed as `config.suite_pass_rate`.

`budgets.per_scenario` stores `max_cost_usd`, `max_latency_ms`, and `max_turns`. `budgets.suite.max_cost_usd` stores the suite budget.

`baseline.branch` and `baseline.show_deltas` describe comparison behavior for higher-level reporting workflows.

`reporting.github_comment` and `reporting.artifact_path` describe reporting output preferences.

## Example With Assertions And Policies

```yaml
version: "1"

scenarios:
  include:
    - "tests/scenarios/**/*.agentrun.json"

replay:
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
    tools:
      - lookup_order
      - check_refund_eligibility
      - process_refund

  - name: confirm-before-refund
    type: requires_confirmation
    tools:
      - process_refund
```

Unknown assertion or policy types fail closed when evaluated by the assertion engine.
