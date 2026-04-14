# Customer Support Example

The repository includes a small end-to-end example under `examples/customer_support/` that demonstrates the intended `live -> record -> replay -> assert` workflow.

## Files

| Path | Purpose |
| --- | --- |
| `examples/customer_support/agent.py` | A deterministic support agent that looks up orders, checks refund eligibility, and processes refunds. |
| `examples/customer_support/test_support.py` | Three pytest tests using the plugin fixtures. |
| `examples/customer_support/scenarios/*.agentrun.json` | Bundled example cassettes for replay. |

## What The Example Covers

The example includes three scenarios:

- `refund-eligible`: happy path refund flow with extra content and schema assertions.
- `refund-not-delivered`: denial flow that asserts `process_refund` was not called.
- `refund-with-policies`: policy-only validation with `tool_allowlist` and `requires_confirmation`.

The tests demonstrate:

- manual turn recording with `ac_recorder.add_turn(...)`
- branching on `ac_mode`
- replaying from `ac_replay_engine.recorded_run`
- extra assertions passed directly to `ac_check_contract(...)`
- direct `AssertionEngine().check(...)` usage for policies

## Run The Example

All commands below are relative to the repository root.

### Live mode

This runs the example agent directly and evaluates assertions without writing cassettes:

```bash
pytest examples/customer_support/test_support.py -v
```

### Record mode

The plugin default cassette directory is `tests/scenarios`, but the bundled example stores its cassettes under `examples/customer_support/scenarios`. Pass `--ac-scenarios` so record mode writes back into the example directory:

```bash
pytest examples/customer_support/test_support.py \
  --ac-record \
  --ac-scenarios examples/customer_support/scenarios \
  -v
```

That refreshes:

- `examples/customer_support/scenarios/refund-eligible.agentrun.json`
- `examples/customer_support/scenarios/refund-not-delivered.agentrun.json`
- `examples/customer_support/scenarios/refund-with-policies.agentrun.json`

### Replay mode

Use the same `--ac-scenarios` override so the plugin loads the bundled example cassettes instead of looking in `tests/scenarios`:

```bash
pytest examples/customer_support/test_support.py \
  --ac-replay \
  --ac-scenarios examples/customer_support/scenarios \
  -v
```

## Example Behavior

`agent.py` models a simple refund assistant:

1. Parse an order ID from the user message.
2. Call `lookup_order`.
3. Call `check_refund_eligibility`.
4. If eligible, wait for an explicit user confirmation turn.
5. Call `process_refund`.

The bundled cassettes show both the happy path and the ineligible-order path:

- `refund-eligible` records three tool calls and a final success message.
- `refund-not-delivered` records two tool calls and stops before any refund is issued.
- `refund-with-policies` reuses the happy path structure to validate policy checks.

## Assertions And Policies Demonstrated

`test_support.py` currently exercises these contract checks:

- `contains` against `final_response`
- `json_schema` against `tool_call:process_refund:arguments`
- `not_called` against `tool:process_refund`
- `tool_allowlist`
- `requires_confirmation`

That makes the example a good template when you want to:

- lock down the final assistant reply
- validate tool argument shape
- assert that dangerous tools were not invoked
- require a user confirmation step before sensitive actions

## Adapting It To Your Own Agent

Start by replacing the toy agent implementation while keeping the test shape:

1. Keep the scenario markers stable so cassette names stay predictable.
2. In live and record modes, push each user, assistant, and tool turn into `ac_recorder`.
3. In replay mode, either reuse `recorded_run` for contract checks or wire your agent to `ac_replay_engine.tool_stub`.
4. Move shared assertions into `agentcontract.yml` once they apply across many tests.
