# Customer Support Example

This example is a deterministic refund support agent used to demonstrate
recording, replay, assertions, and policy checks without calling external
services.

The agent in `agent.py` is a small state machine. It can:

- Look up orders with `lookup_order`.
- Check refund eligibility with `check_refund_eligibility`.
- Process eligible refunds with `process_refund`.

## Scenarios

Pre-recorded cassettes live in `examples/customer_support/scenarios/`.

- `refund-eligible`: customer requests a refund for delivered order `ORD-123`.
  The agent looks up the order, checks eligibility, receives user confirmation,
  and calls `process_refund`.
- `refund-not-delivered`: customer requests a refund for shipped order
  `ORD-456`. The agent looks up the order, checks eligibility, and does not call
  `process_refund`.
- `refund-with-policies`: same happy-path trajectory as `refund-eligible`, then
  validates tool allowlist and confirmation policies.

## Run The Example

From the repository root:

```bash
uv run pytest examples/customer_support/test_support.py -v
```

Replay from the bundled cassettes:

```bash
uv run pytest examples/customer_support/test_support.py --ac-replay \
  --ac-scenarios examples/customer_support/scenarios -v
```

Record the scenarios again into the example scenario directory:

```bash
uv run pytest examples/customer_support/test_support.py --ac-record \
  --ac-scenarios examples/customer_support/scenarios -v
```

Inspect or validate a cassette:

```bash
uv run agentcontract info examples/customer_support/scenarios/refund-eligible.agentrun.json
uv run agentcontract validate examples/customer_support/scenarios/refund-eligible.agentrun.json
```

## What The Tests Demonstrate

`test_refund_happy_path`

Records or replays a successful refund flow and asserts the final response
contains `$79.99`. It also validates the recorded `process_refund` arguments
with JSON Schema.

`test_refund_denied_not_delivered`

Records or replays an ineligible refund flow and asserts that `process_refund`
was not called.

`test_refund_with_policy_enforcement`

Checks the same eligible refund trajectory against two policies:
`tool_allowlist` and `requires_confirmation`.

The tests manually add turns returned by `run_support_agent()` to `ac_recorder`.
In replay mode, they assert the loaded `ac_replay_engine.recorded_run`.
