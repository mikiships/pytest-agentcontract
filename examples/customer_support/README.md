# Customer support example

This example is a deterministic customer support agent that demonstrates the
record/replay/assert workflow without calling an LLM or external API.

The agent in `agent.py` handles refund requests with three simulated tools:

- `lookup_order`
- `check_refund_eligibility`
- `process_refund`

The tests in `test_support.py` cover:

- `refund-eligible`: delivered order, refund is processed.
- `refund-not-delivered`: shipped order, refund is denied.
- `refund-with-policies`: tool allowlist and confirmation policies.

## Run from the repository root

Normal contract assertion mode runs the deterministic agent directly and checks
assertions:

```bash
pytest examples/customer_support/test_support.py -v
```

Record the example cassettes into this directory:

```bash
pytest examples/customer_support/test_support.py \
  --ac-record \
  --ac-scenarios examples/customer_support/scenarios \
  -v
```

Replay from the bundled cassettes:

```bash
pytest examples/customer_support/test_support.py \
  --ac-replay \
  --ac-scenarios examples/customer_support/scenarios \
  -v
```

If `--ac-scenarios` is omitted, the pytest plugin uses `tests/scenarios` by
default.

## What to look for

`test_refund_happy_path` records a final assistant response containing `$79.99`
and checks that `process_refund` arguments satisfy a JSON Schema.

`test_refund_denied_not_delivered` verifies that `process_refund` is not called
and that the final response explains the refund is not eligible.

`test_refund_with_policy_enforcement` uses `AssertionEngine` directly with
`PolicySpec` entries for `tool_allowlist` and `requires_confirmation`.
