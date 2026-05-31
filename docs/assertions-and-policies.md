# Assertions And Policies

Assertions and policies are evaluated by `AssertionEngine.check(run, assertions, policies)`. In pytest, `ac_check_contract(run, extra_assertions=...)` combines configured default assertions, per-scenario overrides, supplied extra assertions, and configured policies.

## Assertion Types

| Type | Required fields | Behavior |
| --- | --- | --- |
| `exact` | `target`, `value` | Passes when the resolved target equals `value`. |
| `contains` | `target`, `value` | Passes when `value` is a substring of the resolved target. |
| `regex` | `target`, `value` | Passes when the regular expression in `value` matches the resolved target. |
| `json_schema` | `target`, `schema` | Validates the resolved target with `jsonschema.validate`. |
| `not_called` | `target` | Passes when the named tool was not called. |
| `called_with` | `target`, `schema` | Passes when the named tool was called with at least the key/value pairs in `schema`. |
| `called_count` | `target`, `value` | Passes when the named tool was called exactly `value` times. |

`AssertionSpec` also parses optional fields such as `threshold`, `prompt`, `judge_model`, `tools`, and `block` from configuration for forward-compatible consumers. The built-in engine currently evaluates the fields listed above.

## Target Syntax

- `final_response`: content from the last assistant turn that has non-null content.
- `turn:N`: content from turn index `N`.
- `full_conversation`: all non-null turn content joined as `role: content` lines.
- `tool_call:function_name:arguments`: arguments from the first matching tool call.
- `tool_call:function_name:result`: result from the first matching tool call.

Tool-specific assertions use `tool:function_name` or just `function_name`:

```yaml
- type: not_called
  target: tool:process_refund
```

## Tool Argument Checks

`called_with` uses the `schema` field as an expected argument subset, not as a JSON Schema document:

```yaml
- type: called_with
  target: tool:process_refund
  schema:
    order_id: "ORD-123"
    method: "original"
```

That assertion passes when a recorded `process_refund` call has those exact key/value pairs, even if it also has other arguments such as `amount`.

Use `json_schema` when you need JSON Schema validation:

```yaml
- type: json_schema
  target: tool_call:process_refund:arguments
  schema:
    type: object
    required: [order_id, amount, method]
    properties:
      order_id:
        type: string
      amount:
        type: number
      method:
        const: original
```

`called_count` accepts integer-like values, including strings such as `"1"`:

```yaml
- type: called_count
  target: tool:lookup_order
  value: "1"
```

## Policies

Policies are configured with `PolicySpec` entries and run after assertions.

### `tool_allowlist`

Only listed tools may be called:

```yaml
policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]
```

Any recorded tool call whose function is not in `tools` fails the policy.

### `requires_confirmation`

Protected tools must be immediately preceded by a user turn:

```yaml
policies:
  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

The built-in policy checks the previous turn role only. It does not parse the user's text for a specific confirmation phrase.

## Python API

```python
from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import AssertionSpec, PolicySpec

result = AssertionEngine().check(
    run,
    assertions=[
        AssertionSpec(type="contains", target="final_response", value="$79.99"),
        AssertionSpec(
            type="called_with",
            target="tool:process_refund",
            schema={"order_id": "ORD-123", "method": "original"},
        ),
    ],
    policies=[
        PolicySpec(
            name="allowed-tools",
            type="tool_allowlist",
            tools=["lookup_order", "check_refund_eligibility", "process_refund"],
        )
    ],
)
assert result.passed, [failure.message for failure in result.failures()]
```
