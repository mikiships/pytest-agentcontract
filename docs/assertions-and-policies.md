# Assertions and policies

The assertion engine evaluates a recorded `AgentRun` against assertion specs and
policy specs. `ac_check_contract` combines config defaults, matching scenario
overrides, optional inline assertions, and configured policies.

## Assertion types

| Type | Required fields | Behavior |
| --- | --- | --- |
| `exact` | `target`, `value` | Target must equal `value`. |
| `contains` | `target`, `value` | `value` must appear as a substring of the target. |
| `regex` | `target`, `value` | Python `re.search(value, target)` must match. |
| `json_schema` | `target`, `schema` | Validates the resolved target with `jsonschema.validate`. |
| `not_called` | `target` | Tool named by `target` must not appear in the run. |
| `called_with` | `target`, `schema` | Tool must be called with expected arguments from `schema`. |
| `called_count` | `target`, `value` | Tool must be called exactly `value` times. |

Unknown assertion types fail closed and produce an assertion failure.

## Target syntax

- `final_response`: last assistant turn with non-`None` content.
- `full_conversation`: all turn content joined as `role: content` lines.
- `turn:N`: content for turn index `N`.
- `tool_call:function_name:arguments`: arguments for the first matching tool
  call.
- `tool_call:function_name:result`: result for the first matching tool call.
- `tool:function_name`: accepted by tool assertions such as `not_called`,
  `called_with`, and `called_count`.
- `function_name`: also accepted by tool assertions.

## `called_with`

`called_with` reuses the `schema` field as expected arguments. It is not JSON
Schema validation. The expected key/value pairs must be a subset of the recorded
tool arguments with exact equality.

```yaml
defaults:
  assertions:
    - type: called_with
      target: "tool:lookup_order"
      schema:
        order_id: "ORD-123"
```

This passes when the recorded call includes `{"order_id": "ORD-123"}` even if
the call has additional arguments.

## `called_count`

`called_count.value` must be an integer or an integer-like value such as `"2"`.
Missing values, booleans, non-integer floats, and non-numeric strings fail.

```yaml
defaults:
  assertions:
    - type: called_count
      target: "tool:process_refund"
      value: 1
```

## Policies

| Type | Required fields | Behavior |
| --- | --- | --- |
| `tool_allowlist` | `name`, `type`, `tools` | Every recorded tool call must be included in `tools`. |
| `requires_confirmation` | `name`, `type`, `tools` | Each protected tool call must be immediately preceded by a user turn. |

Unknown policy types fail closed and produce a policy failure.

`requires_confirmation` checks turn ordering only. It does not parse the content
of the user message to decide whether the text is a valid confirmation.

## YAML example

```yaml
defaults:
  assertions:
    - type: contains
      target: final_response
      value: "refund"
    - type: json_schema
      target: "tool_call:process_refund:arguments"
      schema:
        type: object
        required: [order_id, amount, method]
    - type: called_with
      target: "tool:lookup_order"
      schema:
        order_id: "ORD-123"

overrides:
  refund-not-delivered:
    assertions:
      - type: not_called
        target: "tool:process_refund"

policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]

  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

## Python example

```python
from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import AssertionSpec, PolicySpec


engine = AssertionEngine()
result = engine.check(
    run,
    assertions=[
        AssertionSpec(type="contains", target="final_response", value="$79.99"),
        AssertionSpec(
            type="called_with",
            target="tool:lookup_order",
            schema={"order_id": "ORD-123"},
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
