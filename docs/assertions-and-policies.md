# Assertions And Policies

Assertions check recorded content or tool calls. Policies check cross-cutting behavior across a run.

The assertion engine is available directly:

```python
from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import AssertionSpec, PolicySpec

result = AssertionEngine().check(
    run,
    assertions=[AssertionSpec(type="contains", target="final_response", value="refund")],
    policies=[PolicySpec(name="tools", type="tool_allowlist", tools=["lookup_order"])],
)
assert result.passed, [failure.message for failure in result.failures()]
```

In pytest, `ac_check_contract(run)` applies configured default assertions, matching scenario overrides, and policies. Pass `extra_assertions=[...]` for checks local to one test.

## Assertion Types

`exact` compares the resolved target to `value` with exact equality. `value` must be non-null.

```yaml
- type: exact
  target: final_response
  value: "Your refund has been processed."
```

`contains` checks that `value` appears in the resolved target after both are treated as strings.

```yaml
- type: contains
  target: final_response
  value: "refund"
```

`regex` runs a Python regular expression against the resolved target string.

```yaml
- type: regex
  target: final_response
  value: "\\$\\d+\\.\\d{2}"
```

`json_schema` validates the resolved target with `jsonschema.validate`. Use it for tool arguments or results that are JSON-like objects.

```yaml
- type: json_schema
  target: tool_call:process_refund:arguments
  schema:
    type: object
    required: [order_id, amount, method]
```

`not_called` passes when a tool was not invoked. The target can be `tool:<function_name>` or just the function name.

```yaml
- type: not_called
  target: tool:delete_account
```

`called_with` passes when a tool was called with expected argument key/value pairs. The expected arguments live in `schema`; for this assertion type, `schema` is a subset match object, not a JSON Schema document.

```yaml
- type: called_with
  target: tool:lookup_order
  schema:
    order_id: "ORD-123"
```

`called_count` passes when a tool was called exactly `value` times. `value` may be an integer-like string.

```yaml
- type: called_count
  target: tool:lookup_order
  value: "1"
```

## Target Syntax

`final_response` resolves to the content of the last assistant turn that has non-null content. Empty strings are valid content.

`full_conversation` resolves to all non-null turn contents joined as `role: content` lines.

`turn:N` resolves to the content of turn index `N`.

`tool_call:function_name:arguments` resolves to the arguments object for the first recorded tool call with that function name.

`tool_call:function_name:result` resolves to the result value for the first recorded tool call with that function name.

Tool-call assertions `not_called`, `called_with`, and `called_count` use `tool:<function_name>` or `<function_name>` as their target.

## Policies

`tool_allowlist` fails if any recorded tool call uses a function not listed in `tools`.

```yaml
policies:
  - name: allowed-tools
    type: tool_allowlist
    tools:
      - lookup_order
      - check_refund_eligibility
      - process_refund
```

`requires_confirmation` fails if a protected tool is called without an immediately preceding user turn. The current implementation checks the previous turn role; it does not inspect the content for semantic confirmation text.

```yaml
policies:
  - name: confirm-before-refund
    type: requires_confirmation
    tools:
      - process_refund
      - cancel_subscription
```

The config dataclasses also parse fields such as `threshold`, `prompt`, `judge_model`, `tools`, and `block` on assertions and `block` on policies. The current assertion engine only evaluates the assertion and policy types listed above.
