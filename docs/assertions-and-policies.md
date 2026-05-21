# Assertions and Policies

Assertions and policies are evaluated by `AssertionEngine.check(run, ...)`.
The pytest fixture `ac_check_contract(run, extra_assertions=None)` merges
configured default assertions, matching scenario overrides, and any extra
assertions before running the engine.

## Assertion Types

| Type | Required fields | Behavior |
| --- | --- | --- |
| `exact` | `target`, `value` | Target must equal `value`. |
| `contains` | `target`, `value` | `value` must be a substring of the target converted to text. |
| `regex` | `target`, `value` | Python `re.search()` must match the target converted to text. |
| `json_schema` | `target`, `schema` | Validates the resolved target with `jsonschema.validate()`. |
| `not_called` | `target` | Tool must not appear in recorded tool calls. |
| `called_with` | `target`, `schema` | Tool must be called with expected arguments. |
| `called_count` | `target`, `value` | Tool must be called exactly `int(value)` times. |

Unknown assertion types fail closed and return a failed `AssertionResult`.

## Target Syntax

Content targets:

- `final_response`: the last assistant turn with non-null content. An empty
  string is a valid final response.
- `full_conversation`: all non-null turn content joined as `role: content`.
- `turn:N`: the content from turn index `N`.

Tool-call targets for `json_schema` and content-style checks:

- `tool_call:function_name:arguments`
- `tool_call:function_name:result`

These return the first matching tool call's arguments or result. If the function
is not found, the target resolves to `None`.

Tool-name targets for tool assertions:

- `tool:function_name`
- `function_name`

These forms are accepted by `not_called`, `called_with`, and `called_count`.

## Examples

```yaml
defaults:
  assertions:
    - type: contains
      target: final_response
      value: refund
    - type: json_schema
      target: tool_call:process_refund:arguments
      schema:
        type: object
        required: [order_id, amount, method]
    - type: called_with
      target: tool:process_refund
      schema:
        order_id: ORD-123
    - type: called_count
      target: tool:lookup_order
      value: 1
```

`called_with` intentionally uses the `schema` field as an expected-argument map,
not as JSON Schema. Each key/value pair in `schema` must be present in the
recorded arguments with exact equality. Extra recorded arguments are allowed.

Use `json_schema` when you need JSON Schema validation.

## Policies

Policies are configured as `PolicySpec` entries and are returned as assertion
results with `type` set to `policy:<name>`.

### `tool_allowlist`

Only tools listed in `tools` may be called.

```yaml
policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]
```

Any recorded tool call whose function name is not in `tools` fails the policy.

### `requires_confirmation`

Protected tools must be preceded by a user turn.

```yaml
policies:
  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

The current implementation checks the immediately previous turn's role. It does
not parse the content of the user message to decide whether the user actually
confirmed the action.

Unknown policy types fail closed. The `target` and `block` policy fields are
parsed by configuration but are not used by the current built-in policies.
