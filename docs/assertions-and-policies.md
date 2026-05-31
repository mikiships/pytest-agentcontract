# Assertions And Policies

Assertions and policies are evaluated by `AssertionEngine.check(...)`. Pytest
tests usually call `ac_check_contract(run, extra_assertions=[...])`, which
combines:

- `defaults.assertions` from `agentcontract.yml`
- matching `overrides.<scenario>.assertions`
- `extra_assertions` passed by the test
- top-level `policies` from `agentcontract.yml`

## Assertion Types

| Type | Required fields | Behavior |
| --- | --- | --- |
| `exact` | `target`, `value` | Target value must exactly equal `value`. |
| `contains` | `target`, `value` | String form of `value` must be present in the target. |
| `regex` | `target`, `value` | Python regular expression in `value` must match the target. |
| `json_schema` | `target`, `schema` | Validates the resolved target with `jsonschema.validate`. |
| `not_called` | `target` | Tool must not appear in recorded tool calls. |
| `called_with` | `target`, `schema` | Tool must be called with at least the expected argument keys and values. |
| `called_count` | `target`, `value` | Tool must be called exactly `value` times. |

Unknown assertion types fail closed.

## Target Syntax

General content targets:

- `final_response`: content from the last assistant turn with non-`None`
  content. Empty strings are valid final responses.
- `turn:N`: content from zero-based turn index `N`.
- `full_conversation`: all non-`None` turn content joined as
  `role: content` lines.
- `tool_call:function_name:arguments`: arguments from the first matching tool
  call.
- `tool_call:function_name:result`: result from the first matching tool call.

Tool call assertions use either `tool:function_name` or just `function_name` as
their target:

```yaml
- type: not_called
  target: tool:delete_account
```

## `called_with`

`called_with` reuses the `schema` field as an expected argument subset, not as a
JSON Schema. The assertion passes when any recorded call for the target tool has
matching values for every key in `schema`.

```yaml
- type: called_with
  target: tool:lookup_order
  schema:
    order_id: "123"
```

If the actual call has extra arguments, the assertion can still pass. If
`schema` is missing or is not an object, the assertion fails.

## `called_count`

`called_count` requires an integer-like `value`. String values such as `"1"` are
accepted because the engine converts the value with `int(...)`; booleans and
non-integer floats are rejected.

```yaml
- type: called_count
  target: tool:lookup_order
  value: 1
```

## `json_schema`

Use `json_schema` when you want structural validation for a resolved target,
usually tool arguments or tool results.

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
        type: string
```

The assertion fails if either `target` resolves to `None` or `schema` is missing.

## Policies

Policies are checked after assertions.

### `tool_allowlist`

Only tools listed in `tools` may be called.

```yaml
policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]
```

Any recorded tool call whose function is not in `tools` fails the policy.

### `requires_confirmation`

Protected tools must be immediately preceded by a user turn.

```yaml
policies:
  - name: confirm-before-refund
    type: requires_confirmation
    tools: [process_refund]
```

This policy checks turn order only. It does not inspect the content of the user
turn for particular confirmation words.

Unknown policy types fail closed.
