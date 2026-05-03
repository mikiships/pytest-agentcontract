# Assertions and Policies

Assertions check recorded content and tool calls. Policies enforce suite-wide
rules on the trajectory.

## Assertion Spec

Assertions are usually written in `agentcontract.yml` or passed directly as
`AssertionSpec` objects:

```yaml
defaults:
  assertions:
    - type: contains
      target: final_response
      value: refund
```

Supported fields are:

| Field | Purpose |
| --- | --- |
| `type` | Assertion type. Required. |
| `target` | Target selector. Required by most assertions. |
| `value` | Expected string, regex, or count depending on assertion type. |
| `schema` | JSON Schema or expected argument subset depending on assertion type. |
| `threshold` | Parsed for future judge-style assertions. |
| `prompt` | Parsed for future judge-style assertions. |
| `judge_model` | Parsed for future judge-style assertions. |
| `tools` | Parsed for compatibility with policy-like config. |
| `block` | Parsed for compatibility with policy-like config. |

## Assertion Types

| Type | Required fields | Behavior |
| --- | --- | --- |
| `exact` | `target`, `value` | Target content must equal `value`. |
| `contains` | `target`, `value` | Target content must contain `value`. |
| `regex` | `target`, `value` | Target content must match the regex in `value`. |
| `json_schema` | `target`, `schema` | Target value must validate against the JSON Schema in `schema`. |
| `not_called` | `target` | Tool named by `target` must not be called. |
| `called_with` | `target`, `schema` | Tool named by `target` must be called with arguments containing every key/value from `schema`. |
| `called_count` | `target`, `value` | Tool named by `target` must be called exactly `value` times. |

`called_with` intentionally treats `schema` as an expected argument subset, not
as JSON Schema. Use `json_schema` when you need JSON Schema validation.

## Target Syntax

| Target | Resolves to |
| --- | --- |
| `final_response` | The last assistant turn with non-null content. |
| `full_conversation` | All turn content joined as `role: content` lines. |
| `turn:N` | The content of turn index `N`. |
| `tool_call:function_name:arguments` | Arguments from the first matching tool call. |
| `tool_call:function_name:result` | Result from the first matching tool call. |
| `tool:function_name` | Accepted by tool assertions such as `not_called`, `called_with`, and `called_count`. |
| `function_name` | Also accepted by tool assertions. |

## Examples

```yaml
defaults:
  assertions:
    - type: contains
      target: final_response
      value: processed

    - type: regex
      target: full_conversation
      value: "refund|return"

    - type: json_schema
      target: tool_call:process_refund:arguments
      schema:
        type: object
        required: [order_id, amount, method]

    - type: called_with
      target: tool:lookup_order
      schema:
        order_id: ORD-123

    - type: called_count
      target: process_refund
      value: 1

    - type: not_called
      target: tool:escalate_to_fraud
```

## Policy Spec

Policies are configured under `policies`:

```yaml
policies:
  - name: allowed-tools
    type: tool_allowlist
    tools: [lookup_order, check_refund_eligibility, process_refund]
```

Supported fields are:

| Field | Purpose |
| --- | --- |
| `name` | Stable name used in assertion output. Required. |
| `type` | Policy type. Required. |
| `target` | Parsed for compatibility; not currently used by built-in policies. |
| `tools` | Tool list used by built-in policies. |
| `block` | Parsed for compatibility; not currently used by built-in policies. |

## Policy Types

| Type | Behavior |
| --- | --- |
| `tool_allowlist` | Fails if any recorded tool call function is not listed in `tools`. |
| `requires_confirmation` | Fails if a protected tool is not immediately preceded by a user turn. |

## Tool Allowlist

```yaml
policies:
  - name: customer-support-tools
    type: tool_allowlist
    tools:
      - lookup_order
      - check_refund_eligibility
      - process_refund
```

This is useful for catching accidental access to tools that should not be
reachable in a scenario, such as account deletion or admin-only operations.

## Confirmation Requirements

```yaml
policies:
  - name: confirm-before-refund
    type: requires_confirmation
    tools:
      - process_refund
```

For each matching tool call, the previous recorded turn must have role `user`.
The policy checks role order only; it does not inspect the text of the user
message for a specific confirmation phrase.
