# Assertions And Policies

The assertion engine validates an `AgentRun` without raising for ordinary contract failures. It returns a `ContractResult`:

```python
from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import AssertionSpec


result = AssertionEngine().check(
    run,
    assertions=[
        AssertionSpec(type="contains", target="final_response", value="refund"),
    ],
)
assert result.passed, [failure.message for failure in result.failures()]
```

Unknown assertion or policy types fail closed and are reported in the result. Unexpected assertion or policy exceptions are also converted into failed results.

## Target Syntax

| Target | Resolves to |
| --- | --- |
| `final_response` | Last assistant turn whose `content` is not `None`. Empty string content is valid. |
| `turn:N` | `content` from `run.turns[N]`; this uses list position, not the turn's `index` field. |
| `full_conversation` | All non-`None` turn contents joined as `role: content` lines. |
| `tool_call:function_name` | Arguments from the first matching tool call. |
| `tool_call:function_name:arguments` | Arguments from the first matching tool call. |
| `tool_call:function_name:result` | Result from the first matching tool call. |

Tool-oriented assertions also accept `tool:function_name` or just `function_name` as the target.

## Assertion Types

| Type | Required fields | Behavior |
| --- | --- | --- |
| `exact` | `target`, `value` | Passes when the resolved target equals `value`. |
| `contains` | `target`, `value` | Passes when `str(value)` is a substring of `str(actual)`. |
| `regex` | `target`, `value` | Passes when `re.search(value, actual)` matches. |
| `json_schema` | `target`, `schema` | Validates the resolved target with `jsonschema.validate()`. |
| `not_called` | `target` | Passes when the named tool was not invoked. |
| `called_with` | `target`, `schema` | Passes when the named tool was called with all expected argument key/value pairs. |
| `called_count` | `target`, `value` | Passes when the named tool was called exactly `value` times. |

`called_with` reuses the `schema` field for expected arguments. It is a subset match, not JSON Schema validation:

```yaml
defaults:
  assertions:
    - type: called_with
      target: "tool:lookup_order"
      schema:
        order_id: "ORD-123"
```

Use `json_schema` when you want JSON Schema validation:

```yaml
defaults:
  assertions:
    - type: json_schema
      target: "tool_call:process_refund:arguments"
      schema:
        type: object
        required: [order_id, amount, method]
```

`called_count` accepts integer-like values such as `"1"` or `1`. Booleans, missing values, and non-integral numbers fail.

## Assertion Config Fields

`AssertionSpec` supports these fields:

| Field | Current engine use |
| --- | --- |
| `type` | Required dispatch key. |
| `target` | Used by all target-based assertions. |
| `value` | Used by `exact`, `contains`, `regex`, and `called_count`. |
| `schema` | Used by `json_schema` and `called_with`. |
| `threshold` | Parsed but not used by the current engine. |
| `prompt` | Parsed but not used by the current engine. |
| `judge_model` | Parsed but not used by the current engine. |
| `tools` | Parsed but not used by the current engine. |
| `block` | Parsed but not used by the current engine. |

## Policies

Policies are checked after assertions:

```python
from agentcontract.config import PolicySpec


result = AssertionEngine().check(
    run,
    policies=[
        PolicySpec(
            name="allowed-tools",
            type="tool_allowlist",
            tools=["lookup_order", "check_refund_eligibility", "process_refund"],
        )
    ],
)
```

| Type | Required fields | Behavior |
| --- | --- | --- |
| `tool_allowlist` | `name`, `tools` | Fails if any recorded tool name is not listed in `tools`. |
| `requires_confirmation` | `name`, `tools` | Fails if a protected tool call is not immediately preceded by a user turn. |

`requires_confirmation` checks turn roles only. It does not inspect whether the user text contains words like "yes" or "confirm".

`PolicySpec` supports `name`, `type`, `target`, `tools`, and `block`. The current policy engine uses `name`, `type`, and `tools`.

## Failure Results

Each check produces an `AssertionResult` with:

| Field | Meaning |
| --- | --- |
| `assertion` | The assertion spec, or a synthetic policy assertion spec. |
| `passed` | Boolean pass/fail value. |
| `message` | Empty on pass, explanatory text on failure. |
| `details` | Reserved details dictionary. |

`ContractResult.passed` is true only when every assertion and policy result passed. `ContractResult.failures()` returns only failed results.
