"""Assertion engine: validates agent trajectories against contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from agentcontract.config import AssertionSpec, PolicySpec
from agentcontract.types import AgentRun, TurnRole


@dataclass
class AssertionResult:
    """Result of a single assertion check."""

    assertion: AssertionSpec
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractResult:
    """Result of all assertions + policies against a trajectory."""

    scenario: str
    results: list[AssertionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def failures(self) -> list[AssertionResult]:
        return [r for r in self.results if not r.passed]


class AssertionEngine:
    """Evaluates assertions and policies against an AgentRun trajectory.

    Usage:
        engine = AssertionEngine()
        result = engine.check(run, assertions, policies)
        assert result.passed
    """

    def check(
        self,
        run: AgentRun,
        assertions: list[AssertionSpec] | None = None,
        policies: list[PolicySpec] | None = None,
    ) -> ContractResult:
        """Run all assertions and policies against a trajectory."""
        result = ContractResult(scenario=run.metadata.scenario)

        for assertion in assertions or []:
            ar = self._check_assertion(run, assertion)
            result.results.append(ar)

        for policy in policies or []:
            pr = self._check_policy(run, policy)
            result.results.append(pr)

        return result

    def _check_assertion(self, run: AgentRun, spec: AssertionSpec) -> AssertionResult:
        """Dispatch to the appropriate assertion checker."""
        checkers = {
            "exact": self._check_exact,
            "contains": self._check_contains,
            "regex": self._check_regex,
            "json_schema": self._check_json_schema,
            "not_called": self._check_not_called,
            "called_with": self._check_called_with,
            "called_count": self._check_called_count,
        }

        checker = checkers.get(spec.type)
        if checker is None:
            return AssertionResult(
                assertion=spec,
                passed=False,
                message=f"Unknown assertion type: {spec.type}",
            )

        try:
            return checker(run, spec)
        except Exception as e:
            return AssertionResult(
                assertion=spec,
                passed=False,
                message=f"Assertion error: {e}",
            )

    def _resolve_target(self, run: AgentRun, target: str) -> Any:
        """Resolve a target string to actual content from the run.

        Targets:
            final_response - last assistant message content
            turn:N - specific turn content
            full_conversation - all turns concatenated
            tool_call:function_name:arguments - tool call arguments
            tool_call:function_name:result - tool call result
        """
        if target == "final_response":
            for turn in reversed(run.turns):
                if turn.role == TurnRole.ASSISTANT and turn.content is not None:
                    return turn.content
            return None

        if target == "full_conversation":
            parts = []
            for turn in run.turns:
                if turn.content is not None:
                    parts.append(f"{turn.role.value}: {turn.content}")
            return "\n".join(parts)

        if target.startswith("turn:"):
            _, raw_idx = target.split(":", 1)
            idx = int(raw_idx)
            if 0 <= idx < len(run.turns):
                return run.turns[idx].content
            return None

        if target.startswith("tool_call:"):
            parts = target.split(":", 2)
            if len(parts) < 2 or not parts[1]:
                return None
            func_name = parts[1]
            field_name = parts[2] if len(parts) > 2 else "arguments"

            for turn in run.turns:
                for tc in turn.tool_calls:
                    if tc.function == func_name:
                        if field_name == "arguments":
                            return tc.arguments
                        elif field_name == "result":
                            return tc.result
            return None

        return None

    def _get_all_tool_calls(self, run: AgentRun) -> list[tuple[str, dict[str, Any], Any]]:
        """Get all tool calls as (function, arguments, result) tuples."""
        calls = []
        for turn in run.turns:
            for tc in turn.tool_calls:
                calls.append((tc.function, tc.arguments, tc.result))
        return calls

    def _check_exact(self, run: AgentRun, spec: AssertionSpec) -> AssertionResult:
        if spec.value is None:
            return AssertionResult(
                assertion=spec,
                passed=False,
                message="'exact' requires a non-null 'value'",
            )
        actual = self._resolve_target(run, spec.target)
        passed = actual == spec.value
        return AssertionResult(
            assertion=spec,
            passed=passed,
            message="" if passed else f"Expected exact '{spec.value}', got '{actual}'",
        )

    def _check_contains(self, run: AgentRun, spec: AssertionSpec) -> AssertionResult:
        actual = self._resolve_target(run, spec.target)
        if actual is None or spec.value is None:
            return AssertionResult(
                assertion=spec,
                passed=False,
                message=f"Target '{spec.target}' resolved to None",
            )
        passed = spec.value in str(actual)
        return AssertionResult(
            assertion=spec,
            passed=passed,
            message="" if passed else f"'{spec.value}' not found in target",
        )

    def _check_regex(self, run: AgentRun, spec: AssertionSpec) -> AssertionResult:
        actual = self._resolve_target(run, spec.target)
        if actual is None or spec.value is None:
            return AssertionResult(
                assertion=spec, passed=False, message="Target or pattern is None"
            )
        passed = bool(re.search(spec.value, str(actual)))
        return AssertionResult(
            assertion=spec,
            passed=passed,
            message="" if passed else f"Pattern '{spec.value}' not matched",
        )

    def _check_json_schema(self, run: AgentRun, spec: AssertionSpec) -> AssertionResult:
        actual = self._resolve_target(run, spec.target)
        if actual is None or spec.schema is None:
            return AssertionResult(assertion=spec, passed=False, message="Target or schema is None")
        try:
            jsonschema.validate(instance=actual, schema=spec.schema)
            return AssertionResult(assertion=spec, passed=True)
        except jsonschema.ValidationError as e:
            return AssertionResult(
                assertion=spec,
                passed=False,
                message=f"Schema validation failed: {e.message}",
            )

    def _check_not_called(self, run: AgentRun, spec: AssertionSpec) -> AssertionResult:
        """Assert a tool was NOT called."""
        # target format: "tool:function_name" or just the function name
        func_name = (
            spec.target.replace("tool:", "") if spec.target.startswith("tool:") else spec.target
        )
        calls = self._get_all_tool_calls(run)
        called = any(name == func_name for name, _, _ in calls)
        return AssertionResult(
            assertion=spec,
            passed=not called,
            message="" if not called else f"Tool '{func_name}' was called but should not have been",
        )

    def _check_called_with(self, run: AgentRun, spec: AssertionSpec) -> AssertionResult:
        """Assert a tool was called with specific arguments."""
        func_name = (
            spec.target.replace("tool:", "") if spec.target.startswith("tool:") else spec.target
        )
        if spec.schema is None:
            return AssertionResult(
                assertion=spec,
                passed=False,
                message="'called_with' requires expected arguments in 'schema'",
            )
        if not isinstance(spec.schema, dict):
            return AssertionResult(
                assertion=spec,
                passed=False,
                message="'called_with' expects a dict in 'schema'",
            )

        expected_args = spec.schema  # reuse schema field for expected args
        calls = self._get_all_tool_calls(run)

        for name, args, _ in calls:
            if name == func_name and isinstance(args, dict):
                # Check if expected args are a subset of actual args
                match = all(args.get(k) == v for k, v in expected_args.items())
                if match:
                    return AssertionResult(assertion=spec, passed=True)

        return AssertionResult(
            assertion=spec,
            passed=False,
            message=f"Tool '{func_name}' was not called with expected arguments",
        )

    def _check_called_count(self, run: AgentRun, spec: AssertionSpec) -> AssertionResult:
        """Assert a tool was called exactly N times."""
        func_name = (
            spec.target.replace("tool:", "") if spec.target.startswith("tool:") else spec.target
        )
        if spec.value is None:
            return AssertionResult(
                assertion=spec,
                passed=False,
                message="'called_count' requires an integer 'value'",
            )
        try:
            expected_count = int(spec.value)
        except (TypeError, ValueError):
            return AssertionResult(
                assertion=spec,
                passed=False,
                message="'called_count' expects an integer 'value'",
            )

        calls = self._get_all_tool_calls(run)
        actual_count = sum(1 for name, _, _ in calls if name == func_name)

        passed = actual_count == expected_count
        return AssertionResult(
            assertion=spec,
            passed=passed,
            message=""
            if passed
            else f"Tool '{func_name}' called {actual_count} times, expected {expected_count}",
        )

    def _check_policy(self, run: AgentRun, policy: PolicySpec) -> AssertionResult:
        """Check a policy against the trajectory."""
        policy_checkers = {
            "tool_allowlist": self._policy_tool_allowlist,
            "requires_confirmation": self._policy_requires_confirmation,
        }

        checker = policy_checkers.get(policy.type)
        if checker is None:
            return AssertionResult(
                assertion=AssertionSpec(type=f"policy:{policy.name}"),
                passed=False,
                message=f"Unknown policy type: {policy.type}",
            )

        try:
            return checker(run, policy)
        except Exception as e:
            return AssertionResult(
                assertion=AssertionSpec(type=f"policy:{policy.name}"),
                passed=False,
                message=f"Policy error: {e}",
            )

    def _policy_tool_allowlist(self, run: AgentRun, policy: PolicySpec) -> AssertionResult:
        """Only allowed tools may be called."""
        calls = self._get_all_tool_calls(run)
        violations = [name for name, _, _ in calls if name not in policy.tools]

        spec = AssertionSpec(type=f"policy:{policy.name}", target="all_tool_calls")
        if violations:
            return AssertionResult(
                assertion=spec,
                passed=False,
                message=f"Disallowed tools called: {violations}",
            )
        return AssertionResult(assertion=spec, passed=True)

    def _policy_requires_confirmation(self, run: AgentRun, policy: PolicySpec) -> AssertionResult:
        """Protected tools must be preceded by a user confirmation turn."""
        spec = AssertionSpec(type=f"policy:{policy.name}", target="tool_sequence")

        for i, turn in enumerate(run.turns):
            for tc in turn.tool_calls:
                if tc.function in policy.tools:
                    # Check if previous turn was a user message (confirmation)
                    if i == 0:
                        return AssertionResult(
                            assertion=spec,
                            passed=False,
                            message=(
                                f"Tool '{tc.function}' called at turn 0 "
                                f"with no prior confirmation"
                            ),
                        )
                    prev = run.turns[i - 1]
                    if prev.role != TurnRole.USER:
                        return AssertionResult(
                            assertion=spec,
                            passed=False,
                            message=(
                                f"Tool '{tc.function}' at turn {i} "
                                f"not preceded by user confirmation"
                            ),
                        )

        return AssertionResult(assertion=spec, passed=True)
