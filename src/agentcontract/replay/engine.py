"""Replay engine: stubs tool calls from recorded trajectories for deterministic CI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentcontract.types import AgentRun, Turn


@dataclass
class ReplayResult:
    """Result of replaying a trajectory."""

    run: AgentRun
    matched_tools: int = 0
    mismatched_tools: int = 0
    missing_tools: int = 0
    extra_tools: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0 and self.mismatched_tools == 0


class ToolStub:
    """Provides recorded tool results during replay."""

    def __init__(self, run: AgentRun) -> None:
        self._tool_results: dict[str, list[Any]] = {}
        self._call_counts: dict[str, int] = {}

        # Index all tool calls by function name
        for turn in run.turns:
            for tc in turn.tool_calls:
                if tc.function not in self._tool_results:
                    self._tool_results[tc.function] = []
                self._tool_results[tc.function].append(tc.result)

    def get_result(self, function: str, arguments: dict[str, Any] | None = None) -> Any:
        """Get the next recorded result for a tool function.

        Returns results in order of recording. Raises if no more results available.
        """
        count = self._call_counts.get(function, 0)
        results = self._tool_results.get(function, [])

        if count >= len(results):
            raise ToolStubExhausted(
                f"No more recorded results for tool '{function}' "
                f"(called {count + 1} times, only {len(results)} recorded)"
            )

        result = results[count]
        self._call_counts[function] = count + 1
        return result

    def has_results(self, function: str) -> bool:
        """Check if there are remaining results for a function."""
        count = self._call_counts.get(function, 0)
        results = self._tool_results.get(function, [])
        return count < len(results)


class ToolStubExhausted(Exception):
    """Raised when replay requests a tool call that has no more recorded results."""


class ReplayEngine:
    """Replays a recorded trajectory, providing tool stubs and collecting diffs.

    Usage:
        engine = ReplayEngine(recorded_run)
        stub = engine.tool_stub

        # In your agent loop, replace real tool calls:
        result = stub.get_result("lookup_order", {"order_id": "123"})

        # After replay, check results:
        replay_result = engine.finish(actual_turns)
    """

    def __init__(self, run: AgentRun) -> None:
        self._recorded = run
        self._tool_stub = ToolStub(run)

    @property
    def tool_stub(self) -> ToolStub:
        """Access the tool stub for providing recorded results."""
        return self._tool_stub

    @property
    def recorded_run(self) -> AgentRun:
        """Access the original recorded run."""
        return self._recorded

    def finish(self, actual_turns: list[Turn] | None = None) -> ReplayResult:
        """Compare actual execution against recorded trajectory.

        If actual_turns is None, only checks that all tool stubs were consumed.
        """
        result = ReplayResult(run=self._recorded)

        if actual_turns is None:
            # Just check stub consumption
            for func, results in self._tool_stub._tool_results.items():
                count = self._tool_stub._call_counts.get(func, 0)
                remaining = len(results) - count
                if remaining > 0:
                    result.missing_tools += remaining
                    result.errors.append(
                        f"Tool '{func}' was recorded {len(results)} times "
                        f"but only called {count} times during replay"
                    )
            return result

        # Compare turn-by-turn
        recorded_turns = self._recorded.turns
        for i, actual in enumerate(actual_turns):
            if i >= len(recorded_turns):
                extra_call_count = len(actual.tool_calls)
                if extra_call_count:
                    result.extra_tools += extra_call_count
                    result.mismatched_tools += extra_call_count
                result.errors.append(
                    f"Extra turn {i}: role={actual.role.value}, "
                    f"content={actual.content[:50] if actual.content else '(none)'}..."
                )
                continue

            expected = recorded_turns[i]

            # Check role match
            if actual.role != expected.role:
                result.errors.append(
                    f"Turn {i}: expected role={expected.role.value}, got role={actual.role.value}"
                )

            # Check tool call match
            if len(actual.tool_calls) != len(expected.tool_calls):
                diff = len(actual.tool_calls) - len(expected.tool_calls)
                if diff > 0:
                    result.extra_tools += diff
                else:
                    result.missing_tools += -diff
                result.mismatched_tools += abs(diff)
                result.errors.append(
                    f"Turn {i}: expected {len(expected.tool_calls)} tool calls, "
                    f"got {len(actual.tool_calls)}"
                )
            else:
                for j, (act_tc, exp_tc) in enumerate(
                    zip(actual.tool_calls, expected.tool_calls, strict=False)
                ):
                    if act_tc.function != exp_tc.function:
                        result.mismatched_tools += 1
                        result.errors.append(
                            f"Turn {i}, tool {j}: expected function='{exp_tc.function}', "
                            f"got function='{act_tc.function}'"
                        )
                    elif act_tc.arguments != exp_tc.arguments:
                        result.mismatched_tools += 1
                        result.errors.append(
                            f"Turn {i}, tool {j} ({act_tc.function}): arguments differ"
                        )
                    else:
                        result.matched_tools += 1

        # Check for missing turns
        if len(actual_turns) < len(recorded_turns):
            missing = len(recorded_turns) - len(actual_turns)
            missing_turns = recorded_turns[len(actual_turns) :]
            missing_call_count = sum(len(turn.tool_calls) for turn in missing_turns)
            if missing_call_count:
                result.missing_tools += missing_call_count
                result.mismatched_tools += missing_call_count
            result.errors.append(
                f"Missing {missing} turns (recorded {len(recorded_turns)}, got {len(actual_turns)})"
            )

        return result
