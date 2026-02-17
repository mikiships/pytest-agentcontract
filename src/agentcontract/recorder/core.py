"""Core recorder that intercepts LLM SDK calls and captures trajectories."""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentcontract import __version__
from agentcontract.serialization import save_run
from agentcontract.types import (
    AgentRun,
    ModelInfo,
    RunMetadata,
    RunSummary,
    Timing,
    TokenUsage,
    ToolCall,
    Turn,
    TurnRole,
)


class Recorder:
    """Records agent tool calls and LLM interactions into an AgentRun trajectory.

    Usage:
        recorder = Recorder(scenario="refund-flow")
        with recorder.recording():
            # ... run your agent ...
            recorder.add_turn(role="user", content="I want a refund")
            recorder.add_turn(
                role="assistant",
                content="Let me check your order.",
                tool_calls=[{
                    "id": "1", "function": "lookup_order",
                    "arguments": {"order_id": "123"}, "result": {...}
                }]
            )
        recorder.save("tests/scenarios/refund-flow.agentrun.json")
    """

    def __init__(
        self,
        scenario: str = "",
        tags: list[str] | None = None,
        description: str = "",
        model_provider: str = "",
        model_name: str = "",
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._run = AgentRun(
            run_id=str(uuid.uuid4()),
            recorded_at=datetime.now(timezone.utc).isoformat(),
            recorder_version=__version__,
            model=ModelInfo(
                provider=model_provider,
                model=model_name,
                temperature=temperature,
                seed=seed,
            ),
            metadata=RunMetadata(
                scenario=scenario,
                tags=tags or [],
                description=description,
            ),
        )
        self._start_time: float | None = None
        self._turn_index = 0
        self._total_tool_calls = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    @contextmanager
    def recording(self) -> Generator[Recorder, None, None]:
        """Context manager that tracks total duration."""
        self._start_time = time.monotonic()
        try:
            yield self
        finally:
            elapsed = time.monotonic() - self._start_time
            self._run.summary = RunSummary(
                total_turns=len(self._run.turns),
                total_duration_ms=elapsed * 1000,
                total_tokens=TokenUsage(
                    prompt=self._total_prompt_tokens,
                    completion=self._total_completion_tokens,
                    total=self._total_prompt_tokens + self._total_completion_tokens,
                ),
                total_tool_calls=self._total_tool_calls,
            )

    def add_turn(
        self,
        role: str,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        latency_ms: float | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> Turn:
        """Add a turn to the trajectory."""
        parsed_tool_calls = []
        if tool_calls:
            for tc in tool_calls:
                parsed_tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        function=tc["function"],
                        arguments=tc.get("arguments", {}),
                        result=tc.get("result"),
                        duration_ms=tc.get("duration_ms"),
                    )
                )
            self._total_tool_calls += len(parsed_tool_calls)

        timing = Timing(latency_ms=latency_ms) if latency_ms is not None else None
        tokens = None
        if prompt_tokens or completion_tokens:
            tokens = TokenUsage(
                prompt=prompt_tokens,
                completion=completion_tokens,
                total=prompt_tokens + completion_tokens,
            )
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens

        turn = Turn(
            index=self._turn_index,
            role=TurnRole(role),
            content=content,
            tool_calls=parsed_tool_calls,
            timing=timing,
            tokens=tokens,
        )
        self._run.turns.append(turn)
        self._turn_index += 1
        return turn

    def save(self, path: str | Path) -> Path:
        """Save the recorded trajectory to a .agentrun.json file."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".agentrun.json")
        save_run(self._run, p)
        return p

    @property
    def run(self) -> AgentRun:
        """Access the underlying AgentRun."""
        return self._run
