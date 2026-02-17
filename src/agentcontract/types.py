"""Core types for agent trajectory recording and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TurnRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A single tool/function call within a turn."""

    id: str
    function: str
    arguments: dict[str, Any]
    result: Any = None
    duration_ms: float | None = None


@dataclass
class Timing:
    """Timing information for a turn."""

    latency_ms: float | None = None
    time_to_first_token_ms: float | None = None


@dataclass
class TokenUsage:
    """Token counts for a turn."""

    prompt: int = 0
    completion: int = 0
    total: int = 0


@dataclass
class Turn:
    """A single turn in an agent trajectory."""

    index: int
    role: TurnRole
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    timing: Timing | None = None
    tokens: TokenUsage | None = None


@dataclass
class ModelInfo:
    """Model configuration used for the recording."""

    provider: str
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    seed: int | None = None


@dataclass
class RunMetadata:
    """Metadata about the recorded run."""

    scenario: str = ""
    tags: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RunSummary:
    """Summary statistics for the run."""

    total_turns: int = 0
    total_duration_ms: float = 0.0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_tool_calls: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class AgentRun:
    """A complete recorded agent trajectory.

    This is the top-level recording format (.agentrun.json).
    """

    schema_version: str = "1.0.0"
    run_id: str = ""
    recorded_at: str = ""
    recorder_version: str = ""
    sdk: str = "agentcontract-python"
    model: ModelInfo = field(default_factory=lambda: ModelInfo(provider="", model=""))
    metadata: RunMetadata = field(default_factory=RunMetadata)
    summary: RunSummary = field(default_factory=RunSummary)
    turns: list[Turn] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON output."""
        from agentcontract.serialization import run_to_dict

        return run_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRun:
        """Deserialize from a dictionary."""
        from agentcontract.serialization import run_from_dict

        return run_from_dict(data)
