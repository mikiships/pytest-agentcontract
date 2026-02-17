"""Serialization/deserialization for AgentRun and related types."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def run_to_dict(run: AgentRun) -> dict[str, Any]:
    """Serialize an AgentRun to a JSON-compatible dictionary."""
    return {
        "schema_version": run.schema_version,
        "run_id": run.run_id,
        "source": {
            "recorded_at": run.recorded_at,
            "recorder_version": run.recorder_version,
            "sdk": run.sdk,
        },
        "model": {
            "provider": run.model.provider,
            "model": run.model.model,
            "temperature": run.model.temperature,
            "top_p": run.model.top_p,
            "max_tokens": run.model.max_tokens,
            "seed": run.model.seed,
        },
        "metadata": {
            "scenario": run.metadata.scenario,
            "tags": run.metadata.tags,
            "description": run.metadata.description,
        },
        "summary": {
            "total_turns": run.summary.total_turns,
            "total_duration_ms": run.summary.total_duration_ms,
            "total_tokens": {
                "prompt": run.summary.total_tokens.prompt,
                "completion": run.summary.total_tokens.completion,
                "total": run.summary.total_tokens.total,
            },
            "total_tool_calls": run.summary.total_tool_calls,
            "estimated_cost_usd": run.summary.estimated_cost_usd,
        },
        "turns": [_turn_to_dict(t) for t in run.turns],
    }


def _turn_to_dict(turn: Turn) -> dict[str, Any]:
    d: dict[str, Any] = {
        "index": turn.index,
        "role": turn.role.value,
    }
    if turn.content is not None:
        d["content"] = turn.content
    if turn.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "function": tc.function,
                "arguments": tc.arguments,
                "result": tc.result,
                "duration_ms": tc.duration_ms,
            }
            for tc in turn.tool_calls
        ]
    if turn.timing:
        d["timing"] = {
            "latency_ms": turn.timing.latency_ms,
            "time_to_first_token_ms": turn.timing.time_to_first_token_ms,
        }
    if turn.tokens:
        d["tokens"] = {
            "prompt": turn.tokens.prompt,
            "completion": turn.tokens.completion,
            "total": turn.tokens.total,
        }
    return d


def run_from_dict(data: dict[str, Any]) -> AgentRun:
    """Deserialize a dictionary into an AgentRun."""
    source = data.get("source", {})
    model_data = data.get("model", {})
    meta = data.get("metadata", {})
    summary_data = data.get("summary", {})
    tokens_data = summary_data.get("total_tokens", {})

    return AgentRun(
        schema_version=data.get("schema_version", "1.0.0"),
        run_id=data.get("run_id", ""),
        recorded_at=source.get("recorded_at", ""),
        recorder_version=source.get("recorder_version", ""),
        sdk=source.get("sdk", "agentcontract-python"),
        model=ModelInfo(
            provider=model_data.get("provider", ""),
            model=model_data.get("model", ""),
            temperature=model_data.get("temperature", 0.0),
            top_p=model_data.get("top_p", 1.0),
            max_tokens=model_data.get("max_tokens", 4096),
            seed=model_data.get("seed"),
        ),
        metadata=RunMetadata(
            scenario=meta.get("scenario", ""),
            tags=meta.get("tags", []),
            description=meta.get("description", ""),
        ),
        summary=RunSummary(
            total_turns=summary_data.get("total_turns", 0),
            total_duration_ms=summary_data.get("total_duration_ms", 0.0),
            total_tokens=TokenUsage(
                prompt=tokens_data.get("prompt", 0),
                completion=tokens_data.get("completion", 0),
                total=tokens_data.get("total", 0),
            ),
            total_tool_calls=summary_data.get("total_tool_calls", 0),
            estimated_cost_usd=summary_data.get("estimated_cost_usd", 0.0),
        ),
        turns=[_turn_from_dict(t) for t in data.get("turns", [])],
    )


def _turn_from_dict(data: dict[str, Any]) -> Turn:
    tool_calls = [
        ToolCall(
            id=tc["id"],
            function=tc["function"],
            arguments=tc.get("arguments", {}),
            result=tc.get("result"),
            duration_ms=tc.get("duration_ms"),
        )
        for tc in data.get("tool_calls", [])
    ]

    timing = None
    if "timing" in data:
        timing = Timing(
            latency_ms=data["timing"].get("latency_ms"),
            time_to_first_token_ms=data["timing"].get("time_to_first_token_ms"),
        )

    tokens = None
    if "tokens" in data:
        tokens = TokenUsage(
            prompt=data["tokens"].get("prompt", 0),
            completion=data["tokens"].get("completion", 0),
            total=data["tokens"].get("total", 0),
        )

    return Turn(
        index=data["index"],
        role=TurnRole(data["role"]),
        content=data.get("content"),
        tool_calls=tool_calls,
        timing=timing,
        tokens=tokens,
    )


def save_run(run: AgentRun, path: Path) -> None:
    """Save an AgentRun to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(run_to_dict(run), f, indent=2)


def load_run(path: Path) -> AgentRun:
    """Load an AgentRun from a JSON file."""
    with open(path) as f:
        return run_from_dict(json.load(f))
