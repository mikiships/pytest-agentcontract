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


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Normalize optional object sections that may be null in JSON."""
    if isinstance(value, dict):
        return value
    return {}


def _coerce_list(value: Any) -> list[Any]:
    """Normalize optional array sections that may be null in JSON."""
    if isinstance(value, list):
        return value
    return []


def _coerce_tool_arguments(value: Any) -> dict[str, Any]:
    """Normalize tool-call argument payloads to objects."""
    if isinstance(value, dict):
        return value
    return {}


def _coerce_str(value: Any, default: str = "") -> str:
    """Normalize scalar fields to strings, preserving defaults for nulls."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_int(value: Any, default: int = 0) -> int:
    """Normalize numeric fields that should be integers."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: Any) -> int | None:
    """Normalize optional integer fields."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Normalize numeric fields that should be floats."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(value: Any) -> float | None:
    """Normalize optional float fields."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_json_compatible(value: Any) -> Any:
    """Recursively normalize values so ``json.dump`` never crashes."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return [_to_json_compatible(item) for item in sorted(value, key=repr)]

    if isinstance(value, Path):
        return str(value)

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except TypeError:
            pass

    return str(value)


def run_to_dict(run: AgentRun) -> dict[str, Any]:
    """Serialize an AgentRun to a JSON-compatible dictionary."""
    return _to_json_compatible(
        {
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
    )


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
    source = _coerce_dict(data.get("source"))
    model_data = _coerce_dict(data.get("model"))
    meta = _coerce_dict(data.get("metadata"))
    summary_data = _coerce_dict(data.get("summary"))
    tokens_data = _coerce_dict(summary_data.get("total_tokens"))
    tags = meta.get("tags")
    if not isinstance(tags, list):
        tags = []

    return AgentRun(
        schema_version=_coerce_str(data.get("schema_version"), "1.0.0"),
        run_id=_coerce_str(data.get("run_id"), ""),
        recorded_at=_coerce_str(source.get("recorded_at"), ""),
        recorder_version=_coerce_str(source.get("recorder_version"), ""),
        sdk=_coerce_str(source.get("sdk"), "agentcontract-python"),
        model=ModelInfo(
            provider=_coerce_str(model_data.get("provider"), ""),
            model=_coerce_str(model_data.get("model"), ""),
            temperature=_coerce_float(model_data.get("temperature"), 0.0),
            top_p=_coerce_float(model_data.get("top_p"), 1.0),
            max_tokens=_coerce_int(model_data.get("max_tokens"), 4096),
            seed=_coerce_optional_int(model_data.get("seed")),
        ),
        metadata=RunMetadata(
            scenario=_coerce_str(meta.get("scenario"), ""),
            tags=tags,
            description=_coerce_str(meta.get("description"), ""),
        ),
        summary=RunSummary(
            total_turns=_coerce_int(summary_data.get("total_turns"), 0),
            total_duration_ms=_coerce_float(summary_data.get("total_duration_ms"), 0.0),
            total_tokens=TokenUsage(
                prompt=_coerce_int(tokens_data.get("prompt"), 0),
                completion=_coerce_int(tokens_data.get("completion"), 0),
                total=_coerce_int(tokens_data.get("total"), 0),
            ),
            total_tool_calls=_coerce_int(summary_data.get("total_tool_calls"), 0),
            estimated_cost_usd=_coerce_float(summary_data.get("estimated_cost_usd"), 0.0),
        ),
        turns=[_turn_from_dict(t) for t in _coerce_list(data.get("turns")) if isinstance(t, dict)],
    )


def _turn_from_dict(data: dict[str, Any]) -> Turn:
    tool_calls_data = _coerce_list(data.get("tool_calls"))
    tool_calls = [
        ToolCall(
            id=_coerce_str(tc.get("id"), ""),
            function=_coerce_str(tc.get("function"), ""),
            arguments=_coerce_tool_arguments(tc.get("arguments")),
            result=tc.get("result"),
            duration_ms=_coerce_optional_float(tc.get("duration_ms")),
        )
        for tc in tool_calls_data
        if isinstance(tc, dict)
    ]

    timing = None
    timing_data = data.get("timing")
    if isinstance(timing_data, dict):
        timing = Timing(
            latency_ms=_coerce_optional_float(timing_data.get("latency_ms")),
            time_to_first_token_ms=_coerce_optional_float(timing_data.get("time_to_first_token_ms")),
        )

    tokens = None
    tokens_data = data.get("tokens")
    if isinstance(tokens_data, dict):
        tokens = TokenUsage(
            prompt=_coerce_int(tokens_data.get("prompt"), 0),
            completion=_coerce_int(tokens_data.get("completion"), 0),
            total=_coerce_int(tokens_data.get("total"), 0),
        )

    content = data.get("content")
    if content is not None and not isinstance(content, str):
        content = str(content)

    raw_role = _coerce_str(data.get("role"))
    if not raw_role:
        raise ValueError("Turn is missing required field 'role'")
    try:
        role = TurnRole(raw_role)
    except ValueError as exc:
        raise ValueError(f"Invalid turn role: {raw_role!r}") from exc

    return Turn(
        index=_coerce_int(data.get("index"), 0),
        role=role,
        content=content,
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
